"""Equivalence + benchmark for the two recurrent delta-scan kernels.

The K×K reference path (`_recurrent_delta_scan_kxk`) materialises the full
state-transition matrix and runs a Hillis–Steele prefix scan inside chunks
of 64 tokens — O(T · K^3) FLOPs.  The fast sequential path
(`_recurrent_delta_scan_seq`) exploits rank-1 structure and runs a per-token
O(K · V) update — same forward output within fp32 noise, ~K=64× fewer
FLOPs, ~K=64× less peak memory.

CI tests:
  - both paths agree on a non-trivial gate > 0 (the only regime where
    the rank-1 erase is non-zero, hence the only place the two paths
    can disagree); max abs diff < 1e-4 fp32.
  - both paths give identical at-init reduction to vanilla WKV when
    gate ≡ 0 (cross-check that neither scan introduces a regression).

Manual benchmark: `python tests/test_delta_scan_kernels.py` on GPU prints
wall-clock timing for an ASR-typical (B=8, H=4, T=300, K=64) input.
"""

import os
import time

import torch


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_inputs(B=2, H=4, T=128, K=64, seed=0):
    """Realistic synthetic inputs in the RWKV-6 init regime."""
    torch.manual_seed(seed)
    r = torch.randn(B, H, T, K, device=_DEVICE)
    k = torch.randn(B, H, T, K, device=_DEVICE)
    v = torch.randn(B, H, T, K, device=_DEVICE)
    # Log-decay in the realistic init range, w ∈ [-0.6, -0.1].
    w = -torch.rand(B, H, T, K, device=_DEVICE).mul(0.5).add(0.1)
    u = torch.randn(1, H, 1, K, device=_DEVICE).mul(0.1)
    # Normalised key direction.
    kk = torch.randn(B, H, T, K, device=_DEVICE)
    kk = kk / (kk.norm(dim=-1, keepdim=True) + 1e-8)
    # Erase strength σ-bounded in [0, 2].
    iclr = torch.sigmoid(torch.randn(B, H, T, K, device=_DEVICE)) * 2.0
    # Coupling vector γ_t = exp(p · w) with p=1.0.
    gamma = torch.exp(w)
    return r, k, v, w, u, kk, iclr, gamma


def test_seq_matches_kxk_with_active_gate():
    """Both paths must agree to fp32 noise when the rank-1 erase is live."""
    from src.models.rwkv6_time_mix import (
        _recurrent_delta_scan_seq, _recurrent_delta_scan_kxk,
    )
    r, k, v, w, u, kk, iclr, gamma = _make_inputs(B=2, H=4, T=128, K=64)
    H = r.shape[1]

    # Active gate so the rank-1 term is non-zero — that is the regime where
    # the two scans can numerically differ.
    gate = torch.full((H,), 0.3, device=_DEVICE)

    y_seq, S_seq = _recurrent_delta_scan_seq(r, k, v, w, u, kk, iclr, gate, None, gamma=gamma)
    y_kxk, S_kxk = _recurrent_delta_scan_kxk(r, k, v, w, u, kk, iclr, gate, None, gamma=gamma)

    diff_y = (y_seq - y_kxk).abs().max().item()
    diff_S = (S_seq - S_kxk).abs().max().item()
    rel_y = diff_y / (y_kxk.abs().max().item() + 1e-12)
    print(f"[seq vs kxk, gate=0.3]  diff_y={diff_y:.3e} (rel {rel_y:.3e})  diff_S={diff_S:.3e}")

    # 1e-4 absolute is generous fp32 noise for this size — the two paths
    # accumulate round-off in different orders (sequential vs Hillis-Steele
    # parallel) over T=128 steps with non-trivial rank-1 updates.
    assert diff_y < 1e-4, f"output diverges between scan kernels: max |Δy|={diff_y:.3e}"


def test_both_paths_reduce_to_vanilla_at_init():
    """When gate=0, both scans must reduce to the same output (vanilla
    WKV recurrence).  This is the §2.4 reduction contract — neither
    scan can leak the rank-1 erase term when β_eff=0."""
    from src.models.rwkv6_time_mix import (
        _recurrent_delta_scan_seq, _recurrent_delta_scan_kxk,
    )
    r, k, v, w, u, kk, iclr, gamma = _make_inputs(B=2, H=4, T=128, K=64)
    H = r.shape[1]
    gate = torch.zeros(H, device=_DEVICE)  # at-init: dormant

    y_seq, _ = _recurrent_delta_scan_seq(r, k, v, w, u, kk, iclr, gate, None, gamma=gamma)
    y_kxk, _ = _recurrent_delta_scan_kxk(r, k, v, w, u, kk, iclr, gate, None, gamma=gamma)

    diff = (y_seq - y_kxk).abs().max().item()
    rel = diff / (y_kxk.abs().max().item() + 1e-12)
    print(f"[seq vs kxk, gate=0]    diff_y={diff:.3e} (rel {rel:.3e})")
    # At gate=0 both scans collapse to the same vanilla WKV computation;
    # the only differences are accumulation order.  The K×K reference
    # accumulates into a 4096-float matrix per chunk, whereas the
    # sequential kernel fans the cumulative decay through K-vector ops —
    # ~K=64 fewer rounding operations per token.  Relative diff in the
    # 1e-7 range is fp32 noise floor.
    assert rel < 1e-5, f"at-init scans diverge: rel |Δy|={rel:.3e}"


def _benchmark():
    """Manual wall-clock benchmark — not a CI test."""
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping benchmark.")
        return

    from src.models.rwkv6_time_mix import (
        _recurrent_delta_scan_seq, _recurrent_delta_scan_kxk,
    )

    # Compile the sequential scan to fuse per-token Python+kernel-launch
    # overhead into a single CUDA graph.
    seq_compiled = torch.compile(_recurrent_delta_scan_seq, mode="reduce-overhead",
                                  dynamic=False, fullgraph=False)

    for shape in [(8, 4, 300, 64), (8, 4, 1024, 64)]:
        B, H, T, K = shape
        r, k, v, w, u, kk, iclr, gamma = _make_inputs(B=B, H=H, T=T, K=K, seed=1)
        gate = torch.full((H,), 0.05, device=_DEVICE)

        def time_kernel(fn, n=10):
            # warm-up — torch.compile needs more iters to specialise.
            for _ in range(5):
                y, S = fn(r, k, v, w, u, kk, iclr, gate, None, gamma=gamma)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                y, S = fn(r, k, v, w, u, kk, iclr, gate, None, gamma=gamma)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n, y, S

        print(f"\nshape: B={B} H={H} T={T} K={K}  gate={gate[0].item()}")
        print("forward only (no backward):")

        t_kxk, y_kxk, _ = time_kernel(_recurrent_delta_scan_kxk)
        print(f"  K×K reference (Hillis-Steele):  {t_kxk*1000:8.2f} ms/call")

        t_seq, y_seq, _ = time_kernel(_recurrent_delta_scan_seq)
        print(f"  rank-1 sequential (eager):       {t_seq*1000:8.2f} ms/call  ({t_kxk/t_seq:.2f}×)")

        try:
            t_seq_c, y_seq_c, _ = time_kernel(seq_compiled)
            print(f"  rank-1 sequential (compiled):    {t_seq_c*1000:8.2f} ms/call  ({t_kxk/t_seq_c:.2f}×)")
            print(f"  max |Δy_compiled|: {(y_seq_c - y_kxk).abs().max().item():.3e}")
        except Exception as e:
            print(f"  rank-1 sequential (compiled):    FAILED — {type(e).__name__}: {e}")

        # Memory measurement
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        _ = _recurrent_delta_scan_kxk(r, k, v, w, u, kk, iclr, gate, None, gamma=gamma)
        torch.cuda.synchronize()
        mem_kxk = torch.cuda.max_memory_allocated() / (1024 ** 3)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        _ = _recurrent_delta_scan_seq(r, k, v, w, u, kk, iclr, gate, None, gamma=gamma)
        torch.cuda.synchronize()
        mem_seq = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"  peak mem: K×K {mem_kxk:.2f} GB   seq {mem_seq:.2f} GB   ratio {mem_kxk/mem_seq:.2f}×")


if __name__ == "__main__":
    test_seq_matches_kxk_with_active_gate()
    test_both_paths_reduce_to_vanilla_at_init()
    print()
    _benchmark()
