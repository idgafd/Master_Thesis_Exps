"""Equivalence test + speed benchmark for rse_scan_fast.rse_viscosity_scan.

Compares against the reference RWKV6TimeMix._forward_recurrent_rse with
use_viscosity=True. Measures:
  - max absolute difference of forward outputs (should be < 1e-4 fp32)
  - max absolute difference of grad wrt each input (autograd equivalence)
  - wall-clock speedup (forward + forward+backward)
  - peak VRAM

Usage:
    uv run python scripts/benchmark_rse_fast.py
"""

from __future__ import annotations
import math
import sys
import time
from pathlib import Path

import torch

# Make the repo importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.rwkv6_time_mix import RWKV6TimeMix
from src.models.rse_scan_fast import rse_viscosity_scan


# ── Shapes matching a typical LibriSpeech batch through the Stage-4 encoder ──
# d_model=256, 4 heads, head_size=64, Bk=32 blocks per head
D_MODEL = 256
N_HEAD = 4
HEAD_SIZE = 64
N_LAYERS = 6
LAYER_ID = 3
B, T = 4, 512      # batch × frames (≈ 5 seconds of audio after 4× downsampling)


def make_timemix(seed: int = 0) -> RWKV6TimeMix:
    torch.manual_seed(seed)
    tm = RWKV6TimeMix(
        hidden_size=D_MODEL,
        n_head=N_HEAD,
        head_size=HEAD_SIZE,
        num_hidden_layers=N_LAYERS,
        layer_id=LAYER_ID,
        mode="recurrent",
        rse=True,
        rse_theta_init_scale=math.pi / 16,
        rse_theta_clip=math.pi / 2,
        rse_theta_lora_dim=48,
        rse_viscosity=True,
    ).cuda()
    # Give viscosity_eta a nontrivial value so the equivalence test covers
    # the eta ≠ 0 path (the whole point of the Stage-5 refinement).
    with torch.no_grad():
        tm.viscosity_eta.uniform_(0.02, 0.05)
    tm.eval()
    return tm


def sample_rse_inputs(seed: int = 1):
    """Build (r, k, v, w, theta, state) consistent with the TimeMix's forward."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    K = HEAD_SIZE
    Bk = K // 2

    r = torch.randn(B, N_HEAD, T, K, device=device) * 0.3
    k = torch.randn(B, N_HEAD, T, K, device=device) * 0.3
    v = torch.randn(B, N_HEAD, T, K, device=device) * 0.3
    # log-decay must be negative and not too large in magnitude
    w = -torch.rand(B, N_HEAD, T, K, device=device) * 0.1 - 1e-3
    # theta: small angles per 2x2 block
    theta = torch.randn(B, N_HEAD, T, Bk, device=device) * (math.pi / 16)
    return r, k, v, w, theta


@torch.no_grad()
def forward_ref(tm: RWKV6TimeMix, r, k, v, w, theta, state=None):
    return tm._forward_recurrent_rse(r, k, v, w, theta, state=state, apply_bonus=True)


def forward_fast(tm: RWKV6TimeMix, r, k, v, w, theta, state=None):
    eta = tm.viscosity_eta if tm.use_viscosity else None
    u = None if tm.drop_u else tm.time_faaaa
    return rse_viscosity_scan(r, k, v, w, theta, eta=eta, u=u, state=state)


def equivalence_test() -> None:
    tm = make_timemix()
    r, k, v, w, theta = sample_rse_inputs()

    # Forward equivalence (no grad)
    y_ref, s_ref = forward_ref(tm, r, k, v, w, theta)
    y_fast, s_fast = forward_fast(tm, r, k, v, w, theta)

    y_err = (y_ref - y_fast).abs().max().item()
    s_err = (s_ref - s_fast).abs().max().item()
    print(f"[forward] max |y_ref - y_fast|      = {y_err:.3e}")
    print(f"[forward] max |state_ref - state_f| = {s_err:.3e}")
    assert y_err < 5e-4, f"forward output mismatch: {y_err}"
    assert s_err < 5e-4, f"forward state mismatch: {s_err}"

    # Backward equivalence — gradients wrt r, k, v, w, theta, and parameters
    for tensor in (r, k, v, w, theta):
        tensor.requires_grad_(True)
    tm_grad = make_timemix()  # fresh module, grads on parameters
    loss_ref = forward_ref.__wrapped__ if hasattr(forward_ref, '__wrapped__') else None

    # Re-run forward with grad enabled (we can't use the no_grad wrapper above)
    with torch.enable_grad():
        y_ref_g, _ = tm_grad._forward_recurrent_rse(r, k, v, w, theta, state=None, apply_bonus=True)
        loss_ref = y_ref_g.square().sum()
        g_ref = torch.autograd.grad(
            loss_ref,
            [r, k, v, w, theta, tm_grad.viscosity_eta, tm_grad.time_faaaa],
            retain_graph=False,
            create_graph=False,
        )

    for tensor in (r, k, v, w, theta):
        tensor.grad = None
    tm_grad.zero_grad(set_to_none=True)

    with torch.enable_grad():
        eta_g = tm_grad.viscosity_eta
        u_g = tm_grad.time_faaaa
        y_fast_g, _ = rse_viscosity_scan(r, k, v, w, theta, eta=eta_g, u=u_g, state=None)
        loss_fast = y_fast_g.square().sum()
        g_fast = torch.autograd.grad(
            loss_fast,
            [r, k, v, w, theta, eta_g, u_g],
            retain_graph=False,
            create_graph=False,
        )

    names = ["r", "k", "v", "w", "theta", "eta", "u(time_faaaa)"]
    for name, ga, gb in zip(names, g_ref, g_fast):
        err = (ga - gb).abs().max().item()
        scale = ga.abs().max().item() + 1e-12
        print(f"[backward] grad {name:<16s} max |Δ| = {err:.3e}  (scale {scale:.3e})")
        assert err / scale < 5e-3, f"gradient mismatch on {name}: {err}"

    print("\n✓ numerical equivalence OK (forward + backward)\n")


def time_forward(fn, *args, warmup: int = 3, iters: int = 20) -> float:
    """Returns median ms per iteration."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def time_forward_backward(fn, *args, warmup: int = 3, iters: int = 20) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        y, _ = fn(*args)
        y.square().sum().backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        for a in args:
            if isinstance(a, torch.Tensor) and a.grad is not None:
                a.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y, _ = fn(*args)
        y.square().sum().backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def speed_benchmark() -> None:
    tm = make_timemix()
    r, k, v, w, theta = sample_rse_inputs()
    for t in (r, k, v, w, theta):
        t.requires_grad_(True)

    print(f"Shapes: B={B}, H={N_HEAD}, T={T}, K={HEAD_SIZE}  (Bk={HEAD_SIZE//2})")
    print("-" * 70)

    # --- Forward only ---
    ref_ms = time_forward(lambda: forward_ref(tm, r.detach(), k.detach(), v.detach(), w.detach(), theta.detach()))
    fast_ms = time_forward(lambda: forward_fast(tm, r.detach(), k.detach(), v.detach(), w.detach(), theta.detach()))
    print(f"forward   | ref  {ref_ms:7.2f} ms   fast  {fast_ms:7.2f} ms   speedup  {ref_ms/fast_ms:4.2f}x")

    # --- Forward+backward ---
    def fb_ref():
        return forward_ref(tm, r.detach().requires_grad_(True),
                           k.detach().requires_grad_(True),
                           v.detach().requires_grad_(True),
                           w.detach().requires_grad_(True),
                           theta.detach().requires_grad_(True))

    def fb_fast():
        return forward_fast(tm, r.detach().requires_grad_(True),
                            k.detach().requires_grad_(True),
                            v.detach().requires_grad_(True),
                            w.detach().requires_grad_(True),
                            theta.detach().requires_grad_(True))

    # forward_ref uses @torch.no_grad — strip it for the bwd benchmark
    def fb_ref_grad():
        y, _ = tm._forward_recurrent_rse(
            r.detach().requires_grad_(True),
            k.detach().requires_grad_(True),
            v.detach().requires_grad_(True),
            w.detach().requires_grad_(True),
            theta.detach().requires_grad_(True),
            state=None, apply_bonus=True,
        )
        y.square().sum().backward()

    def fb_fast_grad():
        eta = tm.viscosity_eta
        u = None if tm.drop_u else tm.time_faaaa
        y, _ = rse_viscosity_scan(
            r.detach().requires_grad_(True),
            k.detach().requires_grad_(True),
            v.detach().requires_grad_(True),
            w.detach().requires_grad_(True),
            theta.detach().requires_grad_(True),
            eta=eta, u=u, state=None,
        )
        y.square().sum().backward()

    torch.cuda.synchronize()
    for _ in range(3):
        fb_ref_grad()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        fb_ref_grad()
    torch.cuda.synchronize()
    ref_fb = (time.perf_counter() - t0) * 100.0  # ms per iter

    torch.cuda.synchronize()
    for _ in range(3):
        fb_fast_grad()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        fb_fast_grad()
    torch.cuda.synchronize()
    fast_fb = (time.perf_counter() - t0) * 100.0

    print(f"fwd+bwd   | ref  {ref_fb:7.2f} ms   fast  {fast_fb:7.2f} ms   speedup  {ref_fb/fast_fb:4.2f}x")

    # --- bf16 autocast on fast path ---
    def fb_fast_bf16():
        eta = tm.viscosity_eta
        u = None if tm.drop_u else tm.time_faaaa
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y, _ = rse_viscosity_scan(
                r.detach().requires_grad_(True),
                k.detach().requires_grad_(True),
                v.detach().requires_grad_(True),
                w.detach().requires_grad_(True),
                theta.detach().requires_grad_(True),
                eta=eta, u=u, state=None,
            )
            y.square().sum().backward()

    torch.cuda.synchronize()
    for _ in range(3):
        fb_fast_bf16()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        fb_fast_bf16()
    torch.cuda.synchronize()
    fast_bf16 = (time.perf_counter() - t0) * 100.0

    print(f"fwd+bwd bf16 autocast | fast  {fast_bf16:7.2f} ms   total speedup vs ref  {ref_fb/fast_bf16:4.2f}x")

    # --- VRAM ---
    torch.cuda.reset_peak_memory_stats()
    fb_ref_grad()
    torch.cuda.synchronize()
    ref_mem = torch.cuda.max_memory_allocated() / 1024**3

    torch.cuda.reset_peak_memory_stats()
    fb_fast_grad()
    torch.cuda.synchronize()
    fast_mem = torch.cuda.max_memory_allocated() / 1024**3

    torch.cuda.reset_peak_memory_stats()
    fb_fast_bf16()
    torch.cuda.synchronize()
    fast_mem_bf16 = torch.cuda.max_memory_allocated() / 1024**3

    print("-" * 70)
    print(f"peak VRAM (fwd+bwd) | ref {ref_mem:5.2f} GB   fast {fast_mem:5.2f} GB   fast+bf16 {fast_mem_bf16:5.2f} GB")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    print("=" * 70)
    print("RSE + viscosity: numerical equivalence")
    print("=" * 70)
    equivalence_test()
    print("=" * 70)
    print("RSE + viscosity: speed benchmark")
    print("=" * 70)
    speed_benchmark()
