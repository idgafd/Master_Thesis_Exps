"""Correctness + compile-parity tests for mamba2_kernels.

Runs against a tiny in-file reference implementation of the SSD scan
(hand-coded sequential recurrence) so we don't depend on mamba-ssm being
installed.  Bidirectional tests compare ``ssd_scan_lion`` against
``ssd_scan_lion_chunk`` (both must be bidirectional, must agree) and
against a manual "forward + reversed" construction.

Usage:
    uv run python -m pytest tests/test_mamba2_kernels.py -q
    or
    uv run python tests/test_mamba2_kernels.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.models.mamba2_kernels import (
    ssd_scan_causal,
    ssd_scan_lion,
    ssd_scan_lion_chunk,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Reference: slow, explicit sequential recurrence (O(T · H · P · N)).
# ---------------------------------------------------------------------------

def ref_ssd_causal(X, dt, A, B, C, state=None):
    """Plain sequential recurrence matching the SSD semantics.

    h[t+1] = exp(dt[t]*A) * h[t] + dt[t] * B[t] ⊗ X[t]
    y[t]   = C[t]^T · h[t]                (sum over state-dim)
    """
    Bsz, L, H, P = X.shape
    N = B.shape[-1]
    h = (
        torch.zeros(Bsz, H, P, N, device=X.device, dtype=torch.float32)
        if state is None
        else state.float().clone()
    )
    Ys = []
    for t in range(L):
        dA = torch.exp(dt[:, t] * A.view(1, H))           # (Bsz, H)
        dB = dt[:, t].unsqueeze(-1) * B[:, t]             # (Bsz, H, N)
        x_t = X[:, t].float()                             # (Bsz, H, P)
        h = h * dA.unsqueeze(-1).unsqueeze(-1) + x_t.unsqueeze(-1) * dB.unsqueeze(-2)
        y_t = (h * C[:, t].unsqueeze(-2)).sum(-1)         # (Bsz, H, P)
        Ys.append(y_t)
    return torch.stack(Ys, dim=1).to(X.dtype), h.to(X.dtype)


def ref_ssd_bidir(X, dt, A, B, C):
    """Bidirectional reference: forward recurrence summed with reversed recurrence.

    Matches what ``ssd_scan_lion`` should compute (diagonal counted once:
    forward includes j=i, backward is strict j > i).
    """
    Y_F, _ = ref_ssd_causal(X, dt, A, B, C)
    # Backward: run causal on reversed inputs, then flip result.
    Y_R, _ = ref_ssd_causal(X.flip(1), dt.flip(1), A, B.flip(1), C.flip(1))
    Y_B = Y_R.flip(1)
    # Forward has j ≤ i, Y_B (flipped) has j ≥ i, diagonal counted twice.
    diag_attn = (C * B).sum(-1)                            # (B, L, H)
    diag_val = dt.unsqueeze(-1) * X                        # (B, L, H, P)
    Y_diag = diag_attn.unsqueeze(-1) * diag_val
    return Y_F + Y_B - Y_diag


# ---------------------------------------------------------------------------
# Input generator
# ---------------------------------------------------------------------------

def _rand_inputs(Bsz=2, L=128, H=4, P=32, N=16, seed=0, device=DEVICE, dtype=DTYPE):
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(Bsz, L, H, P, device=device, dtype=dtype, generator=g)
    # dt after softplus: positive, typical scale 0.01-0.1
    dt = torch.nn.functional.softplus(
        torch.randn(Bsz, L, H, device=device, dtype=dtype, generator=g) - 4
    )
    A = -torch.rand(H, device=device, dtype=dtype, generator=g) * 15 - 1  # (-16, -1)
    Bm = torch.randn(Bsz, L, H, N, device=device, dtype=dtype, generator=g)
    Cm = torch.randn(Bsz, L, H, N, device=device, dtype=dtype, generator=g)
    return X, dt, A, Bm, Cm


def _maxerr(a, b):
    return (a - b).abs().max().item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_causal_matches_reference():
    X, dt, A, B, C = _rand_inputs(Bsz=2, L=128, H=4, P=32, N=16)
    for cs in [16, 32, 64, 128]:
        y_ssd, s_ssd = ssd_scan_causal(X, dt, A, B, C, chunk_size=cs)
        y_ref, s_ref = ref_ssd_causal(X, dt, A, B, C)
        err_y = _maxerr(y_ssd, y_ref)
        err_s = _maxerr(s_ssd, s_ref)
        rel_y = err_y / (y_ref.abs().max().item() + 1e-8)
        print(f"[causal cs={cs:3d}]  max |Δy| = {err_y:.2e} (rel {rel_y:.1e})   "
              f"max |Δstate| = {err_s:.2e}")
        assert rel_y < 1e-4, f"chunk {cs}: relative error too large: {rel_y}"
        assert err_s < 1e-4, f"chunk {cs}: state mismatch: {err_s}"


def test_causal_chunk_invariance():
    """Output must be independent of chunk_size (same semantics, different blocking)."""
    X, dt, A, B, C = _rand_inputs(Bsz=2, L=256, H=4, P=32, N=16)
    y_ref, _ = ssd_scan_causal(X, dt, A, B, C, chunk_size=64)
    for cs in [16, 32, 128, 256]:
        y, _ = ssd_scan_causal(X, dt, A, B, C, chunk_size=cs)
        err = _maxerr(y, y_ref) / (y_ref.abs().max().item() + 1e-8)
        print(f"[chunk-invariance cs={cs:3d}]  rel Δ = {err:.1e}")
        assert err < 1e-4, f"chunk_size={cs} diverges: {err}"


def test_causal_initial_state():
    """Running with half the sequence + carry the state must equal full-length."""
    X, dt, A, B, C = _rand_inputs(Bsz=2, L=128, H=4, P=32, N=16)
    L = X.size(1)
    half = L // 2
    y_full, s_full = ssd_scan_causal(X, dt, A, B, C, chunk_size=32)
    y1, s1 = ssd_scan_causal(X[:, :half], dt[:, :half], A, B[:, :half], C[:, :half],
                             chunk_size=32)
    y2, s2 = ssd_scan_causal(X[:, half:], dt[:, half:], A, B[:, half:], C[:, half:],
                             chunk_size=32, state=s1)
    y_split = torch.cat([y1, y2], dim=1)
    err_y = _maxerr(y_split, y_full) / (y_full.abs().max().item() + 1e-8)
    err_s = _maxerr(s2, s_full)
    print(f"[carry-state]  rel Δy = {err_y:.1e}   Δstate = {err_s:.2e}")
    assert err_y < 1e-4
    assert err_s < 1e-4


def test_lion_full_matches_bidir_reference():
    X, dt, A, B, C = _rand_inputs(Bsz=2, L=96, H=4, P=32, N=16)
    y_lion = ssd_scan_lion(X, dt, A, B, C)
    y_ref = ref_ssd_bidir(X, dt, A, B, C)
    err = _maxerr(y_lion, y_ref) / (y_ref.abs().max().item() + 1e-8)
    print(f"[lion full vs bidir-ref]  rel Δ = {err:.1e}")
    assert err < 1e-4


def test_lion_chunk_matches_lion_full():
    X, dt, A, B, C = _rand_inputs(Bsz=2, L=128, H=4, P=32, N=16)
    y_full = ssd_scan_lion(X, dt, A, B, C)
    for cs in [16, 32, 64, 128]:
        y_chunk = ssd_scan_lion_chunk(X, dt, A, B, C, chunk_size=cs)
        err = _maxerr(y_chunk, y_full) / (y_full.abs().max().item() + 1e-8)
        print(f"[lion_chunk cs={cs:3d} vs lion_full]  rel Δ = {err:.1e}")
        assert err < 1e-4, f"chunk {cs}: {err}"


def test_compile_parity():
    X, dt, A, B, C = _rand_inputs(Bsz=1, L=64, H=2, P=16, N=8)

    def _call(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    # Causal
    y_eager, s_eager = ssd_scan_causal(X, dt, A, B, C, chunk_size=32)
    ssd_compiled = torch.compile(ssd_scan_causal, mode="default", dynamic=False)
    y_comp, s_comp = ssd_compiled(X, dt, A, B, C, chunk_size=32)
    err_y = _maxerr(y_comp, y_eager) / (y_eager.abs().max().item() + 1e-8)
    err_s = _maxerr(s_comp, s_eager)
    print(f"[compile causal]  rel Δy = {err_y:.1e}   Δstate = {err_s:.2e}")
    assert err_y < 1e-4 and err_s < 1e-4

    # Lion full
    y_eager = ssd_scan_lion(X, dt, A, B, C)
    y_comp = torch.compile(ssd_scan_lion, mode="default", dynamic=False)(X, dt, A, B, C)
    err = _maxerr(y_comp, y_eager) / (y_eager.abs().max().item() + 1e-8)
    print(f"[compile lion-full]  rel Δ = {err:.1e}")
    assert err < 1e-4


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_causal_matches_reference,
        test_causal_chunk_invariance,
        test_causal_initial_state,
        test_lion_full_matches_bidir_reference,
        test_lion_chunk_matches_lion_full,
        test_compile_parity,
    ]
    n_pass = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  ✓ {name}")
            n_pass += 1
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
    print(f"\n{n_pass}/{len(tests)} passed")
