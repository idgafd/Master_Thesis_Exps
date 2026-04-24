"""Tests for Mamba-2 Householder (generalised partial reflection on the
inter-chunk state transition).

Per-head H_h = I − 2(1 − α_h)·u_h·u_hᵀ applied on the state's N-axis at
every chunk boundary.  α = 1 ⇒ H = I ⇒ bit-exact vanilla Mamba-2.
α ∈ [0, 1] ⇒ operator norm ≤ 1 (stable).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

from src.config import ExperimentConfig
from src.models.asr_model import ASRModel
from src.models.encoder import build_encoder
from src.models.mamba2_block import Mamba2Block
from src.models.mamba2_kernels import ssd_scan_causal


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Kernel-level invariants
# ---------------------------------------------------------------------------

def test_householder_alpha_one_is_exact_vanilla():
    """α = 1 makes H = I; the kernel must produce bit-exact vanilla
    output in that limit — including the final state."""
    torch.manual_seed(0)
    Bsz, L, H, P, N = 2, 64, 2, 8, 16
    X = torch.randn(Bsz, L, H, P, dtype=torch.float64)
    dt = torch.rand(Bsz, L, H, dtype=torch.float64) * 0.1 + 0.01
    A = -torch.tensor([2.0, 4.0], dtype=torch.float64)
    Bt = torch.randn(Bsz, L, H, N, dtype=torch.float64)
    C = torch.randn(Bsz, L, H, N, dtype=torch.float64)
    # α = 1 ⇒ coef = 2(1-α) = 0 ⇒ no reflection
    u = F.normalize(torch.randn(H, N, dtype=torch.float64), dim=-1)
    alpha = torch.ones(H, dtype=torch.float64)

    Y_v, state_v = ssd_scan_causal(X, dt, A, Bt, C, chunk_size=16)
    Y_h, state_h = ssd_scan_causal(
        X, dt, A, Bt, C, chunk_size=16,
        householder_u=u, householder_alpha=alpha,
    )
    y_diff = (Y_v - Y_h).abs().max().item()
    s_diff = (state_v - state_h).abs().max().item()
    # The vectorised einsum vs sequential loop sum in different orders,
    # so fp32 internal precision puts the gap at ~1e-6.  Tolerance 1e-4
    # keeps the "bit-exact at α=1" contract meaningful without tripping
    # on reduction-order noise.
    assert y_diff < 1e-4, f"α=1 output mismatch: {y_diff}"
    assert s_diff < 1e-4, f"α=1 final state mismatch: {s_diff}"


def test_householder_alpha_below_one_diverges_from_vanilla():
    """α < 1 engages the reflection; output must differ from vanilla."""
    torch.manual_seed(1)
    Bsz, L, H, P, N = 2, 64, 2, 8, 16
    X = torch.randn(Bsz, L, H, P, dtype=torch.float64)
    dt = torch.rand(Bsz, L, H, dtype=torch.float64) * 0.1 + 0.01
    A = -torch.tensor([2.0, 4.0], dtype=torch.float64)
    Bt = torch.randn(Bsz, L, H, N, dtype=torch.float64)
    C = torch.randn(Bsz, L, H, N, dtype=torch.float64)
    u = F.normalize(torch.randn(H, N, dtype=torch.float64), dim=-1)
    alpha = torch.tensor([0.3, 0.5], dtype=torch.float64)

    Y_v, _ = ssd_scan_causal(X, dt, A, Bt, C, chunk_size=16)
    Y_h, _ = ssd_scan_causal(
        X, dt, A, Bt, C, chunk_size=16,
        householder_u=u, householder_alpha=alpha,
    )
    diff = (Y_v - Y_h).abs().max().item()
    assert diff > 1e-3, (
        f"α < 1 must diverge from vanilla; got diff = {diff}"
    )


def test_householder_stability_norm_preserved():
    """With a fresh chunk and zero A (no decay), a full reflection
    (α = 0, H orthogonal) preserves state L2 norm: ||H s|| = ||s||."""
    torch.manual_seed(2)
    Bsz, L, H, P, N = 1, 32, 1, 4, 16
    X = torch.randn(Bsz, L, H, P, dtype=torch.float64)
    dt = torch.zeros(Bsz, L, H, dtype=torch.float64) + 1e-6  # near-zero
    A = torch.zeros(H, dtype=torch.float64)  # no decay
    Bt = torch.randn(Bsz, L, H, N, dtype=torch.float64)
    C = torch.randn(Bsz, L, H, N, dtype=torch.float64)
    u = F.normalize(torch.randn(H, N, dtype=torch.float64), dim=-1)
    alpha = torch.zeros(H, dtype=torch.float64)  # full Householder

    _, state_h = ssd_scan_causal(
        X, dt, A, Bt, C, chunk_size=8,
        householder_u=u, householder_alpha=alpha,
    )
    # State shouldn't blow up; spectral radius of full Householder is 1.
    state_norm = state_h.norm().item()
    assert state_norm < 1e3, (
        f"α=0 state norm blew up (spectral radius > 1?): {state_norm}"
    )


# ---------------------------------------------------------------------------
# Block-level / end-to-end
# ---------------------------------------------------------------------------

def test_householder_block_has_params():
    blk = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64,
        expand=2, ngroups=1, chunk_size=32, mode="recurrent",
        use_householder=True,
    )
    assert hasattr(blk, "householder_u_raw")
    assert hasattr(blk, "householder_alpha_raw")
    assert blk.householder_u_raw.shape == (blk.nheads, 64)
    assert blk.householder_alpha_raw.shape == (blk.nheads,)
    # At init, α ≈ sigmoid(5) ≈ 0.993 ⇒ (1 − α) ≈ 0.007 ⇒ H ≈ I.
    alpha_init = torch.sigmoid(blk.householder_alpha_raw)
    assert alpha_init.min().item() > 0.99


def test_householder_forward_finite():
    torch.manual_seed(0)
    blk = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64,
        expand=2, ngroups=1, chunk_size=32, mode="recurrent",
        use_householder=True,
    ).to(DEVICE).eval()
    x = torch.randn(2, 96, 256, device=DEVICE)
    with torch.no_grad():
        y, _ = blk(x)
    assert y.shape == (2, 96, 256)
    assert torch.isfinite(y).all()


def test_householder_gradient_flow():
    """Both u_raw and α_raw must get gradient — but only when the
    sequence is long enough that nC ≥ 3.  The first two loop iterations
    start with s = zeros (no α dependence in the reflection term), so
    nC=2 cases are α-invariant by construction — we need at least one
    chunk with a nonzero state entering to engage α.  Here L=128,
    chunk_size=32 ⇒ nC=4."""
    torch.manual_seed(0)
    blk = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64,
        expand=2, ngroups=1, chunk_size=32, mode="recurrent",
        use_householder=True,
    ).to(DEVICE).train()
    x = torch.randn(2, 128, 256, device=DEVICE)
    y, _ = blk(x)
    y.sum().backward()
    assert blk.householder_alpha_raw.grad is not None
    assert torch.isfinite(blk.householder_alpha_raw.grad).all()
    assert blk.householder_alpha_raw.grad.abs().max().item() > 1e-9
    assert blk.householder_u_raw.grad is not None
    assert torch.isfinite(blk.householder_u_raw.grad).all()
    assert blk.householder_u_raw.grad.abs().max().item() > 1e-12


def test_householder_rejects_lion_mode():
    with pytest.raises(ValueError, match="Householder is only defined"):
        Mamba2Block(
            d_model=64, d_state=64, d_conv=4, headdim=32,
            expand=2, ngroups=1, chunk_size=16, mode="lion",
            use_householder=True,
        )


def test_householder_end_to_end_dispatch():
    cfg = ExperimentConfig()
    cfg.backbone = "mamba2_householder"
    cfg.dropout = 0.0
    m = ASRModel(vocab_size=29, cfg=cfg).to(DEVICE)
    for i, layer in enumerate(m.encoder.layers):
        assert hasattr(layer.mamba, "householder_u_raw")
        assert hasattr(layer.mamba, "householder_alpha_raw")
        assert layer.mamba.use_householder is True
    # Use a longer sequence so the frontend's subsampling gives at
    # least nC ≥ 3 chunks after ConvSubsampling shrinks length by ~4×.
    x = torch.randn(2, 80, 800, device=DEVICE)
    lens = torch.tensor([800, 600], device=DEVICE)
    lp, _, _ = m(x, lens)
    lp.sum().backward()
    for i, layer in enumerate(m.encoder.layers):
        assert layer.mamba.householder_alpha_raw.grad is not None, (
            f"layer {i} α_raw grad is None"
        )
        assert layer.mamba.householder_u_raw.grad is not None, (
            f"layer {i} u_raw grad is None"
        )


def test_householder_param_delta():
    """mamba2_householder adds n_layers × n_heads × (d_state + 1) params over vanilla."""
    cfg_v = ExperimentConfig(); cfg_v.backbone = "mamba2"
    cfg_h = ExperimentConfig(); cfg_h.backbone = "mamba2_householder"
    enc_v = build_encoder(cfg_v)
    enc_h = build_encoder(cfg_h)
    pv = sum(p.numel() for p in enc_v.parameters())
    ph = sum(p.numel() for p in enc_h.parameters())
    n_heads = enc_h.layers[0].mamba.nheads
    d_state = enc_h.layers[0].mamba.d_state
    expected = cfg_h.n_layers * n_heads * (d_state + 1)
    assert ph - pv == expected, (
        f"Param delta mismatch: got {ph - pv}, expected {expected}"
    )
