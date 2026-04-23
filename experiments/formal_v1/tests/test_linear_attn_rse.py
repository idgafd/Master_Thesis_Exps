"""Tests for Stage 11.2b — linear_attn_rse_strong_viscosity.

Contract:
  * Causality: output at t depends only on inputs <= t.
  * Finite output with no NaN / Inf at init (numerical stability).
  * theta_base = 0, LoRA weights = 0, eta = 0 at init ⇒ theta_t ≡ 0 so
    z_t = exp(-lambda_base) is real-valued (no rotation); imaginary part
    of the complex state stays at 0 in the absence of paired imag k.
  * chunk-size invariance: output does not depend on rse_chunk_size.
  * Param parity within 5% of vanilla LA.
  * State carry shape round-trip.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import ExperimentConfig
from src.models.encoder import build_encoder
from src.models.linear_attn_rse import (
    CausalLinearAttentionRSEEncoder,
    CausalLinearAttentionRSELayer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maxerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _mk_layer(d_model: int = 64, n_heads: int = 4, ffn_dim: int = 128,
              rse_chunk_size: int = 16) -> CausalLinearAttentionRSELayer:
    torch.manual_seed(0)
    layer = CausalLinearAttentionRSELayer(
        d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim, dropout=0.0,
        rse_chunk_size=rse_chunk_size,
    ).to(DEVICE)
    layer.eval()
    return layer


def test_forward_is_finite():
    layer = _mk_layer()
    torch.manual_seed(1)
    x = torch.randn(2, 20, 64, device=DEVICE)
    with torch.no_grad():
        y, _ = layer.forward_parallel(x)
    assert torch.isfinite(y).all(), "output contains NaN/Inf"


def test_causality():
    """Perturbing x[:, t_cut:] leaves output at positions < t_cut unchanged."""
    layer = _mk_layer()
    torch.manual_seed(2)
    x = torch.randn(2, 32, 64, device=DEVICE)
    y1, _ = layer.forward_parallel(x)
    t_cut = 16
    x2 = x.clone()
    x2[:, t_cut:, :] = torch.randn_like(x2[:, t_cut:, :])
    y2, _ = layer.forward_parallel(x2)
    err = _maxerr(y1[:, :t_cut], y2[:, :t_cut])
    assert err < 1e-4, f"causality violated: {err}"


def test_chunk_size_invariance():
    """Output must be invariant under rse_chunk_size."""
    torch.manual_seed(3)
    x = torch.randn(2, 48, 64, device=DEVICE)

    def _out_for(cs):
        torch.manual_seed(0)
        layer = CausalLinearAttentionRSELayer(
            d_model=64, n_heads=4, ffn_dim=128, dropout=0.0,
            rse_chunk_size=cs,
        ).to(DEVICE)
        layer.eval()
        with torch.no_grad():
            y, _ = layer.forward_parallel(x)
        return y

    y_small = _out_for(8)
    y_mid = _out_for(16)
    y_large = _out_for(64)  # >= T
    assert _maxerr(y_small, y_mid) < 1e-4, "chunk-size dependence: 8 vs 16"
    assert _maxerr(y_mid, y_large) < 1e-4, "chunk-size dependence: 16 vs 64"


def test_theta_is_data_independent_at_init():
    """At init, LoRA W1=0 ⇒ LoRA contribution is exactly 0 regardless of x,
    so theta_t collapses to theta_base — NO data dependence at step 0.
    Matches the Stage-3 RSE convention: symmetry-breaking uniform init on
    theta_base, zero-init LoRA for exact LoRA reduction."""
    layer = _mk_layer()
    torch.manual_seed(0)
    x = torch.randn(2, 20, 64, device=DEVICE)
    x_n = layer.norm1(x)
    theta_a = layer._compute_theta(x_n)
    # Same layer, different x — theta must be identical along T and B axes
    # since there's no data dependence at init.
    x2 = torch.randn_like(x)
    x2_n = layer.norm1(x2)
    theta_b = layer._compute_theta(x2_n)
    # theta_t should be constant across (B, T) — only (H, Bk) varies.
    assert _maxerr(theta_a, theta_b) < 1e-6, (
        "theta_t depends on x at init — LoRA W1 is not zero-init"
    )
    # Also check: theta stays within the init scale (pi/16).
    assert theta_a.abs().max().item() <= math.pi / 16 + 1e-6, (
        f"theta_base exceeds init scale pi/16: {theta_a.abs().max().item()}"
    )


def test_viscosity_eta_zero_at_init():
    """eta is zero-init ⇒ lambda_eff = lambda_base regardless of theta."""
    layer = _mk_layer()
    assert layer.viscosity_eta.abs().max().item() == 0.0


def test_state_carry_shape_roundtrip():
    """init_state returns correct shapes and forward with state accepts them."""
    cfg = ExperimentConfig()
    cfg.backbone = "linear_attn_rse_strong_viscosity"
    cfg.d_model = 256; cfg.n_heads = 4; cfg.head_size = 64; cfg.n_layers = 6
    cfg.dropout = 0.0
    enc = build_encoder(cfg).to(DEVICE).eval()
    assert isinstance(enc, CausalLinearAttentionRSEEncoder)

    state = enc.init_state(2, DEVICE)
    assert state["offset"] == 0
    assert len(state["layers"]) == 6
    # Bk = head_dim / 2 = 32
    assert state["layers"][0]["c"].shape == (2, 4, 32, 64)
    assert state["layers"][0]["c"].dtype == torch.complex64

    x = torch.randn(2, 40, 256, device=DEVICE)
    lengths = torch.tensor([40, 30], device=DEVICE)
    with torch.no_grad():
        y, new_state = enc(x, lengths, state=state)
    assert y.shape == (2, 40, 256)
    assert new_state is not None
    assert new_state["offset"] == 40


def test_param_parity_within_5pct_vs_vanilla_la():
    cfg_v = ExperimentConfig()
    cfg_v.backbone = "linear_attn_causal"
    cfg_v.d_model = 256; cfg_v.n_heads = 4; cfg_v.n_layers = 6
    cfg_r = ExperimentConfig()
    cfg_r.backbone = "linear_attn_rse_strong_viscosity"
    cfg_r.d_model = 256; cfg_r.n_heads = 4; cfg_r.n_layers = 6
    enc_v = build_encoder(cfg_v)
    enc_r = build_encoder(cfg_r)
    pv = sum(p.numel() for p in enc_v.parameters())
    pr = sum(p.numel() for p in enc_r.parameters())
    delta_pct = (pr - pv) / pv * 100
    assert abs(delta_pct) < 5.0, f"param delta {delta_pct:.2f}% exceeds 5%"


def test_backward_runs_without_nan():
    layer = _mk_layer()
    layer.train()
    torch.manual_seed(4)
    x = torch.randn(2, 16, 64, device=DEVICE, requires_grad=True)
    y, _ = layer.forward_parallel(x)
    loss = y.sum()
    loss.backward()
    # Check grads exist on all RSE parameters and are finite.
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
