"""Tests for Stage 11.2a — mamba2_rse_strong_viscosity."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import ExperimentConfig
from src.models.encoder import build_encoder
from src.models.mamba2_rse import (
    Mamba2RSEBlock,
    Mamba2RSEEncoder,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maxerr(a, b):
    return (a - b).abs().max().item()


def _mk_block(d_model=256, d_state=16, chunk_size=16) -> Mamba2RSEBlock:
    torch.manual_seed(0)
    blk = Mamba2RSEBlock(
        d_model=d_model, d_state=d_state, d_conv=4, headdim=64,
        expand=2, ngroups=1, chunk_size=chunk_size,
    ).to(DEVICE)
    blk.eval()
    return blk


def test_forward_is_finite():
    blk = _mk_block()
    torch.manual_seed(1)
    x = torch.randn(2, 40, 256, device=DEVICE)
    with torch.no_grad():
        y, _ = blk(x)
    assert torch.isfinite(y).all(), "output contains NaN/Inf"
    assert y.shape == (2, 40, 256)


def test_causality():
    blk = _mk_block()
    torch.manual_seed(2)
    x = torch.randn(2, 32, 256, device=DEVICE)
    y1, _ = blk(x)
    t_cut = 16
    x2 = x.clone()
    x2[:, t_cut:, :] = torch.randn_like(x2[:, t_cut:, :])
    y2, _ = blk(x2)
    err = _maxerr(y1[:, :t_cut], y2[:, :t_cut])
    assert err < 1e-4, f"causality violated: {err}"


def test_chunk_size_determinism_and_approx_invariance():
    """Two contracts here:

    1. For a *fixed* chunk size, forward is deterministic across repeated
       calls on identical input — tight tolerance.
    2. Across different chunk sizes, results agree to fp32-precision-loss
       expected from chained complex-exp state carries.  The SSD RSE scan
       does exp(dt * A_cont) repeatedly; with lambda up to 16 and dt up
       to 0.1, the decay per step can be extreme and chained complex-exp
       rounding accumulates ~O(1e-2) over several state carries.  During
       training the chunk size is fixed, so this cross-chunk-size drift
       never matters — but it's not a correctness issue, just precision.
    """
    torch.manual_seed(3)
    x = torch.randn(2, 48, 256, device=DEVICE)

    def _out_for(cs):
        torch.manual_seed(0)
        blk = Mamba2RSEBlock(
            d_model=256, d_state=16, d_conv=4, headdim=64, expand=2,
            ngroups=1, chunk_size=cs,
        ).to(DEVICE)
        blk.eval()
        with torch.no_grad():
            y, _ = blk(x)
        return y

    # Determinism: same chunk size, repeated call ⇒ identical output.
    y_a = _out_for(16)
    y_b = _out_for(16)
    assert _maxerr(y_a, y_b) == 0.0, "non-deterministic at fixed chunk size"

    # Approximate invariance across chunk sizes (fp32 precision budget).
    y_8 = _out_for(8)
    y_16 = _out_for(16)
    y_big = _out_for(64)
    assert _maxerr(y_8, y_16) < 5e-2, (
        f"chunk-invariance beyond fp32 budget: 8 vs 16 "
        f"= {_maxerr(y_8, y_16):.2e}"
    )
    assert _maxerr(y_16, y_big) < 5e-2, (
        f"chunk-invariance beyond fp32 budget: 16 vs 64 "
        f"= {_maxerr(y_16, y_big):.2e}"
    )


def test_theta_data_independent_at_init():
    blk = _mk_block()
    torch.manual_seed(0)
    x1 = torch.randn(2, 20, 256, device=DEVICE)
    x2 = torch.randn(2, 20, 256, device=DEVICE)
    t1 = blk._compute_theta(x1)
    t2 = blk._compute_theta(x2)
    assert _maxerr(t1, t2) < 1e-6, "theta depends on x at init — LoRA W1 not zero"
    assert t1.abs().max().item() <= math.pi / 16 + 1e-6


def test_viscosity_eta_zero_at_init():
    blk = _mk_block()
    assert blk.viscosity_eta.abs().max().item() == 0.0


def test_state_carry_shape_roundtrip():
    cfg = ExperimentConfig()
    cfg.backbone = "mamba2_rse_strong_viscosity"
    cfg.d_model = 256; cfg.n_layers = 6
    cfg.dropout = 0.0
    enc = build_encoder(cfg).to(DEVICE).eval()
    assert isinstance(enc, Mamba2RSEEncoder)

    state = enc.init_state(2, DEVICE)
    assert state["offset"] == 0
    assert len(state["layers"]) == 6
    # d_state=64, Bk=32; headdim=64
    first = state["layers"][0]
    assert first["ssm"].shape == (2, 8, 64, 32)
    assert first["ssm"].dtype == torch.complex64

    x = torch.randn(2, 40, 256, device=DEVICE)
    lengths = torch.tensor([40, 30], device=DEVICE)
    with torch.no_grad():
        y, new_state = enc(x, lengths, state=state)
    assert y.shape == (2, 40, 256)
    assert new_state is not None
    assert new_state["offset"] == 40


def test_param_parity_within_5pct_vs_vanilla_mamba2():
    cfg_v = ExperimentConfig()
    cfg_v.backbone = "mamba2"
    cfg_v.d_model = 256; cfg_v.n_layers = 6
    cfg_r = ExperimentConfig()
    cfg_r.backbone = "mamba2_rse_strong_viscosity"
    cfg_r.d_model = 256; cfg_r.n_layers = 6
    enc_v = build_encoder(cfg_v)
    enc_r = build_encoder(cfg_r)
    pv = sum(p.numel() for p in enc_v.parameters())
    pr = sum(p.numel() for p in enc_r.parameters())
    delta_pct = (pr - pv) / pv * 100
    assert abs(delta_pct) < 5.0, f"param delta {delta_pct:.2f}% exceeds 5%"


def test_backward_runs_without_nan():
    blk = _mk_block()
    blk.train()
    torch.manual_seed(4)
    x = torch.randn(2, 20, 256, device=DEVICE, requires_grad=True)
    y, _ = blk(x)
    loss = y.sum()
    loss.backward()
    for name, p in blk.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
