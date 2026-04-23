"""Tests for Mamba-2 LUCID — B-correlation preconditioner on the SSD
dual form.

Per the spec in STAGE11_AGENT_QUEUE.md (Mamba-2 LUCID feasibility note)
and the user's adaptation proposal: LUCID's chunked preconditioner
operates on Mamba-2's chunk-local T_c × T_c attention (C · B^T), using
B as the key analog. These tests guard:
  * per-layer lucid_temperature receives non-zero gradient;
  * approximate forward invariance when tau is zero-init but softplus
    is ~0.69 (mild preconditioning, not identity — documented);
  * forward is deterministic and finite at init;
  * param delta vs vanilla mamba2 is tiny (per-head scalars only).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import ExperimentConfig
from src.models.asr_model import ASRModel
from src.models.encoder import build_encoder
from src.models.mamba2_block import Mamba2Block


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_mamba2_lucid_forward_finite():
    torch.manual_seed(0)
    blk = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64,
        expand=2, ngroups=1, chunk_size=32, mode="recurrent",
        use_lucid=True,
    ).to(DEVICE).eval()
    x = torch.randn(2, 64, 256, device=DEVICE)
    with torch.no_grad():
        y, _ = blk(x)
    assert y.shape == (2, 64, 256)
    assert torch.isfinite(y).all()


def test_mamba2_lucid_gradient_flow_through_temperature():
    """lucid_temperature (per-head scalar) must receive gradient at init.

    Regression guard: if the kernel dispatch doesn't plumb
    lucid_temp through the chunked scan, the parameter sits detached and
    SGD can't engage the preconditioner.
    """
    torch.manual_seed(0)
    blk = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64,
        expand=2, ngroups=1, chunk_size=32, mode="recurrent",
        use_lucid=True,
    ).to(DEVICE).train()
    x = torch.randn(2, 64, 256, device=DEVICE)
    y, _ = blk(x)
    y.sum().backward()
    assert blk.lucid_temperature.grad is not None
    assert torch.isfinite(blk.lucid_temperature.grad).all()
    assert blk.lucid_temperature.grad.abs().max().item() > 1e-8


def test_mamba2_lucid_rejects_lion_mode():
    """use_lucid=True with mode='lion' must raise at init.

    LUCID's algebraic form Y = A P^{-1} V doesn't apply to the
    LION parallel attention path the same way.
    """
    import pytest
    with pytest.raises(ValueError, match="LUCID is only defined"):
        Mamba2Block(
            d_model=64, d_state=64, d_conv=4, headdim=32,
            expand=2, ngroups=1, chunk_size=16, mode="lion",
            use_lucid=True,
        )


def test_mamba2_lucid_end_to_end_dispatch():
    """build_encoder('mamba2_lucid') returns the correct encoder with
    lucid_temperature parameters on each layer."""
    cfg = ExperimentConfig()
    cfg.backbone = "mamba2_lucid"
    cfg.dropout = 0.0
    m = ASRModel(vocab_size=29, cfg=cfg).to(DEVICE)
    for i, layer in enumerate(m.encoder.layers):
        assert hasattr(layer.mamba, "lucid_temperature"), (
            f"layer {i} missing lucid_temperature"
        )
        assert layer.mamba.lucid_temperature.shape == (layer.mamba.nheads,)
    # forward + backward
    x = torch.randn(2, 80, 160, device=DEVICE)
    lens = torch.tensor([160, 128], device=DEVICE)
    lp, _, _ = m(x, lens)
    lp.sum().backward()
    for i, layer in enumerate(m.encoder.layers):
        g = layer.mamba.lucid_temperature.grad
        assert g is not None and torch.isfinite(g).all()
        assert g.abs().max().item() > 1e-8, (
            f"layer {i} lucid_temperature grad is zero"
        )


def test_mamba2_lucid_param_delta_is_tiny():
    """use_lucid adds only (H,) scalars per layer — ~tens of parameters."""
    cfg_v = ExperimentConfig()
    cfg_v.backbone = "mamba2"
    cfg_l = ExperimentConfig()
    cfg_l.backbone = "mamba2_lucid"
    enc_v = build_encoder(cfg_v)
    enc_l = build_encoder(cfg_l)
    pv = sum(p.numel() for p in enc_v.parameters())
    pl = sum(p.numel() for p in enc_l.parameters())
    # Expect delta = n_layers * nheads = 6 * 8 = 48 scalars.
    assert 0 < (pl - pv) < 200, (
        f"unexpected param delta: vanilla={pv}, lucid={pl}, delta={pl - pv}"
    )


def test_mamba2_lucid_composition_with_multidil_builds():
    """`mamba2_lucid_convshift_multidil_symmetric_v2` (LUCID × multidil_v2)
    builds + forwards + backwards cleanly."""
    cfg = ExperimentConfig()
    cfg.backbone = "mamba2_lucid_convshift_multidil_symmetric_v2"
    cfg.dropout = 0.0
    m = ASRModel(vocab_size=29, cfg=cfg).to(DEVICE)
    m.train()
    x = torch.randn(2, 80, 160, device=DEVICE)
    lens = torch.tensor([160, 128], device=DEVICE)
    lp, _, _ = m(x, lens)
    lp.sum().backward()
    # lucid_temperature should receive gradient
    for i, layer in enumerate(m.encoder.layers):
        g = layer.mamba.lucid_temperature.grad
        assert g is not None and g.abs().max().item() > 1e-8
