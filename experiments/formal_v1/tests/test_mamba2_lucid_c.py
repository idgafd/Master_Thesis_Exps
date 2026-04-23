"""Smoke test for the C-side LUCID variant on Mamba-2.

Verifies the dispatch split between B-correlation and C-correlation
Gram sources works end-to-end, and both branches are reachable.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import ExperimentConfig
from src.models.asr_model import ASRModel
from src.models.mamba2_block import Mamba2Block


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mk_block(key_source: str = "B") -> Mamba2Block:
    torch.manual_seed(0)
    blk = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64,
        expand=2, ngroups=1, chunk_size=32, mode="recurrent",
        use_lucid=True,
        lucid_key_source=key_source,
    ).to(DEVICE).eval()
    return blk


def test_c_side_rejects_invalid_key_source():
    import pytest
    with pytest.raises(ValueError, match="lucid_key_source"):
        Mamba2Block(
            d_model=64, d_state=64, d_conv=4, headdim=32,
            expand=2, ngroups=1, chunk_size=16, mode="recurrent",
            use_lucid=True, lucid_key_source="A",
        )


def test_c_side_forward_finite():
    blk = _mk_block("C")
    x = torch.randn(2, 64, 256, device=DEVICE)
    with torch.no_grad():
        y, _ = blk(x)
    assert y.shape == (2, 64, 256)
    assert torch.isfinite(y).all()


def test_b_and_c_paths_diverge_at_default_init():
    """With default init (tau=1.0), B- and C-side LUCID should produce
    materially different outputs.  Confirms the branch split is live."""
    torch.manual_seed(0)
    blk_b = _mk_block("B")
    torch.manual_seed(0)
    blk_c = _mk_block("C")

    # Copy all shared parameters from B to C so the only difference is
    # the dispatch branch.
    blk_c.load_state_dict(blk_b.state_dict())

    x = torch.randn(2, 64, 256, device=DEVICE)
    with torch.no_grad():
        y_b, _ = blk_b(x)
        y_c, _ = blk_c(x)
    diff = (y_b - y_c).abs().max().item()
    # Tighter than the usual 1e-5 tolerance for "identical".  Expect
    # meaningful divergence (both solves are different systems).
    assert diff > 1e-3, (
        f"B and C paths produce near-identical output ({diff:.2e}); "
        f"dispatch split may not be live."
    )


def test_c_side_end_to_end_dispatch():
    """build_encoder('mamba2_lucid_c') returns the correct encoder with
    lucid_key_source='C' on each layer."""
    cfg = ExperimentConfig()
    cfg.backbone = "mamba2_lucid_c"
    cfg.dropout = 0.0
    m = ASRModel(vocab_size=29, cfg=cfg).to(DEVICE)
    for i, layer in enumerate(m.encoder.layers):
        assert layer.mamba.use_lucid is True
        assert layer.mamba.lucid_key_source == "C"
    # forward + backward
    m.train()
    x = torch.randn(2, 80, 160, device=DEVICE)
    lens = torch.tensor([160, 128], device=DEVICE)
    lp, _, _ = m(x, lens)
    lp.sum().backward()
    for i, layer in enumerate(m.encoder.layers):
        g = layer.mamba.lucid_temperature.grad
        assert g is not None and g.abs().max().item() > 1e-8
