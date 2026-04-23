"""Gradient-flow tests for Stage-11 _v2 dispatch on Mamba-2 and LA.

Priority-2 engineering step per STAGE11_AGENT_QUEUE.md: the
``MultiDilationDWConv1d`` in ``mamba2_block.py`` (Mamba-2's channel-
first analogue) had the same multiplicative-zero init trap as the
pre-fix ``MultiDilationDWConvShift``.  These tests mirror
``test_mechanisms.py::test_multidil_init_gradient_flow_symmetric`` on
the Mamba-2 and LA ``_v2`` dispatch paths to guard against the trap
returning.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import ExperimentConfig
from src.models.asr_model import ASRModel
from src.models.mamba2_block import MultiDilationDWConv1d


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Module-level gradient-flow tests
# ---------------------------------------------------------------------------


def test_mamba2_multidil_v2_init_gradient_flow():
    """Mamba-2's MultiDilationDWConv1d with fixed_init=True (default) must
    have nonzero gradient on every d>1 branch's alpha and weight."""
    torch.manual_seed(0)
    mod = MultiDilationDWConv1d(
        channels=64, kernel_size=4, dilations=(1, 2, 4, 8), fixed_init=True,
    )
    x = torch.randn(2, 64, 32)
    y = mod(x).sum()
    y.backward()

    for i, d in enumerate(mod.dilations):
        if d == 1:
            continue
        assert mod.alpha.grad[i].abs().item() > 1e-8, (
            f"alpha[d={d}] gradient is zero after fix: {mod.alpha.grad}"
        )
        assert mod.branches[i].weight.grad.norm().item() > 1e-8, (
            f"branch[d={d}] weight gradient is zero after fix"
        )


def test_mamba2_multidil_broken_init_reproduces_trap():
    """Verifies the pre-fix broken init still produces the trap when
    fixed_init=False is explicitly opted in (for bit-exact repro of
    11.1a's pre-fix result, not for new runs)."""
    torch.manual_seed(0)
    mod = MultiDilationDWConv1d(
        channels=64, kernel_size=4, dilations=(1, 2, 4, 8), fixed_init=False,
    )
    x = torch.randn(2, 64, 32)
    mod(x).sum().backward()

    # d > 1: both α and branch weight should have zero gradient under the
    # pre-fix init (α_d=0 AND weight=0).
    for i, d in enumerate(mod.dilations):
        if d == 1:
            continue
        assert mod.alpha.grad[i].abs().item() == 0.0, (
            f"broken-init alpha[d={d}] grad should be exactly 0 "
            f"(trap not reproduced): {mod.alpha.grad}"
        )
        assert mod.branches[i].weight.grad.abs().sum().item() == 0.0, (
            f"broken-init branch[d={d}] weight grad should be exactly 0"
        )


# ---------------------------------------------------------------------------
# End-to-end dispatch tests
# ---------------------------------------------------------------------------


def _mk_model(backbone: str) -> ASRModel:
    cfg = ExperimentConfig()
    cfg.backbone = backbone
    cfg.dropout = 0.0
    return ASRModel(vocab_size=29, cfg=cfg).to(DEVICE)


def test_mamba2_v2_dispatch_end_to_end_gradient_flow():
    """Building an end-to-end ASR model with backbone
    ``mamba2_convshift_multidil_symmetric_v2`` and running a forward +
    backward: every layer's d>1 alpha must receive gradient."""
    m = _mk_model("mamba2_convshift_multidil_symmetric_v2")
    m.train()
    x = torch.randn(2, 80, 160, device=DEVICE)
    lens = torch.tensor([160, 128], device=DEVICE)
    lp, out_lens, _ = m(x, lens)
    lp.sum().backward()

    # Walk the model: every Mamba2Block's conv1d should be a
    # MultiDilationDWConv1d with nonzero alpha grads.
    for layer_idx, layer in enumerate(m.encoder.layers):
        conv = layer.mamba.conv1d
        assert isinstance(conv, MultiDilationDWConv1d), (
            f"layer {layer_idx}: mamba.conv1d is {type(conv).__name__}, "
            f"expected MultiDilationDWConv1d"
        )
        assert conv.alpha.grad is not None
        for i, d in enumerate(conv.dilations):
            if d == 1:
                continue
            assert conv.alpha.grad[i].abs().item() > 1e-8, (
                f"layer {layer_idx}: alpha[d={d}] grad is zero "
                f"(gradient trap not broken)"
            )


def test_linear_attn_v2_dispatch_end_to_end_gradient_flow():
    """Same end-to-end check for LA's
    ``linear_attn_convshift_multidil_symmetric_v2`` path.  LA uses
    ``MultiDilationDWConvShift`` from ``conv_shift.py`` (fixed by the
    other instance), so this confirms the fixed path is reached."""
    from src.models.mechanisms.conv_shift import MultiDilationDWConvShift

    m = _mk_model("linear_attn_convshift_multidil_symmetric_v2")
    m.train()
    x = torch.randn(2, 80, 160, device=DEVICE)
    lens = torch.tensor([160, 128], device=DEVICE)
    lp, out_lens, _ = m(x, lens)
    lp.sum().backward()

    for layer_idx, layer in enumerate(m.encoder.layers):
        premix = layer.premix
        assert isinstance(premix, MultiDilationDWConvShift), (
            f"layer {layer_idx}: premix is {type(premix).__name__}"
        )
        assert premix.alpha.grad is not None
        for i, d in enumerate(premix.dilations):
            if d == 1:
                continue
            assert premix.alpha.grad[i].abs().item() > 1e-8, (
                f"layer {layer_idx}: alpha[d={d}] grad is zero"
            )
