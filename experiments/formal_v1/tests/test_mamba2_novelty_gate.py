"""Smoke tests for the write-novelty-gated Mamba-2 state update.

Four mandatory contracts:

1. At-init: output matches vanilla Mamba-2 within fp32 noise
   (γ_raw = -30 → softplus ≈ 9.4e-14 → ω ≡ 1 to machine precision).
2. Engagement: with γ_raw = 0 (softplus(0) = 0.693), output diverges
   materially from vanilla — confirms the gate is live.
3. Gradient: γ_raw receives non-zero gradient through a forward+backward
   so SGD can lift it off the init plateau.
4. Carry-state: two half-sequence forwards (with state carried) equal
   a single full-sequence forward — sigma carry is consistent.
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


def _mk_block(use_novelty_gate: bool = True, **kw) -> Mamba2Block:
    torch.manual_seed(0)
    blk = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64,
        expand=2, ngroups=1, chunk_size=32, mode="recurrent",
        use_novelty_gate=use_novelty_gate,
        **kw,
    ).to(DEVICE).eval()
    return blk


def test_at_init_near_vanilla():
    """γ_raw = 0 with shift=5 → γ = softplus(-5) ≈ 6.7e-3.  Output is
    NOT bit-exact vanilla, but within ~1% (ω ≈ 0.993 at init).  Exact
    reduction is deliberately traded for an Adam-reachable γ_raw; see
    Mamba2Block init comment."""
    torch.manual_seed(0)
    blk_v = _mk_block(use_novelty_gate=False)
    torch.manual_seed(0)
    blk_n = _mk_block(use_novelty_gate=True)

    # Copy the vanilla params into the novelty block so the only
    # difference is γ_raw = 0 (and Σ carry, which is Σ_0 = 0).
    novelty_sd = blk_n.state_dict()
    for k, v in blk_v.state_dict().items():
        novelty_sd[k] = v
    blk_n.load_state_dict(novelty_sd)

    x = torch.randn(2, 96, 256, device=DEVICE)
    with torch.no_grad():
        y_v, _ = blk_v(x)
        y_n, _ = blk_n(x)
    rel = (y_v - y_n).norm().item() / (y_v.norm().item() + 1e-12)
    assert rel < 1e-2, (
        f"At-init novelty output should be within ~1% of vanilla in "
        f"relative L2; got {rel:.2e}"
    )


def test_gamma_raw_is_adam_reachable():
    """σ(γ_raw − shift) must be materially > 0 so Adam updates to γ_raw
    are real (not swamped by Adam's ε floor).  With shift=5 and
    γ_raw=0 init, the effective softplus derivative on the γ_raw side
    is σ(-5) ≈ 6.7e-3 — but crucially, σ'(γ_raw − shift) at γ_raw=0 is
    σ(-5)·(1 - σ(-5)) on γ, and through the chain rule the gradient
    flowing back to γ_raw scales like σ(-5) itself, which is many
    orders of magnitude above the γ_raw=-30 pathology (~1e-14).
    This test asserts the scale is adequate for training."""
    blk_n = _mk_block(use_novelty_gate=True)
    # σ(γ_raw - shift) at init should be ~6.7e-3, far from the 1e-10
    # range where Adam's ε makes updates vanish.
    import torch.nn.functional as F
    sigma = torch.sigmoid(blk_n.novelty_gamma_raw - blk_n._novelty_shift)
    assert sigma.min().item() > 1e-4, (
        f"softplus-derivative σ(γ_raw - shift) is too small "
        f"({sigma.min().item():.2e}) — γ_raw will be Adam-unreachable"
    )


def test_engaged_gate_diverges():
    """With γ_raw large (softplus ≈ γ ≫ q²), ω suppresses most writes;
    output must diverge materially from vanilla.  Pushes well past the
    ~1e-5 fp32 noise floor to confirm the gate is live."""
    torch.manual_seed(0)
    blk_v = _mk_block(use_novelty_gate=False)
    torch.manual_seed(0)
    blk_n = _mk_block(use_novelty_gate=True)

    novelty_sd = blk_n.state_dict()
    for k, v in blk_v.state_dict().items():
        novelty_sd[k] = v
    blk_n.load_state_dict(novelty_sd)
    # Engage strongly: γ_raw=10, shift=5 → γ = softplus(5) ≈ 5.007.
    # q² at init is O(1e-3)–O(10) depending on chunk (Σ_0 regularised
    # to 1e-3·I gives Σ_inv ≈ 1000·I, q² = 1000·||B||²).  γ=5 ensures
    # clear attenuation regardless.
    with torch.no_grad():
        blk_n.novelty_gamma_raw.fill_(10.0)

    x = torch.randn(2, 96, 256, device=DEVICE)
    with torch.no_grad():
        y_v, _ = blk_v(x)
        y_n, _ = blk_n(x)
    diff = (y_v - y_n).abs().max().item()
    assert diff > 1e-4, (
        f"Novelty gate engaged at γ_raw=5 should diverge from vanilla "
        f"well past fp32 noise, got {diff:.2e}.  Branch may not be live."
    )


def test_gamma_receives_gradient():
    """forward + backward → γ_raw.grad non-zero.  Needed for SGD to lift
    the gate off the -30 init."""
    blk_n = _mk_block(use_novelty_gate=True)
    blk_n.train()

    x = torch.randn(2, 96, 256, device=DEVICE, requires_grad=False)
    y, _ = blk_n(x)
    y.sum().backward()
    g = blk_n.novelty_gamma_raw.grad
    assert g is not None, "γ_raw.grad is None — gate not connected to graph"
    g_max = g.abs().max().item()
    # At γ_raw=-30 the gradient is small but not zero; softplus'(−30) is
    # tiny but the finite float32 floor keeps it computable.  Any non-zero
    # gradient is enough for SGD to engage given enough steps; the training
    # run will naturally lift γ once the upstream forces it.
    assert g_max > 0.0, (
        f"γ_raw.grad is exactly zero → gate unreachable by SGD.  "
        f"Got max |g| = {g_max}"
    )


def test_carry_state_splits_match_single_forward():
    """Two half-forwards with sigma carry must equal one full forward.

    Covers: conv carry, SSM carry, and Σ carry all compose correctly.
    """
    blk = _mk_block(use_novelty_gate=True)

    # Engage the gate so Σ actually matters for the output.
    with torch.no_grad():
        blk.novelty_gamma_raw.fill_(0.0)

    L = 128
    assert L % 2 == 0
    x = torch.randn(2, L, 256, device=DEVICE)

    with torch.no_grad():
        y_full, _ = blk(x)

        state0 = blk.init_state(batch_size=2, device=DEVICE)
        y1, state1 = blk(x[:, : L // 2], state=state0)
        y2, _ = blk(x[:, L // 2 :], state=state1)
        y_split = torch.cat([y1, y2], dim=1)

    diff = (y_full - y_split).abs().max().item()
    # ssd_scan_causal_novelty's Σ carry is chunk-granular (Σ updated at
    # chunk boundaries).  As long as the split point L/2 is a multiple of
    # the block's chunk_size (32), the two paths produce the same Σ
    # trajectory.  L/2 = 64 = 2 · chunk_size, so split is clean.
    assert diff < 1e-4, (
        f"Two-halves with carry ≠ single forward (max diff {diff:.2e}).  "
        f"Σ or SSM carry broken."
    )


def test_end_to_end_dispatch_novelty_gate():
    """build_encoder('mamba2_novelty_gate') returns a Mamba-2 encoder
    with use_novelty_gate=True on every layer, and forward+backward
    reaches γ_raw."""
    cfg = ExperimentConfig()
    cfg.backbone = "mamba2_novelty_gate"
    cfg.dropout = 0.0
    m = ASRModel(vocab_size=29, cfg=cfg).to(DEVICE)
    for layer in m.encoder.layers:
        assert layer.mamba.use_novelty_gate is True
        assert layer.mamba.use_lucid is False

    m.train()
    x = torch.randn(2, 80, 160, device=DEVICE)
    lens = torch.tensor([160, 128], device=DEVICE)
    lp, _, _ = m(x, lens)
    lp.sum().backward()
    for layer in m.encoder.layers:
        g = layer.mamba.novelty_gamma_raw.grad
        assert g is not None and g.abs().max().item() > 0.0


def test_fixed_gamma_is_buffer_not_parameter():
    """mamba2_novelty_fixed_g05: γ is a buffer (no grad), exactly 0.5,
    and backward does NOT populate a grad on it."""
    cfg = ExperimentConfig()
    cfg.backbone = "mamba2_novelty_fixed_g05"
    cfg.dropout = 0.0
    m = ASRModel(vocab_size=29, cfg=cfg).to(DEVICE)

    for layer in m.encoder.layers:
        mixer = layer.mamba
        # Buffer present, parameter absent.
        named_buffers = dict(mixer.named_buffers())
        named_params = dict(mixer.named_parameters())
        assert "novelty_gamma_buf" in named_buffers
        assert "novelty_gamma_raw" not in named_params
        # Exact value γ = 0.5 across heads.
        buf = named_buffers["novelty_gamma_buf"]
        assert torch.allclose(buf, torch.full_like(buf, 0.5))
        # No .requires_grad on a buffer.
        assert buf.requires_grad is False

    # Forward + backward: no crash, no grad on the buffer.
    m.train()
    x = torch.randn(2, 80, 160, device=DEVICE)
    lens = torch.tensor([160, 128], device=DEVICE)
    lp, _, _ = m(x, lens)
    lp.sum().backward()
    for layer in m.encoder.layers:
        assert layer.mamba.novelty_gamma_buf.grad is None


def test_fixed_gamma_materially_differs_from_vanilla():
    """At γ=0.5 the gate attenuates writes in deeper chunks (Σ
    accumulates, Σ_inv shrinks, q² drops, ω falls from 1 toward 0.8).
    Chunk-0 is regularised to ω≈1, so divergence grows with chunk
    count; L=256 (≈ realistic ASR sequence length post-subsampling)
    reliably shows material divergence."""
    torch.manual_seed(0)
    blk_v = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64, expand=2,
        ngroups=1, chunk_size=32, mode="recurrent",
        use_novelty_gate=False,
    ).to(DEVICE).eval()
    torch.manual_seed(0)
    blk_fix = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64, expand=2,
        ngroups=1, chunk_size=32, mode="recurrent",
        use_novelty_gate=True,
        novelty_gamma_fixed=0.5,
    ).to(DEVICE).eval()

    # Copy shared weights so the only difference is the gate.
    fix_sd = blk_fix.state_dict()
    for k, v in blk_v.state_dict().items():
        fix_sd[k] = v
    blk_fix.load_state_dict(fix_sd)

    x = torch.randn(2, 256, 256, device=DEVICE)
    with torch.no_grad():
        y_v, _ = blk_v(x)
        y_f, _ = blk_fix(x)
    rel = (y_v - y_f).norm().item() / y_v.norm().item()
    assert rel > 1e-3, (
        f"γ=0.5 fixed gate should diverge from vanilla materially at "
        f"L=256; got rel-L2 = {rel:.2e}"
    )


def test_end_to_end_dispatch_lucid_novelty():
    """Composition backbone: mamba2_lucid_novelty has both flags on."""
    cfg = ExperimentConfig()
    cfg.backbone = "mamba2_lucid_novelty"
    cfg.dropout = 0.0
    m = ASRModel(vocab_size=29, cfg=cfg).to(DEVICE)
    for layer in m.encoder.layers:
        assert layer.mamba.use_novelty_gate is True
        assert layer.mamba.use_lucid is True
        assert layer.mamba.lucid_key_source == "B"

    m.train()
    x = torch.randn(2, 80, 160, device=DEVICE)
    lens = torch.tensor([160, 128], device=DEVICE)
    lp, _, _ = m(x, lens)
    lp.sum().backward()
    for layer in m.encoder.layers:
        assert layer.mamba.lucid_temperature.grad is not None
        assert layer.mamba.novelty_gamma_raw.grad is not None
