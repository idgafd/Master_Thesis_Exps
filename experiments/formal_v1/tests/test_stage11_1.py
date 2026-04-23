"""Zero-regression + param-parity tests for Stage 11.1 variants.

11.1a: ``mamba2_convshift_multidil_symmetric``
  At init with alpha_1=1 and alpha_{2,4,8}=0, the layer reduces to a
  single-dilation k=4 SYMMETRIC DWConv on xBC (not bit-exact vs vanilla
  mamba2 which uses causal padding — the "sym" variant is a deliberate
  structural choice, matching 10.3-sym on RWKV-6).  We verify that the
  multi-dilation branches d>1 are inert (zero-weight + zero-bias) and
  only branch_1 contributes at init.

11.1b: ``linear_attn_convshift_multidil_symmetric``
  At init with branch_1 = center-tap identity and alpha_{2,4,8}=0, the
  pre-mix is exactly identity — bit-exact vs vanilla linear_attn_causal.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import ExperimentConfig
from src.models.encoder import build_encoder
from src.models.linear_attn_causal import CausalLinearAttentionEncoder
from src.models.mamba2_block import MultiDilationDWConv1d, Mamba2Block

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maxerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


# ---------------------------------------------------------------------------
# 11.1b — LA + multidil_sym zero-regression (bit-exact vs vanilla LA)
# ---------------------------------------------------------------------------


def test_la_multidil_approx_zero_regression_vs_vanilla_at_init():
    """Approximate (not bit-exact) zero-regression vs vanilla LA at init.

    Relaxed from bit-exact after the MULTIDIL_INIT_FIX (commit ``3af846d``):
    α_{d>1}=0.01 + branch_{d>1}.weight ~ N(0, 0.01²) introduces a small
    deterministic perturbation of ~3e-4 elementwise at init.  With a
    ~6-layer stack + causal LA recurrence the perturbation accumulates to
    a small but observable output diff; tolerance set above expected fp
    rounding and below any activation-scale signal.
    """
    cfg_v = ExperimentConfig()
    cfg_v.backbone = "linear_attn_causal"
    cfg_v.d_model = 256
    cfg_v.n_heads = 4
    cfg_v.n_layers = 6
    cfg_v.dropout = 0.0

    cfg_m = ExperimentConfig()
    cfg_m.backbone = "linear_attn_convshift_multidil_symmetric"
    cfg_m.d_model = 256
    cfg_m.n_heads = 4
    cfg_m.n_layers = 6
    cfg_m.dropout = 0.0

    torch.manual_seed(0)
    enc_v = build_encoder(cfg_v).to(DEVICE).eval()
    torch.manual_seed(0)
    enc_m = build_encoder(cfg_m).to(DEVICE).eval()

    v_state = enc_v.state_dict()
    m_state = enc_m.state_dict()
    for k in v_state:
        if k in m_state:
            m_state[k].copy_(v_state[k])
    enc_m.load_state_dict(m_state, strict=False)

    x = torch.randn(2, 50, 256, device=DEVICE)
    lengths = torch.tensor([50, 40], device=DEVICE)
    with torch.no_grad():
        y_v, _ = enc_v(x, lengths)
        y_m, _ = enc_m(x, lengths)
    err = _maxerr(y_v, y_m)
    # Post-fix tolerance: well above fp rounding (1e-5), well below
    # activation scale (1.0) — matches the conv_shift.py fix's 5e-3 band
    # at the module level, scaled by the LA recurrence depth.
    assert err < 0.2, f"LA multidil init perturbation too large: {err:.2e}"


def test_la_multidil_param_parity_within_5pct():
    cfg_v = ExperimentConfig()
    cfg_v.backbone = "linear_attn_causal"
    cfg_v.d_model = 256
    cfg_v.n_heads = 4
    cfg_v.n_layers = 6
    cfg_m = ExperimentConfig()
    cfg_m.backbone = "linear_attn_convshift_multidil_symmetric"
    cfg_m.d_model = 256
    cfg_m.n_heads = 4
    cfg_m.n_layers = 6
    enc_v = build_encoder(cfg_v)
    enc_m = build_encoder(cfg_m)
    pv = sum(p.numel() for p in enc_v.parameters())
    pm = sum(p.numel() for p in enc_m.parameters())
    delta_pct = (pm - pv) / pv * 100
    assert abs(delta_pct) < 5.0, f"LA multidil param delta {delta_pct:.2f}% exceeds 5%"


# ---------------------------------------------------------------------------
# 11.1a — mamba2 + multidil_sym structural-reduction contract
# ---------------------------------------------------------------------------


def test_mamba2_multidil_module_alpha_init():
    """Post-fix: alpha_1=1.0 and alpha_{2,4,8}=0.01 at init (Option B).

    Pre-fix semantics (alpha_{d>1}=0 exactly) are available via
    ``fixed_init=False`` for bit-exact repro of the 11.1a broken-init
    run; the default path uses fixed_init=True."""
    conv = MultiDilationDWConv1d(channels=32, kernel_size=4,
                                 dilations=(1, 2, 4, 8))
    alpha = conv.alpha.detach()
    assert alpha[0].item() == 1.0
    for i in (1, 2, 3):
        assert abs(alpha[i].item() - 0.01) < 1e-6, (
            f"post-fix alpha[{conv.dilations[i]}] should be ≈0.01, "
            f"got {alpha[i].item()}"
        )

    # Broken-init path still available for bit-exact pre-v2 reproduction.
    conv_broken = MultiDilationDWConv1d(
        channels=32, kernel_size=4, dilations=(1, 2, 4, 8), fixed_init=False,
    )
    for i in (1, 2, 3):
        assert conv_broken.alpha[i].item() == 0.0


def test_mamba2_multidil_branches_d_gt_1_are_small_nonzero_at_init():
    """Post-fix: branches for d > 1 have small N(0, 0.01²) weights and
    zeroed biases, breaking the multiplicative-zero trap. Pre-fix
    all-zero semantics still reachable via fixed_init=False."""
    conv = MultiDilationDWConv1d(channels=32, kernel_size=4,
                                 dilations=(1, 2, 4, 8), bias=True)
    for idx, d in enumerate(conv.dilations):
        if d == 1:
            continue
        w_max = conv.branches[idx].weight.abs().max().item()
        assert 0.0 < w_max < 0.1, (
            f"branch[d={d}] weight abs-max={w_max}; expected small non-zero "
            f"(N(0, 0.01²) ≈ ±0.03 max for 32 chan × 4 taps)"
        )
        assert conv.branches[idx].bias.abs().max().item() == 0.0

    conv_broken = MultiDilationDWConv1d(
        channels=32, kernel_size=4, dilations=(1, 2, 4, 8), bias=True,
        fixed_init=False,
    )
    for idx, d in enumerate(conv_broken.dilations):
        if d == 1:
            continue
        assert conv_broken.branches[idx].weight.abs().max().item() == 0.0


def test_mamba2_multidil_output_length_matches_input_length():
    """Per-branch symmetric padding ⇒ output length == input length."""
    conv = MultiDilationDWConv1d(channels=16, kernel_size=4,
                                 dilations=(1, 2, 4, 8))
    x = torch.randn(2, 16, 100)
    y = conv(x)
    assert y.shape == (2, 16, 100), f"unexpected output shape {y.shape}"


def test_mamba2_multidil_approx_reduces_to_single_dilation_symmetric_at_init():
    """Post-fix: output ≈ single-dilation at init within ~3e-4 per
    MULTIDIL_INIT_FIX Option-B (small non-zero α_{d>1} + small random
    branch_{d>1} weights instead of zero-zero)."""
    torch.manual_seed(0)
    conv = MultiDilationDWConv1d(channels=16, kernel_size=4,
                                 dilations=(1, 2, 4, 8))
    x = torch.randn(2, 16, 50)
    with torch.no_grad():
        y_full = conv(x)
        b1 = conv.branches[0]
        x_p = torch.nn.functional.pad(x, (1, 2))
        y_expected = torch.nn.functional.conv1d(
            x_p, b1.weight, bias=b1.bias, stride=1, padding=0,
            dilation=1, groups=16,
        ) * conv.alpha[0]
    err = _maxerr(y_full, y_expected)
    # ~3e-4 from α_{d>1}=0.01 * branch_{d>1}.weight ~ N(0, 0.01²) summed
    # over 3 d>1 branches.  5e-3 tolerance matches conv_shift fix.
    assert err < 5e-3, f"multidil approx-reduction drift too large: {err:.2e}"


def test_mamba2_multidil_block_forward_shapes():
    """End-to-end Mamba2Block with use_multidil_sym=True gives correct shapes."""
    blk = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64,
        expand=2, ngroups=1, chunk_size=64, mode="recurrent",
        use_multidil_sym=True,
    ).to(DEVICE).eval()
    x = torch.randn(2, 128, 256, device=DEVICE)
    with torch.no_grad():
        y, new_state = blk(x)
    assert y.shape == (2, 128, 256)
    assert "conv" in new_state and "ssm" in new_state


def test_mamba2_multidil_encoder_param_parity_within_5pct():
    cfg_v = ExperimentConfig()
    cfg_v.backbone = "mamba2"
    cfg_v.d_model = 256
    cfg_v.n_layers = 6
    cfg_m = ExperimentConfig()
    cfg_m.backbone = "mamba2_convshift_multidil_symmetric"
    cfg_m.d_model = 256
    cfg_m.n_layers = 6
    enc_v = build_encoder(cfg_v)
    enc_m = build_encoder(cfg_m)
    pv = sum(p.numel() for p in enc_v.parameters())
    pm = sum(p.numel() for p in enc_m.parameters())
    delta_pct = (pm - pv) / pv * 100
    # Expect small positive delta: 3 extra DWConv branches (zero-init weight
    # + zero-init bias) + a 4-vector alpha per layer.
    assert abs(delta_pct) < 5.0, f"Mamba-2 multidil param delta {delta_pct:.2f}% exceeds 5%"


def test_mamba2_multidil_state_carry_preserves_shape():
    """Stateful forward with multidil produces the same L-length output."""
    blk = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64,
        expand=2, ngroups=1, chunk_size=64, mode="recurrent",
        use_multidil_sym=True,
    ).to(DEVICE).eval()
    x1 = torch.randn(2, 64, 256, device=DEVICE)
    x2 = torch.randn(2, 32, 256, device=DEVICE)
    state0 = blk.init_state(batch_size=2, device=DEVICE)
    with torch.no_grad():
        y1, st1 = blk(x1, state=state0)
        y2, st2 = blk(x2, state=st1)
    assert y1.shape == (2, 64, 256)
    assert y2.shape == (2, 32, 256)
    assert st2["conv"].shape == state0["conv"].shape


# ---------------------------------------------------------------------------
# Smoke test for the encoder factory
# ---------------------------------------------------------------------------


def test_factory_builds_11_1a_and_11_1b():
    for name in (
        "mamba2_convshift_multidil_symmetric",
        "linear_attn_convshift_multidil_symmetric",
    ):
        cfg = ExperimentConfig()
        cfg.backbone = name
        cfg.d_model = 256
        cfg.n_heads = 4
        cfg.n_layers = 6
        cfg.dropout = 0.0
        enc = build_encoder(cfg).to(DEVICE).eval()
        x = torch.randn(2, 40, 256, device=DEVICE)
        lengths = torch.tensor([40, 30], device=DEVICE)
        with torch.no_grad():
            y, _ = enc(x, lengths)
        assert y.shape == (2, 40, 256)
