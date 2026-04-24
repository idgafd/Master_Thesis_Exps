"""Tests for D-LUCID v6 — additive decay-penalty LUCID on Mamba-2 SSD.

v6 adds a per-pair decay-distance penalty to the LUCID exponent:

    scaled_ij = τ · (G_ij/√N − √N)  −  γ · |cs[i] − cs[j]|

Per-head γ ≥ 0 (softplus(γ_raw − 5), γ_raw init 0 ⇒ γ ≈ 0.007 at init).

Invariants guarded here:
  * γ = 0 exactly ⇒ bit-exact vanilla LUCID (any decay profile).
  * Δcs = 0 (zero-decay within chunk) ⇒ bit-exact LUCID for any γ.
  * Unit diagonal preserved for any τ, γ, decay profile.
  * Conditioning stays O(1) across γ ∈ [0, 1].
  * γ > 0 with material decay ⇒ output diverges meaningfully from LUCID.
  * End-to-end dispatch and gradient flow through the new γ parameter.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from src.config import ExperimentConfig
from src.models.asr_model import ASRModel
from src.models.encoder import build_encoder
from src.models.mamba2_block import Mamba2Block
from src.models.mamba2_kernels import (
    _apply_lucid_mamba2_chunked,
    _apply_lucid_mamba2_chunked_decay_aware,
    ssd_scan_causal,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Function-level invariants
# ---------------------------------------------------------------------------

def test_dlucid_gamma_zero_is_exact_lucid():
    """γ = 0 should make v6 bit-exactly equal to vanilla LUCID — for any
    decay profile.  This is the zero-regression contract that v6 preserves
    and the v1–v5 lineage did not."""
    torch.manual_seed(0)
    B_sz, nC, Tc, H, N, P = 2, 3, 16, 4, 8, 12
    X_c = torch.randn(B_sz, nC, Tc, H, P, dtype=torch.float64)
    key_c = torch.randn(B_sz, nC, Tc, H, N, dtype=torch.float64)
    tau = torch.tensor([0.5, 1.0, 1.5, 0.75], dtype=torch.float64)
    # Non-trivial decay — the test that v6's γ=0 is LUCID despite decay.
    ramp = torch.arange(Tc, dtype=torch.float64) * -0.2
    A_cumsum = ramp.view(1, 1, 1, Tc).expand(B_sz, H, nC, Tc).contiguous()
    gamma_zero = torch.zeros(H, dtype=torch.float64)

    y_lucid = _apply_lucid_mamba2_chunked(X_c, key_c, tau)
    y_v6 = _apply_lucid_mamba2_chunked_decay_aware(
        X_c, key_c, tau, A_cumsum, gamma_zero,
    )
    max_diff = (y_lucid - y_v6).abs().max().item()
    assert max_diff < 1e-12, (
        f"γ=0 must reduce to vanilla LUCID bit-exactly; max diff = {max_diff}"
    )


def test_dlucid_zero_decay_is_exact_lucid_for_any_gamma():
    """If within-chunk cs is flat (zero per-chunk decay), Δcs = 0 and the
    penalty term vanishes for any γ ≥ 0.  This matches the init conditions
    where dt · A ≈ 0."""
    torch.manual_seed(1)
    B_sz, nC, Tc, H, N, P = 1, 2, 16, 2, 8, 8
    X_c = torch.randn(B_sz, nC, Tc, H, P, dtype=torch.float64)
    key_c = torch.randn(B_sz, nC, Tc, H, N, dtype=torch.float64)
    tau = torch.tensor([1.0, 1.0], dtype=torch.float64)
    A_zero = torch.zeros(B_sz, H, nC, Tc, dtype=torch.float64)
    for gamma_val in (0.1, 1.0, 10.0):
        gamma = torch.full((H,), gamma_val, dtype=torch.float64)
        y_lucid = _apply_lucid_mamba2_chunked(X_c, key_c, tau)
        y_v6 = _apply_lucid_mamba2_chunked_decay_aware(
            X_c, key_c, tau, A_zero, gamma,
        )
        diff = (y_lucid - y_v6).abs().max().item()
        assert diff < 1e-12, (
            f"zero-decay at γ={gamma_val}: v6 must match LUCID; diff = {diff}"
        )


def test_dlucid_diverges_from_lucid_when_both_gamma_and_decay_present():
    """γ > 0 AND Δcs > 0 is the only regime where v6 differs from LUCID —
    guard against a sign error / broadcast bug that would zero the penalty."""
    torch.manual_seed(2)
    B_sz, nC, Tc, H, N, P = 1, 2, 16, 2, 8, 8
    X_c = torch.randn(B_sz, nC, Tc, H, P, dtype=torch.float64)
    key_c = torch.randn(B_sz, nC, Tc, H, N, dtype=torch.float64)
    tau = torch.tensor([1.0, 1.0], dtype=torch.float64)
    ramp = torch.arange(Tc, dtype=torch.float64) * -0.2
    A_cumsum = ramp.view(1, 1, 1, Tc).expand(B_sz, H, nC, Tc).contiguous()
    gamma = torch.full((H,), 0.3, dtype=torch.float64)

    y_lucid = _apply_lucid_mamba2_chunked(X_c, key_c, tau)
    y_v6 = _apply_lucid_mamba2_chunked_decay_aware(
        X_c, key_c, tau, A_cumsum, gamma,
    )
    assert not torch.allclose(y_lucid, y_v6, rtol=1e-4, atol=1e-4), (
        "v6 with γ > 0 and decay > 0 must differ meaningfully from LUCID"
    )


def test_dlucid_unit_diagonal_preserved():
    """P_ii = 1 for any τ, γ, decay — Δcs_ii = 0 and (G_ii/√N − √N) = 0."""
    torch.manual_seed(3)
    B_sz, nC, Tc, H, N, P = 1, 1, 12, 1, 64, 12
    X_eye = torch.eye(Tc, dtype=torch.float64).view(1, 1, Tc, 1, Tc)
    key_c = torch.randn(B_sz, nC, Tc, H, N, dtype=torch.float64)
    tau = torch.tensor([2.0], dtype=torch.float64)
    ramp = torch.arange(Tc, dtype=torch.float64) * -0.3
    A_cumsum = ramp.view(1, 1, 1, Tc)
    gamma = torch.tensor([0.5], dtype=torch.float64)

    y = _apply_lucid_mamba2_chunked_decay_aware(
        X_eye, key_c, tau, A_cumsum, gamma,
    )
    y_mat = y[0, 0, :, 0, :]
    P_mat = torch.linalg.inv(y_mat)
    diag = torch.diagonal(P_mat)
    # ε = 1e-4 ⇒ diag(P) = 1 + 1e-4 ± numerical slack.
    assert torch.allclose(
        diag, torch.full_like(diag, 1.0001), rtol=0, atol=5e-3,
    ), f"v6 diag should be 1 + ε; got: {diag.tolist()}"


def test_dlucid_conditioning_under_large_gamma():
    """cond(P) stays O(1) across γ ∈ [0, 1] — γ can only push off-diag
    *toward zero*, never toward 1, so no rank-1 collapse path exists."""
    torch.manual_seed(4)
    B_sz, nC, Tc, H, N = 1, 1, 64, 1, 64
    key_c = torch.randn(B_sz, nC, Tc, H, N, dtype=torch.float64)
    tau = torch.tensor([1.2], dtype=torch.float64)
    # Severe decay profile.
    ramp = torch.arange(Tc, dtype=torch.float64) * -0.4
    A_cumsum = ramp.view(1, 1, 1, Tc)
    X_eye = torch.eye(Tc, dtype=torch.float64).view(1, 1, Tc, 1, Tc)
    for gamma_val in (0.01, 0.1, 0.5, 1.0):
        gamma = torch.tensor([gamma_val], dtype=torch.float64)
        y = _apply_lucid_mamba2_chunked_decay_aware(
            X_eye, key_c, tau, A_cumsum, gamma,
        )
        P = torch.linalg.inv(y[0, 0, :, 0, :])
        cond = torch.linalg.cond(P).item()
        assert cond < 50.0, (
            f"γ={gamma_val}: cond(P) should stay O(1); got {cond:.1f}"
        )


def test_dlucid_signal_preservation():
    """Output row norms stay close to input row norms — additive penalty
    only reduces off-diag (never amplifies), so solve is near-identity."""
    torch.manual_seed(5)
    B_sz, nC, Tc, H, N, Pd = 1, 1, 64, 1, 64, 16
    X_c = torch.randn(B_sz, nC, Tc, H, Pd, dtype=torch.float64)
    key_c = torch.randn(B_sz, nC, Tc, H, N, dtype=torch.float64)
    tau = torch.tensor([1.2], dtype=torch.float64)
    ramp = torch.arange(Tc, dtype=torch.float64) * -0.4
    A_cumsum = ramp.view(1, 1, 1, Tc)
    gamma = torch.tensor([0.3], dtype=torch.float64)

    Y = _apply_lucid_mamba2_chunked_decay_aware(
        X_c, key_c, tau, A_cumsum, gamma,
    )
    x_norms = X_c.norm(dim=-1).flatten()
    y_norms = Y.norm(dim=-1).flatten()
    ratio = (y_norms / x_norms.clamp(min=1e-12))
    assert ratio.max().item() < 1.5 and ratio.min().item() > 0.5, (
        f"v6 should keep ||Y/X|| near 1; got range "
        f"[{ratio.min():.3f}, {ratio.max():.3f}]"
    )


# ---------------------------------------------------------------------------
# Block-level / end-to-end
# ---------------------------------------------------------------------------

def test_dlucid_block_has_gamma_parameter():
    """The block must register ``dlucid_gamma_raw`` alongside lucid_temperature."""
    blk = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64,
        expand=2, ngroups=1, chunk_size=32, mode="recurrent",
        use_lucid=True, lucid_decay_aware=True,
    )
    assert hasattr(blk, "dlucid_gamma_raw")
    assert blk.dlucid_gamma_raw.shape == (blk.nheads,)
    # At init, γ = softplus(-5) ≈ 0.0067 per head.
    gamma = torch.nn.functional.softplus(
        blk.dlucid_gamma_raw - blk._dlucid_gamma_shift
    )
    assert torch.allclose(
        gamma, torch.full_like(gamma, 0.006715),
        rtol=1e-2, atol=1e-4,
    ), f"γ_init should be ≈ 0.0067; got {gamma.tolist()}"


def test_dlucid_forward_finite_end_to_end():
    torch.manual_seed(0)
    blk = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64,
        expand=2, ngroups=1, chunk_size=32, mode="recurrent",
        use_lucid=True, lucid_decay_aware=True,
    ).to(DEVICE).eval()
    x = torch.randn(2, 96, 256, device=DEVICE)
    with torch.no_grad():
        y, _ = blk(x)
    assert y.shape == (2, 96, 256)
    assert torch.isfinite(y).all()


def test_dlucid_gradient_flow_through_tau_and_gamma():
    torch.manual_seed(0)
    blk = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64,
        expand=2, ngroups=1, chunk_size=32, mode="recurrent",
        use_lucid=True, lucid_decay_aware=True,
    ).to(DEVICE).train()
    x = torch.randn(2, 64, 256, device=DEVICE)
    y, _ = blk(x)
    y.sum().backward()
    # Both τ and γ must receive non-zero gradient.
    assert blk.lucid_temperature.grad is not None
    assert torch.isfinite(blk.lucid_temperature.grad).all()
    assert blk.lucid_temperature.grad.abs().max().item() > 1e-8
    assert blk.dlucid_gamma_raw.grad is not None
    assert torch.isfinite(blk.dlucid_gamma_raw.grad).all()
    assert blk.dlucid_gamma_raw.grad.abs().max().item() > 1e-9


def test_dlucid_requires_use_lucid():
    with pytest.raises(ValueError, match="lucid_decay_aware=True requires"):
        Mamba2Block(
            d_model=64, d_state=64, d_conv=4, headdim=32,
            expand=2, ngroups=1, chunk_size=16, mode="recurrent",
            use_lucid=False, lucid_decay_aware=True,
        )


def test_dlucid_rejects_lion_mode():
    with pytest.raises(ValueError, match="LUCID is only defined"):
        Mamba2Block(
            d_model=64, d_state=64, d_conv=4, headdim=32,
            expand=2, ngroups=1, chunk_size=16, mode="lion",
            use_lucid=True, lucid_decay_aware=True,
        )


def test_dlucid_end_to_end_dispatch():
    cfg = ExperimentConfig()
    cfg.backbone = "mamba2_dlucid"
    cfg.dropout = 0.0
    m = ASRModel(vocab_size=29, cfg=cfg).to(DEVICE)
    for i, layer in enumerate(m.encoder.layers):
        assert hasattr(layer.mamba, "dlucid_gamma_raw"), (
            f"layer {i} missing dlucid_gamma_raw"
        )
        assert layer.mamba.lucid_decay_aware is True
    x = torch.randn(2, 80, 160, device=DEVICE)
    lens = torch.tensor([160, 128], device=DEVICE)
    lp, _, _ = m(x, lens)
    lp.sum().backward()
    for i, layer in enumerate(m.encoder.layers):
        g_tau = layer.mamba.lucid_temperature.grad
        g_gamma = layer.mamba.dlucid_gamma_raw.grad
        assert g_tau is not None and g_tau.abs().max().item() > 1e-8, (
            f"layer {i} lucid_temperature grad is zero"
        )
        assert g_gamma is not None and g_gamma.abs().max().item() > 1e-9, (
            f"layer {i} dlucid_gamma_raw grad is zero"
        )


def test_dlucid_param_delta_is_one_scalar_per_head_per_layer():
    """v6 adds nheads per layer on top of LUCID (one γ per head)."""
    cfg_l = ExperimentConfig(); cfg_l.backbone = "mamba2_lucid"
    cfg_d = ExperimentConfig(); cfg_d.backbone = "mamba2_dlucid"
    enc_l = build_encoder(cfg_l)
    enc_d = build_encoder(cfg_d)
    pl = sum(p.numel() for p in enc_l.parameters())
    pd = sum(p.numel() for p in enc_d.parameters())
    expected_delta = cfg_d.n_layers * (
        enc_d.layers[0].mamba.nheads
    )
    assert pd - pl == expected_delta, (
        f"v6 should add n_layers·n_heads = {expected_delta} scalars; got {pd - pl}"
    )


def test_dlucid_composition_with_multidil_builds():
    cfg = ExperimentConfig()
    cfg.backbone = "mamba2_dlucid_convshift_multidil_symmetric_v2"
    cfg.dropout = 0.0
    m = ASRModel(vocab_size=29, cfg=cfg).to(DEVICE)
    m.train()
    x = torch.randn(2, 80, 160, device=DEVICE)
    lens = torch.tensor([160, 128], device=DEVICE)
    lp, _, _ = m(x, lens)
    lp.sum().backward()
    for i, layer in enumerate(m.encoder.layers):
        g = layer.mamba.dlucid_gamma_raw.grad
        assert g is not None and g.abs().max().item() > 1e-9


# ---------------------------------------------------------------------------
# Scan-level
# ---------------------------------------------------------------------------

def test_dlucid_scan_matches_lucid_at_gamma_zero():
    """Full scan path: lucid_decay_aware=True with γ=0 bit-exact matches
    lucid_decay_aware=False (vanilla LUCID scan)."""
    torch.manual_seed(6)
    Bsz, L, H, P, N = 2, 32, 2, 8, 8
    X = torch.randn(Bsz, L, H, P, dtype=torch.float64)
    dt = torch.rand(Bsz, L, H, dtype=torch.float64) * 0.1 + 0.01
    A = -torch.tensor([2.0, 4.0], dtype=torch.float64)  # material decay
    B = torch.randn(Bsz, L, H, N, dtype=torch.float64)
    C = torch.randn(Bsz, L, H, N, dtype=torch.float64)
    tau = torch.tensor([1.5, 1.5], dtype=torch.float64)
    gamma_zero = torch.zeros(H, dtype=torch.float64)

    Y_lucid, _ = ssd_scan_causal(
        X, dt, A, B, C, chunk_size=16,
        lucid_temp=tau, lucid_key_source="B", lucid_decay_aware=False,
    )
    Y_v6, _ = ssd_scan_causal(
        X, dt, A, B, C, chunk_size=16,
        lucid_temp=tau, lucid_key_source="B",
        lucid_decay_aware=True, lucid_decay_penalty=gamma_zero,
    )
    diff = (Y_lucid - Y_v6).abs().max().item()
    assert diff < 1e-5, (
        f"γ=0 scan must match vanilla LUCID scan (fp32 tol); got {diff}"
    )


def test_dlucid_scan_diverges_under_gamma_and_decay():
    torch.manual_seed(7)
    Bsz, L, H, P, N = 2, 32, 2, 8, 8
    X = torch.randn(Bsz, L, H, P, dtype=torch.float64)
    dt = torch.rand(Bsz, L, H, dtype=torch.float64) * 0.1 + 0.01
    A = -torch.tensor([3.0, 5.0], dtype=torch.float64)
    B = torch.randn(Bsz, L, H, N, dtype=torch.float64)
    C = torch.randn(Bsz, L, H, N, dtype=torch.float64)
    tau = torch.tensor([1.5, 1.5], dtype=torch.float64)
    gamma = torch.tensor([0.5, 0.5], dtype=torch.float64)

    Y_lucid, _ = ssd_scan_causal(
        X, dt, A, B, C, chunk_size=16,
        lucid_temp=tau, lucid_key_source="B", lucid_decay_aware=False,
    )
    Y_v6, _ = ssd_scan_causal(
        X, dt, A, B, C, chunk_size=16,
        lucid_temp=tau, lucid_key_source="B",
        lucid_decay_aware=True, lucid_decay_penalty=gamma,
    )
    diff = (Y_lucid - Y_v6).abs().max().item()
    assert diff > 1e-3, (
        f"v6 must differ materially from LUCID at γ=0.5 with decay; got {diff}"
    )
