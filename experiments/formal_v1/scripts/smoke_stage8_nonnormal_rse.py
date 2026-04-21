#!/usr/bin/env python3
"""Stage-8 T2 — `rwkv6_nonnormal_rse_viscosity` zero-regression-at-init smoke test.

At init, the non-normal mechanism adds:
  - nonnormal_rho_base = 0  → ρ_raw = 0 + 0 = 0 at init
  - nonnormal_rho_w1 = 0    → LoRA contribution = 0
  - nonnormal_psi_base = U(0, 2π)  (irrelevant when ρ = 0)
  - nonnormal_psi_w1 = 0    → ψ LoRA contribution = 0
  - nonnormal_mu = 0        → μ ρ² = 0 in viscosity
  ⇒ ρ ≡ 0 ⇒ P = I ⇒ G = e^{-λ} R(θ) = exact RSE transition.

However this uses a SEQUENTIAL scan while rwkv6_rse_strong_viscosity uses
the CHUNKED complex scan.  The two differ in FP accumulation order; they
should be bit-exact algorithmically, numerically close within ~1e-4.

Run:
    uv run python scripts/smoke_stage8_nonnormal_rse.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.config import ExperimentConfig
from src.models.asr_model import ASRModel


VOCAB_SIZE = 29


def build(backbone: str, seed: int = 42) -> ASRModel:
    torch.manual_seed(seed)
    cfg = ExperimentConfig()
    cfg.backbone = backbone
    cfg.rwkv_mode = "recurrent"
    return ASRModel(vocab_size=VOCAB_SIZE, cfg=cfg)


def main() -> int:
    print("Building anchor (`rwkv6_rse_strong_viscosity`) and "
          "T2 (`rwkv6_nonnormal_rse_viscosity`) with matched seed…")
    anchor = build("rwkv6_rse_strong_viscosity", seed=42)
    nn_rse = build("rwkv6_nonnormal_rse_viscosity", seed=42)

    anchor_sd = anchor.state_dict()
    nn_sd = nn_rse.state_dict()
    copied = 0
    new_zero, new_nonzero = 0, 0
    for k, v in anchor_sd.items():
        if k in nn_sd and nn_sd[k].shape == v.shape:
            nn_sd[k] = v.clone()
            copied += 1

    # Check new params at init
    new_keys = [k for k in nn_sd.keys() if k not in anchor_sd]
    for k in new_keys:
        t = nn_sd[k]
        mx = t.abs().max().item() if t.numel() > 0 else 0.0
        # psi_base is uniformly initialized on [0, 2π) so it is NON-zero.
        # But ψ is only multiplied by sin(2ψ), cos(2ψ) scaled by sinh(ρ)=0 at init.
        # So its value at init does not enter the forward pass.  Expected non-zero.
        is_psi_base = "psi_base" in k
        if mx > 1e-12:
            if is_psi_base:
                new_zero += 1  # accepted, gated by ρ=0
            else:
                new_nonzero += 1
                print(f"  NON-ZERO new param at init: {k}  max|·|={mx:.3e}")
        else:
            new_zero += 1
    nn_rse.load_state_dict(nn_sd)

    # Verify ρ parameters zero-init
    for i, layer in enumerate(nn_rse.encoder.layers):
        att = layer.att
        if not att.use_rse:
            continue
        assert att.nonnormal_rho_base.abs().max() == 0.0, f"rho_base non-zero at L{i}"
        assert att.nonnormal_rho_w1.abs().max() == 0.0, f"rho_w1 non-zero at L{i}"
        assert att.nonnormal_mu.abs().max() == 0.0, f"mu non-zero at L{i}"
    print("  ✓ all ρ_base, ρ_w1, μ zero-init confirmed")
    print("  (ψ_base is U(0, 2π) by design; irrelevant at ρ=0 since sinh(ρ)=0)")
    print(f"  copied {copied} keys, {new_zero} acceptable new keys, {new_nonzero} unexpected non-zero new keys")

    n_anchor = sum(p.numel() for p in anchor.parameters())
    n_nn = sum(p.numel() for p in nn_rse.parameters())
    print(f"  params: anchor={n_anchor:,}  T2={n_nn:,}  extra={n_nn - n_anchor:,}")

    anchor.eval(); nn_rse.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        B, n_mels, T = 2, 80, 300
        mels = torch.randn(B, n_mels, T)
        mel_lens = torch.tensor([T, T - 40])

        lp_a, len_a, _ = anchor(mels, mel_lens)
        lp_n, len_n, _ = nn_rse(mels, mel_lens)

    assert len_a.tolist() == len_n.tolist(), "output lengths disagree"
    diff = (lp_a - lp_n).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    print(f"\nForward output comparison:")
    print(f"  max|anchor − nonnormal_rse|  = {max_abs:.2e}")
    print(f"  mean|anchor − nonnormal_rse| = {mean_abs:.2e}")
    print(f"  ref std of anchor            = {lp_a.std().item():.4f}")

    # Sequential vs chunked scan; algorithmically bit-exact when ρ = 0,
    # but FP accumulation order differs.  Expect within 1e-3.
    TOL = 1e-3
    if max_abs <= TOL:
        print(f"\n  ✓ zero-regression-at-init CONFIRMED (max|diff| ≤ {TOL:.0e})")
        return 0
    else:
        print(f"\n  ✗ FAIL — max|diff| {max_abs:.3e} > {TOL:.0e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
