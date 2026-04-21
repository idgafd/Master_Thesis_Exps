#!/usr/bin/env python3
"""Stage-9 — `rwkv6_sparse_nonnormal_rse_viscosity` zero-regression-at-init smoke test.

At init:
  - `sparse_nn_gate` = 0 per head per layer
  - `nonnormal_rho_base` = 0, `nonnormal_rho_w1` = 0 (LoRA gate zero) ⇒ ρ_raw ≡ 0
  - Same for ψ_raw

Under the Stage-9 forward:
  rho_h_pre = κ · softplus(λ̃) · tanh(ρ_raw)
  rho_h = rho_h_pre · gate       # zero regardless of κ, regardless of ρ_raw

⇒ ρ_eff ≡ 0 ⇒ P = I ⇒ G = e^{-λ} R(θ) = exact RSE+viscosity transition.

Anchor: `rwkv6_rse_strong_viscosity`.  Expected bit-match to FP tolerance
(sequential-vs-chunked scan order differs; accept 1e-3).

Run:
    uv run python scripts/smoke_stage9_sparse.py
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
    print("Building anchor (`rwkv6_rse_strong_viscosity`) and Stage-9 "
          "(`rwkv6_sparse_nonnormal_rse_viscosity`) with matched seed…")
    anchor = build("rwkv6_rse_strong_viscosity", seed=42)
    s9 = build("rwkv6_sparse_nonnormal_rse_viscosity", seed=42)

    anchor_sd = anchor.state_dict()
    s9_sd = s9.state_dict()
    copied = 0
    new_keys = [k for k in s9_sd.keys() if k not in anchor_sd]
    for k, v in anchor_sd.items():
        if k in s9_sd and s9_sd[k].shape == v.shape:
            s9_sd[k] = v.clone()
            copied += 1
    non_zero_unexpected = 0
    for k in new_keys:
        t = s9_sd[k]
        mx = t.abs().max().item() if t.numel() > 0 else 0.0
        if mx > 1e-12:
            if "psi_base" in k:
                # ψ_base U(0, 2π) by design — irrelevant at gate=0.
                continue
            if "psi_w2" in k or "rho_w2" in k:
                # w2 LoRA is init U(-0.01, 0.01) by convention; zero-gate
                # makes the LoRA contribution irrelevant.
                continue
            non_zero_unexpected += 1
            print(f"  UNEXPECTED NON-ZERO new param: {k}  max|·|={mx:.3e}")
    s9.load_state_dict(s9_sd)
    print(f"  copied {copied} keys, {non_zero_unexpected} unexpected non-zero new params")

    # Verify gate raw is zero-init (so sigmoid(raw)=0.5 neutral).  The
    # zero-regression contract holds via ρ_raw zero-init, not via gate=0.
    import torch.nn.functional as Fcheck
    for i, layer in enumerate(s9.encoder.layers):
        att = layer.att
        assert att.sparse_nn_gate_raw.abs().max() == 0.0, \
            f"sparse_nn_gate_raw not zero at L{i}"
        g_eff = torch.sigmoid(att.sparse_nn_gate_raw)
        assert (g_eff - 0.5).abs().max() < 1e-6, \
            f"gate sigmoid(0) should be 0.5 at init, got {g_eff}"
    print("  ✓ sparse_nn_gate_raw zero-init (sigmoid=0.5) confirmed")
    # Verify psi LoRA weights are NOT present (static ψ)
    for i, layer in enumerate(s9.encoder.layers):
        att = layer.att
        assert not hasattr(att, "nonnormal_psi_w1"), \
            f"ψ LoRA should be absent under static-ψ flag at L{i}"
    print("  ✓ static ψ (no ψ LoRA weights) confirmed")

    # Verify κ = 0.4 for Stage-9 backbones
    for i, layer in enumerate(s9.encoder.layers):
        assert abs(layer.att.nonnormal_rho_kappa - 0.4) < 1e-6, \
            f"κ should be 0.4 for Stage-9, got {layer.att.nonnormal_rho_kappa}"
    print("  ✓ κ = 0.4 (Stage-9 tightened clip) confirmed")

    n_anchor = sum(p.numel() for p in anchor.parameters())
    n_s9 = sum(p.numel() for p in s9.parameters())
    print(f"  params: anchor={n_anchor:,}  s9={n_s9:,}  extra={n_s9 - n_anchor:,}")

    anchor.eval(); s9.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        B, n_mels, T = 2, 80, 300
        mels = torch.randn(B, n_mels, T)
        mel_lens = torch.tensor([T, T - 40])

        lp_a, len_a, _ = anchor(mels, mel_lens)
        lp_s, len_s, _ = s9(mels, mel_lens)

    assert len_a.tolist() == len_s.tolist(), "output lengths disagree"
    diff = (lp_a - lp_s).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    print(f"\nForward output comparison:")
    print(f"  max|anchor − s9|   = {max_abs:.2e}")
    print(f"  mean|anchor − s9|  = {mean_abs:.2e}")
    print(f"  ref std of anchor  = {lp_a.std().item():.4f}")

    TOL = 1e-3
    if max_abs <= TOL:
        print(f"\n  ✓ zero-regression-at-init CONFIRMED (max|diff| ≤ {TOL:.0e})")
        return 0
    print(f"\n  ✗ FAIL — max|diff| {max_abs:.3e} > {TOL:.0e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
