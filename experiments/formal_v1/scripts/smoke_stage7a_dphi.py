#!/usr/bin/env python3
"""Stage-7A A1′ zero-regression-at-init smoke test.

The A1′ backbone `rwkv6_rse_dphi_viscosity` differs from
`rwkv6_rse_strong_viscosity` only by a zero-initialized Linear
(`readphase_proj`) per RSE-enabled layer.  At t=0, that Linear outputs
zero, so φ ≡ 0 and `exp(-iφ) = 1`; the readout collapses to the
anchor's readout bit-exactly.

This script builds both models with the same seed, copies the anchor's
weights into the A1′ model (the A1′ model has a superset of params),
runs the same random input through both, and asserts
    max |y_A1′ - y_anchor|  <  1e-5
per-layer output.  If this fails, the A1′ implementation has an
unintended side effect (e.g. forgot to skip the rotation when φ=0).

Run:
    uv run python scripts/smoke_stage7a_dphi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.config import ExperimentConfig
from src.models.asr_model import ASRModel


# English charset: a-z + space + apostrophe + CTC blank = 29.
# Hard-coded here so the smoke test has no dependency on src.data.__init__
# (which eagerly imports the HuggingFace `datasets` librispeech loader).
VOCAB_SIZE = 29


def build(backbone: str, seed: int = 42) -> ASRModel:
    torch.manual_seed(seed)
    cfg = ExperimentConfig()
    cfg.backbone = backbone
    cfg.rwkv_mode = "recurrent"
    model = ASRModel(vocab_size=VOCAB_SIZE, cfg=cfg)
    return model


def main() -> int:
    print("Building anchor and A1′ models with matched seed …")
    anchor = build("rwkv6_rse_strong_viscosity", seed=42)
    dphi = build("rwkv6_rse_dphi_viscosity", seed=42)

    # Copy anchor weights into dphi model.  dphi has a superset
    # (extra readphase_proj per layer, zero-init).  So copy overlapping
    # keys; leave dphi-only keys at their zero init.
    anchor_sd = anchor.state_dict()
    dphi_sd = dphi.state_dict()
    copied, skipped_new, mismatched = 0, 0, 0
    for k, v in anchor_sd.items():
        if k in dphi_sd:
            if dphi_sd[k].shape == v.shape:
                dphi_sd[k] = v.clone()
                copied += 1
            else:
                print(f"  shape mismatch: {k}  anchor={v.shape}  dphi={dphi_sd[k].shape}")
                mismatched += 1
    # Identify new parameters in dphi and confirm zero-init.
    dphi_only = [k for k in dphi_sd.keys() if k not in anchor_sd]
    dphi.load_state_dict(dphi_sd)
    for k in dphi_only:
        t = dphi.state_dict()[k]
        mx = t.abs().max().item() if t.numel() > 0 else 0.0
        if mx > 1e-12:
            print(f"  NON-ZERO dphi-only param: {k}  max|·|={mx:.3e}")
        else:
            skipped_new += 1
    print(f"  copied {copied} keys, {skipped_new} zero-init new keys, {mismatched} mismatches")
    assert mismatched == 0, "unexpected shape mismatch"
    assert skipped_new == len(dphi_only), "new params not zero-initialised"

    # Sanity check parameter counts — dphi should have 6 × (256 × 4·32) + 6 × (4·32)
    # additional weights for the per-layer readphase_proj (we use n_head=4,
    # Bk=32, hidden=256).
    n_anchor = sum(p.numel() for p in anchor.parameters())
    n_dphi = sum(p.numel() for p in dphi.parameters())
    n_extra = n_dphi - n_anchor
    n_extra_expected = 6 * (256 * (4 * 32) + (4 * 32))   # 6 layers × (W + b)
    print(f"  params: anchor={n_anchor:,}  dphi={n_dphi:,}  extra={n_extra:,}  "
          f"(expected ≈ {n_extra_expected:,})")

    anchor.eval(); dphi.eval()
    with torch.no_grad():
        B, n_mels, T = 2, 80, 400
        mels = torch.randn(B, n_mels, T)
        mel_lens = torch.tensor([T, T - 50])

        lp_a, len_a, _ = anchor(mels, mel_lens)
        lp_d, len_d, _ = dphi(mels, mel_lens)

    assert len_a.tolist() == len_d.tolist(), "output lengths disagree"
    diff = (lp_a - lp_d).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    print(f"\nForward output comparison:")
    print(f"  max|anchor − dphi|  = {max_abs:.2e}")
    print(f"  mean|anchor − dphi| = {mean_abs:.2e}")
    print(f"  ref std of anchor   = {lp_a.std().item():.4f}")

    TOL = 1e-5
    if max_abs <= TOL:
        print(f"\n  ✓ zero-regression-at-init CONFIRMED (max|diff| ≤ {TOL:.0e})")
        return 0
    else:
        print(f"\n  ✗ FAIL — max|diff| {max_abs:.3e} > {TOL:.0e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
