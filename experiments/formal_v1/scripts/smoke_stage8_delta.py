#!/usr/bin/env python3
"""Stage-8 T1 — `rwkv6_delta_warmstart_fixed` zero-regression-at-init smoke test.

At init, the per-head hard gate `delta_recurrent_gate` is zero, so the
effective erasure strength β ≡ 0 and the sequential delta scan reduces
to a sequential implementation of the standard RWKV-6 recurrent update
(`_chunked_wkv` with `L=1` case per step). The two paths should produce
bit-identical output within FP tolerance on the same batch.

The test builds anchor = `rwkv6` (vanilla) and delta = `rwkv6_delta_warmstart_fixed`
with matched seed, copies the overlapping weights, runs identical input
through both, and asserts:

    max|y_anchor − y_delta|  <  1e-4

(Looser tolerance than Stage 7A because the sequential vs chunked
backends differ in FP accumulation order; same algorithm, different
kernel.)

Run:
    uv run python scripts/smoke_stage8_delta.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.config import ExperimentConfig
from src.models.asr_model import ASRModel


VOCAB_SIZE = 29  # English char vocab: 26 + space + apostrophe + blank


def build(backbone: str, seed: int = 42) -> ASRModel:
    torch.manual_seed(seed)
    cfg = ExperimentConfig()
    cfg.backbone = backbone
    cfg.rwkv_mode = "recurrent"
    return ASRModel(vocab_size=VOCAB_SIZE, cfg=cfg)


def main() -> int:
    print("Building anchor `rwkv6` and delta `rwkv6_delta_warmstart_fixed` with matched seed…")
    anchor = build("rwkv6", seed=42)
    delta = build("rwkv6_delta_warmstart_fixed", seed=42)

    anchor_sd = anchor.state_dict()
    delta_sd = delta.state_dict()
    copied, new_zero, new_nonzero = 0, 0, 0
    for k, v in anchor_sd.items():
        if k in delta_sd and delta_sd[k].shape == v.shape:
            delta_sd[k] = v.clone()
            copied += 1
    for k in delta_sd.keys():
        if k not in anchor_sd:
            t = delta_sd[k]
            mx = t.abs().max().item() if t.numel() > 0 else 0.0
            if mx > 1e-12:
                new_nonzero += 1
                print(f"  NON-ZERO delta-only param at init: {k}  max|·|={mx:.3e}")
            else:
                new_zero += 1
    delta.load_state_dict(delta_sd)

    # The delta_params include k_k, k_a, a0, a1, a2 — some of which are
    # non-zero by design (k_k=0.85, k_a=1.0, a0=-5 for warmstart).  These
    # would normally contribute to the erase, but the hard gate
    # delta_recurrent_gate is zero-init, so β_effective = gate * iclr = 0.
    print(f"  copied {copied} keys, {new_zero} zero-init new keys, {new_nonzero} non-zero new keys")
    print("  (non-zero new params are internal to DeltaRuleParams;"
          " their effect is gated by `delta_recurrent_gate = 0`)")

    # Verify the gate is actually zero at init:
    for i, layer in enumerate(delta.encoder.layers):
        gate = layer.att.delta_recurrent_gate
        assert gate.abs().max() == 0.0, (
            f"delta_recurrent_gate at L{i} is not zero-init: max|·|={gate.abs().max()}")
    print("  ✓ all delta_recurrent_gate parameters zero-init confirmed")

    n_anchor = sum(p.numel() for p in anchor.parameters())
    n_delta = sum(p.numel() for p in delta.parameters())
    print(f"  params: anchor={n_anchor:,}  delta={n_delta:,}  extra={n_delta - n_anchor:,}")

    anchor.eval(); delta.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        B, n_mels, T = 2, 80, 300
        mels = torch.randn(B, n_mels, T)
        mel_lens = torch.tensor([T, T - 40])

        lp_a, len_a, _ = anchor(mels, mel_lens)
        lp_d, len_d, _ = delta(mels, mel_lens)

    assert len_a.tolist() == len_d.tolist(), "output lengths disagree"
    diff = (lp_a - lp_d).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    print(f"\nForward output comparison:")
    print(f"  max|anchor − delta|  = {max_abs:.2e}")
    print(f"  mean|anchor − delta| = {mean_abs:.2e}")
    print(f"  ref std of anchor    = {lp_a.std().item():.4f}")

    # Tolerance: sequential vs chunked scan differ in FP accumulation
    # order. Expect bit-exact algorithmically but small numerical drift.
    TOL = 1e-4
    if max_abs <= TOL:
        print(f"\n  ✓ zero-regression-at-init CONFIRMED (max|diff| ≤ {TOL:.0e})")
        return 0
    else:
        print(f"\n  ✗ FAIL — max|diff| {max_abs:.3e} > {TOL:.0e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
