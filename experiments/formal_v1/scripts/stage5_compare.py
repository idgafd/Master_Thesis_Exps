#!/usr/bin/env python3
"""Side-by-side Stage-5 P²-RSE vs Stage-4 reference trajectories.

Reads live history.csv from the two Stage-5 runs and prints a per-epoch
comparison against the Stage-3 / Stage-4 reference trajectories from the
thesis (RSE_DISTRIBUTION_ANALYSIS.md §5.1 and §5.3).

Usage:
    uv run python scripts/stage5_compare.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


# ── Stage-3 and Stage-4 reference trajectories (best dev CER per epoch) ──
# Hand-compiled from RSE_DISTRIBUTION_ANALYSIS.md §5.1 (per-epoch) and §5.2 (final).
# "nan" means the trajectory was not reported at that epoch in the document.
REFERENCE_TRAJECTORIES = {
    "rwkv6_rse      (S3 uniform, 0.1251 best)":      {
        5: 0.2365, 10: 0.1748, 14: 0.1540, 15: 0.1485,
        19: 0.1367, 20: 0.1360, 25: 0.1271, 30: 0.1251,
    },
    "rwkv6_rse_depth (S4 depth-graded, 0.1207 best)": {
        5: 0.2282, 10: 0.1695, 14: 0.1488, 15: 0.1437,
        19: 0.1317, 20: 0.1294, 25: 0.1223, 30: 0.1208,
    },
    "rwkv6_rse_strong (S4 uniform-large, 0.1192 best — ceiling to beat)": {
        5: 0.2312, 10: 0.1680, 15: 0.1430,
        19: 0.1304, 20: 0.1280, 25: 0.1214, 30: 0.1197,
    },
}


def load_history(path: Path) -> dict[int, float]:
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out[int(row["epoch"])] = float(row["dev_cer"])
    return out


def main():
    runs = {
        "P1 p2rse  (linear)":             REPO / "outputs/stage5_01_rwkv6_p2rse_seed42/history.csv",
        "P1 p2rse  (softmax)":            REPO / "outputs/stage5_02_rwkv6_p2rse_softmax_seed42/history.csv",
        "P2 p2rse_depth  (stacked)":      REPO / "outputs/stage5_03_rwkv6_p2rse_depth_seed42/history.csv",
        "P2 p2rse_strong (stacked)":      REPO / "outputs/stage5_04_rwkv6_p2rse_strong_seed42/history.csv",
        "P3 rse_strong + viscosity":      REPO / "outputs/stage5_05_rwkv6_rse_strong_viscosity_seed42/history.csv",
        "P3 rse_depth  + viscosity":      REPO / "outputs/stage5_06_rwkv6_rse_depth_viscosity_seed42/history.csv",
    }
    stage5_data = {name: load_history(p) for name, p in runs.items()}

    max_ep = max((max(d.keys()) if d else 0 for d in stage5_data.values()), default=0)
    if max_ep == 0:
        print("No Stage-5 epochs completed yet.")
        return

    # All epochs we want to show: every Stage-5 epoch, plus reference checkpoints.
    ref_ckpts = sorted(set().union(*[r.keys() for r in REFERENCE_TRAJECTORIES.values()]))
    display_eps = sorted(set(range(1, max_ep + 1)) | set(ref_ckpts))
    display_eps = [e for e in display_eps if e <= 30]

    # Header
    headers = ["Ep"] + list(stage5_data.keys()) + list(REFERENCE_TRAJECTORIES.keys())
    widths = [max(len(h), 8) for h in headers]
    widths[0] = 4
    row_fmt = " | ".join(f"{{:>{w}}}" for w in widths)
    print()
    print(row_fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))

    for ep in display_eps:
        row = [str(ep)]
        for name in stage5_data.keys():
            v = stage5_data[name].get(ep)
            row.append(f"{v:.4f}" if v is not None else " "*6)
        for name in REFERENCE_TRAJECTORIES.keys():
            v = REFERENCE_TRAJECTORIES[name].get(ep)
            row.append(f"{v:.4f}" if v is not None else " "*6)
        print(row_fmt.format(*row))

    # Current best
    print("\n── Best dev CER so far ─────────────────────────────────")
    for name, hist in stage5_data.items():
        if hist:
            best_ep, best_cer = min(hist.items(), key=lambda kv: kv[1])
            print(f"  {name}: {best_cer:.4f} (epoch {best_ep})")

    # Stage-4 ceiling target
    print("\n── Stage-4 ceilings (targets to beat) ──────────────────")
    print("  rwkv6_rse_depth  best dev CER = 0.1207  (test 0.1200)")
    print("  rwkv6_rse_strong best dev CER = 0.1192  (test 0.1188)   ← break: ≤ 0.1160")

    # Projection if current best trajectory holds
    print("\n── Classification thresholds from STAGE5_PLAN.md §3 ────")
    print("  BREAK    dev CER ≤ 0.1160  (≥ 2× seed-noise below Stage-4 strong)")
    print("  MARGINAL dev CER ∈ (0.1160, 0.1180]")
    print("  PLATEAU  dev CER > 0.1180")


if __name__ == "__main__":
    main()
