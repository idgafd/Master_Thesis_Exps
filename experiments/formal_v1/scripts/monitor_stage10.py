#!/usr/bin/env python3
"""Stage 10 training monitor.

Reads history.csv for each of the five Stage-10 backbones and emits a
concise per-epoch progress table (dev CER, train loss, epoch time, peak
mem).  Also flags matched-epoch halt conditions per STAGE10_PLAN §8.1:
  - Halt at ep 15 if dev CER ≥ +0.006 behind primary reference.
  - Flag REGRESSION if dev CER at ep 30 > 0.1258 + 2σ (σ ≈ 0.0014).

Run repeatedly to watch progress; non-interactive, exits cleanly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# STAGE10_PLAN §6 primary-cohort dev CERs (pre-loaded from §9.1).
PRIMARY_REF = {
    "rwkv6_loglinear":                    ("rwkv6",                0.1258, "vanilla"),
    "rwkv6_m2rnn_sparse":                 ("rwkv6",                0.1258, "vanilla"),
    "rwkv6_convshift_multidil":           ("rwkv6_convshift_trap", 0.1150, "convshift_trap"),
    "rwkv6_convshift_multidil_symmetric": ("rwkv6_convshift_trap", 0.1150, "convshift_trap"),
    "rwkv6_chanmix_bypass":               ("rwkv6",                0.1258, "vanilla"),
}
VANILLA_CER = 0.1258
SEED_NOISE = 0.0014
HALT_DELTA = 0.006
HALT_EPOCH = 15


def _fmt_cer(x, col_width=8):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return " " * col_width
    return f"{x:.4f}".rjust(col_width)


def _summarize(backbone: str) -> dict | None:
    run_dir = Path(f"outputs/{backbone}_seed42")
    hist_path = run_dir / "history.csv"
    if not hist_path.exists():
        return None
    try:
        df = pd.read_csv(hist_path)
    except Exception:
        return None
    if df.empty:
        return None

    last_ep = int(df["epoch"].max())
    last_row = df.loc[df["epoch"].idxmax()]

    # Matched-epoch 15 halt check
    halt_flag = ""
    if last_ep >= HALT_EPOCH and backbone in PRIMARY_REF:
        ref_name, ref_cer, _ = PRIMARY_REF[backbone]
        row15 = df[df["epoch"] == HALT_EPOCH]
        if not row15.empty:
            cer15 = float(row15.iloc[0]["dev_cer"])
            if cer15 - ref_cer >= HALT_DELTA:
                halt_flag = f"  ⚠ HALT (ep15 = {cer15:.4f}, +{cer15-ref_cer:.4f} vs {ref_name})"

    # 30-ep regression flag
    regression_flag = ""
    if last_ep >= 30:
        cer30 = float(last_row["dev_cer"])
        if cer30 > VANILLA_CER + 2 * SEED_NOISE:
            regression_flag = f"  ⚠ REGRESSION (ep30 = {cer30:.4f} > vanilla + 2σ = {VANILLA_CER + 2*SEED_NOISE:.4f})"

    return {
        "backbone": backbone,
        "last_ep": last_ep,
        "dev_cer": float(last_row["dev_cer"]),
        "dev_wer": float(last_row.get("dev_wer", float("nan"))),
        "train_loss": float(last_row["train_loss"]),
        "epoch_time_sec": float(last_row.get("epoch_time_sec", float("nan"))),
        "peak_mem_gb": float(last_row.get("peak_mem_gb", float("nan"))),
        "halt_flag": halt_flag,
        "regression_flag": regression_flag,
        "df": df,
    }


def main():
    backbones = [
        "rwkv6_loglinear",
        "rwkv6_m2rnn_sparse",
        "rwkv6_convshift_multidil",
        "rwkv6_convshift_multidil_symmetric",
        "rwkv6_chanmix_bypass",
    ]

    print(f"\n{'=' * 100}")
    print(f"Stage 10 training progress monitor")
    print(f"{'=' * 100}")
    print(f"{'Backbone':<40} {'Ep':>4} {'Dev CER':>9} {'Dev WER':>9} "
          f"{'Tr Loss':>8} {'Ep sec':>7} {'Mem GB':>7}")
    print("-" * 100)

    any_running = False
    all_done = True
    for bb in backbones:
        s = _summarize(bb)
        if s is None:
            print(f"{bb:<40} {'—':>4} {'not started':>50}")
            all_done = False
            continue

        any_running = s["last_ep"] < 30
        if s["last_ep"] < 30:
            all_done = False

        line = (
            f"{s['backbone']:<40} {s['last_ep']:>4} "
            f"{_fmt_cer(s['dev_cer']):>9} "
            f"{_fmt_cer(s['dev_wer']):>9} "
            f"{s['train_loss']:>8.3f} "
            f"{s['epoch_time_sec']:>7.0f} "
            f"{s['peak_mem_gb']:>7.2f}"
        )
        print(line)
        if s["halt_flag"]:
            print(f"  {s['halt_flag']}")
        if s["regression_flag"]:
            print(f"  {s['regression_flag']}")

    print("-" * 100)

    # Every 5 epochs: show milestone table
    print("\nMatched-epoch tracking (dev CER at ep 5, 10, 15, 20, 25, 30):")
    print(f"{'Backbone':<40} " + " ".join(f"{f'ep{e:>2}':>8}" for e in [5, 10, 15, 20, 25, 30]))
    print("-" * 100)
    for bb in backbones:
        s = _summarize(bb)
        if s is None:
            continue
        df = s["df"]
        cells = []
        for ep in [5, 10, 15, 20, 25, 30]:
            row = df[df["epoch"] == ep]
            if row.empty:
                cells.append(f"{'—':>8}")
            else:
                cells.append(f"{float(row.iloc[0]['dev_cer']):.4f}".rjust(8))
        print(f"{bb:<40} " + " ".join(cells))

    # Summary status
    print()
    if all_done:
        print("✓ All runs complete.")
    elif any_running:
        print("⋯ Runs still in progress.")
    else:
        print("(No runs found yet.)")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
