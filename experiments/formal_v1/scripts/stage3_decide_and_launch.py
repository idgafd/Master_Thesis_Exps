#!/usr/bin/env python3
"""Stage-3 autonomous follow-up.

Reads the final dev CER from the two Stage-3 RSE runs (rwkv6_rse and
rwkv6_rse_convshift), classifies the joint outcome against the decision tree,
chooses the next 2-GPU batch, and launches the two jobs.

Decision tree (vs baselines: rwkv6=0.1258, rwkv6_convshift_trap=0.1150):

  GPU0=WIN, GPU1=WIN   →  Scenario A:  rwkv6_rse_m2 + rwkv6_rse_m2_convshift
  GPU0=FLAT, GPU1=WIN  →  Scenario B:  rwkv6_rse_convshift_headscale + rwkv6_rse_m2_convshift
  GPU0=WIN, GPU1=FLAT  →  Scenario C:  rwkv6_rse_headscale + rwkv6_rse_m2
  GPU0=FLAT, GPU1=FLAT →  Scenario D:  rwkv6_rse_formant_init + rwkv6_rse_m4
  GPU0=REGRESS or GPU1=REGRESS → Scenario E:  rwkv6_rse_tightclip + rwkv6_convshift_k5

Thresholds (relative to direct baseline):
  WIN     ≤ -2 % relative
  FLAT    -2 % < δ < +2 %
  REGRESS ≥ +2 %

Run as:  uv run python scripts/stage3_decide_and_launch.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_RSE = ROOT / "outputs" / "stage3_01_rwkv6_rse_seed42"
RUN_RSE_CS = ROOT / "outputs" / "stage3_02_rwkv6_rse_convshift_seed42"
LOG_DIR = ROOT / "outputs" / "logs"

# Direct comparison baselines (Stage 2 reference dev CERs)
BASELINE_RSE = 0.1258         # rwkv6 baseline for rwkv6_rse
BASELINE_RSE_CS = 0.1150      # rwkv6_convshift_trap for rwkv6_rse_convshift

# Decision thresholds (relative to baseline)
THRESH_WIN = -0.02
THRESH_REG = +0.02


def best_dev_cer(run_dir: Path) -> float | None:
    """Read history.csv and return the minimum dev_cer over all completed epochs.

    Skips rows where dev_cer is NaN/Inf (avoids first-row poisoning in early
    epochs where eval can produce NaN before normalization stabilizes).
    """
    import math as _m
    h = run_dir / "history.csv"
    if not h.exists():
        return None
    best = None
    for row in csv.DictReader(open(h)):
        try:
            cer = float(row["dev_cer"])
        except (KeyError, ValueError):
            continue
        if _m.isnan(cer) or _m.isinf(cer):
            continue
        if best is None or cer < best:
            best = cer
    return best


def classify(cer: float, baseline: float) -> str:
    rel = (cer - baseline) / baseline
    if rel <= THRESH_WIN:
        return "WIN"
    if rel >= THRESH_REG:
        return "REGRESS"
    return "FLAT"


def pick_next_batch(rse_label: str, rse_cs_label: str) -> tuple[str, str, str]:
    """Returns (scenario, gpu0_backbone, gpu1_backbone).

    All chosen backbones are registered in src/models/encoder.py; the script
    will not pick a name that fails to build.
    """
    if rse_label == "REGRESS" or rse_cs_label == "REGRESS":
        # Gradient pathology in rotation params.  Don't auto-launch anything —
        # blindly committing 4 GPU-hours to a broken mechanism is wasteful.
        # The user gets the diagnosis on return and can decide.
        return ("E", "<NONE>", "<NONE>")
    if rse_label == "WIN" and rse_cs_label == "WIN":
        return ("A", "rwkv6_rse_m2", "rwkv6_rse_m2_convshift")
    if rse_label == "WIN" and rse_cs_label == "FLAT":
        return ("C", "rwkv6_rse_headscale", "rwkv6_rse_m2")
    if rse_label == "FLAT" and rse_cs_label == "WIN":
        return ("B", "rwkv6_rse_convshift_headscale", "rwkv6_rse_m2_convshift")
    # Both flat: try multi-scale to add capacity, paired with M=4 to test scale-richness.
    return ("D", "rwkv6_rse_m2", "rwkv6_rse_m4")


def launch(backbone: str, gpu: int, output_dir: Path, log_path: Path,
           epochs: int = 30, seed: int = 42, dry_run: bool = False) -> int | None:
    cmd = [
        "uv", "run", "python", "scripts/run_experiment.py",
        "--backbone", backbone,
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--output-dir", str(output_dir),
        "--gpu", str(gpu),
    ]
    print(f"  launch:  GPU{gpu}  {backbone}  →  {output_dir}", flush=True)
    print(f"           {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    if dry_run:
        return None
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        cwd=str(ROOT),
        start_new_session=True,
    )
    return proc.pid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print decision and command but do not launch.")
    ap.add_argument("--prefix", default="stage3p5",
                    help="Output dir prefix for the next batch.")
    args = ap.parse_args()

    print("=" * 64)
    print("Stage-3 RSE follow-up decision script")
    print("=" * 64)

    rse_cer = best_dev_cer(RUN_RSE)
    rse_cs_cer = best_dev_cer(RUN_RSE_CS)

    if rse_cer is None or rse_cs_cer is None:
        print(f"ERROR: incomplete history — rse={rse_cer}  rse_cs={rse_cs_cer}")
        print(f"  expected: {RUN_RSE / 'history.csv'} and {RUN_RSE_CS / 'history.csv'}")
        return 1

    rse_label = classify(rse_cer, BASELINE_RSE)
    rse_cs_label = classify(rse_cs_cer, BASELINE_RSE_CS)

    print(f"Stage-3 results:")
    print(f"  rwkv6_rse           best dev CER = {rse_cer:.4f}   "
          f"(baseline {BASELINE_RSE:.4f}, Δ={100*(rse_cer-BASELINE_RSE)/BASELINE_RSE:+.1f} %)  →  {rse_label}")
    print(f"  rwkv6_rse_convshift best dev CER = {rse_cs_cer:.4f}   "
          f"(baseline {BASELINE_RSE_CS:.4f}, Δ={100*(rse_cs_cer-BASELINE_RSE_CS)/BASELINE_RSE_CS:+.1f} %)  →  {rse_cs_label}")

    scenario, gpu0_bb, gpu1_bb = pick_next_batch(rse_label, rse_cs_label)
    print(f"\nScenario {scenario}.  Next batch:")
    print(f"  GPU 0: {gpu0_bb}")
    print(f"  GPU 1: {gpu1_bb}")

    if gpu0_bb == "<NONE>" or gpu1_bb == "<NONE>":
        print("\nNo auto-launch (regression scenario). Awaiting user triage.")
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out0 = ROOT / "outputs" / f"{args.prefix}_01_{gpu0_bb}_seed42"
    out1 = ROOT / "outputs" / f"{args.prefix}_02_{gpu1_bb}_seed42"
    log0 = LOG_DIR / f"{args.prefix}_01_{gpu0_bb}.log"
    log1 = LOG_DIR / f"{args.prefix}_02_{gpu1_bb}.log"

    pid0 = launch(gpu0_bb, 0, out0, log0, dry_run=args.dry_run)
    pid1 = launch(gpu1_bb, 1, out1, log1, dry_run=args.dry_run)

    print(f"\nLaunched PIDs: GPU0={pid0}  GPU1={pid1}")
    print(f"Logs:")
    print(f"  {log0}")
    print(f"  {log1}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
