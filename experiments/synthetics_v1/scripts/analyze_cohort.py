"""Build the cohort summary table from completed runs.

Walks `outputs/cohort_reduced/*/results.json`, extracts headline metrics,
and prints a human-readable table + emits `outputs/cohort_reduced/_index.csv`.

Usage:
    cd experiments/synthetics_v1
    uv run python scripts/analyze_cohort.py
    uv run python scripts/analyze_cohort.py --root outputs/_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tabulate import tabulate


def _verdict(per_seq_acc: float | None) -> str:
    if per_seq_acc is None:
        return "—"
    if per_seq_acc >= 0.99:
        return "PASS"
    if per_seq_acc >= 0.70:
        return "PARTIAL"
    return "FAIL"


def _steps_to(metrics_path: Path, threshold: float) -> int | None:
    """First eval step where per_query_acc crosses `threshold`. None if never."""
    if not metrics_path.exists():
        return None
    with metrics_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("split") not in ("eval", "eval_final"):
                continue
            pq = rec.get("per_query_acc")
            if pq is None:
                continue
            if pq >= threshold:
                return rec.get("step")
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/cohort_reduced")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"output root does not exist: {root}")

    rows = []
    for run_dir in sorted(root.iterdir()):
        results_path = run_dir / "results.json"
        if not results_path.exists():
            continue
        with results_path.open() as f:
            r = json.load(f)
        metrics_path = run_dir / "metrics.jsonl"
        steps_50 = _steps_to(metrics_path, 0.5)
        steps_90 = _steps_to(metrics_path, 0.9)
        rows.append({
            "backbone": r.get("backbone", "?"),
            "T": r.get("seq_len", 0),
            "K": r.get("n_kv_pairs", 0),
            "seed": r.get("seed", 0),
            "params_M": round(r.get("n_parameters", 0) / 1e6, 2),
            "steps": r.get("steps_taken", 0),
            "early_stop": r.get("early_stopped", False),
            "best_per_seq": round(r.get("best_per_seq_acc", -1), 4),
            "final_per_seq": round(r.get("final_per_seq_acc", -1), 4),
            "final_per_query": round(r.get("final_per_query_acc", -1), 4),
            "final_loss": round(r.get("final_loss", float("nan")), 4),
            # Zoology-style steps-to-solve at per_query_acc thresholds.
            # `—` = never crossed within the trained budget.
            "steps_to_0.5": steps_50 if steps_50 is not None else "—",
            "steps_to_0.9": steps_90 if steps_90 is not None else "—",
            "wall_min": round(r.get("wall_sec", 0) / 60, 1),
            "verdict": _verdict(r.get("best_per_seq_acc")),
        })

    if not rows:
        print(f"no completed runs found under {root}")
        return

    rows.sort(key=lambda x: (x["backbone"], x["T"], x["seed"]))
    print(tabulate(rows, headers="keys", tablefmt="github"))

    csv_path = root / "_index.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n→ {csv_path}")


if __name__ == "__main__":
    main()
