"""Scan `outputs/*/results.json` and build the master run index.

Produces `outputs/_index.csv` — one row per completed run, used as the
single source of truth by `reporting.tables` and `reporting.plots`.

Usage:
    uv run python -m src.reporting.collect
    uv run python -m src.reporting.collect --output outputs/_index.csv

Idempotent. Runs with no results.json are skipped. Runs whose parent
directory matches an entry in `configs/experiments.yaml` inherit the tags
from the registry; runs that don't match (e.g. ad-hoc experiments) still
appear in the index with an empty tags field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


# Columns we extract from results.json. Order matters — it becomes the
# default CSV column order.
COLUMNS = [
    "run_id", "backbone", "seed", "tags", "shortlist",
    "params_total", "params_encoder",
    "best_dev_cer", "best_dev_epoch",
    "test_cer", "test_wer",
    "chunked_2s_reset_cer", "chunked_5s_reset_cer", "chunked_10s_reset_cer",
    "chunked_2s_carry_cer", "chunked_5s_carry_cer", "chunked_10s_carry_cer",
    "avg_epoch_sec", "total_train_sec", "n_epochs",
    "peak_mem_gb", "git_sha", "gpu_name", "torch_version",
]


def _best_dev_epoch(history: list[dict]) -> int | None:
    if not history:
        return None
    best = min(history, key=lambda h: float(h.get("dev_cer", 1.0)))
    try:
        return int(best["epoch"])
    except (KeyError, ValueError):
        return None


def _chunked(results: dict, key: str, metric: str) -> float | None:
    chunk = results.get("chunked", {}).get(key)
    if chunk is None:
        return None
    return float(chunk.get(metric)) if chunk.get(metric) is not None else None


def _load_registry(registry_path: Path) -> dict[str, dict]:
    """Return {run_id_prefix: {tags, shortlist}} keyed by the exp id."""
    if not registry_path.exists():
        return {}
    with open(registry_path) as f:
        data = yaml.safe_load(f) or {}
    out: dict[str, dict] = {}
    for e in data.get("experiments", []):
        out[e["id"]] = {
            "tags": ",".join(e.get("tags", [])),
            "shortlist": bool(e.get("shortlist", False)),
        }
    return out


def _match_registry(run_dir_name: str, registry: dict[str, dict]) -> dict:
    """Find the registry entry whose id is a prefix of the run directory name."""
    for rid, meta in registry.items():
        if run_dir_name == rid or run_dir_name.startswith(rid + "_seed"):
            return meta
    return {"tags": "", "shortlist": False}


def _parse_run(run_dir: Path, registry: dict[str, dict]) -> dict | None:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return None
    try:
        with open(results_path) as f:
            r = json.load(f)
    except json.JSONDecodeError:
        return None

    history = r.get("history", [])
    params = r.get("params", {})
    cfg = r.get("config_snapshot", {})
    meta = _match_registry(run_dir.name, registry)

    epoch_times = [float(h.get("epoch_time_sec", 0)) for h in history]
    peak_mem = max((float(h.get("peak_mem_gb", 0)) for h in history), default=0.0)

    row = {
        "run_id": run_dir.name,
        "backbone": r.get("backbone", cfg.get("backbone", "")),
        "seed": cfg.get("seed"),
        "tags": meta["tags"],
        "shortlist": meta["shortlist"],
        "params_total": params.get("total"),
        "params_encoder": params.get("encoder"),
        "best_dev_cer": r.get("best_dev_cer"),
        "best_dev_epoch": _best_dev_epoch(history),
        "test_cer": r.get("test", {}).get("cer"),
        "test_wer": r.get("test", {}).get("wer"),
        "chunked_2s_reset_cer": _chunked(r, "2.0s_reset", "cer"),
        "chunked_5s_reset_cer": _chunked(r, "5.0s_reset", "cer"),
        "chunked_10s_reset_cer": _chunked(r, "10.0s_reset", "cer"),
        "chunked_2s_carry_cer": _chunked(r, "2.0s_carry", "cer"),
        "chunked_5s_carry_cer": _chunked(r, "5.0s_carry", "cer"),
        "chunked_10s_carry_cer": _chunked(r, "10.0s_carry", "cer"),
        "avg_epoch_sec": sum(epoch_times) / max(len(epoch_times), 1),
        "total_train_sec": sum(epoch_times),
        "n_epochs": len(history),
        "peak_mem_gb": peak_mem,
        "git_sha": r.get("git_sha", ""),
        "gpu_name": r.get("gpu_name", ""),
        "torch_version": r.get("torch_version", ""),
    }
    return row


def collect(output_root: Path, registry_path: Path) -> pd.DataFrame:
    """Walk `output_root` and build a DataFrame of all runs with a results.json."""
    registry = _load_registry(registry_path)
    rows: list[dict] = []
    for path in sorted(output_root.rglob("results.json")):
        parsed = _parse_run(path.parent, registry)
        if parsed is not None:
            rows.append(parsed)
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows)
    # Preserve column order where possible
    ordered = [c for c in COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--registry", default="configs/experiments.yaml")
    parser.add_argument("--output", default="outputs/_index.csv")
    args = parser.parse_args()

    df = collect(Path(args.output_root), Path(args.registry))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} run(s) → {out_path}")
    if len(df):
        cols_to_show = ["run_id", "backbone", "best_dev_cer", "test_cer", "avg_epoch_sec"]
        visible = [c for c in cols_to_show if c in df.columns]
        print(df[visible].to_string(index=False))


if __name__ == "__main__":
    main()
