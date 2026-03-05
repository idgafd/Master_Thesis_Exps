#!/usr/bin/env python3
"""
Commit and push experiment results to GitHub.

Usage:
    python scripts/push_results.py --output-dir ./outputs/run-003_bidir-cosine

    # Dry-run: show what would be committed without actually doing it:
    python scripts/push_results.py --dry-run

What gets committed:
    - results.json, results_tables.txt
    - history_*.json
    - config_snapshot.yaml, run_info.json, vocab.json
    - samples_*.txt
    - plots/*.png
    NOT committed (excluded by outputs/.gitignore):
    - *.pt  (model weights — too large)
    - *.pdf (redundant with PNG)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]   # Master_Thesis_Exps/
EXP_DIR   = Path(__file__).resolve().parents[1]   # asr_backbone_comparison/


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    kwargs = dict(cwd=REPO_ROOT, check=check)
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    return subprocess.run(cmd, **kwargs)


def get_run_id(output_dir: Path) -> str:
    run_info = output_dir / "run_info.json"
    if run_info.exists():
        with open(run_info) as f:
            data = json.load(f)
        return data.get("run_id", output_dir.name)
    return output_dir.name


def get_completed_backbones(output_dir: Path) -> list[str]:
    results_file = output_dir / "results.json"
    if not results_file.exists():
        return []
    with open(results_file) as f:
        results = json.load(f)
    return [bb for bb, r in results.items() if "error" not in r]


def stage_results(output_dir: Path, dry_run: bool) -> list[str]:
    """Stage all result files (respecting .gitignore) and return list of staged paths."""
    rel = output_dir.relative_to(REPO_ROOT)

    patterns = [
        rel / "results.json",
        rel / "results_tables.txt",
        rel / "run_info.json",
        rel / "config_snapshot.yaml",
        rel / "vocab.json",
    ]
    # Glob for per-backbone files
    for f in output_dir.glob("history_*.json"):
        patterns.append(rel / f.name)
    for f in output_dir.glob("samples_*.txt"):
        patterns.append(rel / f.name)
    # PNG plots
    plots_dir = output_dir / "plots"
    if plots_dir.exists():
        for f in plots_dir.glob("*.png"):
            patterns.append(rel / "plots" / f.name)
    # .gitignore itself
    patterns.append(rel / ".gitignore" if (output_dir / ".gitignore").exists() else None)
    patterns = [str(p) for p in patterns if p is not None]

    existing = [p for p in patterns if (REPO_ROOT / p).exists()]

    if dry_run:
        print("Would stage:")
        for p in existing:
            print(f"  {p}")
    else:
        run(["git", "add"] + existing)

    return existing


def main():
    parser = argparse.ArgumentParser(description="Commit and push experiment results")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to the run output directory (e.g. outputs/run-003_bidir-cosine)",
    )
    parser.add_argument(
        "--message", "-m",
        help="Custom commit message (auto-generated if omitted)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without actually committing or pushing",
    )
    parser.add_argument(
        "--no-push", action="store_true",
        help="Commit but do not push to remote",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        print(f"Error: output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    run_id = get_run_id(output_dir)
    backbones = get_completed_backbones(output_dir)

    print(f"Run ID  : {run_id}")
    print(f"Backbones: {', '.join(backbones) if backbones else '(none completed)'}")

    staged = stage_results(output_dir, dry_run=args.dry_run)
    if not staged:
        print("Nothing to commit.")
        sys.exit(0)

    # Check if there is actually anything new to commit
    if not args.dry_run:
        status = run(["git", "status", "--porcelain"] + staged, capture=True).stdout.strip()
        if not status:
            print("Nothing new to commit (all result files already up to date).")
            sys.exit(0)

    # Build commit message
    if args.message:
        msg = args.message
    else:
        bb_str = ", ".join(backbones) if backbones else "no backbones"
        msg = f"results: {run_id} — {bb_str}"

    if args.dry_run:
        print(f"\nWould commit with message:\n  {msg.splitlines()[0]}")
        print("(dry-run — no changes made)")
        return

    run(["git", "commit", "-m", msg])
    print("Committed.")

    if not args.no_push:
        run(["git", "push", "origin", "main"])
        print("Pushed to origin/main.")
    else:
        print("Skipped push (--no-push).")


if __name__ == "__main__":
    main()
