#!/usr/bin/env python3
"""Registry-driven experiment executor.

Reads `configs/experiments.yaml`, resolves each selected entry to a
subprocess invocation of `scripts/run_experiment.py`, and launches the
runs (sequentially by default, optionally distributed across GPUs).

Usage:
    # Run everything on GPU 0
    uv run scripts/run_registry.py --all --gpu 0

    # Just the shortlist, multi-seed
    uv run scripts/run_registry.py --shortlist --seeds 42,123,777 --gpu 0

    # Filter by id or tag
    uv run scripts/run_registry.py --ids exp09_lion,exp11_lion_convshift
    uv run scripts/run_registry.py --tag mechanism --gpu 1

    # Dry-run: print what would be executed without running
    uv run scripts/run_registry.py --all --dry-run

    # Distribute across multiple GPUs (round-robin)
    uv run scripts/run_registry.py --all --gpus 0,1 --parallel

Output policy:
  * Each run writes to outputs/{id}_seed{seed}/
  * Successful runs produce results.json; failed runs do not (so the
    reporter can distinguish).
  * A master _index.csv can be regenerated with
    `python -m src.reporting.collect` after any set of runs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False


@dataclass
class Experiment:
    id: str
    backbone: str
    seed: int
    epochs: int
    tags: list[str]
    shortlist: bool = False

    @classmethod
    def from_yaml(cls, d: dict) -> "Experiment":
        return cls(
            id=d["id"],
            backbone=d["backbone"],
            seed=int(d.get("seed", 42)),
            epochs=int(d.get("epochs", 80)),
            tags=list(d.get("tags", [])),
            shortlist=bool(d.get("shortlist", False)),
        )


def load_registry(path: str | Path) -> list[Experiment]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return [Experiment.from_yaml(d) for d in data["experiments"]]


def filter_experiments(
    experiments: Sequence[Experiment],
    *,
    ids: Iterable[str] | None = None,
    tag: str | None = None,
    shortlist_only: bool = False,
) -> list[Experiment]:
    selected = list(experiments)
    if ids:
        wanted = set(ids)
        selected = [e for e in selected if e.id in wanted]
    if tag:
        selected = [e for e in selected if tag in e.tags]
    if shortlist_only:
        selected = [e for e in selected if e.shortlist]
    return selected


def run_id_for(exp: Experiment, seed: int) -> str:
    """Output directory name for a given (experiment, seed) pair."""
    return f"{exp.id}_seed{seed}"


def build_command(
    exp: Experiment,
    seed: int,
    gpu: int,
    output_root: Path,
    resume: bool,
) -> list[str]:
    out_dir = output_root / run_id_for(exp, seed)
    cmd = [
        sys.executable, "scripts/run_experiment.py",
        "--config", "configs/default.yaml",
        "--backbone", exp.backbone,
        "--epochs", str(exp.epochs),
        "--seed", str(seed),
        "--output-dir", str(out_dir),
        "--gpu", str(gpu),
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def launch(cmd: list[str], log_file: Path) -> subprocess.Popen:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_file, "a")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)


def print_plan(experiments: list[Experiment], seeds: list[int]) -> None:
    rows = []
    for e in experiments:
        for s in seeds:
            rows.append((run_id_for(e, s), e.backbone, ",".join(e.tags), s, e.epochs))
    if _RICH:
        console = Console()
        table = Table(title=f"{len(rows)} run(s) planned", show_lines=False)
        table.add_column("run_id")
        table.add_column("backbone")
        table.add_column("tags")
        table.add_column("seed", justify="right")
        table.add_column("epochs", justify="right")
        for r in rows:
            table.add_row(r[0], r[1], r[2], str(r[3]), str(r[4]))
        console.print(table)
    else:
        print(f"{len(rows)} run(s) planned:")
        for r in rows:
            print(f"  {r[0]:40s} {r[1]:24s} [{r[2]}] seed={r[3]} epochs={r[4]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments.yaml",
                        help="Path to the experiment registry YAML.")
    parser.add_argument("--output-root", default="outputs",
                        help="Base directory for run outputs.")
    parser.add_argument("--all", action="store_true",
                        help="Run every experiment in the registry.")
    parser.add_argument("--ids", default=None,
                        help="Comma-separated run ids to select.")
    parser.add_argument("--tag", default=None,
                        help="Select only experiments with this tag.")
    parser.add_argument("--shortlist", action="store_true",
                        help="Only run experiments marked shortlist=true.")
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated list of seeds (default: 42).")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Single GPU index for sequential execution.")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU indices for parallel execution.")
    parser.add_argument("--parallel", action="store_true",
                        help="Launch runs in parallel across --gpus. Requires --gpus.")
    parser.add_argument("--resume", action="store_true",
                        help="Pass --resume to each run (resume from last_model.pt).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    args = parser.parse_args()

    experiments = load_registry(args.config)

    ids = [x.strip() for x in args.ids.split(",")] if args.ids else None
    selected = filter_experiments(
        experiments, ids=ids, tag=args.tag, shortlist_only=args.shortlist,
    )
    if args.all and not (ids or args.tag or args.shortlist):
        pass  # no filter applied → all
    elif not (ids or args.tag or args.shortlist):
        parser.error("Pass one of --all / --ids / --tag / --shortlist.")

    if not selected:
        print("No experiments match the filter.")
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    output_root = Path(args.output_root)
    print_plan(selected, seeds)

    if args.dry_run:
        print("\n[dry-run] no commands executed")
        return

    # ── Execution ─────────────────────────────────────────────────────────
    if args.parallel and args.gpus:
        gpu_list = [int(g) for g in args.gpus.split(",")]
        _run_parallel(selected, seeds, gpu_list, output_root, args.resume)
    else:
        gpu_list = [int(g) for g in args.gpus.split(",")] if args.gpus else [args.gpu]
        _run_sequential(selected, seeds, gpu_list, output_root, args.resume)


def _run_sequential(
    experiments: list[Experiment],
    seeds: list[int],
    gpu_list: list[int],
    output_root: Path,
    resume: bool,
) -> None:
    """Run each (experiment, seed) pair one after another."""
    gpu = gpu_list[0]
    for exp in experiments:
        for seed in seeds:
            run_id = run_id_for(exp, seed)
            out_dir = output_root / run_id
            log_file = out_dir / "registry.log"
            cmd = build_command(exp, seed, gpu, output_root, resume)

            print(f"\n[registry] → {run_id} (gpu={gpu})")
            print(f"           {' '.join(cmd)}")
            t0 = time.time()
            p = launch(cmd, log_file)
            rc = p.wait()
            dt = time.time() - t0

            status = "ok" if rc == 0 else f"FAIL (rc={rc})"
            results_json = out_dir / "results.json"
            if results_json.exists():
                print(f"[registry] ← {run_id}: {status} in {dt:.0f}s — results.json present")
            else:
                print(f"[registry] ← {run_id}: {status} in {dt:.0f}s — NO results.json")


def _run_parallel(
    experiments: list[Experiment],
    seeds: list[int],
    gpu_list: list[int],
    output_root: Path,
    resume: bool,
) -> None:
    """Distribute (experiment, seed) pairs across GPUs, one run per GPU.

    At most len(gpu_list) runs execute concurrently; when one finishes the
    next queued run launches on its GPU.
    """
    queue: list[tuple[Experiment, int]] = [
        (e, s) for e in experiments for s in seeds
    ]
    in_flight: dict[int, tuple[subprocess.Popen, str, float]] = {}  # gpu → (proc, run_id, t0)
    free_gpus = list(gpu_list)

    def _launch_next() -> None:
        if not queue or not free_gpus:
            return
        gpu = free_gpus.pop(0)
        exp, seed = queue.pop(0)
        run_id = run_id_for(exp, seed)
        out_dir = output_root / run_id
        log_file = out_dir / "registry.log"
        cmd = build_command(exp, seed, gpu, output_root, resume)
        print(f"[registry] → {run_id} (gpu={gpu})")
        p = launch(cmd, log_file)
        in_flight[gpu] = (p, run_id, time.time())

    # Prime the pipeline
    while queue and free_gpus:
        _launch_next()

    while in_flight:
        time.sleep(5)
        done: list[int] = []
        for gpu, (p, run_id, t0) in in_flight.items():
            if p.poll() is not None:
                rc = p.returncode
                dt = time.time() - t0
                status = "ok" if rc == 0 else f"FAIL (rc={rc})"
                print(f"[registry] ← {run_id}: {status} in {dt:.0f}s")
                done.append(gpu)
        for gpu in done:
            del in_flight[gpu]
            free_gpus.append(gpu)
            _launch_next()


if __name__ == "__main__":
    main()
