"""Checkpoint save/load with full training state for resume-on-crash.

A checkpoint is a single `.pt` file containing the model weights, optimizer
and scheduler state, RNG state for all three libraries, and run metadata.
Designed so that `--resume` restarts from the exact pre-crash state.

File layout inside a run directory:
    best_model.pt         — updated whenever dev CER improves
    last_model.pt         — full resume state, overwritten every epoch
    checkpoint_ep{N}.pt   — sparse snapshots at epochs {1, 5, 10, 20, 40, final}

Checkpoint dict schema:
    epoch              : int
    model              : state_dict
    optimizer          : state_dict
    scheduler          : state_dict
    rng_state          : {python, numpy, torch, cuda}
    best_cer           : float
    patience_counter   : int
    config             : dict (snapshot of the resolved config)
    git_sha            : str
"""

from __future__ import annotations

import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


SNAPSHOT_EPOCHS = {1, 5, 10, 20, 40}


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _config_to_dict(cfg: Any) -> dict:
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return cfg
    return {"_repr": repr(cfg)}


def save_checkpoint(
    run_dir: str | Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_cer: float,
    patience_counter: int,
    config: Any,
    git_sha: str,
    is_best: bool,
    total_epochs: int,
) -> None:
    """Save `last_model.pt`, optionally `best_model.pt`, and sparse snapshots.

    The same serialized dict is written to every destination; symbolic links
    are avoided because git + some filesystems handle them poorly.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else {},
        "rng_state": _rng_state(),
        "best_cer": best_cer,
        "patience_counter": patience_counter,
        "config": _config_to_dict(config),
        "git_sha": git_sha,
    }

    # last: resume target
    torch.save(state, run_dir / "last_model.pt")

    # best: the checkpoint we report
    if is_best:
        torch.save(state, run_dir / "best_model.pt")

    # sparse snapshots for convergence analysis
    if epoch in SNAPSHOT_EPOCHS or epoch == total_epochs:
        torch.save(state, run_dir / f"checkpoint_ep{epoch}.pt")


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device | None = None,
    restore_rng: bool = True,
) -> dict:
    """Load a checkpoint and restore model/optimizer/scheduler/rng.

    Returns the full checkpoint dict (caller pulls epoch, best_cer, etc.).
    """
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])

    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])

    if scheduler is not None and state.get("scheduler") and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(state["scheduler"])

    if restore_rng and state.get("rng_state") is not None:
        _restore_rng(state["rng_state"])

    return state


def find_resume_point(run_dir: str | Path) -> Path | None:
    """Return `last_model.pt` if it exists, else None."""
    p = Path(run_dir) / "last_model.pt"
    return p if p.exists() else None


def get_git_sha(default: str = "unknown") -> str:
    """Return the current git SHA, or `default` if we're not in a repo."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return default
