"""Per-run matplotlib plots, rendered at the end of every epoch.

Produces PNG files under `{run_dir}/plots/` which are overwritten in place
so the latest state is always at the same path. These files are committed
to git — a reviewer browsing `outputs/{run_id}/plots/cer_curve.png` on the
GitHub web UI sees the training dynamics immediately.

Intentionally minimal: four plots per run, each self-contained, readable
from history.csv + metrics.jsonl without loading the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import pandas as pd


PLOT_STYLE = {
    "figure.figsize": (7, 4),
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
    "savefig.dpi": 120,
}


def _apply_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def _read_history(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "history.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _read_metrics(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "metrics.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_json(path, lines=True)
    except Exception:
        return None


def render_run_plots(run_dir: str | Path, title_prefix: str = "") -> None:
    """Regenerate all in-run plots. Safe to call at the end of every epoch."""
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    history = _read_history(run_dir)
    metrics = _read_metrics(run_dir)

    _apply_style()

    _plot_loss_curve(history, metrics, plots_dir, title_prefix)
    _plot_cer_curve(history, plots_dir, title_prefix)
    _plot_lr_schedule(metrics, plots_dir, title_prefix)
    _plot_grad_norm(metrics, plots_dir, title_prefix)


# ── individual plots ──────────────────────────────────────────────────────

def _plot_loss_curve(history, metrics, plots_dir: Path, title_prefix: str) -> None:
    fig, ax = plt.subplots()
    plotted = False

    if metrics is not None and "train_loss" in metrics.columns:
        ax.plot(metrics["step"], metrics["train_loss"], alpha=0.35,
                color="steelblue", label="train (per step)")
        plotted = True
    if history is not None and "train_loss" in history.columns:
        ax.plot(_epoch_x(history, metrics), history["train_loss"],
                marker="o", color="darkblue", label="train (epoch avg)")
        plotted = True
    if history is not None and "dev_loss" in history.columns:
        ax.plot(_epoch_x(history, metrics), history["dev_loss"],
                marker="s", color="darkorange", label="dev")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("step" if metrics is not None and "train_loss" in metrics.columns else "epoch")
    ax.set_ylabel("CTC loss")
    ax.set_title(f"{title_prefix}loss curve".strip())
    ax.legend()
    fig.savefig(plots_dir / "loss_curve.png")
    plt.close(fig)


def _plot_cer_curve(history, plots_dir: Path, title_prefix: str) -> None:
    if history is None or "dev_cer" not in history.columns:
        return
    fig, ax = plt.subplots()
    ax.plot(history["epoch"], history["dev_cer"], marker="o",
            color="darkgreen", label="dev CER")
    if "dev_wer" in history.columns:
        ax2 = ax.twinx()
        ax2.plot(history["epoch"], history["dev_wer"], marker="s",
                 color="firebrick", linestyle="--", alpha=0.6, label="dev WER")
        ax2.set_ylabel("dev WER", color="firebrick")
        ax2.tick_params(axis="y", labelcolor="firebrick")
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev CER", color="darkgreen")
    ax.tick_params(axis="y", labelcolor="darkgreen")
    ax.set_title(f"{title_prefix}dev CER / WER".strip())
    fig.savefig(plots_dir / "cer_curve.png")
    plt.close(fig)


def _plot_lr_schedule(metrics, plots_dir: Path, title_prefix: str) -> None:
    if metrics is None or "lr" not in metrics.columns:
        return
    fig, ax = plt.subplots()
    ax.plot(metrics["step"], metrics["lr"], color="purple")
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title(f"{title_prefix}LR schedule".strip())
    fig.savefig(plots_dir / "lr_schedule.png")
    plt.close(fig)


def _plot_grad_norm(metrics, plots_dir: Path, title_prefix: str) -> None:
    if metrics is None:
        return
    cols = [c for c in ("grad_norm_raw", "grad_norm") if c in metrics.columns]
    if not cols:
        return
    fig, ax = plt.subplots()
    for col, color in zip(cols, ["crimson", "navy"]):
        ax.plot(metrics["step"], metrics[col], label=col,
                color=color, alpha=0.7, linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("gradient norm")
    ax.set_yscale("log")
    ax.set_title(f"{title_prefix}gradient norm".strip())
    ax.legend()
    fig.savefig(plots_dir / "grad_norm.png")
    plt.close(fig)


# ── helpers ───────────────────────────────────────────────────────────────

def _epoch_x(history: pd.DataFrame, metrics: pd.DataFrame | None):
    """Return x-axis values matching the loss-curve x-axis.

    If the loss curve is per-step (metrics available), anchor the per-epoch
    points at the step where each epoch ended. Otherwise use plain epoch
    numbers.
    """
    if metrics is None or "step" not in metrics.columns or "epoch" not in metrics.columns:
        return history["epoch"]
    # last step of each epoch
    end_of_epoch = metrics.groupby("epoch")["step"].max()
    return [end_of_epoch.get(e, e) for e in history["epoch"]]
