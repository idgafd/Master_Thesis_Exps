"""Cross-run comparison plots.

Reads:
    outputs/_index.csv               — one row per run
    outputs/{run_id}/history.csv     — per-epoch metrics per run

Writes to `outputs/_plots/`:
    convergence_groupA.png           — dev CER vs epoch, all Group A runs
    convergence_groupB.png           — dev CER vs epoch, all Group B runs
    cer_vs_params.png                — test CER vs parameter count scatter
    chunked_cer_vs_chunk_size.png    — reset-mode CER as a function of chunk size
    training_time_bar.png            — avg epoch time, grouped by backbone

All charts skip rows that haven't completed (missing metrics). Designed
to be safe to rerun at any point during the experiment campaign — even
with a single run in _index.csv the plots still render meaningfully.

Usage:
    uv run python -m src.reporting.plots.cross_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PLOT_STYLE = {
    "figure.figsize": (8, 5),
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
    "savefig.dpi": 130,
}


# Group-consistent colors so the same backbone keeps the same color
# across every plot.
BACKBONE_COLORS = {
    # Group A
    "mamba":              "tab:blue",
    "mamba_cuda":         "tab:cyan",
    "rwkv6":              "tab:green",
    "rwkv6_lucid":        "lightgreen",
    "rwkv6_delta":        "darkgreen",
    "rwkv6_lucid_delta":  "forestgreen",
    "transformer_causal": "tab:red",
    # Group B
    "transformer":              "tab:orange",
    "lion":                     "tab:purple",
    "lion_convshift":           "mediumpurple",
    "lion_lucid":               "plum",
    "lion_lucid_chunked":       "orchid",
    "lion_delta":               "darkorchid",
    "lion_headscale":           "violet",
    "lion_convshift_headscale": "indigo",
    "mamba_bidir":              "tab:brown",
}


def _color(backbone: str) -> str:
    return BACKBONE_COLORS.get(backbone, "tab:gray")


def _load_history(output_root: Path, run_id: str) -> pd.DataFrame | None:
    path = output_root / run_id / "history.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ── plots ──────────────────────────────────────────────────────────────────

def convergence_curves(index: pd.DataFrame, output_root: Path, plots_dir: Path) -> None:
    for group_tag, fname in [("groupA", "convergence_groupA.png"),
                              ("groupB", "convergence_groupB.png")]:
        sub = index[index["tags"].fillna("").str.contains(group_tag, case=False)]
        if sub.empty:
            continue

        fig, ax = plt.subplots()
        plotted = 0
        for _, r in sub.iterrows():
            hist = _load_history(output_root, r["run_id"])
            if hist is None or "dev_cer" not in hist.columns:
                continue
            ax.plot(
                hist["epoch"], hist["dev_cer"],
                marker="o", markersize=3, linewidth=1.5,
                label=r["backbone"], color=_color(r["backbone"]),
            )
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        ax.set_xlabel("epoch")
        ax.set_ylabel("Dev CER")
        ax.set_title(f"{group_tag} convergence (Dev CER vs epoch)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        fig.savefig(plots_dir / fname)
        plt.close(fig)


def cer_vs_params(index: pd.DataFrame, plots_dir: Path) -> None:
    sub = index.dropna(subset=["params_total", "test_cer"])
    if sub.empty:
        return
    fig, ax = plt.subplots()
    for _, r in sub.iterrows():
        ax.scatter(
            r["params_total"] / 1e6, r["test_cer"],
            color=_color(r["backbone"]), s=80,
            edgecolors="black", linewidths=0.5,
        )
        ax.annotate(
            r["backbone"], (r["params_total"] / 1e6, r["test_cer"]),
            xytext=(5, 5), textcoords="offset points", fontsize=8,
        )
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Test CER")
    ax.set_title("Test CER vs parameter count")
    fig.savefig(plots_dir / "cer_vs_params.png")
    plt.close(fig)


def chunked_cer_vs_chunk_size(index: pd.DataFrame, plots_dir: Path) -> None:
    chunk_cols = [
        ("chunked_2s_reset_cer", 2.0),
        ("chunked_5s_reset_cer", 5.0),
        ("chunked_10s_reset_cer", 10.0),
    ]
    cols_present = [c for c, _ in chunk_cols]
    sub = index.dropna(subset=["test_cer"]).copy()
    if sub.empty or not any(c in sub.columns for c in cols_present):
        return

    fig, ax = plt.subplots()
    plotted = 0
    for _, r in sub.iterrows():
        xs, ys = [], []
        if pd.notna(r.get("test_cer")):
            xs.append(1000.0)  # sentinel for "full utterance" (right-most)
            ys.append(r["test_cer"])
        for col, chunk_sec in chunk_cols:
            if col in sub.columns and pd.notna(r.get(col)):
                xs.append(chunk_sec)
                ys.append(r[col])
        if len(ys) < 2:
            continue
        pairs = sorted(zip(xs, ys))
        xs_sorted, ys_sorted = zip(*pairs)
        ax.plot(
            xs_sorted, ys_sorted,
            marker="o", linewidth=1.5,
            label=r["backbone"], color=_color(r["backbone"]),
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_xlabel("Chunk length (s) — right-most point is full utterance")
    ax.set_ylabel("Dev CER")
    ax.set_title("Chunked reset CER vs chunk length")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.savefig(plots_dir / "chunked_cer_vs_chunk_size.png")
    plt.close(fig)


def training_time_bar(index: pd.DataFrame, plots_dir: Path) -> None:
    sub = index.dropna(subset=["avg_epoch_sec", "backbone"])
    if sub.empty:
        return
    agg = sub.groupby("backbone")["avg_epoch_sec"].mean().sort_values()
    if agg.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [_color(b) for b in agg.index]
    ax.bar(range(len(agg)), agg.values, color=colors,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Average epoch time (s)")
    ax.set_title("Training time per epoch, by backbone")
    fig.tight_layout()
    fig.savefig(plots_dir / "training_time_bar.png")
    plt.close(fig)


def render_all(index_csv: Path, output_root: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not index_csv.exists():
        raise SystemExit(
            f"{index_csv} not found. Run `python -m src.reporting.collect` first."
        )
    df = pd.read_csv(index_csv)

    plt.rcParams.update(PLOT_STYLE)
    convergence_curves(df, output_root, plots_dir)
    cer_vs_params(df, plots_dir)
    chunked_cer_vs_chunk_size(df, plots_dir)
    training_time_bar(df, plots_dir)
    print(f"Regenerated cross-run plots → {plots_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="outputs/_index.csv")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--plots-dir", default="outputs/_plots")
    args = parser.parse_args()
    render_all(Path(args.index), Path(args.output_root), Path(args.plots_dir))


if __name__ == "__main__":
    main()
