"""Streaming-inference state size vs audio duration plot.

Input : outputs/_streaming_memory.csv (produced by
        `scripts/measure_streaming_memory.py`)
Output: outputs/_plots/streaming_memory_vs_duration.png

Thesis narrative: Mamba and RWKV-6 carry state is constant regardless of
audio length (few hundred KB). A causal Transformer's KV cache grows
linearly with the number of frames streamed so far, reaching several MB
at tens of seconds and gigabytes at long durations. This plot is the
single strongest practical-relevance figure in the thesis.

Usage:
    uv run python -m src.reporting.plots.streaming_memory
    # optional explicit paths:
    uv run python -m src.reporting.plots.streaming_memory \\
        --input outputs/_streaming_memory.csv \\
        --output outputs/_plots/streaming_memory_vs_duration.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# Display labels and colors for each Group A backbone.
BACKBONE_STYLE = {
    "mamba":              {"label": "Mamba (ours)",         "color": "tab:blue"},
    "rwkv6":              {"label": "RWKV-6 recurrent",     "color": "tab:green"},
    "transformer_causal": {"label": "Causal Transformer",   "color": "tab:red"},
}


def _fmt_bytes(n: float, _pos) -> str:
    """Human-readable byte formatter for the y-axis."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def render(input_csv: Path, output_png: Path) -> None:
    df = pd.read_csv(input_csv)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    # Use the maximum chunk length per backbone as a proxy for "stream at
    # full resolution"; x-axis is total duration streamed so far.
    # For a constant-state backbone, the curve is flat regardless of chunk
    # size. For a growing-state backbone, each chunk size gives one point
    # at duration=30s; we instead walk cumulative duration to show growth.
    # Simplest: plot state_bytes vs duration_sec grouped by chunk_sec=max.
    # That gives one point per backbone. For the growth story we also
    # emit a "theoretical" linear line for causal Transformer by computing
    # bytes-per-frame from the measured point.

    plt.rcParams.update({
        "figure.figsize": (7.5, 4.5),
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "savefig.dpi": 130,
    })
    fig, ax = plt.subplots()

    for backbone, sub in df.groupby("backbone"):
        sub_sorted = sub.sort_values("duration_sec")
        style = BACKBONE_STYLE.get(backbone, {"label": backbone, "color": "tab:gray"})
        ax.plot(
            sub_sorted["duration_sec"], sub_sorted["state_bytes"],
            marker="o", label=style["label"], color=style["color"], linewidth=2,
        )

        # For the causal Transformer, project a theoretical growth line
        # from the smallest-chunk data point out to a longer duration.
        if backbone == "transformer_causal" and len(sub_sorted) >= 1:
            point = sub_sorted.iloc[0]
            bytes_per_sec = point["state_bytes"] / max(point["duration_sec"], 1e-6)
            proj_x = [1, 5, 10, 30, 60, 300, 1800]
            proj_y = [bytes_per_sec * d for d in proj_x]
            ax.plot(
                proj_x, proj_y, "--",
                color=style["color"], alpha=0.5, linewidth=1.2,
                label=f"{style['label']} (projected, linear growth)",
            )

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Audio duration streamed (s)")
    ax.set_ylabel("Encoder state size (bytes, log scale)")
    ax.set_title("Streaming-inference state size vs audio duration")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_bytes))
    ax.legend(loc="upper left", framealpha=0.9)

    fig.savefig(output_png)
    plt.close(fig)
    print(f"Saved → {output_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/_streaming_memory.csv")
    parser.add_argument("--output", default="outputs/_plots/streaming_memory_vs_duration.png")
    args = parser.parse_args()
    render(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
