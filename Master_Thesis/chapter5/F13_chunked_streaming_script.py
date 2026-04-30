"""F13 chunked-streaming evaluation (7M causal vanilla).

Per architecture, a small grouped-bar panel showing test CER under
chunked-streaming evaluation at 2.0s / 5.0s / 10.0s windows in both
reset and carry modes, with a horizontal dashed reference line at
the full-utterance test CER. Three panels stacked vertically.

Source: each cell's `results.json::chunked` block. The chunked
block has six entries (three window lengths x reset|carry); the
`test.cer` field carries the full-utterance reference. 7M causal
cells only; the chapter's chunked-streaming reading is a
robustness-to-streaming footnote and does not need cross-mode or
cross-scale comparison.

Reading the figure: the reset modes lose accuracy because each
chunk is processed independently with a fresh RNN state; the
carry modes recover most of the lost accuracy by carrying the
recurrent state across chunk boundaries. The 10s windows are the
closest to the full-utterance reference (most utterances are
shorter than 10s); the 2s windows are the worst.

Outputs:
  * F13_chunked_streaming.{pdf,png}
  * F13_chunked_streaming_data.csv
  * F13_chunked_streaming_script.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/tmp/Master_Thesis_Exps")
FIG_DIR = REPO / "Master_Thesis" / "figures" / "chapter5"
STEM = "F13_chunked_streaming"
SOURCE_DIR = REPO / "experiments" / "final" / "outputs"

sys.path.insert(0, str(FIG_DIR))
from _style import (  # noqa: E402
    ARCH_COLOR,
    PAGE_WIDTH_IN,
    SPINE_COLOR,
    apply_typography,
    clean_spines,
)

ARCH_LABEL = {"rwkv6": "RWKV-6", "mamba2": "Mamba-2", "linear_attn": "LA"}
ARCH_ORDER = ["rwkv6", "mamba2", "linear_attn"]

WINDOW_LENGTHS = [2.0, 5.0, 10.0]
MODES = ["reset", "carry"]


def load_cell(cell_dir: Path) -> dict:
    with (cell_dir / "results.json").open() as f:
        results = json.load(f)
    full_cer = float(results["test"]["cer"])
    chunked = results.get("chunked", {})
    chunked_cer: dict[tuple[float, str], float] = {}
    for win in WINDOW_LENGTHS:
        for mode in MODES:
            key = f"{win}s_{mode}"
            entry = chunked.get(key)
            if entry is None or "cer" not in entry:
                continue
            chunked_cer[(win, mode)] = float(entry["cer"])
    return {"full_cer": full_cer, "chunked": chunked_cer}


def collect() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for arch in ARCH_ORDER:
        cell = SOURCE_DIR / f"7m_{arch}_causal_vanilla_seed42"
        out[arch] = load_cell(cell)
    return out


def render(data: dict[str, dict], out_pdf: Path, out_png: Path) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(PAGE_WIDTH_IN, 5.0),
        sharex=True, constrained_layout=True,
    )
    bar_w = 0.36
    for ax, arch in zip(axes, ARCH_ORDER):
        clean_spines(ax)
        ax.set_axisbelow(True)
        d = data[arch]
        full = d["full_cer"]
        # Horizontal dashed line for full-utterance reference.
        ax.axhline(full, color=SPINE_COLOR, linewidth=0.7, linestyle="--",
                    zorder=1)
        ax.text(
            len(WINDOW_LENGTHS) - 0.5, full,
            f"  full-utt {full:.3f}",
            fontsize=6.5, color=SPINE_COLOR,
            ha="left", va="center",
        )
        for j, win in enumerate(WINDOW_LENGTHS):
            for k, mode in enumerate(MODES):
                cer = d["chunked"].get((win, mode))
                if cer is None:
                    continue
                x = j + (k - 0.5) * bar_w
                hatch = None if mode == "reset" else "////"
                ax.bar(
                    x, cer, width=bar_w,
                    color=ARCH_COLOR[arch],
                    hatch=hatch, edgecolor="white", linewidth=0.5,
                    zorder=2,
                )
                ax.text(
                    x, cer + 0.005, f"{cer:.3f}",
                    fontsize=6, color="#212529",
                    ha="center", va="bottom",
                )
        ax.set_xticks(range(len(WINDOW_LENGTHS)))
        ax.set_xticklabels([f"{int(w)} s" for w in WINDOW_LENGTHS])
        ax.set_ylabel("Test CER")
        ax.set_title(ARCH_LABEL[arch])
        # y-axis: from a touch below full-utt to a touch above the
        # worst chunked CER, so the dashed line is visible.
        ymin = min(full, min(d["chunked"].values())) - 0.01
        ymax = max(d["chunked"].values()) + 0.03
        ax.set_ylim(ymin, ymax)

    fig.suptitle("Chunked-streaming evaluation (7M causal)")

    legend_handles = [
        mpatches.Patch(facecolor="#adb5bd", edgecolor="white", label="reset"),
        mpatches.Patch(facecolor="#adb5bd", edgecolor="white", hatch="////",
                        label="carry"),
        plt.Line2D([0], [0], color=SPINE_COLOR, linestyle="--",
                    linewidth=0.8, label="full-utt"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3, frameon=False, fontsize=7,
    )
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)


def write_csv(data: dict[str, dict], path: Path) -> None:
    rows = []
    for arch, d in data.items():
        rows.append({
            "arch": arch, "window_s": "full", "mode": "full-utt",
            "test_cer": d["full_cer"],
        })
        for (win, mode), cer in d["chunked"].items():
            rows.append({
                "arch": arch, "window_s": win, "mode": mode,
                "test_cer": cer,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_typography()
    data = collect()
    print("[F13] data:")
    for arch, d in data.items():
        print(f"  {arch} full {d['full_cer']:.4f}; chunked: "
              + ", ".join(f"{w}s_{m}={c:.4f}"
                           for (w, m), c in d["chunked"].items()))
    write_csv(data, FIG_DIR / f"{STEM}_data.csv")
    render(data, FIG_DIR / f"{STEM}.pdf", FIG_DIR / f"{STEM}.png")


if __name__ == "__main__":
    main()
