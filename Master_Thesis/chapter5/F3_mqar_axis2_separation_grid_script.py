"""F3 MQAR axis-2 separation grid.

Canonical 10-row PASS/FAIL/PARTIAL/SKIP/OOM grid for the MQAR
length sweep at T in {64, 256, 1024}. Source: the cohort index at
`experiments/synthetics_v1/outputs/cohort_reduced/_index.csv` on the
`main` branch (read-only via `git show`).

Headline: at T=1024, the causal Transformer FAILs while every
linear-time backbone with either Multi-Scale Depthwise Convolution
or Chunked Value Decorrelation PASSes - the cleanest single
empirical demonstration of axis-2 mechanism transfer in the matrix.

Caption note: the Damped Harmonic Oscillator (DHO) row is NOT in
the cohort by design. DHO targets axis 1 (short-range temporal
hierarchy / damped-oscillator dynamics); MQAR exercises axis 2
(associative memory under interference). Excluding DHO from the
axis-2 cohort is itself an empirical validation of the axis
decomposition: an axis-1 mechanism is not expected to lift an
axis-2 deficit.

Outputs:
  * F3_mqar_axis2_separation_grid.{pdf,png}    main-text grid
  * F3_mqar_axis2_separation_grid_data.csv     filtered cohort rows
  * F3_mqar_axis2_separation_grid_script.py    this file
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO = Path("/tmp/Master_Thesis_Exps")
FIG_DIR = REPO / "Master_Thesis" / "figures" / "chapter5"
STEM = "F3_mqar_axis2_separation_grid"
SOURCE_REF = "origin/main"
COHORT_PATH = "experiments/synthetics_v1/outputs/cohort_reduced/_index.csv"

sys.path.insert(0, str(FIG_DIR))
from _style import (  # noqa: E402
    ARCH_COLOR,
    PAGE_WIDTH_IN,
    apply_typography,
    clean_spines,
)

# Canonical row order (10 rows). Each tuple is
#   (label_for_axis, csv_backbone_name, mechanism_class).
ROWS = [
    ("Transformer (causal)",            "transformer_causal",                          "vanilla"),
    ("RWKV-6",                          "rwkv6",                                       "vanilla"),
    ("RWKV-6 + CVD",                    "rwkv6_lucid",                                 "cvd"),
    ("RWKV-6 + MSDC",                   "rwkv6_convshift_multidil_symmetric_v2",       "msdc"),
    ("Mamba-2",                         "mamba2",                                      "vanilla"),
    ("Mamba-2 + CVD",                   "mamba2_lucid_c",                              "cvd"),
    ("Mamba-2 + MSDC",                  "mamba2_convshift_multidil_symmetric_v2",      "msdc"),
    ("Linear Attention",                "linear_attn_causal",                          "vanilla"),
    ("Linear Attention + CVD",          "linear_attn_lucid",                           "cvd"),
    ("Linear Attention + MSDC",         "linear_attn_convshift_multidil_symmetric_v2", "msdc"),
]

T_VALUES = [64, 256, 1024]

# Verdict palette. PASS uses the locked architecture-cerulean colour
# (helpful, blue); FAIL uses the diverging-cmap red endpoint
# (harmful, red); PARTIAL uses a saffron amber for "near pass";
# SKIP / OOM are hatched grey.
VERDICT_FACE = {
    "PASS":    ARCH_COLOR["rwkv6"],
    "FAIL":    "#f94144",
    "PARTIAL": "#f9c74f",
    "SKIP":    "#dee2e6",
    "OOM":     "#dee2e6",
}
VERDICT_HATCH = {
    "PASS":    None,
    "FAIL":    None,
    "PARTIAL": None,
    "SKIP":    "////",
    "OOM":     "xxxx",
}
VERDICT_TEXT_COLOR = {
    "PASS":    "white",
    "FAIL":    "white",
    "PARTIAL": "#212529",
    "SKIP":    "#495057",
    "OOM":     "#495057",
}


def git_show(path: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{SOURCE_REF}:{path}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout


def load_cohort() -> pd.DataFrame:
    raw = git_show(COHORT_PATH)
    df = pd.read_csv(io.StringIO(raw))
    # Normalise PARTIAL / SKIP / OOM verdict strings.
    df["verdict"] = df["verdict"].fillna("").str.strip().str.upper()
    return df


def build_grid(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, csv_name, mech in ROWS:
        for T in T_VALUES:
            sub = df[(df["backbone"] == csv_name) & (df["T"] == T)]
            if sub.empty:
                rows.append({"label": label, "backbone": csv_name, "mechanism": mech,
                              "T": T, "verdict": "MISSING", "steps_to_0_9": None,
                              "best_per_seq": None})
                continue
            row = sub.iloc[0]
            verdict = row["verdict"] if row["verdict"] else "MISSING"
            steps = row["steps_to_0.9"]
            try:
                steps = int(steps) if str(steps) not in ("-", "—", "", "nan") else None
            except (ValueError, TypeError):
                steps = None
            try:
                best_ps = float(row["best_per_seq"])
            except (ValueError, TypeError):
                best_ps = None
            rows.append({
                "label": label, "backbone": csv_name, "mechanism": mech,
                "T": T, "verdict": verdict, "steps_to_0_9": steps,
                "best_per_seq": best_ps,
            })
    return pd.DataFrame(rows)


def render(grid: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    n_rows = len(ROWS)
    n_cols = len(T_VALUES)
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 4.6), constrained_layout=True)
    clean_spines(ax)
    ax.set_axisbelow(True)
    ax.grid(False)
    # Draw cell rectangles with the appropriate verdict styling.
    for i, (label, csv_name, mech) in enumerate(ROWS):
        # y axis goes top -> bottom in the canonical order so we
        # invert the row index.
        y = n_rows - 1 - i
        for j, T in enumerate(T_VALUES):
            cell = grid[(grid["backbone"] == csv_name) & (grid["T"] == T)]
            if cell.empty:
                continue
            verdict = cell.iloc[0]["verdict"]
            face = VERDICT_FACE.get(verdict, "white")
            hatch = VERDICT_HATCH.get(verdict)
            edge = "white"
            rect = mpatches.Rectangle(
                (j - 0.5, y - 0.5), 1.0, 1.0,
                facecolor=face, edgecolor=edge, hatch=hatch, linewidth=0.8,
            )
            ax.add_patch(rect)
            # Annotate: PASS gets steps_to_0.9 (in thousands, "kS"),
            # FAIL/PARTIAL/SKIP/OOM get the verdict label.
            if verdict == "PASS":
                steps = cell.iloc[0]["steps_to_0_9"]
                if steps is None:
                    txt = "PASS"
                else:
                    txt = f"PASS\n{int(steps) // 1000} kS"
            else:
                txt = verdict
            ax.text(j, y, txt, ha="center", va="center",
                    color=VERDICT_TEXT_COLOR.get(verdict, "#212529"),
                    fontsize=7)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"T = {T}" for T in T_VALUES])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([ROWS[n_rows - 1 - i][0] for i in range(n_rows)])
    ax.tick_params(axis="both", length=0)
    # Light separators between cells.
    for x in range(1, n_cols):
        ax.axvline(x - 0.5, color="white", linewidth=1.0, zorder=3)
    for y in range(1, n_rows):
        ax.axhline(y - 0.5, color="white", linewidth=1.0, zorder=3)
    # Legend.
    legend_handles = [
        mpatches.Patch(facecolor=VERDICT_FACE["PASS"], label="PASS"),
        mpatches.Patch(facecolor=VERDICT_FACE["FAIL"], label="FAIL"),
        mpatches.Patch(facecolor=VERDICT_FACE["PARTIAL"], label="PARTIAL"),
        mpatches.Patch(facecolor=VERDICT_FACE["SKIP"],
                       hatch=VERDICT_HATCH["SKIP"], edgecolor="white",
                       label="SKIP"),
        mpatches.Patch(facecolor=VERDICT_FACE["OOM"],
                       hatch=VERDICT_HATCH["OOM"], edgecolor="white",
                       label="OOM"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=False)
    ax.set_title("MQAR axis-2 separation grid (seed 42)")
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_typography()
    cohort = load_cohort()
    grid = build_grid(cohort)
    print("[F3] grid:")
    print(grid.to_string(index=False))
    grid.to_csv(FIG_DIR / f"{STEM}_data.csv", index=False)
    render(grid, FIG_DIR / f"{STEM}.pdf", FIG_DIR / f"{STEM}.png")
    # Headline verification.
    head = grid[(grid["backbone"] == "transformer_causal") & (grid["T"] == 1024)]
    if not head.empty:
        print(f"[F3] HEADLINE: transformer_causal at T=1024 -> {head.iloc[0]['verdict']}")
    lt_passes_at_1024 = grid[(grid["T"] == 1024) & (grid["verdict"] == "PASS")]
    print(f"[F3] linear-time PASSes at T=1024: {len(lt_passes_at_1024)} / 9")


if __name__ == "__main__":
    main()
