"""F10 CVD trained per-head temperature heatmaps (7M + 14M).

Per cell where CVD (Chunked Value Decorrelation) is deployed, render
a heatmap of the trained tau_h = softplus(tau_raw) values; rows =
layers, columns = heads. Heatmap colour encodes magnitude of the
trained temperature: pale near zero (preconditioner near-identity),
darker for engaged values.

Panels are organised as a 4 x 4 GridSpec at width_ratios = [1, 2, 1, 1]
so the 8-head Mamba-2 panels are visually twice as wide as the 4-head
RWKV-6 / LA panels (uniform per-cell aspect). Empty cells in rows 1-2
column 4 are hidden because LA causal does not have a LION-S
counterpart distinct from the LA causal cell already shown.

    Row 1 (causal-7M LS):  RWKV-6  | Mamba-2 | LA         | (n/a)
    Row 2 (causal-14M LS): RWKV-6  | Mamba-2 | LA         | (n/a)
    Row 3 (LION-7M LS):    RWKV-6     Mamba-2   LA          LA
                           (LION-S)  (LION-S)  (LION-LIT)  (LION-S)
    Row 4 (causal-7M CV):  RWKV-6  | Mamba-2 | LA         | (n/a)

The LA LION-LIT / LA LION-S contrast sits in row 3 cols 3-4 so the
reader can compare trained tau on the harmful-inversion substrate
vs the decay-recovered one.

Caption-text record (script docstring; chapter author copies into
LaTeX):

  Some LA causal heads operate at trained tau below the at-init
  value of softplus(0) ≈ 0.693, indicating that the preconditioner
  on those heads softens (rather than sharpens) the value
  distribution. Linear Attention's lack of native decay leaves the
  LUCID-style sharpening guarantee unsupported on those heads,
  which is consistent with the smaller helpful Δ observed on
  LA × CVD relative to the same mechanism on RWKV-6 and Mamba-2.

  The LA LION-LIT and LA LION-S panels show a similar trained tau
  range (means 1.06 and 1.34 respectively, both above init). The
  mechanism is engaged in both cases at the parameter level, but
  the test-CER outcome differs in sign: harmful Δ +0.024 on
  LION-LIT, helpful Δ -0.007 on LION-S. This is direct
  parameter-level evidence for the decay-as-prerequisite finding:
  engagement does not by itself determine whether CVD converts
  into measured gain; the substrate's decay configuration does.

Outputs:
  * F10_cvd_tau_heatmaps.{pdf,png}
  * F10_cvd_tau_heatmaps_data.csv     per-cell (layer, head, tau)
  * F10_cvd_tau_heatmaps_script.py    this file
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

REPO = Path("/tmp/Master_Thesis_Exps")
FIG_DIR = REPO / "Master_Thesis" / "figures" / "chapter5"
STEM = "F10_cvd_tau_heatmaps"
SOURCE_DIR = REPO / "experiments" / "final" / "outputs"

sys.path.insert(0, str(FIG_DIR))
from _style import (  # noqa: E402
    ARCH_COLOR,
    PAGE_WIDTH_IN,
    apply_typography,
    clean_spines,
)
from _extract_mechanism_params import extract_for_cell  # noqa: E402

# Cells to render. Each entry: (panel_title, cell_dir).
# Layout is 3 rows x 4 cols. Empty slots (rows 1-2 col 4) get None.
PANEL_LAYOUT: list[list[tuple[str, str | None]]] = [
    [  # Row 1: causal, 7M, LibriSpeech
        ("RWKV-6 (causal, 7M)",   "7m_rwkv6_causal_lucid_chunked_seed42"),
        ("Mamba-2 (causal, 7M)",  "7m_mamba2_causal_lucid_c_seed42"),
        ("LA (causal, 7M)",       "7m_linear_attn_causal_lucid_seed42"),
        ("",                       None),
    ],
    [  # Row 2: causal, 14M, LibriSpeech
        ("RWKV-6 (causal, 14M)",  "14m_rwkv6_causal_lucid_chunked_seed42"),
        ("Mamba-2 (causal, 14M)", "14m_mamba2_causal_lucid_c_seed42"),
        ("LA (causal, 14M)",      "14m_linear_attn_causal_lucid_seed42"),
        ("",                       None),
    ],
    [  # Row 3: LION, 7M, LibriSpeech
        ("RWKV-6 (LION-S, 7M)",   "7m_rwkv6_lion_lucid_chunked_seed42"),
        ("Mamba-2 (LION-S, 7M)",  "7m_mamba2_lion_lucid_c_seed42"),
        ("LA (LION-LIT, 7M)",     "7m_linear_attn_lion_lucid_seed42"),
        ("LA (LION-S, 7M)",       "7m_linear_attn_lion_s_lucid_seed42"),
    ],
    [  # Row 4: causal, 7M, Common Voice
        ("RWKV-6 (causal, 7M)",   "cv_pilot_rwkv6_lucid_chunked_seed42"),
        ("Mamba-2 (causal, 7M)",  "cv_pilot_mamba2_lucid_c_seed42"),
        ("LA (causal, 7M)",       "cv_pilot_linear_attn_lucid_seed42"),
        ("",                       None),
    ],
]

# Width ratios mirror the head-count disparity (Mamba-2 = 8 heads,
# RWKV-6 / LA = 4 heads). The rightmost column is narrow because
# only row 3 uses it.
PANEL_WIDTH_RATIOS = [1.0, 2.0, 1.0, 1.0]


# Sequential cerulean cmap aligned with the locked palette: zero =
# white (preconditioner near-identity), high = locked cerulean
# (`#277da1`). The softplus(0) = 0.693 init sits in the upper-pale
# range; trained engagement pushes the cells darker.
TAU_CMAP = LinearSegmentedColormap.from_list(
    "thesis_tau",
    ["#FFFFFF", "#aac8d9", "#277da1"],
    N=256,
)


def collect_all() -> list[list[dict | None]]:
    grid: list[list[dict | None]] = []
    for row in PANEL_LAYOUT:
        out_row: list[dict | None] = []
        for title, cell in row:
            if cell is None:
                out_row.append(None)
                continue
            cell_path = SOURCE_DIR / cell
            extracted = extract_for_cell(cell_path)
            if extracted is None or "tau" not in extracted:
                out_row.append({"title": title, "cell": cell, "tau": None})
            else:
                out_row.append({
                    "title": title, "cell": cell,
                    "tau": extracted["tau"],
                    "tau_raw": extracted.get("tau_raw"),
                })
        grid.append(out_row)
    return grid


def render(grid: list[list[dict | None]], out_pdf: Path, out_png: Path) -> None:
    # Compute a single global colour scale so per-cell heatmaps are
    # directly comparable.
    all_taus = []
    for row in grid:
        for entry in row:
            if entry is None or entry.get("tau") is None:
                continue
            all_taus.append(entry["tau"].flatten())
    if not all_taus:
        fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 3.0), constrained_layout=True)
        ax.text(0.5, 0.5, "no CVD cells available", ha="center", va="center")
        fig.savefig(out_pdf)
        fig.savefig(out_png)
        plt.close(fig)
        return
    pooled = np.concatenate(all_taus)
    vmax = float(np.max(pooled)) * 1.02
    vmin = 0.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    nrows, ncols = len(grid), len(grid[0])
    fig = plt.figure(
        figsize=(7.0, 0.9 + 1.6 * nrows),
        constrained_layout=True,
    )
    # Reserve right-margin space for LibriSpeech / Common Voice
    # dataset labels and brackets (mirror F8 / F9 / F11). The panel
    # grid stops at x=0.94, leaving ~0.06 on the right.
    try:
        fig.get_layout_engine().set(rect=(0.0, 0.0, 0.94, 1.0))
    except (AttributeError, TypeError):
        pass
    gs = fig.add_gridspec(
        nrows=nrows, ncols=ncols,
        width_ratios=PANEL_WIDTH_RATIOS,
    )
    axes: list[list] = []
    for r in range(nrows):
        row_axes = []
        for c in range(ncols):
            entry = grid[r][c] if r < len(grid) and c < len(grid[r]) else None
            row_axes.append(fig.add_subplot(gs[r, c]))
        axes.append(row_axes)
    # Inset axis for the colourbar: spans rows 0-1, col 3 (the empty
    # block above the LA LION-S 7M panel). The two axes that
    # nominally sit there (rows 0 col 3 and row 1 col 3) are
    # immediately turned off below in the per-entry loop because
    # `grid[0][3]` and `grid[1][3]` are None; we then drop them and
    # add a new axis spanning that 2-row block for the colourbar.
    axes[0][3].remove()
    axes[1][3].remove()
    # The colourbar is positioned manually after layout finalises;
    # see the explicit `fig.add_axes(...)` block at the end of this
    # function. subgridspec-based positioning would align to GridSpec
    # cell borders (which include title / tick padding), not to the
    # heatmap data box itself.

    for r in range(nrows):
        for c in range(ncols):
            if r in (0, 1) and c == 3:
                # This block holds the colourbar; skip the per-cell
                # heatmap rendering for it.
                continue
            ax = axes[r][c]
            entry = grid[r][c]
            if entry is None:
                ax.axis("off")
                continue
            title = entry["title"]
            tau = entry.get("tau")
            if tau is None:
                ax.axis("off")
                ax.set_title(title + "\n(unavailable)", fontsize=7)
                continue
            n_layers, n_heads = tau.shape
            ax.imshow(tau, cmap=TAU_CMAP, norm=norm, aspect="auto")
            ax.set_xticks(range(n_heads))
            ax.set_xticklabels([f"h{h}" for h in range(n_heads)], fontsize=6)
            ax.set_yticks(range(n_layers))
            ax.set_yticklabels([f"L{L}" for L in range(n_layers)], fontsize=6)
            ax.tick_params(axis="both", length=0)
            ax.set_title(title, fontsize=7.5)
            ax.grid(False)
            # Annotate each cell with the tau value if the heatmap is
            # small enough to fit without overlap.
            if n_heads <= 8:
                for L in range(n_layers):
                    for h in range(n_heads):
                        v = float(tau[L, h])
                        text_color = "white" if v > vmax * 0.55 else "#212529"
                        ax.text(h, L, f"{v:.2f}", ha="center", va="center",
                                fontsize=5.5, color=text_color)

    fig.suptitle("CVD trained per-head temperature")

    # Shared colour bar aligned to the actual heatmap boxes of
    # LA causal 7M and LA causal 14M, not to GridSpec cell borders
    # (which include title / tick padding). Force layout to settle
    # before reading axes positions.
    fig.canvas.draw()
    top_bbox = axes[0][2].get_position()      # LA (causal, 7M)
    bottom_bbox = axes[1][2].get_position()   # LA (causal, 14M)
    col4_bbox = axes[2][3].get_position()     # LA LION-S 7M (col 4)

    cbar_y0 = bottom_bbox.y0
    cbar_y1 = top_bbox.y1
    cbar_h = cbar_y1 - cbar_y0
    cbar_w = col4_bbox.width * 0.18
    cbar_x = col4_bbox.x0 + (col4_bbox.width - cbar_w) / 2.0

    cbar_ax = fig.add_axes([cbar_x, cbar_y0, cbar_w, cbar_h])
    sm = plt.cm.ScalarMappable(cmap=TAU_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(
        r"trained $\tau_h = \mathrm{softplus}(\tau_{\mathrm{raw}, h})$",
        fontsize=7,
    )
    cbar.ax.tick_params(labelsize=6)

    # Dataset labels in the right-margin space reserved by
    # constrained-layout's `rect`. LibriSpeech spans rows 1-3
    # (causal-7M, causal-14M, LION-7M); Common Voice on row 4.
    libr_top = axes[0][0].get_position().y1
    libr_bottom = axes[2][0].get_position().y0
    cv_top = axes[3][0].get_position().y1
    cv_bottom = axes[3][0].get_position().y0
    label_x = 0.965
    fig.text(
        label_x, (libr_top + libr_bottom) / 2.0, "LibriSpeech",
        rotation=270, ha="center", va="center",
        fontsize=8.5, color="#495057", weight="bold",
    )
    fig.text(
        label_x, (cv_top + cv_bottom) / 2.0, "Common Voice",
        rotation=270, ha="center", va="center",
        fontsize=8.5, color="#495057", weight="bold",
    )
    bracket_x = label_x - 0.013
    hook_len = 0.006
    bracket_lw = 0.7
    from matplotlib.lines import Line2D

    def _draw_bracket(y_top: float, y_bottom: float) -> None:
        fig.add_artist(Line2D(
            [bracket_x, bracket_x], [y_bottom, y_top],
            color="#495057", linewidth=bracket_lw,
        ))
        fig.add_artist(Line2D(
            [bracket_x - hook_len, bracket_x], [y_top, y_top],
            color="#495057", linewidth=bracket_lw,
        ))
        fig.add_artist(Line2D(
            [bracket_x - hook_len, bracket_x], [y_bottom, y_bottom],
            color="#495057", linewidth=bracket_lw,
        ))

    _draw_bracket(libr_top, libr_bottom)
    _draw_bracket(cv_top, cv_bottom)

    # Plain savefig (no bbox_inches="tight") so the manually-placed
    # colourbar position is preserved exactly as computed.
    fig.savefig(out_pdf, bbox_inches=None)
    fig.savefig(out_png, bbox_inches=None, dpi=300)
    plt.close(fig)


def write_csv(grid: list[list[dict | None]], path: Path) -> None:
    rows: list[dict] = []
    for r, panel_row in enumerate(grid):
        for c, entry in enumerate(panel_row):
            if entry is None or entry.get("tau") is None:
                continue
            tau = entry["tau"]
            n_layers, n_heads = tau.shape
            for L in range(n_layers):
                for h in range(n_heads):
                    rows.append({
                        "panel_row": r, "panel_col": c,
                        "panel_title": entry["title"], "cell": entry["cell"],
                        "layer": L, "head": h,
                        "tau": float(tau[L, h]),
                        "tau_raw": float(entry["tau_raw"][L, h])
                                    if entry.get("tau_raw") is not None else float("nan"),
                    })
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_typography()
    grid = collect_all()
    print("[F10] cells loaded:")
    for r, row in enumerate(grid):
        for c, entry in enumerate(row):
            if entry is None:
                print(f"  ({r},{c}) (empty slot)")
                continue
            tau = entry.get("tau")
            if tau is None:
                print(f"  ({r},{c}) {entry['title']}: UNAVAILABLE")
            else:
                print(f"  ({r},{c}) {entry['title']}: tau shape {tau.shape}, "
                       f"min={tau.min():.3f} max={tau.max():.3f} mean={tau.mean():.3f}")
    write_csv(grid, FIG_DIR / f"{STEM}_data.csv")
    render(grid, FIG_DIR / f"{STEM}.pdf", FIG_DIR / f"{STEM}.png")


if __name__ == "__main__":
    main()
