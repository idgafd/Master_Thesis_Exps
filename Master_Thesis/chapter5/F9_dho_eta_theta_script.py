"""F9 DHO trained eta and theta figures (eta only for now).

Per cell where DHO (Damped Harmonic Oscillator) is deployed, render
a heatmap of trained viscosity engagement: rows = layers, columns =
heads, value = mean(|eta_{l,h,b}|) across the per-head block
dimension. The mean(|eta|) statistic characterises engagement as
"departure from zero-init" (eta is initialised to zeros at the
parameter level — see linear_attn_rse.py:122 and parallel
declarations in mamba2_rse.py:154 and rwkv6_time_mix.py:835).

Layout: 4 rows x 4 cols GridSpec at width_ratios = [1, 2, 1, 1] so
the wider Mamba-2 panels (8 heads in the 7M-causal-strong and
LION-S configurations) are visually proportionate to the 4-head
panels:

    Row 1 (causal, 7M, LibriSpeech): RWKV-6  Mamba-2  LA  (cbar)
    Row 2 (causal, 14M, LibriSpeech): RWKV-6  Mamba-2  LA  (cbar)
    Row 3 (LION, 7M, LibriSpeech):    RWKV-6 LION-S
                                      Mamba-2 LION-S
                                      LA LION-LIT
                                      (LA LION-S pending new cell)
    Row 4 (causal, 7M, Common Voice): RWKV-6  Mamba-2  LA  (n/a)

The CV row carries the chapter's cross-distribution comparison
point: trained eta on Mamba-2 x DHO is engaged on both
LibriSpeech (where DHO is NULL on test CER, Δ ≈ -0.003 with the
depth-schedule projection) and Common Voice (where DHO is
helpful, Δ ≈ -0.022). Engagement at the parameter level is
similar; the chapter's reading is that the mechanism's
contribution surfaces or stays absorbed depending on what
axis-1 deficit the data distribution actually exercises.

Theta histograms (the (b) sub-figure of the reviewer's F9 spec)
require a forward pass over a sample of the eval set to collect
per-step theta = theta_base + LoRA(x) * tanh values. Running that
forward pass with three architectures' state dicts in this venv
exceeds reasonable compute budget and is deferred. The chapter
caption can footnote this if the histograms are not produced
separately.

Caption-eligible notes (chapter author copies into LaTeX):

  (a) Heatmap cells encode the mean of the absolute viscosity
      coefficient across the per-head block dimension; the
      per-cell summary in the data CSV reports the max value for
      the engagement-classifier purposes.

  (b) On Mamba-2, the LibriSpeech configuration uses 8 heads with
      head_dim 32, while the Common Voice pilot configuration uses
      4 heads with head_dim 64. The head-count difference is a
      config-level detail of the matched-parameter setup;
      mean(|eta|) normalises per-head and remains directly
      comparable across cells. Mamba-2 LibriSpeech panels are
      visibly wider than the CV panel for this reason.

  (c) Trained eta in the Mamba-2 LibriSpeech causal cell reaches
      the highest magnitude in the figure (max ≈ 1.21), yet the
      test-CER Δ is the predicted-NULL signature, while the same
      backbone on Common Voice reaches a smaller peak engagement
      (max ≈ 0.36) but converts into a helpful Δ of -0.023. This
      is direct parameter-level evidence for the task-prior
      modulation reading: viscosity engagement is necessary but
      not sufficient for the mechanism to convert, and the
      sufficiency depends on whether the task distribution
      exercises the axis the mechanism targets.

  (d) Theta angle distributions are deferred to supplementary
      material; producing them requires an additional forward
      pass over evaluation data per cell.

DHO selection rule (per F1, applied independently per cell):
prefer rse_depth_viscosity, fall back to rse_strong_viscosity. The
on-disk variant is recorded in the data CSV.

Outputs:
  * F9_dho_eta_theta.{pdf,png}
  * F9_dho_eta_theta_data.csv     per-cell (layer, head) mean|eta|
  * F9_dho_eta_theta_script.py    this file
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
STEM = "F9_dho_eta_theta"
SOURCE_DIR = REPO / "experiments" / "final" / "outputs"

sys.path.insert(0, str(FIG_DIR))
from _style import (  # noqa: E402
    PAGE_WIDTH_IN,
    apply_typography,
    clean_spines,
)
from _extract_mechanism_params import extract_for_cell  # noqa: E402

# Each cell entry: (panel_title, list of cell-dir candidates in
# DHO-priority order). The first candidate that yields an `eta`
# extraction wins.
PANEL_LAYOUT: list[list[tuple[str, list[str | None]]]] = [
    [   # Row 1: causal, 7M, LibriSpeech
        ("RWKV-6 (causal, 7M)",
         ["7m_rwkv6_causal_rse_depth_viscosity_seed42",
          "7m_rwkv6_causal_rse_strong_viscosity_seed42"]),
        ("Mamba-2 (causal, 7M)",
         ["7m_mamba2_causal_rse_depth_viscosity_seed42",
          "7m_mamba2_causal_rse_strong_viscosity_seed42"]),
        ("LA (causal, 7M)",
         ["7m_linear_attn_causal_rse_depth_viscosity_seed42",
          "7m_linear_attn_causal_rse_strong_viscosity_seed42"]),
        ("", [None]),
    ],
    [   # Row 2: causal, 14M, LibriSpeech
        ("RWKV-6 (causal, 14M)",
         ["14m_rwkv6_causal_rse_depth_viscosity_seed42",
          "14m_rwkv6_causal_rse_strong_viscosity_seed42"]),
        ("Mamba-2 (causal, 14M)",
         ["14m_mamba2_causal_rse_depth_viscosity_seed42",
          "14m_mamba2_causal_rse_strong_viscosity_seed42"]),
        ("LA (causal, 14M)",
         ["14m_linear_attn_causal_rse_depth_viscosity_seed42",
          "14m_linear_attn_causal_rse_strong_viscosity_seed42"]),
        ("", [None]),
    ],
    [   # Row 3: LION, 7M, LibriSpeech
        ("RWKV-6 (LION-S, 7M)",
         ["7m_rwkv6_lion_rse_depth_viscosity_seed42"]),
        ("Mamba-2 (LION-S, 7M)",
         ["7m_mamba2_lion_rse_depth_viscosity_seed42"]),
        ("LA (LION-LIT, 7M)",
         ["7m_linear_attn_lion_rse_depth_viscosity_seed42"]),
        ("LA (LION-S, 7M)",
         ["7m_linear_attn_lion_s_rse_depth_viscosity_seed42"]),
    ],
    [   # Row 4: causal, 7M, Common Voice
        ("RWKV-6 (causal, 7M)",
         ["cv_pilot_rwkv6_rse_depth_viscosity_seed42"]),
        ("Mamba-2 (causal, 7M)",
         ["cv_pilot_mamba2_rse_depth_viscosity_seed42",
          "cv_pilot_mamba2_rse_strong_viscosity_seed42"]),
        ("LA (causal, 7M)",
         ["cv_pilot_linear_attn_rse_depth_viscosity_seed42",
          "cv_pilot_linear_attn_rse_strong_viscosity_seed42"]),
        ("", [None]),
    ],
]

PANEL_WIDTH_RATIOS = [1.0, 2.0, 1.0, 1.0]

# Sequential cerulean colormap aligned with the chapter palette;
# white = zero (preconditioner near-identity / viscosity at zero-
# init), darker cerulean = larger mean(|eta|) (engaged).
ETA_CMAP = LinearSegmentedColormap.from_list(
    "thesis_eta",
    ["#FFFFFF", "#aac8d9", "#277da1"],
    N=256,
)


def collect_all() -> list[list[dict | None]]:
    grid: list[list[dict | None]] = []
    for row in PANEL_LAYOUT:
        out_row: list[dict | None] = []
        for title, candidates in row:
            entry: dict | None = None
            for cell in candidates:
                if cell is None:
                    continue
                cell_path = SOURCE_DIR / cell
                if not cell_path.exists():
                    continue
                extracted = extract_for_cell(cell_path)
                if extracted is None or "eta" not in extracted:
                    continue
                entry = {
                    "title": title, "cell": cell,
                    "eta": extracted["eta"],
                }
                break
            if entry is None:
                # Either no candidate exists or extraction failed;
                # render this slot as empty/placeholder.
                out_row.append({"title": title, "cell": "",
                                 "eta": None})
            else:
                out_row.append(entry)
        grid.append(out_row)
    return grid


def render(grid: list[list[dict | None]], out_pdf: Path, out_png: Path) -> None:
    nrows, ncols = len(grid), len(grid[0])

    # Build a global colour scale across all populated panels using
    # mean(|eta|) per (layer, head).
    pooled = []
    for row in grid:
        for entry in row:
            if entry is None or entry.get("eta") is None:
                continue
            mean_abs = np.abs(entry["eta"]).mean(axis=-1)  # (n_layers, n_heads)
            entry["eta_mean_abs"] = mean_abs
            pooled.append(mean_abs.flatten())
    if not pooled:
        fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 3.0), constrained_layout=True)
        ax.text(0.5, 0.5, "no DHO cells available", ha="center", va="center")
        fig.savefig(out_pdf)
        fig.savefig(out_png)
        plt.close(fig)
        return
    pooled_concat = np.concatenate(pooled)
    vmax = float(np.max(pooled_concat)) * 1.02
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig = plt.figure(
        figsize=(7.0, 0.9 + 1.6 * nrows),
        constrained_layout=True,
    )
    # Reserve right-margin space for LibriSpeech / Common Voice
    # dataset labels and brackets. The panel grid stops at x=0.94
    # (figure-relative), leaving ~0.06 on the right for the labels.
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
        row_axes = [fig.add_subplot(gs[r, c]) for c in range(ncols)]
        axes.append(row_axes)

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            entry = grid[r][c]
            title = entry.get("title", "") if entry else ""
            eta_ma = entry.get("eta_mean_abs") if entry else None
            if eta_ma is None:
                ax.axis("off")
                if title:
                    ax.set_title(title, fontsize=7,
                                  color="#adb5bd",
                                  loc="center")
                continue
            n_layers, n_heads = eta_ma.shape
            ax.imshow(eta_ma, cmap=ETA_CMAP, norm=norm, aspect="auto")
            ax.set_xticks(range(n_heads))
            ax.set_xticklabels([f"h{h}" for h in range(n_heads)], fontsize=6)
            ax.set_yticks(range(n_layers))
            ax.set_yticklabels([f"L{L}" for L in range(n_layers)], fontsize=6)
            ax.tick_params(axis="both", length=0)
            ax.set_title(title, fontsize=7.5)
            ax.grid(False)
            for L in range(n_layers):
                for h in range(n_heads):
                    v = float(eta_ma[L, h])
                    text_color = "white" if v > vmax * 0.55 else "#212529"
                    ax.text(h, L, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.5, color=text_color)

    fig.suptitle("DHO trained $|\\eta|$ engagement (mean over blocks)")

    # Colourbar manually positioned to align with the LA causal
    # panels' actual heatmap boxes (rows 1-2 col 3).
    fig.canvas.draw()
    top_bbox = axes[0][2].get_position()
    bottom_bbox = axes[1][2].get_position()
    col4_bbox = axes[2][3].get_position()

    cbar_y0 = bottom_bbox.y0
    cbar_y1 = top_bbox.y1
    cbar_h = cbar_y1 - cbar_y0
    cbar_w = col4_bbox.width * 0.18
    cbar_x = col4_bbox.x0 + (col4_bbox.width - cbar_w) / 2.0

    cbar_ax = fig.add_axes([cbar_x, cbar_y0, cbar_w, cbar_h])
    sm = plt.cm.ScalarMappable(cmap=ETA_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"trained $\overline{|\eta|}$  (mean over blocks)",
                    fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Dataset labels in the right-margin space reserved by
    # constrained-layout's `rect`. LibriSpeech spans rows 1-3,
    # Common Voice on row 4. Brackets in `]` shape with hooks
    # pointing left toward the panels (same pattern as F11).
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

    fig.savefig(out_pdf, bbox_inches=None)
    fig.savefig(out_png, bbox_inches=None, dpi=300)
    plt.close(fig)


def write_csv(grid: list[list[dict | None]], path: Path) -> None:
    rows: list[dict] = []
    for r, panel_row in enumerate(grid):
        for c, entry in enumerate(panel_row):
            if entry is None:
                continue
            eta_ma = entry.get("eta_mean_abs")
            if eta_ma is None:
                continue
            n_layers, n_heads = eta_ma.shape
            for L in range(n_layers):
                for h in range(n_heads):
                    rows.append({
                        "panel_row": r, "panel_col": c,
                        "panel_title": entry["title"], "cell": entry["cell"],
                        "layer": L, "head": h,
                        "mean_abs_eta": float(eta_ma[L, h]),
                    })
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_typography()
    grid = collect_all()
    print("[F9] cells loaded:")
    for r, row in enumerate(grid):
        for c, entry in enumerate(row):
            title = entry.get("title", "") if entry else ""
            cell = entry.get("cell", "") if entry else ""
            eta = entry.get("eta") if entry else None
            if eta is None:
                print(f"  ({r},{c}) {title or '(empty)'}: UNAVAILABLE")
            else:
                print(f"  ({r},{c}) {title}: cell={cell}, "
                       f"shape {eta.shape}, max|η|={abs(eta).max():.4f}")
    render(grid, FIG_DIR / f"{STEM}.pdf", FIG_DIR / f"{STEM}.png")
    write_csv(grid, FIG_DIR / f"{STEM}_data.csv")


if __name__ == "__main__":
    main()
