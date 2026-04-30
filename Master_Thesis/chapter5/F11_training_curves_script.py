"""F11 training-curve consolidated grid (full scope).

Four rows x four columns:

    Row 1 (7M causal LS):    RWKV-6  Mamba-2  LA  (empty)
    Row 2 (7M LION LS):      RWKV-6  Mamba-2  LA           LA
                             (LION-S) (LION-S) (LION-LIT)   (LION-S)
    Row 3 (14M causal LS):   RWKV-6  Mamba-2  LA  (empty)
    Row 4 (7M causal CV):    RWKV-6  Mamba-2  LA  (empty)

LION at 14M is out of scope per Master_Plan §LibriSpeech 14M. Each
panel shows dev CER over training epochs for the canonical
mechanism cells of that (architecture, mode, scale) configuration.
Lines are coloured by ARCH_COLOR per architecture; mechanism is
encoded by line style:

    vanilla        - solid, no marker
    MSDC           - dashed
    CVD            - dotted
    DHO            - dash-dot
    composition    - solid + open-circle markers every 5 ep,
                     alpha 0.7

The DHO line uses the per-cell selection rule from F1
(rse_depth_viscosity preferred, rse_strong_viscosity fallback).
Per-row sharey; per-row y-axes differ because 7M, 14M, and LION-LIT
absolute CER ranges are not comparable on a single scale.

Callouts on row 3 (14M causal):
  * RWKV-6 vanilla: "plateaued at ep 50" (slope ≈ 0 falsifies the
    undertraining-at-14M hypothesis carried in
    Thesis_Positioning §14; the NEG-scaling story stands as
    structural rather than budget-limited).
  * Mamba-2 × MSDC: "matrix ceiling 0.063" at the final-epoch
    endpoint. Slope at ep 50 is also ≈ 0, confirming the ceiling
    is structural at the matched budget rather than truncation.

Outputs:
  * F11_training_curves.{pdf,png}
  * F11_training_curves_data.csv
  * F11_training_curves_script.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/tmp/Master_Thesis_Exps")
FIG_DIR = REPO / "Master_Thesis" / "figures" / "chapter5"
STEM = "F11_training_curves"
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

# Per-(panel) cells. Each entry: list of (mechanism_label, cell_dir,
# line_style, marker, alpha). Composition line uses markers + alpha.
LINE_STYLES = {
    "vanilla":     ("solid",   None, 1.0),
    "MSDC":        ("dashed",  None, 1.0),
    "CVD":         ("dotted",  None, 1.0),
    "DHO":         ("dashdot", None, 1.0),
    "composition": ("solid",   "o",  0.7),
}


def _entry(label: str, cell_basename: str) -> tuple[str, str, str, str | None, float]:
    ls, marker, alpha = LINE_STYLES[label if label in LINE_STYLES else "composition"]
    return (label, cell_basename, ls, marker, alpha)


PANEL_LAYOUT: list[list[tuple[str, str, list]]] = [
    # ---- Row 1: 7M causal ----
    [
        ("rwkv6", "RWKV-6 (causal, 7M)", [
            _entry("vanilla",     "7m_rwkv6_causal_vanilla_seed42"),
            _entry("MSDC",        "7m_rwkv6_causal_multidil_v2_seed42"),
            _entry("CVD",         "7m_rwkv6_causal_lucid_chunked_seed42"),
            _entry("DHO",         "7m_rwkv6_causal_rse_depth_viscosity_seed42"),
            _entry("composition", "7m_rwkv6_causal_p7_seed42"),
        ]),
        ("mamba2", "Mamba-2 (causal, 7M)", [
            _entry("vanilla",     "7m_mamba2_causal_vanilla_seed42"),
            _entry("MSDC",        "7m_mamba2_causal_multidil_v2_seed42"),
            _entry("CVD",         "7m_mamba2_causal_lucid_c_seed42"),
            _entry("DHO",         "7m_mamba2_causal_rse_strong_viscosity_seed42"),
            _entry("composition", "7m_mamba2_causal_lucid_c_x_multidil_v2_seed42"),
        ]),
        ("linear_attn", "LA (causal, 7M)", [
            _entry("vanilla",     "7m_linear_attn_causal_vanilla_seed42"),
            _entry("MSDC",        "7m_linear_attn_causal_multidil_v2_seed42"),
            _entry("CVD",         "7m_linear_attn_causal_lucid_seed42"),
            _entry("DHO",         "7m_linear_attn_causal_rse_strong_viscosity_seed42"),
            _entry("composition", "7m_linear_attn_causal_rse_x_multidil_v2_seed42"),
        ]),
        (None, "", []),
    ],
    # ---- Row 2: 7M LION ----
    [
        ("rwkv6", "RWKV-6 (LION-S, 7M)", [
            _entry("vanilla",     "7m_rwkv6_lion_vanilla_seed42"),
            _entry("MSDC",        "7m_rwkv6_lion_multidil_v2_seed42"),
            _entry("CVD",         "7m_rwkv6_lion_lucid_chunked_seed42"),
            _entry("DHO",         "7m_rwkv6_lion_rse_depth_viscosity_seed42"),
            _entry("composition", "7m_rwkv6_lion_p7_seed42"),
        ]),
        ("mamba2", "Mamba-2 (LION-S, 7M)", [
            _entry("vanilla",     "7m_mamba2_lion_vanilla_seed42"),
            _entry("MSDC",        "7m_mamba2_lion_multidil_v2_seed42"),
            _entry("CVD",         "7m_mamba2_lion_lucid_c_seed42"),
            _entry("DHO",         "7m_mamba2_lion_rse_depth_viscosity_seed42"),
            _entry("composition", "7m_mamba2_lion_p7_seed42"),
        ]),
        ("linear_attn", "LA (LION-LIT, 7M)", [
            _entry("vanilla",     "7m_linear_attn_lion_vanilla_seed42"),
            _entry("MSDC",        "7m_linear_attn_lion_multidil_v2_seed42"),
            _entry("CVD",         "7m_linear_attn_lion_lucid_seed42"),
            _entry("DHO",         "7m_linear_attn_lion_rse_depth_viscosity_seed42"),
            _entry("composition", "7m_linear_attn_lion_rse_x_multidil_v2_seed42"),
        ]),
        ("linear_attn", "LA (LION-S, 7M)", [
            _entry("vanilla",     "7m_linear_attn_lion_s_vanilla_seed42"),
            _entry("MSDC",        "7m_linear_attn_lion_s_multidil_v2_seed42"),
            _entry("CVD",         "7m_linear_attn_lion_s_lucid_seed42"),
            _entry("DHO",         "7m_linear_attn_lion_s_rse_depth_viscosity_seed42"),
            _entry("composition", "7m_linear_attn_lion_s_lucid_multidil_v2_seed42"),
        ]),
    ],
    # ---- Row 3: 14M causal ----
    [
        ("rwkv6", "RWKV-6 (causal, 14M)", [
            _entry("vanilla",     "14m_rwkv6_causal_vanilla_seed42"),
            _entry("MSDC",        "14m_rwkv6_causal_multidil_v2_seed42"),
            _entry("CVD",         "14m_rwkv6_causal_lucid_chunked_seed42"),
            _entry("DHO",         "14m_rwkv6_causal_rse_strong_viscosity_seed42"),
            _entry("composition", "14m_rwkv6_causal_p7_seed42"),
        ]),
        ("mamba2", "Mamba-2 (causal, 14M)", [
            _entry("vanilla",     "14m_mamba2_causal_vanilla_seed42"),
            _entry("MSDC",        "14m_mamba2_causal_multidil_v2_seed42"),
            _entry("CVD",         "14m_mamba2_causal_lucid_c_seed42"),
            _entry("DHO",         "14m_mamba2_causal_rse_depth_viscosity_seed42"),
            _entry("composition", "14m_mamba2_causal_lucid_c_x_multidil_v2_seed42"),
        ]),
        ("linear_attn", "LA (causal, 14M)", [
            _entry("vanilla",     "14m_linear_attn_causal_vanilla_seed42"),
            _entry("MSDC",        "14m_linear_attn_causal_multidil_v2_seed42"),
            _entry("CVD",         "14m_linear_attn_causal_lucid_seed42"),
            _entry("DHO",         "14m_linear_attn_causal_rse_strong_viscosity_seed42"),
            _entry("composition", "14m_linear_attn_causal_rse_x_multidil_v2_seed42"),
        ]),
        (None, "", []),
    ],
    # ---- Row 4: 7M causal Common Voice ----
    [
        ("rwkv6", "RWKV-6 (causal, 7M)", [
            _entry("vanilla",     "cv_pilot_rwkv6_seed42"),
            _entry("MSDC",        "cv_pilot_rwkv6_multidil_v2_seed42"),
            _entry("CVD",         "cv_pilot_rwkv6_lucid_chunked_seed42"),
            _entry("DHO",         "cv_pilot_rwkv6_rse_depth_viscosity_seed42"),
            _entry("composition", "cv_pilot_rwkv6_lucid_multidil_v2_seed42"),
        ]),
        ("mamba2", "Mamba-2 (causal, 7M)", [
            _entry("vanilla",     "cv_pilot_mamba2_seed42"),
            _entry("MSDC",        "cv_pilot_mamba2_multidil_v2_seed42"),
            _entry("CVD",         "cv_pilot_mamba2_lucid_c_seed42"),
            _entry("DHO",         "cv_pilot_mamba2_rse_depth_viscosity_seed42"),
            _entry("composition", "cv_pilot_mamba2_lucid_c_multidil_v2_seed42"),
        ]),
        ("linear_attn", "LA (causal, 7M)", [
            _entry("vanilla",     "cv_pilot_linear_attn_seed42"),
            _entry("MSDC",        "cv_pilot_linear_attn_multidil_v2_seed42"),
            _entry("CVD",         "cv_pilot_linear_attn_lucid_seed42"),
            _entry("DHO",         "cv_pilot_linear_attn_rse_strong_viscosity_seed42"),
            _entry("composition", "cv_pilot_linear_attn_rse_x_multidil_v2_seed42"),
        ]),
        (None, "", []),
    ],
]

CER_METRIC = "dev_cer"


def load_history(cell_basename: str) -> pd.DataFrame | None:
    h = SOURCE_DIR / cell_basename / "history.csv"
    if not h.exists():
        return None
    return pd.read_csv(h)


def render(out_pdf: Path, out_png: Path) -> list[dict]:
    nrows = 4
    ncols = 4
    # F11 carries four columns including the LA-LION-S panel in
    # row 2, which would crowd at the standard 5.5 in text width.
    # Widen to 8.0 in to give per-panel ~1.8 in plus reserve a
    # small right margin for LibriSpeech / Common Voice labels.
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(8.0, 9.4),
        constrained_layout=True,
    )
    # Reserve right-margin space for the dataset labels: tell
    # constrained_layout to leave a wider gap on the right so the
    # rightmost panel does not abut the figure edge.
    try:
        fig.get_layout_engine().set(rect=(0.0, 0.0, 0.94, 1.0))
    except (AttributeError, TypeError):
        pass
    rows_for_csv: list[dict] = []

    for r in range(nrows):
        # Per-row sharey: collect lines first to compute ymin/ymax,
        # then re-apply.
        row_axes = [axes[r, c] for c in range(ncols)]
        plotted_data: list[tuple] = []  # (ax, arch, panel_title, lines)
        for c in range(ncols):
            ax = row_axes[c]
            arch, title, entries = PANEL_LAYOUT[r][c]
            if arch is None:
                ax.axis("off")
                continue
            clean_spines(ax)
            ax.set_axisbelow(True)
            color = ARCH_COLOR[arch]
            for label, cell, ls, marker, alpha in entries:
                df = load_history(cell)
                if df is None or CER_METRIC not in df.columns:
                    continue
                kwargs = dict(
                    color=color, linewidth=1.0, linestyle=ls,
                    alpha=alpha, label=label, zorder=2,
                )
                if marker is not None:
                    kwargs.update(dict(
                        marker=marker, markersize=3.5,
                        markerfacecolor="white", markevery=5,
                    ))
                ax.plot(df["epoch"], df[CER_METRIC], **kwargs)
                for _, row_v in df.iterrows():
                    rows_for_csv.append({
                        "panel_row": r,
                        "arch": arch, "mechanism": label, "cell": cell,
                        "epoch": int(row_v["epoch"]),
                        "dev_cer": float(row_v[CER_METRIC]),
                    })
            ax.set_title(title, fontsize=7.5)
            ax.set_xlabel("Epoch", fontsize=7)
            ax.tick_params(axis="both", labelsize=6.5)
            ax.legend(fontsize=5.5, loc="upper right")
            plotted_data.append((ax, arch, title, entries))
        # Sharey across this row's populated panels.
        if plotted_data:
            ymins = []
            ymaxs = []
            for ax, _, _, _ in plotted_data:
                ymin, ymax = ax.get_ylim()
                ymins.append(ymin)
                ymaxs.append(ymax)
            for ax, _, _, _ in plotted_data:
                ax.set_ylim(min(ymins), max(ymaxs))
        # y-axis label only on the leftmost panel of each row.
        if plotted_data:
            row_axes[0].set_ylabel("Dev CER", fontsize=7.5)

    # Callouts placed inside panels using axes-relative xytext so the
    # figure grid (panel widths and inter-panel spacing) is not
    # influenced by annotation offsets. Anchor (xy) is in data coords
    # at a specific epoch on the target curve. fontsize 6.5 pt to fit
    # crowded panels (LA-LION-LIT carries three callouts).
    callouts = []

    def _at(cell: str, ep: int) -> tuple[float, float]:
        df = pd.read_csv(SOURCE_DIR / cell / "history.csv")
        row = df[df["epoch"] == ep]
        if row.empty:
            row = df.iloc[[-1]]
        return float(row["epoch"].iloc[0]), float(row["dev_cer"].iloc[0])

    # Row 1 col 1 — RWKV-6 7M causal: DHO marginal + composition.
    callouts.append((axes[0, 0],
        "DHO marginal\nWKV absorbs decay\n(Δ −0.006)",
        _at("7m_rwkv6_causal_rse_depth_viscosity_seed42", 45),
        (0.62, 0.55)))
    callouts.append((axes[0, 0],
        "CVD × MSDC\ncomposition\n(test 0.079, Δ −0.026)",
        _at("7m_rwkv6_causal_p7_seed42", 50),
        (0.45, 0.30)))
    # Row 1 col 2 — Mamba-2 7M causal: DHO NULL + composition.
    callouts.append((axes[0, 1],
        "DHO NULL\ntracks vanilla\n(Δ ≈ 0)",
        _at("7m_mamba2_causal_rse_strong_viscosity_seed42", 45),
        (0.66, 0.55)))
    callouts.append((axes[0, 1],
        "CVD × MSDC\ncomposition\n(test 0.080, Δ −0.024)",
        _at("7m_mamba2_causal_lucid_c_x_multidil_v2_seed42", 50),
        (0.45, 0.30)))
    # Row 1 col 3 — LA 7M causal: DHO causal BREAK + composition.
    callouts.append((axes[0, 2],
        "DHO\nBREAK (Δ −0.068)",
        _at("7m_linear_attn_causal_rse_strong_viscosity_seed42", 50),
        (0.62, 0.42)))
    callouts.append((axes[0, 2],
        "DHO × MSDC\nΔ −0.088",
        _at("7m_linear_attn_causal_rse_x_multidil_v2_seed42", 50),
        (0.22, 0.05)))

    # Row 2 col 3 — LA LION-LIT: 3 callouts.
    callouts.append((axes[1, 2],
        "CVD\nharmful without decay\n(Δ +0.024)",
        _at("7m_linear_attn_lion_lucid_seed42", 35),
        (0.32, 0.85)))
    callouts.append((axes[1, 2],
        "DHO\nLION-LIT BREAK (Δ −0.191)",
        _at("7m_linear_attn_lion_rse_depth_viscosity_seed42", 50),
        (0.52, 0.40)))
    callouts.append((axes[1, 2],
        "DHO × MSDC\nbest Δ in matrix (Δ −0.199)",
        _at("7m_linear_attn_lion_rse_x_multidil_v2_seed42", 50),
        (0.30, 0.05)))

    # Row 2 col 4 — LA LION-S: best LION-S cell is DHO at 0.099
    # (lower than the P7-style composition at 0.113).
    callouts.append((axes[1, 3],
        "DHO\nbest LION-S cell\n(test 0.099, Δ −0.039)",
        _at("7m_linear_attn_lion_s_rse_depth_viscosity_seed42", 50),
        (0.46, 0.42)))

    # Row 3 col 1 — RWKV-6 14M causal: MSDC Δ growth.
    # (the "still descending at ep 50" callout was removed at the
    # senior reviewer's request — the slope at ep 50 is essentially
    # zero, which falsifies the still-descending hypothesis; the
    # plateau reading is left to chapter prose.)
    callouts.append((axes[2, 0],
        "MSDC\nΔ grows with scale\n(−0.009 vs 7M)",
        _at("14m_rwkv6_causal_multidil_v2_seed42", 50),
        (0.55, 0.30)))

    # Row 3 col 2 — Mamba-2 14M causal: lowest CER in matrix,
    # POS-scaling vanilla, DHO absorption persists.
    callouts.append((axes[2, 1],
        "MSDC\nlowest CER in matrix\n(test 0.063)",
        _at("14m_mamba2_causal_multidil_v2_seed42", 50),
        (0.40, 0.20)))
    callouts.append((axes[2, 1],
        "vanilla\nimproves with scale\n(Δ −0.021 vs 7M)",
        _at("14m_mamba2_causal_vanilla_seed42", 30),
        (0.42, 0.58)))
    callouts.append((axes[2, 1],
        "DHO\nNULL persists at scale\n(Δ +0.001)",
        _at("14m_mamba2_causal_rse_depth_viscosity_seed42", 45),
        (0.74, 0.40)))

    # Row 3 col 3 — LA 14M causal: DHO×MSDC composition is the
    # deepest helpful Δ at 14M.
    callouts.append((axes[2, 2],
        "DHO × MSDC\nΔ −0.057",
        _at("14m_linear_attn_causal_rse_x_multidil_v2_seed42", 50),
        (0.55, 0.30)))

    # Row 4 col 2 — Mamba-2 CV: DHO is NULL on LibriSpeech but
    # converts to helpful on Common Voice.
    callouts.append((axes[3, 1],
        "DHO\nNULL on LibriSpeech\nhelpful on CV (Δ −0.023)",
        _at("cv_pilot_mamba2_rse_depth_viscosity_seed42", 45),
        (0.50, 0.45)))

    for ax, text, xy, xytext_axfrac in callouts:
        ax.annotate(
            text, xy=xy, xycoords="data",
            xytext=xytext_axfrac, textcoords="axes fraction",
            ha="center", va="center",
            fontsize=6, color=SPINE_COLOR,
            arrowprops=dict(arrowstyle="->", color=SPINE_COLOR, lw=0.5,
                             connectionstyle="arc3,rad=.2",
                             shrinkA=2, shrinkB=2),
        )

    fig.suptitle("Training-curve consolidated grid (dev CER)")

    # Dataset labels in the right-margin space reserved by the
    # constrained-layout `rect` above. LibriSpeech spans rows 1-3
    # (LS row group); Common Voice on row 4. Labels rotated 270°
    # for vertical text. A thin horizontal divider line between
    # LS and CV groups makes the dataset boundary explicit.
    fig.canvas.draw()
    libr_top = axes[0, 0].get_position().y1
    libr_bottom = axes[2, 0].get_position().y0
    cv_top = axes[3, 0].get_position().y1
    cv_bottom = axes[3, 0].get_position().y0
    label_x = 0.965
    fig.text(
        label_x, (libr_top + libr_bottom) / 2.0, "LibriSpeech",
        rotation=270, ha="center", va="center",
        fontsize=8.5, color=SPINE_COLOR, weight="bold",
    )
    fig.text(
        label_x, (cv_top + cv_bottom) / 2.0, "Common Voice",
        rotation=270, ha="center", va="center",
        fontsize=8.5, color=SPINE_COLOR, weight="bold",
    )
    # Vertical brackets grouping panels by dataset. Each bracket is
    # a `]` shape: a vertical line just to the right of the
    # rightmost panel, with two short horizontal hooks pointing
    # left toward the panels. LibriSpeech bracket spans rows 1-3,
    # Common Voice bracket spans row 4.
    from matplotlib.lines import Line2D
    bracket_x = label_x - 0.013
    hook_len = 0.006
    bracket_lw = 0.7

    def _draw_bracket(y_top: float, y_bottom: float) -> None:
        # Vertical spine.
        fig.add_artist(Line2D(
            [bracket_x, bracket_x], [y_bottom, y_top],
            color=SPINE_COLOR, linewidth=bracket_lw,
        ))
        # Top hook (pointing left toward the panels).
        fig.add_artist(Line2D(
            [bracket_x - hook_len, bracket_x], [y_top, y_top],
            color=SPINE_COLOR, linewidth=bracket_lw,
        ))
        # Bottom hook (pointing left toward the panels).
        fig.add_artist(Line2D(
            [bracket_x - hook_len, bracket_x], [y_bottom, y_bottom],
            color=SPINE_COLOR, linewidth=bracket_lw,
        ))

    _draw_bracket(libr_top, libr_bottom)
    _draw_bracket(cv_top, cv_bottom)

    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    return rows_for_csv


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_typography()
    rows = render(FIG_DIR / f"{STEM}.pdf", FIG_DIR / f"{STEM}.png")
    pd.DataFrame(rows).to_csv(FIG_DIR / f"{STEM}_data.csv", index=False)
    print(f"[F11] {len(rows)} (cell, epoch) rows written")


if __name__ == "__main__":
    main()
