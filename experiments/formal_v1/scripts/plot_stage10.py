#!/usr/bin/env python3
"""Stage-10 training-curve comparison plots.

Reads per-epoch history.csv from each Stage-10 run and its comparison
cohort references, then emits a small set of thesis-relevant plots under
``outputs/_stage10_plots/``:

    01_stage10_all_vs_vanilla.png
        Dev CER trajectory for all Stage-10 runs against vanilla / anchor.
    02_family_D_parametrisations.png
        Family-D quadratic-lift cross-parametrisation saturation
        (hadamard_n2, pom_vlift, chanmix_bypass, qtail_lowrank_all).
    03_family_A_input_vs_structural.png
        Family-A input-side (convshift variants) vs structural (loglinear).
    04_cayley_vs_t2.png
        Stage 10.5 Cayley diagnostic vs T2 primary (matched-epoch tracking).
    05_cer_vs_compute.png
        Final best-dev-CER vs 30-epoch wallclock scatter (Pareto view).

Safe to re-run mid-training; skips runs whose history.csv is empty.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


STYLE = {
    "figure.figsize": (9, 5.2),
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "savefig.bbox": "tight",
    "savefig.dpi": 140,
    "lines.linewidth": 1.8,
}
plt.rcParams.update(STYLE)


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "_stage10_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical run-dir mapping so we find each backbone's history.
RUN_DIRS = {
    # Stage 10 runs
    "rwkv6_loglinear":                        "outputs/rwkv6_loglinear_seed42",
    "rwkv6_m2rnn_sparse":                     "outputs/rwkv6_m2rnn_sparse_seed42",
    "rwkv6_convshift_multidil":               "outputs/rwkv6_convshift_multidil_seed42",
    "rwkv6_convshift_multidil_symmetric":     "outputs/rwkv6_convshift_multidil_symmetric_seed42",
    "rwkv6_chanmix_bypass":                   "outputs/rwkv6_chanmix_bypass_seed42",
    "rwkv6_pom_vlift":                        "outputs/rwkv6_pom_vlift_seed42",
    "rwkv6_orthogonal":                       "outputs/rwkv6_orthogonal_seed42",
    # Cohort references (Stages 2–9)
    "rwkv6":                                  "outputs/exp02_rwkv6_seed42",
    "rwkv6_rse_strong_viscosity":             "outputs/stage5_05_rwkv6_rse_strong_viscosity_seed42",
    "rwkv6_convshift_trap":                   "outputs/disc06_rwkv6_convshift_trap_seed42",
    "rwkv6_nonnormal_rse_viscosity":          "outputs/s8_t2_rwkv6_nonnormal_rse_viscosity_seed42",
    "rwkv6_delta_warmstart_fixed":            "outputs/s8_t1a_rwkv6_delta_warmstart_fixed_seed42",
    "rwkv6_rse_dphi_viscosity":               "outputs/s7a_dphi_rwkv6_rse_dphi_viscosity_seed42",
    # Family-D reference parametrisations (if present)
    "rwkv6_hadamard_n2":                      "outputs/stage6_02_rwkv6_hadamard_n2_seed42",
    "rwkv6_qtail_lowrank_all":                "outputs/lra_rwkv6_qtail_lowrank_all_seed42",
}

# Curated colour palette so the same backbone keeps its colour across plots.
COLORS = {
    "rwkv6":                                  "#888888",   # neutral grey = baseline
    "rwkv6_rse_strong_viscosity":             "#3b82f6",   # anchor
    "rwkv6_convshift_trap":                   "#22c55e",
    "rwkv6_nonnormal_rse_viscosity":          "#7c3aed",   # T2
    "rwkv6_delta_warmstart_fixed":            "#94a3b8",   # T1
    "rwkv6_rse_dphi_viscosity":               "#a78bfa",   # A1′
    "rwkv6_hadamard_n2":                      "#f59e0b",
    "rwkv6_qtail_lowrank_all":                "#ef4444",
    # Stage 10 backbones
    "rwkv6_loglinear":                        "#14b8a6",
    "rwkv6_m2rnn_sparse":                     "#ec4899",
    "rwkv6_convshift_multidil":               "#f97316",
    "rwkv6_convshift_multidil_symmetric":     "#10b981",
    "rwkv6_chanmix_bypass":                   "#eab308",
    "rwkv6_pom_vlift":                        "#6366f1",
    "rwkv6_orthogonal":                       "#dc2626",
}

# Verdict bands per STAGE10_PLAN §6.10.1 (uniform reference for plots)
BAND_EDGES_VS_VANILLA = {
    "BREAK": 0.1200,
    "MARGINAL": 0.1230,
    "PLATEAU": 0.1260,
}


def _load(backbone: str) -> pd.DataFrame | None:
    run_dir = RUN_DIRS.get(backbone)
    if run_dir is None:
        return None
    hist = ROOT / run_dir / "history.csv"
    if not hist.exists():
        return None
    try:
        df = pd.read_csv(hist)
    except Exception:
        return None
    if df.empty or "dev_cer" not in df.columns or "epoch" not in df.columns:
        return None
    return df.sort_values("epoch").reset_index(drop=True)


def _plot_curves(
    backbones: list[str],
    title: str,
    out_name: str,
    *,
    annotations: list[tuple[float, str, str]] | None = None,
    xlim: tuple[float, float] = (1, 30),
    ylim: tuple[float, float] = (0.10, 0.30),
    highlight: list[str] | None = None,
):
    highlight = highlight or []
    fig, ax = plt.subplots()

    for bb in backbones:
        df = _load(bb)
        if df is None:
            continue
        x = df["epoch"].to_numpy()
        y = df["dev_cer"].to_numpy()
        color = COLORS.get(bb, "black")
        lw = 2.4 if bb in highlight else 1.5
        alpha = 1.0 if bb in highlight else 0.85
        ls = "-" if not bb.endswith("_in_progress") else "--"
        ax.plot(x, y, label=bb, color=color, linewidth=lw, alpha=alpha, linestyle=ls)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dev CER")
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    if annotations:
        for y, label, color in annotations:
            ax.axhline(y=y, linestyle=":", linewidth=0.9, color=color, alpha=0.75)
            ax.text(ax.get_xlim()[1] * 0.995, y, f" {label}", color=color,
                    fontsize=8, verticalalignment="center",
                    horizontalalignment="right")
    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)
    print(f"  wrote {OUT_DIR / out_name}")


def _plot_delta_to_reference(
    backbones: list[str],
    reference: str,
    title: str,
    out_name: str,
    *,
    ylim: tuple[float, float] = (-0.015, 0.015),
    highlight: list[str] | None = None,
):
    """Δ dev CER vs matched-epoch reference. Positive = behind reference.

    `reference` is a backbone key in RUN_DIRS. Use anchor
    (`rwkv6_rse_strong_viscosity`) instead of vanilla-on-exp02 to avoid
    the 80-ep schedule bias (PI review §1.1 / §1.2).
    """
    highlight = highlight or []
    ref_df = _load(reference)
    if ref_df is None:
        print(f"  skip (reference {reference} missing)")
        return
    ref_idx = ref_df.set_index("epoch")["dev_cer"]

    fig, ax = plt.subplots()
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.axhspan(-0.0014, 0.0014, color="lightgrey", alpha=0.5, label="±1σ (seed noise)")

    for bb in backbones:
        df = _load(bb)
        if df is None:
            continue
        joined = df.set_index("epoch").join(ref_idx.rename("ref_cer"), how="inner")
        x = joined.index.to_numpy()
        dy = (joined["dev_cer"] - joined["ref_cer"]).to_numpy()
        color = COLORS.get(bb, "black")
        lw = 2.4 if bb in highlight else 1.5
        ax.plot(x, dy, label=bb, color=color, linewidth=lw)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Dev CER − {reference} Dev CER")
    ax.set_title(title)
    ax.set_xlim(1, 30)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)
    print(f"  wrote {OUT_DIR / out_name}")


def _plot_cer_vs_compute(out_name: str):
    """Scatter of final best-dev-CER against 30-epoch wallclock (Pareto view)."""
    COMPUTE_HR = {  # from training history (median sec/ep × 30 / 3600)
        "rwkv6":                               0.95,
        "rwkv6_rse_strong_viscosity":          1.60,
        "rwkv6_convshift_trap":                1.02,
        "rwkv6_nonnormal_rse_viscosity":       1.72,
        "rwkv6_loglinear":                     3.00,
        "rwkv6_m2rnn_sparse":                  4.26,
        "rwkv6_convshift_multidil":            1.43,
        "rwkv6_convshift_multidil_symmetric":  1.48,
        "rwkv6_chanmix_bypass":                1.45,
        "rwkv6_pom_vlift":                     1.43,
        "rwkv6_orthogonal":                    7.65,
    }
    BEST = {}
    for bb in COMPUTE_HR.keys():
        df = _load(bb)
        if df is None:
            continue
        # Cap at epoch 30 for apples-to-apples comparison. Vanilla in
        # exp02_rwkv6_seed42 ran 80 ep on a different LR schedule; only
        # the first 30 epochs are directly comparable to Stage-10 runs.
        df30 = df[df["epoch"] <= 30]
        BEST[bb] = float(df30["dev_cer"].min())

    fig, ax = plt.subplots()
    for bb, cer in BEST.items():
        color = COLORS.get(bb, "black")
        marker = "*" if bb == "rwkv6_convshift_multidil_symmetric" else "o"
        size = 120 if bb == "rwkv6_convshift_multidil_symmetric" else 60
        ax.scatter(COMPUTE_HR[bb], cer, color=color, s=size, marker=marker, edgecolors="black", linewidths=0.5)
        # label offset avoiding overlap
        ax.annotate(
            bb.replace("rwkv6_", ""),
            (COMPUTE_HR[bb], cer),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7.5, alpha=0.85,
        )
    # Pareto frontier approximation
    pts = sorted(((COMPUTE_HR[bb], BEST[bb]) for bb in BEST.keys()), key=lambda p: p[0])
    frontier_x, frontier_y = [], []
    best_so_far = 1.0
    for x, y in pts:
        if y < best_so_far:
            frontier_x.append(x); frontier_y.append(y); best_so_far = y
    ax.plot(frontier_x, frontier_y, "--", color="grey", alpha=0.5, label="Pareto frontier")

    ax.set_xlabel("30-ep wallclock (h, 1× RTX PRO 6000)")
    ax.set_ylabel("Best Dev CER")
    ax.set_title("Stage-10 compute–quality frontier")
    ax.legend(fontsize=8)
    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)
    print(f"  wrote {OUT_DIR / out_name}")


def main():
    print(f"Writing plots to: {OUT_DIR}")

    # 1. All Stage-10 trajectories vs vanilla / anchor reference lines only.
    # (Dropped the exp02 vanilla curve; it's the 80-ep-schedule run whose
    # ep-30 lands at 0.1173 — visually contradicts the §9.1 0.1258 dashed
    # reference. Keep the dashed reference line instead.)
    _plot_curves(
        [
            "rwkv6_rse_strong_viscosity",
            "rwkv6_loglinear",
            "rwkv6_m2rnn_sparse",
            "rwkv6_convshift_multidil",
            "rwkv6_convshift_multidil_symmetric",
            "rwkv6_chanmix_bypass",
            "rwkv6_pom_vlift",
            "rwkv6_orthogonal",
        ],
        title="Stage 10 dev-CER trajectories (vs reference lines)",
        out_name="01_stage10_all_vs_vanilla.png",
        annotations=[
            (0.1258, "vanilla §9.1 (0.1258)", "#888888"),
            (0.1191, "anchor on-disk (0.1191)",  "#3b82f6"),
            (0.1145, "abs-best (0.1145)", "#22c55e"),
        ],
        ylim=(0.11, 0.30),
        highlight=["rwkv6_convshift_multidil_symmetric"],
    )

    # 2. Family-D FEATURE-SIDE mechanisms — quadratic-lift cluster only.
    # Dropped chanmix_bypass from this plot (PI review §1.1): the bypass is
    # α-gated activation interpolation, not a polynomial lift. It has its
    # own plot now (02b_family_D_chanmix_vs_quadratic.png).
    _plot_curves(
        [
            "rwkv6_hadamard_n2",
            "rwkv6_qtail_lowrank_all",
            "rwkv6_pom_vlift",
        ],
        title="Family-D quadratic-lift parametrisations saturate at ~0.125 (params: +0 vs +mod vs +198K)",
        out_name="02_family_D_quadratic_parametrisations.png",
        annotations=[
            (0.1258, "vanilla §9.1",     "#888888"),
            (0.1253, "hadamard_n2 (+0)", "#f59e0b"),
            (0.1254, "pom_vlift (+198K)", "#6366f1"),
            (0.1240, "qtail_lr_all (+mod)", "#ef4444"),
        ],
        ylim=(0.12, 0.27),
    )

    # 2b. Chanmix bypass (α-gated activation interpolation) as its own
    # plot — structurally distinct from the quadratic lifts.
    _plot_curves(
        [
            "rwkv6_chanmix_bypass",
            "rwkv6_hadamard_n2",
        ],
        title="Family-D α-gated activation interpolation vs quadratic lift (reference)",
        out_name="02b_family_D_chanmix_vs_quadratic.png",
        annotations=[
            (0.1258, "vanilla §9.1", "#888888"),
            (0.1251, "chanmix_bypass (+6 scalars)", "#eab308"),
            (0.1253, "hadamard_n2 (+24 scalars)", "#f59e0b"),
        ],
        ylim=(0.12, 0.27),
    )

    # 3. Family-A input-side vs structural
    _plot_curves(
        [
            "rwkv6",
            "rwkv6_convshift_trap",
            "rwkv6_loglinear",
            "rwkv6_convshift_multidil",
            "rwkv6_convshift_multidil_symmetric",
        ],
        title="Family-A: input-side (symmetric multi-dil) clears the ceiling",
        out_name="03_family_A_input_vs_structural.png",
        annotations=[
            (0.1258, "vanilla",       "#888888"),
            (0.1150, "convshift_trap","#22c55e"),
            (0.1153, "multidil_sym",  "#10b981"),
        ],
        ylim=(0.11, 0.27),
        highlight=["rwkv6_convshift_multidil_symmetric"],
    )

    # 4. Stage 10.5 Cayley diagnostic vs T2
    _plot_curves(
        [
            "rwkv6",
            "rwkv6_nonnormal_rse_viscosity",   # T2, the primary reference
            "rwkv6_orthogonal",                 # Stage 10.5 (in progress)
            "rwkv6_rse_strong_viscosity",      # anchor, lower bound
        ],
        title="Stage 10.5 Cayley-orthogonal diagnostic vs T2 primary",
        out_name="04_cayley_vs_t2.png",
        annotations=[
            (0.1258, "vanilla", "#888888"),
            (0.1202, "T2 final", "#7c3aed"),
            (0.1185, "anchor",   "#3b82f6"),
        ],
        ylim=(0.12, 0.30),
        highlight=["rwkv6_orthogonal"],
    )

    # 5. Δ-to-anchor trajectory (reproducible reference, avoids exp02 bias).
    # PI review §1.1 / §1.2: the exp02 vanilla reference has a +0.0085
    # systematic bias at ep 30; switching the reference to the anchor
    # gives a clean on-disk comparator with no LR-schedule mismatch.
    _plot_delta_to_reference(
        [
            "rwkv6_loglinear",
            "rwkv6_m2rnn_sparse",
            "rwkv6_convshift_multidil",
            "rwkv6_convshift_multidil_symmetric",
            "rwkv6_chanmix_bypass",
            "rwkv6_pom_vlift",
        ],
        reference="rwkv6_rse_strong_viscosity",
        title="Δ Dev CER vs matched-epoch ANCHOR (rse_strong_viscosity, 0.1191 on disk)",
        out_name="05_delta_vs_anchor.png",
        ylim=(-0.010, 0.020),
        highlight=["rwkv6_convshift_multidil_symmetric"],
    )

    # 6. CER vs compute (Pareto)
    _plot_cer_vs_compute("06_cer_vs_compute_pareto.png")

    print("done")


if __name__ == "__main__":
    main()
