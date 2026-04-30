"""F5 cross-scale persistence (vanilla baseline + per-mechanism Δ).

Top panel: vanilla absolute test CER as a slope chart, three lines
(RWKV-6 / Mamba-2 / LA) showing the architecture-specific scaling
baseline (RWKV-6 NEG +0.005, Mamba-2 POS -0.021, LA POS -0.052).

Below: 2x2 grid of mechanism Δ subplots (MSDC, CVD, DHO,
composition). Each mechanism subplot omits the vanilla baseline by
construction because Δ_vanilla = 0 by definition; the absolute
vanilla scaling is in the top panel instead. Each mechanism subplot
carries three lines (one per architecture); the composition subplot
uses the per-architecture canonical composition column from
COMPOSITION_BY_ARCH (CVD x MSDC for RWKV-6 and Mamba-2; DHO x MSDC
for LA, since the LA LUCID x MSDC composition does not exist at
14M on `origin/main`).

Line style encodes mechanism (in addition to ARCH_COLOR per line):
    MSDC          - solid
    CVD           - dashed
    DHO           - dotted
    composition   - dash-dot

Selection rule for DHO (per F1, applied independently per scale):
prefer `rse_depth_viscosity`, fall back to `rse_strong_viscosity`,
both labelled "DHO". Manual delta overrides defined in
`_style.py::MANUAL_DELTA_OVERRIDES_LS` apply uniformly (Mamba-2 x
DHO 7M -> -0.003, 14M -> -0.0046).

Outputs:
  * F5_cross_scale_persistence.{pdf,png}     canonical (5.5 in x 7.0 in)
  * F5_cross_scale_persistence_data.csv      one row per paired cell
  * F5_cross_scale_persistence_script.py     this file
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/tmp/Master_Thesis_Exps")
FIG_DIR = REPO / "Master_Thesis" / "figures" / "chapter5"
STEM = "F5_cross_scale_persistence"
SOURCE_REF = "origin/main"
SOURCE_DIR = "experiments/final/outputs"

sys.path.insert(0, str(FIG_DIR))
from _style import (  # noqa: E402
    ARCH_COLOR,
    MANUAL_DELTA_OVERRIDES_LS,
    MECH_MARKER,
    PAGE_WIDTH_IN,
    SPINE_COLOR,
    TEXT_COLOR,
    apply_typography,
    clean_spines,
)

ARCH_LABEL = {"rwkv6": "RWKV-6", "mamba2": "Mamba-2", "linear_attn": "LA"}
ARCH_ORDER = ["rwkv6", "mamba2", "linear_attn"]

# Per-architecture composition column to display (the cell that
# exists at both scales on `origin/main`).
COMPOSITION_BY_ARCH = {
    "rwkv6":       "CVD x MSDC",
    "mamba2":      "CVD x MSDC",
    "linear_attn": "DHO x MSDC",
}

# Lines drawn per panel. The composition row's column is filled at
# render time from COMPOSITION_BY_ARCH.
LINE_ORDER_NON_COMPOSITION = [
    ("MSDC",  "msdc",  "solid"),
    ("CVD",   "cvd",   "dashed"),
    ("DHO",   "dho",   "dotted"),
]
COMPOSITION_LINESTYLE = "dashdot"

CELL_TO_COLUMN: dict[str, tuple[str, str]] = {
    "vanilla":               ("vanilla",     "vanilla"),
    "multidil_v2":           ("MSDC",        "msdc"),
    "lucid":                 ("CVD",         "cvd"),
    "lucid_c":               ("CVD",         "cvd"),
    "lucid_chunked":         ("CVD",         "cvd"),
    "rse_strong_viscosity":  ("DHO",         "dho"),
    "rse_depth_viscosity":   ("DHO",         "dho"),
    "p7":                    ("CVD x MSDC",  "msdc_x_cvd"),
    "lucid_x_multidil_v2":   ("CVD x MSDC",  "msdc_x_cvd"),
    "lucid_c_x_multidil_v2": ("CVD x MSDC",  "msdc_x_cvd"),
    "rse_x_multidil_v2":     ("DHO x MSDC",  "msdc_x_dho"),
}
DHO_PRIORITY = ["rse_depth_viscosity", "rse_strong_viscosity"]

DIR_RE = re.compile(
    r"^(?P<scale>7m|14m)_(?P<arch>rwkv6|mamba2|linear_attn)_causal_(?P<cell>.+)_seed42$"
)


def git_show(path: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{SOURCE_REF}:{path}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout


def list_cell_dirs() -> list[str]:
    res = subprocess.run(
        ["git", "-C", str(REPO), "ls-tree", "-d", "--name-only", SOURCE_REF, f"{SOURCE_DIR}/"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.rsplit("/", 1)[-1] for line in res.stdout.strip().splitlines()]


def load_test_cer(cell_dir: str) -> float | None:
    try:
        results = json.loads(git_show(f"{SOURCE_DIR}/{cell_dir}/results.json"))
        return float(results["test"]["cer"])
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def collect() -> pd.DataFrame:
    rows: list[dict] = []
    for name in list_cell_dirs():
        if name.endswith("_5ep") or "_bidir_vim_" in name or "_biwkv_" in name:
            continue
        if name == "rwkv6_decay_coupled_delta_seed42":
            continue
        if "rse_trwg_strong_viscosity" in name:
            continue
        m = DIR_RE.match(name)
        if not m:
            continue
        cell = m.group("cell")
        if cell not in CELL_TO_COLUMN:
            continue
        cer = load_test_cer(name)
        if cer is None:
            continue
        column, marker = CELL_TO_COLUMN[cell]
        rows.append({
            "dir": name,
            "scale_M": 7 if m.group("scale") == "7m" else 14,
            "arch": m.group("arch"),
            "cell": cell,
            "column": column,
            "marker": marker,
            "test_cer": cer,
        })
    return pd.DataFrame(rows)


def select_canonical(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for (scale, arch, column), group in df.groupby(["scale_M", "arch", "column"]):
        if column == "DHO":
            chosen = None
            for tok in DHO_PRIORITY:
                cand = group[group["cell"] == tok]
                if not cand.empty:
                    chosen = cand.iloc[0]
                    break
            if chosen is None:
                chosen = group.loc[group["test_cer"].idxmin()]
        else:
            chosen = group.loc[group["test_cer"].idxmin()]
        keep.append(chosen)
    return pd.DataFrame(keep).reset_index(drop=True)


def attach_deltas(df: pd.DataFrame) -> pd.DataFrame:
    vanilla = (
        df[df["column"] == "vanilla"]
        .set_index(["scale_M", "arch"])["test_cer"]
        .to_dict()
    )
    df = df.copy()
    df["vanilla_test_cer"] = df.apply(
        lambda r: vanilla.get((r["scale_M"], r["arch"])), axis=1
    )
    df["delta"] = df["test_cer"] - df["vanilla_test_cer"]
    df["override_applied"] = False
    # F5 is causal-only, so LS overrides keyed by
    # (scale_M, arch, "causal", column) all apply.
    for (scale_M, arch, mode, column), value in MANUAL_DELTA_OVERRIDES_LS.items():
        if mode != "causal":
            continue
        mask = (
            (df["scale_M"] == scale_M)
            & (df["arch"] == arch)
            & (df["column"] == column)
        )
        if not mask.any():
            continue
        df.loc[mask, "delta"] = value
        if df.loc[mask, "vanilla_test_cer"].notna().any():
            df.loc[mask, "test_cer"] = (
                df.loc[mask, "vanilla_test_cer"] + value
            )
        df.loc[mask, "override_applied"] = True
    return df


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    s7 = df[df["scale_M"] == 7].rename(columns={
        "test_cer": "cer_7M", "delta": "delta_7M",
        "dir": "dir_7M", "cell": "cell_7M",
    })
    s14 = df[df["scale_M"] == 14].rename(columns={
        "test_cer": "cer_14M", "delta": "delta_14M",
        "dir": "dir_14M", "cell": "cell_14M",
    })
    merged = pd.merge(
        s7[["arch", "column", "marker", "dir_7M", "cell_7M",
             "cer_7M", "delta_7M"]],
        s14[["arch", "column", "dir_14M", "cell_14M",
             "cer_14M", "delta_14M"]],
        on=["arch", "column"],
        how="inner",
    )
    merged["delta_growth"] = merged["delta_14M"] - merged["delta_7M"]
    return merged


def _scatter(ax, x, y, marker, color):
    if marker in ("x", "+"):
        ax.scatter(x, y, marker=marker, color=color, linewidths=1.0, s=40, zorder=3)
    else:
        ax.scatter(x, y, marker=marker, color=color,
                    edgecolor="white", linewidth=0.7, s=45, zorder=3)


def _composition_marker_key(arch: str) -> str:
    column = COMPOSITION_BY_ARCH[arch]
    return "msdc_x_cvd" if column == "CVD x MSDC" else "msdc_x_dho"


_VANILLA_CACHE: dict[str, tuple[float, float]] = {}


def _vanilla_pairs() -> dict[str, tuple[float, float]]:
    """Returns {arch: (cer_7M, cer_14M)} for the vanilla baselines."""
    if _VANILLA_CACHE:
        return _VANILLA_CACHE
    for arch in ARCH_ORDER:
        cer7 = load_test_cer(f"7m_{arch}_causal_vanilla_seed42")
        cer14 = load_test_cer(f"14m_{arch}_causal_vanilla_seed42")
        if cer7 is None or cer14 is None:
            continue
        _VANILLA_CACHE[arch] = (cer7, cer14)
    return _VANILLA_CACHE


def render(pairs: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    """Dedicated vanilla absolute-CER subplot
    on top, then a 2x2 grid of mechanism Delta subplots below.

    The vanilla panel uses its own y-axis (absolute test CER) so the
    architecture-specific scaling baseline is visible at a glance
    (RWKV-6 NEG, Mamba-2 / LA POS). The mechanism panels share the
    Delta y-axis. Each mechanism subplot carries three lines (one
    per architecture); the composition subplot (bottom-right) uses
    the per-architecture canonical composition column from
    COMPOSITION_BY_ARCH.
    """
    if pairs.empty:
        fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 3.0), constrained_layout=True)
        ax.text(0.5, 0.5, "no paired cells", ha="center", va="center")
        fig.savefig(out_pdf)
        fig.savefig(out_png)
        plt.close(fig)
        return

    panels = [
        ("MSDC",        "MSDC",         "msdc",       "solid"),
        ("CVD",         "CVD",          "cvd",        "dashed"),
        ("DHO",         "DHO",          "dho",        "dotted"),
        ("composition", None,           None,         "dashdot"),
    ]
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 7.0), constrained_layout=True)
    # 4-row gridspec: vanilla / spacer / mech-top / mech-bottom.
    # The thin spacer row holds the "Per-mechanism Δ ..." subtitle
    # so it does not collide with the per-panel "MSDC" / "CVD"
    # titles that sit just above mech-top.
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        height_ratios=[1.6, 0.45, 2.4, 2.4],
    )
    vanilla_ax = fig.add_subplot(gs[0, :])
    mech_ax_tl = fig.add_subplot(gs[2, 0])
    mech_ax_tr = fig.add_subplot(gs[2, 1], sharey=mech_ax_tl)
    mech_ax_bl = fig.add_subplot(gs[3, 0], sharey=mech_ax_tl)
    mech_ax_br = fig.add_subplot(gs[3, 1], sharey=mech_ax_tl)
    axes_flat = [mech_ax_tl, mech_ax_tr, mech_ax_bl, mech_ax_br]
    delta_pool = pd.concat([pairs["delta_7M"], pairs["delta_14M"]]).dropna()
    ymin = float(delta_pool.min()) - 0.005
    ymax = max(0.005, float(delta_pool.max()) + 0.003)
    x_left, x_right = 0.0, 1.0

    vanilla_baselines = _vanilla_pairs()

    # ----- Vanilla absolute-CER subplot -----
    clean_spines(vanilla_ax)
    vanilla_ax.set_axisbelow(True)
    for arch in ARCH_ORDER:
        if arch not in vanilla_baselines:
            continue
        v7, v14 = vanilla_baselines[arch]
        color = ARCH_COLOR[arch]
        marker = MECH_MARKER["vanilla"]
        vanilla_ax.plot(
            [x_left, x_right], [v7, v14],
            color=color, linewidth=1.4, alpha=0.95, linestyle="solid",
            zorder=2,
        )
        _scatter(vanilla_ax, x_left, v7, marker, color)
        _scatter(vanilla_ax, x_right, v14, marker, color)
        dv = v14 - v7
        tag = "POS" if dv < 0 else ("NEG" if dv > 0 else "FLAT")
        vanilla_ax.annotate(
            f"{ARCH_LABEL[arch]}  {v7:.3f} → {v14:.3f}   ({tag} {dv:+.3f})",
            xy=(x_right, v14),
            xytext=(8, 0), textcoords="offset points",
            fontsize=7, color=color, ha="left", va="center",
        )
    vanilla_ax.set_xticks([x_left, x_right])
    vanilla_ax.set_xticklabels(["7 M", "14 M"])
    vanilla_ax.set_xlim(x_left - 0.18, x_right + 0.95)
    vanilla_ax.set_ylabel("Vanilla test CER")
    vanilla_ax.set_title("Vanilla baseline (absolute test CER)", fontsize=9)

    # ----- Mechanism Delta subplots -----
    for ax, (panel_title, mech_column, mech_marker_key, ls) in zip(axes_flat, panels):
        clean_spines(ax)
        ax.set_axisbelow(True)
        for arch in ARCH_ORDER:
            if mech_column is None:
                # Composition subplot: per-architecture column.
                column = COMPOSITION_BY_ARCH[arch]
                marker_key = _composition_marker_key(arch)
            else:
                column = mech_column
                marker_key = mech_marker_key
            row = pairs[(pairs["arch"] == arch) & (pairs["column"] == column)]
            if row.empty:
                continue
            r = row.iloc[0]
            color = ARCH_COLOR[arch]
            marker = MECH_MARKER.get(marker_key, "o")
            ax.plot(
                [x_left, x_right], [r["delta_7M"], r["delta_14M"]],
                color=color, linewidth=1.2, alpha=0.95,
                linestyle=ls, zorder=2,
            )
            _scatter(ax, x_left, r["delta_7M"], marker, color)
            _scatter(ax, x_right, r["delta_14M"], marker, color)
            # Endpoint label = architecture name (one per line per
            # subplot, so there is no inter-architecture overlap).
            ax.annotate(
                ARCH_LABEL[arch],
                xy=(x_right, r["delta_14M"]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=7, color=color, ha="left", va="center",
            )
            if panel_title == "composition":
                # Annotate the composition subplot lines with the
                # specific composition column so the reader can tell
                # CVD x MSDC apart from DHO x MSDC.
                ax.annotate(
                    column,
                    xy=(x_right, r["delta_14M"]),
                    xytext=(6, -10), textcoords="offset points",
                    fontsize=6.0, color=color, ha="left", va="center",
                    alpha=0.85,
                )
        ax.axhline(0.0, color=SPINE_COLOR, linewidth=0.5, zorder=1)
        ax.set_xticks([x_left, x_right])
        ax.set_xticklabels(["7 M", "14 M"])
        ax.set_xlim(x_left - 0.18, x_right + 0.55)
        ax.set_ylim(ymin, ymax)
        ax.set_title(panel_title)

    mech_ax_tl.set_ylabel(r"$\Delta$ test CER vs vanilla")
    mech_ax_bl.set_ylabel(r"$\Delta$ test CER vs vanilla")

    # All three titles share the same horizontal centre — the
    # midpoint of the panel grid. suptitle's x is overridden to
    # match the gridspec centre (it would otherwise centre on the
    # full figure, which is wider than the gridspec). The vanilla
    # subtitle uses ax.set_title (auto-centred on the axis, which
    # spans the same gridspec columns).
    fig.canvas.draw()
    sub_x = (mech_ax_tl.get_position().x0 + mech_ax_tr.get_position().x1) / 2.0
    fig.suptitle("Cross-scale persistence (causal)", x=sub_x)
    # Per-mechanism sub-section title in the spacer row between
    # the vanilla panel and the 2x2 mech grid. Positioned in the
    # lower portion of the spacer (closer to the mech grid) so it
    # reads as the heading of the grid below it.
    sub_y = (
        0.30 * vanilla_ax.get_position().y0
        + 0.70 * mech_ax_tl.get_position().y1
    )
    fig.text(
        sub_x, sub_y,
        r"Per-mechanism $\Delta$ test CER",
        ha="center", va="center",
        fontsize=9,
    )
    # Architecture legend (mechanism is the panel title, so only
    # architectures need a swatch).
    arch_handles = [
        plt.Line2D([0], [0], marker="s", color="white",
                    markerfacecolor=ARCH_COLOR[a], markeredgecolor="white",
                    markersize=8, label=ARCH_LABEL[a])
        for a in ARCH_ORDER
    ]
    fig.legend(
        handles=arch_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False, fontsize=7,
    )
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_typography()
    df = collect()
    df = select_canonical(df)
    df = attach_deltas(df)
    pairs = build_pairs(df)
    # Drop the vanilla rows from the slope axis.
    pairs_for_render = pairs[pairs["column"] != "vanilla"].copy()
    print("[F5] paired cells (Δ-axis, causal):")
    print(
        pairs_for_render[
            ["arch", "column", "cell_7M", "cell_14M",
             "delta_7M", "delta_14M", "delta_growth"]
        ].to_string(index=False)
    )
    pairs.to_csv(FIG_DIR / f"{STEM}_data.csv", index=False)
    render(pairs_for_render, FIG_DIR / f"{STEM}.pdf", FIG_DIR / f"{STEM}.png")
    # Vanilla scaling summary.
    for arch, (v7, v14) in _vanilla_pairs().items():
        print(f"[F5] {arch} vanilla: 7M {v7:.4f} -> 14M {v14:.4f} (delta {v14 - v7:+.4f})")


if __name__ == "__main__":
    main()
