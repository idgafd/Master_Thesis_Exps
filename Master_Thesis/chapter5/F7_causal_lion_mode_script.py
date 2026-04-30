"""F7 causal vs LION mode comparison (7M).

Single panel grouped-bar chart at 5.5 in x 4.0 in. x-axis: 12 groups
(3 architectures x 4 mechanism columns: vanilla, MSDC, CVD, DHO).
Per group: 2 bars (causal, LION) for RWKV-6 and Mamba-2; 3 bars
(causal, LION-LIT, LION-S) for LA. Bars coloured by ARCH_COLOR per
architecture; mode encoded by hatch (causal solid, LION /
LION-LIT `///`, LA LION-S `xxx`).

y-axis: absolute test CER (the LA × DHO LION-LIT BREAK, the
LA × CVD LION-LIT harmful inversion, and the LA × CVD LION-S
recovery are all visible at the matrix-relevant scale).

Caption-text record (script docstring; chapter copies into LaTeX):
  * LION-LIT is the table-default mapping for LA per Afzal et al.
    2025 Table 1: Linear Attention has no native decay in causal
    mode, so the natural LION mapping preserves the no-decay class.
  * LION-S on LA is the controlled experiment that adds per-token
    sigmoid decay not present in the underlying causal LA backbone.
  * The LA × CVD LION-LIT vs LA × CVD LION-S contrast (test
    0.319 vs 0.131) falsifies the reading "LUCID always helps on
    linear-time recurrent backbones" and isolates decay as the
    structural prerequisite for the value-decorrelation mechanism
    in the bidirectional setting.

DHO selection per (arch, mode), per F1: prefer rse_depth_viscosity,
fall back to rse_strong_viscosity; both labelled "DHO". Manual delta
overrides from `_style.py` (Mamba-2 x DHO 7M causal -> -0.003) apply
to the causal column only.

Outputs:
  * F7_causal_lion_mode.{pdf,png}    main-text figure
  * F7_causal_lion_mode_data.csv     bar-level data
  * F7_causal_lion_mode_script.py    this file
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/tmp/Master_Thesis_Exps")
FIG_DIR = REPO / "Master_Thesis" / "figures" / "chapter5"
STEM = "F7_causal_lion_mode"
SOURCE_REF = "origin/main"
SOURCE_DIR = "experiments/final/outputs"

sys.path.insert(0, str(FIG_DIR))
from _style import (  # noqa: E402
    ARCH_COLOR,
    MANUAL_DELTA_OVERRIDES_LS,
    PAGE_WIDTH_IN,
    SPINE_COLOR,
    TEXT_COLOR,
    apply_typography,
    clean_spines,
)

ARCH_LABEL = {"rwkv6": "RWKV-6", "mamba2": "Mamba-2", "linear_attn": "LA"}

# Mechanism column -> on-disk cell directory token preferences (in
# preference order; first match wins).
MECH_TOKENS = {
    "vanilla":  ["vanilla"],
    "MSDC":     ["multidil_v2"],
    "CVD":      ["lucid", "lucid_c", "lucid_chunked"],
    "DHO":      ["rse_depth_viscosity", "rse_strong_viscosity"],
}
MECH_ORDER = ["vanilla", "MSDC", "CVD", "DHO"]

# Mode rows per architecture. Each entry is
#   (mode_label, dir_prefix, hatch).
# RWKV-6 and Mamba-2 LION cells use the standard `_lion_` prefix
# (LION-S by natural mapping per Master_Plan §6); LA has both
# `_lion_` (LION-LIT) and `_lion_s_` (LION-S, controlled).
MODES_BY_ARCH = {
    "rwkv6": [
        ("causal", "7m_rwkv6_causal_",   None),
        ("LION-S", "7m_rwkv6_lion_",     "///"),
    ],
    "mamba2": [
        ("causal", "7m_mamba2_causal_",  None),
        ("LION-S", "7m_mamba2_lion_",    "///"),
    ],
    "linear_attn": [
        ("causal",   "7m_linear_attn_causal_",   None),
        ("LION-LIT", "7m_linear_attn_lion_",     "xxx"),
        ("LION-S",   "7m_linear_attn_lion_s_",   "///"),
    ],
}
ARCH_ORDER = ["rwkv6", "mamba2", "linear_attn"]


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


def find_cell(prefix: str, mech: str, all_dirs: set) -> str | None:
    """Find the on-disk cell for `prefix + token + _seed42` matching
    one of the mechanism's accepted tokens. Returns the directory
    name or None."""
    if mech == "vanilla":
        # Vanilla cells use the bare prefix, e.g.
        # `7m_rwkv6_causal_vanilla_seed42` or
        # `7m_linear_attn_lion_s_vanilla_seed42`.
        target = f"{prefix}vanilla_seed42"
        return target if target in all_dirs else None
    for tok in MECH_TOKENS[mech]:
        target = f"{prefix}{tok}_seed42"
        if target in all_dirs:
            return target
    return None


def collect() -> pd.DataFrame:
    rows: list[dict] = []
    all_dirs = set(list_cell_dirs())
    for arch in ARCH_ORDER:
        for mode_label, prefix, hatch in MODES_BY_ARCH[arch]:
            for mech in MECH_ORDER:
                target = find_cell(prefix, mech, all_dirs)
                if target is None:
                    rows.append({
                        "arch": arch, "mode": mode_label, "mechanism": mech,
                        "dir": "", "test_cer": None, "hatch": hatch,
                    })
                    continue
                cer = load_test_cer(target)
                rows.append({
                    "arch": arch, "mode": mode_label, "mechanism": mech,
                    "dir": target, "test_cer": cer, "hatch": hatch,
                })
    df = pd.DataFrame(rows)
    # Apply manual overrides on causal cells only.
    for (scale_M, arch, mode, column), value in MANUAL_DELTA_OVERRIDES_LS.items():
        if scale_M != 7 or mode != "causal":
            continue
        # Compute the corresponding override test_cer = vanilla + Δ.
        vanilla_row = df[(df["arch"] == arch) & (df["mode"] == "causal") & (df["mechanism"] == "vanilla")]
        if vanilla_row.empty or pd.isna(vanilla_row.iloc[0]["test_cer"]):
            continue
        v = float(vanilla_row.iloc[0]["test_cer"])
        mask = (df["arch"] == arch) & (df["mode"] == "causal") & (df["mechanism"] == column)
        df.loc[mask, "test_cer"] = v + value
    return df


def render(df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 4.0), constrained_layout=True)
    clean_spines(ax)
    ax.set_axisbelow(True)

    # x-positions: 12 groups, each centred at i + 0.5.
    group_centers = []
    group_labels = []
    bar_specs: list[tuple] = []  # (x, height, color, hatch, dir, arch, mode, mech)
    inter_group = 0.35  # space between groups
    bar_w = 0.22
    cursor = 0.0
    for arch in ARCH_ORDER:
        modes = MODES_BY_ARCH[arch]
        n_modes = len(modes)
        for mech in MECH_ORDER:
            # Centre this group at `cursor + (n_modes * bar_w) / 2`.
            group_x0 = cursor
            for k, (mode_label, prefix, hatch) in enumerate(modes):
                x = group_x0 + (k + 0.5) * bar_w
                row = df[(df["arch"] == arch) & (df["mode"] == mode_label) & (df["mechanism"] == mech)]
                if row.empty:
                    continue
                r = row.iloc[0]
                if pd.isna(r["test_cer"]):
                    continue
                bar_specs.append((
                    x, float(r["test_cer"]), ARCH_COLOR[arch], hatch,
                    r["dir"], arch, mode_label, mech,
                ))
            group_centers.append(group_x0 + (n_modes * bar_w) / 2)
            group_labels.append(mech)
            cursor += n_modes * bar_w + inter_group
        # Insert a slightly larger gap between architectures.
        cursor += inter_group * 0.6

    for x, h, color, hatch, *_ in bar_specs:
        ax.bar(
            x, h, width=bar_w, color=color,
            hatch=hatch, edgecolor="white", linewidth=0.6, zorder=2,
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=7.5)
    ax.set_ylabel("Test CER")
    # Headroom above the tallest bar for in-figure annotations.
    max_cer = max(b[1] for b in bar_specs)
    ax.set_ylim(0, max_cer * 1.30)

    # Architecture row labels: positioned just below the suptitle, in
    # ARCH_COLOR, slightly above the panel area. The y-coordinate sits
    # below the annotations on the LION-LIT CVD bar so the two do not
    # overlap.
    arch_x_centers: list[float] = []
    for arch in ARCH_ORDER:
        idxs = [i for i, _ in enumerate(group_labels) if i // 4 == ARCH_ORDER.index(arch)]
        if idxs:
            arch_x_centers.append(np.mean([group_centers[i] for i in idxs]))
    for arch, cx in zip(ARCH_ORDER, arch_x_centers):
        ax.text(
            cx, 1.06, ARCH_LABEL[arch],
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom",
            fontsize=9, color=ARCH_COLOR[arch], weight="bold",
        )

    # Legend: mode hatch-pattern key (architecture is in the per-arch
    # x-axis label above each group).
    legend_handles = [
        mpatches.Patch(facecolor="#adb5bd", edgecolor="white",
                        hatch=None, label="causal"),
        mpatches.Patch(facecolor="#adb5bd", edgecolor="white",
                        hatch="///", label="LION-S"),
        mpatches.Patch(facecolor="#adb5bd", edgecolor="white",
                        hatch="xxx", label="LION-LIT"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right",
        ncol=1, frameon=False, fontsize=7,
    )
    ax.set_xlim(-0.4, cursor)
    fig.suptitle("Causal vs LION mode (7M, absolute test CER)")
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_typography()
    df = collect()
    print("[F7] data:")
    print(df.to_string(index=False))
    df.to_csv(FIG_DIR / f"{STEM}_data.csv", index=False)
    render(df, FIG_DIR / f"{STEM}.pdf", FIG_DIR / f"{STEM}.png")


if __name__ == "__main__":
    main()
