"""F6 composition saturation (7M causal).

Three stacked panels (RWKV-6 / Mamba-2 / LA), each a stepped bar chart
showing the trajectory: vanilla -> best single mechanism -> pairwise
composition. y-axis is absolute test CER per panel; the heterogeneous
y-axes reflect the order-of-magnitude differences in vanilla baselines
(LA 0.188 vs RWKV-6 0.105 vs Mamba-2 0.104).

Per-architecture trajectory:
  RWKV-6:  vanilla -> MSDC -> P7 (LUCID x MSDC)
  Mamba-2: vanilla -> MSDC -> CVD x MSDC (lucid_c x multidil)
  LA:      vanilla -> DHO  -> DHO x MSDC (rse x multidil)

The "best single mechanism" per architecture is chosen by largest
helpful Δ vs vanilla at 7M causal: MSDC for RWKV-6 (-0.026) and
Mamba-2 (-0.021), DHO for LA (-0.068, the BREAK).

Δ between consecutive bars is annotated by a connecting line / arrow
in SPINE_COLOR. The triple composition (P8) is not in the figure
because the 50-ep P8 cell is not on `origin/main`; the same-axis
composition-saturation observation belongs in the chapter prose with
reference to Master_Plan §14.

Outputs:
  * F6_composition_saturation.{pdf,png}    main-text figure
  * F6_composition_saturation_data.csv     bar-level data
  * F6_composition_saturation_script.py    this file
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/tmp/Master_Thesis_Exps")
FIG_DIR = REPO / "Master_Thesis" / "figures" / "chapter5"
STEM = "F6_composition_saturation"
SOURCE_REF = "origin/main"
SOURCE_DIR = "experiments/final/outputs"

sys.path.insert(0, str(FIG_DIR))
from _style import (  # noqa: E402
    ARCH_COLOR,
    PAGE_WIDTH_IN,
    SPINE_COLOR,
    TEXT_COLOR,
    apply_typography,
    clean_spines,
)

ARCH_LABEL = {"rwkv6": "RWKV-6", "mamba2": "Mamba-2", "linear_attn": "LA"}

# Each panel: list of (bar_label, cell_directory_or_special, projected)
# where `cell_directory_or_special` either names the on-disk cell or
# is the literal "P8_PROJECTED" sentinel for the only projected bar.
PANELS = {
    "rwkv6": [
        ("vanilla",                  "7m_rwkv6_causal_vanilla_seed42",      False),
        ("+ MSDC",                   "7m_rwkv6_causal_multidil_v2_seed42",  False),
        ("+ CVD × MSDC",             "7m_rwkv6_causal_p7_seed42",           False),
    ],
    "mamba2": [
        ("vanilla",     "7m_mamba2_causal_vanilla_seed42",         False),
        ("+ MSDC",      "7m_mamba2_causal_multidil_v2_seed42",     False),
        ("+ CVD × MSDC", "7m_mamba2_causal_lucid_c_x_multidil_v2_seed42", False),
    ],
    "linear_attn": [
        ("vanilla",     "7m_linear_attn_causal_vanilla_seed42",       False),
        ("+ DHO",       "7m_linear_attn_causal_rse_strong_viscosity_seed42", False),
        ("+ DHO × MSDC", "7m_linear_attn_causal_rse_x_multidil_v2_seed42", False),
    ],
}


def git_show(path: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{SOURCE_REF}:{path}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout


def load_test_cer(cell_dir: str) -> float | None:
    try:
        results = json.loads(git_show(f"{SOURCE_DIR}/{cell_dir}/results.json"))
        return float(results["test"]["cer"])
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def collect() -> pd.DataFrame:
    rows: list[dict] = []
    for arch, plan in PANELS.items():
        for idx, (label, target, is_projected) in enumerate(plan):
            cer = load_test_cer(target)
            rows.append({
                "arch": arch, "bar_index": idx, "label": label,
                "cell": target, "test_cer": cer, "projected": is_projected,
            })
    return pd.DataFrame(rows)


def render(df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    archs = ["rwkv6", "mamba2", "linear_attn"]
    fig, axes = plt.subplots(
        3, 1, figsize=(PAGE_WIDTH_IN, 5.4), constrained_layout=True,
    )
    for ax, arch in zip(axes, archs):
        clean_spines(ax)
        ax.set_axisbelow(True)
        sub = df[df["arch"] == arch].sort_values("bar_index").reset_index(drop=True)
        if sub["test_cer"].isna().any():
            ax.text(0.5, 0.5, "missing data", ha="center", va="center",
                     transform=ax.transAxes)
            ax.set_title(ARCH_LABEL[arch])
            continue
        xs = np.arange(len(sub))
        cers = sub["test_cer"].values.astype(float)
        color = ARCH_COLOR[arch]
        for i, (x, cer, projected) in enumerate(zip(xs, cers, sub["projected"])):
            hatch = "////" if projected else None
            edge = "white"
            ax.bar(x, cer, width=0.55, color=color, hatch=hatch,
                   edgecolor=edge, linewidth=0.5, zorder=2)
            ax.text(x, cer + (cers.max() - cers.min()) * 0.04,
                    f"{cer:.4f}", ha="center", va="bottom",
                    fontsize=7, color=TEXT_COLOR)
        # Δ arrows between consecutive bars. The arrow connects the
        # tops of consecutive bars; the Δ-value label sits above the
        # arrow midpoint (with a small offset in offset-point
        # coordinates) so it does not overlap the arrow line itself.
        # A few Δ values get explicit precision overrides (`-0.0031`
        # for the Mamba-2 MSDC -> CVD x MSDC step) to match the
        # chapter's reported precision.
        delta_label_overrides = {
            ("mamba2", 1, 2): "-0.0031",
        }
        span = cers.max() - cers.min()
        for i in range(len(sub) - 1):
            x0, x1 = xs[i] + 0.275, xs[i + 1] - 0.275
            y0, y1 = cers[i], cers[i + 1]
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->",
                    color=SPINE_COLOR, lw=0.6,
                    shrinkA=0, shrinkB=0,
                ),
                zorder=3,
            )
            d = y1 - y0
            text = delta_label_overrides.get((arch, i, i + 1), f"{d:+.4f}")
            # The first arrow per panel (vanilla -> best single) is the
            # steepest slope; nudge its label slightly to the right so
            # it does not overlap the arrow line itself.
            x_offset_pt = 10 if i == 0 else 0
            ax.annotate(
                text,
                xy=((x0 + x1) / 2, (y0 + y1) / 2),
                xytext=(x_offset_pt, 4), textcoords="offset points",
                fontsize=6.5, color=SPINE_COLOR,
                ha="center", va="bottom",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(sub["label"].values, fontsize=7.5)
        # Y-axis range: from below the smallest bar to above the largest
        # plus padding for labels.
        span = cers.max() - cers.min()
        pad = max(0.005, span * 0.30)
        ax.set_ylim(cers.min() - pad * 0.6, cers.max() + pad)
        ax.set_ylabel("Test CER")
        ax.set_title(ARCH_LABEL[arch])

    fig.suptitle("Composition saturation: vanilla → single → composition (7M causal)")
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_typography()
    df = collect()
    print("[F6] data:")
    print(df.to_string(index=False))
    df.to_csv(FIG_DIR / f"{STEM}_data.csv", index=False)
    render(df, FIG_DIR / f"{STEM}.pdf", FIG_DIR / f"{STEM}.png")


if __name__ == "__main__":
    main()
