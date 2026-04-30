"""F12 four-cell engagement classifier.

Combines parameter-mobility extractions from F8 (MSDC alpha), F9
(DHO eta), F10 (CVD tau) with the matching Delta test CER values
read from each cell's `results.json::test.cer`. Each reported single-
mechanism cell is classified into one of four buckets:

  (a) engaged-and-helpful — mechanism's free parameters move under
       SGD AND the cell delivers a helpful Delta test CER
  (b) engaged-without-gain — parameters move but no CER improvement
       (the engaged-null catalogue)
  (c) fail-to-engage — parameters frozen, no movement
  (d) residual / other — anything that does not fit

Engagement criteria (per Master_Plan §13 plus mechanism-specific
init knowledge from the source files):
  * MSDC: alpha at d in {2, 4, 8} differs from its small init by
    more than 0.10 in any layer (init from
    mechanisms/conv_shift.py:117-123, alphas init small near 0).
  * CVD: |trained tau - softplus(0)| = |tau - 0.693| > 0.3 in any
    head/layer (raw temperature initialised to zero per
    rwkv6_time_mix.py:681 / mamba2_block.py:362 /
    linear_attn_causal.py:172 / linear_attn_lion.py:122).
  * DHO: max |eta| > 0.1 across heads/blocks/layers (eta initialised
    to zeros per rwkv6_time_mix.py:835 / mamba2_rse.py:154 /
    linear_attn_rse.py:122).

Helpfulness threshold: Delta test CER < -0.005 (helpful), |Delta|
≤ 0.005 (marginal / null), Delta > +0.005 (harmful). Helpful
cells go to bucket (a); marginal and harmful cells with engagement
go to bucket (b).

The chips inside each quadrant are colour-coded by architecture
(ARCH_COLOR) and labelled with a short cell identifier in the
form "<arch> × <mech> (<mode>, <scale>, <dataset>)".

Outputs:
  * F12_engagement_classifier.{pdf,png}
  * F12_engagement_classifier_data.csv
  * F12_engagement_classifier_script.py
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
STEM = "F12_engagement_classifier"
SOURCE_DIR = REPO / "experiments" / "final" / "outputs"
SOURCE_REF = "origin/main"

sys.path.insert(0, str(FIG_DIR))
from _style import (  # noqa: E402
    ARCH_COLOR,
    PAGE_WIDTH_IN,
    SPINE_COLOR,
    TEXT_COLOR,
    apply_typography,
    clean_spines,
)
from _extract_mechanism_params import extract_for_cell  # noqa: E402

ARCH_LABEL = {"rwkv6": "RWKV-6", "mamba2": "Mamba-2", "linear_attn": "LA"}
ARCH_ORDER = ["rwkv6", "mamba2", "linear_attn"]

HELPFUL_THRESHOLD = -0.005

# Engagement thresholds per mechanism.
ENGAGEMENT_MSDC_THRESHOLD = 0.10        # max alpha at d>0 above this
CVD_INIT = 0.6931472                     # softplus(0) = ln(2)
ENGAGEMENT_CVD_THRESHOLD = 0.30
ENGAGEMENT_DHO_THRESHOLD = 0.10

# (arch, mode, scale, dataset, mechanism, cell_dir, vanilla_cell_dir)
# Listed as a single flat catalogue covering every reported single-
# mechanism cell across the F8 / F9 / F10 input set.
CELLS: list[dict] = [
    # RWKV-6
    dict(arch="rwkv6", mode="causal", scale="7M", dataset="LS", mech="MSDC",
         cell="7m_rwkv6_causal_multidil_v2_seed42",
         vanilla="7m_rwkv6_causal_vanilla_seed42"),
    dict(arch="rwkv6", mode="causal", scale="7M", dataset="LS", mech="CVD",
         cell="7m_rwkv6_causal_lucid_chunked_seed42",
         vanilla="7m_rwkv6_causal_vanilla_seed42"),
    dict(arch="rwkv6", mode="causal", scale="7M", dataset="LS", mech="DHO",
         cell="7m_rwkv6_causal_rse_depth_viscosity_seed42",
         vanilla="7m_rwkv6_causal_vanilla_seed42"),
    dict(arch="rwkv6", mode="causal", scale="14M", dataset="LS", mech="MSDC",
         cell="14m_rwkv6_causal_multidil_v2_seed42",
         vanilla="14m_rwkv6_causal_vanilla_seed42"),
    dict(arch="rwkv6", mode="causal", scale="14M", dataset="LS", mech="CVD",
         cell="14m_rwkv6_causal_lucid_chunked_seed42",
         vanilla="14m_rwkv6_causal_vanilla_seed42"),
    dict(arch="rwkv6", mode="causal", scale="14M", dataset="LS", mech="DHO",
         cell="14m_rwkv6_causal_rse_strong_viscosity_seed42",
         vanilla="14m_rwkv6_causal_vanilla_seed42"),
    dict(arch="rwkv6", mode="LION-S", scale="7M", dataset="LS", mech="MSDC",
         cell="7m_rwkv6_lion_multidil_v2_seed42",
         vanilla="7m_rwkv6_lion_vanilla_seed42"),
    dict(arch="rwkv6", mode="LION-S", scale="7M", dataset="LS", mech="CVD",
         cell="7m_rwkv6_lion_lucid_chunked_seed42",
         vanilla="7m_rwkv6_lion_vanilla_seed42"),
    dict(arch="rwkv6", mode="LION-S", scale="7M", dataset="LS", mech="DHO",
         cell="7m_rwkv6_lion_rse_depth_viscosity_seed42",
         vanilla="7m_rwkv6_lion_vanilla_seed42"),
    dict(arch="rwkv6", mode="causal", scale="7M", dataset="CV", mech="MSDC",
         cell="cv_pilot_rwkv6_multidil_v2_seed42",
         vanilla="cv_pilot_rwkv6_seed42"),
    dict(arch="rwkv6", mode="causal", scale="7M", dataset="CV", mech="CVD",
         cell="cv_pilot_rwkv6_lucid_chunked_seed42",
         vanilla="cv_pilot_rwkv6_seed42"),
    dict(arch="rwkv6", mode="causal", scale="7M", dataset="CV", mech="DHO",
         cell="cv_pilot_rwkv6_rse_depth_viscosity_seed42",
         vanilla="cv_pilot_rwkv6_seed42"),
    # Mamba-2
    dict(arch="mamba2", mode="causal", scale="7M", dataset="LS", mech="MSDC",
         cell="7m_mamba2_causal_multidil_v2_seed42",
         vanilla="7m_mamba2_causal_vanilla_seed42"),
    dict(arch="mamba2", mode="causal", scale="7M", dataset="LS", mech="CVD",
         cell="7m_mamba2_causal_lucid_c_seed42",
         vanilla="7m_mamba2_causal_vanilla_seed42"),
    dict(arch="mamba2", mode="causal", scale="7M", dataset="LS", mech="DHO",
         cell="7m_mamba2_causal_rse_strong_viscosity_seed42",
         vanilla="7m_mamba2_causal_vanilla_seed42"),
    dict(arch="mamba2", mode="causal", scale="14M", dataset="LS", mech="MSDC",
         cell="14m_mamba2_causal_multidil_v2_seed42",
         vanilla="14m_mamba2_causal_vanilla_seed42"),
    dict(arch="mamba2", mode="causal", scale="14M", dataset="LS", mech="CVD",
         cell="14m_mamba2_causal_lucid_c_seed42",
         vanilla="14m_mamba2_causal_vanilla_seed42"),
    dict(arch="mamba2", mode="causal", scale="14M", dataset="LS", mech="DHO",
         cell="14m_mamba2_causal_rse_depth_viscosity_seed42",
         vanilla="14m_mamba2_causal_vanilla_seed42"),
    dict(arch="mamba2", mode="LION-S", scale="7M", dataset="LS", mech="MSDC",
         cell="7m_mamba2_lion_multidil_v2_seed42",
         vanilla="7m_mamba2_lion_vanilla_seed42"),
    dict(arch="mamba2", mode="LION-S", scale="7M", dataset="LS", mech="CVD",
         cell="7m_mamba2_lion_lucid_c_seed42",
         vanilla="7m_mamba2_lion_vanilla_seed42"),
    dict(arch="mamba2", mode="LION-S", scale="7M", dataset="LS", mech="DHO",
         cell="7m_mamba2_lion_rse_depth_viscosity_seed42",
         vanilla="7m_mamba2_lion_vanilla_seed42"),
    dict(arch="mamba2", mode="causal", scale="7M", dataset="CV", mech="MSDC",
         cell="cv_pilot_mamba2_multidil_v2_seed42",
         vanilla="cv_pilot_mamba2_seed42"),
    dict(arch="mamba2", mode="causal", scale="7M", dataset="CV", mech="CVD",
         cell="cv_pilot_mamba2_lucid_c_seed42",
         vanilla="cv_pilot_mamba2_seed42"),
    dict(arch="mamba2", mode="causal", scale="7M", dataset="CV", mech="DHO",
         cell="cv_pilot_mamba2_rse_depth_viscosity_seed42",
         vanilla="cv_pilot_mamba2_seed42"),
    # LA
    dict(arch="linear_attn", mode="causal", scale="7M", dataset="LS", mech="MSDC",
         cell="7m_linear_attn_causal_multidil_v2_seed42",
         vanilla="7m_linear_attn_causal_vanilla_seed42"),
    dict(arch="linear_attn", mode="causal", scale="7M", dataset="LS", mech="CVD",
         cell="7m_linear_attn_causal_lucid_seed42",
         vanilla="7m_linear_attn_causal_vanilla_seed42"),
    dict(arch="linear_attn", mode="causal", scale="7M", dataset="LS", mech="DHO",
         cell="7m_linear_attn_causal_rse_strong_viscosity_seed42",
         vanilla="7m_linear_attn_causal_vanilla_seed42"),
    dict(arch="linear_attn", mode="causal", scale="14M", dataset="LS", mech="MSDC",
         cell="14m_linear_attn_causal_multidil_v2_seed42",
         vanilla="14m_linear_attn_causal_vanilla_seed42"),
    dict(arch="linear_attn", mode="causal", scale="14M", dataset="LS", mech="CVD",
         cell="14m_linear_attn_causal_lucid_seed42",
         vanilla="14m_linear_attn_causal_vanilla_seed42"),
    dict(arch="linear_attn", mode="causal", scale="14M", dataset="LS", mech="DHO",
         cell="14m_linear_attn_causal_rse_strong_viscosity_seed42",
         vanilla="14m_linear_attn_causal_vanilla_seed42"),
    dict(arch="linear_attn", mode="LION-LIT", scale="7M", dataset="LS", mech="MSDC",
         cell="7m_linear_attn_lion_multidil_v2_seed42",
         vanilla="7m_linear_attn_lion_vanilla_seed42"),
    dict(arch="linear_attn", mode="LION-LIT", scale="7M", dataset="LS", mech="CVD",
         cell="7m_linear_attn_lion_lucid_seed42",
         vanilla="7m_linear_attn_lion_vanilla_seed42"),
    dict(arch="linear_attn", mode="LION-LIT", scale="7M", dataset="LS", mech="DHO",
         cell="7m_linear_attn_lion_rse_depth_viscosity_seed42",
         vanilla="7m_linear_attn_lion_vanilla_seed42"),
    dict(arch="linear_attn", mode="LION-S", scale="7M", dataset="LS", mech="MSDC",
         cell="7m_linear_attn_lion_s_multidil_v2_seed42",
         vanilla="7m_linear_attn_lion_s_vanilla_seed42"),
    dict(arch="linear_attn", mode="LION-S", scale="7M", dataset="LS", mech="CVD",
         cell="7m_linear_attn_lion_s_lucid_seed42",
         vanilla="7m_linear_attn_lion_s_vanilla_seed42"),
    dict(arch="linear_attn", mode="LION-S", scale="7M", dataset="LS", mech="DHO",
         cell="7m_linear_attn_lion_s_rse_depth_viscosity_seed42",
         vanilla="7m_linear_attn_lion_s_vanilla_seed42"),
    dict(arch="linear_attn", mode="causal", scale="7M", dataset="CV", mech="MSDC",
         cell="cv_pilot_linear_attn_multidil_v2_seed42",
         vanilla="cv_pilot_linear_attn_seed42"),
    dict(arch="linear_attn", mode="causal", scale="7M", dataset="CV", mech="CVD",
         cell="cv_pilot_linear_attn_lucid_seed42",
         vanilla="cv_pilot_linear_attn_seed42"),
    dict(arch="linear_attn", mode="causal", scale="7M", dataset="CV", mech="DHO",
         cell="cv_pilot_linear_attn_rse_strong_viscosity_seed42",
         vanilla="cv_pilot_linear_attn_seed42"),
]


def git_show(path: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{SOURCE_REF}:{path}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout


def load_test_cer_via_git(rel_cell: str) -> float | None:
    try:
        res = json.loads(git_show(f"experiments/final/outputs/{rel_cell}/results.json"))
        return float(res["test"]["cer"])
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def engaged_msdc(extracted: dict | None) -> tuple[bool, float | None]:
    if extracted is None or "alpha" not in extracted:
        return False, None
    alpha = extracted["alpha"]                # (n_layers, n_dilations)
    if alpha.shape[1] < 2:
        return False, None
    # max α at d>0 across layers, normalised against init noise.
    metric = float(np.max(np.abs(alpha[:, 1:])))
    return metric > ENGAGEMENT_MSDC_THRESHOLD, metric


def engaged_cvd(extracted: dict | None) -> tuple[bool, float | None]:
    if extracted is None or "tau" not in extracted:
        return False, None
    tau = extracted["tau"]                    # (n_layers, n_heads)
    metric = float(np.max(np.abs(tau - CVD_INIT)))
    return metric > ENGAGEMENT_CVD_THRESHOLD, metric


def engaged_dho(extracted: dict | None) -> tuple[bool, float | None]:
    if extracted is None or "eta" not in extracted:
        return False, None
    eta = extracted["eta"]                    # (n_layers, n_heads, n_blocks)
    metric = float(np.max(np.abs(eta)))
    return metric > ENGAGEMENT_DHO_THRESHOLD, metric


ENGAGEMENT_FN = {"MSDC": engaged_msdc, "CVD": engaged_cvd, "DHO": engaged_dho}


def collect_classified() -> pd.DataFrame:
    rows: list[dict] = []
    for entry in CELLS:
        cell_dir = SOURCE_DIR / entry["cell"]
        if not cell_dir.exists() or not (cell_dir / "best_model.pt").exists():
            rows.append({**entry, "test_cer": None, "vanilla_cer": None,
                         "delta": None, "engaged": None,
                         "engagement_metric": None, "bucket": "missing"})
            continue
        # Try to load test_cer locally first; fall back to git show.
        try:
            with (cell_dir / "results.json").open() as f:
                test_cer = float(json.load(f)["test"]["cer"])
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            test_cer = load_test_cer_via_git(entry["cell"])
        vanilla_dir = SOURCE_DIR / entry["vanilla"]
        try:
            with (vanilla_dir / "results.json").open() as f:
                vanilla_cer = float(json.load(f)["test"]["cer"])
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            vanilla_cer = load_test_cer_via_git(entry["vanilla"])
        if test_cer is None or vanilla_cer is None:
            rows.append({**entry, "test_cer": test_cer, "vanilla_cer": vanilla_cer,
                         "delta": None, "engaged": None,
                         "engagement_metric": None, "bucket": "missing"})
            continue
        delta = test_cer - vanilla_cer
        extracted = extract_for_cell(cell_dir)
        engaged, metric = ENGAGEMENT_FN[entry["mech"]](extracted)
        helpful = delta < HELPFUL_THRESHOLD
        if engaged is None:
            bucket = "missing"
        elif engaged and helpful:
            bucket = "engaged-and-helpful"
        elif engaged and not helpful:
            bucket = "engaged-without-gain"
        elif not engaged and not helpful:
            bucket = "fail-to-engage"
        else:
            bucket = "residual"
        rows.append({
            **entry, "test_cer": test_cer, "vanilla_cer": vanilla_cer,
            "delta": delta, "engaged": engaged,
            "engagement_metric": metric, "helpful": helpful,
            "bucket": bucket,
        })
    return pd.DataFrame(rows)


BUCKET_LABELS = {
    "engaged-and-helpful":  "engaged-and-helpful",
    "engaged-without-gain": "engaged-without-gain\n(engaged-null catalogue)",
    "fail-to-engage":       "fail-to-engage",
    "residual":             "residual / other",
}
BUCKET_BG = {
    "engaged-and-helpful":  "#e8f1f5",   # very pale cerulean
    "engaged-without-gain": "#f5ece1",   # very pale tangerine
    "fail-to-engage":       "#eef0ec",   # very pale willow
    "residual":             "#f1f3f5",   # neutral grey
}
BUCKET_ORDER = [
    "engaged-and-helpful",
    "engaged-without-gain",
    "fail-to-engage",
    "residual",
]


def render(df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.6), constrained_layout=True)
    bucket_axes = {
        "engaged-and-helpful":  axes[0, 0],
        "engaged-without-gain": axes[0, 1],
        "fail-to-engage":       axes[1, 0],
        "residual":             axes[1, 1],
    }
    arch_short = {"rwkv6": "R6", "mamba2": "M2", "linear_attn": "LA"}
    mode_short = {
        "causal": "c", "LION-S": "S", "LION-LIT": "LIT",
    }
    for bucket, ax in bucket_axes.items():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(SPINE_COLOR)
            spine.set_linewidth(0.6)
        ax.set_facecolor(BUCKET_BG[bucket])
        ax.set_title(BUCKET_LABELS[bucket], fontsize=9, pad=6)

        sub = df[df["bucket"] == bucket].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "(no cells)", ha="center", va="center",
                    fontsize=9, color=SPINE_COLOR, style="italic",
                    transform=ax.transAxes)
            continue
        # Sort by architecture, mechanism, scale, mode, dataset for
        # deterministic ordering; also bring helpful Δ first.
        sub = sub.sort_values(
            by=["arch", "mech", "scale", "mode", "dataset"]
        )
        n = len(sub)
        # Pack chips into 2 columns when the bucket exceeds 10 cells,
        # otherwise single column. Each chip is one line of text
        # with a small architecture-coloured square marker.
        n_cols = 2 if n > 10 else 1
        n_rows = int(np.ceil(n / n_cols))
        line_h = min(0.072, 0.86 / max(1, n_rows))
        col_w = 0.96 / n_cols
        x_left = 0.02
        y_top = 0.94
        for i, (_, r) in enumerate(sub.iterrows()):
            row_i = i % n_rows
            col_i = i // n_rows
            x = x_left + col_i * col_w
            y = y_top - (row_i + 0.5) * line_h
            color = ARCH_COLOR[r["arch"]]
            # Small filled square as architecture marker.
            marker_size = min(0.022, line_h * 0.55)
            rect = mpatches.Rectangle(
                (x, y - marker_size / 2),
                marker_size, marker_size,
                facecolor=color, edgecolor="none",
            )
            ax.add_patch(rect)
            # Chip label: short identifier + Δ.
            label = (
                f"{arch_short[r['arch']]} × {r['mech']:<4} "
                f"({mode_short.get(r['mode'], r['mode'])}-{r['scale']}, {r['dataset']})  "
                f"Δ {r['delta']:+.3f}"
            )
            ax.text(
                x + marker_size + 0.012,
                y,
                label,
                ha="left", va="center",
                fontsize=6.5, color=TEXT_COLOR,
                family="monospace",
            )

    fig.suptitle("Engagement classifier: every reported single-mechanism cell")

    arch_handles = [
        mpatches.Patch(facecolor=ARCH_COLOR[a], edgecolor="white",
                       label=ARCH_LABEL[a])
        for a in ARCH_ORDER
    ]
    fig.legend(handles=arch_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02),
               ncol=3, frameon=False, fontsize=8)

    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_typography()
    df = collect_classified()
    print("[F12] classification:")
    summary = df.groupby("bucket").size().to_dict()
    for b in BUCKET_ORDER + ["missing"]:
        print(f"  {b}: {summary.get(b, 0)}")
    print("\n[F12] per-cell:")
    print(df[[
        "arch", "mech", "mode", "scale", "dataset",
        "delta", "engaged", "engagement_metric", "bucket",
    ]].to_string(index=False))
    df.to_csv(FIG_DIR / f"{STEM}_data.csv", index=False)
    render(df, FIG_DIR / f"{STEM}.pdf", FIG_DIR / f"{STEM}.png")


if __name__ == "__main__":
    main()
