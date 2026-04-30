"""F4 LibriSpeech vs Common Voice scatter.

For every Common Voice pilot cell on disk, pair it with the matching
LibriSpeech 7M causal cell and scatter Delta_CV vs Delta_LibriSpeech.
Diagonal y = x is the reference: points on the diagonal show
consistent transfer; off-diagonal points are task-prior modulated.

Data sources (read-only via `git show origin/main`):
  * `experiments/final/outputs/cv_pilot_<arch>[_<cell>]_seed42/results.json`
  * `experiments/final/outputs/7m_<arch>_causal_<cell>_seed42/results.json`

Selection rules (per F1, applied independently to each dataset):
  * DHO: prefer `rse_depth_viscosity`; fall back to
    `rse_strong_viscosity`; relabel either as the canonical "DHO".
    The independent-per-dataset rule means Mamba-2 reports DHO
    via `rse_depth` on CV (preferred, present) and via the
    senior-reviewer-supplied depth-schedule delta of -0.003 on
    LibriSpeech (depth result not yet committed under its
    canonical directory name on `origin/main`; see
    MANUAL_DELTA_OVERRIDES below). The (Delta_LS, Delta_CV) pair
    on Mamba-2 x DHO is therefore the task-prior-modulation
    anchor reported in the figure (-0.003, -0.023): both helpful,
    but the CV gain is roughly an order of magnitude larger.
  * Skip TRWG, bidir_vim, biwkv, decay_coupled_delta, _5ep.

Outputs:
  * F4_libri_cv_scatter.{pdf,png}
  * F4_libri_cv_scatter_data.csv
  * F4_libri_cv_scatter_script.py
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
STEM = "F4_libri_cv_scatter"
SOURCE_REF = "origin/main"
SOURCE_DIR = "experiments/final/outputs"

sys.path.insert(0, str(FIG_DIR))
from _style import (  # noqa: E402
    ARCH_COLOR,
    GRID_COLOR,
    MANUAL_DELTA_OVERRIDES_CV,
    MANUAL_DELTA_OVERRIDES_LS,
    MECH_MARKER,
    PAGE_WIDTH_IN,
    apply_typography,
    clean_spines,
)

ARCH_LABEL = {"rwkv6": "RWKV-6", "mamba2": "Mamba-2", "linear_attn": "LA"}
ARCH_ORDER = ["rwkv6", "mamba2", "linear_attn"]

# Mechanism column normalisation. Each (cell-string-on-disk) maps to
# (mechanism_class, marker_key).
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
    "lucid_multidil_v2":     ("CVD x MSDC",  "msdc_x_cvd"),
    "lucid_c_multidil_v2":   ("CVD x MSDC",  "msdc_x_cvd"),
    "rse_x_multidil_v2":     ("DHO x MSDC",  "msdc_x_dho"),
}

# DHO selection priority per dataset (depth before strong).
DHO_PRIORITY = ["rse_depth_viscosity", "rse_strong_viscosity"]


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


def parse_cv(name: str) -> tuple[str, str] | None:
    """Returns (arch, cell) for a CV pilot directory, or None."""
    if not name.startswith("cv_pilot_") or not name.endswith("_seed42"):
        return None
    body = name[len("cv_pilot_"):-len("_seed42")]
    for arch in ARCH_ORDER:
        if body == arch:
            return (arch, "vanilla")
        prefix = f"{arch}_"
        if body.startswith(prefix):
            cell = body[len(prefix):]
            return (arch, cell)
    return None


def parse_ls(name: str) -> tuple[str, str] | None:
    """Returns (arch, cell) for a 7M causal LibriSpeech directory."""
    if not name.startswith("7m_") or not name.endswith("_seed42"):
        return None
    body = name[len("7m_"):-len("_seed42")]
    for arch in ARCH_ORDER:
        prefix = f"{arch}_causal_"
        if body.startswith(prefix):
            cell = body[len(prefix):]
            return (arch, cell)
    return None


def build_per_dataset_table(
    dirs: list[str],
    parser,
    dataset_label: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for name in dirs:
        if name.endswith("_5ep") or "_bidir_vim_" in name or "_biwkv_" in name:
            continue
        if name == "rwkv6_decay_coupled_delta_seed42":
            continue
        if "rse_trwg_strong_viscosity" in name:
            continue
        parsed = parser(name)
        if parsed is None:
            continue
        arch, cell = parsed
        if cell not in CELL_TO_COLUMN:
            continue
        cer = load_test_cer(name)
        if cer is None:
            continue
        column, marker = CELL_TO_COLUMN[cell]
        rows.append({
            "dataset": dataset_label,
            "dir": name,
            "arch": arch,
            "cell": cell,
            "column": column,
            "marker": marker,
            "test_cer": cer,
        })
    return pd.DataFrame(rows)


def select_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Apply DHO depth/strong fallback per (dataset, arch). Other
    columns just take the lowest-CER cell when several map to the
    same column.
    """
    keep = []
    for (ds, arch, column), group in df.groupby(["dataset", "arch", "column"]):
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
        .set_index(["dataset", "arch"])["test_cer"]
        .to_dict()
    )
    df = df.copy()
    df["vanilla_test_cer"] = df.apply(
        lambda r: vanilla.get((r["dataset"], r["arch"])), axis=1
    )
    df["delta"] = df["test_cer"] - df["vanilla_test_cer"]
    df["override_applied"] = False
    # F4 only loads 7M causal cells, so the LS override key
    # (scale_M=7, arch, mode="causal", column) reduces to (arch, column)
    # restricted to the LS dataset.
    for (scale_M, arch, mode, column), value in MANUAL_DELTA_OVERRIDES_LS.items():
        if scale_M != 7 or mode != "causal":
            continue
        mask = (
            (df["dataset"] == "LS")
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
    for (arch, column), value in MANUAL_DELTA_OVERRIDES_CV.items():
        mask = (
            (df["dataset"] == "CV")
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


def pair(ls: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    """Inner-join LibriSpeech and CV tables on (arch, column).
    Vanilla rows are excluded from the scatter (Δ = 0 by definition
    on both axes).
    """
    ls = ls[ls["column"] != "vanilla"].rename(
        columns={"test_cer": "ls_test_cer", "delta": "ls_delta",
                  "dir": "ls_dir", "cell": "ls_cell",
                  "vanilla_test_cer": "ls_vanilla"}
    )
    cv = cv[cv["column"] != "vanilla"].rename(
        columns={"test_cer": "cv_test_cer", "delta": "cv_delta",
                  "dir": "cv_dir", "cell": "cv_cell",
                  "vanilla_test_cer": "cv_vanilla"}
    )
    merged = pd.merge(
        ls[["arch", "column", "marker", "ls_dir", "ls_cell",
             "ls_vanilla", "ls_test_cer", "ls_delta"]],
        cv[["arch", "column", "cv_dir", "cv_cell",
             "cv_vanilla", "cv_test_cer", "cv_delta"]],
        on=["arch", "column"],
        how="inner",
    )
    return merged


def render(merged: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 4.4), constrained_layout=True)
    clean_spines(ax)
    if merged.empty:
        ax.text(0.5, 0.5, "no paired cells", ha="center", va="center")
        fig.savefig(out_pdf)
        fig.savefig(out_png)
        plt.close(fig)
        return
    xs = merged["ls_delta"].values
    ys = merged["cv_delta"].values
    margin = 0.01
    xmin = float(min(xs.min(), 0.0)) - margin
    xmax = float(max(xs.max(), 0.0)) + margin
    ymin = float(min(ys.min(), 0.0)) - margin
    ymax = float(max(ys.max(), 0.0)) + margin
    # Diagonal y = x reference.
    diag_lo = min(xmin, ymin)
    diag_hi = max(xmax, ymax)
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color=GRID_COLOR,
             linestyle=":", linewidth=1.0, zorder=1)
    ax.axvline(0.0, color=GRID_COLOR, linestyle=":", linewidth=0.8, zorder=1)
    ax.axhline(0.0, color=GRID_COLOR, linestyle=":", linewidth=0.8, zorder=1)
    # Quadrant labels (cells whose Δ is helpful on both, helpful only
    # on one, etc.). The quadrant is defined by sign of Δ.
    ax.text(xmin + margin * 0.4, ymin + margin * 0.4, "consistent helpful",
             color="#adb5bd", fontsize=7, ha="left", va="bottom")
    ax.text(xmax - margin * 0.4, ymin + margin * 0.4, "CV-only helpful",
             color="#adb5bd", fontsize=7, ha="right", va="bottom")
    ax.text(xmin + margin * 0.4, ymax - margin * 0.4, "LS-only helpful",
             color="#adb5bd", fontsize=7, ha="left", va="top")
    ax.text(xmax - margin * 0.4, ymax - margin * 0.4, "consistent harmful",
             color="#adb5bd", fontsize=7, ha="right", va="top")
    # Plot points.
    for _, r in merged.iterrows():
        marker = MECH_MARKER.get(r["marker"], "o")
        face = ARCH_COLOR[r["arch"]]
        ax.scatter(
            r["ls_delta"], r["cv_delta"],
            s=55, marker=marker, color=face, edgecolor="white",
            linewidth=0.8, zorder=3,
        )
    # Annotate notable points.
    annotate_targets = [
        ("mamba2", "DHO",        "Mamba-2 × DHO"),
        ("linear_attn", "DHO",   "LA × DHO"),
        ("linear_attn", "CVD x MSDC", "LA × CVD×MSDC"),
        ("linear_attn", "DHO x MSDC", "LA × DHO×MSDC"),
    ]
    annotated = set()
    for arch, column, label in annotate_targets:
        match = merged[(merged["arch"] == arch) & (merged["column"] == column)]
        if not match.empty:
            r = match.iloc[0]
            ax.annotate(
                label, xy=(r["ls_delta"], r["cv_delta"]),
                xytext=(8, 8), textcoords="offset points", fontsize=7,
                color="#212529",
                arrowprops=dict(arrowstyle="-", color="#adb5bd", lw=0.6),
            )
            annotated.add((arch, column))
    # Label any other point > 0.015 from the diagonal.
    for _, r in merged.iterrows():
        key = (r["arch"], r["column"])
        if key in annotated:
            continue
        if abs(r["cv_delta"] - r["ls_delta"]) <= 0.015:
            continue
        ax.annotate(
            f"{ARCH_LABEL[r['arch']]} × {r['column']}",
            xy=(r["ls_delta"], r["cv_delta"]),
            xytext=(8, -10), textcoords="offset points", fontsize=7,
            color="#495057",
            arrowprops=dict(arrowstyle="-", color="#dee2e6", lw=0.5),
        )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$\Delta_{\mathrm{LibriSpeech}}$ test CER")
    ax.set_ylabel(r"$\Delta_{\mathrm{CV}}$ test CER")
    ax.set_title("LibriSpeech vs Common Voice cross-distribution scatter (7M causal)")
    # Legend: architecture swatches + mechanism markers, single legend.
    arch_handles = [
        plt.Line2D([0], [0], marker="s", color="white",
                    markerfacecolor=ARCH_COLOR[a], markeredgecolor="white",
                    markersize=8, label=ARCH_LABEL[a])
        for a in ARCH_ORDER
    ]
    mech_keys = [
        ("MSDC",        "msdc"),
        ("CVD",         "cvd"),
        ("DHO",         "dho"),
        ("CVD × MSDC",  "msdc_x_cvd"),
        ("DHO × MSDC",  "msdc_x_dho"),
    ]
    mech_handles = [
        plt.Line2D([0], [0], marker=MECH_MARKER[k], color="#495057",
                    markerfacecolor="#495057", markeredgecolor="white",
                    markersize=7, linestyle="", label=lbl)
        for lbl, k in mech_keys
    ]
    ax.legend(
        handles=arch_handles + mech_handles,
        loc="lower center", bbox_to_anchor=(0.5, -0.30),
        ncol=4, frameon=False, fontsize=7,
    )
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_typography()
    dirs = list_cell_dirs()
    cv_dirs = [d for d in dirs if d.startswith("cv_pilot_")]
    ls_dirs = [d for d in dirs if d.startswith("7m_") and "_causal_" in d]
    print(f"[F4] CV pilot cells: {len(cv_dirs)}")
    print(f"[F4] LS 7M causal cells: {len(ls_dirs)}")
    cv = build_per_dataset_table(cv_dirs, parse_cv, "CV")
    ls = build_per_dataset_table(ls_dirs, parse_ls, "LS")
    cv_canon = select_canonical(cv)
    ls_canon = select_canonical(ls)
    cv_canon = attach_deltas(cv_canon)
    ls_canon = attach_deltas(ls_canon)
    merged = pair(ls_canon, cv_canon)
    print("[F4] merged pairs:")
    print(
        merged[["arch", "column", "ls_cell", "cv_cell",
                "ls_delta", "cv_delta"]].to_string(index=False)
    )
    full = pd.concat([ls_canon.assign(side="LS"), cv_canon.assign(side="CV")])
    full.to_csv(FIG_DIR / f"{STEM}_data.csv", index=False)
    render(merged, FIG_DIR / f"{STEM}.pdf", FIG_DIR / f"{STEM}.png")


if __name__ == "__main__":
    main()
