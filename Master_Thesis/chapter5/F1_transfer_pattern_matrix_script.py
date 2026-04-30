"""F1 transfer-pattern matrix.

Reads test CER for every reported cell of the experimental matrix from
the `experiments/final/outputs/` tree on the `main` branch (read-only via
`git show`) and renders a faceted bar chart (preferred main-text form)
plus heatmap companions. Rows = architecture (with LION decay class
where relevant), columns = mechanism. Three panels: causal-7M,
causal-14M, LION-7M.

Outputs:
  * F1_transfer_pattern_matrix.{pdf,png}        main-text bars, 5.5 in wide
  * F1_transfer_pattern_matrix_heatmap.{pdf,png} combined heatmap (appendix),
                                                global symmetric norm of
                                                +/-0.199 for cross-panel
                                                comparability
  * F1_transfer_pattern_matrix_heatmap_causal_7m.{pdf,png}  per-panel heatmap,
                                                bound = +/-0.088 (local
                                                symmetric norm, this panel only)
  * F1_transfer_pattern_matrix_heatmap_causal_14m.{pdf,png} per-panel heatmap,
                                                bound = +/-0.057 (local
                                                symmetric norm, this panel only)
  * F1_transfer_pattern_matrix_heatmap_lion_7m.{pdf,png}    per-panel heatmap,
                                                bound = +/-0.199 (local
                                                symmetric norm, this panel only;
                                                coincides with the global bound)
  * F1_transfer_pattern_matrix_data.csv         one row per loaded cell

Selection rules (per senior reviewer 2026-04-29):
  * DHO column: prefer the `_rse_depth_viscosity_` cell. Fall back to
    `_rse_strong_viscosity_` when no depth cell exists; treat the
    fallback as the canonical DHO entry (the chapter reports DHO with
    the depth-graded theta-clip schedule).
  * Skip `_rse_trwg_strong_viscosity_` (engaged-null probe, not part
    of the canonical reporting).
  * Skip `_bidir_vim_`, `_biwkv_`, `_5ep`, `cv_pilot_*`,
    `rwkv6_decay_coupled_delta`. The bidirectional `_bidir_vim_` and
    `_biwkv_` cells implement the operator-level bidirectional
    alternative and are intentionally outside the locked
    mechanism-level (LION) matrix.
  * LA x LION exists in two variants: `_lion_*` is LION-LIT
    (table-default); `_lion_s_*` is LION-S (controlled experiment).
    Both are reported as separate rows of the single LION panel.
  * RWKV-6 LION and Mamba-2 LION use `_lion_*` directories which are
    LION-S by natural mapping (Master_Plan §6).

Suggested caption text for the chapter LaTeX (record-only, the figure
file itself does not contain this text):

    Transfer-pattern matrix: $\\Delta$ test CER vs the matching
    (architecture, mode, scale) vanilla baseline. Three panels
    (causal-7M, causal-14M, LION-7M) span the locked $3 \\times 2
    \\times 2$ matrix; the LION panel carries Linear Attention twice,
    once on the table-default LION-LIT substrate and once on the
    controlled LION-S substrate (hatched). Bars below zero are
    helpful; bars above zero are harmful. Of particular note, on the
    no-decay LION-LIT substrate on Linear Attention, the Damped
    Harmonic Oscillator composes with Multi-Scale Depthwise
    Convolution to a $\\Delta$ test CER of -0.199, surpassing the
    single-mechanism BREAK of -0.191. By contrast, Chunked Value
    Decorrelation on the same LION-LIT substrate inverts to a
    harmful $\\Delta$ of +0.024. The asymmetry between the two
    mechanisms on a common substrate is consistent with the reading
    that decay is a structural prerequisite for the value-
    decorrelation operator but not for the block-complex transition
    operator.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/tmp/Master_Thesis_Exps")
FIG_DIR = REPO / "Master_Thesis" / "figures" / "chapter5"
STEM = "F1_transfer_pattern_matrix"
SOURCE_REF = "origin/main"
SOURCE_DIR = "experiments/final/outputs"

# Import the shared chapter-5 style (locked palette 2026-04-29).
sys.path.insert(0, str(FIG_DIR))
from _style import (  # noqa: E402
    ARCH_COLOR,
    DELTA_CMAP,
    LION_HATCH,
    PAGE_WIDTH_IN,
    apply_manual_overrides,
    apply_typography,
    clean_spines,
    delta_norm,
)

# Directory naming convention (locked in STATUS.md):
#   <scale>m_<arch>_<mode>_<cell>_seed42
# where <arch> ∈ {rwkv6, mamba2, linear_attn},
#       <mode> ∈ {causal, lion, lion_s},
#       <cell> ∈ {vanilla, multidil_v2, lucid, lucid_c, lucid_chunked,
#                 rse_strong_viscosity, rse_depth_viscosity, p7,
#                 lucid_x_multidil_v2, rse_x_multidil_v2,
#                 lucid_c_x_multidil_v2, lucid_multidil_v2}.
DIR_RE = re.compile(
    r"^(?P<scale>7m|14m)_(?P<arch>rwkv6|mamba2|linear_attn)"
    r"_(?P<mode>causal|lion_s|lion)_(?P<cell>.+)_seed42$"
)

# Cells dropped from F1 reporting.
CELL_DROP = {"rse_trwg_strong_viscosity"}

# Mechanism-column normalisation (the on-disk directory string is
# irrelevant past this point).
CELL_TO_COLUMN = {
    "vanilla": "vanilla",
    "multidil_v2": "MSDC",
    "lucid": "CVD",
    "lucid_c": "CVD",
    "lucid_chunked": "CVD",
    "rse_strong_viscosity": "DHO",
    "rse_depth_viscosity": "DHO",
    "p7": "CVD x MSDC",
    "lucid_x_multidil_v2": "CVD x MSDC",
    "lucid_c_x_multidil_v2": "CVD x MSDC",
    "lucid_multidil_v2": "CVD x MSDC",
    "rse_x_multidil_v2": "DHO x MSDC",
}

COLUMN_ORDER = ["vanilla", "MSDC", "CVD", "DHO", "CVD x MSDC", "DHO x MSDC"]
ARCH_ORDER = ["rwkv6", "mamba2", "linear_attn"]
ARCH_LABEL = {"rwkv6": "RWKV-6", "mamba2": "Mamba-2", "linear_attn": "LA"}


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
    names = []
    for line in res.stdout.strip().splitlines():
        name = line.rsplit("/", 1)[-1]
        names.append(name)
    return names


def is_excluded(name: str) -> bool:
    if "_bidir_vim_" in name or "_biwkv_" in name:
        return True
    if name.startswith("cv_pilot_"):
        return True
    if name == "rwkv6_decay_coupled_delta_seed42":
        return True
    if name.endswith("_5ep"):
        return True
    return False


def collect_cells() -> pd.DataFrame:
    rows: list[dict] = []
    for name in list_cell_dirs():
        if is_excluded(name):
            continue
        m = DIR_RE.match(name)
        if not m:
            continue
        cell = m.group("cell")
        if cell in CELL_DROP:
            continue
        column = CELL_TO_COLUMN.get(cell)
        if column is None:
            continue
        try:
            results = json.loads(git_show(f"{SOURCE_DIR}/{name}/results.json"))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        try:
            test_cer = float(results["test"]["cer"])
        except (KeyError, ValueError, TypeError):
            continue
        scale = 7 if m.group("scale") == "7m" else 14
        rows.append(
            {
                "dir": name,
                "scale_M": scale,
                "arch": m.group("arch"),
                "mode": m.group("mode"),
                "cell": cell,
                "column": column,
                "test_cer": test_cer,
            }
        )
    return pd.DataFrame(rows)


def apply_dho_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the canonical DHO entry per (arch, mode, scale).

    Prefer `rse_depth_viscosity`; fall back to `rse_strong_viscosity`.
    Re-label the cell field of the kept row to a single canonical
    string `dho_canonical` so the data CSV does not surface either
    on-disk variant name.
    """
    keep = []
    drop_idx: list[int] = []
    dho_rows = df[df["column"] == "DHO"]
    for (arch, mode, scale), group in dho_rows.groupby(["arch", "mode", "scale_M"]):
        depth = group[group["cell"] == "rse_depth_viscosity"]
        if not depth.empty:
            keep.append(depth.index[0])
            drop_idx.extend([i for i in group.index if i != depth.index[0]])
        else:
            strong = group[group["cell"] == "rse_strong_viscosity"]
            if not strong.empty:
                keep.append(strong.index[0])
                drop_idx.extend([i for i in group.index if i != strong.index[0]])
            else:
                drop_idx.extend(list(group.index))
    df = df.drop(index=drop_idx).copy()
    df.loc[df["column"] == "DHO", "cell"] = "dho_canonical"
    return df


def attach_deltas(df: pd.DataFrame) -> pd.DataFrame:
    vanilla = (
        df[df["cell"] == "vanilla"]
        .set_index(["arch", "mode", "scale_M"])["test_cer"]
        .to_dict()
    )
    df = df.copy()
    df["vanilla_test_cer"] = df.apply(
        lambda r: vanilla.get((r["arch"], r["mode"], r["scale_M"])), axis=1
    )
    df["delta"] = df["test_cer"] - df["vanilla_test_cer"]
    return df


# A "row key" labels each row of a panel uniquely. For the LION panel
# we have RWKV-6 (LION-S), Mamba-2 (LION-S), LA (LION-LIT), LA (LION-S).
def row_keys_for_panel(panel: str, present: pd.DataFrame) -> list[tuple]:
    """Return ordered list of (arch, mode, label) row keys present."""
    if panel == "causal":
        order = [(a, "causal", ARCH_LABEL[a]) for a in ARCH_ORDER]
    elif panel == "lion":
        order = [
            ("rwkv6", "lion", "RWKV-6 (LION-S)"),
            ("mamba2", "lion", "Mamba-2 (LION-S)"),
            ("linear_attn", "lion", "LA (LION-LIT)"),
            ("linear_attn", "lion_s", "LA (LION-S)"),
        ]
    else:
        order = []
    keys = []
    for arch, mode, label in order:
        if not present[(present["arch"] == arch) & (present["mode"] == mode)].empty:
            keys.append((arch, mode, label))
    return keys


def render_bars(df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    panels = [("causal", 7, "causal, 7M"), ("causal", 14, "causal, 14M"), ("lion", None, "LION, 7M")]
    # Thesis text width is ~14 cm (PAGE_WIDTH_IN). Stack panels vertically
    # so the per-panel readability survives the column-width constraint.
    fig, axes = plt.subplots(3, 1, figsize=(PAGE_WIDTH_IN, 7.0), constrained_layout=True)
    for ax, (panel, scale, title) in zip(axes, panels):
        clean_spines(ax)
        ax.set_axisbelow(True)
        if panel == "causal":
            sub = df[(df["mode"] == "causal") & (df["scale_M"] == scale)]
            keys = row_keys_for_panel("causal", sub)
        else:
            sub = df[(df["mode"].isin(["lion", "lion_s"])) & (df["scale_M"] == 7)]
            keys = row_keys_for_panel("lion", sub)
        cols_present = [c for c in COLUMN_ORDER if c in sub["column"].unique() and c != "vanilla"]
        n_rows = len(keys)
        n_cols = len(cols_present)
        if n_rows == 0 or n_cols == 0:
            ax.set_axis_off()
            ax.set_title(title)
            continue
        bar_w = 0.85 / n_rows
        for i, (arch, mode, label) in enumerate(keys):
            ys = []
            for c in cols_present:
                cell = sub[(sub["arch"] == arch) & (sub["mode"] == mode) & (sub["column"] == c)]
                if cell.empty:
                    ys.append(np.nan)
                else:
                    best = cell.loc[cell["test_cer"].idxmin()]
                    ys.append(best["delta"])
            xs = np.arange(n_cols) + (i - (n_rows - 1) / 2) * bar_w
            face = ARCH_COLOR[arch]
            # LA appears twice in the LION panel: solid for LION-LIT,
            # hatched for LION-S. Other rows are always solid.
            is_la_lion_s = (panel == "lion" and arch == "linear_attn" and mode == "lion_s")
            hatch = LION_HATCH["lion_s"] if is_la_lion_s else LION_HATCH["lion_lit"]
            edge = "white" if is_la_lion_s else "none"
            ax.bar(
                xs, ys, width=bar_w, color=face, label=label,
                hatch=hatch, edgecolor=edge, linewidth=0.0,
            )
        ax.axhline(0.0, color="#212529", linewidth=0.6)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(cols_present, rotation=20, ha="right")
        ax.set_ylabel(r"$\Delta$ test CER vs vanilla")
        ax.set_title(title)
        ax.legend(loc="best")
    fig.suptitle(r"Transfer-pattern matrix: $\Delta$ test CER vs vanilla")
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)


def _draw_heatmap_panel(ax, df: pd.DataFrame, panel: str, scale: int | None, title: str,
                        norm, cmap, fontsize_text: float = 7.0) -> bool:
    if panel == "causal":
        sub = df[(df["mode"] == "causal") & (df["scale_M"] == scale)]
        keys = row_keys_for_panel("causal", sub)
    else:
        sub = df[(df["mode"].isin(["lion", "lion_s"])) & (df["scale_M"] == 7)]
        keys = row_keys_for_panel("lion", sub)
    cols_present = [c for c in COLUMN_ORDER if c in sub["column"].unique()]
    if not keys or not cols_present:
        ax.set_axis_off()
        ax.set_title(title, fontsize=9)
        return False
    delta_grid = np.full((len(keys), len(cols_present)), np.nan)
    cer_grid = np.full_like(delta_grid, np.nan)
    for i, (arch, mode, _label) in enumerate(keys):
        for j, c in enumerate(cols_present):
            cell = sub[(sub["arch"] == arch) & (sub["mode"] == mode) & (sub["column"] == c)]
            if cell.empty:
                continue
            best = cell.loc[cell["test_cer"].idxmin()]
            cer_grid[i, j] = best["test_cer"]
            delta_grid[i, j] = best["delta"]
    ax.imshow(delta_grid, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols_present)))
    ax.set_xticklabels(cols_present, rotation=20, ha="right")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k[2] for k in keys])
    ax.set_title(title)
    ax.grid(False)
    # Hide the imshow tick marks (they sit between cells and look noisy).
    ax.tick_params(axis="both", length=0)
    bound = max(abs(norm.vmin), abs(norm.vmax))
    text_threshold = bound * 0.55
    for i in range(len(keys)):
        for j, c in enumerate(cols_present):
            if np.isnan(cer_grid[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=fontsize_text, color="#495057")
                continue
            d = delta_grid[i, j]
            if c == "vanilla":
                txt = f"{cer_grid[i, j]:.3f}"
            else:
                txt = f"{cer_grid[i, j]:.3f}\n({d:+.3f})"
            color = "white" if abs(d) > text_threshold else "#212529"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fontsize_text, color=color)
    return True


def render_heatmap_panel(df: pd.DataFrame, panel: str, scale: int | None, title: str,
                          out_pdf: Path, out_png: Path, norm) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(PAGE_WIDTH_IN, 2.8), constrained_layout=True)
    drawn = _draw_heatmap_panel(ax, df, panel, scale, title, norm, DELTA_CMAP, fontsize_text=8.0)
    if drawn:
        bound = max(abs(norm.vmin), abs(norm.vmax))
        sm = plt.cm.ScalarMappable(cmap=DELTA_CMAP, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(
            rf"$\Delta$ test CER  (blue helpful; local bound $\pm${bound:.3f})"
        )
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)


def _panel_local_norm(df: pd.DataFrame, panel: str, scale: int | None):
    if panel == "causal":
        sub = df[(df["mode"] == "causal") & (df["scale_M"] == scale) & (df["column"] != "vanilla")]
    else:
        sub = df[(df["mode"].isin(["lion", "lion_s"])) & (df["scale_M"] == 7) & (df["column"] != "vanilla")]
    deltas = sub["delta"].dropna().values
    if deltas.size == 0:
        return delta_norm(-0.01, 0.01)
    return delta_norm(deltas.min(), deltas.max())


def render_heatmap(df: pd.DataFrame, out_pdf: Path, out_png: Path, norm) -> None:
    panels = [("causal", 7, "causal, 7M"), ("causal", 14, "causal, 14M"), ("lion", None, "LION, 7M")]
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.4), constrained_layout=True)
    for ax, (panel, scale, title) in zip(axes, panels):
        _draw_heatmap_panel(ax, df, panel, scale, title, norm, DELTA_CMAP, fontsize_text=6.0)
    sm = plt.cm.ScalarMappable(cmap=DELTA_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02, location="right")
    cbar.set_label(r"$\Delta$ test CER (blue helpful)")
    fig.suptitle(r"Transfer-pattern matrix: test CER and $\Delta$ vs vanilla")
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_typography()
    df = collect_cells()
    df = apply_dho_rule(df)
    df = attach_deltas(df)
    df = apply_manual_overrides(df, dataset="LS")
    print(f"[F1] cells loaded: {len(df)}")
    bases = (
        df[df["cell"] == "vanilla"][["arch", "mode", "scale_M", "test_cer"]]
        .sort_values(["mode", "scale_M", "arch"])
    )
    print("[F1] vanilla baselines:")
    print(bases.to_string(index=False))
    df.sort_values(["scale_M", "mode", "arch", "column"], inplace=True)
    df.to_csv(FIG_DIR / f"{STEM}_data.csv", index=False)
    render_bars(df, FIG_DIR / f"{STEM}.pdf", FIG_DIR / f"{STEM}.png")
    delta_vals = df.loc[df["column"] != "vanilla", "delta"].dropna().values
    norm = delta_norm(delta_vals.min(), delta_vals.max())
    print(f"[F1] symmetric Δ-bound = ±{max(abs(norm.vmin), abs(norm.vmax)):.3f}")
    render_heatmap(
        df, FIG_DIR / f"{STEM}_heatmap.pdf", FIG_DIR / f"{STEM}_heatmap.png", norm
    )
    for panel, scale, title, suffix in [
        ("causal", 7, "Causal, 7M", "causal_7m"),
        ("causal", 14, "Causal, 14M", "causal_14m"),
        ("lion", None, "LION, 7M", "lion_7m"),
    ]:
        local_norm = _panel_local_norm(df, panel, scale)
        local_bound = max(abs(local_norm.vmin), abs(local_norm.vmax))
        print(f"[F1] per-panel local norm for {suffix}: ±{local_bound:.3f}")
        render_heatmap_panel(
            df, panel, scale, title,
            FIG_DIR / f"{STEM}_heatmap_{suffix}.pdf",
            FIG_DIR / f"{STEM}_heatmap_{suffix}.png",
            local_norm,
        )
    notable = df[df["column"] != "vanilla"].dropna(subset=["delta"])
    if not notable.empty:
        biggest = notable.loc[notable["delta"].idxmin()]
        worst = notable.loc[notable["delta"].idxmax()]
        print(
            f"[F1] biggest helpful Δ: {biggest['delta']:+.4f} "
            f"({biggest['arch']} {biggest['mode']} {biggest['column']} {biggest['scale_M']}M)"
        )
        print(
            f"[F1] worst harmful Δ:   {worst['delta']:+.4f} "
            f"({worst['arch']} {worst['mode']} {worst['column']} {worst['scale_M']}M)"
        )


if __name__ == "__main__":
    main()
