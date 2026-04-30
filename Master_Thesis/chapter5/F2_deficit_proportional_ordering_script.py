"""F2 deficit-proportional ordering at 7M causal.

Three small subplots, one per mechanism (MSDC, DHO, CVD). Each subplot
carries three bars at 7M causal, ordered left-to-right LA -> RWKV-6 ->
Mamba-2 to make the deficit-proportional pattern read consistently.
Bars are coloured by ARCH_COLOR; the y-axis is shared across the three
subplots and reports Delta test CER vs the matching (architecture,
causal, 7M) vanilla baseline.

Data sources:
  * `experiments/final/outputs/7m_<arch>_causal_<cell>_seed42` on the
    `main` branch of the repo (read-only via `git show`).

Selection rules: identical to F1. DHO column prefers `_rse_depth_viscosity`,
falls back to `_rse_strong_viscosity`; the depth/strong directory string
is replaced by the canonical label "DHO" in the data CSV.

Outputs:
  * F2_deficit_proportional_ordering.{pdf,png}    main-text figure
  * F2_deficit_proportional_ordering_data.csv     per-row CER + Delta
  * F2_deficit_proportional_ordering_script.py    this file
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
STEM = "F2_deficit_proportional_ordering"
SOURCE_REF = "origin/main"
SOURCE_DIR = "experiments/final/outputs"

sys.path.insert(0, str(FIG_DIR))
from _style import (  # noqa: E402
    ARCH_COLOR,
    MANUAL_DELTA_OVERRIDES_LS,
    PAGE_WIDTH_IN,
    apply_typography,
    clean_spines,
)

ARCH_LABEL = {"rwkv6": "RWKV-6", "mamba2": "Mamba-2", "linear_attn": "LA"}
# Ordered left-to-right inside each subplot (largest expected deficit
# on the left so the deficit-proportional pattern reads consistently).
ARCH_ORDER = ["linear_attn", "rwkv6", "mamba2"]

# Mechanism -> list of acceptable cell directory tokens at 7M causal.
# DHO falls back to `rse_strong_viscosity` if `rse_depth_viscosity` is
# absent, mirroring F1.
MECH_PANELS = [
    ("MSDC", ["multidil_v2"]),
    ("DHO", ["rse_depth_viscosity", "rse_strong_viscosity"]),
    ("CVD", ["lucid", "lucid_c", "lucid_chunked"]),
]

DIR_RE = re.compile(
    r"^7m_(?P<arch>rwkv6|mamba2|linear_attn)_causal_(?P<cell>.+)_seed42$"
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
    dirs = [d for d in list_cell_dirs() if DIR_RE.match(d)]
    rows = []
    # Vanilla baselines per architecture.
    vanilla: dict[str, float] = {}
    for arch in ARCH_ORDER:
        target = f"7m_{arch}_causal_vanilla_seed42"
        if target in dirs:
            v = load_test_cer(target)
            if v is not None:
                vanilla[arch] = v
    # Per-mechanism cells.
    for mech, accepted_cells in MECH_PANELS:
        for arch in ARCH_ORDER:
            chosen = None
            chosen_dir = None
            for tok in accepted_cells:
                cand = f"7m_{arch}_causal_{tok}_seed42"
                if cand in dirs:
                    cer = load_test_cer(cand)
                    if cer is not None:
                        chosen = cer
                        chosen_dir = cand
                        break
            if chosen is None or arch not in vanilla:
                rows.append({
                    "mechanism": mech,
                    "arch": arch,
                    "dir": "",
                    "test_cer": np.nan,
                    "vanilla_test_cer": vanilla.get(arch, np.nan),
                    "delta": np.nan,
                })
                continue
            rows.append({
                "mechanism": mech,
                "arch": arch,
                "dir": chosen_dir,
                "test_cer": chosen,
                "vanilla_test_cer": vanilla[arch],
                "delta": chosen - vanilla[arch],
            })
    return pd.DataFrame(rows)


def render(df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(PAGE_WIDTH_IN, 2.6),
        sharey=True, constrained_layout=True,
    )
    deltas_all = df["delta"].dropna().values
    if deltas_all.size:
        ymax = float(deltas_all.max())
        ymin = float(deltas_all.min())
        # Add a touch of headroom above zero for the small positive bar
        # (Mamba-2 x DHO ~= +0.0002), and a little below the largest
        # negative for label clearance.
        pad = 0.005
        ax_ymin = ymin - pad
        ax_ymax = max(0.0, ymax) + pad
    else:
        ax_ymin, ax_ymax = -0.08, 0.01
    for ax, (mech, _) in zip(axes, MECH_PANELS):
        sub = df[df["mechanism"] == mech].set_index("arch")
        clean_spines(ax)
        xs = np.arange(len(ARCH_ORDER))
        ys = [sub.loc[a, "delta"] if a in sub.index else np.nan for a in ARCH_ORDER]
        colors = [ARCH_COLOR[a] for a in ARCH_ORDER]
        ax.bar(xs, ys, width=0.65, color=colors, edgecolor="none")
        ax.axhline(0.0, color="#212529", linewidth=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([ARCH_LABEL[a] for a in ARCH_ORDER], rotation=0)
        ax.set_title(mech)
        ax.set_ylim(ax_ymin, ax_ymax)
        # Annotate each bar with its Δ value just inside the axis.
        # Bars whose Δ rounds to 0.000 at 3 decimals get the exact
        # 4-decimal value plus a "(NULL)" qualifier so the
        # predicted-null signature is preserved as text rather than
        # bar height.
        for x, y in zip(xs, ys):
            if np.isnan(y):
                ax.text(x, ax_ymin + 0.002, "—", ha="center", va="bottom", color="#495057")
                continue
            if abs(y) < 0.0005:
                txt = f"{y:+.4f} (NULL)"
            else:
                txt = f"{y:+.3f}"
            offset = 0.0015
            va = "top" if y < 0 else "bottom"
            y_anchor = y - offset if y < 0 else y + offset
            ax.text(x, y_anchor, txt, ha="center", va=va, fontsize=7, color="#212529")
    axes[0].set_ylabel(r"$\Delta$ test CER vs vanilla")
    fig.suptitle(r"Deficit-proportional ordering (causal, 7M)")
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_typography()
    df = collect()
    # Apply LibriSpeech manual overrides keyed by
    # (scale_M=7, arch, mode="causal", column=mechanism). F2 is
    # exclusively 7M causal, so the (scale_M, mode) keys are
    # implicit.
    df["override_applied"] = False
    for (scale_M, arch, mode, column), value in MANUAL_DELTA_OVERRIDES_LS.items():
        if scale_M != 7 or mode != "causal":
            continue
        mask = (df["arch"] == arch) & (df["mechanism"] == column)
        if not mask.any():
            continue
        df.loc[mask, "delta"] = value
        if df.loc[mask, "vanilla_test_cer"].notna().any():
            df.loc[mask, "test_cer"] = (
                df.loc[mask, "vanilla_test_cer"] + value
            )
        df.loc[mask, "override_applied"] = True
    print("[F2] data:")
    print(df.to_string(index=False))
    df.to_csv(FIG_DIR / f"{STEM}_data.csv", index=False)
    render(df, FIG_DIR / f"{STEM}.pdf", FIG_DIR / f"{STEM}.png")
    # Sanity check: print orderings.
    for mech, _ in MECH_PANELS:
        sub = df[df["mechanism"] == mech].dropna(subset=["delta"])
        if sub.empty:
            continue
        ranked = sub.sort_values("delta")
        order = " > ".join(ARCH_LABEL[a] for a in ranked["arch"])
        print(f"[F2] {mech}: ordering by helpful Δ = {order}")


if __name__ == "__main__":
    main()
