"""Experiment result plotting utilities."""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BACKBONE_STYLE = {
    "transformer":      ("Transformer",      "#1f77b4", "-", "o"),
    "linear_attention": ("Linear Attention",  "#ff7f0e", "-", "s"),
    "mamba":            ("Mamba",             "#2ca02c", "-", "^"),
    "rwkv6":            ("RWKV-6",            "#d62728", "-", "D"),
    "rwkv7":            ("RWKV-7",            "#9467bd", "-", "v"),
}


def _style(bb: str):
    return BACKBONE_STYLE.get(bb, (bb, "grey", "-", "x"))


def plot_convergence(all_results: dict, backbones: list, plot_dir: str) -> None:
    """3-panel convergence figure: train loss / dev loss / dev CER vs epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panels = [
        ("train_loss", "Train Loss (CTC)", axes[0]),
        ("dev_loss",   "Dev Loss (CTC)",   axes[1]),
        ("dev_cer",    "Dev CER",          axes[2]),
    ]
    for key, ylabel, ax in panels:
        for bb in backbones:
            if bb not in all_results:
                continue
            h = all_results[bb]["history"]
            label, color, ls, marker = _style(bb)
            epochs = h["epoch"]
            values = h[key]
            n = len(epochs)
            mark_every = max(1, n // 10)
            ax.plot(
                epochs, values,
                color=color, linestyle=ls, linewidth=1.8,
                marker=marker, markersize=5, markevery=mark_every,
                label=label, alpha=0.9,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if key == "dev_cer":
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(backbones),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Training Convergence", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    _save(fig, plot_dir, "convergence")


def plot_chunked_cer_bars(
    all_results: dict, backbones: list, chunk_sizes_sec: list, plot_dir: str
) -> None:
    """Grouped bar chart: test CER for full utterance and each reset-state chunk size."""
    present = [bb for bb in backbones if bb in all_results]
    conditions = ["Full"] + [f"{cs}s" for cs in chunk_sizes_sec]
    n_cond = len(conditions)
    n_bb = len(present)

    bar_width = 0.8 / n_cond
    fig, ax = plt.subplots(figsize=(max(8, n_bb * 2.2), 5))

    for j, cond in enumerate(conditions):
        cers = []
        for bb in present:
            r = all_results[bb]
            if cond == "Full":
                cers.append(r["test"]["cer"])
            else:
                cers.append(r["chunked_reset"].get(cond, {}).get("cer", float("nan")))

        x = np.arange(n_bb) + j * bar_width
        bars = ax.bar(x, [c * 100 for c in cers], bar_width * 0.9, label=cond, alpha=0.85)
        for bar, cer in zip(bars, cers):
            if not np.isnan(cer):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{cer * 100:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(np.arange(n_bb) + bar_width * (n_cond - 1) / 2)
    ax.set_xticklabels([_style(bb)[0] for bb in present], fontsize=10)
    ax.set_ylabel("CER (%)")
    ax.set_title("Test CER: Full Utterance vs Chunked Inference (Reset-State)",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Context", fontsize=9, title_fontsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, plot_dir, "chunked_cer_reset")


def plot_carry_vs_reset(
    all_results: dict, backbones: list, chunk_sizes_sec: list, plot_dir: str
) -> None:
    """Side-by-side reset vs carry CER bars for stateful backbones."""
    carry_bbs = [bb for bb in backbones if bb in all_results and all_results[bb]["chunked_carry"]]
    if not carry_bbs:
        print("  carry_vs_reset: no stateful backbones with results, skipping")
        return

    chunk_labels = [f"{cs}s" for cs in chunk_sizes_sec]
    n_bb = len(carry_bbs)
    fig, axes = plt.subplots(1, n_bb, figsize=(5 * n_bb, 5), squeeze=False)

    for idx, bb in enumerate(carry_bbs):
        ax = axes[0, idx]
        r = all_results[bb]
        label, color, _, _ = _style(bb)

        reset_cers = [r["chunked_reset"].get(cl, {}).get("cer", float("nan")) * 100
                      for cl in chunk_labels]
        carry_cers = [r["chunked_carry"][cl]["cer"] * 100
                      for cl in chunk_labels if cl in r["chunked_carry"]]

        x = np.arange(len(carry_cers))
        w = 0.35
        bars_r = ax.bar(x - w / 2, reset_cers[: len(carry_cers)], w,
                        label="Reset", color=color, alpha=0.5)
        bars_c = ax.bar(x + w / 2, carry_cers, w,
                        label="Carry", color=color, alpha=0.9)

        for bars in [bars_r, bars_c]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

        for i in range(len(carry_cers)):
            delta = reset_cers[i] - carry_cers[i]
            if abs(delta) > 0.01:
                y_pos = max(reset_cers[i], carry_cers[i]) + 1.5
                sign = "−" if delta > 0 else "+"
                ax.text(x[i], y_pos, f"{sign}{abs(delta):.1f}pp",
                        ha="center", fontsize=8, fontweight="bold",
                        color="green" if delta > 0 else "red")

        ax.set_xticks(x)
        ax.set_xticklabels(chunk_labels[: len(carry_cers)])
        ax.set_ylabel("CER (%)")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Carry-State vs Reset-State Chunked CER",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, plot_dir, "carry_vs_reset")


def plot_all(all_results: dict, backbones: list, chunk_sizes_sec: list, output_dir: str) -> None:
    """Generate and save all experiment plots."""
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plot_convergence(all_results, backbones, plot_dir)
    plot_chunked_cer_bars(all_results, backbones, chunk_sizes_sec, plot_dir)
    plot_carry_vs_reset(all_results, backbones, chunk_sizes_sec, plot_dir)
    print(f"All plots saved to {plot_dir}/")


def _save(fig, plot_dir: str, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(plot_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf / .png")
