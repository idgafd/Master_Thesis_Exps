#!/usr/bin/env python3
"""θ-mobility diagnostic for the trained RSE model.

Loads stage3_01_rwkv6_rse_seed42/best_model.pt and characterises:

1. **Static parameter mobility:** how far did time_theta, time_theta_w1,
   time_theta_w2 drift from their initialization?  Initialization values:
       time_theta_init    ~ U(-π/16, +π/16)         per (head, block)
       time_theta_w1_init = 0                        (LoRA gate)
       time_theta_w2_init ~ U(-0.01, +0.01)
2. **Dynamic θ usage:** forward-pass on dev-clean samples, extract the
   actual θ_t values the model produces, summarise their distribution
   per layer / per head / per block.

Output:
   prints a per-layer summary table to stdout
   writes outputs/stage3_01_rwkv6_rse_seed42/theta_diagnostic.json
   writes outputs/stage3_01_rwkv6_rse_seed42/theta_diagnostic.png

Run:  uv run python scripts/rse_theta_diagnostic.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config
from src.data.vocab import CharVocab
from src.data.dataset import ASRDataset, DurationBatchSampler, collate_fn
from src.data.librispeech import load_librispeech
from src.models.asr_model import ASRModel
from src.training.checkpoint import load_checkpoint
from torch.utils.data import DataLoader

CKPT = ROOT / "outputs" / "stage3_01_rwkv6_rse_seed42" / "best_model.pt"
OUT_DIR = ROOT / "outputs" / "stage3_01_rwkv6_rse_seed42"
THETA_INIT_SCALE = math.pi / 16
THETA_CLIP = math.pi / 4


def main() -> int:
    cfg = load_config(str(ROOT / "outputs" / "stage3_01_rwkv6_rse_seed42" / "config.yaml"), {})

    vocab = CharVocab.build_english()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = ASRModel(vocab_size=vocab.size, cfg=cfg).to(device)
    load_checkpoint(CKPT, model=model, map_location=device, restore_rng=False)
    model.eval()

    encoder = model.encoder
    n_layers = encoder.n_layers

    # ── 1. Static parameter mobility ───────────────────────────────────
    print("=" * 76)
    print("STATIC PARAMETER MOBILITY (vs. initialization)")
    print("=" * 76)
    print(f"{'Layer':<6} | {'time_theta':<32} | {'w1 ‖.‖_F':<9} | {'w2 ‖.‖_F':<9} | {'w2 init ‖.‖_F':<14}")
    print("-" * 76)

    static = {"layers": []}
    for i, layer in enumerate(encoder.layers):
        att = layer.att
        if not att.use_rse:
            continue
        H, K = att.n_head, att.head_size
        Bk = K // 2
        D_THETA = att.time_theta_w1.shape[1]

        theta = att.time_theta.detach().float().view(H, Bk).cpu()
        w1 = att.time_theta_w1.detach().float().cpu()
        w2 = att.time_theta_w2.detach().float().cpu()

        theta_mean = theta.mean().item()
        theta_std = theta.std().item()
        theta_max_abs = theta.abs().max().item()

        # Init reference: U(-π/16, π/16) has mean 0, std π/(16·√3) ≈ 0.113
        init_std = THETA_INIT_SCALE / math.sqrt(3.0)
        # w2 init: U(-0.01, 0.01) per element; ‖.‖_F = 0.01 / √3 · √(D·H·Bk)
        w2_init_fnorm = 0.01 / math.sqrt(3.0) * math.sqrt(D_THETA * H * Bk)

        w1_fnorm = w1.norm().item()
        w2_fnorm = w2.norm().item()

        print(
            f"L{i:<5}| "
            f"mean={theta_mean:+.3f} std={theta_std:.3f} max|·|={theta_max_abs:.3f}"
            f"   (init std≈{init_std:.3f}) | "
            f"{w1_fnorm:.3f}    | {w2_fnorm:.3f}    | {w2_init_fnorm:.3f}"
        )

        static["layers"].append({
            "layer": i,
            "theta_mean": theta_mean,
            "theta_std": theta_std,
            "theta_max_abs": theta_max_abs,
            "theta_init_std_ref": init_std,
            "w1_fnorm": w1_fnorm,
            "w1_init_fnorm": 0.0,
            "w2_fnorm": w2_fnorm,
            "w2_init_fnorm": w2_init_fnorm,
        })

    # Coarse interpretation of static mobility
    print()
    n_w1_grew = sum(1 for L in static["layers"] if L["w1_fnorm"] > 0.5)
    n_theta_drift = sum(
        1 for L in static["layers"]
        if abs(L["theta_mean"]) > L["theta_init_std_ref"] * 0.5
        or L["theta_std"] > L["theta_init_std_ref"] * 1.5
    )
    print(f"Layers where w1 (LoRA gate) grew above 0.5 Frobenius:           {n_w1_grew}/{len(static['layers'])}")
    print(f"Layers where time_theta drifted noticeably from U(-π/16, π/16): {n_theta_drift}/{len(static['layers'])}")

    # ── 2. Dynamic θ usage on dev set ──────────────────────────────────
    print()
    print("=" * 76)
    print("DYNAMIC θ USAGE on dev-clean (300 utterances)")
    print("=" * 76)

    dev_entries = load_librispeech("dev", cfg.data_cache_dir, cfg.min_audio_sec, cfg.max_audio_sec)
    dev_ds = ASRDataset(dev_entries[:300], vocab, cfg)
    dev_loader = DataLoader(
        dev_ds,
        batch_sampler=DurationBatchSampler(dev_entries[:300], cfg.batch_max_seconds, False, cfg.seed),
        collate_fn=collate_fn, num_workers=2,
    )

    # Hook the time_theta computation: monkey-patch the time-mix _compute_rkv_gw
    # to also save the produced theta tensor per layer.
    captured = {i: [] for i in range(n_layers)}

    def make_hook(layer_id):
        att = encoder.layers[layer_id].att
        orig = att._compute_rkv_gw

        def hook(x):
            r, k, v, g, w, theta = orig(x)
            # theta has shape (B, T, H*Bk); reshape for storage
            B, T, _ = theta.shape
            captured[layer_id].append(theta.detach().float().view(B, T, att.n_head, -1).cpu())
            return r, k, v, g, w, theta

        return hook, orig

    hooks = []
    for i, layer in enumerate(encoder.layers):
        if layer.att.use_rse:
            new_fn, orig = make_hook(i)
            layer.att._compute_rkv_gw = new_fn
            hooks.append((i, orig))

    with torch.no_grad():
        n_processed = 0
        for batch in dev_loader:
            mels, mel_lens, *_ = batch
            mels = mels.to(device); mel_lens = mel_lens.to(device)
            try:
                _ = model(mels, mel_lens)
            except Exception as e:
                print(f"  batch failed: {e}")
                continue
            n_processed += mels.size(0)
            if n_processed >= 300:
                break

    # Restore originals
    for i, orig in hooks:
        encoder.layers[i].att._compute_rkv_gw = orig

    print(f"Processed {n_processed} utterances\n")

    print(f"{'Layer':<6} | {'mean θ':<9} | {'std θ':<9} | {'|θ|>π/16':<10} | {'|θ|>π/8':<10} | {'sat |θ|=π/4':<11}")
    print("-" * 76)
    dynamic = {"layers": []}
    for i in range(n_layers):
        if not captured[i]:
            continue
        all_theta = torch.cat([t.reshape(-1) for t in captured[i]])
        mean = all_theta.mean().item()
        std = all_theta.std().item()
        # Fraction of values in informative ranges
        frac_above_init = (all_theta.abs() > THETA_INIT_SCALE).float().mean().item()
        frac_above_2init = (all_theta.abs() > THETA_INIT_SCALE * 2).float().mean().item()
        frac_saturated = (all_theta.abs() > 0.95 * THETA_CLIP).float().mean().item()
        print(
            f"L{i:<5}| "
            f"{mean:+.4f} | {std:.4f}  | "
            f"{frac_above_init:.3f}     | {frac_above_2init:.3f}     | {frac_saturated:.3f}"
        )
        dynamic["layers"].append({
            "layer": i,
            "mean": mean, "std": std,
            "frac_above_init": frac_above_init,
            "frac_above_2init": frac_above_2init,
            "frac_saturated": frac_saturated,
            "n_samples": all_theta.numel(),
        })

    print()
    print("Reading guide:")
    print(f"  init: |θ| ≤ π/16 ≈ {THETA_INIT_SCALE:.3f}; clip: |θ| ≤ π/4 ≈ {THETA_CLIP:.3f}")
    print(f"  Healthy mobility: 'frac |θ|>π/16' substantially > 0.5 in deeper layers")
    print(f"  Saturation:       'frac sat'      > 0.05 indicates clip is binding")

    # ── 3. Per-layer histograms ────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=False)
    for i in range(n_layers):
        if not captured[i]:
            continue
        ax = axes.flatten()[i]
        all_theta = torch.cat([t.reshape(-1) for t in captured[i]]).numpy()
        ax.hist(all_theta, bins=80, color="C0", alpha=0.7, density=True, range=(-THETA_CLIP, THETA_CLIP))
        ax.axvspan(-THETA_INIT_SCALE, THETA_INIT_SCALE, color="C1", alpha=0.15, label="init range" if i == 0 else None)
        ax.axvline(-THETA_CLIP, color="r", lw=0.8, ls=":")
        ax.axvline(+THETA_CLIP, color="r", lw=0.8, ls=":")
        ax.set_title(f"L{i}: mean={dynamic['layers'][i]['mean']:+.3f}, std={dynamic['layers'][i]['std']:.3f}")
        ax.set_xlabel("θ (rad)")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Dynamic θ distribution per layer (rwkv6_rse @ stage3_01)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "theta_diagnostic.png", dpi=120)
    print(f"\nHistogram → {OUT_DIR / 'theta_diagnostic.png'}")

    # Save numbers
    with open(OUT_DIR / "theta_diagnostic.json", "w") as f:
        json.dump({"static": static, "dynamic": dynamic, "params": {
            "theta_init_scale": THETA_INIT_SCALE,
            "theta_clip": THETA_CLIP,
        }}, f, indent=2)
    print(f"Numbers   → {OUT_DIR / 'theta_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
