#!/usr/bin/env python3
"""Post-training diagnostics probe for linear_attn_causal runs.

Implements the §7.5 mandate in STAGE10_PLAN.md: mechanism-specific probes
captured at epoch 15 and epoch 30 minimum and written to
``<run_dir>/diagnostics.json``.

For a run at ``outputs/linear_attn_causal_seed42/``, captures per-layer:

  * ``phi_k_mean`` / ``phi_k_frac_tiny`` — Katharopoulos feature-map health.
    ``frac_tiny`` is the fraction of phi(k) components below 1e-4 (signal
    that elu(k)+1 saturates at 0 for large negative k).
  * ``S_norm_p{p}`` — Frobenius norm of the running state S_t at t = last
    valid position per sequence, summarised by percentile p ∈ {50, 90, 99}
    across (batch, head).
  * ``Z_norm_p{p}`` — L1 norm of running z_t, same percentiles. If Z grows
    unbounded with T, the denominator grows too — indicator that the L1
    normaliser is (or isn't) doing real work.
  * ``den_percentiles`` — magnitude of phi(q)^T z + eps across all valid
    (b, h, t) positions. If the lower percentiles approach eps, the
    denominator is epsilon-floored — the L1 normaliser is effectively
    inactive, which is the failure mode the explicit-denom spec exists
    to rule out.
  * ``t_trajectory`` — state norms at the quartile positions (T/4, T/2,
    3T/4, T) so temporal drift is visible in the JSON.

Usage:
    uv run scripts/diagnose_linear_attn.py \\
        --run-dir outputs/linear_attn_causal_seed42 \\
        --n-utterances 64

Writes:
    <run_dir>/diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data.dataset import ASRDataset, DurationBatchSampler, collate_fn
from src.data.librispeech import load_librispeech
from src.data.vocab import CharVocab
from src.models.asr_model import ASRModel
from src.models.linear_attn_causal import CausalLinearAttentionLayer
from src.training.checkpoint import load_checkpoint
from src.utils.misc import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostics collector
# ---------------------------------------------------------------------------


class DiagnosticsCollector:
    """Accumulates per-layer intermediate statistics across forward passes.

    Registered as ``CausalLinearAttentionLayer._diag_hook``.  Keeps only
    summary statistics in memory — never the full (B, H, T, K, K) tensors.
    """

    def __init__(self) -> None:
        self.batches_per_layer: dict[int, list] = {}

    def __call__(self, layer: CausalLinearAttentionLayer, intermediates: dict) -> None:
        # Identify layer by its id() — stable within a process.
        lid = id(layer)
        bucket = self.batches_per_layer.setdefault(lid, [])

        phi_q = intermediates["phi_q"]
        phi_k = intermediates["phi_k"]
        S = intermediates["S"]
        Z = intermediates["Z"]
        den = intermediates["den"]
        mask = intermediates["key_padding_mask"]  # (B, T) True=pad, or None

        B, H, T, K = phi_q.shape

        # Valid (non-pad) position mask for (b, t).
        if mask is not None:
            valid = (~mask)  # (B, T)
        else:
            valid = torch.ones(B, T, dtype=torch.bool, device=phi_q.device)

        valid_bht = valid.unsqueeze(1).expand(B, H, T)

        # phi(k) saturation stats — only over valid positions.
        phi_k_valid = phi_k[valid_bht.unsqueeze(-1).expand(B, H, T, K)]
        phi_k_mean = phi_k_valid.mean().item()
        phi_k_frac_tiny = (phi_k_valid < 1e-4).float().mean().item()

        # Denominator distribution over valid positions.
        den_valid = den[valid_bht]
        den_percentiles = torch.quantile(
            den_valid, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device=den.device)
        ).tolist()
        den_frac_eps = (den_valid < 1e-5).float().mean().item()

        # Per-sequence last-valid-position index.
        last_idx = valid.long().sum(dim=1) - 1           # (B,)
        last_idx = torch.clamp(last_idx, min=0)
        t_idx = last_idx.view(B, 1, 1, 1).expand(-1, H, 1, K)

        # S at the last valid position: (B, H, K, K)
        S_last = torch.gather(
            S, 2,
            last_idx.view(B, 1, 1, 1, 1).expand(-1, H, 1, K, K),
        ).squeeze(2)
        S_norm = torch.linalg.norm(S_last.reshape(B, H, -1), dim=-1)  # (B, H)
        S_norm_percentiles = torch.quantile(
            S_norm.flatten(), torch.tensor([0.5, 0.9, 0.99], device=S.device)
        ).tolist()

        Z_last = torch.gather(Z, 2, t_idx).squeeze(2)                  # (B, H, K)
        Z_norm = Z_last.abs().sum(dim=-1)                              # (B, H)
        Z_norm_percentiles = torch.quantile(
            Z_norm.flatten(), torch.tensor([0.5, 0.9, 0.99], device=Z.device)
        ).tolist()

        # Trajectory: S-norm at quartile positions of the longest-valid
        # sequence in the batch (approximates population drift over t).
        T_eff = int(last_idx.max().item()) + 1
        t_quartiles = [max(0, T_eff // 4 - 1), max(0, T_eff // 2 - 1),
                       max(0, 3 * T_eff // 4 - 1), max(0, T_eff - 1)]
        traj = []
        for tq in t_quartiles:
            S_tq = S[:, :, tq]                                         # (B, H, K, K)
            s_norm = torch.linalg.norm(S_tq.reshape(B, H, -1), dim=-1)
            traj.append(float(s_norm.median().item()))

        bucket.append({
            "phi_k_mean": phi_k_mean,
            "phi_k_frac_tiny": phi_k_frac_tiny,
            "den_percentiles": den_percentiles,   # [p01, p10, p50, p90, p99]
            "den_frac_below_1e-5": den_frac_eps,
            "S_norm_percentiles": S_norm_percentiles,  # [p50, p90, p99]
            "Z_norm_percentiles": Z_norm_percentiles,
            "S_norm_trajectory_median": traj,
            "T_eff_max": T_eff,
        })

    def summarise(self, layer_order: list) -> list:
        """Return per-layer aggregated dicts in the order ``layer_order``
        (list of layer ids matching the encoder's ``self.layers``)."""
        out = []
        for depth, lid in enumerate(layer_order):
            samples = self.batches_per_layer.get(lid, [])
            if not samples:
                out.append({"depth": depth, "n_batches": 0})
                continue
            # Simple mean of each scalar / elementwise mean of each list.
            def _mean_of_list(key: str) -> list:
                arrs = [s[key] for s in samples]
                n = len(arrs[0])
                return [sum(a[i] for a in arrs) / len(arrs) for i in range(n)]
            out.append({
                "depth": depth,
                "n_batches": len(samples),
                "phi_k_mean": sum(s["phi_k_mean"] for s in samples) / len(samples),
                "phi_k_frac_tiny": sum(s["phi_k_frac_tiny"] for s in samples) / len(samples),
                "den_percentiles": _mean_of_list("den_percentiles"),
                "den_frac_below_1e-5": sum(s["den_frac_below_1e-5"] for s in samples) / len(samples),
                "S_norm_percentiles": _mean_of_list("S_norm_percentiles"),
                "Z_norm_percentiles": _mean_of_list("Z_norm_percentiles"),
                "S_norm_trajectory_median": _mean_of_list("S_norm_trajectory_median"),
                "T_eff_max": max(s["T_eff_max"] for s in samples),
            })
        return out


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def capture_for_checkpoint(
    checkpoint_path: Path,
    cfg: ExperimentConfig,
    vocab: CharVocab,
    dev_loader,
    device: torch.device,
    n_utterances: int,
) -> dict:
    """Load weights from ``checkpoint_path`` into a fresh model and capture
    diagnostics over the first ``n_utterances`` of ``dev_loader``."""
    model = ASRModel(vocab_size=vocab.size, cfg=cfg).to(device)
    state = load_checkpoint(checkpoint_path, model=model, map_location=device, restore_rng=False)
    model.eval()

    encoder = model.encoder
    layer_order = [id(layer) for layer in encoder.layers]

    collector = DiagnosticsCollector()
    CausalLinearAttentionLayer._diag_hook = collector
    try:
        seen = 0
        with torch.no_grad():
            for mels, targets, mel_lengths, target_lengths in dev_loader:
                if seen >= n_utterances:
                    break
                mels = mels.to(device)
                mel_lengths = mel_lengths.to(device)
                _ = model(mels, mel_lengths)
                seen += mels.size(0)
    finally:
        CausalLinearAttentionLayer._diag_hook = None

    return {
        "checkpoint": str(checkpoint_path.name),
        "epoch": int(state.get("epoch", -1)),
        "best_cer": float(state.get("best_cer", float("nan"))),
        "n_utterances_probed": seen,
        "per_layer": collector.summarise(layer_order),
    }


def find_checkpoints(run_dir: Path, targets: list) -> dict:
    """Return a dict label -> Path for whichever of the requested targets
    actually exist.  Labels are strings: "ep15", "ep30", "best".
    Also includes surrogate fallbacks for ep15 if it is missing."""
    found = {}
    for t in targets:
        if t == "best":
            p = run_dir / "best_model.pt"
            if p.exists():
                found["best"] = p
        elif t == "ep15":
            p = run_dir / "checkpoint_ep15.pt"
            if p.exists():
                found["ep15"] = p
            else:
                # Surrogate: use ep10 and/or ep20 if present.
                for fallback in ("checkpoint_ep10.pt", "checkpoint_ep20.pt"):
                    q = run_dir / fallback
                    if q.exists():
                        label = fallback.replace("checkpoint_", "").replace(".pt", "")
                        found[label + "_surrogate_for_ep15"] = q
        else:
            # Arbitrary "epN" label.
            name = f"checkpoint_{t}.pt"
            p = run_dir / name
            if p.exists():
                found[t] = p
    return found


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-utterances", type=int, default=64,
                        help="Number of dev utterances to probe per checkpoint.")
    parser.add_argument("--checkpoints", nargs="*",
                        default=["ep15", "ep30", "best"],
                        help="Checkpoint labels to probe.  Supports "
                             "'best', 'ep15', 'ep30', 'ep20', 'ep10'.")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"no config.yaml in {run_dir}")
    with open(cfg_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = ExperimentConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.output_dir = str(run_dir)

    if not cfg.backbone.startswith("linear_attn"):
        raise SystemExit(
            f"diagnose_linear_attn.py supports the linear_attn_* backbone "
            f"family (linear_attn_causal, linear_attn_convshift_multidil_symmetric, "
            f"etc.); got {cfg.backbone!r}. Run dir: {run_dir}"
        )

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(cfg.seed)

    # Dev loader (clean).
    vocab = CharVocab.build_english()
    dev_entries = load_librispeech(
        "dev", cache_dir=cfg.data_cache_dir,
        min_audio_sec=cfg.min_audio_sec, max_audio_sec=cfg.max_audio_sec,
    )
    dev_ds = ASRDataset(dev_entries, vocab, cfg)
    dev_loader = DataLoader(
        dev_ds,
        batch_sampler=DurationBatchSampler(
            dev_entries, cfg.batch_max_seconds, shuffle=False, seed=cfg.seed,
        ),
        collate_fn=collate_fn, num_workers=2,
    )

    cps = find_checkpoints(run_dir, args.checkpoints)
    if not cps:
        raise SystemExit(
            f"no checkpoints matching {args.checkpoints} found in {run_dir}"
        )
    logger.info(f"Probing checkpoints: {list(cps.keys())}")

    results: dict = {
        "run_dir": str(run_dir),
        "backbone": cfg.backbone,
        "spec": "Stage 11.0a diagnostics per STAGE10_PLAN.md §7.5",
        "probes": {},
    }
    for label, path in cps.items():
        logger.info(f"[{label}] loading {path.name}...")
        results["probes"][label] = capture_for_checkpoint(
            path, cfg, vocab, dev_loader, device, args.n_utterances,
        )

    out_path = run_dir / "diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
