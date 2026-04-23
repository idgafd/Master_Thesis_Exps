#!/usr/bin/env python3
"""Novelty-gate-specific diagnostics for mamba2_novelty_* runs.

Captures per-layer, per-chunk-depth statistics of:

  * γ_per_head — effective γ after softplus(γ_raw - shift) [trainable]
    or the fixed buffer value [fixed_g05].
  * ω_stats — mean, p10, p50, p90 of ω_t across tokens × heads × batch,
    bucketed by chunk-depth (chunk 0, chunk 1, …).
  * q2_stats — mean and p50/p90 of the Mahalanobis q² (denominator of
    the gate) per layer × chunk-depth.
  * sigma_fro — Frobenius norm of Σ_c per layer × chunk-depth.
  * sigma_cond — condition number of Σ_c (max/min non-zero eigenvalue)
    per layer × chunk-depth.
  * write_suppression — mean (1 - ω) per layer, the effective fraction
    of writes attenuated.

Telemetry is captured by monkey-patching ``_compute_novelty_gates`` to
record stats then delegating to the original.

Usage:
    uv run scripts/diagnose_mamba2_novelty.py \\
        --run-dir outputs/mamba2_novelty_gate_seed42 \\
        --n-utterances 64
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data.dataset import ASRDataset, DurationBatchSampler, collate_fn
from src.data.librispeech import load_librispeech
from src.data.vocab import CharVocab
from src.models import mamba2_kernels
from src.models.asr_model import ASRModel
from src.models.mamba2_block import Mamba2Block
from src.training.checkpoint import load_checkpoint
from src.utils.misc import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_quant_ptl = torch.tensor([0.10, 0.50, 0.90])


def _percentiles(t: torch.Tensor) -> list:
    if t.numel() == 0:
        return [float("nan")] * 3
    flat = t.flatten().cpu()
    q = _quant_ptl.to(flat.dtype)
    return torch.quantile(flat, q).tolist()


class NoveltyProbe:
    """Captures per-layer × per-chunk-depth novelty-gate statistics.

    Works by wrapping ``mamba2_kernels._compute_novelty_gates`` and
    tagging calls with the current layer index (set by a forward hook
    on each Mamba2Block).  Each call records ω, q², and Σ spectrum
    stats per chunk index within that layer's forward.
    """

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self.current_layer: int = -1
        self.per_layer: list[dict] = [
            {
                "omega_by_chunk": {},   # chunk_idx -> concat of (B, Tc, H)
                "q2_by_chunk": {},      # chunk_idx -> concat of (B, Tc, H)
                "sigma_fro_by_chunk": {},   # chunk_idx -> concat of (B, H)
                "sigma_cond_by_chunk": {},  # chunk_idx -> concat of (B, H)
                "gamma_snapshot": None,
            }
            for _ in range(n_layers)
        ]
        self._orig_fn = mamba2_kernels._compute_novelty_gates

    def install(self, encoder) -> None:
        probe = self

        for depth, layer in enumerate(encoder.layers):
            blk: Mamba2Block = layer.mamba
            if not getattr(blk, "use_novelty_gate", False):
                continue

            # Record γ snapshot.
            if getattr(blk, "_novelty_trainable", False):
                gamma = F.softplus(blk.novelty_gamma_raw - blk._novelty_shift)
            else:
                gamma = blk.novelty_gamma_buf
            probe.per_layer[depth]["gamma_snapshot"] = gamma.detach().float().cpu().tolist()

            # Wrap forward to tag the layer index for the kernel patch.
            orig_forward = blk.forward

            def make_hook(d, orig_fn):
                def hooked(x, state=None):
                    prev = probe.current_layer
                    probe.current_layer = d
                    try:
                        return orig_fn(x, state=state)
                    finally:
                        probe.current_layer = prev
                return hooked

            blk.forward = make_hook(depth, orig_forward)

        # Monkey-patch the kernel helper.
        def patched_compute(B_c, A_cumsum, gamma, sigma_init,
                            eps_reg=1e-3, eps_num=1e-6):
            L = probe.current_layer
            Bsz, nC, Tc, H, N = B_c.shape
            device = B_c.device
            dtype = B_c.dtype

            if sigma_init is None:
                sigma = torch.zeros(Bsz, H, N, N, device=device, dtype=dtype)
            else:
                sigma = sigma_init.to(dtype)

            eye = torch.eye(N, device=device, dtype=dtype).view(1, 1, N, N)
            gamma_h = gamma.view(1, 1, H).to(dtype)

            omega_chunks = []
            q2_chunks = []
            sigma_fro_chunks = []
            sigma_cond_chunks = []

            for c in range(nC):
                sigma_reg = sigma + eps_reg * eye
                sigma_inv = torch.linalg.inv(sigma_reg)

                B_here = B_c[:, c]
                q2 = torch.einsum("bthn,bhnm,bthm->bth", B_here, sigma_inv, B_here)
                omega_c = 1.0 / (1.0 + gamma_h / (q2 + eps_num))
                omega_chunks.append(omega_c)

                # Diagnostics
                if L >= 0:
                    # Filter to real (non-padding) tokens: real tokens
                    # have ||B_here||² > 0; padding is exactly zero.
                    b_norm = (B_here ** 2).sum(dim=-1)             # (B, Tc, H)
                    real_mask = b_norm > 1e-10                      # (B, Tc, H)
                    # Apply mask by flattening masked values only.
                    omega_real = omega_c[real_mask]
                    q2_real = q2[real_mask]

                    # Frobenius of Σ (over (B, H)).
                    sigma_fro = torch.linalg.norm(sigma.reshape(Bsz, H, -1), dim=-1)
                    # Condition number of Σ_reg (regularised for stability).
                    eigs = torch.linalg.eigvalsh(sigma_reg)      # (B, H, N)
                    eig_max = eigs[..., -1]
                    eig_min = eigs[..., 0].clamp(min=1e-12)
                    sigma_cond = eig_max / eig_min              # (B, H)

                    probe.per_layer[L]["omega_by_chunk"].setdefault(c, []).append(omega_real.detach().float().cpu())
                    probe.per_layer[L]["q2_by_chunk"].setdefault(c, []).append(q2_real.detach().float().cpu())
                    probe.per_layer[L]["sigma_fro_by_chunk"].setdefault(c, []).append(sigma_fro.detach().float().cpu())
                    probe.per_layer[L]["sigma_cond_by_chunk"].setdefault(c, []).append(sigma_cond.detach().float().cpu())

                alpha_bar = torch.exp(A_cumsum[:, :, c, -1])
                BtB = torch.einsum("bthn,bthm->bhnm", B_here, B_here)
                sigma = alpha_bar.view(Bsz, H, 1, 1) * sigma + BtB

            omega = torch.stack(omega_chunks, dim=1)
            return omega, sigma

        mamba2_kernels._compute_novelty_gates = patched_compute
        self._patched = True

    def uninstall(self) -> None:
        if getattr(self, "_patched", False):
            mamba2_kernels._compute_novelty_gates = self._orig_fn
            self._patched = False

    def summarise(self) -> list:
        out = []
        for depth, rec in enumerate(self.per_layer):
            if rec["gamma_snapshot"] is None:
                out.append({"depth": depth, "novelty_gate": False})
                continue

            chunks_seen = sorted(rec["omega_by_chunk"].keys())
            omega_stats = {}
            q2_stats = {}
            sigma_fro_stats = {}
            sigma_cond_stats = {}
            for c in chunks_seen:
                omega_cat = torch.cat([t.flatten() for t in rec["omega_by_chunk"][c]])
                q2_cat = torch.cat([t.flatten() for t in rec["q2_by_chunk"][c]])
                sf_cat = torch.cat([t.flatten() for t in rec["sigma_fro_by_chunk"][c]])
                sc_cat = torch.cat([t.flatten() for t in rec["sigma_cond_by_chunk"][c]])
                omega_stats[f"chunk{c}"] = {
                    "mean": float(omega_cat.mean()),
                    "p10_p50_p90": _percentiles(omega_cat),
                    "n": int(omega_cat.numel()),
                }
                q2_stats[f"chunk{c}"] = {
                    "mean": float(q2_cat.mean()),
                    "p10_p50_p90": _percentiles(q2_cat),
                }
                sigma_fro_stats[f"chunk{c}"] = {
                    "mean": float(sf_cat.mean()),
                    "p10_p50_p90": _percentiles(sf_cat),
                }
                sigma_cond_stats[f"chunk{c}"] = {
                    "mean": float(sc_cat.mean()),
                    "p10_p50_p90": _percentiles(sc_cat),
                }

            # Aggregate write suppression (over all captured chunks).
            all_omega = torch.cat([
                torch.cat([t.flatten() for t in v])
                for v in rec["omega_by_chunk"].values()
            ])
            write_suppression = float((1.0 - all_omega).mean())

            out.append({
                "depth": depth,
                "novelty_gate": True,
                "gamma_per_head": rec["gamma_snapshot"],
                "gamma_mean": float(torch.tensor(rec["gamma_snapshot"]).mean()),
                "gamma_max": float(torch.tensor(rec["gamma_snapshot"]).max()),
                "omega_by_chunk": omega_stats,
                "q2_by_chunk": q2_stats,
                "sigma_fro_by_chunk": sigma_fro_stats,
                "sigma_cond_by_chunk": sigma_cond_stats,
                "write_suppression_mean": write_suppression,
                "n_chunks_seen": len(chunks_seen),
            })
        return out


def capture_for_checkpoint(
    checkpoint_path: Path,
    cfg: ExperimentConfig,
    vocab: CharVocab,
    dev_loader,
    device: torch.device,
    n_utterances: int,
) -> dict:
    model = ASRModel(vocab_size=vocab.size, cfg=cfg).to(device)
    state = load_checkpoint(
        checkpoint_path, model=model, map_location=device, restore_rng=False,
    )
    model.eval()
    encoder = model.encoder

    probe = NoveltyProbe(n_layers=len(encoder.layers))
    probe.install(encoder)

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
        probe.uninstall()

    return {
        "checkpoint": str(checkpoint_path.name),
        "epoch": int(state.get("epoch", -1)),
        "best_cer": float(state.get("best_cer", float("nan"))),
        "n_utterances_probed": seen,
        "per_layer": probe.summarise(),
    }


def find_checkpoints(run_dir: Path, targets: list) -> dict:
    found = {}
    for t in targets:
        if t == "best":
            p = run_dir / "best_model.pt"
            if p.exists():
                found["best"] = p
        else:
            p = run_dir / f"checkpoint_{t}.pt"
            if p.exists():
                found[t] = p
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-utterances", type=int, default=64)
    parser.add_argument("--checkpoints", nargs="*",
                        default=["ep1", "ep5", "ep15", "ep30", "best"])
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

    if "novelty" not in cfg.backbone:
        raise SystemExit(
            f"diagnose_mamba2_novelty.py is for novelty-gate runs; "
            f"got backbone {cfg.backbone!r}"
        )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(cfg.seed)

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
        raise SystemExit(f"no checkpoints matching {args.checkpoints} in {run_dir}")
    logger.info(f"Probing checkpoints: {list(cps.keys())}")

    results = {
        "run_dir": str(run_dir),
        "backbone": cfg.backbone,
        "spec": "Novelty-gate diagnostics: ω/q²/Σ per layer × chunk-depth",
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
