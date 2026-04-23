#!/usr/bin/env python3
"""Post-training diagnostics probe for mamba2* runs.

Companion to ``diagnose_linear_attn.py`` — same output schema (per-layer
stats, multi-checkpoint capture), different probes tuned to Mamba-2's
SSD state dynamics.

Per-layer stats captured:

  * ``dt_percentiles`` — softplus(dt_raw + dt_bias), the per-head step
    size.  p01/p10/p50/p90/p99 over all valid (b, l, h).  Mamba-2's
    selective Δt gives continuous decay; if p50 is very small the SSD
    state updates are nearly identity, and the per-head variance tells
    us how much Δt specialises across heads.
  * ``A_cont`` — -exp(A_log), the learned per-head continuous decay.
    Reported as (min, p50, max) over heads × layers.
  * ``ssm_state_norm_p{p}`` — ||h||_F of the SSD state at the last valid
    position, percentiles p50/p90/p99 over (B, H).
  * ``conv_branch_alpha`` — (mamba2_convshift_multidil_symmetric only)
    the learned per-layer alpha over dilations {1, 2, 4, 8}.  This is
    the key mechanism-engagement probe for Stage 11.1a: if SGD grows
    alpha_{2,4,8} meaningfully away from 0, the multi-dilation axis has
    been engaged.
  * ``D_skip`` — the per-head D skip scalars (min, p50, max over H×L).

Usage:
    uv run scripts/diagnose_mamba2.py \\
        --run-dir outputs/mamba2_convshift_multidil_symmetric_seed42 \\
        --n-utterances 64
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
from src.models.mamba2_block import Mamba2Block, MultiDilationDWConv1d
from src.training.checkpoint import load_checkpoint
from src.utils.misc import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Capture via forward_pre_hook on Mamba2Block
# ---------------------------------------------------------------------------


class Mamba2Probe:
    """Attaches forward hooks on each Mamba2Block to capture ssm_state norm
    and dt distribution as the scan runs."""

    def __init__(self):
        self.batches_per_layer: dict[int, list] = {}
        self._handles: list = []

    def install(self, encoder) -> None:
        for layer in encoder.layers:
            blk: Mamba2Block = layer.mamba
            lid = id(blk)
            self.batches_per_layer.setdefault(lid, [])

            # Wrap forward so we can capture intermediates cheaply.
            original_forward = blk.forward

            def make_hook(block_id, orig_fn, module):
                def hooked(x, state=None):
                    out, new_state = orig_fn(x, state=state)
                    # Recompute dt just for telemetry (cheap — one softplus).
                    zxbcdt = module.in_proj(x)
                    _, _, dt_raw = torch.split(
                        zxbcdt,
                        [module.d_inner, module.conv_dim, module.nheads],
                        dim=-1,
                    )
                    dt = torch.nn.functional.softplus(dt_raw + module.dt_bias)
                    # SSM state norm: (B, H, P, N).
                    ssm = new_state.get("ssm") if new_state is not None else None
                    A_cont = -torch.exp(module.A_log.float())
                    self.batches_per_layer[block_id].append({
                        "dt": dt.detach().float(),
                        "ssm": ssm.detach().float() if ssm is not None else None,
                        "A_cont": A_cont.detach().float().cpu().tolist(),
                        "D": module.D.detach().float().cpu().tolist(),
                        "alpha": (module.conv1d.alpha.detach().float().cpu().tolist()
                                  if isinstance(module.conv1d, MultiDilationDWConv1d)
                                  else None),
                    })
                    return out, new_state
                return hooked

            blk.forward = make_hook(lid, original_forward, blk)

    def summarise(self, layer_order: list) -> list:
        out = []
        for depth, lid in enumerate(layer_order):
            samples = self.batches_per_layer.get(lid, [])
            if not samples:
                out.append({"depth": depth, "n_batches": 0})
                continue

            dts = torch.cat([s["dt"].flatten() for s in samples])
            q = torch.quantile(
                dts, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device=dts.device)
            ).tolist()

            ssms = [s["ssm"] for s in samples if s["ssm"] is not None]
            ssm_p = []
            if ssms:
                norms = []
                for ssm in ssms:
                    B, H, P, N = ssm.shape
                    norms.append(torch.linalg.norm(ssm.reshape(B, H, -1), dim=-1).flatten())
                norms_cat = torch.cat(norms)
                ssm_p = torch.quantile(
                    norms_cat,
                    torch.tensor([0.5, 0.9, 0.99], device=norms_cat.device),
                ).tolist()

            # A_cont and D — take from first sample (constant across batches for a given state).
            A_cont = samples[0]["A_cont"]
            D = samples[0]["D"]

            alpha = samples[0]["alpha"]  # None for vanilla mamba2

            out.append({
                "depth": depth,
                "n_batches": len(samples),
                "dt_percentiles": q,  # [p01, p10, p50, p90, p99]
                "ssm_state_norm_percentiles": ssm_p,  # [p50, p90, p99]
                "A_cont_per_head": A_cont,
                "D_per_head": D,
                "conv_branch_alpha": alpha,  # only if multidil
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
    model = ASRModel(vocab_size=vocab.size, cfg=cfg).to(device)
    state = load_checkpoint(checkpoint_path, model=model, map_location=device, restore_rng=False)
    model.eval()
    encoder = model.encoder

    probe = Mamba2Probe()
    probe.install(encoder)

    layer_order = [id(layer.mamba) for layer in encoder.layers]

    seen = 0
    with torch.no_grad():
        for mels, targets, mel_lengths, target_lengths in dev_loader:
            if seen >= n_utterances:
                break
            mels = mels.to(device)
            mel_lengths = mel_lengths.to(device)
            _ = model(mels, mel_lengths)
            seen += mels.size(0)

    return {
        "checkpoint": str(checkpoint_path.name),
        "epoch": int(state.get("epoch", -1)),
        "best_cer": float(state.get("best_cer", float("nan"))),
        "n_utterances_probed": seen,
        "per_layer": probe.summarise(layer_order),
    }


def find_checkpoints(run_dir: Path, targets: list) -> dict:
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
                for fb in ("checkpoint_ep10.pt", "checkpoint_ep20.pt"):
                    q = run_dir / fb
                    if q.exists():
                        label = fb.replace("checkpoint_", "").replace(".pt", "")
                        found[label + "_surrogate_for_ep15"] = q
        else:
            p = run_dir / f"checkpoint_{t}.pt"
            if p.exists():
                found[t] = p
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-utterances", type=int, default=64)
    parser.add_argument("--checkpoints", nargs="*",
                        default=["ep15", "ep30", "best"])
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

    if not cfg.backbone.startswith("mamba2"):
        raise SystemExit(
            f"diagnose_mamba2.py supports the mamba2_* backbone family; "
            f"got {cfg.backbone!r}. Run dir: {run_dir}"
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
        "spec": "Stage 11.0b / 11.1a diagnostics (Mamba-2 SSD state + Δt + alpha)",
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
