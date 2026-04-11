#!/usr/bin/env python3
"""Measure streaming-inference state size vs audio duration.

For each causal encoder in Group A, stream a 30-second utterance through
the model in chunks of increasing size and record:

  * state_bytes     : total bytes in the encoder carry state after the
                      last chunk (i.e. `sum(t.numel()*t.element_size())`
                      over every tensor in the state dict).
  * peak_vram_gb    : `torch.cuda.max_memory_allocated()` during the
                      forward pass.
  * chunk_sec       : the chunk length used (duration of one streaming step).
  * duration_sec    : total audio duration streamed so far.

Writes `outputs/_streaming_memory.csv`. Pair with
`src/reporting/plots/streaming_memory.py` to produce the thesis figure.

Usage:
    uv run scripts/measure_streaming_memory.py --gpu 0
    uv run scripts/measure_streaming_memory.py --backbones mamba,rwkv6,transformer_causal

The backbones this script knows how to measure (Group A — causal, streaming-
capable) are `mamba`, `rwkv6`, `transformer_causal`. Non-causal backbones
have no meaningful carry state.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import ExperimentConfig
from src.models.encoder import build_encoder


SUPPORTED_BACKBONES = ["mamba", "rwkv6", "transformer_causal"]

# The chunk lengths we sweep, in seconds of raw audio. With the standard
# 10 ms hop and 4× downsampling in the frontend, each chunk corresponds to
# chunk_sec * 25 post-frontend frames.
CHUNK_SECONDS = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
TOTAL_DURATION_SEC = 30.0


def _state_bytes(state) -> int:
    """Recursively sum bytes of all tensors in a carry state structure."""
    if state is None:
        return 0
    if isinstance(state, torch.Tensor):
        return state.numel() * state.element_size()
    if isinstance(state, dict):
        return sum(_state_bytes(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return sum(_state_bytes(v) for v in state)
    return 0


def _make_fake_frontend_output(
    duration_sec: float, cfg: ExperimentConfig, device: torch.device,
) -> torch.Tensor:
    """Produce a random (B=1, T', d_model) tensor matching what the
    frontend would output for an audio clip of `duration_sec` seconds."""
    hop_ms = cfg.hop_length_ms  # e.g. 10
    frames = int(duration_sec * 1000 / hop_ms)
    # ConvSubsampling does 4× temporal downsampling
    T = max(1, frames // 4)
    return torch.randn(1, T, cfg.d_model, device=device)


@torch.no_grad()
def measure_backbone(
    backbone: str,
    device: torch.device,
    chunk_seconds: Iterable[float] = CHUNK_SECONDS,
    total_duration_sec: float = TOTAL_DURATION_SEC,
) -> list[dict]:
    """Stream a simulated utterance through an encoder, measure state size."""
    cfg = ExperimentConfig()
    cfg.backbone = backbone
    encoder = build_encoder(cfg).to(device).eval()

    rows: list[dict] = []

    for chunk_sec in chunk_seconds:
        x_full = _make_fake_frontend_output(total_duration_sec, cfg, device)
        T_full = x_full.shape[1]

        # Number of frames per chunk, clamped to at least 1.
        chunk_frames = max(1, int(chunk_sec * 1000 / cfg.hop_length_ms) // 4)

        state = encoder.init_state(batch_size=1, device=device)
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

        frames_seen = 0
        final_state = state
        for t0 in range(0, T_full, chunk_frames):
            t1 = min(t0 + chunk_frames, T_full)
            lengths = torch.tensor([t1 - t0], dtype=torch.long, device=device)
            _, final_state = encoder(x_full[:, t0:t1], lengths, state=final_state)
            frames_seen = t1

        state_bytes = _state_bytes(final_state)
        peak_vram = (
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda" else 0.0
        )

        rows.append({
            "backbone": backbone,
            "chunk_sec": chunk_sec,
            "duration_sec": frames_seen * cfg.hop_length_ms * 4 / 1000,
            "frames_seen": frames_seen,
            "state_bytes": state_bytes,
            "peak_vram_gb": peak_vram,
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbones",
        default=",".join(SUPPORTED_BACKBONES),
        help="Comma-separated list of causal backbones to measure.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--output",
        default="outputs/_streaming_memory.csv",
        help="CSV path for measurements.",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    all_rows: list[dict] = []
    for backbone in backbones:
        if backbone not in SUPPORTED_BACKBONES:
            print(f"[skip] {backbone}: not in SUPPORTED_BACKBONES {SUPPORTED_BACKBONES}")
            continue
        print(f"[measure] {backbone}")
        rows = measure_backbone(backbone, device)
        for r in rows:
            print(
                f"  chunk={r['chunk_sec']:>5.1f}s  "
                f"state={r['state_bytes']:>10d} B  "
                f"peak_vram={r['peak_vram_gb']:.3f} GB"
            )
        all_rows.extend(rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "backbone", "chunk_sec", "duration_sec", "frames_seen",
            "state_bytes", "peak_vram_gb",
        ])
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"\nSaved {len(all_rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
