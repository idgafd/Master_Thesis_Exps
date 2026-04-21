#!/usr/bin/env python3
"""Benchmark the Stage-8 scan variants after the chunked affine rewrite."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.config import ExperimentConfig
from src.models.asr_model import ASRModel


VOCAB_SIZE = 29


def bench(backbone: str, device: torch.device, n_warmup: int = 3, n_iter: int = 8) -> tuple[float, float]:
    torch.manual_seed(42)
    cfg = ExperimentConfig()
    cfg.backbone = backbone
    cfg.rwkv_mode = "recurrent"
    model = ASRModel(vocab_size=VOCAB_SIZE, cfg=cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    B, n_mels, T = 10, 80, 1200
    mels = torch.randn(B, n_mels, T, device=device)
    mel_lens = torch.tensor([T] * B, device=device)
    target_len = 50
    targets = torch.randint(0, VOCAB_SIZE, (B, target_len), device=device)

    model.train()
    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(n_warmup):
        optim.zero_grad(set_to_none=True)
        log_probs, _, _ = model(mels, mel_lens)
        loss = -log_probs[:, :target_len].gather(-1, targets.unsqueeze(-1)).mean()
        loss.backward()
        optim.step()
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        optim.zero_grad(set_to_none=True)
        log_probs, _, _ = model(mels, mel_lens)
        loss = -log_probs[:, :target_len].gather(-1, targets.unsqueeze(-1)).mean()
        loss.backward()
        optim.step()
    torch.cuda.synchronize(device)
    dt = (time.perf_counter() - t0) / n_iter
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    return dt, peak_mem


def main() -> int:
    device = torch.device("cuda:0")
    print(f"Benchmarking on {torch.cuda.get_device_name(device)}\n")

    configs = [
        "rwkv6",
        "rwkv6_rse_strong_viscosity",          # RSE anchor (chunked complex scan)
        "rwkv6_delta_warmstart_fixed",         # T1 (sequential delta)
        "rwkv6_nonnormal_rse_viscosity",       # T2 (chunked affine AFTER rewrite)
    ]
    results = []
    for backbone in configs:
        print(f"[{backbone}] …", flush=True)
        try:
            dt, mem = bench(backbone, device)
            print(f"  {backbone:50s}  {dt*1000:7.1f} ms/iter   {mem:5.2f} GB peak", flush=True)
            results.append((backbone, dt, mem))
        except Exception as e:
            print(f"  {backbone}: FAILED — {type(e).__name__}: {e}", flush=True)
            results.append((backbone, None, None))

    print("\nSpeed relative to rwkv6 baseline:")
    baseline_dt = results[0][1] if results[0][1] else None
    for label, dt, mem in results:
        if dt is None:
            print(f"  {label:50s}  FAILED")
        elif baseline_dt:
            print(f"  {label:50s}  {dt*1000:7.1f} ms   {dt/baseline_dt:5.2f}×  {mem:5.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
