#!/usr/bin/env python3
"""Benchmark the Stage-8 scan forward+backward with and without torch.compile.

Measures average wall-clock time per forward+backward on a realistic-shape
batch (B=10, T=300, H=4, K=64) for:
  - vanilla rwkv6 (reference chunked _chunked_wkv, baseline)
  - rwkv6_delta_warmstart_fixed (sequential delta scan)
  - rwkv6_nonnormal_rse_viscosity (sequential nonnormal_rse scan)

Each variant is tested with and without torch.compile(mode="reduce-overhead").
"""

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


def bench(backbone: str, device: torch.device, n_warmup: int = 3, n_iter: int = 10,
          compile_encoder: bool = False) -> tuple[float, float]:
    """Return (fwd+bwd time avg, peak memory GB)."""
    torch.manual_seed(42)
    cfg = ExperimentConfig()
    cfg.backbone = backbone
    cfg.rwkv_mode = "recurrent"
    if compile_encoder:
        cfg.compile_encoder = True
    model = ASRModel(vocab_size=VOCAB_SIZE, cfg=cfg).to(device)
    if compile_encoder:
        model.encoder = torch.compile(model.encoder, mode="reduce-overhead", dynamic=True)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    B, n_mels, T = 10, 80, 1200  # T=1200 mel → ~300 encoded tokens
    mels = torch.randn(B, n_mels, T, device=device)
    mel_lens = torch.tensor([T] * B, device=device)
    # Dummy targets — token classification loss just to trigger backward
    target_len = 50
    targets = torch.randint(0, VOCAB_SIZE, (B, target_len), device=device)

    model.train()
    torch.cuda.reset_peak_memory_stats(device)

    # Warmup
    for _ in range(n_warmup):
        optim.zero_grad(set_to_none=True)
        log_probs, out_lens, _ = model(mels, mel_lens)
        # crude loss: mean of log_probs at first target_len positions
        loss = -log_probs[:, :target_len].gather(-1, targets.unsqueeze(-1)).mean()
        loss.backward()
        optim.step()
    torch.cuda.synchronize(device)

    # Time
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        optim.zero_grad(set_to_none=True)
        log_probs, out_lens, _ = model(mels, mel_lens)
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
        ("rwkv6",                           False),   # baseline (chunked, already fast)
        ("rwkv6_delta_warmstart_fixed",     False),
        ("rwkv6_delta_warmstart_fixed",     True),    # + torch.compile
        ("rwkv6_nonnormal_rse_viscosity",   False),
        ("rwkv6_nonnormal_rse_viscosity",   True),    # + torch.compile
    ]
    results = []
    for backbone, compile_enc in configs:
        label = f"{backbone}{' [compiled]' if compile_enc else ''}"
        print(f"[{label}] building + warmup …")
        try:
            dt, mem = bench(backbone, device, n_warmup=3, n_iter=5, compile_encoder=compile_enc)
            print(f"  {label:60s}  {dt*1000:8.1f} ms/iter   {mem:6.2f} GB peak")
            results.append((label, dt, mem))
        except Exception as e:
            print(f"  {label}: FAILED — {type(e).__name__}: {e}")
            results.append((label, None, None))

    print("\nSummary:")
    baseline_dt = results[0][1] if results[0][1] is not None else None
    for label, dt, mem in results:
        if dt is None:
            print(f"  {label:60s}  FAILED")
        elif baseline_dt is not None and dt > 0:
            print(f"  {label:60s}  {dt*1000:8.1f} ms   {dt/baseline_dt:5.2f}×  {mem:5.2f} GB")
        else:
            print(f"  {label:60s}  {dt*1000:8.1f} ms   {mem:5.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
