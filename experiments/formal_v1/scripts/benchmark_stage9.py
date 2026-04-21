#!/usr/bin/env python3
"""Benchmark Stage-9 sparse nonnormal_rse vs T2 dense vs RSE anchor."""
from __future__ import annotations

import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.config import ExperimentConfig
from src.models.asr_model import ASRModel


VOCAB_SIZE = 29


def bench(backbone: str, device, n_warmup=3, n_iter=8):
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
        lp, _, _ = model(mels, mel_lens)
        loss = -lp[:, :target_len].gather(-1, targets.unsqueeze(-1)).mean()
        loss.backward(); optim.step()
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        optim.zero_grad(set_to_none=True)
        lp, _, _ = model(mels, mel_lens)
        loss = -lp[:, :target_len].gather(-1, targets.unsqueeze(-1)).mean()
        loss.backward(); optim.step()
    torch.cuda.synchronize(device)
    dt = (time.perf_counter() - t0) / n_iter
    mem = torch.cuda.max_memory_allocated(device) / 1e9
    return dt, mem


def main():
    device = torch.device("cuda:1")
    print(f"Benchmarking on {torch.cuda.get_device_name(device)}\n")
    for bb in [
        "rwkv6_rse_strong_viscosity",
        "rwkv6_nonnormal_rse_viscosity",
        "rwkv6_sparse_nonnormal_rse_viscosity",
        "rwkv6_sparse_nonnormal_rse_edge_only_viscosity",
    ]:
        print(f"[{bb}] …", flush=True)
        dt, mem = bench(bb, device)
        print(f"  {bb:50s}  {dt*1000:7.1f} ms/iter   {mem:5.2f} GB peak", flush=True)


if __name__ == "__main__":
    main()
