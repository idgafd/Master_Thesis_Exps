#!/usr/bin/env python3
"""Benchmark: Mamba-2 (3 modes) vs existing Mamba-1 encoder.

Shapes are representative of LibriSpeech-clean ASR after 4× down-sample.

Usage:
    uv run scripts/benchmark_mamba2.py
    uv run scripts/benchmark_mamba2.py --compile
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.set_float32_matmul_precision("high")

from src.models.mamba_encoder import MambaEncoder
from src.models.mamba2_encoder import Mamba2Encoder

D, N_LAYERS, FFN = 256, 6, 896


def _bench_fwdbwd(fn, warmup=3, repeat=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000


def _peak_mem(fn):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    fn()
    return torch.cuda.max_memory_allocated() / 1e9


def _make(encoder_cls, mode, device):
    if encoder_cls is MambaEncoder:
        return MambaEncoder(d_model=D, n_layers=N_LAYERS, dropout=0.1,
                            ffn_dim=FFN).to(device).train()
    return Mamba2Encoder(
        d_model=D, n_layers=N_LAYERS, dropout=0.1, ffn_dim=FFN,
        d_state=64, headdim=64, ngroups=1, chunk_size=64,
        mode=mode,
    ).to(device).train()


def run(B, T, compile_mode, device):
    x = torch.randn(B, T, D, device=device)
    lengths = torch.full((B,), T, device=device, dtype=torch.long)
    rows = []
    specs = [
        ("mamba-1 (HS scan)", MambaEncoder, None),
        ("mamba-2 recurrent", Mamba2Encoder, "recurrent"),
        ("mamba-2 lion",      Mamba2Encoder, "lion"),
        ("mamba-2 lion_chunk", Mamba2Encoder, "lion_chunk"),
    ]
    for name, cls, mode in specs:
        enc = _make(cls, mode, device)
        if compile_mode:
            enc = torch.compile(enc, mode=compile_mode, dynamic=False)

        def step():
            out, _ = enc(x, lengths)
            out.sum().backward()
            enc.zero_grad(set_to_none=True)

        try:
            ms = _bench_fwdbwd(step)
            mem = _peak_mem(step)
            rows.append((name, ms, mem, "ok"))
        except torch.cuda.OutOfMemoryError as e:
            rows.append((name, float("nan"), float("nan"), "OOM"))
        del enc
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", nargs="?", const="default", default=None,
                    choices=[None, "default", "reduce-overhead", "max-autotune"],
                    help="torch.compile mode (None = eager)")
    args = ap.parse_args()

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}   "
          f"compile: {args.compile or 'eager'}")
    print(f"d_model={D}  n_layers={N_LAYERS}  ffn={FFN}\n")

    for B, T in [(8, 500), (8, 1000), (16, 500)]:
        print(f"  B={B}  T={T}")
        rows = run(B, T, args.compile, device)
        for name, ms, mem, status in rows:
            if status == "OOM":
                print(f"    {name:<22s}  OOM")
            else:
                print(f"    {name:<22s}  {ms:7.1f} ms   peak {mem:5.1f} GB")
        print()


if __name__ == "__main__":
    main()
