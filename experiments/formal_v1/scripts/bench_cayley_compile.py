#!/usr/bin/env python3
"""Quick torch.compile benchmark for the Cayley rank-1 scan."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import load_config
from src.data.vocab import CharVocab
from src.models.asr_model import ASRModel
from src.utils.misc import seed_everything


def bench(mode: str | None, n_iters: int = 10, warmup: int = 3):
    cfg = load_config("configs/default.yaml", {"backbone": "rwkv6_orthogonal", "seed": 42})
    vocab = CharVocab.build_english()
    seed_everything(42)
    m = ASRModel(vocab_size=vocab.size, cfg=cfg).cuda().train()

    if mode is not None:
        m.encoder = torch.compile(m.encoder, mode=mode, dynamic=False, fullgraph=False)

    B, T_mel = 10, 1200
    mels = torch.randn(B, cfg.n_mels, T_mel, device="cuda")
    lengths = torch.full((B,), T_mel, device="cuda", dtype=torch.long)

    print(f"  [warmup={warmup}, iters={n_iters}, mode={mode}]")
    for _ in range(warmup):
        m.zero_grad(set_to_none=True)
        lp, ol, _ = m(mels, lengths)
        lp.sum().backward()
    torch.cuda.synchronize()

    ts = []
    t0_all = time.time()
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.time()
        m.zero_grad(set_to_none=True)
        lp, ol, _ = m(mels, lengths)
        lp.sum().backward()
        torch.cuda.synchronize()
        ts.append((time.time() - t0) * 1000)

    peak = torch.cuda.max_memory_allocated() / 1e9
    mean = sum(ts) / len(ts)
    stdev = (sum((t - mean) ** 2 for t in ts) / len(ts)) ** 0.5
    print(f"  mean: {mean:.1f} ms  stdev: {stdev:.1f}  min: {min(ts):.1f}  max: {max(ts):.1f}  peak VRAM: {peak:.2f} GB")
    return mean


if __name__ == "__main__":
    print("=== eager (no compile) ===")
    t_eager = bench(None)
    torch.cuda.empty_cache()

    print("\n=== torch.compile mode='default' ===")
    try:
        t_default = bench("default")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e!r}")
        t_default = None

    print("\n=== Speedup vs eager ===")
    if t_default:
        print(f"  default:         {t_eager / t_default:.2f}×  ({t_default:.0f} ms)")
