#!/usr/bin/env python3
"""Quick benchmark: does torch.jit.script help the Stage-8 scans?

Tests the sequential scans in three modes:
  1. eager (current)
  2. torch.jit.script on the scan function
  3. torch.compile with dynamic shape hint (small-T test)

Prints forward+backward times on a realistic shape.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.models.rwkv6_time_mix import (
    _recurrent_delta_scan, _recurrent_nonnormal_rse_scan,
)


def make_delta_inputs(B=10, H=4, T=300, K=64, device="cuda:0"):
    r = torch.randn(B, H, T, K, device=device, requires_grad=True)
    k = torch.randn(B, H, T, K, device=device, requires_grad=True)
    v = torch.randn(B, H, T, K, device=device, requires_grad=True)
    w = torch.randn(B, H, T, K, device=device, requires_grad=True) * 0.01 - 0.1
    u = torch.randn(1, H, 1, K, device=device, requires_grad=True)
    kk = torch.randn(B, H, T, K, device=device, requires_grad=True)
    iclr = torch.rand(B, H, T, K, device=device, requires_grad=True) * 0.2
    gate = torch.zeros(H, device=device, requires_grad=True)
    state = torch.zeros(B, H, K, K, device=device)
    return (r, k, v, w, u, kk, iclr, gate, state)


def make_nn_inputs(B=10, H=4, T=300, K=64, device="cuda:0"):
    Bk = K // 2
    r = torch.randn(B, H, T, K, device=device, requires_grad=True)
    k = torch.randn(B, H, T, K, device=device, requires_grad=True)
    v = torch.randn(B, H, T, K, device=device, requires_grad=True)
    w = torch.randn(B, H, T, K, device=device, requires_grad=True) * 0.01 - 0.1
    theta = torch.randn(B, H, T, Bk, device=device, requires_grad=True) * 0.1
    rho = torch.randn(B, H, T, Bk, device=device, requires_grad=True) * 0.05
    psi = torch.randn(B, H, T, Bk, device=device, requires_grad=True)
    u = torch.randn(1, H, 1, K, device=device, requires_grad=True)
    state = torch.zeros(B, H, K, K, device=device)
    return (r, k, v, w, theta, rho, psi, u, state)


def time_call(fn, args, n_warmup=3, n_iter=5):
    device = args[0].device
    def step():
        # Re-create fresh leaf tensors so backward can retrace
        fresh = tuple(a.detach().clone().requires_grad_(a.requires_grad) if isinstance(a, torch.Tensor) else a for a in args)
        out = fn(*fresh)
        loss = out[0].sum()
        loss.backward()
    for _ in range(n_warmup):
        step()
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        step()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / n_iter


def main() -> int:
    device = torch.device("cuda:0")
    print(f"Bench on {torch.cuda.get_device_name(device)}\n")

    # Delta
    print("=== Delta scan ===")
    args = make_delta_inputs(device=device)
    t_eager = time_call(_recurrent_delta_scan, args)
    print(f"  eager:          {t_eager*1000:7.1f} ms")

    try:
        scripted = torch.jit.script(_recurrent_delta_scan)
        t_script = time_call(scripted, args)
        print(f"  torch.jit.script: {t_script*1000:7.1f} ms   {t_eager/t_script:.2f}×")
    except Exception as e:
        print(f"  torch.jit.script: FAILED — {type(e).__name__}: {str(e)[:120]}")

    # Non-normal RSE
    print("\n=== Non-normal RSE scan ===")
    args = make_nn_inputs(device=device)
    t_eager = time_call(_recurrent_nonnormal_rse_scan, args)
    print(f"  eager:          {t_eager*1000:7.1f} ms")

    try:
        scripted = torch.jit.script(_recurrent_nonnormal_rse_scan)
        t_script = time_call(scripted, args)
        print(f"  torch.jit.script: {t_script*1000:7.1f} ms   {t_eager/t_script:.2f}×")
    except Exception as e:
        print(f"  torch.jit.script: FAILED — {type(e).__name__}: {str(e)[:120]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
