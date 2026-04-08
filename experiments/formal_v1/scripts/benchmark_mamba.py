#!/usr/bin/env python3
"""Benchmark: PyTorch Mamba reimplementation vs CUDA mamba-ssm.

Compares training speed, inference speed, and memory usage for both
implementations at various sequence lengths and batch sizes.

Usage:
    uv run scripts/benchmark_mamba.py
    uv run scripts/benchmark_mamba.py --compile   # also benchmark torch.compile
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.set_float32_matmul_precision("high")

from src.models.mamba_encoder import MambaEncoder
from src.models.mamba_cuda_encoder import MambaCudaEncoder


def benchmark_fn(fn, warmup=5, repeat=20):
    """Time a function with CUDA synchronization."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat


def measure_memory(fn):
    """Measure peak GPU memory of a function."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    fn()
    return torch.cuda.max_memory_allocated() / 1e6


def run_benchmark(B, T, D, n_layers, use_compile, device):
    """Run a single benchmark configuration."""
    x = torch.randn(B, T, D, device=device)
    lengths = torch.full((B,), T, device=device, dtype=torch.long)

    pytorch_enc = MambaEncoder(d_model=D, n_layers=n_layers, dropout=0.1).to(device).train()
    cuda_enc = MambaCudaEncoder(d_model=D, n_layers=n_layers, dropout=0.1).to(device).train()

    results = {}

    # --- PyTorch eager ---
    def fwd_bwd_eager():
        out, _ = pytorch_enc(x, lengths)
        out.sum().backward()
        pytorch_enc.zero_grad()

    results["pytorch_eager_fwd_bwd_ms"] = benchmark_fn(fwd_bwd_eager) * 1000
    results["pytorch_eager_mem_mb"] = measure_memory(fwd_bwd_eager)

    # --- torch.compile ---
    if use_compile:
        compiled_enc = torch.compile(pytorch_enc, mode="default")
        # Warmup compilation
        print(f"    Compiling for B={B}, T={T}...", end=" ", flush=True)
        t0 = time.perf_counter()
        for _ in range(3):
            out, _ = compiled_enc(x, lengths)
            out.sum().backward()
            compiled_enc.zero_grad()
        torch.cuda.synchronize()
        compile_time = time.perf_counter() - t0
        print(f"{compile_time:.0f}s")

        def fwd_bwd_compiled():
            out, _ = compiled_enc(x, lengths)
            out.sum().backward()
            compiled_enc.zero_grad()

        results["pytorch_compiled_fwd_bwd_ms"] = benchmark_fn(fwd_bwd_compiled) * 1000
        results["pytorch_compiled_mem_mb"] = measure_memory(fwd_bwd_compiled)
        results["compile_time_s"] = compile_time

    # --- CUDA mamba-ssm ---
    def fwd_bwd_cuda():
        out, _ = cuda_enc(x, lengths)
        out.sum().backward()
        cuda_enc.zero_grad()

    results["cuda_fwd_bwd_ms"] = benchmark_fn(fwd_bwd_cuda) * 1000
    results["cuda_mem_mb"] = measure_memory(fwd_bwd_cuda)

    # --- Inference ---
    pytorch_enc.eval()
    cuda_enc.eval()

    def infer_pytorch():
        with torch.no_grad():
            pytorch_enc(x, lengths)

    def infer_cuda():
        with torch.no_grad():
            cuda_enc(x, lengths)

    results["pytorch_infer_ms"] = benchmark_fn(infer_pytorch, warmup=3, repeat=30) * 1000
    results["cuda_infer_ms"] = benchmark_fn(infer_cuda, warmup=3, repeat=30) * 1000

    results["params"] = sum(p.numel() for p in pytorch_enc.parameters())
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Also benchmark torch.compile (slow first-run)")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")

    # Standard config: d_model=256, 6 layers (matching thesis experiments)
    D, n_layers = 256, 6

    configs = [
        # (batch_size, seq_len) — representative of ASR after 4× downsample
        (4, 250),   # short utterances
        (8, 500),   # typical batch
        (4, 1000),  # long utterances
        (16, 250),  # large batch, short
    ]

    all_results = []
    for B, T in configs:
        print(f"  B={B}, T={T}, D={D}, layers={n_layers}")
        result = run_benchmark(B, T, D, n_layers, args.compile, device)
        result["B"] = B
        result["T"] = T
        all_results.append(result)

        # Print summary
        eager = result["pytorch_eager_fwd_bwd_ms"]
        cuda = result["cuda_fwd_bwd_ms"]
        ratio = eager / cuda
        print(f"    Eager fwd+bwd: {eager:6.1f} ms | CUDA: {cuda:6.1f} ms | ratio: {ratio:.1f}×")
        if args.compile and "pytorch_compiled_fwd_bwd_ms" in result:
            compiled = result["pytorch_compiled_fwd_bwd_ms"]
            ratio_c = compiled / cuda
            print(f"    Compiled fwd+bwd: {compiled:6.1f} ms | ratio vs CUDA: {ratio_c:.2f}×")
        print(f"    Infer: PyTorch {result['pytorch_infer_ms']:.1f}ms | CUDA {result['cuda_infer_ms']:.1f}ms")
        print(f"    Memory: PyTorch {result['pytorch_eager_mem_mb']:.0f}MB | CUDA {result['cuda_mem_mb']:.0f}MB")
        print()

    # Summary table
    print("=" * 80)
    print(f"{'Config':>12} | {'Eager':>10} | {'Compiled':>10} | {'CUDA':>10} | {'Eager/CUDA':>10} | {'Comp/CUDA':>10}")
    print("-" * 80)
    for r in all_results:
        config = f"B{r['B']}×T{r['T']}"
        eager = f"{r['pytorch_eager_fwd_bwd_ms']:.1f}ms"
        compiled = f"{r.get('pytorch_compiled_fwd_bwd_ms', 0):.1f}ms" if args.compile else "—"
        cuda = f"{r['cuda_fwd_bwd_ms']:.1f}ms"
        ratio_e = f"{r['pytorch_eager_fwd_bwd_ms'] / r['cuda_fwd_bwd_ms']:.1f}×"
        ratio_c = f"{r.get('pytorch_compiled_fwd_bwd_ms', 0) / r['cuda_fwd_bwd_ms']:.2f}×" if args.compile else "—"
        print(f"{config:>12} | {eager:>10} | {compiled:>10} | {cuda:>10} | {ratio_e:>10} | {ratio_c:>10}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
