#!/usr/bin/env python3
"""Run Mamba comparison: PyTorch reimplementation vs CUDA mamba-ssm.

Runs both experiments IN PARALLEL on separate GPUs, then compares:
  - Accuracy: CER, WER on dev/test
  - Speed: wall-clock training time per epoch
  - Memory: peak GPU memory during training

Usage:
    uv run scripts/run_mamba_comparison.py --epochs 10
    uv run scripts/run_mamba_comparison.py --epochs 10 --compile
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def launch_experiment(backbone, epochs, seed, compile_flag, output_dir, gpu_id):
    """Launch a single experiment as a subprocess on a specific GPU."""
    cmd = [
        sys.executable, "scripts/run_experiment.py",
        "--config", "configs/default.yaml",
        "--backbone", backbone,
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--gpu", str(gpu_id),
    ]
    if compile_flag:
        cmd.append("--compile")

    log_path = os.path.join(output_dir, "train.log")
    os.makedirs(output_dir, exist_ok=True)
    log_file = open(log_path, "w")

    logger.info(f"Launching on GPU {gpu_id}: {backbone} → {output_dir}")
    logger.info(f"  Command: {' '.join(cmd)}")
    logger.info(f"  Log: {log_path}")

    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def load_results(output_dir):
    """Load results.json from an experiment directory."""
    path = os.path.join(output_dir, "results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for PyTorch Mamba")
    parser.add_argument("--output-dir", default="outputs/mamba_comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    suffix = "_compiled" if args.compile else ""
    pytorch_dir = os.path.join(args.output_dir, f"mamba_pytorch{suffix}_ep{args.epochs}_seed{args.seed}")
    cuda_dir = os.path.join(args.output_dir, f"mamba_cuda_ep{args.epochs}_seed{args.seed}")

    # Launch both experiments in parallel on separate GPUs
    t0 = time.time()
    proc_pt, log_pt = launch_experiment("mamba", args.epochs, args.seed, args.compile, pytorch_dir, gpu_id=0)
    proc_cu, log_cu = launch_experiment("mamba_cuda", args.epochs, args.seed, False, cuda_dir, gpu_id=1)

    logger.info("Both experiments running in parallel (GPU 0: PyTorch, GPU 1: CUDA)")
    logger.info("Waiting for completion...")

    # Wait for both to finish
    rc_pt = proc_pt.wait()
    log_pt.close()
    logger.info(f"PyTorch Mamba finished (exit code {rc_pt})")

    rc_cu = proc_cu.wait()
    log_cu.close()
    logger.info(f"CUDA Mamba finished (exit code {rc_cu})")

    wall_time = time.time() - t0
    logger.info(f"Total wall-clock time: {wall_time:.0f}s")

    if rc_pt != 0:
        logger.error(f"PyTorch experiment failed. Check log: {pytorch_dir}/train.log")
    if rc_cu != 0:
        logger.error(f"CUDA experiment failed. Check log: {cuda_dir}/train.log")

    # Compare results
    pytorch_results = load_results(pytorch_dir)
    cuda_results = load_results(cuda_dir)

    if pytorch_results and cuda_results:
        print("\n" + "=" * 70)
        print("MAMBA COMPARISON RESULTS")
        print("=" * 70)

        for name, results in [("PyTorch Mamba" + (" (compiled)" if args.compile else ""), pytorch_results),
                               ("CUDA mamba-ssm", cuda_results)]:
            print(f"\n{name}:")
            print(f"  Params:       {results['params']['total']:,}")
            print(f"  Best dev CER: {results['best_dev_cer']:.4f}")
            print(f"  Test CER:     {results['test']['cer']:.4f}")
            print(f"  Test WER:     {results['test']['wer']:.4f}")

            epoch_times = [h["epoch_time_sec"] for h in results["history"]]
            avg_time = sum(epoch_times) / len(epoch_times)
            total_time = sum(epoch_times)
            print(f"  Avg epoch:    {avg_time:.1f}s")
            print(f"  Total time:   {total_time:.0f}s")

        # Save comparison summary
        comparison = {
            "wall_clock_sec": wall_time,
            "pytorch": {
                "compiled": args.compile,
                "gpu": 0,
                "best_dev_cer": pytorch_results["best_dev_cer"],
                "test_cer": pytorch_results["test"]["cer"],
                "test_wer": pytorch_results["test"]["wer"],
                "params": pytorch_results["params"]["total"],
                "avg_epoch_sec": sum(h["epoch_time_sec"] for h in pytorch_results["history"]) / len(pytorch_results["history"]),
                "total_time_sec": sum(h["epoch_time_sec"] for h in pytorch_results["history"]),
                "history": [{"epoch": h["epoch"], "train_loss": h["train_loss"],
                             "dev_cer": h["dev_cer"], "epoch_time": h["epoch_time_sec"]}
                            for h in pytorch_results["history"]],
            },
            "cuda": {
                "gpu": 1,
                "best_dev_cer": cuda_results["best_dev_cer"],
                "test_cer": cuda_results["test"]["cer"],
                "test_wer": cuda_results["test"]["wer"],
                "params": cuda_results["params"]["total"],
                "avg_epoch_sec": sum(h["epoch_time_sec"] for h in cuda_results["history"]) / len(cuda_results["history"]),
                "total_time_sec": sum(h["epoch_time_sec"] for h in cuda_results["history"]),
                "history": [{"epoch": h["epoch"], "train_loss": h["train_loss"],
                             "dev_cer": h["dev_cer"], "epoch_time": h["epoch_time_sec"]}
                            for h in cuda_results["history"]],
            },
        }
        out_path = os.path.join(args.output_dir, "comparison.json")
        with open(out_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {out_path}")
    else:
        if not pytorch_results:
            logger.error(f"No results from PyTorch experiment. Check: {pytorch_dir}/train.log")
        if not cuda_results:
            logger.error(f"No results from CUDA experiment. Check: {cuda_dir}/train.log")


if __name__ == "__main__":
    main()
