#!/bin/bash
# Extra LUCID experiments — run on GPU 0 after the baseline finishes
#
# Exp 1: lion_convshift (30 ep, ~107s/ep ≈ 55 min) — best mechanism from draft
# Exp 2: rwkv6_lucid_sr (30 ep, ~110s/ep ≈ 55 min) — RKHS delta self-regulation
#
# Usage:
#   bash scripts/launch_lucid_extras.sh

set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

echo "=== Extra LUCID experiments on GPU 0 at $(date) ==="

echo "[1/2] Starting lion_convshift (30 epochs)..."
uv run python scripts/run_experiment.py \
    --backbone lion_convshift \
    --epochs 30 \
    --output-dir outputs/lucid_exp05_lion_convshift_seed42 \
    --gpu 0 \
    2>&1 | tee "$LOG_DIR/lucid_gpu0_convshift.log"

echo "[2/2] Starting rwkv6_lucid_sr (30 epochs)..."
uv run python scripts/run_experiment.py \
    --backbone rwkv6_lucid_sr \
    --epochs 30 \
    --output-dir outputs/lucid_exp06_rwkv6_lucid_sr_seed42 \
    --gpu 0 \
    2>&1 | tee "$LOG_DIR/lucid_gpu0_sr.log"

echo "=== Extra experiments complete at $(date) ==="
