#!/bin/bash
# LUCID comparison: 4 experiments (2 per GPU), 30 epochs each
#
# GPU 0: rwkv6 baseline → lion baseline  (sequential)
# GPU 1: rwkv6_lucid   → lion_lucid     (sequential)
#
# Expected time: ~30 epochs × ~85s/epoch ≈ 45 min per run, ~90 min per GPU
#
# Usage:
#   bash scripts/launch_lucid_comparison.sh

set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

echo "=== LUCID Comparison launched at $(date) ==="
echo "4 runs: {rwkv6, lion} × {baseline, lucid}, 30 epochs each"
echo ""

# ── GPU 0: Baselines (sequential) ──────────────────────────────────
(
    echo "[GPU0] Starting rwkv6 baseline..."
    uv run python scripts/run_experiment.py \
        --backbone rwkv6 \
        --epochs 30 \
        --output-dir outputs/lucid_exp01_rwkv6_seed42 \
        --gpu 0

    echo "[GPU0] Starting lion baseline..."
    uv run python scripts/run_experiment.py \
        --backbone lion \
        --epochs 30 \
        --output-dir outputs/lucid_exp02_lion_seed42 \
        --gpu 0
) > "$LOG_DIR/lucid_gpu0.log" 2>&1 &
PID_GPU0=$!
echo "GPU 0 launched (PID=$PID_GPU0) → $LOG_DIR/lucid_gpu0.log"

# ── GPU 1: LUCID variants (sequential) ────────────────────────────
(
    echo "[GPU1] Starting rwkv6_lucid..."
    uv run python scripts/run_experiment.py \
        --backbone rwkv6_lucid \
        --epochs 30 \
        --output-dir outputs/lucid_exp03_rwkv6_lucid_seed42 \
        --gpu 1

    echo "[GPU1] Starting lion_lucid..."
    uv run python scripts/run_experiment.py \
        --backbone lion_lucid \
        --epochs 30 \
        --output-dir outputs/lucid_exp04_lion_lucid_seed42 \
        --gpu 1
) > "$LOG_DIR/lucid_gpu1.log" 2>&1 &
PID_GPU1=$!
echo "GPU 1 launched (PID=$PID_GPU1) → $LOG_DIR/lucid_gpu1.log"

echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/lucid_gpu0.log"
echo "  tail -f $LOG_DIR/lucid_gpu1.log"
echo ""
echo "Waiting for both GPUs to finish..."

wait $PID_GPU0
RC0=$?
echo "[GPU0] Finished (exit=$RC0) at $(date)"

wait $PID_GPU1
RC1=$?
echo "[GPU1] Finished (exit=$RC1) at $(date)"

echo ""
echo "=== LUCID Comparison complete at $(date) ==="
echo "  GPU 0 (baselines): exit=$RC0"
echo "  GPU 1 (LUCID):     exit=$RC1"
echo ""
echo "Check results:"
echo "  cat outputs/lucid_exp01_rwkv6_seed42/results.json | python -m json.tool | head -20"
echo "  cat outputs/lucid_exp03_rwkv6_lucid_seed42/results.json | python -m json.tool | head -20"
