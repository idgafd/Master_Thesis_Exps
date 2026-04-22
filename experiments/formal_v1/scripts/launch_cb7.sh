#!/bin/bash
# CB-7: rwkv6_qtail_lowrank_all_convshift_multidil_symmetric.
# Cross-channel × temporal composition — cheapest orthogonal test post-CB-1.
# GPU 0, ~2.2 h dry-run / ~3 h real training.

set -u
cd "$(dirname "$0")/.."

LOG_DIR="outputs/_cb7_launch_logs"
mkdir -p "$LOG_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "[$(date +%H:%M:%S)] GPU 0: starting rwkv6_qtail_lowrank_all_convshift_multidil_symmetric"
uv run scripts/run_experiment.py \
    --backbone rwkv6_qtail_lowrank_all_convshift_multidil_symmetric \
    --gpu 0 \
    --epochs 30 \
    --seed 42 \
    2>&1 | tee -a "$LOG_DIR/gpu0.log"
echo "[$(date +%H:%M:%S)] GPU 0: finished CB-7"
