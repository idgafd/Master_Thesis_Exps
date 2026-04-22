#!/bin/bash
# CB-3: rwkv6_convshift_multidil_symmetric_gated.
# Content-conditional α_d on multi-dilation ConvShift.
# GPU 1, ~1.8 h dry-run / ~3 h real.

set -u
cd "$(dirname "$0")/.."

LOG_DIR="outputs/_cb3_launch_logs"
mkdir -p "$LOG_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "[$(date +%H:%M:%S)] GPU 1: starting rwkv6_convshift_multidil_symmetric_gated"
uv run scripts/run_experiment.py \
    --backbone rwkv6_convshift_multidil_symmetric_gated \
    --gpu 1 \
    --epochs 30 \
    --seed 42 \
    2>&1 | tee -a "$LOG_DIR/gpu1.log"
echo "[$(date +%H:%M:%S)] GPU 1: finished CB-3"
