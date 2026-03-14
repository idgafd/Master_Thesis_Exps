#!/bin/bash
# Launch run-016 (bidir_rwkv6 vs temperature) on GPU 0
# and    run-017 (conv_nogate vs layerconv)    on GPU 1
# Each run trains 2 backbones sequentially with stronger regularization.
# Separate output_dirs → no results.json conflicts.

set -e
cd "$(dirname "$0")/.."

echo "Starting run-016 (bidir_rwkv6 vs temperature, stronger reg) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup uv run scripts/run_experiment.py \
    --config configs/run016_temperature_reg.yaml \
    > outputs/log_run016_temperature_reg.txt 2>&1 &
PID1=$!
echo "  PID: $PID1 → outputs/log_run016_temperature_reg.txt"

echo "Starting run-017 (conv_nogate vs layerconv, stronger reg) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup uv run scripts/run_experiment.py \
    --config configs/run017_layerconv_reg.yaml \
    > outputs/log_run017_layerconv_reg.txt 2>&1 &
PID2=$!
echo "  PID: $PID2 → outputs/log_run017_layerconv_reg.txt"

echo ""
echo "Both experiments launched. Monitor with:"
echo "  tail -f outputs/log_run016_temperature_reg.txt"
echo "  tail -f outputs/log_run017_layerconv_reg.txt"
echo ""
echo "PIDs: $PID1 (GPU 0, run-016), $PID2 (GPU 1, run-017)"
