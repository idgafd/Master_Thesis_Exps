#!/bin/bash
# Launch run-014 (LayerConv) on GPU 0 and run-015 (Temperature) on GPU 1
# Both run in background; logs go to outputs/ directory

set -e
cd "$(dirname "$0")/.."

echo "Starting run-014 (LayerConv) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup uv run scripts/run_experiment.py \
    --config configs/run014_layerconv.yaml \
    > outputs/log_layerconv.txt 2>&1 &
PID1=$!
echo "  PID: $PID1 → outputs/log_layerconv.txt"

echo "Starting run-015 (Temperature) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup uv run scripts/run_experiment.py \
    --config configs/run015_temperature.yaml \
    > outputs/log_temperature.txt 2>&1 &
PID2=$!
echo "  PID: $PID2 → outputs/log_temperature.txt"

echo ""
echo "Both experiments launched. Monitor with:"
echo "  tail -f outputs/log_layerconv.txt"
echo "  tail -f outputs/log_temperature.txt"
echo ""
echo "PIDs: $PID1 (GPU 0, LayerConv), $PID2 (GPU 1, Temperature)"
