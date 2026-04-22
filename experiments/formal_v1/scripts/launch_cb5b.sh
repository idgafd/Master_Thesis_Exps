#!/bin/bash
# CB-5b param-matched frontend_v2 variant.
#   Runs on GPU 0 after rwkv6_orthogonal (Stage 10.5) finishes.
#   Launch manually when GPU 0 frees.
#
#   Usage:  bash scripts/launch_cb5b.sh
#
#   Will exit immediately if GPU 0 still has an active training job
#   (checks for large memory usage on GPU 0).

set -u
cd "$(dirname "$0")/.."

LOG_DIR="outputs/_cb15_launch_logs"
mkdir -p "$LOG_DIR"

GPU0_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0 | tr -d ' ')
if [ "$GPU0_MEM" -gt 5000 ]; then
    echo "GPU 0 still has $GPU0_MEM MiB in use — another training run is active."
    echo "Wait for rwkv6_orthogonal to finish before launching CB-5b."
    exit 1
fi

run_one() {
    local gpu="$1"
    local backbone="$2"
    echo "[$(date +%H:%M:%S)] GPU $gpu: starting $backbone"
    uv run scripts/run_experiment.py \
        --backbone "$backbone" \
        --gpu "$gpu" \
        --epochs 30 \
        --seed 42 \
        2>&1 | tee -a "$LOG_DIR/gpu${gpu}.log"
    echo "[$(date +%H:%M:%S)] GPU $gpu: finished $backbone"
}

( run_one 0 rwkv6_frontend_v2_matched ) > "$LOG_DIR/gpu0_cb5b_pipeline.log" 2>&1 &
GPU0_PID=$!
echo "GPU 0 pipeline PID: $GPU0_PID"
echo "$GPU0_PID" > "$LOG_DIR/gpu0_cb5b.pid"
echo "Logs: $LOG_DIR/gpu0_cb5b_pipeline.log"

wait $GPU0_PID
echo "[$(date +%H:%M:%S)] CB-5b done. Exit: $?"
