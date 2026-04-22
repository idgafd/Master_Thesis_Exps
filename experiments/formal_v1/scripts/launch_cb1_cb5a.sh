#!/bin/bash
# GPU 1 pipeline: CB-1 then CB-5a (lean frontend_v2).
#   CB-1  rwkv6_rse_convshift_multidil_symmetric  ~1.7 h
#   CB-5a rwkv6_frontend_v2                        ~1.9 h
# Total ~3.6 h wallclock on GPU 1.
#
# CB-5b (param-matched frontend_v2_matched) runs on GPU 0 after
# rwkv6_orthogonal finishes; launched separately.

set -u
cd "$(dirname "$0")/.."

LOG_DIR="outputs/_cb15_launch_logs"
mkdir -p "$LOG_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID

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

(
    run_one 1 rwkv6_rse_convshift_multidil_symmetric
    run_one 1 rwkv6_frontend_v2
) > "$LOG_DIR/gpu1_pipeline.log" 2>&1 &
GPU1_PID=$!

echo "GPU 1 pipeline PID: $GPU1_PID"
echo "$GPU1_PID" > "$LOG_DIR/gpu1.pid"
echo "Logs: $LOG_DIR/gpu1_pipeline.log"

wait $GPU1_PID
echo "[$(date +%H:%M:%S)] GPU 1 pipeline done. Exit: $?"
