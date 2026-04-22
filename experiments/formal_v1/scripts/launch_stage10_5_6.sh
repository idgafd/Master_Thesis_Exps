#!/bin/bash
# Stage 10.5 + 10.6 training launcher across 2 GPUs (Phase II of STAGE10_PLAN §5).
#
# GPU 0: rwkv6_orthogonal (rank-1 chunked Cayley)   — ~3.4 h projected
# GPU 1: rwkv6_pom_vlift (thin PoM v-lift)          — ~0.9 h projected
#
# Each run writes to outputs/{backbone}_seed{seed}/ with:
#   history.csv   — per-epoch dev CER / loss / timing
#   train.log     — streaming log
#   best_model.pt, last_model.pt, checkpoint_ep{N}.pt

set -u
cd "$(dirname "$0")/.."

LOG_DIR="outputs/_stage10_phase2_launch_logs"
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

# GPU 0 — longest run solo
( run_one 0 rwkv6_orthogonal ) > "$LOG_DIR/gpu0_pipeline.log" 2>&1 &
GPU0_PID=$!

# GPU 1 — fast run, then idle (can be extended later)
( run_one 1 rwkv6_pom_vlift ) > "$LOG_DIR/gpu1_pipeline.log" 2>&1 &
GPU1_PID=$!

echo "GPU 0 pipeline PID: $GPU0_PID"
echo "GPU 1 pipeline PID: $GPU1_PID"
echo "Logs:"
echo "  $LOG_DIR/gpu0_pipeline.log"
echo "  $LOG_DIR/gpu1_pipeline.log"
echo ""
echo "$GPU0_PID" > "$LOG_DIR/gpu0.pid"
echo "$GPU1_PID" > "$LOG_DIR/gpu1.pid"

wait $GPU0_PID $GPU1_PID
EXIT0=$?
EXIT1=$?
echo ""
echo "All pipelines finished. GPU 0 exit: $EXIT0, GPU 1 exit: $EXIT1"
