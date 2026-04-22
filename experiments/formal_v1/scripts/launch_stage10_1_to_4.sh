#!/bin/bash
# Stage 10.1–10.4 + 10.3-symmetric training launcher across 2 GPUs.
#
# GPU 0 pipeline (~4.2 h total):
#   rwkv6_m2rnn_sparse  → rwkv6_chanmix_bypass
# GPU 1 pipeline (~3.7 h total):
#   rwkv6_loglinear  → rwkv6_convshift_multidil  → rwkv6_convshift_multidil_symmetric
#
# Each run writes to outputs/{backbone}_seed{seed}/ with:
#   history.csv   — per-epoch dev CER / loss / timing
#   train.log     — streaming log
#   best_model.pt, last_model.pt, checkpoint_ep{N}.pt

set -u
cd "$(dirname "$0")/.."

LOG_DIR="outputs/_stage10_launch_logs"
mkdir -p "$LOG_DIR"

# Normalise the environment so the two pipelines don't fight over GPU 0.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Don't cap visibility via CUDA_VISIBLE_DEVICES here — each job sets --gpu N
# itself and run_experiment.py calls torch.cuda.set_device(N).

run_one() {
    # $1 = gpu, $2 = backbone
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

# ── GPU 0 pipeline ────────────────────────────────────────────────
(
    run_one 0 rwkv6_m2rnn_sparse
    run_one 0 rwkv6_chanmix_bypass
) > "$LOG_DIR/gpu0_pipeline.log" 2>&1 &
GPU0_PID=$!

# ── GPU 1 pipeline ────────────────────────────────────────────────
(
    run_one 1 rwkv6_loglinear
    run_one 1 rwkv6_convshift_multidil
    run_one 1 rwkv6_convshift_multidil_symmetric
) > "$LOG_DIR/gpu1_pipeline.log" 2>&1 &
GPU1_PID=$!

echo "GPU 0 pipeline PID: $GPU0_PID"
echo "GPU 1 pipeline PID: $GPU1_PID"
echo "Logs:"
echo "  $LOG_DIR/gpu0_pipeline.log"
echo "  $LOG_DIR/gpu1_pipeline.log"
echo ""
echo "Wait with:   wait $GPU0_PID $GPU1_PID"
echo "Or poll history.csv in outputs/{backbone}_seed42/ for per-epoch progress."

# Persist PIDs so the monitoring script can check them.
echo "$GPU0_PID" > "$LOG_DIR/gpu0.pid"
echo "$GPU1_PID" > "$LOG_DIR/gpu1.pid"

wait $GPU0_PID $GPU1_PID
EXIT0=$?
EXIT1=$?
echo ""
echo "All pipelines finished. GPU 0 exit: $EXIT0, GPU 1 exit: $EXIT1"
