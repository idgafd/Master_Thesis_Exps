#!/bin/bash
# Launch all 4 baseline experiments in parallel on 2 GPUs.
# GPU 0: Transformer + RWKV-6 (sequential — Transformer is fast, RWKV-6 is slower)
# GPU 1: Linear Attention + Mamba (sequential — similar reasoning)
#
# Each GPU runs 2 models sequentially, but both GPUs run in parallel.
# This maximizes GPU utilization without OOM.

set -e
cd "$(dirname "$0")/.."

EPOCHS=${1:-10}
SEED=${2:-42}
LOGDIR="./outputs/logs"
mkdir -p "$LOGDIR"

echo "=========================================="
echo " Baseline experiments: 4 models, 2 GPUs"
echo " Epochs: $EPOCHS, Seed: $SEED"
echo "=========================================="
echo ""

# GPU 0: Transformer then RWKV-6
(
    echo "[GPU 0] Starting Transformer..."
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_baseline.py \
        --backbone transformer --epochs "$EPOCHS" --seed "$SEED" \
        2>&1 | tee "$LOGDIR/transformer.log"

    echo "[GPU 0] Starting RWKV-6..."
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_baseline.py \
        --backbone rwkv6 --epochs "$EPOCHS" --seed "$SEED" \
        2>&1 | tee "$LOGDIR/rwkv6.log"

    echo "[GPU 0] Done."
) &
PID_GPU0=$!

# GPU 1: Linear Attention then Mamba
(
    echo "[GPU 1] Starting Linear Attention..."
    CUDA_VISIBLE_DEVICES=1 python3 scripts/run_baseline.py \
        --backbone linear_attention --epochs "$EPOCHS" --seed "$SEED" \
        2>&1 | tee "$LOGDIR/linear_attention.log"

    echo "[GPU 1] Starting Mamba..."
    CUDA_VISIBLE_DEVICES=1 python3 scripts/run_baseline.py \
        --backbone mamba --epochs "$EPOCHS" --seed "$SEED" \
        2>&1 | tee "$LOGDIR/mamba.log"

    echo "[GPU 1] Done."
) &
PID_GPU1=$!

echo "Launched GPU 0 (PID $PID_GPU0): Transformer + RWKV-6"
echo "Launched GPU 1 (PID $PID_GPU1): Linear Attention + Mamba"
echo ""
echo "Logs: $LOGDIR/{transformer,rwkv6,linear_attention,mamba}.log"
echo "Waiting for all jobs to finish..."

wait $PID_GPU0
wait $PID_GPU1

echo ""
echo "=========================================="
echo " All 4 baselines complete!"
echo "=========================================="

# Print summary
echo ""
for bb in transformer linear_attention rwkv6 mamba; do
    result_file="./outputs/baseline_${bb}_ep${EPOCHS}_seed${SEED}/results.json"
    if [ -f "$result_file" ]; then
        cer=$(python3 -c "import json; r=json.load(open('$result_file')); print(f'{r[\"best_dev_cer\"]:.4f}')")
        params=$(python3 -c "import json; r=json.load(open('$result_file')); print(f'{r[\"params\"][\"total\"]/1e6:.2f}M')")
        echo "  $bb: CER=$cer, Params=$params"
    else
        echo "  $bb: NO RESULTS (check log)"
    fi
done
