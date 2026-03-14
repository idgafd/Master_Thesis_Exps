#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run RWKV-7 fix experiments in parallel on 2 GPUs.
#
# GPU 0: rwkv7_fix_decay  — only decay init fix (isolates hypothesis)
# GPU 1: rwkv7_fix_all    — decay + v_first + k_a fixes (best convergence chance)
#
# Both write to outputs/run-018_rwkv7_fix/ (different backbone names, no clash).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_DIR="/workspace/Master_Thesis_Exps/experiments/asr_backbone_comparison"
RWKV_DIR="$EXP_DIR/RWKV-block"
SRC_DIR="$EXP_DIR/src"
PYTHON="/venv/main/bin/python3"

mkdir -p "$EXP_DIR/outputs"

echo "=== Launching GPU 0: rwkv7_fix_decay ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$RWKV_DIR:$SRC_DIR" \
    $PYTHON "$EXP_DIR/scripts/run_experiment.py" \
    --config "$EXP_DIR/configs/rwkv7_fix_decay.yaml" \
    > "$EXP_DIR/outputs/log_rwkv7_fix_decay.txt" 2>&1 &
PID0=$!
echo "GPU0 PID: $PID0"

echo "=== Launching GPU 1: rwkv7_fix_all ==="
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$RWKV_DIR:$SRC_DIR" \
    $PYTHON "$EXP_DIR/scripts/run_experiment.py" \
    --config "$EXP_DIR/configs/rwkv7_fix_all.yaml" \
    > "$EXP_DIR/outputs/log_rwkv7_fix_all.txt" 2>&1 &
PID1=$!
echo "GPU1 PID: $PID1"

echo ""
echo "=== Both runs launched ==="
echo "Monitor logs:"
echo "  tail -f $EXP_DIR/outputs/log_rwkv7_fix_decay.txt"
echo "  tail -f $EXP_DIR/outputs/log_rwkv7_fix_all.txt"
echo ""
echo "=== Waiting for both runs to complete ==="
wait $PID0 && echo "GPU0 (fix_decay) DONE" || echo "GPU0 (fix_decay) FAILED"
wait $PID1 && echo "GPU1 (fix_all) DONE" || echo "GPU1 (fix_all) FAILED"
