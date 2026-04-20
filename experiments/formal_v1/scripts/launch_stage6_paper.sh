#!/usr/bin/env bash
# Stage 6 — EXPRESSIVENESS paper (Mongaras & Larson 2025) adaptations on
# pure causal RWKV-6. Tests the two load-bearing claims of the paper in
# isolation — no RSE, no viscosity, no P²-RSE.
#
# GPU 0: rwkv6_rmsnorm     — GroupNorm → per-head RMSNorm readout
# GPU 1: rwkv6_hadamard_n2 — rmsnorm + diagonal (k⊙k, r⊙r) second-order branch
# Queued: rwkv6_qtail      — rmsnorm + Kronecker (k⊗k, r⊗r) at top 2 layers
#
# Usage:
#   cd experiments/formal_v1
#   bash scripts/launch_stage6_paper.sh [SEED] [EPOCHS]
# Monitor:
#   tmux ls
#   tail -f outputs/logs/stage6_01.log
# Stop:
#   tmux kill-session -t stage6_01

set -euo pipefail

SEED=${1:-42}
EPOCHS=${2:-30}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p outputs/logs

launch_tmux () {
    local session=$1 backbone=$2 gpu=$3
    local outdir="outputs/${session}_${backbone}_seed${SEED}"
    mkdir -p "$outdir"
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "[launch] tmux session '$session' already exists — refusing to overwrite."
        echo "         Inspect with: tmux attach -t $session"
        echo "         Or kill with:  tmux kill-session -t $session"
        return 1
    fi
    tmux new-session -d -s "$session" -c "$REPO_ROOT" "\
        echo '[tmux] session=$session backbone=$backbone gpu=$gpu seed=$SEED epochs=$EPOCHS' && \
        uv run python scripts/run_experiment.py \
            --backbone $backbone \
            --epochs $EPOCHS \
            --seed $SEED \
            --output-dir $outdir \
            --gpu $gpu 2>&1 | tee outputs/logs/${session}.log; \
        echo '[tmux] $session exit code: '\${PIPESTATUS[0]}; \
        sleep 60"
    echo "[launch] $session started: gpu $gpu, backbone $backbone, output $outdir"
}

launch_tmux stage6_01 rwkv6_rmsnorm     0
launch_tmux stage6_02 rwkv6_hadamard_n2 1

echo ""
echo "[launch] Stage 6 launched (rmsnorm on GPU 0, hadamard_n2 on GPU 1)."
echo "[launch] rwkv6_qtail is queued — launch on whichever GPU frees first."
echo ""
echo "[monitor]"
echo "  tmux ls"
echo "  tail -f outputs/logs/stage6_01.log  # rmsnorm"
echo "  tail -f outputs/logs/stage6_02.log  # hadamard_n2"
