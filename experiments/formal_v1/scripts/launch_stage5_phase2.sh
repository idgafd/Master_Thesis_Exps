#!/usr/bin/env bash
# Stage 5 Phase 2 — orthogonal stacking: P²-RSE × Stage-4 budget refinements.
#
# Two parallel 30-epoch runs, one per GPU:
#   GPU 0: rwkv6_p2rse_depth   — P²-RSE + depth-graded budget (vs rse_depth ceiling 0.1207)
#   GPU 1: rwkv6_p2rse_strong  — P²-RSE + uniform large budget (vs rse_strong ceiling 0.1192)
#
# Both use the softmax mixer (Phase-1 winner between linear and softmax).
#
# Usage:
#   bash scripts/launch_stage5_phase2.sh [SEED] [EPOCHS] [GPU_D] [GPU_S]
#     GPU_D = GPU for p2rse_depth  (default 0)
#     GPU_S = GPU for p2rse_strong (default 1)

set -euo pipefail

SEED=${1:-42}
EPOCHS=${2:-30}
GPU_D=${3:-0}
GPU_S=${4:-1}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p outputs/logs

launch_tmux () {
    local session=$1 backbone=$2 gpu=$3
    local outdir="outputs/${session}_${backbone}_seed${SEED}"
    mkdir -p "$outdir"
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "[launch] tmux session '$session' already exists — refusing to overwrite."
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
        sleep 30"
    echo "[launch] $session started: gpu $gpu, backbone $backbone"
}

launch_tmux stage5_03 rwkv6_p2rse_depth  $GPU_D
launch_tmux stage5_04 rwkv6_p2rse_strong $GPU_S

echo ""
echo "[launch] Phase 2 launched. Monitor:"
echo "  tail -f outputs/logs/stage5_03.log  # depth, GPU $GPU_D"
echo "  tail -f outputs/logs/stage5_04.log  # strong, GPU $GPU_S"
