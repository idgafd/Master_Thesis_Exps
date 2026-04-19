#!/usr/bin/env bash
# Stage 5 Phase 1 — parallel launch (2 GPUs) of the decisive P²-RSE pair.
#
# Experiment A (GPU 0): rwkv6_p2rse             — unconstrained linear β mixer
# Experiment B (GPU 1): rwkv6_p2rse_softmax     — convex softmax mixer (control)
#
# Each experiment runs in its own detached tmux session so it survives
# disconnection of the parent shell.  Output is tee'd to a log file that
# the launcher prints at the end.
#
# Usage:
#   cd experiments/formal_v1
#   bash scripts/launch_stage5_phase1.sh [SEED] [EPOCHS]
#
# Monitor:
#   tmux ls
#   tail -f outputs/stage5_01_rwkv6_p2rse_seed42/train.log
#   tail -f outputs/stage5_02_rwkv6_p2rse_softmax_seed42/train.log
#
# Stop:
#   tmux kill-session -t stage5_01
#   tmux kill-session -t stage5_02

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
    # If a session with this name already exists, refuse rather than clobber.
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
        sleep 30"
    echo "[launch] $session started: gpu $gpu, backbone $backbone, output $outdir"
}

launch_tmux stage5_01 rwkv6_p2rse         0
launch_tmux stage5_02 rwkv6_p2rse_softmax 1

echo ""
echo "[launch] Phase 1 launched. Monitor with:"
echo "  tmux ls"
echo "  tail -f outputs/logs/stage5_01.log"
echo "  tail -f outputs/logs/stage5_02.log"
echo ""
echo "[launch] Attach to see live: tmux attach -t stage5_01"
