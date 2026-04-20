#!/usr/bin/env bash
# Autonomous launcher for qtail-γ refinement (rwkv6_qtail_gamma).
#
# Waits for the stage6_03 (qtail baseline) tmux session to end, then
# launches rwkv6_qtail_gamma on GPU 0 at matched seed. This is the
# Stage-2-style refinement iteration: single mechanism refinement
# (learnable per-head γ decay coupling) with zero-regression-at-init
# back to vanilla qtail.  NOT a composition — it's a principled math
# refinement INSIDE the Kronecker mechanism (see TODO_FUTURE_IDEAS).
#
# Replaces the earlier phase2b_ext_watcher (that one stacked extkv on
# top of indep-λ, which the user correctly identified as the wrong
# direction before we know whether the single changes work).

set -euo pipefail

SEED=${1:-42}
EPOCHS=${2:-30}
WAIT_ON=${3:-stage6_03}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BACKBONE=rwkv6_qtail_gamma
SESSION=qtail_gamma
OUTDIR="outputs/${SESSION}_${BACKBONE}_seed${SEED}"
LOG="outputs/logs/${SESSION}.log"
WATCHER_LOG="outputs/logs/${SESSION}_watcher.log"

mkdir -p "$OUTDIR" outputs/logs

{
  echo "[watcher] $(date -u +'%Y-%m-%dT%H:%M:%SZ') — start. Waiting on tmux session '$WAIT_ON'..."

  while tmux has-session -t "$WAIT_ON" 2>/dev/null; do
    sleep 60
  done

  echo "[watcher] $(date -u +'%Y-%m-%dT%H:%M:%SZ') — '$WAIT_ON' ended. Launching $SESSION on GPU 0."

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[watcher] ERROR — tmux session '$SESSION' already exists. Aborting."
    exit 1
  fi

  tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "\
    echo '[tmux] session=$SESSION backbone=$BACKBONE gpu=0 seed=$SEED epochs=$EPOCHS' && \
    uv run python scripts/run_experiment.py \
      --backbone $BACKBONE \
      --epochs $EPOCHS \
      --seed $SEED \
      --output-dir $OUTDIR \
      --gpu 0 2>&1 | tee $LOG; \
    echo '[tmux] $SESSION exit: '\${PIPESTATUS[0]}; \
    sleep 60"

  echo "[watcher] $(date -u +'%Y-%m-%dT%H:%M:%SZ') — $SESSION started on GPU 0 → $OUTDIR"
} 2>&1 | tee -a "$WATCHER_LOG"
