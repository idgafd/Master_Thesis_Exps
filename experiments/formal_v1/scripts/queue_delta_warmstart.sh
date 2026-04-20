#!/usr/bin/env bash
# Autonomous launcher for rwkv6_delta_warmstart.
# Polls for lowrank tmux session to end, then launches delta_warmstart on GPU 1.
# Uses EXACT match (tmux list-sessions | grep -qx) to avoid the prefix-match
# bug that caused qtail_gamma_watcher to detect itself.

set -euo pipefail

SEED=${1:-42}
EPOCHS=${2:-30}
WAIT_ON=${3:-lowrank}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BACKBONE=rwkv6_delta_warmstart
SESSION=dwarm
OUTDIR="outputs/${SESSION}_${BACKBONE}_seed${SEED}"
LOG="outputs/logs/${SESSION}.log"
WATCHER_LOG="outputs/logs/${SESSION}_watcher.log"

mkdir -p "$OUTDIR" outputs/logs

{
  echo "[watcher] $(date -u +'%Y-%m-%dT%H:%M:%SZ') — start. Waiting on tmux session '$WAIT_ON'..."

  # EXACT match to avoid prefix collision with watcher's own session name.
  while tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -qx "$WAIT_ON"; do
    sleep 60
  done

  echo "[watcher] $(date -u +'%Y-%m-%dT%H:%M:%SZ') — '$WAIT_ON' ended. Launching $SESSION on GPU 1."

  if tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -qx "$SESSION"; then
    echo "[watcher] ERROR — tmux session '$SESSION' already exists. Aborting."
    exit 1
  fi

  tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "\
    echo '[tmux] session=$SESSION backbone=$BACKBONE gpu=1 seed=$SEED epochs=$EPOCHS' && \
    uv run python scripts/run_experiment.py \
      --backbone $BACKBONE \
      --epochs $EPOCHS \
      --seed $SEED \
      --output-dir $OUTDIR \
      --gpu 1 2>&1 | tee $LOG; \
    echo '[tmux] $SESSION exit: '\${PIPESTATUS[0]}; \
    sleep 60"

  echo "[watcher] $(date -u +'%Y-%m-%dT%H:%M:%SZ') — $SESSION started on GPU 1 → $OUTDIR"
} 2>&1 | tee -a "$WATCHER_LOG"
