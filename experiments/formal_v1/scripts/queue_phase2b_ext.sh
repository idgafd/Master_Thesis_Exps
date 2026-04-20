#!/usr/bin/env bash
# Autonomous launcher for Phase 2b-ext (rwkv6_p2rse_indeplam_extkv_strong_viscosity).
#
# Blocks until the stage6_03 (qtail) tmux session ends, then launches
# Phase 2b-ext on GPU 0. Phase 2b continues in parallel on GPU 1.
#
# This script should itself be run inside a detached tmux session so the
# user can disconnect without killing it:
#   bash scripts/queue_phase2b_ext.sh
#   (or via tmux wrapping — see the spawn command at the bottom of the script)
#
# Args:
#   SEED    — default 42
#   EPOCHS  — default 30
#   WAIT_ON — default stage6_03 (tmux session to wait for)

set -euo pipefail

SEED=${1:-42}
EPOCHS=${2:-30}
WAIT_ON=${3:-stage6_03}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BACKBONE=rwkv6_p2rse_indeplam_extkv_strong_viscosity
SESSION=phase2b_ext
OUTDIR="outputs/${SESSION}_${BACKBONE}_seed${SEED}"
LOG="outputs/logs/${SESSION}.log"
WATCHER_LOG="outputs/logs/${SESSION}_watcher.log"

mkdir -p "$OUTDIR" outputs/logs

{
  echo "[watcher] $(date -u +'%Y-%m-%dT%H:%M:%SZ') — start. Waiting on tmux session '$WAIT_ON'..."

  # Poll for the watched tmux session to disappear.
  while tmux has-session -t "$WAIT_ON" 2>/dev/null; do
    sleep 60
  done

  echo "[watcher] $(date -u +'%Y-%m-%dT%H:%M:%SZ') — '$WAIT_ON' ended. Launching $SESSION on GPU 0."

  # Refuse to clobber an existing session of the same name.
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
