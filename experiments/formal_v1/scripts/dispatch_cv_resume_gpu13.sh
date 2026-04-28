#!/usr/bin/env bash
# Re-launch GPU 1 + GPU 3 sub-dispatchers for the CV full-matrix queue.
# Used after the original GPU 1/3 sub-loops were killed (they were
# sleeping; no children lost) and we need to keep working the queue
# on those GPUs once they free.
set -uo pipefail

cd "$(dirname "$0")/.."

QUEUE_FILE=/tmp/cv_full_matrix_queue.txt
LOCK_FILE=/tmp/cv_full_matrix_queue.lock
DISPATCH_LOG=/tmp/cv_full_matrix_dispatcher.log
FINAL_OUT=../final/outputs

pop_queue() {
  ( flock -x 9
    if [ ! -s "$QUEUE_FILE" ]; then exit 0; fi
    head -1 "$QUEUE_FILE"
    tail -n +2 "$QUEUE_FILE" > "${QUEUE_FILE}.tmp"
    mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"
  ) 9>"$LOCK_FILE"
}

is_gpu_idle() {
  local gpu=$1
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
  [ -n "$used" ] && [ "$used" -lt 1000 ]
}

dispatch_loop() {
  local gpu=$1
  echo "[$(date -u +%FT%TZ)] [GPU $gpu] dispatcher (resume) started" >> "$DISPATCH_LOG"
  while : ; do
    if ! is_gpu_idle "$gpu"; then sleep 60; continue; fi
    local entry; entry=$(pop_queue)
    if [ -z "$entry" ]; then
      echo "[$(date -u +%FT%TZ)] [GPU $gpu] queue empty — exiting" >> "$DISPATCH_LOG"
      return 0
    fi
    local backbone="${entry%%:*}"
    local run_name="${entry##*:}"
    local out="${FINAL_OUT}/${run_name}_seed42"
    if [ -f "$out/results.json" ]; then
      echo "[$(date -u +%FT%TZ)] [GPU $gpu] SKIP $run_name" >> "$DISPATCH_LOG"
      continue
    fi
    mkdir -p "$out"
    echo "[$(date -u +%FT%TZ)] [GPU $gpu] LAUNCH $run_name (backbone=$backbone)" >> "$DISPATCH_LOG"
    uv run python scripts/run_experiment.py \
      --config configs/cv_pilot.yaml \
      --backbone "$backbone" \
      --dataset common_voice_en_100h \
      --output-dir "$out" \
      --gpu "$gpu" \
      --fast >> "$DISPATCH_LOG" 2>&1
    echo "[$(date -u +%FT%TZ)] [GPU $gpu] FINISHED $run_name" >> "$DISPATCH_LOG"
  done
}

dispatch_loop 1 &
dispatch_loop 3 &
wait
echo "[$(date -u +%FT%TZ)] resume dispatcher exited" >> "$DISPATCH_LOG"
