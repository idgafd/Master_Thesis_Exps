#!/usr/bin/env bash
# Dispatch the missing 11 CV-EN-100h 7M-causal cells across all 4 GPUs.
# Each per-GPU dispatcher loop:
#   - Waits until its GPU is idle (no large allocation by another run).
#   - Pops the next (backbone, run_name) tuple from the shared queue file.
#     `flock` makes the pop atomic across the 4 dispatchers.
#   - Skips a cell whose run dir already has a results.json.
#   - Launches the run via run_experiment.py and waits for it to finish.
# Loops until the queue is empty. The queue order is intentional:
#   - 8 fast cells first (vanilla / multidil / lucid / P7 — all ~80 min)
#   - 3 slow cells last (RSE-based — ~3-5 h each on this codebase)
#
# This honours the master prompt's GPU-isolation rule: it never preempts a
# run that didn't come from this dispatcher (only launches when memory is
# below 1 GB on a GPU). 14M Mamba-2 P7 on GPU 1 and the LA-RSE pilot on
# GPU 3 are left alone; their slots are picked up only after they free.
set -uo pipefail

cd "$(dirname "$0")/.."

QUEUE_FILE=/tmp/cv_full_matrix_queue.txt
LOCK_FILE=/tmp/cv_full_matrix_queue.lock
DISPATCH_LOG=/tmp/cv_full_matrix_dispatcher.log
FINAL_OUT=../final/outputs

cat > "$QUEUE_FILE" <<'EOF'
mamba2:cv_pilot_mamba2
mamba2_convshift_multidil_symmetric_v2:cv_pilot_mamba2_multidil_v2
mamba2_lucid_c:cv_pilot_mamba2_lucid_c
mamba2_lucid_c_convshift_multidil_symmetric_v2:cv_pilot_mamba2_lucid_c_multidil_v2
rwkv6_convshift_multidil_symmetric_v2:cv_pilot_rwkv6_multidil_v2
rwkv6_lucid:cv_pilot_rwkv6_lucid_chunked
linear_attn_convshift_multidil_symmetric_v2:cv_pilot_linear_attn_multidil_v2
linear_attn_lucid:cv_pilot_linear_attn_lucid
mamba2_rse_strong_viscosity:cv_pilot_mamba2_rse_strong_viscosity
rwkv6_rse_strong_viscosity:cv_pilot_rwkv6_rse_strong_viscosity
linear_attn_rse_strong_viscosity_convshift_multidil_symmetric_v2:cv_pilot_linear_attn_rse_x_multidil_v2
EOF
touch "$LOCK_FILE"

pop_queue() {
  # Atomic line pop. Echoes the popped line (or empty when queue empty).
  (
    flock -x 9
    if [ ! -s "$QUEUE_FILE" ]; then
      exit 0
    fi
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
  echo "[$(date -u +%FT%TZ)] [GPU $gpu] dispatcher started" >> "$DISPATCH_LOG"
  while : ; do
    if ! is_gpu_idle "$gpu"; then
      sleep 60
      continue
    fi
    local entry
    entry=$(pop_queue)
    if [ -z "$entry" ]; then
      echo "[$(date -u +%FT%TZ)] [GPU $gpu] queue empty — exiting" >> "$DISPATCH_LOG"
      return 0
    fi
    local backbone="${entry%%:*}"
    local run_name="${entry##*:}"
    local out="${FINAL_OUT}/${run_name}_seed42"
    if [ -f "$out/results.json" ]; then
      echo "[$(date -u +%FT%TZ)] [GPU $gpu] SKIP $run_name (results.json exists)" >> "$DISPATCH_LOG"
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
    local rc=$?
    if [ "$rc" = 0 ]; then
      echo "[$(date -u +%FT%TZ)] [GPU $gpu] FINISHED $run_name" >> "$DISPATCH_LOG"
    else
      echo "[$(date -u +%FT%TZ)] [GPU $gpu] FAILED $run_name (exit=$rc)" >> "$DISPATCH_LOG"
    fi
  done
}

for gpu in 0 1 2 3; do
  dispatch_loop "$gpu" &
done
wait
echo "[$(date -u +%FT%TZ)] all dispatchers exited" >> "$DISPATCH_LOG"
