#!/usr/bin/env bash
# launch_7m_50ep_lion_multidil_v2.sh
#
# Multidil_v2 LION batch — 3 archs × {LION + multidil_v2} at 7M, 50 ep, seed 42.
# Run after the vanilla LION batch lands.
#
# Backbone strings:
#   - lion_convshift_multidil_symmetric_v2              — RWKV-6 LION + multidil_v2
#   - mamba2_lion_convshift_multidil_symmetric_v2       — Mamba-2 LION + multidil_v2
#   - linear_attn_lion_convshift_multidil_symmetric_v2  — LA LION + multidil_v2
#
# Distribution across 2 GPUs:
#   GPU 0: lion_convshift → linear_attn_lion_convshift
#   GPU 1: mamba2_lion_convshift

set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=50
SEED=42
FINAL_OUT_ROOT="$(realpath ../final/outputs)"
LOG_DIR="${FINAL_OUT_ROOT}/_50ep_lion_multidil_v2_launch_logs"
mkdir -p "$LOG_DIR"

JOBS=(
  "0|lion_convshift_multidil_symmetric_v2|7m_rwkv6_lion_multidil_v2_seed42"
  "0|linear_attn_lion_convshift_multidil_symmetric_v2|7m_linear_attn_lion_multidil_v2_seed42"

  "1|mamba2_lion_convshift_multidil_symmetric_v2|7m_mamba2_lion_multidil_v2_seed42"
)

run_queue () {
  local gpu="$1"; shift
  for spec in "$@"; do
    IFS='|' read -r _ backbone outdir <<< "$spec"
    out_path="${FINAL_OUT_ROOT}/${outdir}"
    log_path="${LOG_DIR}/${outdir}.log"

    if [[ -f "$out_path/results.json" ]]; then
      echo "[GPU ${gpu}] SKIP ${outdir} (results.json present)" | tee -a "$log_path"
      continue
    fi

    echo "[GPU ${gpu}] START ${backbone} → ${outdir} at $(date -Is)" | tee -a "$log_path"
    uv run scripts/run_experiment.py \
      --backbone "${backbone}" \
      --epochs ${EPOCHS} \
      --seed ${SEED} \
      --gpu ${gpu} \
      --output-dir "${out_path}" \
      >> "$log_path" 2>&1
    rc=$?
    echo "[GPU ${gpu}] DONE  ${backbone} rc=${rc} at $(date -Is)" | tee -a "$log_path"
  done
}

GPU0=(); GPU1=()
for spec in "${JOBS[@]}"; do
  case "$spec" in
    0\|*) GPU0+=("$spec") ;;
    1\|*) GPU1+=("$spec") ;;
  esac
done

run_queue 0 "${GPU0[@]}" &  P0=$!
run_queue 1 "${GPU1[@]}" &  P1=$!

echo "Launched 2 queues at $(date -Is). PIDs: $P0 $P1"
wait $P0 $P1
echo "All multidil_v2 LION runs complete at $(date -Is)"
