#!/usr/bin/env bash
# launch_7m_50ep_lion_vanilla.sh
#
# Vanilla LION batch — Master_Plan §2 modes 2/4/6 at 7M, 50 ep, seed 42.
# 3 architectures × 1 cell (vanilla) = 3 runs.
#
# Backbone strings:
#   - lion             — RWKV-6 LION (existing, mode="lion")
#   - mamba2_lion      — Mamba-2 LION full bidir (existing, ssd_scan_lion)
#   - linear_attn_lion — LION-LIT for LA (new, see linear_attn_lion.py)
#
# Distribution across 2 GPUs (this machine):
#   GPU 0: lion → linear_attn_lion         (~1h + ~1h)
#   GPU 1: mamba2_lion                     (~1.4h)
#
# Each run writes to outputs/7m_<arch>_lion_vanilla_seed42/ under
# experiments/final/ per Master_Plan §13 / §18 entry 14.

set -euo pipefail
cd "$(dirname "$0")/.."   # → experiments/formal_v1

EPOCHS=50
SEED=42
FINAL_OUT_ROOT="$(realpath ../final/outputs)"
LOG_DIR="${FINAL_OUT_ROOT}/_50ep_lion_vanilla_launch_logs"
mkdir -p "$LOG_DIR"

# (gpu | codebase_backbone | output_dirname)
JOBS=(
  "0|lion|7m_rwkv6_lion_vanilla_seed42"
  "0|linear_attn_lion|7m_linear_attn_lion_vanilla_seed42"

  "1|mamba2_lion|7m_mamba2_lion_vanilla_seed42"
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
echo "All vanilla LION runs complete at $(date -Is)"
