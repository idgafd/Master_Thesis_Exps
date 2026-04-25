#!/usr/bin/env bash
# launch_7m_50ep_singles.sh
#
# First batch of the 50-ep rerun cycle.
# 12 LibriSpeech ASR runs, 7M, causal, 50 ep, single seed (42).
# 3 architectures (RWKV-6, Mamba-2, Linear Attention) × {vanilla + 3 single mechanisms}.
# Compositions (cell 5 per Master_Plan §4) and LION variants land in subsequent batches.
#
# Distribution across 4 GPUs, 3 sequential jobs per GPU:
#   GPU 0: RWKV-6 column     (vanilla → lucid_chunked → rse_strong_viscosity)
#   GPU 1: Mamba-2 column    (vanilla → lucid_c       → rse_strong_viscosity)
#   GPU 2: LA column         (vanilla → lucid         → rse_strong_viscosity)
#   GPU 3: multidil_v2 row   (rwkv6 → mamba2 → linear_attn)
#
# Each run writes to outputs/7m_<arch>_causal_<cellname>_seed42/ per Master_Plan §13.

set -euo pipefail
cd "$(dirname "$0")/.."   # → experiments/formal_v1

EPOCHS=50
SEED=42
# Final-stage output root per Master_Plan §13 / §18 entry 14.
# Must live under experiments/final/ so best_model.pt is committable
# (formal_v1/.gitignore excludes *.pt; final/.gitignore permits them).
FINAL_OUT_ROOT="$(realpath ../final/outputs)"
LOG_DIR="${FINAL_OUT_ROOT}/_50ep_singles_launch_logs"
mkdir -p "$LOG_DIR"

# (gpu | codebase_backbone | output_dirname)
JOBS=(
  "0|rwkv6|7m_rwkv6_causal_vanilla_seed42"
  "0|rwkv6_lucid|7m_rwkv6_causal_lucid_chunked_seed42"
  "0|rwkv6_rse_strong_viscosity|7m_rwkv6_causal_rse_strong_viscosity_seed42"

  "1|mamba2|7m_mamba2_causal_vanilla_seed42"
  "1|mamba2_lucid_c|7m_mamba2_causal_lucid_c_seed42"
  "1|mamba2_rse_strong_viscosity|7m_mamba2_causal_rse_strong_viscosity_seed42"

  "2|linear_attn_causal|7m_linear_attn_causal_vanilla_seed42"
  "2|linear_attn_lucid|7m_linear_attn_causal_lucid_seed42"
  "2|linear_attn_rse_strong_viscosity|7m_linear_attn_causal_rse_strong_viscosity_seed42"

  "3|rwkv6_convshift_multidil_symmetric_v2|7m_rwkv6_causal_multidil_v2_seed42"
  "3|mamba2_convshift_multidil_symmetric_v2|7m_mamba2_causal_multidil_v2_seed42"
  "3|linear_attn_convshift_multidil_symmetric_v2|7m_linear_attn_causal_multidil_v2_seed42"
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

GPU0=(); GPU1=(); GPU2=(); GPU3=()
for spec in "${JOBS[@]}"; do
  case "$spec" in
    0\|*) GPU0+=("$spec") ;;
    1\|*) GPU1+=("$spec") ;;
    2\|*) GPU2+=("$spec") ;;
    3\|*) GPU3+=("$spec") ;;
  esac
done

run_queue 0 "${GPU0[@]}" &  P0=$!
run_queue 1 "${GPU1[@]}" &  P1=$!
run_queue 2 "${GPU2[@]}" &  P2=$!
run_queue 3 "${GPU3[@]}" &  P3=$!

echo "Launched 4 queues at $(date -Is). PIDs: $P0 $P1 $P2 $P3"
wait $P0 $P1 $P2 $P3
echo "All 12 runs complete at $(date -Is)"
