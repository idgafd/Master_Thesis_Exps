#!/usr/bin/env bash
# launch_7m_50ep_lion_lucid.sh
#
# LUCID × LION batch — 3 archs at 7M, 50 ep, seed 42.
# Backbones (per encoder.py registration):
#   - lion_lucid_chunked          — RWKV-6 LION + LUCID (chunk_size=64)
#   - mamba2_lion_lucid_c         — Mamba-2 LION + LUCID-c (C-correlation)
#   - linear_attn_lion_lucid      — LA LION-LIT + LUCID (parallel form)
#
# Distribution: 3 cells across GPUs 0/1/2 (GPU 3 reserved for RSE-LION
# engineering work in parallel).

set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=50
SEED=42
FINAL_OUT_ROOT="$(realpath ../final/outputs)"
LOG_DIR="${FINAL_OUT_ROOT}/_50ep_lion_lucid_launch_logs"
mkdir -p "$LOG_DIR"

JOBS=(
  "0|lion_lucid_chunked|7m_rwkv6_lion_lucid_chunked_seed42"
  "1|mamba2_lion_lucid_c|7m_mamba2_lion_lucid_c_seed42"
  "2|linear_attn_lion_lucid|7m_linear_attn_lion_lucid_seed42"
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

GPU0=(); GPU1=(); GPU2=()
for spec in "${JOBS[@]}"; do
  case "$spec" in
    0\|*) GPU0+=("$spec") ;;
    1\|*) GPU1+=("$spec") ;;
    2\|*) GPU2+=("$spec") ;;
  esac
done

run_queue 0 "${GPU0[@]}" &  P0=$!
run_queue 1 "${GPU1[@]}" &  P1=$!
run_queue 2 "${GPU2[@]}" &  P2=$!

echo "Launched 3 queues at $(date -Is). PIDs: $P0 $P1 $P2"
wait $P0 $P1 $P2
echo "All LUCID-LION runs complete at $(date -Is)"
