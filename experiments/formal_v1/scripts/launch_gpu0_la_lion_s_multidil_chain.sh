#!/usr/bin/env bash
# GPU 0 — sequential dispatch for the LA LION-S × multidil follow-up cells.
#   1. linear_attn_lion_s_convshift_multidil_symmetric_v2
#      (LA LION-S + multidil_v2 — fills the missing LION-S × multidil cell.
#       Compare against LION-LIT × multidil_v2 = 0.1404 test.)
#   2. linear_attn_lion_s_lucid_convshift_multidil_symmetric_v2
#      (LA LION-S × LUCID × multidil_v2 — P7-style composition on LA with the
#       decay-bounded LION-S backbone.  Tests if multidil_v2 stacks on top of
#       the encouraging LION-S × LUCID descent.)

set -uo pipefail
cd "$(dirname "$0")/.."

ROOT="$(realpath ../final/outputs)"
LOG_DIR="${ROOT}/_50ep_lion_lucid_launch_logs"
mkdir -p "$LOG_DIR"

GPU=0
EPOCHS=50
SEED=42

dispatch () {
  local backbone="$1"; local outdir="$2"
  local out_path="${ROOT}/${outdir}"
  local log_path="${LOG_DIR}/${outdir}.log"
  if [[ -f "$out_path/results.json" ]]; then
    echo "[gpu0-chain] SKIP ${outdir} (results.json present)" | tee -a "$log_path"
    return 0
  fi
  echo "[gpu0-chain] LAUNCH ${backbone} → GPU ${GPU} → ${outdir} at $(date -Is)" | tee -a "$log_path"
  uv run scripts/run_experiment.py \
    --backbone "${backbone}" \
    --epochs ${EPOCHS} \
    --seed ${SEED} \
    --gpu ${GPU} \
    --output-dir "${out_path}" \
    >> "$log_path" 2>&1
  rc=$?
  echo "[gpu0-chain] DONE ${backbone} rc=${rc} at $(date -Is)" | tee -a "$log_path"
  return $rc
}

dispatch "linear_attn_lion_s_convshift_multidil_symmetric_v2" \
         "7m_linear_attn_lion_s_multidil_v2_seed42"

dispatch "linear_attn_lion_s_lucid_convshift_multidil_symmetric_v2" \
         "7m_linear_attn_lion_s_lucid_multidil_v2_seed42"

echo "[gpu0-chain] complete at $(date -Is)"
