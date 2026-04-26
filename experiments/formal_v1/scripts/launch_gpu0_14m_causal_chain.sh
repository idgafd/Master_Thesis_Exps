#!/usr/bin/env bash
# GPU 0 — 14M causal chain (sequential, ~2-3 h per cell).
#
# Per Master_Plan §8 14M is mandatory for all 60 base cells.  This chain
# closes the four highest-priority causal 14M cells beyond the two
# RWKV-6 scouts already on disk (vanilla 0.1103, multidil_v2 0.0751):
#   1. 14m_mamba2_causal_vanilla              (mamba2)
#   2. 14m_mamba2_causal_multidil_v2          (mamba2_convshift_multidil_symmetric_v2)
#   3. 14m_linear_attn_causal_vanilla         (linear_attn_causal)
#   4. 14m_linear_attn_causal_multidil_v2     (linear_attn_convshift_multidil_symmetric_v2)
#
# Triggered after the GPU 0 LA LION-S × multidil chain (LION-S × LUCID ×
# multidil_v2) writes results.json.  Uses configs/14m.yaml (n_layers=12,
# d_model/n_heads/head_size unchanged from the 7M default — clean depth-
# only scaling per Master_Plan §8 footnote on shape).

set -uo pipefail
cd "$(dirname "$0")/.."

ROOT="$(realpath ../final/outputs)"
LOG_DIR="${ROOT}/_50ep_14m_causal_launch_logs"
mkdir -p "$LOG_DIR"

GPU=0
EPOCHS=50
SEED=42
CONFIG=configs/14m.yaml

# Wait for the upstream P7-style cell to finish.
WAIT_FOR="${ROOT}/7m_linear_attn_lion_s_lucid_multidil_v2_seed42/results.json"
while [[ ! -f "$WAIT_FOR" ]]; do
  sleep 60
done
echo "[gpu0-14m] upstream done; starting 14M causal chain at $(date -Is)"

dispatch () {
  local backbone="$1"; local outdir="$2"
  local out_path="${ROOT}/${outdir}"
  local log_path="${LOG_DIR}/${outdir}.log"
  if [[ -f "$out_path/results.json" ]]; then
    echo "[gpu0-14m] SKIP ${outdir} (results.json present)" | tee -a "$log_path"
    return 0
  fi
  echo "[gpu0-14m] LAUNCH ${backbone} → GPU ${GPU} → ${outdir} at $(date -Is)" | tee -a "$log_path"
  uv run scripts/run_experiment.py \
    --config ${CONFIG} \
    --backbone "${backbone}" \
    --epochs ${EPOCHS} \
    --seed ${SEED} \
    --gpu ${GPU} \
    --output-dir "${out_path}" \
    >> "$log_path" 2>&1
  rc=$?
  echo "[gpu0-14m] DONE ${backbone} rc=${rc} at $(date -Is)" | tee -a "$log_path"
  return $rc
}

dispatch "mamba2"                                       "14m_mamba2_causal_vanilla_seed42"
dispatch "mamba2_convshift_multidil_symmetric_v2"       "14m_mamba2_causal_multidil_v2_seed42"
dispatch "linear_attn_causal"                           "14m_linear_attn_causal_vanilla_seed42"
dispatch "linear_attn_convshift_multidil_symmetric_v2"  "14m_linear_attn_causal_multidil_v2_seed42"

echo "[gpu0-14m] chain complete at $(date -Is)"
