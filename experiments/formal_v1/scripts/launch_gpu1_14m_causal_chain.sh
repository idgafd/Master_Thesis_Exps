#!/usr/bin/env bash
# GPU 1 — 14M causal chain (sequential, ~3 h per cell, ~14 h total).
# Triggered manually after the Mamba-2 LION × RSE-depth-viscosity cell
# was killed at ep10 (null reproduced — see partial cell in
# 7m_mamba2_lion_rse_depth_viscosity_seed42_10ep_NULL_REPRODUCED/).
#
# Priority order matches the §3 / closed-cell evidence: LA RSE was the
# BREAK at 7M (Δ −0.068), Mamba-2 LUCID-c was the only non-null Mamba-2
# single mechanism (Δ −0.008), then LA LUCID + the RWKV-6 cells fill
# the matrix.

set -uo pipefail
cd "$(dirname "$0")/.."

ROOT="$(realpath ../final/outputs)"
LOG_DIR="${ROOT}/_50ep_14m_causal_launch_logs"
mkdir -p "$LOG_DIR"

GPU=1
EPOCHS=50
SEED=42
CONFIG=configs/14m.yaml

dispatch () {
  local backbone="$1"; local outdir="$2"
  local out_path="${ROOT}/${outdir}"
  local log_path="${LOG_DIR}/${outdir}.log"
  if [[ -f "$out_path/results.json" ]]; then
    echo "[gpu1-14m] SKIP ${outdir} (results.json present)" | tee -a "$log_path"
    return 0
  fi
  echo "[gpu1-14m] LAUNCH ${backbone} → GPU ${GPU} → ${outdir} at $(date -Is)" | tee -a "$log_path"
  uv run scripts/run_experiment.py \
    --config ${CONFIG} \
    --backbone "${backbone}" \
    --epochs ${EPOCHS} \
    --seed ${SEED} \
    --gpu ${GPU} \
    --output-dir "${out_path}" \
    >> "$log_path" 2>&1
  rc=$?
  echo "[gpu1-14m] DONE ${backbone} rc=${rc} at $(date -Is)" | tee -a "$log_path"
  return $rc
}

dispatch "linear_attn_rse_strong_viscosity"            "14m_linear_attn_causal_rse_strong_viscosity_seed42"
dispatch "mamba2_lucid_c"                              "14m_mamba2_causal_lucid_c_seed42"
dispatch "linear_attn_lucid"                           "14m_linear_attn_causal_lucid_seed42"
dispatch "rwkv6_lucid"                                 "14m_rwkv6_causal_lucid_chunked_seed42"
dispatch "rwkv6_rse_strong_viscosity"                  "14m_rwkv6_causal_rse_strong_viscosity_seed42"

echo "[gpu1-14m] chain complete at $(date -Is)"
