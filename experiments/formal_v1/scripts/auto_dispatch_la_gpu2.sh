#!/usr/bin/env bash
# GPU 2 — sequential LA dispatch:
#   1. Wait for `7m_linear_attn_lion_lucid_seed42` (LION-LIT × LUCID, currently
#      in flight on GPU 2) to write results.json.
#   2. Launch `linear_attn_lion_s_lucid` (LION-S × LUCID, the cleaner LUCID
#      composition test that gives the preconditioner a decay-bounded value
#      distribution).
#   3. Wait for that to finish.
#   4. Launch `linear_attn_lion_rse_depth_viscosity` (LA LION × RSE).

set -uo pipefail
cd "$(dirname "$0")/.."

ROOT="$(realpath ../final/outputs)"
LUCID_LIT="${ROOT}/7m_linear_attn_lion_lucid_seed42"
LUCID_S="${ROOT}/7m_linear_attn_lion_s_lucid_seed42"
RSE_LA="${ROOT}/7m_linear_attn_lion_rse_depth_viscosity_seed42"
LUCID_S_LOG_DIR="${ROOT}/_50ep_lion_lucid_launch_logs"
RSE_LOG_DIR="${ROOT}/_50ep_lion_rse_depth_viscosity_launch_logs"
mkdir -p "$LUCID_S_LOG_DIR" "$RSE_LOG_DIR"

GPU=2

wait_for_results () {
  local watch_path="$1"
  while [[ ! -f "$watch_path/results.json" ]]; do
    sleep 60
  done
}

dispatch () {
  local backbone="$1"; local outdir="$2"; local log_path="$3"
  if [[ -f "${ROOT}/${outdir}/results.json" ]]; then
    echo "[gpu2-chain] SKIP ${outdir} (results.json present)" | tee -a "$log_path"
    return 0
  fi
  echo "[gpu2-chain] LAUNCH ${backbone} → GPU ${GPU} → ${outdir} at $(date -Is)" | tee -a "$log_path"
  uv run scripts/run_experiment.py \
    --backbone "${backbone}" \
    --epochs 50 \
    --seed 42 \
    --gpu "${GPU}" \
    --output-dir "${ROOT}/${outdir}" \
    >> "$log_path" 2>&1
  rc=$?
  echo "[gpu2-chain] DONE ${backbone} rc=${rc} at $(date -Is)" | tee -a "$log_path"
  return $rc
}

# Step 1 — wait for LA LION-LIT × LUCID to finish.
wait_for_results "$LUCID_LIT"

# Step 2 — launch LA LION-S × LUCID.
dispatch "linear_attn_lion_s_lucid" "7m_linear_attn_lion_s_lucid_seed42" \
         "${LUCID_S_LOG_DIR}/7m_linear_attn_lion_s_lucid_seed42.log"

# Step 3 — wait for results (the dispatch above is blocking, so this is just
# a safety check).
wait_for_results "$LUCID_S"

# Step 4 — launch LA LION × RSE-depth-viscosity.
dispatch "linear_attn_lion_rse_depth_viscosity" "7m_linear_attn_lion_rse_depth_viscosity_seed42" \
         "${RSE_LOG_DIR}/7m_linear_attn_lion_rse_depth_viscosity_seed42.log"

echo "[gpu2-chain] complete at $(date -Is)"
