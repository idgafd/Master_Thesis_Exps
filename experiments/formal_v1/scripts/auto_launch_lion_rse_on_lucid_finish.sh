#!/usr/bin/env bash
# Watch the 3 LUCID-LION runs; when each finishes (results.json appears),
# launch the corresponding queued RSE-LION cell on its freed GPU.
# RWKV-6 RSE-LION is already running on GPU 3 — this script handles
# Mamba-2 (will go on GPU 1) and LA (will go on GPU 2) only.
#
# Mapping (GPU index → which LUCID-LION cell is on it → which RSE-LION cell to dispatch on it):
#   GPU 0 (RWKV-6 LION × LUCID) → already-running RWKV-6 RSE-LION is on GPU 3, so
#                                   GPU 0 just becomes idle (we leave it idle for now).
#   GPU 1 (Mamba-2 LION × LUCID) → after finish, dispatch Mamba-2 RSE-LION on GPU 1.
#   GPU 2 (LA LION × LUCID)      → after finish, dispatch LA RSE-LION on GPU 2.

set -uo pipefail
cd "$(dirname "$0")/.."

ROOT="$(realpath ../final/outputs)"
LOG_DIR="${ROOT}/_50ep_lion_rse_depth_viscosity_launch_logs"
mkdir -p "$LOG_DIR"

dispatch () {
  local gpu="$1"; local backbone="$2"; local outdir="$3"
  local out_path="${ROOT}/${outdir}"
  local log_path="${LOG_DIR}/${outdir}.log"

  if [[ -f "$out_path/results.json" ]]; then
    echo "[auto-dispatch] SKIP ${outdir} (results.json present)"
    return 0
  fi

  echo "[auto-dispatch] LAUNCH ${backbone} → GPU ${gpu} → ${outdir} at $(date -Is)"
  echo "[auto-dispatch] LAUNCH ${backbone} on GPU ${gpu} at $(date -Is)" >> "$log_path"
  uv run scripts/run_experiment.py \
    --backbone "${backbone}" \
    --epochs 50 \
    --seed 42 \
    --gpu "${gpu}" \
    --output-dir "${out_path}" \
    >> "$log_path" 2>&1 &
  disown
}

# Wait for the LUCID cell on a given GPU to finish (results.json appears).
# Returns when seen.  If results.json already exists, returns immediately.
wait_for_results () {
  local watch_path="$1"
  while [[ ! -f "$watch_path/results.json" ]]; do
    sleep 60
  done
}

# GPU 1 — Mamba-2 LION × LUCID-c → Mamba-2 LION × RSE-depth-viscosity
( wait_for_results "${ROOT}/7m_mamba2_lion_lucid_c_seed42"
  dispatch 1 "mamba2_lion_rse_depth_viscosity" "7m_mamba2_lion_rse_depth_viscosity_seed42"
) &
P1=$!

# GPU 2 — LA LION × LUCID → LA LION × RSE-depth-viscosity
( wait_for_results "${ROOT}/7m_linear_attn_lion_lucid_seed42"
  dispatch 2 "linear_attn_lion_rse_depth_viscosity" "7m_linear_attn_lion_rse_depth_viscosity_seed42"
) &
P2=$!

echo "auto-dispatch watchers running PIDs: $P1 $P2 at $(date -Is)"
wait $P1 $P2
echo "all auto-dispatches done at $(date -Is)"
