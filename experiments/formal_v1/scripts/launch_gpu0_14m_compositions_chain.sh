#!/usr/bin/env bash
# GPU 0 — 14M causal compositions chain (3 cells, ~10 h total).
# Each cell is the §5 pre-registered composition for its architecture
# at 14M, extending the 7M closed-cell composition results.
#
#   1. LA §5 RSE × multidil_v2  — 7M closed at test 0.0999 (BIG comp gain Δ −0.041 vs multidil-alone)
#   2. Mamba-2 P7 lucid_c × multidil_v2 — 7M closed at test 0.0795 (Δ −0.0030 vs multidil-alone)
#   3. RWKV-6 P7 LUCID × multidil_v2 — 7M closed at test 0.0785 (saturated, tied multidil-alone)

set -uo pipefail
cd "$(dirname "$0")/.."

ROOT="$(realpath ../final/outputs)"
LOG_DIR="${ROOT}/_50ep_14m_causal_launch_logs"
mkdir -p "$LOG_DIR"

GPU=0
EPOCHS=50
SEED=42
CONFIG=configs/14m.yaml

dispatch () {
  local backbone="$1"; local outdir="$2"
  local out_path="${ROOT}/${outdir}"
  local log_path="${LOG_DIR}/${outdir}.log"
  if [[ -f "$out_path/results.json" ]]; then
    echo "[gpu0-14m-comp] SKIP ${outdir} (results.json present)" | tee -a "$log_path"
    return 0
  fi
  echo "[gpu0-14m-comp] LAUNCH ${backbone} → GPU ${GPU} → ${outdir} at $(date -Is)" | tee -a "$log_path"
  uv run scripts/run_experiment.py \
    --config ${CONFIG} \
    --backbone "${backbone}" \
    --epochs ${EPOCHS} \
    --seed ${SEED} \
    --gpu ${GPU} \
    --output-dir "${out_path}" \
    >> "$log_path" 2>&1
  rc=$?
  echo "[gpu0-14m-comp] DONE ${backbone} rc=${rc} at $(date -Is)" | tee -a "$log_path"
  return $rc
}

dispatch "linear_attn_rse_strong_viscosity_convshift_multidil_symmetric_v2" \
         "14m_linear_attn_causal_rse_x_multidil_v2_seed42"

dispatch "mamba2_lucid_c_convshift_multidil_symmetric_v2" \
         "14m_mamba2_causal_lucid_c_x_multidil_v2_seed42"

dispatch "rwkv6_lucid_convshift_multidil_symmetric_v2" \
         "14m_rwkv6_causal_p7_seed42"

echo "[gpu0-14m-comp] chain complete at $(date -Is)"
