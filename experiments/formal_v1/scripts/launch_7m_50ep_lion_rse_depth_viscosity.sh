#!/usr/bin/env bash
# launch_7m_50ep_lion_rse_depth_viscosity.sh
#
# RSE-depth-viscosity × LION batch — 3 archs at 7M, 50 ep, seed 42.
# Backbones (per encoder.py registration):
#   - rwkv6_lion_rse_depth_viscosity        — RWKV-6 LION + RSE-depth-viscosity
#   - mamba2_lion_rse_depth_viscosity       — Mamba-2 LION + RSE-depth-viscosity
#   - linear_attn_lion_rse_depth_viscosity  — LA LION + RSE-depth-viscosity
#
# Mechanism: bidirectional T×T complex attention with the Hermitian-
# symmetric LION pattern over a complex log-decay
# (log_z_t = -lambda_eff_t + i · theta_t).  Depth-graded θ clip:
# π/8 (L0–L1) → π/4 (L2–L3) → π/2 (L4–L5).  Viscosity coupling
# (η · θ²) added to the magnitude part of log_z.  Per-layer θ_init
# scaled accordingly.
#
# Usage: pass GPU index(es) on the command line.  If unspecified the
# script schedules one cell per GPU 0/1/2/3 (skips already-done cells).
#
# Example:
#   bash launch_7m_50ep_lion_rse_depth_viscosity.sh           # all 3 across GPUs 0/1/2
#   bash launch_7m_50ep_lion_rse_depth_viscosity.sh 3         # all 3 sequentially on GPU 3

set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=50
SEED=42
FINAL_OUT_ROOT="$(realpath ../final/outputs)"
LOG_DIR="${FINAL_OUT_ROOT}/_50ep_lion_rse_depth_viscosity_launch_logs"
mkdir -p "$LOG_DIR"

JOBS=(
  "rwkv6_lion_rse_depth_viscosity|7m_rwkv6_lion_rse_depth_viscosity_seed42"
  "mamba2_lion_rse_depth_viscosity|7m_mamba2_lion_rse_depth_viscosity_seed42"
  "linear_attn_lion_rse_depth_viscosity|7m_linear_attn_lion_rse_depth_viscosity_seed42"
)

run_one () {
  local gpu="$1"; local backbone="$2"; local outdir="$3"
  local out_path="${FINAL_OUT_ROOT}/${outdir}"
  local log_path="${LOG_DIR}/${outdir}.log"

  if [[ -f "$out_path/results.json" ]]; then
    echo "[GPU ${gpu}] SKIP ${outdir} (results.json present)" | tee -a "$log_path"
    return 0
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
  return $rc
}

if [[ $# -gt 0 ]]; then
  GPU="$1"
  for spec in "${JOBS[@]}"; do
    IFS='|' read -r backbone outdir <<< "$spec"
    run_one "$GPU" "$backbone" "$outdir"
  done
else
  # one per GPU 0..2 in parallel
  PIDS=()
  i=0
  for spec in "${JOBS[@]}"; do
    IFS='|' read -r backbone outdir <<< "$spec"
    run_one "$i" "$backbone" "$outdir" &
    PIDS+=($!)
    i=$((i+1))
  done
  wait "${PIDS[@]}"
fi
echo "All RSE-LION runs complete at $(date -Is)"
