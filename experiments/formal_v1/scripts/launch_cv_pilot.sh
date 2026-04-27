#!/usr/bin/env bash
# Launch the 4-run CV EN 100h pilot cohort across 4 GPUs (one run per GPU).
#
# Per task brief §4 + §6 step 5:
#   GPU 0: cv_pilot_rwkv6                            (vanilla baseline)
#   GPU 1: cv_pilot_rwkv6_lucid_multidil_v2          (LibriSpeech P7 composition)
#   GPU 2: cv_pilot_linear_attn                      (vanilla baseline)
#   GPU 3: cv_pilot_linear_attn_rse_strong_viscosity (LA's BREAK mechanism)
#
# Output dirs follow the §13 naming convention as relayed in the task brief
# (`cv_pilot_<backbone>_seed42/`) and live under experiments/final/outputs/.
#
# Hyperparameters come from configs/cv_pilot.yaml (50 ep, AdamW lr=3e-4,
# batch <= 300s, etc. — identical to the LibriSpeech 50-ep schedule).
#
# Logs stream to outputs/_cv_pilot_launch_logs/<run_name>.log; the
# canonical per-run train.log is inside each run dir per the existing
# run_experiment.py convention.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG=configs/cv_pilot.yaml
FINAL_OUT=../final/outputs
LAUNCH_LOG=outputs/_cv_pilot_launch_logs
mkdir -p "$FINAL_OUT" "$LAUNCH_LOG"

launch () {
  local gpu="$1"
  local run_name="$2"
  local backbone="$3"
  local out_dir="${FINAL_OUT}/${run_name}_seed42"
  mkdir -p "$out_dir"
  echo "[$(date -u +%FT%TZ)] launching ${run_name} on GPU ${gpu} (backbone=${backbone})"
  nohup uv run python scripts/run_experiment.py \
        --config "$CONFIG" \
        --backbone "$backbone" \
        --dataset common_voice_en_100h \
        --output-dir "$out_dir" \
        --gpu "$gpu" \
        --fast \
    > "${LAUNCH_LOG}/${run_name}.log" 2>&1 &
  local pid=$!
  echo "  PID=${pid} — log: ${LAUNCH_LOG}/${run_name}.log"
}

launch 0 cv_pilot_rwkv6                            rwkv6
launch 1 cv_pilot_rwkv6_lucid_multidil_v2          rwkv6_lucid_multidil_v2
launch 2 cv_pilot_linear_attn                      linear_attn_causal
launch 3 cv_pilot_linear_attn_rse_strong_viscosity linear_attn_rse_strong_viscosity

wait
echo "[$(date -u +%FT%TZ)] all 4 pilot runs finished."
