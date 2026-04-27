#!/usr/bin/env bash
# Auto-launches 7M Mamba-2 LION × P7 (LUCID-c × multidil_v2) on GPU 0
# after the in-flight 7M RWKV-6 LION × P7 cell finishes.

set -uo pipefail
cd "$(dirname "$0")/.."

ROOT="$(realpath ../final/outputs)"
LOG_DIR="${ROOT}/_50ep_lion_lucid_launch_logs"
WAIT_FOR="${ROOT}/7m_rwkv6_lion_p7_seed42/results.json"
OUTDIR="7m_mamba2_lion_p7_seed42"
LOG="${LOG_DIR}/${OUTDIR}.log"

while [[ ! -f "$WAIT_FOR" ]]; do
  sleep 60
done

echo "[gpu0-mamba2-lion-p7] upstream done; launching at $(date -Is)" | tee -a "$LOG"
uv run scripts/run_experiment.py \
  --backbone mamba2_lion_lucid_c_convshift_multidil_symmetric_v2 \
  --epochs 50 \
  --seed 42 \
  --gpu 0 \
  --output-dir "${ROOT}/${OUTDIR}" \
  >> "$LOG" 2>&1
echo "[gpu0-mamba2-lion-p7] DONE rc=$? at $(date -Is)" | tee -a "$LOG"
