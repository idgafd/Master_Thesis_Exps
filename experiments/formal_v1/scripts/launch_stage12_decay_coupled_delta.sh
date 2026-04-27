#!/usr/bin/env bash
# launch_stage12_decay_coupled_delta.sh
#
# Stage 12 — Decay-Coupled Delta (RWKV-6) ASR run.
#   Pre-registered in STAGE12_DECAY_COUPLED_DELTA.md.
#   7M, causal RWKV-6, 50 ep, single seed (42), LibriSpeech clean-100.
#
# Output directory follows §3.3 / §7 of the spec and matches the existing
# 50-ep convention (lives under experiments/final/outputs/ so checkpoints
# are committable; formal_v1/.gitignore excludes *.pt).
#
# Halt criterion (§4.1): if at ep 15 dev CER ≥ vanilla rwkv6 ep-15 + 0.006,
# halt and write final_status: early_stopped.  This is enforced by the
# matched-epoch tracking in src/training/train.py — no separate watchdog.
#
# Usage (from experiments/formal_v1/):
#     bash scripts/launch_stage12_decay_coupled_delta.sh           # foreground
#     bash scripts/launch_stage12_decay_coupled_delta.sh --bg      # background

set -euo pipefail
cd "$(dirname "$0")/.."   # → experiments/formal_v1

EPOCHS=50
SEED=42
GPU=0
BACKBONE="rwkv6_decay_coupled_delta"

FINAL_OUT_ROOT="$(realpath ../final/outputs)"
OUT_DIR="${FINAL_OUT_ROOT}/rwkv6_decay_coupled_delta_seed42"
LOG_DIR="${FINAL_OUT_ROOT}/_stage12_launch_logs"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/rwkv6_decay_coupled_delta_seed42.log"

if [[ -f "${OUT_DIR}/results.json" ]]; then
  echo "SKIP — ${OUT_DIR} already has results.json"
  exit 0
fi

echo "[stage12] START backbone=${BACKBONE} epochs=${EPOCHS} seed=${SEED} gpu=${GPU} at $(date -Is)"
echo "[stage12] out=${OUT_DIR}"
echo "[stage12] log=${LOG_PATH}"

CMD=(
  uv run scripts/run_experiment.py
  --backbone "${BACKBONE}"
  --epochs ${EPOCHS}
  --seed ${SEED}
  --gpu ${GPU}
  --output-dir "${OUT_DIR}"
  --fast
)
# --fast: TF32 matmul + cudnn.benchmark (auto-tunes per shape).  Standard
# production-training optimisations the codebase had off by default; should
# give 2-3x step-rate improvement on Blackwell tensor cores at the cost of
# losing bit-exact cudnn-kernel reproducibility (loss curves still match
# seeded reference within fp32 noise).
#
# torch.compile NOT used here: the K×K delta scan + many dispatch branches
# (RSE / lucid / multidil / delta) hang Inductor's tracer for 10+ min at
# 0% GPU.  See scripts/launch_*.sh for context.

if [[ "${1:-}" == "--bg" ]]; then
  nohup "${CMD[@]}" > "${LOG_PATH}" 2>&1 &
  echo "[stage12] background PID $!"
else
  "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
fi
