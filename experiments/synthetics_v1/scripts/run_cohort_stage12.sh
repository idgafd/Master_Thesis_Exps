#!/usr/bin/env bash
# Run the Stage 12 — Decay-Coupled Delta MQAR cohort defined in
# configs/cohort_stage12.yaml.
#
# Pre-registered in formal_v1/STAGE12_DECAY_COUPLED_DELTA.md §4.2 / §5.2:
#   T ∈ {64, 256, 1024}
#   backbones: rwkv6, rwkv6_lucid, rwkv6_delta (NEG-CTRL), rwkv6_decay_coupled_delta
#
# Each run lives at outputs/cohort_stage12/<backbone>_T<seq_len>_seed42/.
# After all runs complete, scripts/analyze_cohort.py builds the summary table.
#
# Usage (from experiments/synthetics_v1/):
#     bash scripts/run_cohort_stage12.sh             # all 12 runs
#     bash scripts/run_cohort_stage12.sh --T64       # phase 1 only
#     bash scripts/run_cohort_stage12.sh --T256      # phase 2 only
#     bash scripts/run_cohort_stage12.sh --T1024     # phase 3 only
#
# Per spec §4.2: "Adding only the coupled variant without the negative
# control is not sufficient evidence." rwkv6_delta is mandatory.

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

PHASE_T64=1
PHASE_T256=1
PHASE_T1024=1
case "${1:-}" in
    --T64)   PHASE_T256=0; PHASE_T1024=0 ;;
    --T256)  PHASE_T64=0;  PHASE_T1024=0 ;;
    --T1024) PHASE_T64=0;  PHASE_T256=0 ;;
    "")      ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
esac

BACKBONES=(
    rwkv6
    rwkv6_lucid
    rwkv6_delta                # negative control — mandatory per spec §4.2
    rwkv6_decay_coupled_delta  # Stage 12 probe
)

SEED=42
OUT_ROOT="outputs/cohort_stage12"

run_one() {
    local backbone="$1"
    local T="$2"
    local K="$3"
    local out="${OUT_ROOT}/${backbone}_T${T}_seed${SEED}"

    if [[ -f "${out}/results.json" ]]; then
        echo "SKIP (already done): ${out}"
        return
    fi

    echo "═══════════════════════════════════════════════════════════════════"
    echo "RUN  backbone=${backbone}  T=${T}  K=${K}  seed=${SEED}"
    echo "out=${out}"
    echo "═══════════════════════════════════════════════════════════════════"

    uv run python scripts/run_experiment.py \
        --config configs/default.yaml \
        --backbone "${backbone}" \
        --seq-len "${T}" \
        --n-kv-pairs "${K}" \
        --seed "${SEED}" \
        --output-dir "${out}"
}

if [[ $PHASE_T64 -eq 1 ]]; then
    echo "── Stage 12 Phase A: T=64 ─────────────────────────────────────────"
    for bb in "${BACKBONES[@]}"; do
        run_one "$bb" 64 16
    done
fi

if [[ $PHASE_T256 -eq 1 ]]; then
    echo "── Stage 12 Phase B: T=256 ────────────────────────────────────────"
    for bb in "${BACKBONES[@]}"; do
        run_one "$bb" 256 64
    done
fi

if [[ $PHASE_T1024 -eq 1 ]]; then
    echo "── Stage 12 Phase C: T=1024 ───────────────────────────────────────"
    for bb in "${BACKBONES[@]}"; do
        run_one "$bb" 1024 256
    done
fi

echo "── DONE ────────────────────────────────────────────────────────────"
echo "Run scripts/analyze_cohort.py with --output-root ${OUT_ROOT} to summarise."
