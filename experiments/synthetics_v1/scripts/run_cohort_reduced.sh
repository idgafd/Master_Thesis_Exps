#!/usr/bin/env bash
# Run the 16-run reduced cohort defined in configs/cohort_reduced.yaml.
#
# Phase 1a: T=64 across all 8 backbones (~30 min)
# Phase 1b: T=256 across all 8 backbones (~1 GPU-hour)
#
# Each run lives at outputs/cohort_reduced/<backbone>_T<seq_len>_seed<N>/.
# After all runs complete, scripts/analyze_cohort.py builds the summary table.
#
# Usage (from experiments/synthetics_v1/):
#     bash scripts/run_cohort_reduced.sh           # all 16 runs
#     bash scripts/run_cohort_reduced.sh --T64     # phase 1a only
#     bash scripts/run_cohort_reduced.sh --T256    # phase 1b only

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

ONLY_T64=0
ONLY_T256=0
case "${1:-}" in
    --T64)  ONLY_T64=1 ;;
    --T256) ONLY_T256=1 ;;
    "")     ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
esac

BACKBONES=(
    transformer_causal
    rwkv6
    rwkv6_lucid
    rwkv6_delta
    rwkv6_lucid_delta
    mamba
    mamba2
    transformer
)

SEED=42
OUT_ROOT="outputs/cohort_reduced"

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

if [[ $ONLY_T256 -eq 0 ]]; then
    echo "── Phase 1a: T=64 ─────────────────────────────────────────────────"
    for bb in "${BACKBONES[@]}"; do
        run_one "$bb" 64 16
    done
fi

if [[ $ONLY_T64 -eq 0 ]]; then
    echo "── Phase 1b: T=256 ────────────────────────────────────────────────"
    for bb in "${BACKBONES[@]}"; do
        run_one "$bb" 256 64
    done
fi

echo "── DONE ────────────────────────────────────────────────────────────"
echo "Run scripts/analyze_cohort.py to build the summary table."
