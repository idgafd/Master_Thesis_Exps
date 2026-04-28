#!/usr/bin/env bash
# Parallel-friendly T=1024 cohort runner — runs an arbitrary subset of the
# §3 single-mech 13-backbone cohort on a single GPU. Use CUDA_VISIBLE_DEVICES
# to pin to a GPU. Auto-skips runs whose results.json already exists.
#
# Usage (from experiments/synthetics_v1/):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_T1024_subset.sh transformer_causal rwkv6 rwkv6_lucid …
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

T=1024
K=256
SEED=42
OUT_ROOT="outputs/cohort_reduced"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <backbone> [<backbone> ...]" >&2
    exit 2
fi

for backbone in "$@"; do
    out="${OUT_ROOT}/${backbone}_T${T}_seed${SEED}"
    if [[ -f "${out}/results.json" ]]; then
        echo "SKIP (already done): ${out}"
        continue
    fi
    echo "═══════════════════════════════════════════════════════════════════"
    echo "RUN  backbone=${backbone}  T=${T}  K=${K}  seed=${SEED}  CUDA=${CUDA_VISIBLE_DEVICES:-(default)}"
    echo "out=${out}"
    echo "═══════════════════════════════════════════════════════════════════"
    uv run python scripts/run_experiment.py \
        --config configs/default.yaml \
        --backbone "${backbone}" \
        --seq-len "${T}" \
        --n-kv-pairs "${K}" \
        --seed "${SEED}" \
        --output-dir "${out}"
done

echo "── DONE (subset) ──"
