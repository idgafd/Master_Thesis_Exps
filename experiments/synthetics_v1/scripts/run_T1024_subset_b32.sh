#!/usr/bin/env bash
# Same as run_T1024_subset.sh but with batch_size=32 for memory-tight cells.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
T=1024
K=256
SEED=42
B=32
OUT_ROOT="outputs/cohort_reduced"
for backbone in "$@"; do
    out="${OUT_ROOT}/${backbone}_T${T}_seed${SEED}"
    if [[ -f "${out}/results.json" ]]; then
        echo "SKIP (already done): ${out}"
        continue
    fi
    echo "═══════════════════════════════════════════════════════════════════"
    echo "RUN  backbone=${backbone}  T=${T}  K=${K}  B=${B}  seed=${SEED}  CUDA=${CUDA_VISIBLE_DEVICES:-?}"
    echo "═══════════════════════════════════════════════════════════════════"
    uv run python scripts/run_experiment.py \
        --config configs/default.yaml \
        --backbone "${backbone}" \
        --seq-len "${T}" \
        --n-kv-pairs "${K}" \
        --batch-size "${B}" \
        --seed "${SEED}" \
        --output-dir "${out}"
done
echo "── DONE (subset b=32) ──"
