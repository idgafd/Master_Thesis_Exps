#!/usr/bin/env bash
# §3 single-mechanism MQAR cohort.
#
# Causal backbones (3 architectures × {vanilla + 3 single mechanisms} = 12)
# plus transformer_causal as reference = 13 cells per length.
# Compositions (e.g. rwkv6_lucid_multidil_v2) are NOT run here per the
# user's priority directive (compositions are deferred to a follow-up
# cohort once all single mechanisms are characterised across lengths).
#
# Output:  outputs/cohort_reduced/<backbone>_T<seq_len>_seed<N>/
# (Reused across phases — script auto-skips runs whose results.json exists.)
#
# Usage (from experiments/synthetics_v1/):
#     bash scripts/run_cohort_single_mech.sh           # all lengths
#     bash scripts/run_cohort_single_mech.sh --T64     # T=64 only
#     bash scripts/run_cohort_single_mech.sh --T256    # T=256 only
#     bash scripts/run_cohort_single_mech.sh --T1024   # T=1024 only

set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

ONLY_T=""
case "${1:-}" in
    --T64)   ONLY_T=64   ;;
    --T256)  ONLY_T=256  ;;
    --T1024) ONLY_T=1024 ;;
    "")      ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
esac

BACKBONES=(
    # Reference
    transformer_causal
    # RWKV-6 family — vanilla + 2 mechanisms (RSE removed: confirmed axis-1 NULL on MQAR
    #   at T=64 with both rse_strong and rse_depth schedules, per_seq=0.0).
    rwkv6
    rwkv6_convshift_multidil_symmetric_v2
    rwkv6_lucid
    # Mamba-2 family — vanilla + 2 mechanisms (RSE removed: same reason)
    mamba2
    mamba2_convshift_multidil_symmetric_v2
    mamba2_lucid_c
    # Linear Attention family — vanilla + 2 mechanisms (RSE removed: same reason)
    linear_attn_causal
    linear_attn_convshift_multidil_symmetric_v2
    linear_attn_lucid
)

# Length × n_kv_pairs schedule (RWKV-7 ratio: K = T/4)
LENGTHS=(64 256 1024)
KV_PAIRS=(16 64 256)

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

for i in "${!LENGTHS[@]}"; do
    T="${LENGTHS[$i]}"
    K="${KV_PAIRS[$i]}"
    if [[ -n "${ONLY_T}" && "${T}" != "${ONLY_T}" ]]; then
        continue
    fi
    echo "── Phase: T=${T} (K=${K}) ─────────────────────────────────────────"
    for bb in "${BACKBONES[@]}"; do
        run_one "$bb" "$T" "$K"
    done
done

echo "── DONE ────────────────────────────────────────────────────────────"
