#!/usr/bin/env bash
# Stage 12 — focused MQAR cohort: rwkv6_delta vs rwkv6_decay_coupled_delta
# at T ∈ {64, 256, 1024}.  6 runs total.
#
# Trimmed from run_cohort_stage12.sh at the user's request — drops the
# vanilla and lucid baselines.  See cohort_stage12_focused.yaml for the
# rationale and the deviation note.
#
# Output:  outputs/cohort_stage12_focused/<backbone>_T<seq_len>_seed42/
#
# Usage (from experiments/synthetics_v1/):
#     bash scripts/run_cohort_stage12_focused.sh             # all 6 runs
#     bash scripts/run_cohort_stage12_focused.sh --T64       # phase 1 only
#     bash scripts/run_cohort_stage12_focused.sh --T256      # phase 2 only
#     bash scripts/run_cohort_stage12_focused.sh --T1024     # phase 3 only

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
    rwkv6_delta                # T1 reference / negative control
    rwkv6_decay_coupled_delta  # Stage 12 probe
)

SEED=42
OUT_ROOT="outputs/cohort_stage12_focused"

# Kernel pick — set per-run inside run_one() based on T (see below):
#   T=64,256 → kxk (fits, fastest forward)
#   T=1024   → ckpt (kxk OOMs at any sane batch size)
#
# Bootstrap delta to a *useful* β at init.  Two env-var overrides combined:
#   gate (multiplicative, learnable):     0.1
#   a0  (LoRA bias, sigmoid sets iclr):   0.0  ⇒ iclr ≈ sigmoid(0)*2 = 1.0
# Therefore β_eff at init = gate × iclr = 0.1 × 1.0 = 0.1.
#
# Why both knobs:
#   - gate=0.1 alone (with default warmstart a0=-5 ⇒ iclr=0.013) gives
#     β_eff = 0.0013 — three orders of magnitude too small.  Empirically
#     validated: 2700 steps of T=256 with gate=0 → 0% per-query accuracy;
#     1000 steps with gate=0.1 + warmstart a0=-5 → still 0%.
#   - a0=0 alone (β=1.0 at init) without gate damping is in the
#     destructive regime per delta_rule.py docstring (wipes wkv state
#     before associations form).
#   - gate=0.1 × a0=0 → β=0.1 is the sweet spot: meaningful erase,
#     enough state retained for SGD to learn associations.
#
# Trade-off: at-init output no longer matches vanilla RWKV-6 (β=0.1 is
# a real perturbation, ≈100× the §2.4 noise floor).  Documented as a
# load-bearing methodological deviation specific to MQAR — without it,
# rwkv6_delta cannot learn the task on this codebase, full stop.
export RWKV6_DELTA_GATE_INIT=0.1
export RWKV6_DELTA_A0_INIT=0.0

run_one() {
    local backbone="$1"
    local T="$2"
    local K="$3"
    local out="${OUT_ROOT}/${backbone}_T${T}_seed${SEED}"

    # The K×K delta scan materialises a (B, H, T, K, K) tensor up front;
    # at default B=64 / T=1024 that's 4.3 GB before chunking and OOMs the
    # 97 GB GPU even with ckpt's recompute path.  Drop B with T to keep
    # peak memory tractable.  K=64 head dim means convergence is mainly
    # gradient-noise-controlled, not batch-size-controlled.
    local BS=64
    local KERNEL=kxk
    if [[ "$T" -ge 1024 ]]; then
        BS=16
        KERNEL=ckpt   # kxk OOMs at this length even with B=16
    elif [[ "$T" -ge 256 ]]; then
        BS=32
        # kxk fits at (B=32, T=256, 6L) — ~5-10 GB peak — and is ~22%
        # faster than ckpt because no forward recompute on backward.
    fi
    export RWKV6_DELTA_SCAN_KERNEL="${KERNEL}"

    if [[ -f "${out}/results.json" ]]; then
        echo "SKIP (already done): ${out}"
        return
    fi

    echo "═══════════════════════════════════════════════════════════════════"
    echo "RUN  backbone=${backbone}  T=${T}  K=${K}  seed=${SEED}  batch=${BS}"
    echo "out=${out}"
    echo "═══════════════════════════════════════════════════════════════════"

    uv run python scripts/run_experiment.py \
        --config configs/default.yaml \
        --backbone "${backbone}" \
        --seq-len "${T}" \
        --n-kv-pairs "${K}" \
        --seed "${SEED}" \
        --batch-size "${BS}" \
        --output-dir "${out}"
}

if [[ $PHASE_T64 -eq 1 ]]; then
    echo "── Stage 12 (focused) Phase A: T=64 ──────────────────────────────"
    for bb in "${BACKBONES[@]}"; do
        run_one "$bb" 64 16
    done
fi

if [[ $PHASE_T256 -eq 1 ]]; then
    echo "── Stage 12 (focused) Phase B: T=256 ─────────────────────────────"
    for bb in "${BACKBONES[@]}"; do
        run_one "$bb" 256 64
    done
fi

if [[ $PHASE_T1024 -eq 1 ]]; then
    echo "── Stage 12 (focused) Phase C: T=1024 ────────────────────────────"
    for bb in "${BACKBONES[@]}"; do
        run_one "$bb" 1024 256
    done
fi

echo "── DONE ────────────────────────────────────────────────────────────"
