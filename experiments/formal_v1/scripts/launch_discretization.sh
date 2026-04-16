#!/bin/bash
# Stage 2 — Higher-order discretization study, 8 runs across 2 GPUs.
#
# Layout (each GPU runs 4 jobs sequentially, ~75 min/run, ~5 h per GPU):
#   GPU 0:  disc01 (rwkv6 baseline) → disc02 (trap) → disc03 (trap_var) → disc04 (gen2)
#   GPU 1:  disc05 (ab3) → disc06 (convshift_trap) → disc07 (lion_trap) → disc08 (lion_convshift_trap)
#
# AB3 is on GPU 1 first so that if it explodes (see PLAN.md §4.1 dangers)
# the GPU 1 queue still proceeds with the remaining jobs.
#
# Usage:
#   bash scripts/launch_discretization.sh
# Then monitor:
#   tail -f outputs/logs/disc_gpu0.log
#   tail -f outputs/logs/disc_gpu1.log

set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

EPOCHS=30

run_one() {
    local backbone="$1"
    local id="$2"
    local gpu="$3"
    echo "[GPU${gpu}] $(date +%H:%M:%S) starting ${id} (${backbone})..."
    uv run python scripts/run_experiment.py \
        --backbone "${backbone}" \
        --epochs ${EPOCHS} \
        --output-dir "outputs/${id}_seed42" \
        --gpu "${gpu}"
    echo "[GPU${gpu}] $(date +%H:%M:%S) finished ${id}"
}

echo "=== Stage 2 — Discretization study launched at $(date) ==="
echo "8 runs, 30 epochs each, 2 GPUs in parallel."
echo ""

# GPU 0 — causal core (baseline + 3 closely-related trap/gen2 variants)
(
    run_one rwkv6                disc01_rwkv6_baseline       0
    run_one rwkv6_trap           disc02_rwkv6_trap           0
    run_one rwkv6_trap_var       disc03_rwkv6_trap_var       0
    run_one rwkv6_gen2           disc04_rwkv6_gen2           0
) > "${LOG_DIR}/disc_gpu0.log" 2>&1 &
PID_GPU0=$!
echo "GPU 0 launched (PID=${PID_GPU0}) → ${LOG_DIR}/disc_gpu0.log"

# GPU 1 — AB3 (risky) first, then stacking and LION transfer
(
    run_one rwkv6_ab3            disc05_rwkv6_ab3            1
    run_one rwkv6_convshift_trap disc06_rwkv6_convshift_trap 1
    run_one lion_trap            disc07_lion_trap            1
    run_one lion_convshift_trap  disc08_lion_convshift_trap  1
) > "${LOG_DIR}/disc_gpu1.log" 2>&1 &
PID_GPU1=$!
echo "GPU 1 launched (PID=${PID_GPU1}) → ${LOG_DIR}/disc_gpu1.log"

echo ""
echo "Monitor with:"
echo "  tail -f ${LOG_DIR}/disc_gpu0.log"
echo "  tail -f ${LOG_DIR}/disc_gpu1.log"
echo ""
echo "Waiting for both GPUs to finish..."

wait ${PID_GPU0}
RC0=$?
echo "[GPU0] Finished (exit=${RC0}) at $(date)"

wait ${PID_GPU1}
RC1=$?
echo "[GPU1] Finished (exit=${RC1}) at $(date)"

echo ""
echo "=== Discretization study complete at $(date) ==="
echo "  GPU 0 (baseline+trap+gen2): exit=${RC0}"
echo "  GPU 1 (ab3+stacking+lion):  exit=${RC1}"
echo ""
echo "Refresh the index and the RESULTS.md tables:"
echo "  uv run python -m src.reporting.collect"
echo "  uv run python -m src.reporting.tables"
