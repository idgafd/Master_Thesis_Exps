#!/bin/bash
# Full D3+D4 campaign launcher
# Waits for current validation runs (D0/D1) to finish, then launches
# Group A on GPU 0 and Group B on GPU 1 in parallel.

set -euo pipefail
cd /workspace/Master_Thesis_Exps/experiments/formal_v1

LOG="outputs/campaign.log"
mkdir -p outputs
exec > >(tee -a "$LOG") 2>&1

echo "=== Campaign launcher started at $(date) ==="

# ── Wait for any running training processes to finish ──────────────
echo "[campaign] Waiting for existing training processes to finish..."
while pgrep -f "run_experiment.py" > /dev/null 2>&1; do
    echo "[campaign] $(date +%H:%M:%S) — training processes still running, waiting 30s..."
    sleep 30
done
echo "[campaign] All validation runs finished at $(date)"

# ── Brief pause to let GPU memory clear ────────────────────────────
sleep 5

# ── Validate D0/D1 results ────────────────────────────────────────
echo ""
echo "=== D0/D1 Validation ==="
if [ -f "outputs/d0_rwkv6_ep10_seed42/results.json" ]; then
    echo "[D0] RWKV-6 10ep results.json found — SUCCESS"
    python3 -c "
import json
with open('outputs/d0_rwkv6_ep10_seed42/results.json') as f:
    r = json.load(f)
print(f'  Dev CER: {r[\"best_dev_cer\"]:.4f}')
print(f'  Test CER: {r[\"test\"][\"cer\"]:.4f}')
print(f'  Test WER: {r[\"test\"][\"wer\"]:.4f}')
"
else
    echo "[D0] WARNING: No results.json for RWKV-6 10ep"
fi

if [ -f "outputs/d1_mamba_compiled_ep10_seed42/results.json" ]; then
    echo "[D1] Mamba compiled 10ep results.json found — SUCCESS"
    python3 -c "
import json
with open('outputs/d1_mamba_compiled_ep10_seed42/results.json') as f:
    r = json.load(f)
print(f'  Dev CER: {r[\"best_dev_cer\"]:.4f}')
print(f'  Test CER: {r[\"test\"][\"cer\"]:.4f}')
print(f'  Test WER: {r[\"test\"][\"wer\"]:.4f}')
# Check epoch timing from history
h = r.get('history', [])
if h:
    times = [e['epoch_time_sec'] for e in h]
    print(f'  Avg epoch time: {sum(times)/len(times):.0f}s')
    print(f'  First epoch time: {times[0]:.0f}s (includes compile)')
    print(f'  Steady-state epoch time: {sum(times[1:])/len(times[1:]):.0f}s')
"
else
    echo "[D1] WARNING: No results.json for Mamba compiled 10ep"
fi

echo ""
echo "=== Launching D3 (Group A) + D4 (Group B) at $(date) ==="
echo "[campaign] Group A: 7 causal runs x 80 epochs on GPU 0"
echo "[campaign] Group B: 9 bidirectional runs x 80 epochs on GPU 1"

# ── Launch Group A on GPU 0 ───────────────────────────────────────
uv run scripts/run_registry.py \
    --tag groupA \
    --gpu 0 \
    --resume \
    > outputs/logs/groupA_campaign.log 2>&1 &
PID_A=$!
echo "[campaign] Group A launched (PID=$PID_A) → outputs/logs/groupA_campaign.log"

# ── Launch Group B on GPU 1 ───────────────────────────────────────
uv run scripts/run_registry.py \
    --tag groupB \
    --gpu 1 \
    --resume \
    > outputs/logs/groupB_campaign.log 2>&1 &
PID_B=$!
echo "[campaign] Group B launched (PID=$PID_B) → outputs/logs/groupB_campaign.log"

# ── Wait for both to finish ───────────────────────────────────────
echo ""
echo "[campaign] Waiting for both groups to finish..."
echo "[campaign] Monitor with: tail -f outputs/logs/group{A,B}_campaign.log"

wait $PID_A
RC_A=$?
echo "[campaign] Group A finished (exit=$RC_A) at $(date)"

wait $PID_B
RC_B=$?
echo "[campaign] Group B finished (exit=$RC_B) at $(date)"

echo ""
echo "=== Campaign complete at $(date) ==="
echo "  Group A exit: $RC_A"
echo "  Group B exit: $RC_B"
echo ""
echo "Next steps:"
echo "  uv run python -m src.reporting.collect"
echo "  uv run python -m src.reporting.tables"
echo "  uv run python -m src.reporting.plots"
