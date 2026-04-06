#!/bin/bash
# Compare Mamba implementations: our parallel-scan PyTorch vs official CUDA mamba-ssm
# GPU 0: Our implementation (backbone=mamba)
# GPU 1: Official CUDA (backbone=mamba_cuda)
#
# Both run 10 epochs on LibriSpeech train-clean-100.

set -e
cd "$(dirname "$0")/.."

EPOCHS=${1:-10}
SEED=${2:-42}
LOGDIR="./outputs/logs"
mkdir -p "$LOGDIR"

echo "=========================================="
echo " Mamba comparison: PyTorch vs CUDA"
echo " Epochs: $EPOCHS, Seed: $SEED"
echo "=========================================="
echo ""

# GPU 0: Our parallel-scan PyTorch Mamba
(
    echo "[GPU 0] Starting Mamba (parallel-scan PyTorch)..."
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_experiment.py \
        --config configs/default.yaml \
        --backbone mamba \
        --epochs "$EPOCHS" \
        --output-dir "./outputs/mamba_pytorch_ep${EPOCHS}_seed${SEED}" \
        --seed "$SEED" \
        2>&1 | tee "$LOGDIR/mamba_pytorch.log"
    echo "[GPU 0] Done."
) &
PID_GPU0=$!

# GPU 1: Official CUDA mamba-ssm
(
    echo "[GPU 1] Starting Mamba (official CUDA mamba-ssm)..."
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_experiment.py \
        --config configs/default.yaml \
        --backbone mamba_cuda \
        --epochs "$EPOCHS" \
        --output-dir "./outputs/mamba_cuda_ep${EPOCHS}_seed${SEED}" \
        --seed "$SEED" \
        2>&1 | tee "$LOGDIR/mamba_cuda.log"
    echo "[GPU 1] Done."
) &
PID_GPU1=$!

echo "Launched GPU 0 (PID $PID_GPU0): Mamba parallel-scan PyTorch"
echo "Launched GPU 1 (PID $PID_GPU1): Mamba official CUDA"
echo ""
echo "Logs: $LOGDIR/{mamba_pytorch,mamba_cuda}.log"
echo "Waiting for all jobs to finish..."

wait $PID_GPU0
STATUS0=$?
wait $PID_GPU1
STATUS1=$?

echo ""
echo "=========================================="
echo " Comparison complete!"
echo "=========================================="

# Print summary
echo ""
uv run python -c "
import json, os

variants = [
    ('Mamba (PyTorch parallel)', f'./outputs/mamba_pytorch_ep${EPOCHS}_seed${SEED}/results.json'),
    ('Mamba (CUDA mamba-ssm)', f'./outputs/mamba_cuda_ep${EPOCHS}_seed${SEED}/results.json'),
]

print(f'{\"Variant\":35s} | {\"Params\":>8s} | {\"Best CER\":>9s} | {\"Avg Epoch\":>10s}')
print('-' * 75)

for name, path in variants:
    if os.path.exists(path):
        r = json.load(open(path))
        params = f'{r[\"params\"][\"total\"]/1e6:.2f}M'
        cer = f'{r[\"best_dev_cer\"]:.4f}'
        if r.get('history'):
            avg_time = sum(e.get('epoch_time_sec', 0) for e in r['history'] if 'epoch_time_sec' in e)
            n = sum(1 for e in r['history'] if 'epoch_time_sec' in e)
            if n > 0:
                avg_time = f'{avg_time/n:.0f}s'
            else:
                avg_time = 'N/A'
        else:
            avg_time = 'N/A'
        print(f'{name:35s} | {params:>8s} | {cer:>9s} | {avg_time:>10s}')
    else:
        print(f'{name:35s} | FAILED (no results file)')
"
