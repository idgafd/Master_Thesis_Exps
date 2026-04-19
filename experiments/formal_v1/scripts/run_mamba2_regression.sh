#!/usr/bin/env bash
# 10-epoch regression: mamba2 (causal SSD) vs mamba2_lion (bidir, same params).
# Compares against the mamba baseline in RESULTS.md (best dev CER ≈ 0.189).
set -euo pipefail

cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-42}"
OUT_BASE="outputs/mamba2_regression_ep${EPOCHS}_seed${SEED}"

for backbone in mamba2 mamba2_lion; do
    outdir="${OUT_BASE}/${backbone}"
    echo ">>> training ${backbone} -> ${outdir}"
    .venv/bin/python scripts/run_experiment.py \
        --backbone "${backbone}" \
        --epochs "${EPOCHS}" \
        --seed "${SEED}" \
        --output-dir "${outdir}" \
        2>&1 | tee "${outdir}.log" || { echo "FAIL: ${backbone}"; exit 1; }
done

echo ">>> regression complete. Summary:"
for backbone in mamba2 mamba2_lion; do
    outdir="${OUT_BASE}/${backbone}"
    if [[ -f "${outdir}/results.json" ]]; then
        .venv/bin/python -c "
import json
with open('${outdir}/results.json') as f: r = json.load(f)
print(f'  ${backbone:<14s}  best_dev_cer={r[\"best_dev_cer\"]:.4f}  test_cer={r[\"test\"][\"cer\"]:.4f}  params={r[\"params\"][\"total\"]/1e6:.2f}M')
"
    fi
done
