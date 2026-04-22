#!/usr/bin/env bash
# Mirror reusable LEAF backbone files from formal_v1 into synthetics_v1
# via symlinks, so backbone code stays a single source of truth.
#
# We deliberately DO NOT symlink:
#   - formal_v1/src/models/encoder.py  → has its own slim version here that
#     handles only the 8 backbones in our reduced cohort. Expanding the
#     cohort means extending OUR encoder.py, not symlinking the 500-line
#     formal_v1 dispatcher (which imports `from src.config import
#     ExperimentConfig` — we use `SyntheticsConfig` instead).
#   - formal_v1/src/models/asr_model.py  → ASR-specific (CTC head, mel input).
#
# Usage:  bash scripts/setup_symlinks.sh   (run from experiments/synthetics_v1/)

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
FORMAL="$HERE/../formal_v1"

if [[ ! -d "$FORMAL" ]]; then
    echo "ERROR: formal_v1 not found at $FORMAL" >&2
    exit 1
fi

link() {
    local rel="$1"
    local src="$FORMAL/$rel"
    local dst="$HERE/$rel"
    if [[ ! -e "$src" ]]; then
        echo "  skip (missing source): $rel"
        return
    fi
    mkdir -p "$(dirname "$dst")"
    ln -sfn "$src" "$dst"
    echo "  linked: $rel"
}

echo "Linking leaf backbones from formal_v1 → synthetics_v1..."
# Transformer family
link "src/models/transformer.py"
link "src/models/transformer_causal.py"
# RWKV-6 family
link "src/models/rwkv6_encoder.py"
link "src/models/rwkv6_block.py"
link "src/models/rwkv6_time_mix.py"
link "src/models/rwkv6_channel_mix.py"
# Mamba / Mamba-2
link "src/models/mamba_encoder.py"
link "src/models/mamba_block.py"
link "src/models/mamba2_encoder.py"
link "src/models/mamba2_block.py"
link "src/models/mamba2_kernels.py"
# (mamba_cuda_encoder intentionally NOT linked — optional CUDA dep,
#  not in our reduced cohort.)
# Shared internals used by the leaves
link "src/models/lion_attention.py"
link "src/models/blocks.py"
link "src/models/p2rse_indep_lambda.py"
link "src/models/rse_scan_fast.py"
link "src/models/components.py"   # SinusoidalPE — used by transformer*
link "src/models/mechanisms"

echo "Done."
echo
echo "NOT linked (intentionally): encoder.py, asr_model.py, mamba_cuda_encoder.py"
echo "  → synthetics_v1 has its own src/models/encoder.py dispatcher."
