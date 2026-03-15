#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run 020: Delta Rule & LUCID experiments
#
# GPU 0: rwkv6_delta       → rwkv6_lucid        (sequential on same GPU)
# GPU 1: bidir_rwkv6_delta → bidir_rwkv6_lucid   (sequential on same GPU)
#
# Each backbone writes to its own output_dir — no race conditions.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/outputs/run-020/logs"

mkdir -p "$LOG_DIR"

run_queue() {
    local gpu="$1"
    shift
    local -a configs=("$@")

    (
        set -euo pipefail
        cd "$ROOT_DIR"
        source .venv/bin/activate

        for config in "${configs[@]}"; do
            local_name="$(basename "$config" .yaml)"
            log_path="$LOG_DIR/${local_name}.log"
            ln -sfn "$log_path" "$LOG_DIR/gpu${gpu}_current.log"
            echo "[$(date -Is)] START gpu=${gpu} config=${config}"
            CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_experiment.py \
                --config "$config" \
                --no-resume \
                --force 2>&1 | tee "$log_path"
            echo "[$(date -Is)] END gpu=${gpu} config=${config}"
        done
    ) > "$LOG_DIR/gpu${gpu}_queue.log" 2>&1 &

    echo $! > "$LOG_DIR/gpu${gpu}_queue.pid"
    echo "gpu${gpu} queue pid: $(cat "$LOG_DIR/gpu${gpu}_queue.pid")"
}

# GPU 0: unidirectional variants (recurrent, supports carry-state)
GPU0_CONFIGS=(
    "configs/run020/rwkv6_delta.yaml"
    "configs/run020/rwkv6_lucid.yaml"
)

# GPU 1: bidirectional variants (LION parallel form)
GPU1_CONFIGS=(
    "configs/run020/bidir_rwkv6_delta.yaml"
    "configs/run020/bidir_rwkv6_lucid.yaml"
)

run_queue 0 "${GPU0_CONFIGS[@]}"
run_queue 1 "${GPU1_CONFIGS[@]}"

echo ""
echo "Run 020 launched on 2 GPUs"
echo ""
echo "Logs:"
echo "  GPU 0 queue: $LOG_DIR/gpu0_queue.log"
echo "  GPU 1 queue: $LOG_DIR/gpu1_queue.log"
echo "  GPU 0 current: $LOG_DIR/gpu0_current.log"
echo "  GPU 1 current: $LOG_DIR/gpu1_current.log"
echo ""
echo "Individual experiment logs:"
echo "  $LOG_DIR/rwkv6_delta.log"
echo "  $LOG_DIR/rwkv6_lucid.log"
echo "  $LOG_DIR/bidir_rwkv6_delta.log"
echo "  $LOG_DIR/bidir_rwkv6_lucid.log"
echo ""
echo "Output directories (independent, no race conditions):"
echo "  ./outputs/run-020_rwkv6_delta/"
echo "  ./outputs/run-020_rwkv6_lucid/"
echo "  ./outputs/run-020_bidir_rwkv6_delta/"
echo "  ./outputs/run-020_bidir_rwkv6_lucid/"
echo ""
echo "Monitor with: tail -f $LOG_DIR/gpu{0,1}_current.log"
