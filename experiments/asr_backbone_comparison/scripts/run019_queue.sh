#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/outputs/run-019/logs"

mkdir -p "$LOG_DIR"

run_queue() {
    local gpu="$1"
    shift
    local -a configs=("$@")
    local queue_log="$LOG_DIR/gpu${gpu}_queue.log"

    cd "$ROOT_DIR"
    source .venv/bin/activate

    for config in "${configs[@]}"; do
        local name
        local log_path
        name="$(basename "$config" .yaml)"
        log_path="$LOG_DIR/${name}.log"
        ln -sfn "$log_path" "$LOG_DIR/gpu${gpu}_current.log"

        echo "[$(date -Is)] START gpu=${gpu} config=${config}" | tee -a "$queue_log"
        CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_experiment.py \
            --config "$config" \
            --no-resume \
            --force 2>&1 | tee "$log_path"
        echo "[$(date -Is)] END gpu=${gpu} config=${config}" | tee -a "$queue_log"
    done
}

case "${1:-}" in
    gpu0)
        run_queue 0 \
            "configs/run019/rwkv6_default_cosine100.yaml"
        ;;
    gpu1)
        run_queue 1 \
            "configs/run019/rwkv7_default_cosine100.yaml"
        ;;
    *)
        echo "Usage: $0 gpu0|gpu1" >&2
        exit 2
        ;;
esac
