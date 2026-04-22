"""Single-run entry point for synthetics_v1 MQAR experiments.

Usage:
    cd experiments/synthetics_v1
    bash scripts/setup_symlinks.sh        # one-time
    uv run python scripts/run_experiment.py \
        --config configs/default.yaml \
        --backbone rwkv6_delta \
        --seq-len 256 \
        --output-dir outputs/rwkv6_delta_T256_seed42

CLI overrides win over YAML values. Anything not on the CLI uses YAML;
anything not in YAML uses the SyntheticsConfig dataclass default.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from src.config import SyntheticsConfig, load_config            # noqa: E402
from src.data.dataset import build_eval_loader, build_train_loader  # noqa: E402
from src.models.synthetic_model import SyntheticModel           # noqa: E402
from src.tasks.mqar import MQARSpec                             # noqa: E402
from src.training.train import train                            # noqa: E402


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="synthetics_v1 single-run")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--backbone")
    p.add_argument("--seq-len", type=int)
    p.add_argument("--n-kv-pairs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default=None,
                   help="cuda | cpu (default: cuda if available)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("run_experiment")

    overrides = {}
    if args.backbone:
        overrides["backbone"] = args.backbone
    if args.seq_len is not None:
        overrides["seq_len"] = args.seq_len
    if args.n_kv_pairs is not None:
        overrides["n_kv_pairs"] = args.n_kv_pairs
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps

    if args.device is not None:
        overrides["device"] = args.device
    elif not torch.cuda.is_available():
        overrides["device"] = "cpu"

    overrides["output_dir"] = args.output_dir

    cfg = load_config(args.config, overrides=overrides)

    # Default n_kv_pairs ≈ seq_len / 4 if not specified anywhere.
    if cfg.n_kv_pairs <= 0:
        cfg.n_kv_pairs = max(1, cfg.seq_len // 4)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist resolved config for later analysis.
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.__dict__, sort_keys=True)
    )
    (output_dir / "cli_args.txt").write_text(" ".join(sys.argv) + "\n")

    log.info("resolved config: %s", cfg)

    _seed_all(cfg.seed)

    spec = MQARSpec(
        seq_len=cfg.seq_len,
        n_kv_pairs=cfg.n_kv_pairs,
        n_queries=cfg.resolved_n_queries,
        vocab_size=cfg.vocab_size,
        key_vocab_size=cfg.key_vocab_size,
        value_vocab_size=cfg.value_vocab_size,
    )

    steps_per_epoch = max(1, cfg.train_examples_per_epoch // cfg.batch_size)
    train_loader = build_train_loader(
        spec, steps_per_epoch, cfg.batch_size,
        base_seed=cfg.seed, num_workers=cfg.num_workers,
    )
    eval_loader = build_eval_loader(
        spec, cfg.eval_examples, batch_size=cfg.batch_size, seed=0,
    )

    model = SyntheticModel(cfg)

    train(cfg, model, train_loader, eval_loader, output_dir)


if __name__ == "__main__":
    main()
