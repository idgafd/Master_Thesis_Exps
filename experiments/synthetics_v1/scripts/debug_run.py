"""Phase-0 smoke test — fail-fast across all 8 reduced-cohort backbones.

Trains each backbone for 200 steps at T=64 and reports loss + per-sequence
accuracy. The point is NOT to converge — it is to verify that every
backbone instantiates, forwards, backwards, and produces a non-trivial
loss reduction. Total wall-clock: ~5–10 minutes on RTX PRO 6000.

Usage:
    cd experiments/synthetics_v1
    bash scripts/setup_symlinks.sh        # one-time
    uv run python scripts/debug_run.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from src.config import SyntheticsConfig                          # noqa: E402
from src.data.dataset import build_eval_loader, build_train_loader  # noqa: E402
from src.models.encoder import SUPPORTED_BACKBONES               # noqa: E402
from src.models.synthetic_model import SyntheticModel            # noqa: E402
from src.tasks.mqar import MQARSpec                              # noqa: E402
from src.training.train import train                             # noqa: E402


SMOKE_BACKBONES = sorted(SUPPORTED_BACKBONES)


def _smoke_cfg(backbone: str) -> SyntheticsConfig:
    return SyntheticsConfig(
        task="mqar",
        vocab_size=8192,
        key_vocab_size=4096,
        value_vocab_size=4096,
        seq_len=64,
        n_kv_pairs=16,
        n_queries=16,
        train_examples_per_epoch=8_000,
        eval_examples=512,
        batch_size=32,
        num_workers=0,                # single-process for clearer error traces
        backbone=backbone,
        d_model=256,
        n_layers=6,
        n_heads=4,
        head_size=64,
        dropout=0.1,
        max_steps=200,
        eval_every_steps=100,
        log_every_steps=50,
        early_stop_threshold=1.01,    # disable early stop
        early_stop_patience_evals=999,
        warmup_steps=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir="",                # filled in per-backbone below
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("debug_run")

    out_root = HERE / "outputs" / "_smoke"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    t0 = time.time()
    for i, backbone in enumerate(SMOKE_BACKBONES, 1):
        log.info("─" * 70)
        log.info("[%d/%d] SMOKE %s", i, len(SMOKE_BACKBONES), backbone)
        log.info("─" * 70)

        cfg = _smoke_cfg(backbone)
        cfg.output_dir = str(out_root / backbone)

        spec = MQARSpec(
            seq_len=cfg.seq_len, n_kv_pairs=cfg.n_kv_pairs,
            n_queries=cfg.resolved_n_queries, vocab_size=cfg.vocab_size,
            key_vocab_size=cfg.key_vocab_size, value_vocab_size=cfg.value_vocab_size,
        )
        steps_per_epoch = max(1, cfg.train_examples_per_epoch // cfg.batch_size)
        train_loader = build_train_loader(
            spec, steps_per_epoch, cfg.batch_size,
            base_seed=cfg.seed, num_workers=0,
        )
        eval_loader = build_eval_loader(
            spec, cfg.eval_examples, batch_size=cfg.batch_size, seed=0,
        )

        model = SyntheticModel(cfg)

        try:
            results = train(cfg, model, train_loader, eval_loader, Path(cfg.output_dir))
            verdict = "OK"
        except Exception as e:                              # noqa: BLE001
            log.exception("FAILED %s: %s", backbone, e)
            verdict = f"FAIL: {type(e).__name__}: {e}"
            results = {}

        summary.append({
            "backbone": backbone,
            "verdict": verdict,
            **{k: results.get(k) for k in (
                "n_parameters", "best_per_seq_acc", "final_per_seq_acc",
                "final_per_query_acc", "final_loss", "wall_sec",
            )},
        })

        # Free GPU memory between runs.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("─" * 70)
    log.info("SMOKE SUMMARY (total wall: %.0f s)", time.time() - t0)
    log.info("─" * 70)
    for r in summary:
        log.info(
            "  %-25s %-10s params=%s best_per_seq=%.3f final_per_query=%.3f loss=%.3f",
            r["backbone"], r["verdict"][:10],
            (f"{r['n_parameters']/1e6:.1f}M" if r["n_parameters"] else "—"),
            r.get("best_per_seq_acc") or 0.0,
            r.get("final_per_query_acc") or 0.0,
            r.get("final_loss") or float("nan"),
        )

    summary_path = out_root / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("summary written to %s", summary_path)

    failed = [r for r in summary if r["verdict"] != "OK"]
    if failed:
        log.error("SMOKE FAILED for %d backbones", len(failed))
        sys.exit(1)
    log.info("SMOKE OK for all %d backbones", len(summary))


if __name__ == "__main__":
    main()
