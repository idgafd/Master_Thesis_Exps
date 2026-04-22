"""Training loop for synthetic tasks (MQAR Stage 1).

Self-contained: AdamW + warmup-cosine, masked cross-entropy, periodic eval,
early-stop on per-sequence accuracy. Artifacts written in the same flat-file
layout as `formal_v1` so existing analysis tooling can be pointed at the
synthetics outputs.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import SyntheticsConfig
from src.tasks.mqar import IGNORE_INDEX
from src.training.evaluate import evaluate
from src.training.schedulers import WarmupCosineScheduler


log = logging.getLogger(__name__)


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(
    cfg: SyntheticsConfig,
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    output_dir: Path,
) -> dict:
    """Run training. Returns final results dict (also written to results.json)."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    results_path = output_dir / "results.json"
    log_path = output_dir / "train.log"

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    device = torch.device(cfg.device)
    model.to(device)

    n_params = _count_parameters(model)
    log.info(
        "backbone=%s params=%dM seq_len=%d K=%d Q=%d batch=%d",
        cfg.backbone, n_params // 1_000_000, cfg.seq_len, cfg.n_kv_pairs,
        cfg.resolved_n_queries, cfg.batch_size,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=cfg.warmup_steps,
        total_steps=cfg.max_steps,
    )

    # Track BOTH metrics. per_query_acc is the fine-grained learning signal
    # (continuous, monotone-ish in real training); per_seq_acc is the headline
    # convergence target but stays at 0 until the model is essentially perfect,
    # which makes it useless for patience-based early-stop. So patience and
    # threshold both run off per_query_acc; per_seq_acc is reported for sanity.
    best_per_query_acc = -1.0
    best_per_seq_acc = -1.0
    best_step = -1
    evals_since_best = 0
    PATIENCE_MIN_DELTA = 1e-3
    start_t = time.time()

    metrics_f = open(metrics_path, "a", buffering=1)  # line-buffered

    def write_metric(record: dict) -> None:
        metrics_f.write(json.dumps(record) + "\n")

    step = 0
    epoch = 0
    early_stopped = False

    while step < cfg.max_steps and not early_stopped:
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        epoch += 1

        for batch in train_loader:
            input_ids = batch.input_ids.to(device, non_blocking=True)
            targets = batch.targets.to(device, non_blocking=True)
            lengths = batch.lengths.to(device, non_blocking=True)

            logits, _ = model(input_ids, lengths)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            )
            optimizer.step()
            scheduler.step()

            step += 1

            if step % cfg.log_every_steps == 0:
                lr = scheduler.get_last_lr()[0]
                log.info(
                    "step=%d epoch=%d loss=%.4f grad=%.3f lr=%.2e",
                    step, epoch, loss.item(), grad_norm.item(), lr,
                )
                write_metric({
                    "step": step, "epoch": epoch, "split": "train",
                    "loss": loss.item(),
                    "grad_norm": grad_norm.item(),
                    "lr": lr,
                    "wall_sec": time.time() - start_t,
                })

            if step % cfg.eval_every_steps == 0 or step >= cfg.max_steps:
                ev = evaluate(model, eval_loader, device)
                log.info(
                    "EVAL step=%d loss=%.4f per_query_acc=%.4f per_seq_acc=%.4f",
                    step, ev.loss, ev.per_query_acc, ev.per_seq_acc,
                )
                write_metric({
                    "step": step, "epoch": epoch, "split": "eval",
                    "loss": ev.loss,
                    "per_query_acc": ev.per_query_acc,
                    "per_seq_acc": ev.per_seq_acc,
                    "n_eval_examples": ev.n_examples,
                    "wall_sec": time.time() - start_t,
                })

                # Headline metric for results.json — track best independently.
                if ev.per_seq_acc > best_per_seq_acc:
                    best_per_seq_acc = ev.per_seq_acc

                # Patience metric — per_query_acc is what actually moves during
                # training. Require a min-delta to avoid noise-triggered resets.
                if ev.per_query_acc > best_per_query_acc + PATIENCE_MIN_DELTA:
                    best_per_query_acc = ev.per_query_acc
                    best_step = step
                    evals_since_best = 0
                else:
                    evals_since_best += 1

                # Early stop: per_seq_acc ≥ threshold (Zoology convergence
                # convention) OR per_query_acc plateau for N evals.
                if ev.per_seq_acc >= cfg.early_stop_threshold:
                    log.info(
                        "EARLY STOP: per_seq_acc %.4f ≥ threshold %.4f at step %d",
                        ev.per_seq_acc, cfg.early_stop_threshold, step,
                    )
                    early_stopped = True
                    break
                if evals_since_best >= cfg.early_stop_patience_evals:
                    log.info(
                        "EARLY STOP: %d evals without per_query_acc improvement "
                        "(best=%.4f at step %d, per_seq_acc=%.4f)",
                        evals_since_best, best_per_query_acc, best_step,
                        best_per_seq_acc,
                    )
                    early_stopped = True
                    break

            if step >= cfg.max_steps:
                break

    # Final eval (one last reading at exit).
    final = evaluate(model, eval_loader, device)
    write_metric({
        "step": step, "epoch": epoch, "split": "eval_final",
        "loss": final.loss,
        "per_query_acc": final.per_query_acc,
        "per_seq_acc": final.per_seq_acc,
        "n_eval_examples": final.n_examples,
        "wall_sec": time.time() - start_t,
    })
    metrics_f.close()

    results = {
        "backbone": cfg.backbone,
        "seq_len": cfg.seq_len,
        "n_kv_pairs": cfg.n_kv_pairs,
        "n_queries": cfg.resolved_n_queries,
        "vocab_size": cfg.vocab_size,
        "n_parameters": n_params,
        "steps_taken": step,
        "epochs_taken": epoch,
        "early_stopped": early_stopped,
        "best_per_seq_acc": best_per_seq_acc,
        "best_per_query_acc": best_per_query_acc,
        "best_step": best_step,
        "final_per_seq_acc": final.per_seq_acc,
        "final_per_query_acc": final.per_query_acc,
        "final_loss": final.loss,
        "wall_sec": time.time() - start_t,
        "seed": cfg.seed,
    }
    results_path.write_text(json.dumps(results, indent=2))
    log.info("DONE: %s", json.dumps(results, indent=2))
    return results
