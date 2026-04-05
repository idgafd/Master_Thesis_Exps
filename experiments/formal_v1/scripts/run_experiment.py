#!/usr/bin/env python3
"""Full experiment runner for formal_v1.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml --backbone lion
    python scripts/run_experiment.py --config configs/default.yaml --backbone lion_convshift
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.vocab import CharVocab
from src.data.dataset import ASRDataset, DurationBatchSampler, collate_fn
from src.data.augment import SpecAugment
from src.data.librispeech import load_librispeech
from src.models.asr_model import ASRModel
from src.training.train import train_one_epoch
from src.training.evaluate import evaluate, evaluate_chunked
from src.training.schedulers import WarmupCosineScheduler
from src.utils.misc import seed_everything, count_parameters, format_param_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    overrides = {"backbone": args.backbone}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.seed:
        overrides["seed"] = args.seed

    cfg = load_config(args.config, overrides)
    if not args.output_dir:
        cfg.output_dir = f"./outputs/{args.backbone}_seed{cfg.seed}"

    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    # Vocab
    vocab = CharVocab.build_english()

    # Data
    train_entries = load_librispeech("train", cfg.data_cache_dir, cfg.min_audio_sec, cfg.max_audio_sec)
    dev_entries = load_librispeech("dev", cfg.data_cache_dir, cfg.min_audio_sec, cfg.max_audio_sec)
    test_entries = load_librispeech("test", cfg.data_cache_dir, cfg.min_audio_sec, cfg.max_audio_sec)

    train_ds = ASRDataset(train_entries, vocab, cfg)
    dev_ds = ASRDataset(dev_entries, vocab, cfg)
    test_ds = ASRDataset(test_entries, vocab, cfg)

    train_loader = DataLoader(
        train_ds, batch_sampler=DurationBatchSampler(train_entries, cfg.batch_max_seconds, True, cfg.seed),
        collate_fn=collate_fn, num_workers=4,
    )
    dev_loader = DataLoader(
        dev_ds, batch_sampler=DurationBatchSampler(dev_entries, cfg.batch_max_seconds, False, cfg.seed),
        collate_fn=collate_fn, num_workers=4,
    )
    test_loader = DataLoader(
        test_ds, batch_sampler=DurationBatchSampler(test_entries, cfg.batch_max_seconds, False, cfg.seed),
        collate_fn=collate_fn, num_workers=4,
    )

    spec_aug = SpecAugment(cfg.freq_mask_param, cfg.time_mask_param, cfg.num_freq_masks, cfg.num_time_masks) if cfg.spec_augment else None

    # Model
    model = ASRModel(vocab_size=vocab.size, cfg=cfg).to(device)
    pc = count_parameters(model)
    logger.info(f"Backbone: {args.backbone} | Params: {format_param_count(pc['total'])}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = WarmupCosineScheduler(optimizer, cfg.warmup_steps, total_steps)

    # Training loop
    best_cer = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, spec_aug, cfg, epoch, device)
        dev_metrics = evaluate(model, dev_loader, vocab, device, tag="dev")

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"dev_{k}": v for k, v in dev_metrics.items()},
        }
        history.append(entry)
        logger.info(f"Epoch {epoch} | Train: {train_loss:.4f} | Dev CER: {dev_metrics['cer']:.4f} | Dev WER: {dev_metrics['wer']:.4f}")

        # Checkpointing
        if dev_metrics["cer"] < best_cer:
            best_cer = dev_metrics["cer"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model.pt"))
            logger.info(f"  New best CER: {best_cer:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= cfg.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Final test evaluation
    model.load_state_dict(torch.load(os.path.join(cfg.output_dir, "best_model.pt"), weights_only=True))
    test_metrics = evaluate(model, test_loader, vocab, device, tag="test")

    # Chunked evaluation
    chunked_results = {}
    for chunk_sec in cfg.chunk_sizes_sec:
        for carry in [False, True]:
            result = evaluate_chunked(model, dev_ds, vocab, chunk_sec, cfg, device, carry_state=carry)
            if result is not None:
                mode = "carry" if carry else "reset"
                chunked_results[f"{chunk_sec}s_{mode}"] = result

    # Save results
    final_results = {
        "backbone": args.backbone,
        "params": pc,
        "best_dev_cer": best_cer,
        "test": test_metrics,
        "chunked": chunked_results,
        "history": history,
    }
    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Done. Best dev CER: {best_cer:.4f} | Test CER: {test_metrics['cer']:.4f}")


if __name__ == "__main__":
    main()
