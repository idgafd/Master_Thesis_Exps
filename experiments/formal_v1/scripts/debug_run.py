#!/usr/bin/env python3
"""Debug run: 5-epoch training for all core backbones.

Validates:
- All backbones converge (loss decreasing)
- Parameter counts are matched (within 5%)
- Forward/backward pass shapes are correct
- No NaN/Inf in outputs
"""

import sys
import os
import logging

# Add project root to path
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
from src.training.evaluate import evaluate
from src.training.schedulers import WarmupCosineScheduler
from src.utils.misc import seed_everything, count_parameters, format_param_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Core backbones for debug validation
DEBUG_BACKBONES = [
    "transformer",
    "rwkv6",
    "mamba",
    "lion",
    "lion_convshift",
    "lion_lucid",
    "lion_delta",
]

DEBUG_EPOCHS = 5


def run_debug():
    cfg = load_config("configs/default.yaml", overrides={
        "num_epochs": DEBUG_EPOCHS,
        "batch_max_seconds": 120,
        "warmup_steps": 50,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    seed_everything(cfg.seed)

    # Build vocab
    vocab = CharVocab.build_english()
    logger.info(f"Vocab size: {vocab.size}")

    # Load data (small subset for debug)
    logger.info("Loading LibriSpeech train-clean-100...")
    train_entries = load_librispeech(
        "train", cache_dir=cfg.data_cache_dir,
        min_audio_sec=cfg.min_audio_sec, max_audio_sec=cfg.max_audio_sec,
    )
    dev_entries = load_librispeech(
        "dev", cache_dir=cfg.data_cache_dir,
        min_audio_sec=cfg.min_audio_sec, max_audio_sec=cfg.max_audio_sec,
    )
    logger.info(f"Train: {len(train_entries)}, Dev: {len(dev_entries)}")

    # Datasets
    train_dataset = ASRDataset(train_entries, vocab, cfg)
    dev_dataset = ASRDataset(dev_entries, vocab, cfg)

    train_sampler = DurationBatchSampler(train_entries, cfg.batch_max_seconds, shuffle=True, seed=cfg.seed)
    dev_sampler = DurationBatchSampler(dev_entries, cfg.batch_max_seconds, shuffle=False, seed=cfg.seed)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_sampler=dev_sampler, collate_fn=collate_fn, num_workers=4)

    spec_aug = SpecAugment(
        freq_mask_param=cfg.freq_mask_param,
        time_mask_param=cfg.time_mask_param,
        num_freq_masks=cfg.num_freq_masks,
        num_time_masks=cfg.num_time_masks,
    ) if cfg.spec_augment else None

    # Run each backbone
    results = {}
    param_counts = {}

    for backbone in DEBUG_BACKBONES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing backbone: {backbone}")
        logger.info(f"{'='*60}")

        seed_everything(cfg.seed)
        cfg.backbone = backbone

        model = ASRModel(vocab_size=vocab.size, cfg=cfg).to(device)
        pc = count_parameters(model)
        param_counts[backbone] = pc
        logger.info(f"  Total params: {format_param_count(pc['total'])}")
        logger.info(f"  Trainable:    {format_param_count(pc['trainable'])}")
        for name, count in pc.items():
            if name not in ("total", "trainable"):
                logger.info(f"    {name}: {format_param_count(count)}")

        # Quick shape test
        dummy_mel = torch.randn(2, cfg.n_mels, 200, device=device)
        dummy_len = torch.tensor([200, 150], device=device)
        with torch.no_grad():
            log_probs, out_lens, _ = model(dummy_mel, dummy_len)
        logger.info(f"  Shape test: input (2, {cfg.n_mels}, 200) -> output {tuple(log_probs.shape)}")
        assert not torch.isnan(log_probs).any(), f"{backbone}: NaN in output!"
        assert not torch.isinf(log_probs).any(), f"{backbone}: Inf in output!"

        # Train
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        total_steps = len(train_loader) * DEBUG_EPOCHS
        scheduler = WarmupCosineScheduler(optimizer, cfg.warmup_steps, total_steps)

        losses = []
        for epoch in range(1, DEBUG_EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, optimizer, scheduler, spec_aug, cfg, epoch, device)
            losses.append(loss)
            logger.info(f"  Epoch {epoch}/{DEBUG_EPOCHS} — Train loss: {loss:.4f}")

        # Evaluate
        dev_metrics = evaluate(model, dev_loader, vocab, device, tag=f"dev-{backbone}")
        results[backbone] = {
            "train_losses": losses,
            "dev_cer": dev_metrics["cer"],
            "dev_wer": dev_metrics["wer"],
            "dev_loss": dev_metrics["loss"],
            "params": pc["total"],
        }

        # Check convergence
        if losses[-1] < losses[0]:
            logger.info(f"  PASS: loss decreased ({losses[0]:.4f} -> {losses[-1]:.4f})")
        else:
            logger.warning(f"  WARN: loss did NOT decrease ({losses[0]:.4f} -> {losses[-1]:.4f})")

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    ref_params = param_counts.get("lion", {}).get("total", 1)
    for backbone in DEBUG_BACKBONES:
        r = results[backbone]
        pct = (r["params"] / ref_params - 1) * 100
        match_str = "OK" if abs(pct) < 5 else "MISMATCH"
        logger.info(
            f"  {backbone:30s} | {format_param_count(r['params']):>8s} ({pct:+.1f}% vs LION) [{match_str}] | "
            f"CER: {r['dev_cer']:.4f} | WER: {r['dev_wer']:.4f}"
        )


if __name__ == "__main__":
    run_debug()
