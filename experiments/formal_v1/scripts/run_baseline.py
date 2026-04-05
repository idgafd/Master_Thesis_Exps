#!/usr/bin/env python3
"""Run a single baseline encoder for N epochs on LibriSpeech.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_baseline.py --backbone transformer --epochs 10
    CUDA_VISIBLE_DEVICES=1 python3 scripts/run_baseline.py --backbone mamba --epochs 10
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.vocab import CharVocab
from src.data.dataset import ASRDataset, DurationBatchSampler, collate_fn
from src.data.augment import SpecAugment
from src.data.librispeech import load_librispeech
from src.models.blocks import (
    ConvSubsampling, CTCHead, SinusoidalPE,
    TransformerEncoder, LinearAttentionEncoder,
    RWKV6Encoder, MambaEncoder,
)
from src.training.schedulers import WarmupCosineScheduler
from src.training.decode import greedy_ctc_decode, compute_cer
from src.utils.misc import seed_everything, count_parameters, format_param_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class ASRModel(nn.Module):
    """CTC-based ASR: ConvSubsampling -> Encoder -> CTCHead."""

    def __init__(self, encoder: nn.Module, n_mels: int, d_model: int, vocab_size: int, conv_channels: int = 256):
        super().__init__()
        self.frontend = ConvSubsampling(n_mels, d_model, conv_channels)
        self.encoder = encoder
        self.ctc_head = CTCHead(d_model, vocab_size)

    def forward(self, mels, mel_lengths, state=None):
        x, lengths = self.frontend(mels, mel_lengths)
        x, new_state = self.encoder(x, lengths, state=state)
        logits = self.ctc_head(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, lengths, new_state


def build_encoder(backbone: str, d_model: int, n_layers: int, ffn_dim: int, dropout: float, head_size: int = 64, n_heads: int = 4):
    if backbone == "transformer":
        return TransformerEncoder(d_model, n_layers, n_heads, ffn_dim, dropout)
    elif backbone == "linear_attention":
        return LinearAttentionEncoder(d_model, n_layers, n_heads, ffn_dim, dropout)
    elif backbone == "rwkv6":
        return RWKV6Encoder(d_model, n_layers, dropout, head_size)
    elif backbone == "mamba":
        return MambaEncoder(d_model, n_layers, dropout, ffn_dim)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


@torch.no_grad()
def evaluate(model, dataloader, vocab, device):
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    all_hyps, all_refs = [], []
    total_loss, n_batches = 0.0, 0

    for mels, targets, mel_lengths, target_lengths in dataloader:
        mels = mels.to(device)
        targets = targets.to(device)
        mel_lengths = mel_lengths.to(device)
        target_lengths = target_lengths.to(device)

        log_probs, output_lengths, _ = model(mels, mel_lengths)
        output_lengths = torch.clamp(output_lengths, max=log_probs.size(1))

        loss = ctc_loss_fn(log_probs.permute(1, 0, 2), targets, output_lengths, target_lengths)
        total_loss += loss.item()
        n_batches += 1

        hyps = greedy_ctc_decode(log_probs.cpu(), output_lengths.cpu(), vocab)
        for i in range(targets.size(0)):
            ref = vocab.decode(targets[i, :target_lengths[i]].tolist())
            all_refs.append(ref)
        all_hyps.extend(hyps)

    from jiwer import wer as compute_wer
    cer = compute_cer(all_hyps, all_refs)
    wer_val = compute_wer(
        [" ".join(r.split()) if r.strip() else "<empty>" for r in all_refs],
        [" ".join(h.split()) if h.strip() else "<empty>" for h in all_hyps],
    )
    return {"loss": total_loss / max(n_batches, 1), "cer": cer, "wer": wer_val}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["transformer", "linear_attention", "rwkv6", "mamba"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.seed = args.seed
    cfg.num_epochs = args.epochs

    output_dir = f"./outputs/baseline_{args.backbone}_ep{args.epochs}_seed{args.seed}"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    logger.info(f"Backbone: {args.backbone} | Device: {device} | Epochs: {args.epochs}")

    # Vocab
    vocab = CharVocab.build_english()

    # Data
    logger.info("Loading LibriSpeech...")
    train_entries = load_librispeech("train", cfg.data_cache_dir, cfg.min_audio_sec, cfg.max_audio_sec)
    dev_entries = load_librispeech("dev", cfg.data_cache_dir, cfg.min_audio_sec, cfg.max_audio_sec)
    logger.info(f"Train: {len(train_entries)}, Dev: {len(dev_entries)}")

    train_ds = ASRDataset(train_entries, vocab, cfg)
    dev_ds = ASRDataset(dev_entries, vocab, cfg)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=DurationBatchSampler(train_entries, cfg.batch_max_seconds, True, cfg.seed),
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_sampler=DurationBatchSampler(dev_entries, cfg.batch_max_seconds, False, cfg.seed),
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    spec_aug = SpecAugment(cfg.freq_mask_param, cfg.time_mask_param, cfg.num_freq_masks, cfg.num_time_masks)

    # Model
    ffn_dim = cfg.ffn_dim
    encoder = build_encoder(args.backbone, cfg.d_model, cfg.n_layers, ffn_dim, cfg.dropout, cfg.head_size, cfg.n_heads)
    model = ASRModel(encoder, cfg.n_mels, cfg.d_model, vocab.size, cfg.conv_channels).to(device)

    pc = count_parameters(model)
    logger.info(f"Total params: {format_param_count(pc['total'])} | Trainable: {format_param_count(pc['trainable'])}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = WarmupCosineScheduler(optimizer, cfg.warmup_steps, total_steps)

    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    # Training
    history = []
    best_cer = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()

        for batch_idx, (mels, targets, mel_lengths, target_lengths) in enumerate(train_loader):
            mels = mels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mel_lengths = mel_lengths.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            mels = spec_aug(mels)

            log_probs, output_lengths, _ = model(mels, mel_lengths)
            output_lengths = torch.clamp(output_lengths, max=log_probs.size(1))

            loss = ctc_loss_fn(log_probs.permute(1, 0, 2), targets, output_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Skipping batch {batch_idx} (loss={loss.item():.4f})")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

            if batch_idx % 100 == 0:
                logger.info(
                    f"  [{args.backbone}] Ep {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_time = time.time() - t0

        # Evaluate
        dev_metrics = evaluate(model, dev_loader, vocab, device)

        entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_cer": dev_metrics["cer"],
            "dev_wer": dev_metrics["wer"],
            "epoch_time_sec": epoch_time,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(entry)

        improved = " *NEW BEST*" if dev_metrics["cer"] < best_cer else ""
        logger.info(
            f"[{args.backbone}] Epoch {epoch}/{args.epochs} | "
            f"Train: {avg_loss:.4f} | Dev CER: {dev_metrics['cer']:.4f} | "
            f"Dev WER: {dev_metrics['wer']:.4f} | Time: {epoch_time:.0f}s{improved}"
        )

        if dev_metrics["cer"] < best_cer:
            best_cer = dev_metrics["cer"]
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

    # Save results
    results = {
        "backbone": args.backbone,
        "params": pc,
        "epochs": args.epochs,
        "best_dev_cer": best_cer,
        "history": history,
        "config": {
            "d_model": cfg.d_model, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
            "head_size": cfg.head_size, "ffn_dim": ffn_dim, "dropout": cfg.dropout,
            "lr": cfg.lr, "batch_max_seconds": cfg.batch_max_seconds,
        },
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"[{args.backbone}] Done. Best dev CER: {best_cer:.4f}. Results in {output_dir}/")


if __name__ == "__main__":
    main()
