#!/usr/bin/env python3
"""Full training script for PyTorch Mamba with torch.compile.

Designed for GPUs with >=48GB VRAM (A100/H100). On the RTX 5090 (32GB)
this will OOM because torch.compile is incompatible with gradient
checkpointing — see RESULTS.md "torch.compile — faster than CUDA mamba-ssm,
but blocked by VRAM".

Compared to `run_experiment.py --backbone mamba --compile`, this script:
  - Forces bf16 autocast (halves activation memory)
  - Uses `mode="max-autotune"` for the best steady-state throughput
  - Sets `TORCH_INDUCTOR_CACHE_DIR` so the compiled graph survives reruns
  - Falls back gracefully if `torch.compiler.is_compiling()` is detected
    inside any module that still uses gradient checkpointing

Usage on a big GPU:
    uv run scripts/run_mamba_compiled.py \
        --epochs 80 --seed 42 \
        --output-dir outputs/mamba_compiled_ep80_seed42

Expected speed (based on fixed-batch microbenchmark, B=8, T=500):
    PyTorch eager (checkpointed): ~420s/epoch
    torch.compile (no checkpoint): target ~60-80s/epoch (matches CUDA mamba-ssm)
"""

import argparse
import json
import logging
import os
import sys
import time
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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode. 'max-autotune' tunes kernels per shape (slow first batch).",
    )
    parser.add_argument(
        "--batch-max-seconds",
        type=float,
        default=None,
        help="Override batch_max_seconds. Reduce (e.g. 150) if OOM on smaller GPU.",
    )
    parser.add_argument(
        "--autocast",
        default="bf16",
        choices=["off", "fp16", "bf16"],
        help="Mixed precision (halves activation memory). bf16 preferred on RTX 5090/A100/H100.",
    )
    args = parser.parse_args()

    # ── Config ─────────────────────────────────────────────────────────────
    overrides = {"backbone": "mamba", "compile_encoder": True}
    if args.seed:
        overrides["seed"] = args.seed
    if args.epochs:
        overrides["num_epochs"] = args.epochs
    if args.batch_max_seconds:
        overrides["batch_max_seconds"] = args.batch_max_seconds

    cfg = load_config(args.config, overrides)
    if not args.output_dir:
        cfg.output_dir = f"./outputs/mamba_compiled_ep{cfg.num_epochs}_seed{cfg.seed}"
    else:
        cfg.output_dir = args.output_dir
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    seed_everything(cfg.seed)

    # Persistent compile cache — lets reruns skip compilation
    cache_dir = os.path.abspath(os.path.join(cfg.output_dir, ".inductor_cache"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)
    logger.info(f"Inductor cache: {cache_dir}")

    # ── Data ───────────────────────────────────────────────────────────────
    vocab = CharVocab.build_english()
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

    spec_aug = (
        SpecAugment(cfg.freq_mask_param, cfg.time_mask_param, cfg.num_freq_masks, cfg.num_time_masks)
        if cfg.spec_augment else None
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = ASRModel(vocab_size=vocab.size, cfg=cfg).to(device)
    pc = count_parameters(model)
    logger.info(f"Backbone: mamba (compiled) | Params: {format_param_count(pc['total'])}")
    logger.info(f"  Frontend: {format_param_count(pc['frontend'])}")
    logger.info(f"  Encoder:  {format_param_count(pc['encoder'])}")
    logger.info(f"  Head:     {format_param_count(pc['ctc_head'])}")

    # Compile the encoder. Gradient checkpointing inside MambaEncoder is
    # auto-disabled via torch.compiler.is_compiling() check.
    logger.info(f"Compiling encoder with torch.compile(mode={args.compile_mode!r})...")
    model.encoder = torch.compile(model.encoder, mode=args.compile_mode, dynamic=True)

    # ── Optim ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = WarmupCosineScheduler(optimizer, cfg.warmup_steps, total_steps)

    # AMP setup
    autocast_dtype = {"off": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.autocast]
    use_amp = autocast_dtype is not None
    grad_scaler = torch.amp.GradScaler("cuda") if autocast_dtype == torch.float16 else None
    logger.info(f"Autocast: {args.autocast}")

    # ── Training loop ──────────────────────────────────────────────────────
    best_cer = float("inf")
    patience_counter = 0
    history = []

    ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0

        for batch_idx, (mels, targets, mel_lens, tgt_lens) in enumerate(train_loader):
            mels = mels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mel_lens = mel_lens.to(device, non_blocking=True)
            tgt_lens = tgt_lens.to(device, non_blocking=True)

            if spec_aug is not None:
                mels = spec_aug(mels)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast("cuda", dtype=autocast_dtype):
                    log_probs, out_lens, _ = model(mels, mel_lens)
                    log_probs_ctc = log_probs.permute(1, 0, 2).float()  # CTC wants fp32
                    out_lens = torch.clamp(out_lens, max=log_probs.size(1))
                    loss = ctc_loss_fn(log_probs_ctc, targets, out_lens, tgt_lens)
            else:
                log_probs, out_lens, _ = model(mels, mel_lens)
                log_probs_ctc = log_probs.permute(1, 0, 2)
                out_lens = torch.clamp(out_lens, max=log_probs.size(1))
                loss = ctc_loss_fn(log_probs_ctc, targets, out_lens, tgt_lens)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Skipping batch {batch_idx} (loss={loss.item():.4f})")
                continue

            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            n_batches += 1

            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        train_loss = total_loss / max(n_batches, 1)
        dev_metrics = evaluate(model, dev_loader, vocab, device, tag="dev")
        epoch_time = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_time_sec": epoch_time,
            **{f"dev_{k}": v for k, v in dev_metrics.items()},
        }
        history.append(entry)
        logger.info(
            f"Epoch {epoch} | Train: {train_loss:.4f} | Dev CER: {dev_metrics['cer']:.4f} | "
            f"Dev WER: {dev_metrics['wer']:.4f} | Time: {epoch_time:.0f}s"
        )

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

    # ── Final evaluation ───────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(cfg.output_dir, "best_model.pt"), weights_only=True))
    test_metrics = evaluate(model, test_loader, vocab, device, tag="test")

    chunked_results = {}
    for chunk_sec in cfg.chunk_sizes_sec:
        for carry in [False, True]:
            result = evaluate_chunked(model, dev_ds, vocab, chunk_sec, cfg, device, carry_state=carry)
            if result is not None:
                mode = "carry" if carry else "reset"
                chunked_results[f"{chunk_sec}s_{mode}"] = result

    final_results = {
        "backbone": "mamba_compiled",
        "compile_mode": args.compile_mode,
        "autocast": args.autocast,
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
