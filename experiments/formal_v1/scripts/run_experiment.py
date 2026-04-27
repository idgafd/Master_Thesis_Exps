#!/usr/bin/env python3
"""Full experiment runner for formal_v1.

Writes all artifacts to `{output_dir}/`:
    history.csv, metrics.jsonl, plots/*.png    (training)
    best_model.pt, last_model.pt, checkpoint_ep{N}.pt  (weights)
    results.json, config.yaml, cli_args.txt, git_sha.txt (metadata)
    train.log (streaming)

Usage:
    uv run scripts/run_experiment.py --backbone lion
    uv run scripts/run_experiment.py --backbone mamba --compile --gpu 1
    uv run scripts/run_experiment.py --backbone lion --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.vocab import CharVocab
from src.data.dataset import ASRDataset, DurationBatchSampler, collate_fn, compute_mel
from src.data.augment import SpecAugment
from src.data import load_dataset_split
from src.data.common_voice import TRAIN_MANIFEST_NAME
from src.models.asr_model import ASRModel
from src.training.train import train_one_epoch
from src.training.evaluate import evaluate, evaluate_chunked
from src.training.schedulers import WarmupCosineScheduler
from src.training.metrics import MetricLogger
from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_resume_point,
    get_git_sha,
)
from src.training.run_plots import render_run_plots
from src.utils.misc import seed_everything, count_parameters, format_param_count

logger = logging.getLogger(__name__)


@torch.no_grad()
def _compute_wer_by_speaker(model, entries, vocab, cfg, device) -> dict:
    """Per-speaker WER on the test split — used by Common Voice runs only.

    Single-utterance forward passes so we can keep `client_id` alongside
    each (hyp, ref) pair without changing the dataset/dataloader contract.
    Cost: ~10–20× the batched eval; acceptable for a once-per-run summary.
    """
    from src.training.decode import greedy_ctc_decode
    from jiwer import wer as compute_wer

    model.eval()
    by_speaker: dict[str, dict] = {}
    for e in entries:
        client_id = e.get("client_id", "<unknown>")
        mel = compute_mel(e["audio_array"], e["sample_rate"], cfg).unsqueeze(0).to(device)
        mel_lengths = torch.tensor([mel.shape[2]], device=device)
        log_probs, output_lengths, _ = model(mel, mel_lengths)
        output_lengths = torch.clamp(output_lengths, max=log_probs.size(1))
        hyp = greedy_ctc_decode(log_probs.cpu(), output_lengths.cpu(), vocab)[0]
        ref = e["text"]
        d = by_speaker.setdefault(client_id, {"refs": [], "hyps": []})
        d["refs"].append(ref if ref.strip() else "<empty>")
        d["hyps"].append(hyp if hyp.strip() else "<empty>")

    out: dict[str, dict] = {}
    for client_id, d in by_speaker.items():
        try:
            w = float(compute_wer(d["refs"], d["hyps"]))
        except Exception:
            w = None
        out[client_id] = {"wer": w, "n_utterances": len(d["refs"])}
    return out


def _setup_logging(run_dir: Path) -> None:
    """Log to both stderr and `run_dir/train.log`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(run_dir / "train.log", mode="a"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def _write_run_metadata(run_dir: Path, cfg, cli_args: list[str]) -> None:
    """Snapshot config, CLI args, and git SHA to the run dir."""
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
    with open(run_dir / "cli_args.txt", "w") as f:
        f.write(" ".join(cli_args) + "\n")
    with open(run_dir / "git_sha.txt", "w") as f:
        f.write(get_git_sha() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the encoder (OOMs on 32GB GPUs)")
    parser.add_argument("--fast", action="store_true",
                        help="Enable TF32 matmul + cudnn.benchmark autotuner. "
                             "Sacrifices strict bit-exact reproducibility "
                             "across cudnn kernel choices for ~2-3x speedup. "
                             "Loss curves match seeded reference within "
                             "fp32/cudnn-kernel-noise; standard for production "
                             "training. Stage 12 default to fight the K×K "
                             "delta-scan cost.")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device index (default: auto)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last_model.pt if present in output_dir")
    parser.add_argument("--dataset", default=None,
                        choices=["librispeech_clean", "librispeech_other",
                                 "common_voice_en_100h"],
                        help="Dataset selector. Overrides cfg.dataset from YAML.")
    args = parser.parse_args()

    overrides = {"backbone": args.backbone}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.seed:
        overrides["seed"] = args.seed
    if args.epochs:
        overrides["num_epochs"] = args.epochs
    if args.compile:
        overrides["compile_encoder"] = True
    if args.dataset:
        overrides["dataset"] = args.dataset

    cfg = load_config(args.config, overrides)
    if not args.output_dir:
        cfg.output_dir = f"./outputs/{args.backbone}_seed{cfg.seed}"
    run_dir = Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(run_dir)
    _write_run_metadata(run_dir, cfg, sys.argv[1:])

    device = (
        torch.device(f"cuda:{args.gpu}") if args.gpu is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.gpu is not None:
        torch.cuda.set_device(device)
    seed_everything(cfg.seed)

    # ── --fast: TF32 matmul only, applied AFTER seed_everything ────────────
    # `seed_everything()` sets cudnn.deterministic=True and benchmark=False
    # for bit-exact reproducibility.  --fast turns on TF32 matmul precision
    # (use Blackwell tensor cores for fp32 BMMs in the K×K delta scan and
    # frontend convs).  cudnn.benchmark is INTENTIONALLY left off because
    # the duration-bucketed dataloader produces variable (B, T) shapes,
    # and benchmark mode re-tunes per shape — measured to be ~50% SLOWER
    # in our setup than benchmark=False.  TF32 alone gave the win.
    if args.fast:
        torch.set_float32_matmul_precision("high")
        logger.info("--fast: TF32 matmul precision enabled (cudnn.benchmark left off).")

    # ── Data ───────────────────────────────────────────────────────────────
    vocab = CharVocab.build_english()
    train_entries = load_dataset_split(cfg, "train")
    dev_entries = load_dataset_split(cfg, "dev")
    test_entries = load_dataset_split(cfg, "test")

    train_ds = ASRDataset(train_entries, vocab, cfg)
    dev_ds = ASRDataset(dev_entries, vocab, cfg)
    test_ds = ASRDataset(test_entries, vocab, cfg)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=DurationBatchSampler(train_entries, cfg.batch_max_seconds, True, cfg.seed),
        collate_fn=collate_fn, num_workers=4,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_sampler=DurationBatchSampler(dev_entries, cfg.batch_max_seconds, False, cfg.seed),
        collate_fn=collate_fn, num_workers=4,
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=DurationBatchSampler(test_entries, cfg.batch_max_seconds, False, cfg.seed),
        collate_fn=collate_fn, num_workers=4,
    )

    spec_aug = (
        SpecAugment(cfg.freq_mask_param, cfg.time_mask_param, cfg.num_freq_masks, cfg.num_time_masks)
        if cfg.spec_augment else None
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = ASRModel(vocab_size=vocab.size, cfg=cfg).to(device)
    pc = count_parameters(model)
    logger.info(f"Backbone: {args.backbone} | Params: {format_param_count(pc['total'])}")
    logger.info(f"  Frontend: {format_param_count(pc['frontend'])} | "
                f"Encoder: {format_param_count(pc['encoder'])} | "
                f"Head: {format_param_count(pc['ctc_head'])}")

    if cfg.compile_encoder:
        torch.set_float32_matmul_precision("high")
        logger.info("Compiling encoder with torch.compile (first batch will be slow)...")
        model.encoder = torch.compile(model.encoder, mode="default", dynamic=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = WarmupCosineScheduler(optimizer, cfg.warmup_steps, total_steps)

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch = 1
    best_cer = float("inf")
    patience_counter = 0
    global_step = 0

    if args.resume:
        resume_path = find_resume_point(run_dir)
        if resume_path is None:
            logger.info("No last_model.pt found — starting from scratch")
        else:
            logger.info(f"Resuming from {resume_path}")
            state = load_checkpoint(
                resume_path, model=model, optimizer=optimizer,
                scheduler=scheduler, map_location=device,
            )
            start_epoch = state["epoch"] + 1
            best_cer = state["best_cer"]
            patience_counter = state["patience_counter"]
            # Best-effort global step: reconstruct from history if available
            history_csv = run_dir / "history.csv"
            if history_csv.exists():
                import pandas as pd
                prev = pd.read_csv(history_csv)
                if "last_step" in prev.columns and len(prev):
                    global_step = int(prev["last_step"].iloc[-1])

    # ── Training loop ──────────────────────────────────────────────────────
    git_sha = get_git_sha()
    metric_logger = MetricLogger(run_dir, step_log_interval=50)

    try:
        for epoch in range(start_epoch, cfg.num_epochs + 1):
            epoch_t0 = time.time()
            torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

            train_stats = train_one_epoch(
                model, train_loader, optimizer, scheduler, spec_aug,
                cfg, epoch, device,
                metric_logger=metric_logger,
                global_step_offset=global_step,
            )
            global_step = train_stats["last_step"]

            dev_metrics = evaluate(model, dev_loader, vocab, device, tag="dev")
            epoch_time = time.time() - epoch_t0
            peak_mem = (
                torch.cuda.max_memory_allocated(device) / 1e9
                if device.type == "cuda" else 0.0
            )

            row = {
                "epoch": epoch,
                "train_loss": train_stats["train_loss"],
                "train_loss_std": train_stats["train_loss_std"],
                "dev_loss": dev_metrics["loss"],
                "dev_cer": dev_metrics["cer"],
                "dev_wer": dev_metrics["wer"],
                "epoch_time_sec": epoch_time,
                "learning_rate_end": scheduler.get_last_lr()[0],
                "grad_norm_mean": train_stats["grad_norm_mean"],
                "tokens_per_sec": train_stats["tokens_per_sec"],
                "peak_mem_gb": peak_mem,
                "last_step": global_step,
            }
            metric_logger.log_epoch(row)

            logger.info(
                f"Epoch {epoch} | Train: {train_stats['train_loss']:.4f} | "
                f"Dev CER: {dev_metrics['cer']:.4f} | "
                f"Dev WER: {dev_metrics['wer']:.4f} | "
                f"Time: {epoch_time:.0f}s | Peak: {peak_mem:.1f} GB"
            )

            # Checkpointing
            is_best = dev_metrics["cer"] < best_cer
            if is_best:
                best_cer = dev_metrics["cer"]
                patience_counter = 0
                logger.info(f"  New best CER: {best_cer:.4f}")
            else:
                patience_counter += 1

            save_checkpoint(
                run_dir,
                epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler,
                best_cer=best_cer, patience_counter=patience_counter,
                config=cfg, git_sha=git_sha, is_best=is_best,
                total_epochs=cfg.num_epochs,
            )

            # Regenerate in-run plots (cheap: reads history.csv + metrics.jsonl)
            render_run_plots(run_dir, title_prefix=f"{args.backbone} — ")

            if patience_counter >= cfg.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    finally:
        metric_logger.close()

    # ── Final evaluation on best checkpoint ────────────────────────────────
    best_path = run_dir / "best_model.pt"
    load_checkpoint(best_path, model=model, map_location=device, restore_rng=False)
    test_metrics = evaluate(model, test_loader, vocab, device, tag="test")

    chunked_results = {}
    for chunk_sec in cfg.chunk_sizes_sec:
        for carry in [False, True]:
            max_utt = (
                cfg.max_carry_eval_utterances if carry
                else cfg.max_reset_eval_utterances
            )
            result = evaluate_chunked(
                model, dev_ds, vocab, chunk_sec, cfg, device,
                carry_state=carry,
                batch_size=cfg.chunked_eval_batch_size,
                max_utterances=max_utt,
            )
            if result is not None:
                mode = "carry" if carry else "reset"
                chunked_results[f"{chunk_sec}s_{mode}"] = result

    # Read history.csv back into results.json for a single-artifact summary
    import pandas as pd
    history_csv = run_dir / "history.csv"
    history_rows = pd.read_csv(history_csv).to_dict(orient="records") if history_csv.exists() else []

    final_results = {
        "backbone": args.backbone,
        "params": pc,
        "best_dev_cer": best_cer,
        "test": test_metrics,
        "chunked": chunked_results,
        "history": history_rows,
        "git_sha": git_sha,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config_snapshot": asdict(cfg),
        "cli_args": sys.argv[1:],
        "dataset": cfg.dataset,
    }

    # CV-specific provenance + per-speaker WER (option A: kept inside results.json
    # rather than spinning up a §13-style eval_full.json — see CV pilot decision).
    if cfg.dataset == "common_voice_en_100h":
        from src.data.common_voice import CV_25_RELEASE_DIR
        manifest_path = Path(cfg.data_cache_dir) / TRAIN_MANIFEST_NAME
        stats_path = Path(cfg.data_cache_dir) / "filter_stats.json"
        cv_meta = {
            "dataset_version": "common_voice_en_25.0",
            "release_dir": CV_25_RELEASE_DIR,
        }
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            cv_meta["dataset_subset_sha256"] = stats.get("manifest_sha256")
            cv_meta["filter_stats"] = stats
        if manifest_path.exists():
            cv_meta["manifest_path"] = str(manifest_path)
        final_results["cv_meta"] = cv_meta

        logger.info("Computing wer_by_speaker on CV test split (single-utterance pass)...")
        final_results["wer_by_speaker"] = _compute_wer_by_speaker(
            model, test_entries, vocab, cfg, device,
        )
        logger.info(
            f"  wer_by_speaker covers {len(final_results['wer_by_speaker'])} client_ids"
        )

    with open(run_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    # Final plot regeneration (covers the last epoch)
    render_run_plots(run_dir, title_prefix=f"{args.backbone} — ")

    logger.info(f"Done. Best dev CER: {best_cer:.4f} | Test CER: {test_metrics['cer']:.4f}")


if __name__ == "__main__":
    main()
