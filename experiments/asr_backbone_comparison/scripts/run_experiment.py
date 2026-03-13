#!/usr/bin/env python3
"""
ASR Encoder Backbone Comparison — experiment runner.

Usage:
    uv run scripts/run_experiment.py
    uv run scripts/run_experiment.py --config configs/default.yaml
    uv run scripts/run_experiment.py --backbone transformer mamba
    uv run scripts/run_experiment.py --resume           # skip completed backbones
    uv run scripts/run_experiment.py --eval-only        # load saved models and eval

The runner is fully restartable:
  - Completed backbone results are stored in <output_dir>/results.json.
  - On restart, any backbone already present in results.json is skipped
    (unless --force is passed).
  - Checkpoints are saved per backbone as best_<backbone>.pt.
  - The vocabulary is saved once as vocab.json and reloaded on restart.
"""

import argparse
import dataclasses
import datetime
import json
import os
import platform
import sys
import time
from pathlib import Path

import yaml

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asr_exp.config import ExperimentConfig, load_config
from asr_exp.data import (
    ASRDataset,
    CharVocab,
    DurationBatchSampler,
    SpecAugment,
    collate_fn,
    load_and_prepare_data,
)
from asr_exp.models import ASRModel
from asr_exp.training import (
    WarmupScheduler,
    WSDScheduler,
    evaluate,
    evaluate_chunked,
    train_one_epoch,
)
from asr_exp.utils.misc import count_parameters, count_parameters_by_module, make_serializable, set_seed
from asr_exp.utils.plots import plot_all


# ── Helper functions ─────────────────────────────────────────────────────────

def _save_sample_predictions(refs: list, hyps: list, path: str, n: int = 50) -> None:
    """Write the first n ref/hyp pairs to a plain-text file."""
    with open(path, "w", encoding="utf-8") as f:
        for i, (ref, hyp) in enumerate(zip(refs[:n], hyps[:n])):
            f.write(f"[{i+1:03d}]\n")
            f.write(f"REF: {ref}\n")
            f.write(f"HYP: {hyp}\n\n")
    print(f"  Sample predictions saved to {path}")


def _make_run_id(output_dir: str) -> str:
    """Generate a human-readable run ID: <dir_name>_<YYYYMMDD>_<HHMM>."""
    dir_name = Path(output_dir).name
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    return f"{dir_name}_{ts}"


def _save_run_info(output_dir: str, device: torch.device, run_id: str) -> None:
    """Save hardware, software environment, and run_id to run_info.json."""
    info = {
        "run_id": run_id,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
    path = os.path.join(output_dir, "run_info.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Run info saved to {path}  [run_id={run_id}]")


def _save_config_snapshot(cfg, output_dir: str) -> None:
    """Save a snapshot of the fully-resolved config to <output_dir>/config_snapshot.yaml.

    Captures any CLI overrides — the source YAML alone may not reflect them.
    """
    snapshot_path = os.path.join(output_dir, "config_snapshot.yaml")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.dump(dataclasses.asdict(cfg), f, allow_unicode=True, sort_keys=True)
    print(f"Config snapshot saved to {snapshot_path}")


# ── Per-backbone experiment ───────────────────────────────────────────────────

def run_backbone(
    backbone_type: str,
    train_entries,
    dev_entries,
    test_entries,
    vocab: CharVocab,
    cfg: ExperimentConfig,
    device: torch.device,
    resume: bool = True,
) -> dict:
    """Train and evaluate a single backbone.

    If resume=True and a checkpoint already exists for this backbone,
    training is skipped and only evaluation is re-run.

    Returns a result dict suitable for serialization.
    """
    print(f"\n{'=' * 40}")
    print(f"  BACKBONE: {backbone_type.upper()}")
    print(f"{'=' * 40}\n")

    checkpoint_path = os.path.join(cfg.output_dir, f"best_{backbone_type}.pt")
    history_path = os.path.join(cfg.output_dir, f"history_{backbone_type}.json")

    # ── Datasets / DataLoaders ────────────────────────────────────────────────
    train_dataset = ASRDataset(train_entries, vocab, cfg)
    dev_dataset = ASRDataset(dev_entries, vocab, cfg)
    test_dataset = ASRDataset(test_entries, vocab, cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=DurationBatchSampler(
            train_entries, cfg.batch_max_seconds, shuffle=True, seed=cfg.seed
        ),
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ASRModel(backbone_type, vocab.size, cfg).to(device)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # ── Training ──────────────────────────────────────────────────────────────
    history = {
        "epoch": [], "train_loss": [], "dev_loss": [],
        "dev_cer": [], "dev_wer": [], "lr": [], "epoch_time_s": [],
    }
    best_dev_cer = float("inf")
    best_epoch = 0
    start_epoch = 1

    if resume and os.path.exists(checkpoint_path):
        print(f"Checkpoint found at {checkpoint_path} — loading for evaluation only.")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            best_dev_cer = min(history["dev_cer"]) if history["dev_cer"] else float("inf")
            best_epoch = (
                history["epoch"][history["dev_cer"].index(best_dev_cer)]
                if history["dev_cer"]
                else 0
            )
        start_epoch = cfg.num_epochs + 1  # skip training loop
    else:
        total_steps = len(train_loader) * cfg.num_epochs
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        if cfg.scheduler_type == "wsd":
            decay_steps = cfg.wsd_decay_epochs * len(train_loader)
            scheduler = WSDScheduler(
                optimizer,
                total_steps=total_steps,
                warmup_steps=cfg.warmup_steps,
                decay_steps=decay_steps,
            )
            print(
                f"Scheduler: WSD | warmup={cfg.warmup_steps} steps | "
                f"stable={total_steps - cfg.warmup_steps - decay_steps} steps | "
                f"decay={decay_steps} steps ({cfg.wsd_decay_epochs} epochs)"
            )
        else:
            base_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
            scheduler = WarmupScheduler(optimizer, base_scheduler, cfg.warmup_steps)
            print(f"Scheduler: cosine | warmup={cfg.warmup_steps} steps")

        spec_aug = (
            SpecAugment(
                cfg.freq_mask_param,
                cfg.time_mask_param,
                cfg.num_freq_masks,
                cfg.num_time_masks,
            )
            if cfg.spec_augment
            else None
        )

        no_improve = 0
        for epoch in range(start_epoch, cfg.num_epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, spec_aug, cfg, epoch, device
            )
            dev_metrics = evaluate(model, dev_loader, vocab, device, tag="dev")
            elapsed = time.time() - t0

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["dev_loss"].append(dev_metrics["loss"])
            history["dev_cer"].append(dev_metrics["cer"])
            history["dev_wer"].append(dev_metrics["wer"])
            history["lr"].append(scheduler.get_last_lr()[0])
            history["epoch_time_s"].append(elapsed)

            print(
                f"Epoch {epoch}/{cfg.num_epochs} | Train loss: {train_loss:.4f} | "
                f"Dev CER: {dev_metrics['cer']:.4f} | Dev loss: {dev_metrics['loss']:.4f} | "
                f"Time: {elapsed:.0f}s"
            )
            # Log learned theta per layer if backbone supports it (Config D)
            if hasattr(model.encoder, "get_theta_info"):
                theta_info = model.encoder.get_theta_info()
                if theta_info:
                    print(f"  {theta_info}")

            if dev_metrics["cer"] < best_dev_cer:
                best_dev_cer = dev_metrics["cer"]
                best_epoch = epoch
                no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  New best! Saved to {checkpoint_path}")
            else:
                no_improve += 1
                if no_improve >= cfg.early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                    break

            # Save history after every epoch so we can inspect / restart mid-run
            with open(history_path, "w") as f:
                json.dump(make_serializable(history), f, indent=2)

        # Load best checkpoint for final evaluation
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\n--- Final Test Evaluation ({backbone_type}) ---")
    test_metrics = evaluate(
        model, test_loader, vocab, device, tag="test", return_predictions=True
    )
    # Save sample predictions (first 50) to a text file
    _save_sample_predictions(
        refs=test_metrics.pop("refs"),
        hyps=test_metrics.pop("hyps"),
        path=os.path.join(cfg.output_dir, f"samples_{backbone_type}.txt"),
        n=50,
    )

    print(f"\n--- Chunked Inference: RESET-STATE ({backbone_type}) ---")
    chunked_reset = {}
    for chunk_sec in cfg.chunk_sizes_sec:
        print(f"  Chunk {chunk_sec}s (reset) ...")
        m = evaluate_chunked(model, test_dataset, vocab, chunk_sec, cfg, device, carry_state=False)
        chunked_reset[f"{chunk_sec}s"] = m
        print(f"    CER: {m['cer']:.4f} | WER: {m['wer']:.4f}")

    chunked_carry = {}
    if model.supports_carry_state:
        print(f"\n--- Chunked Inference: CARRY-STATE ({backbone_type}) ---")
        for chunk_sec in cfg.chunk_sizes_sec:
            print(f"  Chunk {chunk_sec}s (carry) ...")
            m = evaluate_chunked(
                model, test_dataset, vocab, chunk_sec, cfg, device, carry_state=True
            )
            if m is not None:
                chunked_carry[f"{chunk_sec}s"] = m
                print(f"    CER: {m['cer']:.4f} | WER: {m['wer']:.4f} (n={m['n_evaluated']})")
    else:
        print(f"\n  Carry-state: N/A for {backbone_type} (stateless architecture)")

    training_time_s = sum(history["epoch_time_s"])
    return {
        "backbone": backbone_type,
        "params": count_parameters_by_module(model),  # {frontend, encoder, ctc_head, total}
        "n_params": n_params,                          # kept for backwards compat
        "best_epoch": best_epoch,
        "best_dev_cer": float(best_dev_cer),
        "training_wall_time_s": training_time_s,
        "test": test_metrics,
        "chunked_reset": chunked_reset,
        "chunked_carry": chunked_carry,
        "history": history,
    }


# ── Results I/O ───────────────────────────────────────────────────────────────

def _load_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_results(results: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"Results saved to {path}")


def _build_tables(all_results: dict, backbones: list, chunk_sizes_sec: list) -> str:
    """Build the results summary as a plain-text string."""
    present = [bb for bb in backbones if bb in all_results]
    lines = []
    lines.append("─" * 30)
    lines.append("  RESULTS SUMMARY")
    lines.append("─" * 30)

    # Table 1: full utterance metrics (loss + CER + WER)
    lines.append("\n  Table 1: Full utterance metrics")
    header1 = (
        f"{'Backbone':<26} {'Params':>8} {'BestDev':>8}"
        f" {'TestLoss':>9} {'TestCER':>8} {'TestWER':>8}"
    )
    lines.append(header1)
    lines.append("-" * len(header1))
    for bb in present:
        r = all_results[bb]
        row = (
            f"{bb:<26} {r['n_params'] / 1e6:>7.2f}M"
            f" {r['best_dev_cer']:>8.4f}"
            f" {r['test']['loss']:>9.4f}"
            f" {r['test']['cer']:>8.4f}"
            f" {r['test']['wer']:>8.4f}"
        )
        lines.append(row)

    # Table 2: reset-state chunked CER + WER
    lines.append("\n  Table 2: Reset-state chunked evaluation")
    header2 = f"{'Backbone':<26}"
    for cs in chunk_sizes_sec:
        lbl = f"{cs:g}s"
        header2 += f" {'R@'+lbl+'_CER':>11} {'R@'+lbl+'_WER':>11}"
    lines.append(header2)
    lines.append("-" * len(header2))
    for bb in present:
        r = all_results[bb]
        row = f"{bb:<26}"
        for cs in chunk_sizes_sec:
            key = f"{cs}s"
            cer = r["chunked_reset"].get(key, {}).get("cer", float("nan"))
            wer = r["chunked_reset"].get(key, {}).get("wer", float("nan"))
            row += f" {cer:>11.4f} {wer:>11.4f}"
        lines.append(row)

    # Table 3: carry-state chunks (if any)
    has_carry = any(all_results[bb]["chunked_carry"] for bb in present)
    if has_carry:
        lines.append("\n  Table 3: Carry-state chunked evaluation (recurrent/SSM backbones)")
        header3 = f"{'Backbone':<26}"
        for cs in chunk_sizes_sec:
            lbl = f"{cs:g}s"
            header3 += f" {'C@'+lbl+'_CER':>11} {'C@'+lbl+'_WER':>11}"
        lines.append(header3)
        lines.append("-" * len(header3))
        for bb in present:
            r = all_results[bb]
            if not r["chunked_carry"]:
                lines.append(f"{bb:<26}  (stateless — no carry)")
            else:
                row = f"{bb:<26}"
                for cs in chunk_sizes_sec:
                    key = f"{cs}s"
                    cer = r["chunked_carry"].get(key, {}).get("cer", float("nan"))
                    wer = r["chunked_carry"].get(key, {}).get("wer", float("nan"))
                    row += f" {cer:>11.4f} {wer:>11.4f}"
                lines.append(row)

        lines.append("\n  Table 4: CER improvement from carry-state (Reset − Carry)")
        header4 = f"{'Backbone':<26}"
        for cs in chunk_sizes_sec:
            lbl = f"{cs:g}s"
            header4 += f" {'Δ@'+lbl:>9}"
        lines.append(header4)
        lines.append("-" * len(header4))
        for bb in present:
            r = all_results[bb]
            if not r["chunked_carry"]:
                lines.append(f"{bb:<26}  N/A (stateless)")
                continue
            row = f"{bb:<26}"
            for cs in chunk_sizes_sec:
                key = f"{cs}s"
                if key in r["chunked_carry"] and key in r["chunked_reset"]:
                    delta = r["chunked_reset"][key]["cer"] - r["chunked_carry"][key]["cer"]
                    sign = "+" if delta > 0 else ""
                    row += f" {sign}{delta:>8.4f}"
                else:
                    row += f" {'N/A':>9}"
            lines.append(row)

    return "\n".join(lines)


def _print_tables(
    all_results: dict, backbones: list, chunk_sizes_sec: list, output_dir: str
) -> None:
    text = _build_tables(all_results, backbones, chunk_sizes_sec)
    print("\n" + text)
    tables_path = os.path.join(output_dir, "results_tables.txt")
    with open(tables_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"Tables saved to {tables_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ASR backbone comparison experiment")
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--backbone", nargs="+", metavar="BACKBONE",
        help="Override backbone list (e.g. --backbone transformer mamba)"
    )
    parser.add_argument(
        "--output-dir", metavar="DIR",
        help="Override output directory"
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip backbones already in results.json (default: True)"
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Re-run all backbones from scratch"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even completed backbones (overwrite results)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, load saved checkpoints and run evaluation only"
    )
    parser.add_argument(
        "--seed", type=int, help="Override random seed"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides["seed"] = args.seed

    cfg = load_config(args.config, overrides)
    if args.backbone:
        cfg.backbones = args.backbone

    os.makedirs(cfg.output_dir, exist_ok=True)
    _save_config_snapshot(cfg, cfg.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    run_id = _make_run_id(cfg.output_dir)
    _save_run_info(cfg.output_dir, device, run_id)
    set_seed(cfg.seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    vocab_path = os.path.join(cfg.output_dir, "vocab.json")
    if os.path.exists(vocab_path):
        print(f"Loading existing vocab from {vocab_path}")
        vocab = CharVocab.load(vocab_path)
        # We still need entries for dataset creation — parse without rebuilding vocab
        train_entries, dev_entries, test_entries, _ = load_and_prepare_data(cfg)
    else:
        train_entries, dev_entries, test_entries, vocab = load_and_prepare_data(cfg)
        vocab.save(vocab_path)
        print(f"Vocab saved to {vocab_path}")

    # ── Per-backbone runs ─────────────────────────────────────────────────────
    results_path = os.path.join(cfg.output_dir, "results.json")
    all_results = _load_results(results_path)

    for bb in cfg.backbones:
        if not args.force and bb in all_results:
            print(f"\n[skip] {bb} already in results.json — use --force to re-run")
            continue

        set_seed(cfg.seed)
        try:
            result = run_backbone(
                backbone_type=bb,
                train_entries=train_entries,
                dev_entries=dev_entries,
                test_entries=test_entries,
                vocab=vocab,
                cfg=cfg,
                device=device,
                resume=args.resume or args.eval_only,
            )
        except Exception as exc:
            print(f"\n!!! Backbone {bb} failed: {exc}")
            import traceback
            traceback.print_exc()
            # Save partial results and continue with next backbone
            all_results[bb] = {"backbone": bb, "error": str(exc)}
            _save_results(all_results, results_path)
            continue

        all_results[bb] = result
        _save_results(all_results, results_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_tables(all_results, cfg.backbones, cfg.chunk_sizes_sec, cfg.output_dir)

    completed = [bb for bb in cfg.backbones if bb in all_results and "error" not in all_results[bb]]
    if completed:
        plot_all(all_results, completed, cfg.chunk_sizes_sec, cfg.output_dir)


if __name__ == "__main__":
    main()
