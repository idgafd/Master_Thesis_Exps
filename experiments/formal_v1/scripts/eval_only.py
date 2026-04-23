#!/usr/bin/env python3
"""Eval-only rescue: load best_model.pt from a completed run-dir, run
test + chunked eval, write results.json.

Use this when training finished but the post-training eval failed
(e.g., numerical issue in LUCID solve at eval batch).  Training state
and checkpoints remain untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data.vocab import CharVocab
from src.data.dataset import ASRDataset, DurationBatchSampler, collate_fn
from src.data.librispeech import load_librispeech
from src.models.asr_model import ASRModel
from src.training.evaluate import evaluate, evaluate_chunked
from src.training.checkpoint import load_checkpoint, get_git_sha
from src.utils.misc import seed_everything, count_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = ExperimentConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.output_dir = str(run_dir)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(cfg.seed)

    vocab = CharVocab.build_english()
    dev_entries = load_librispeech(
        "dev", cache_dir=cfg.data_cache_dir,
        min_audio_sec=cfg.min_audio_sec, max_audio_sec=cfg.max_audio_sec,
    )
    test_entries = load_librispeech(
        "test", cache_dir=cfg.data_cache_dir,
        min_audio_sec=cfg.min_audio_sec, max_audio_sec=cfg.max_audio_sec,
    )
    dev_ds = ASRDataset(dev_entries, vocab, cfg)
    test_ds = ASRDataset(test_entries, vocab, cfg)
    test_loader = DataLoader(
        test_ds,
        batch_sampler=DurationBatchSampler(
            test_entries, cfg.batch_max_seconds, shuffle=False, seed=cfg.seed,
        ),
        collate_fn=collate_fn, num_workers=2,
    )

    model = ASRModel(vocab_size=vocab.size, cfg=cfg).to(device)
    best_path = run_dir / "best_model.pt"
    state = load_checkpoint(
        best_path, model=model, map_location=device, restore_rng=False,
    )
    model.eval()

    print(f"Loaded best_model.pt from epoch {state.get('epoch', '?')} "
          f"(best_dev_cer {state.get('best_cer', '?'):.4f})")

    test_metrics = evaluate(model, test_loader, vocab, device, tag="test")
    print(f"test CER {test_metrics['cer']:.4f}  WER {test_metrics['wer']:.4f}")

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
                print(f"  chunked {chunk_sec}s {mode}: CER {result['cer']:.4f}")

    pc = count_parameters(model)

    import pandas as pd
    history_csv = run_dir / "history.csv"
    history_rows = (
        pd.read_csv(history_csv).to_dict(orient="records")
        if history_csv.exists() else []
    )

    final_results = {
        "backbone": cfg.backbone,
        "params": pc,
        "best_dev_cer": state.get("best_cer", None),
        "test": test_metrics,
        "chunked": chunked_results,
        "history": history_rows,
        "git_sha": get_git_sha(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "config_snapshot": asdict(cfg),
        "cli_args": sys.argv[1:],
        "_note": "eval_only rerun after training-time eval failure",
    }
    out_path = run_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
