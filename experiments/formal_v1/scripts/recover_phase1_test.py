#!/usr/bin/env python3
"""Recover test-set CER/WER for Phase-1 P²-RSE runs.

Both Phase 1 runs (stage5_01, stage5_02) completed 30 epochs of training
and saved `best_model.pt` but crashed during the auxiliary chunked
carry-state evaluation (pre-fix `init_state` shape bug on multi-pole
state). The standard full-utterance test-split CER was computed but
never persisted to results.json because the crash happened before the
final artifact write.

This script loads each best_model.pt, runs the standard test-split
evaluation, and writes a single `test_metrics.json` per run directory.

Usage:
    uv run python scripts/recover_phase1_test.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.vocab import CharVocab
from src.data.dataset import ASRDataset, DurationBatchSampler, collate_fn
from src.data.librispeech import load_librispeech
from src.models.asr_model import ASRModel
from src.training.evaluate import evaluate
from src.training.checkpoint import load_checkpoint
from src.utils.misc import seed_everything


RUNS = [
    ("stage5_01", "rwkv6_p2rse"),
    ("stage5_02", "rwkv6_p2rse_softmax"),
]


def recover_one(tag: str, backbone: str, gpu: int = 0) -> dict:
    run_dir = Path(f"outputs/{tag}_{backbone}_seed42")
    ckpt = run_dir / "best_model.pt"
    assert ckpt.exists(), f"no best_model.pt at {ckpt}"

    cfg = load_config("configs/default.yaml", {"backbone": backbone, "seed": 42})
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    seed_everything(cfg.seed)

    vocab = CharVocab.build_english()
    test_entries = load_librispeech("test", cfg.data_cache_dir, cfg.min_audio_sec, cfg.max_audio_sec)
    dev_entries  = load_librispeech("dev",  cfg.data_cache_dir, cfg.min_audio_sec, cfg.max_audio_sec)

    test_ds = ASRDataset(test_entries, vocab, cfg)
    dev_ds  = ASRDataset(dev_entries,  vocab, cfg)

    test_loader = DataLoader(
        test_ds,
        batch_sampler=DurationBatchSampler(test_entries, cfg.batch_max_seconds, False, cfg.seed),
        collate_fn=collate_fn, num_workers=4,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_sampler=DurationBatchSampler(dev_entries, cfg.batch_max_seconds, False, cfg.seed),
        collate_fn=collate_fn, num_workers=4,
    )

    model = ASRModel(vocab_size=vocab.size, cfg=cfg).to(device)
    load_checkpoint(ckpt, model=model, map_location=device, restore_rng=False)

    print(f"[{tag}] running dev-split re-eval (sanity)...", flush=True)
    dev_m = evaluate(model, dev_loader, vocab, device, tag="dev")
    print(f"[{tag}] dev: CER={dev_m['cer']:.4f}  WER={dev_m['wer']:.4f}", flush=True)

    print(f"[{tag}] running test-split eval...", flush=True)
    test_m = evaluate(model, test_loader, vocab, device, tag="test")
    print(f"[{tag}] test: CER={test_m['cer']:.4f}  WER={test_m['wer']:.4f}", flush=True)

    out = {
        "backbone": backbone,
        "run_tag": tag,
        "dev": {"cer": dev_m["cer"], "wer": dev_m["wer"], "loss": dev_m["loss"]},
        "test": {"cer": test_m["cer"], "wer": test_m["wer"], "loss": test_m["loss"]},
    }
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{tag}] wrote {run_dir / 'test_metrics.json'}", flush=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    results = {}
    for tag, backbone in RUNS:
        try:
            results[tag] = recover_one(tag, backbone, gpu=args.gpu)
        except Exception as e:
            print(f"[{tag}] FAILED: {e!r}", flush=True)
            results[tag] = {"error": repr(e)}

    print("\n=== SUMMARY ===")
    for tag, r in results.items():
        if "error" in r:
            print(f"{tag}: ERROR — {r['error']}")
        else:
            print(
                f"{tag:10s} backbone={r['backbone']:25s} "
                f"dev CER={r['dev']['cer']:.4f}  test CER={r['test']['cer']:.4f}  "
                f"test WER={r['test']['wer']:.4f}"
            )


if __name__ == "__main__":
    main()
