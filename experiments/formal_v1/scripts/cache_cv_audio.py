#!/usr/bin/env python3
"""One-time pre-decode of CV needed clips into a memory-mappable cache.

Reads the 96,998 needed clip filenames (100h manifest train + full dev +
full test), decodes each mp3 at 48 kHz to 16 kHz mono float32 via
`torchaudio.functional.resample(... lowpass_filter_width=64)` (per task
brief §2.6), and writes:

    data/common_voice/audio_cache_seed42.f32       — concatenated raw f32 samples
    data/common_voice/audio_cache_seed42.idx.json  — {clip_id: [offset, n_samples]}

Why: when 4 pilot runs each pre-decode 96k clips into separate numpy
arrays in their own process memory, they hammer DRAM bandwidth and CPU
contend each other to a crawl. Memory-mapping a single shared cache file
lets all 4 processes share OS page cache → one ~35 GB physical allocation
instead of 4 × 30 GB, and zero decode work at training startup.

Idempotent: if cache files exist and the index already covers every clip
in the needed set, exits silently. Re-run is a no-op.

Run from `experiments/formal_v1/`:

    uv run python scripts/cache_cv_audio.py

Decoding is parallelised across a thread pool — torchaudio's mp3 decode
and `functional.resample` both release the GIL, so threads scale well.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

CACHE_DIR = Path("./data/common_voice")
CV_25_RELEASE_DIR = "cv-corpus-25.0-2026-03-09"
CV_LOCALE = "en"
CACHE_F32 = "audio_cache_seed42.f32"
CACHE_IDX = "audio_cache_seed42.idx.json"

DEFAULT_THREADS = max(os.cpu_count() // 2, 8)


def collect_needed() -> List[str]:
    """All clip filenames the pilot needs: manifest train + dev + test (deduped, sorted)."""
    needed = set()
    with open(CACHE_DIR / "100h_train_manifest_seed42.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            needed.add(r["clip_id"])
    cv_root = CACHE_DIR / "raw" / CV_25_RELEASE_DIR / CV_LOCALE
    for split in ("dev", "test"):
        with open(cv_root / f"{split}.tsv", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                needed.add(r["path"])
    return sorted(needed)


def _decode_one(clip_path: Path) -> Optional[np.ndarray]:
    try:
        wav, sr = torchaudio.load(str(clip_path))
    except Exception as e:
        logging.warning(f"decode fail {clip_path.name}: {e}")
        return None
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(
            wav.float(), sr, 16000, lowpass_filter_width=64
        )
    return wav.squeeze(0).contiguous().numpy().astype(np.float32, copy=False)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    cv_root = CACHE_DIR / "raw" / CV_25_RELEASE_DIR / CV_LOCALE
    clips_dir = cv_root / "clips"
    if not clips_dir.exists():
        logger.error(f"clips dir not found: {clips_dir}")
        return 2

    cache_f32 = CACHE_DIR / CACHE_F32
    cache_idx = CACHE_DIR / CACHE_IDX

    needed = collect_needed()
    logger.info(f"needed {len(needed)} clips")

    if cache_f32.exists() and cache_idx.exists():
        with open(cache_idx) as f:
            idx_existing = json.load(f)
        missing = [c for c in needed if c not in idx_existing]
        if not missing:
            logger.info(f"cache already covers all {len(needed)} clips; nothing to do")
            return 0
        logger.info(f"cache exists but missing {len(missing)} clips; rebuilding from scratch")

    n_threads = int(os.environ.get("CV_CACHE_THREADS", DEFAULT_THREADS))
    logger.info(f"decoding with {n_threads} threads")

    # Decode in chunks so we can stream-write to disk and bound peak memory.
    chunk_size = 1024
    started = time.time()
    idx: Dict[str, List[int]] = {}
    offset = 0
    failed = 0

    with open(cache_f32, "wb", buffering=4 << 20) as out:
        for chunk_start in range(0, len(needed), chunk_size):
            chunk = needed[chunk_start:chunk_start + chunk_size]
            results: Dict[str, Optional[np.ndarray]] = {}
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                future_to_clip = {
                    pool.submit(_decode_one, clips_dir / c): c for c in chunk
                }
                for fut in as_completed(future_to_clip):
                    c = future_to_clip[fut]
                    results[c] = fut.result()
            # Preserve sorted order on disk so the index is sequential.
            for c in chunk:
                arr = results.get(c)
                if arr is None:
                    failed += 1
                    continue
                arr.tofile(out)
                n = int(arr.shape[0])
                idx[c] = [offset, n]
                offset += n
            elapsed = time.time() - started
            done = chunk_start + len(chunk)
            rate = done / max(elapsed, 1)
            eta = (len(needed) - done) / max(rate, 1)
            logger.info(
                f"  {done}/{len(needed)} clips ({rate:.0f}/s; ETA {eta/60:.1f} min); "
                f"size={offset*4/1e9:.2f} GB; failed={failed}"
            )

    with open(cache_idx, "w") as f:
        json.dump(idx, f)
    total_gb = offset * 4 / 1e9
    elapsed = time.time() - started
    logger.info(
        f"DONE. cached {len(idx)} clips ({failed} failed) in {total_gb:.1f} GB; "
        f"wall {elapsed/60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
