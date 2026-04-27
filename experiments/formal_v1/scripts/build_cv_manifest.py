#!/usr/bin/env python3
"""Build the deterministic Common Voice EN 100h training manifest.

Per task brief §2.2:

1. Load CV's `train.tsv` (validated-derived, the official one).
2. Apply audio-quality filter (§2.4): up_votes >= 2, down_votes == 0,
   duration >= 1.0 s, non-empty sentence (after §2.5 normalisation).
3. Sort by (sha256("seed42:" + client_id), clip_id) for stable
   speaker-grouped order independent of any HuggingFace shuffle.
4. Walk the sorted list, accumulating durations; stop at the first
   index where cumulative duration >= 100.0 hours. Take the prefix.
5. Persist (client_id, clip_id, duration_seconds, sentence_normalised)
   to data/common_voice/100h_train_manifest_seed42.csv.
6. Compute the manifest sha256, write filter_stats.json with rejection
   counts and the post-normalisation character coverage stat (must be
   >= 90% per §2.5).

Speaker-disjointness (§2.3) is verified post-construction: train manifest
client_ids must be disjoint from dev.tsv and test.tsv client_ids. If
overlap exists, the script aborts and writes
`data/common_voice/speaker_overlap_diagnosis.txt`.

Idempotent: a second run will recompute the manifest in memory and
overwrite the CSV byte-for-byte (deterministic). The filter_stats.json
is rewritten too. The recorded sha256 is the canonical pin.

Run from `experiments/formal_v1/`:

    uv run scripts/build_cv_manifest.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Allow standalone execution: add formal_v1/ to PYTHONPATH so `from src...` works.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.common_voice import (
    CV_25_RELEASE_DIR,
    CV_LOCALE,
    TRAIN_MANIFEST_NAME,
    cv_normalize_sentence,
    _read_clip_durations,
    _read_tsv_rows,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path("./data/common_voice")
TARGET_HOURS = 100.0
HASH_SALT = "seed42:"

FILTER_STATS_NAME = "filter_stats.json"
SPEAKER_OVERLAP_DIAGNOSIS_NAME = "speaker_overlap_diagnosis.txt"


def _client_id_hash(client_id: str) -> str:
    return hashlib.sha256((HASH_SALT + client_id).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def filter_train_rows(
    rows: List[Dict[str, str]],
    durations: Dict[str, float],
) -> Tuple[List[Tuple[str, str, float, str]], Dict[str, int]]:
    """Apply §2.4 + §2.5 filter. Returns (kept_tuples, rejection_counts).

    kept tuples are (client_id, clip_id, duration_sec, sentence_normalised).
    """
    rejected = {
        "missing_duration": 0,
        "duration_lt_1s": 0,
        "low_up_votes": 0,
        "any_down_votes": 0,
        "non_int_votes": 0,
        "empty_sentence_raw": 0,
        "empty_sentence_after_norm": 0,
    }
    kept: List[Tuple[str, str, float, str]] = []
    for r in rows:
        clip = r["path"]
        dur = durations.get(clip)
        if dur is None:
            rejected["missing_duration"] += 1
            continue
        if dur < 1.0:
            rejected["duration_lt_1s"] += 1
            continue
        try:
            uv = int(r.get("up_votes", "0"))
            dv = int(r.get("down_votes", "0"))
        except ValueError:
            rejected["non_int_votes"] += 1
            continue
        if uv < 2:
            rejected["low_up_votes"] += 1
            continue
        if dv >= 1:
            rejected["any_down_votes"] += 1
            continue
        sentence_raw = r.get("sentence", "")
        if not sentence_raw.strip():
            rejected["empty_sentence_raw"] += 1
            continue
        sentence = cv_normalize_sentence(sentence_raw)
        if not sentence:
            rejected["empty_sentence_after_norm"] += 1
            continue
        kept.append((r["client_id"], clip, float(dur), sentence))
    return kept, rejected


def select_100h_prefix(
    kept: List[Tuple[str, str, float, str]],
    target_hours: float = TARGET_HOURS,
) -> List[Tuple[str, str, float, str]]:
    """Sort by (client_id_hash, clip_id), accumulate to >= target_hours."""
    keyed = [(_client_id_hash(c[0]), c[1], c[0], c[2], c[3]) for c in kept]
    keyed.sort(key=lambda t: (t[0], t[1]))
    target_seconds = target_hours * 3600.0
    out: List[Tuple[str, str, float, str]] = []
    cum = 0.0
    for _h, clip, client_id, dur, sentence in keyed:
        out.append((client_id, clip, dur, sentence))
        cum += dur
        if cum >= target_seconds:
            break
    return out


def compute_char_coverage(rows: List[Dict[str, str]], sample_n: int = 50_000) -> float:
    """§2.5 coverage = (non-space chars surviving normalisation) / (non-space chars before)."""
    rng = random.Random(42)
    sample = rng.sample(rows, k=min(sample_n, len(rows)))
    before = 0
    after = 0
    for r in sample:
        s = r.get("sentence", "")
        before += sum(1 for c in s if not c.isspace())
        n = cv_normalize_sentence(s)
        after += sum(1 for c in n if not c.isspace())
    return (after / before) if before > 0 else 1.0


def write_manifest(rows: List[Tuple[str, str, float, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["client_id", "clip_id", "duration_seconds", "sentence_normalised"])
        for client_id, clip, dur, sent in rows:
            w.writerow([client_id, clip, f"{dur:.3f}", sent])


def check_speaker_disjointness(
    train_client_ids: set,
    cv_root: Path,
    cache_dir: Path,
) -> bool:
    """§2.3: train ⊥ dev ⊥ test on client_id."""
    dev_ids = {r["client_id"] for r in _read_tsv_rows(cv_root / "dev.tsv")}
    test_ids = {r["client_id"] for r in _read_tsv_rows(cv_root / "test.tsv")}

    overlaps: List[Tuple[str, str, int]] = []
    if train_client_ids & dev_ids:
        overlaps.append(("train", "dev", len(train_client_ids & dev_ids)))
    if train_client_ids & test_ids:
        overlaps.append(("train", "test", len(train_client_ids & test_ids)))
    if dev_ids & test_ids:
        overlaps.append(("dev", "test", len(dev_ids & test_ids)))

    if not overlaps:
        return True

    diag = cache_dir / SPEAKER_OVERLAP_DIAGNOSIS_NAME
    diag.parent.mkdir(parents=True, exist_ok=True)
    with open(diag, "w", encoding="utf-8") as f:
        f.write("Common Voice EN 25.0 — speaker-overlap diagnosis (§2.3)\n")
        f.write(f"date: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")
        for a, b, n in overlaps:
            f.write(f"{a} ∩ {b}: {n} client_ids\n")
        f.write("\nManifest construction ABORTED. Resolve before retrying.\n")
    return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cv_root = CACHE_DIR / "raw" / CV_25_RELEASE_DIR / CV_LOCALE
    if not cv_root.exists():
        logger.error(f"CV release dir not found at {cv_root}; download/extract first.")
        return 2

    manifest_path = CACHE_DIR / TRAIN_MANIFEST_NAME

    logger.info("loading train.tsv + clip_durations.tsv …")
    train_rows = _read_tsv_rows(cv_root / "train.tsv")
    durations = _read_clip_durations(cv_root)
    logger.info(f"  train rows: {len(train_rows)}")
    logger.info(f"  clip-duration entries: {len(durations)}")

    logger.info("applying §2.4 + §2.5 quality filter …")
    kept, rejected = filter_train_rows(train_rows, durations)
    logger.info(f"  kept {len(kept)} / {len(train_rows)}")
    for k, v in rejected.items():
        logger.info(f"  rejected.{k}: {v}")

    logger.info("sorting (sha256(client_id), clip_id) and accumulating to 100h …")
    selected = select_100h_prefix(kept)
    cum_hours = sum(r[2] for r in selected) / 3600.0
    logger.info(f"  selected {len(selected)} clips, cumulative {cum_hours:.4f} h")

    write_manifest(selected, manifest_path)
    sha = _file_sha256(manifest_path)
    logger.info(f"  manifest written: {manifest_path}")
    logger.info(f"  manifest sha256: {sha}")

    coverage = compute_char_coverage(train_rows)
    logger.info(f"  character coverage post-normalisation: {coverage:.4f}")

    stats = {
        "dataset_version": "common_voice_en_25.0",
        "release_dir": CV_25_RELEASE_DIR,
        "locale": CV_LOCALE,
        "target_hours": TARGET_HOURS,
        "selected_clips": len(selected),
        "selected_hours": cum_hours,
        "input_train_rows": len(train_rows),
        "kept_after_filter": len(kept),
        "rejected_counts": rejected,
        "character_coverage_post_normalisation": coverage,
        "manifest_filename": TRAIN_MANIFEST_NAME,
        "manifest_sha256": sha,
        "hash_salt": HASH_SALT,
        "constructed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    stats_path = CACHE_DIR / FILTER_STATS_NAME
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  filter_stats.json written: {stats_path}")

    if coverage < 0.90:
        logger.error(f"character coverage {coverage:.4f} < 0.90 — escalate per §2.5")
        return 3

    train_cids = {row[0] for row in selected}
    if not check_speaker_disjointness(train_cids, cv_root, CACHE_DIR):
        logger.error(
            f"speaker overlap detected; see {CACHE_DIR / SPEAKER_OVERLAP_DIAGNOSIS_NAME}"
        )
        return 4

    logger.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
