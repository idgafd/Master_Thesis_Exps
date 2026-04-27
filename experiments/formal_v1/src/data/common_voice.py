"""Common Voice EN 25.0 dataset loading.

Mirrors the public API of `librispeech.py`:

    load_common_voice(split, cache_dir, min_audio_sec, max_audio_sec) -> List[dict]

For `split == "train"`, reads from the deterministic 100h manifest at
`<cache_dir>/100h_train_manifest_seed42.csv` constructed once by
`scripts/build_cv_manifest.py`.

For `split == "dev" | "test"`, reads CV's official `dev.tsv` / `test.tsv`
unmodified, then applies the §2.4 audio-quality filter (votes, duration,
non-empty sentence) and the §2.5 sentence normalisation pipeline.

Audio is decoded from mp3, resampled 48 kHz → 16 kHz with
`lowpass_filter_width=64` per task brief §2.6, and stored as numpy
arrays in the same shape as the LibriSpeech loader emits.

Each returned entry has keys:
    audio_array  np.ndarray, 16 kHz mono float32
    sample_rate  int, always 16000
    text         str, normalised sentence (a-z, space, apostrophe only)
    duration_sec float
    id           str, clip filename (e.g. "common_voice_en_32818178.mp3")
    client_id    str, CV speaker id (used for wer_by_speaker)
"""

from __future__ import annotations

import csv
import logging
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio

from src.data.vocab import normalize_text

logger = logging.getLogger(__name__)


# CV 25.0 release directory inside the extracted MDC tar. The exact name
# is part of the dataset version pin; it appears in cv_pilot.yaml as
# `cv_version: "25.0"` and in filter_stats.json as `release_dir`.
CV_25_RELEASE_DIR = "cv-corpus-25.0-2026-03-09"
CV_LOCALE = "en"

TRAIN_MANIFEST_NAME = "100h_train_manifest_seed42.csv"


def cv_normalize_sentence(s: str) -> str:
    """Sentence normalisation per task brief §2.5.

    Order: NFC unicode → lowercase → keep only [a-z, space, apostrophe] →
    collapse whitespace → strip. Returns "" if nothing survives (caller
    drops the clip per the §2.4 extension).

    The existing `vocab.normalize_text` already does lowercase + the
    [a-z ']-only filter + whitespace collapse. We add the NFC step in
    front to handle CV's wider unicode range (accented characters,
    smart quotes, etc.) — without NFC, an "é" composed of "e" + combining
    acute would be filtered to just "e" only after decomposition, making
    coverage unstable across releases.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    return normalize_text(s)


def _read_clip_durations(cv_root: Path) -> Dict[str, float]:
    """Read CV's `clip_durations.tsv` → {clip_filename: duration_seconds}.

    The file's header row is duplicated on line 2 in the CV 25.0 release
    we observed; we tolerate that by skipping any row whose first cell
    equals the header literal.
    """
    path = cv_root / "clip_durations.tsv"
    durations: Dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if i == 0 or line.startswith("clip\t") or not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                durations[parts[0]] = int(parts[1]) / 1000.0
            except ValueError:
                continue
    return durations


def _read_tsv_rows(tsv_path: Path) -> List[Dict[str, str]]:
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def _resample_to_16k(waveform: torch.Tensor, sr: int) -> np.ndarray:
    """Resample to 16 kHz with `lowpass_filter_width=64` per §2.6."""
    if sr != 16000:
        waveform = torchaudio.functional.resample(
            waveform, sr, 16000, lowpass_filter_width=64
        )
    return waveform.squeeze(0).contiguous().numpy()


def _load_audio(clip_path: Path) -> Optional[np.ndarray]:
    """Decode an mp3 → 16 kHz mono float32 numpy array, or None if unreadable."""
    try:
        wav, sr = torchaudio.load(str(clip_path))
    except Exception as e:
        logger.debug(f"failed to decode {clip_path.name}: {e}")
        return None
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return _resample_to_16k(wav.float(), sr)


def _passes_quality_filter(row: Dict[str, str], duration_sec: Optional[float]) -> bool:
    """§2.4 quality filter on a single TSV row.

    Drops if duration < 1.0s, up_votes < 2, down_votes >= 1, or empty
    sentence. Audio-readability is checked separately at audio-load time.
    """
    if duration_sec is None or duration_sec < 1.0:
        return False
    try:
        if int(row.get("up_votes", "0")) < 2:
            return False
        if int(row.get("down_votes", "0")) >= 1:
            return False
    except ValueError:
        return False
    if not row.get("sentence", "").strip():
        return False
    return True


def _load_split_from_tsv(
    cv_root: Path,
    tsv_name: str,
    durations: Dict[str, float],
    min_audio_sec: float,
    max_audio_sec: float,
    split_label: str,
) -> List[dict]:
    """Filter a TSV split, normalise sentences, decode audio."""
    rows = _read_tsv_rows(cv_root / tsv_name)
    clips_dir = cv_root / "clips"
    entries: List[dict] = []
    skipped_filter = 0
    skipped_normtext = 0
    skipped_band = 0
    skipped_decode = 0
    for row in rows:
        clip = row["path"]
        dur = durations.get(clip)
        if not _passes_quality_filter(row, dur):
            skipped_filter += 1
            continue
        text = cv_normalize_sentence(row["sentence"])
        if not text:
            skipped_normtext += 1
            continue
        if dur < min_audio_sec or dur > max_audio_sec:
            skipped_band += 1
            continue
        arr = _load_audio(clips_dir / clip)
        if arr is None:
            skipped_decode += 1
            continue
        entries.append({
            "audio_array": arr,
            "sample_rate": 16000,
            "text": text,
            "duration_sec": float(dur),
            "id": clip,
            "client_id": row["client_id"],
        })
    logger.info(
        f"{split_label}: kept {len(entries)} / {len(rows)} "
        f"(quality {skipped_filter}, normtext {skipped_normtext}, "
        f"band {skipped_band}, decode {skipped_decode})"
    )
    return entries


def load_common_voice(
    split: str = "train",
    cache_dir: str = "./data/common_voice",
    min_audio_sec: float = 1.0,
    max_audio_sec: float = 20.0,
) -> List[dict]:
    """Load Common Voice EN 25.0 split → list of LibriSpeech-shaped entries.

    For `split == "train"`, the deterministic 100h manifest at
    `<cache_dir>/100h_train_manifest_seed42.csv` is the source of truth.
    For `dev` / `test`, the CV-official TSVs are the source.
    """
    cache_root = Path(cache_dir)
    cv_root = cache_root / "raw" / CV_25_RELEASE_DIR / CV_LOCALE
    if not cv_root.exists():
        raise FileNotFoundError(
            f"CV release dir not found at {cv_root}. "
            f"Run scripts/build_cv_manifest.py to download/extract."
        )

    durations = _read_clip_durations(cv_root)
    clips_dir = cv_root / "clips"

    if split == "train":
        manifest_path = cache_root / TRAIN_MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Train manifest not found at {manifest_path}. "
                f"Run scripts/build_cv_manifest.py first."
            )
        with open(manifest_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        entries: List[dict] = []
        skipped_band = 0
        skipped_decode = 0
        for r in rows:
            clip = r["clip_id"]
            dur = float(r["duration_seconds"])
            if dur < min_audio_sec or dur > max_audio_sec:
                skipped_band += 1
                continue
            arr = _load_audio(clips_dir / clip)
            if arr is None:
                skipped_decode += 1
                continue
            entries.append({
                "audio_array": arr,
                "sample_rate": 16000,
                "text": r["sentence_normalised"],
                "duration_sec": dur,
                "id": clip,
                "client_id": r["client_id"],
            })
        logger.info(
            f"train: kept {len(entries)} / {len(rows)} from manifest "
            f"(band {skipped_band}, decode {skipped_decode})"
        )
        return entries

    if split == "dev":
        return _load_split_from_tsv(
            cv_root, "dev.tsv", durations, min_audio_sec, max_audio_sec, "dev"
        )
    if split == "test":
        return _load_split_from_tsv(
            cv_root, "test.tsv", durations, min_audio_sec, max_audio_sec, "test"
        )
    raise ValueError(f"unknown split: {split!r}")
