"""Tests for the Common Voice EN 100h pipeline.

Five invariants per task brief §3.1:
  (a) manifest construction is deterministic across two runs (sha256 match)
  (b) speaker-disjointness train ⊥ dev ⊥ test
  (c) sentence normalisation outputs only [a-z, space, apostrophe]
  (d) audio loads at 16 kHz with expected (channel-collapsed mono) shape
  (e) train manifest cumulative duration is in [99.5h, 100.5h]

Tests in groups (a/c) operate on synthetic fixtures and always run.
Tests in groups (b/d/e) need the extracted CV release dir and the built
manifest; they are auto-skipped when those are missing, so the suite
passes during pipeline bring-up before the first manifest build.
"""

from __future__ import annotations

import csv
import hashlib
import os
import string
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from src.data.common_voice import (
    CV_25_RELEASE_DIR,
    CV_LOCALE,
    TRAIN_MANIFEST_NAME,
    cv_normalize_sentence,
    _read_clip_durations,
    _read_tsv_rows,
    _load_audio,
    load_common_voice,
)
from scripts.build_cv_manifest import (  # noqa: E402
    CACHE_DIR,
    HASH_SALT,
    _client_id_hash,
    filter_train_rows,
    select_100h_prefix,
    check_speaker_disjointness,
)


CV_ROOT = CACHE_DIR / "raw" / CV_25_RELEASE_DIR / CV_LOCALE
MANIFEST_PATH = CACHE_DIR / TRAIN_MANIFEST_NAME

ALLOWED_VOCAB = set(string.ascii_lowercase) | {" ", "'"}


# ── (c) Sentence normalisation outputs only {a-z, space, apostrophe} ─────


@pytest.mark.parametrize("raw,expected", [
    ("Hello World", "hello world"),
    ("It's a Test.", "it's a test"),
    ("multiple   spaces", "multiple spaces"),
    ("café résumé", "caf rsum"),               # accented chars stripped post-NFC
    ("smart “quotes”", "smart quotes"),
    ("punctuation, removed!", "punctuation removed"),
    ("", ""),
    ("\t\n  ", ""),
    ("123 numbers", "numbers"),
])
def test_normalise_sentence_table(raw: str, expected: str) -> None:
    assert cv_normalize_sentence(raw) == expected


def test_normalise_sentence_only_allowed_chars() -> None:
    """Property test: the result only contains [a-z, space, apostrophe]."""
    samples = [
        "Random text 123 with é, ü, and emoji 🎉.",
        "  multiple\t\nwhitespace   collapsed  ",
        "I'll, won't; can't",
        "MIXED CaSe & symbols!",
        "naïve façade — em-dash",
        "1234567890",
        "",
    ]
    for s in samples:
        out = cv_normalize_sentence(s)
        assert all(c in ALLOWED_VOCAB for c in out), \
            f"unexpected char in {out!r} from {s!r}"


# ── (a) Manifest construction is deterministic ──────────────────────────


def _synthetic_train_rows(n_speakers: int = 25, n_clips_per: int = 80) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for s in range(n_speakers):
        cid = f"speaker_{s:04d}"
        for c in range(n_clips_per):
            rows.append({
                "client_id": cid,
                "path": f"common_voice_en_{s:04d}_{c:04d}.mp3",
                "sentence": f"this is utterance number {c} from speaker {s}.",
                "up_votes": "2",
                "down_votes": "0",
                "sentence_id": "x", "sentence_domain": "", "age": "",
                "gender": "", "accents": "", "variant": "",
                "locale": "en", "segment": "",
            })
    return rows


def _synthetic_durations(rows: List[Dict[str, str]], dur_s: float = 5.0) -> Dict[str, float]:
    return {r["path"]: dur_s for r in rows}


def test_manifest_construction_deterministic() -> None:
    """Two in-memory builds from identical inputs must produce identical outputs."""
    rows = _synthetic_train_rows()
    durs = _synthetic_durations(rows)

    kept_a, _ = filter_train_rows(rows, durs)
    kept_b, _ = filter_train_rows(list(rows), dict(durs))
    sel_a = select_100h_prefix(kept_a, target_hours=1.0)  # smaller target for speed
    sel_b = select_100h_prefix(kept_b, target_hours=1.0)
    assert sel_a == sel_b

    # Hash the serialised form too — guarantees CSV-on-disk equality.
    def serialise(sel):
        return hashlib.sha256(
            "\n".join(
                f"{c}\t{p}\t{d:.3f}\t{s}" for (c, p, d, s) in sel
            ).encode("utf-8")
        ).hexdigest()

    assert serialise(sel_a) == serialise(sel_b)


def test_client_id_hash_deterministic_and_salted() -> None:
    h1 = _client_id_hash("speaker_xyz")
    h2 = _client_id_hash("speaker_xyz")
    h3 = _client_id_hash("speaker_xyy")
    assert h1 == h2
    assert h1 != h3
    # Salt is the documented one — protects against silent salt drift.
    expected = hashlib.sha256((HASH_SALT + "speaker_xyz").encode("utf-8")).hexdigest()
    assert h1 == expected


def test_filter_drops_low_quality_rows() -> None:
    rows = [
        {"client_id": "a", "path": "1.mp3", "sentence": "good utterance",
         "up_votes": "2", "down_votes": "0"},
        {"client_id": "a", "path": "2.mp3", "sentence": "should drop downvoted",
         "up_votes": "5", "down_votes": "1"},
        {"client_id": "b", "path": "3.mp3", "sentence": "low up votes",
         "up_votes": "1", "down_votes": "0"},
        {"client_id": "c", "path": "4.mp3", "sentence": "",
         "up_votes": "3", "down_votes": "0"},
        {"client_id": "d", "path": "5.mp3", "sentence": "missing duration",
         "up_votes": "2", "down_votes": "0"},
        {"client_id": "e", "path": "6.mp3", "sentence": "too short",
         "up_votes": "2", "down_votes": "0"},
    ]
    durs = {"1.mp3": 5.0, "2.mp3": 5.0, "3.mp3": 5.0, "4.mp3": 5.0, "6.mp3": 0.5}
    kept, rejected = filter_train_rows(rows, durs)
    assert len(kept) == 1
    assert kept[0][1] == "1.mp3"
    assert rejected["any_down_votes"] == 1
    assert rejected["low_up_votes"] == 1
    assert rejected["empty_sentence_raw"] == 1
    assert rejected["missing_duration"] == 1
    assert rejected["duration_lt_1s"] == 1


# ── (b/d/e) data-dependent tests — auto-skip when the dataset isn't local ─


needs_data = pytest.mark.skipif(
    not CV_ROOT.exists(),
    reason=f"CV release dir not extracted at {CV_ROOT}",
)
needs_manifest = pytest.mark.skipif(
    not MANIFEST_PATH.exists(),
    reason=f"Manifest not built at {MANIFEST_PATH}; run scripts/build_cv_manifest.py",
)


@needs_data
def test_speaker_disjointness_dev_test() -> None:
    """§2.3 — dev and test must be speaker-disjoint independent of train build."""
    dev = {r["client_id"] for r in _read_tsv_rows(CV_ROOT / "dev.tsv")}
    test = {r["client_id"] for r in _read_tsv_rows(CV_ROOT / "test.tsv")}
    assert not (dev & test), \
        f"dev/test client_id overlap: {len(dev & test)} ids"


@needs_data
@needs_manifest
def test_speaker_disjointness_train_dev_test(tmp_path) -> None:
    with open(MANIFEST_PATH, encoding="utf-8", newline="") as f:
        train_cids = {row["client_id"] for row in csv.DictReader(f)}
    assert check_speaker_disjointness(train_cids, CV_ROOT, tmp_path)


@needs_data
def test_audio_loads_at_16k_with_correct_shape() -> None:
    """§2.6 — sample mp3 decodes to mono 16 kHz numpy array."""
    clips_dir = CV_ROOT / "clips"
    sample = next(iter(clips_dir.glob("common_voice_en_*.mp3")), None)
    assert sample is not None, "no clips found in extracted CV release"
    arr = _load_audio(sample)
    assert arr is not None, f"failed to decode {sample.name}"
    assert arr.ndim == 1, f"expected mono 1D array, got shape {arr.shape}"
    durations = _read_clip_durations(CV_ROOT)
    expected_samples = int(round(durations[sample.name] * 16000))
    # Resampling can introduce ±1 sample at the boundary; allow 2-sample slack.
    assert abs(len(arr) - expected_samples) <= 2, \
        f"len={len(arr)} expected≈{expected_samples} for {sample.name}"


@needs_manifest
def test_manifest_cumulative_duration_in_band() -> None:
    """§2.2 — cumulative duration in [99.5h, 100.5h]."""
    cum_seconds = 0.0
    n_rows = 0
    with open(MANIFEST_PATH, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cum_seconds += float(row["duration_seconds"])
            n_rows += 1
    cum_hours = cum_seconds / 3600.0
    assert 99.5 <= cum_hours <= 100.5, \
        f"manifest cumulative {cum_hours:.4f}h not in [99.5, 100.5]; n_rows={n_rows}"
