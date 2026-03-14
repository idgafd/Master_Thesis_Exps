"""Common Voice data loading and preparation."""

import csv
import os
import subprocess
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

import soundfile as sf

from asr_exp.config import ExperimentConfig
from asr_exp.data.vocab import CharVocab, normalize_text


def _get_duration(path: str) -> float:
    """Get audio duration in seconds without loading full audio."""
    try:
        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:
        return 0.0


def _load_tsv_entries(
    tsv_path: str,
    clips_dir: str,
    min_sec: float,
    max_sec: float,
    max_hours: float,
) -> List[dict]:
    """Load entries from a Common Voice TSV file.

    Filters by duration, caps at max_hours, and normalizes text.
    Skips entries with empty normalized text or missing audio files.
    """
    entries = []
    total_dur = 0.0
    max_sec_total = max_hours * 3600.0

    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if total_dur >= max_sec_total:
                break

            clip_name = row.get("path", "").strip()
            sentence = row.get("sentence", "").strip()

            if not clip_name or not sentence:
                continue

            # CV clips can come with or without extension in TSV
            if not clip_name.endswith(".mp3"):
                clip_name = clip_name + ".mp3"
            audio_path = os.path.join(clips_dir, clip_name)

            if not os.path.exists(audio_path):
                continue

            text = normalize_text(sentence)
            if not text:
                continue

            dur = _get_duration(audio_path)
            if dur < min_sec or dur > max_sec:
                continue

            entries.append({"path": audio_path, "text": text, "duration_sec": dur})
            total_dur += dur

    return entries


def _download_cv(cfg: ExperimentConfig, data_dir: Path) -> None:
    """Download and extract Common Voice tarball via Mozilla datacollective API."""
    import json as _json
    import urllib.error
    import urllib.request

    api_key = cfg.cv_api_key or os.environ.get("CV_API_KEY", "")
    tarball_path = data_dir.parent / cfg.cv_tarball_name

    if not tarball_path.exists():
        if not api_key:
            raise RuntimeError(
                "Common Voice data not found and CV_API_KEY is not set.\n"
                f"  Expected data at: {data_dir}\n"
                f"  Set CV_API_KEY env var or place the dataset at the above path.\n"
                f"  Dataset ID: {cfg.cv_dataset_id}"
            )

        print("Requesting download URL from Mozilla datacollective API...")
        req = urllib.request.Request(
            f"https://datacollective.mozillafoundation.org/api/datasets"
            f"/{cfg.cv_dataset_id}/download",
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                body = _json.loads(resp.read().decode())
            download_url = body["downloadUrl"]
        except (urllib.error.HTTPError, KeyError) as e:
            raise RuntimeError(
                f"Failed to get download URL: {e}\n"
                "Check your API key and dataset ID."
            ) from e

        print(f"Downloading {cfg.cv_tarball_name} ...")
        subprocess.run(["curl", "-L", "-o", str(tarball_path), download_url], check=True)
        print(f"Downloaded to {tarball_path} ({os.path.getsize(tarball_path) / 1e9:.2f} GB)")
    else:
        print(f"Tarball already exists: {tarball_path}")

    print(f"Extracting {tarball_path} → {data_dir.parent} ...")
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(data_dir.parent)
    print("Extraction complete.")


def _find_cv_root(data_dir: Path) -> Path:
    """Return path that contains clips/ and train.tsv.

    Search under data_dir first, then fall back to its parent. The downloader
    extracts the Mozilla archive into data_dir.parent, which can leave the
    configured data_dir empty on first run.
    """
    search_roots = [data_dir]
    if data_dir.parent != data_dir:
        search_roots.append(data_dir.parent)

    for root in search_roots:
        subdirs = list(root.iterdir()) if root.is_dir() else []
        for candidate in [root, *subdirs]:
            if isinstance(candidate, Path) and (candidate / "clips").is_dir() and (candidate / "train.tsv").exists():
                return candidate
        # Common Voice sometimes nests one level deep (uk/, cv-corpus-*/uk/, etc.)
        for subdir in sorted(root.rglob("train.tsv")):
            return subdir.parent
    raise FileNotFoundError(
        f"Could not find Common Voice split files (train.tsv + clips/) under {data_dir} or {data_dir.parent}"
    )


def load_and_prepare_data(
    cfg: ExperimentConfig,
    build_vocab: bool = True,
) -> Tuple[List[dict], List[dict], List[dict], Optional[CharVocab]]:
    """Load Common Voice train/dev/test splits.

    Downloads and extracts data if not present.
    Returns (train_entries, dev_entries, test_entries, vocab_or_None).
    vocab is None when build_vocab=False (caller provides existing vocab).
    """
    data_dir = Path(cfg.data_dir)

    if not data_dir.exists() or not any(data_dir.rglob("train.tsv")):
        print(f"Data not found at {data_dir}, attempting download ...")
        data_dir.mkdir(parents=True, exist_ok=True)
        _download_cv(cfg, data_dir)

    cv_root = _find_cv_root(data_dir)
    clips_dir = str(cv_root / "clips")
    print(f"Loading Common Voice data from {cv_root}")

    split_specs = [
        ("train", cfg.max_train_hours),
        ("dev",   cfg.max_val_hours),
        ("test",  cfg.max_test_hours),
    ]
    splits = {}
    for split_name, max_hours in split_specs:
        tsv_path = cv_root / f"{split_name}.tsv"
        if not tsv_path.exists():
            raise FileNotFoundError(f"Missing split file: {tsv_path}")
        print(f"  Loading {split_name}.tsv (max {max_hours}h) ...")
        entries = _load_tsv_entries(
            str(tsv_path), clips_dir,
            cfg.min_audio_sec, cfg.max_audio_sec, max_hours,
        )
        total_h = sum(e["duration_sec"] for e in entries) / 3600.0
        print(f"    {len(entries):,} utterances | {total_h:.1f}h")
        splits[split_name] = entries

    vocab: Optional[CharVocab] = None
    if build_vocab:
        all_texts = [e["text"] for e in splits["train"]]
        vocab = CharVocab.from_texts(all_texts)
        print(f"Vocabulary: {vocab.size} tokens")

    return splits["train"], splits["dev"], splits["test"], vocab
