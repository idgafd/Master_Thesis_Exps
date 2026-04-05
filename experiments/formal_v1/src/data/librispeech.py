"""LibriSpeech dataset loading via HuggingFace datasets."""

import logging
from typing import Dict, List

from datasets import load_dataset

from src.data.vocab import normalize_text

logger = logging.getLogger(__name__)


def load_librispeech(
    split: str = "train",
    cache_dir: str = "./data/librispeech",
    min_audio_sec: float = 0.5,
    max_audio_sec: float = 20.0,
) -> List[dict]:
    """Load LibriSpeech clean from HuggingFace and return list of entries.

    Args:
        split: "train" (train-clean-100), "dev" (dev-clean), "test" (test-clean)
        cache_dir: where HuggingFace caches the dataset
        min_audio_sec: minimum utterance duration
        max_audio_sec: maximum utterance duration

    Returns:
        List of dicts with keys: audio_array, sample_rate, text, duration_sec, id
    """
    hf_split_map = {
        "train": "train.100",
        "dev": "validation",
        "test": "test",
    }
    hf_split = hf_split_map.get(split, split)

    logger.info(f"Loading LibriSpeech split={hf_split} from HuggingFace...")
    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split=hf_split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    entries = []
    skipped = 0
    for item in ds:
        audio = item["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]

        if duration < min_audio_sec or duration > max_audio_sec:
            skipped += 1
            continue

        text = normalize_text(item["text"])
        if not text.strip():
            skipped += 1
            continue

        entries.append({
            "audio_array": audio["array"],
            "sample_rate": audio["sampling_rate"],
            "text": text,
            "duration_sec": duration,
            "id": item.get("id", f"{split}_{len(entries)}"),
        })

    logger.info(
        f"Loaded {len(entries)} utterances from {hf_split} "
        f"(skipped {skipped} outside [{min_audio_sec}s, {max_audio_sec}s])"
    )
    return entries
