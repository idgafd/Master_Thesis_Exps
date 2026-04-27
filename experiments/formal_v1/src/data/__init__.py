from src.data.vocab import CharVocab
from src.data.dataset import ASRDataset, DurationBatchSampler, collate_fn
from src.data.augment import SpecAugment
from src.data.librispeech import load_librispeech
from src.data.common_voice import load_common_voice


def load_dataset_split(cfg, split: str):
    """Dispatch dataset loading by `cfg.dataset`.

    Returns a `List[dict]` in the LibriSpeech-shaped format: each entry has
    `audio_array, sample_rate, text, duration_sec, id` (CV entries also
    include `client_id` for `wer_by_speaker`).
    """
    if cfg.dataset in ("librispeech_clean", "librispeech_other"):
        return load_librispeech(
            split=split,
            cache_dir=cfg.data_cache_dir,
            min_audio_sec=cfg.min_audio_sec,
            max_audio_sec=cfg.max_audio_sec,
        )
    if cfg.dataset == "common_voice_en_100h":
        return load_common_voice(
            split=split,
            cache_dir=cfg.data_cache_dir,
            min_audio_sec=cfg.min_audio_sec,
            max_audio_sec=cfg.max_audio_sec,
        )
    raise ValueError(f"unknown cfg.dataset: {cfg.dataset!r}")
