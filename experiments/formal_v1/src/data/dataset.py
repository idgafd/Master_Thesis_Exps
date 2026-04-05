"""ASRDataset, DurationBatchSampler, collate_fn for LibriSpeech."""

import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, Sampler

from src.config import ExperimentConfig
from src.data.vocab import CharVocab


def compute_mel(audio_array: np.ndarray, sample_rate: int, cfg: ExperimentConfig) -> torch.Tensor:
    """Compute log-mel spectrogram from raw audio array. Returns (n_mels, T)."""
    waveform = torch.from_numpy(audio_array).float().unsqueeze(0)

    if sample_rate != cfg.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, cfg.sample_rate)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.win_length_samples,
        win_length=cfg.win_length_samples,
        hop_length=cfg.hop_length_samples,
        n_mels=cfg.n_mels,
        f_min=20.0,
        f_max=8000.0,
    )
    mel = mel_transform(waveform).squeeze(0)  # (n_mels, T)
    mel = torch.log(mel + 1e-9)
    return mel


class ASRDataset(Dataset):
    """Dataset returning (mel, target_ids) pairs for CTC training."""

    def __init__(self, entries: List[dict], vocab: CharVocab, cfg: ExperimentConfig):
        self.entries = entries
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        mel = compute_mel(entry["audio_array"], entry["sample_rate"], self.cfg)
        targets = torch.tensor(self.vocab.encode(entry["text"]), dtype=torch.long)
        return mel, targets


def collate_fn(batch):
    """Collate (mel, targets) pairs into padded tensors.

    Returns: mels (B, n_mels, T), targets (B, max_L), mel_lengths (B,), target_lengths (B,)
    """
    mels, targets = zip(*batch)

    mel_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    n_mels = mels[0].shape[0]
    max_T = max(m.shape[1] for m in mels)
    max_L = max(len(t) for t in targets)

    mels_padded = torch.zeros(len(mels), n_mels, max_T)
    for i, m in enumerate(mels):
        mels_padded[i, :, :m.shape[1]] = m

    targets_padded = torch.zeros(len(targets), max_L, dtype=torch.long)
    for i, t in enumerate(targets):
        targets_padded[i, :len(t)] = t

    return mels_padded, targets_padded, mel_lengths, target_lengths


class DurationBatchSampler(Sampler):
    """Groups utterances into batches by total duration <= batch_max_seconds."""

    def __init__(
        self,
        entries: List[dict],
        batch_max_seconds: float,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.entries = entries
        self.batch_max_seconds = batch_max_seconds
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._batches = self._build_batches(seed)

    def _build_batches(self, seed: int) -> List[List[int]]:
        indices = sorted(range(len(self.entries)), key=lambda i: self.entries[i]["duration_sec"])

        batches = []
        batch: List[int] = []
        batch_dur = 0.0
        for i in indices:
            dur = self.entries[i]["duration_sec"]
            if batch and batch_dur + dur > self.batch_max_seconds:
                batches.append(batch)
                batch = []
                batch_dur = 0.0
            batch.append(i)
            batch_dur += dur
        if batch:
            batches.append(batch)

        if self.shuffle:
            rng = random.Random(seed)
            rng.shuffle(batches)

        return batches

    def __iter__(self):
        batches = self._build_batches(self.seed + self._epoch)
        self._epoch += 1
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)
