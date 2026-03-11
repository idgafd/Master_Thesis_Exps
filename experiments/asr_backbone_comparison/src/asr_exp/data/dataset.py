"""ASRDataset, DurationBatchSampler, collate_fn, SpecAugment."""

import random
from typing import List

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, Sampler

from asr_exp.config import ExperimentConfig
from asr_exp.data.vocab import CharVocab


def _load_mel(path: str, cfg: ExperimentConfig) -> torch.Tensor:
    """Load audio and compute log-mel spectrogram. Returns (n_mels, T)."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != cfg.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, cfg.sample_rate)

    win_len = int(cfg.win_length_ms * cfg.sample_rate / 1000)
    hop_len = int(cfg.hop_length_ms * cfg.sample_rate / 1000)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=win_len,
        win_length=win_len,
        hop_length=hop_len,
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
        mel = _load_mel(entry["path"], self.cfg)          # (n_mels, T)
        targets = torch.tensor(
            self.vocab.encode(entry["text"]), dtype=torch.long
        )
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
        mels_padded[i, :, : m.shape[1]] = m

    targets_padded = torch.zeros(len(targets), max_L, dtype=torch.long)
    for i, t in enumerate(targets):
        targets_padded[i, : len(t)] = t

    return mels_padded, targets_padded, mel_lengths, target_lengths


class DurationBatchSampler(Sampler):
    """Groups utterances into batches whose total duration ≤ batch_max_seconds.

    Sorts by duration to minimize padding, then shuffles within buckets so
    each epoch sees different orderings while keeping padding overhead low.
    """

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
        # Rebuild with a new seed each call to get epoch-level shuffling
        batches = self._build_batches(self.seed + self._epoch)
        self._epoch += 1
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)


class SpecAugment:
    """SpecAugment: frequency and time masking on log-mel spectrograms.

    Applied to a batch of mel spectrograms (B, n_mels, T).
    """

    def __init__(
        self,
        freq_mask_param: int,
        time_mask_param: int,
        num_freq_masks: int,
        num_time_masks: int,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, mels: torch.Tensor) -> torch.Tensor:
        """mels: (B, n_mels, T) — modified in place and returned."""
        mels = mels.clone()
        _, n_mels, T = mels.shape

        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, n_mels - f))
            mels[:, f0: f0 + f, :] = 0.0

        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, T))
            t0 = random.randint(0, max(0, T - t))
            mels[:, :, t0: t0 + t] = 0.0

        return mels
