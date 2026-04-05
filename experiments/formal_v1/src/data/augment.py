"""SpecAugment: frequency and time masking on log-mel spectrograms."""

import random

import torch


class SpecAugment:
    """SpecAugment applied to batched mel spectrograms (B, n_mels, T)."""

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, mels: torch.Tensor) -> torch.Tensor:
        mels = mels.clone()
        _, n_mels, T = mels.shape

        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, n_mels - f))
            mels[:, f0:f0 + f, :] = 0.0

        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, T))
            t0 = random.randint(0, max(0, T - t))
            mels[:, :, t0:t0 + t] = 0.0

        return mels
