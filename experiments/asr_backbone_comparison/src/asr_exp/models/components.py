"""Shared model components: ConvSubsampling, CTCHead, SinusoidalPE."""

import math

import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 8000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """x: (B, T, D); offset shifts position index for carry-state PE continuity."""
        x = x + self.pe[:, offset : offset + x.size(1)]
        return self.dropout(x)


class ConvSubsampling(nn.Module):
    """Two stride-2 Conv2d layers → 4× temporal downsampling."""

    def __init__(self, n_mels: int, d_model: int, channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        freq_out = math.ceil(math.ceil(n_mels / 2) / 2)
        self.proj = nn.Linear(channels * freq_out, d_model)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, n_mels, T)
        lengths: (B,)  original mel frame counts

        Returns: (B, T', d_model), new_lengths
        """
        x = x.unsqueeze(1)           # (B, 1, n_mels, T)
        x = self.conv(x)              # (B, C, n_mels/4, T/4)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)  # (B, T', C*F)
        x = self.proj(x)              # (B, T', d_model)

        new_lengths = ((lengths - 1) // 2 + 1)
        new_lengths = ((new_lengths - 1) // 2 + 1)
        new_lengths = torch.clamp(new_lengths, min=1)
        return x, new_lengths


class CTCHead(nn.Module):
    """LayerNorm + linear projection to vocabulary logits."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))
