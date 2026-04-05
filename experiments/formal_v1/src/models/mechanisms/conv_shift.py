"""DWConvShift: learned depthwise Conv1d replacing fixed token shift.

Initialized to symmetric [0.5, 0, 0.5] to match the bidirectional
(x[t-1] + x[t+1]) / 2 token shift at initialization.
"""

import torch
import torch.nn as nn


class DWConvShift(nn.Module):
    """Learned depthwise Conv1d token shift."""

    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        # Initialize to [0.5, 0, 0.5] — matches bidirectional average
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, 0, 0] = 0.5   # left neighbor
            self.conv.weight[:, 0, -1] = 0.5   # right neighbor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        return self.conv(x.transpose(1, 2)).transpose(1, 2)
