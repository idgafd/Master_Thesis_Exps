"""Bidirectional Mamba encoder — Vim-style flip-and-sum approach.

Each layer has two separate Mamba SSM blocks with independent parameters:
  - Forward block processes x in natural order
  - Backward block processes x.flip() (reversed sequence)
  - Outputs are summed (divided by 2)

This is the same strategy as Vision Mamba (Vim) v2 (Zhu et al. 2024).
Uses standard mamba-ssm Mamba blocks (no forked kernels needed).

Unlike LION, there is no correction term — token i's self-contribution
is counted in both directions. The approach is heuristic but simple.

Not compatible with carry-state inference (requires full sequence).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from asr_exp.models.components import SinusoidalPE


class BidirVimMambaEncoder(nn.Module):
    """Bidirectional Mamba encoder using Vim-style dual scan.

    Two independent Mamba blocks per layer (forward + backward),
    plus shared LayerNorms and FFN.
    Requires: mamba-ssm and causal-conv1d packages.
    """

    supports_carry_state = False

    def __init__(self, d_model: int, n_layers: int, dropout: float):
        super().__init__()
        from mamba_ssm import Mamba

        self.d_model = d_model
        self.n_layers = n_layers

        self.pos_enc = SinusoidalPE(d_model, max_len=8000)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(d_model),
                        "mamba_fwd": Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                        "mamba_bwd": Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                        "norm2": nn.LayerNorm(d_model),
                        "ffn": nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_model * 4, d_model),
                            nn.Dropout(dropout),
                        ),
                    }
                )
            )
        self.dropout = nn.Dropout(dropout)
        print(
            f"Bidirectional Mamba (Vim) encoder initialized "
            f"({n_layers} layers, d_model={d_model}, d_state=16, d_conv=4, expand=2)"
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[list] = None,
    ) -> Tuple[torch.Tensor, None]:
        x = self.pos_enc(x)
        B, T, _ = x.shape

        # Padding mask
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()  # (B, T, 1)

        for layer in self.layers:
            residual = x
            x_norm = layer["norm1"](x)

            # Forward and backward Mamba scans
            y_fwd = layer["mamba_fwd"](x_norm)
            y_bwd = layer["mamba_bwd"](x_norm.flip([1])).flip([1])

            # Vim v2: sum and divide by 2
            x = residual + self.dropout((y_fwd + y_bwd) * 0.5)
            x = x + layer["ffn"](layer["norm2"](x))

        # Zero out padding
        x = x * mask_f

        return x, None
