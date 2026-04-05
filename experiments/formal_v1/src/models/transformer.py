"""Matched Transformer baseline encoder.

Fixed from draft:
- norm_first=True (pre-norm, matches RWKV-6/LION architecture)
- FFN dim = int((d_model * 3.5) // 32 * 32) = 896 (matched to RWKV-6 ChannelMix)
- ln0 at layer 0 (input normalization, matches RWKV convention)
- Sinusoidal PE
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.components import SinusoidalPE


class TransformerEncoder(nn.Module):
    """Pre-norm Transformer encoder with matched FFN dim and ln0."""

    supports_carry_state = False

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.pos_enc = SinusoidalPE(d_model, max_len=8000, dropout=dropout)
        self.ln0 = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[object] = None,
    ) -> Tuple[torch.Tensor, None]:
        x = self.ln0(x)
        x = self.pos_enc(x)

        B, T, _ = x.shape
        padding_mask = ~(torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x, None
