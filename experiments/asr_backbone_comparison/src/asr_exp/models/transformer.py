"""Transformer encoder backbone."""

import torch
import torch.nn as nn

from asr_exp.models.components import SinusoidalPE


class TransformerEncoder(nn.Module):
    """Standard Transformer encoder with sinusoidal PE.

    Stateless — full attention is recomputed per chunk.
    """

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
        self.pos_enc = SinusoidalPE(d_model, max_len=8000)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state=None,
    ) -> tuple[torch.Tensor, None]:
        """x: (B, T, D); stateless — state is always None."""
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask = (
            torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        )
        x = self.encoder(x, src_key_padding_mask=mask)
        return x, None
