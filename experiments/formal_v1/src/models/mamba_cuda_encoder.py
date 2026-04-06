"""Mamba encoder using the official mamba-ssm CUDA kernels.

Drop-in replacement for MambaEncoder, but uses the high-performance
CUDA selective scan from the `mamba-ssm` package.

Usage:
    from src.models.mamba_cuda_encoder import MambaCudaEncoder
    encoder = MambaCudaEncoder(d_model=256, n_layers=6, dropout=0.1)
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from mamba_ssm import Mamba as MambaCuda

from src.models.components import SinusoidalPE


class MambaCudaEncoderLayer(nn.Module):
    """Single Mamba encoder layer using official CUDA Mamba block.

    Structure: LN -> Mamba(CUDA) -> Drop -> LN -> FFN -> Drop
    """

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        layer_id: int = 0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(d_model)
        else:
            self.ln0 = nn.Identity()

        self.mamba = MambaCuda(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_id,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln0(x)
        mamba_out = self.mamba(self.ln1(x))
        x = x + self.drop(mamba_out)
        x = x + self.ffn(self.ln2(x))
        return x


class MambaCudaEncoder(nn.Module):
    """Mamba encoder using official CUDA kernels for max speed."""

    supports_carry_state = False  # CUDA Mamba doesn't expose carry-state in training mode

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        ffn_dim: int = 896,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.pos_enc = SinusoidalPE(d_model, max_len=8000, dropout=dropout)
        self.layers = nn.ModuleList([
            MambaCudaEncoderLayer(
                d_model=d_model,
                ffn_dim=ffn_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                layer_id=i,
            )
            for i in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[List] = None,
    ) -> Tuple[torch.Tensor, None]:
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()

        for layer in self.layers:
            x = layer(x)
            x = x * mask_f

        return x, None
