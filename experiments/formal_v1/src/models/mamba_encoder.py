"""Mamba encoder — unidirectional and bidirectional variants."""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from src.models.components import SinusoidalPE
from src.models.mamba_block import MambaBlock


class MambaEncoderLayer(nn.Module):
    """Single Mamba encoder layer: LN -> Mamba -> Drop -> LN -> FFN -> Drop."""

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        layer_id: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, dtype=dtype)
        self.ln2 = nn.LayerNorm(d_model, dtype=dtype)

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(d_model, dtype=dtype)
        else:
            self.ln0 = nn.Identity()

        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dtype=dtype,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model, dtype=dtype),
            nn.Dropout(dropout),
        )

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        x = self.ln0(x)
        mamba_out, new_state = self.mamba(self.ln1(x), state=state)
        x = x + self.drop(mamba_out)
        x = x + self.ffn(self.ln2(x))
        return x, new_state


class MambaEncoder(nn.Module):
    """Unidirectional Mamba encoder."""

    supports_carry_state = True

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        ffn_dim: int = 896,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.pos_enc = SinusoidalPE(d_model, max_len=8000, dropout=dropout)
        self.layers = nn.ModuleList([
            MambaEncoderLayer(
                d_model=d_model,
                ffn_dim=ffn_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                layer_id=i,
                dtype=dtype,
            )
            for i in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()

        new_states = [] if state is not None else None

        for i, layer in enumerate(self.layers):
            layer_state = state[i] if state is not None else None
            x, ns = layer(x, state=layer_state)
            x = x * mask_f
            if new_states is not None:
                new_states.append(ns)

        return x, new_states


class BidirMambaEncoder(nn.Module):
    """Bidirectional Mamba: forward SSM + backward SSM, outputs summed."""

    supports_carry_state = False

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        ffn_dim: int = 896,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.pos_enc = SinusoidalPE(d_model, max_len=8000, dropout=dropout)
        self.fwd_layers = nn.ModuleList([
            MambaEncoderLayer(
                d_model=d_model, ffn_dim=ffn_dim, d_state=d_state,
                d_conv=d_conv, expand=expand, dropout=dropout,
                layer_id=i, dtype=dtype,
            )
            for i in range(n_layers)
        ])
        self.bwd_layers = nn.ModuleList([
            MambaEncoderLayer(
                d_model=d_model, ffn_dim=ffn_dim, d_state=d_state,
                d_conv=d_conv, expand=expand, dropout=dropout,
                layer_id=i, dtype=dtype,
            )
            for i in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[object] = None,
    ) -> Tuple[torch.Tensor, None]:
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()

        x_fwd = x
        x_bwd = x.flip(1)

        for fwd_layer, bwd_layer in zip(self.fwd_layers, self.bwd_layers):
            x_fwd, _ = fwd_layer(x_fwd)
            x_fwd = x_fwd * mask_f

            x_bwd, _ = bwd_layer(x_bwd)
            x_bwd = x_bwd * mask_f.flip(1)

        x_bwd = x_bwd.flip(1)
        return x_fwd + x_bwd, None
