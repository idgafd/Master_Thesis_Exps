"""Bidirectional sequential (Bi-WKV / Vim) encoders for RWKV-6 and Mamba-2.

These encoders implement the AudioRWKV-style "Bi-WKV" pattern: per-layer
forward + backward causal scans over the sequence, output averaged.  The
"clean" variant uses a fixed 0.5/0.5 average (no gate, no ConvShift) — the
simplest possible bidirectional adaptation, equivalent to a BiLSTM
fusion strategy applied to the token-mixer.

This is *different* from LION (`mode="lion"` in the existing causal encoders),
which uses a parallel matrix form with a correction term.  Bi-WKV runs two
sequential causal RNN passes per layer with no correction — token i's
self-contribution is counted in both directions.

Param budget calibration: each Bi-WKV layer contains 2 full causal blocks
(plus shared layer-norms / FFN already inside each block).  To match the
~7M parameter envelope used by the 6-layer causal RWKV-6 / Mamba-2 cells,
use ``n_layers ≈ causal_n_layers / 2`` (e.g. 3 Bi-WKV layers ≈ 6 causal
blocks ≈ 7.7M params on the d_model=256 spine).

Token-shift / per-layer dropouts / FFN are inherited from the underlying
block — same as the causal cells.

Not compatible with carry-state inference (requires full sequence to
construct the backward pass).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.components import SinusoidalPE
from src.models.rwkv6_block import RWKV6Block
from src.models.mamba2_encoder import Mamba2EncoderLayer


class BiWKVRWKV6Encoder(nn.Module):
    """Bi-WKV RWKV-6 encoder (clean variant: per-layer fwd + bwd average).

    Per layer:
      p_fwd = block_fwd(x)
      p_bwd = flip(block_bwd(flip(x)))
      x_next = 0.5 * (p_fwd + p_bwd)
    """

    supports_carry_state = False

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        head_size: int = 64,
    ):
        super().__init__()
        assert d_model % 32 == 0
        assert d_model % head_size == 0
        self.d_model = d_model
        self.n_layers = n_layers

        n_head = d_model // head_size
        n_rwkv_blocks = 2 * n_layers  # for layer-id-dependent decay init
        self.layers_fwd = nn.ModuleList()
        self.layers_bwd = nn.ModuleList()
        for i in range(n_layers):
            self.layers_fwd.append(
                RWKV6Block(
                    hidden_size=d_model,
                    n_head=n_head,
                    head_size=head_size,
                    num_hidden_layers=n_rwkv_blocks,
                    layer_id=2 * i,
                    dropout=dropout,
                    mode="recurrent",
                )
            )
            self.layers_bwd.append(
                RWKV6Block(
                    hidden_size=d_model,
                    n_head=n_head,
                    head_size=head_size,
                    num_hidden_layers=n_rwkv_blocks,
                    layer_id=2 * i + 1,
                    dropout=dropout,
                    mode="recurrent",
                )
            )

        self.pos_enc = SinusoidalPE(d_model, max_len=8000, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[list] = None,
    ) -> Tuple[torch.Tensor, None]:
        if state is not None:
            raise RuntimeError("BiWKVRWKV6Encoder does not support carry-state")
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask_f = (
            (torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))
            .unsqueeze(-1).float()
        )
        for fwd_block, bwd_block in zip(self.layers_fwd, self.layers_bwd):
            p_fwd, _ = fwd_block(x)
            p_bwd_flipped, _ = bwd_block(x.flip(1))
            x = 0.5 * (p_fwd + p_bwd_flipped.flip(1))
        x = x * mask_f
        return x, None


class BidirVimMamba2Encoder(nn.Module):
    """Vim-style bidirectional Mamba-2 encoder (clean variant).

    Per layer:
      p_fwd = block_fwd(x)
      p_bwd = flip(block_bwd(flip(x)))
      x_next = 0.5 * (p_fwd + p_bwd)
    """

    supports_carry_state = False

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        ffn_dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        headdim: int = 64,
        expand: int = 2,
        ngroups: int = 1,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.layers_fwd = nn.ModuleList()
        self.layers_bwd = nn.ModuleList()
        for i in range(n_layers):
            common = dict(
                d_model=d_model,
                ffn_dim=ffn_dim,
                d_state=d_state,
                d_conv=d_conv,
                headdim=headdim,
                expand=expand,
                ngroups=ngroups,
                chunk_size=chunk_size,
                mode="recurrent",
                dropout=dropout,
            )
            self.layers_fwd.append(Mamba2EncoderLayer(layer_id=2 * i, **common))
            self.layers_bwd.append(Mamba2EncoderLayer(layer_id=2 * i + 1, **common))

        self.pos_enc = SinusoidalPE(d_model, max_len=8000, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[list] = None,
    ) -> Tuple[torch.Tensor, None]:
        if state is not None:
            raise RuntimeError("BidirVimMamba2Encoder does not support carry-state")
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask_f = (
            (torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))
            .unsqueeze(-1).float()
        )
        for fwd_block, bwd_block in zip(self.layers_fwd, self.layers_bwd):
            p_fwd, _ = fwd_block(x)
            p_bwd_flipped, _ = bwd_block(x.flip(1))
            x = 0.5 * (p_fwd + p_bwd_flipped.flip(1))
        x = x * mask_f
        return x, None
