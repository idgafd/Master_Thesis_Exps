"""Mamba-2 encoder — unidirectional or bidirectional via ``mode`` flag.

No ``BidirMamba2Encoder`` class: bidirectionality lives inside the block
(``mode="lion"`` / ``"lion_chunk"``), matching the LION pattern used for
RWKV-6 in this repo.  Parameter count is identical to the causal encoder.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.models.components import SinusoidalPE
from src.models.mamba2_block import Mamba2Block


class Mamba2EncoderLayer(nn.Module):
    """Single layer: LN → Mamba2 → drop → LN → FFN → drop."""

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        headdim: int = 64,
        expand: int = 2,
        ngroups: int = 1,
        chunk_size: int = 64,
        mode: str = "recurrent",
        dropout: float = 0.1,
        layer_id: int = 0,
        dtype: torch.dtype = torch.float32,
        use_multidil_sym: bool = False,
        use_convshift_sym: bool = False,
        use_lucid: bool = False,
        lucid_key_source: str = "B",
        use_novelty_gate: bool = False,
        novelty_gamma_fixed: Optional[float] = None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, dtype=dtype)
        self.ln2 = nn.LayerNorm(d_model, dtype=dtype)
        self.ln0 = nn.LayerNorm(d_model, dtype=dtype) if layer_id == 0 else nn.Identity()

        self.mamba = Mamba2Block(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            headdim=headdim,
            expand=expand,
            ngroups=ngroups,
            chunk_size=chunk_size,
            mode=mode,
            dtype=dtype,
            use_multidil_sym=use_multidil_sym,
            use_convshift_sym=use_convshift_sym,
            use_lucid=use_lucid,
            lucid_key_source=lucid_key_source,
            use_novelty_gate=use_novelty_gate,
            novelty_gamma_fixed=novelty_gamma_fixed,
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
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        x = self.ln0(x)
        mamba_out, new_state = self.mamba(self.ln1(x), state=state)
        x = x + self.drop(mamba_out)
        x = x + self.ffn(self.ln2(x))
        return x, new_state


class Mamba2Encoder(nn.Module):
    """Mamba-2 encoder; ``mode`` selects unidirectional / bidirectional.

    - ``mode="recurrent"``   — causal SSD scan.  ``supports_carry_state = True``.
    - ``mode="lion"``        — LION full-attention bidirectional.  No carry.
    - ``mode="lion_chunk"``  — LION chunkwise bidirectional.        No carry.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        ffn_dim: int = 896,
        d_state: int = 64,
        d_conv: int = 4,
        headdim: int = 64,
        expand: int = 2,
        ngroups: int = 1,
        chunk_size: int = 64,
        mode: str = "recurrent",
        dtype: torch.dtype = torch.float32,
        use_multidil_sym: bool = False,
        use_convshift_sym: bool = False,
        use_lucid: bool = False,
        lucid_key_source: str = "B",
        use_novelty_gate: bool = False,
        novelty_gamma_fixed: Optional[float] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.mode = mode

        self.pos_enc = SinusoidalPE(d_model, max_len=8000, dropout=dropout)
        self.layers = nn.ModuleList([
            Mamba2EncoderLayer(
                d_model=d_model,
                ffn_dim=ffn_dim,
                d_state=d_state,
                d_conv=d_conv,
                headdim=headdim,
                expand=expand,
                ngroups=ngroups,
                chunk_size=chunk_size,
                mode=mode,
                dropout=dropout,
                layer_id=i,
                dtype=dtype,
                use_multidil_sym=use_multidil_sym,
                use_convshift_sym=use_convshift_sym,
                use_lucid=use_lucid,
                lucid_key_source=lucid_key_source,
                use_novelty_gate=use_novelty_gate,
                novelty_gamma_fixed=novelty_gamma_fixed,
            )
            for i in range(n_layers)
        ])

    @property
    def supports_carry_state(self) -> bool:
        return self.mode == "recurrent"

    def init_state(self, batch_size: int, device: torch.device) -> dict:
        if not self.supports_carry_state:
            raise RuntimeError(f"mode={self.mode!r} does not support carry-state")
        return {
            "layers": [
                layer.mamba.init_state(batch_size, device) for layer in self.layers
            ],
            "offset": 0,
        }

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        B, T, _ = x.shape
        offset = state["offset"] if state is not None else 0
        layer_states = state["layers"] if state is not None else None

        x = self.pos_enc(x, offset=offset)
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()

        new_layer_states: Optional[list] = [] if state is not None else None

        for i, layer in enumerate(self.layers):
            ls = layer_states[i] if layer_states is not None else None
            # Gradient-checkpoint only in eager mode without carry-state.
            # Disabled under torch.compile (incompatible) and when carrying state.
            if (
                self.training
                and state is None
                and not torch.compiler.is_compiling()
            ):
                x, ns = checkpoint(layer, x, ls, use_reentrant=False)
            else:
                x, ns = layer(x, state=ls)
            x = x * mask_f
            if new_layer_states is not None:
                new_layer_states.append(ns)

        new_state: Optional[dict] = None
        if new_layer_states is not None:
            new_state = {"layers": new_layer_states, "offset": offset + T}
        return x, new_state
