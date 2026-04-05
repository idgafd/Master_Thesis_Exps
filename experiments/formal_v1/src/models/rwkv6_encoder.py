"""RWKV-6 encoder — wraps N RWKV6Blocks with positional encoding and masking."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.components import SinusoidalPE
from src.models.rwkv6_block import RWKV6Block


class RWKV6Encoder(nn.Module):
    """RWKV-6 encoder supporting recurrent, LION, and bidir_serial modes.

    All configuration is passed through to RWKV6Block/TimeMix.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        head_size: int = 64,
        mode: str = "lion",
        conv_shift: bool = False,
        headscale: bool = False,
        delta_rule: bool = False,
        lucid: bool = False,
        lucid_chunk_size: Optional[int] = None,
        temperature: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert d_model % 32 == 0
        assert d_model % head_size == 0

        self.d_model = d_model
        self.n_layers = n_layers
        self.mode = mode
        n_head = d_model // head_size

        self.pos_enc = SinusoidalPE(d_model, max_len=8000, dropout=dropout)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                RWKV6Block(
                    hidden_size=d_model,
                    n_head=n_head,
                    head_size=head_size,
                    num_hidden_layers=n_layers,
                    layer_id=i,
                    dropout=dropout,
                    mode=mode,
                    conv_shift=conv_shift,
                    headscale=headscale,
                    delta_rule=delta_rule,
                    lucid=lucid,
                    lucid_chunk_size=lucid_chunk_size,
                    temperature=temperature,
                    dtype=dtype,
                )
            )

    @property
    def supports_carry_state(self) -> bool:
        return self.mode == "recurrent"

    def init_state(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initialize per-layer WKV states for carry-state inference."""
        n_head = self.d_model // (self.d_model // len(self.layers[0].att.receptance.weight))
        # Actually compute from first layer
        H = self.layers[0].att.n_head
        K = self.layers[0].att.head_size
        return [
            torch.zeros(batch_size, H, K, K, dtype=torch.float32, device=device)
            for _ in range(self.n_layers)
        ]

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
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
