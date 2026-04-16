"""RWKV-6 layer block: ln0 (layer 0 only) -> ln1 -> TimeMix -> drop -> ln2 -> ChannelMix -> drop."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.rwkv6_time_mix import RWKV6TimeMix
from src.models.rwkv6_channel_mix import RWKV6ChannelMix


class RWKV6Block(nn.Module):
    """Single RWKV-6 encoder layer block."""

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        num_hidden_layers: int,
        layer_id: int,
        dropout: float,
        mode: str = "lion",
        conv_shift: bool = False,
        headscale: bool = False,
        delta_rule: bool = False,
        lucid: bool = False,
        lucid_chunk_size: Optional[int] = None,
        lucid_self_reg: bool = False,
        temperature: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(hidden_size, dtype=dtype)
        else:
            self.ln0 = nn.Identity()

        self.att = RWKV6TimeMix(
            hidden_size=hidden_size,
            n_head=n_head,
            head_size=head_size,
            num_hidden_layers=num_hidden_layers,
            layer_id=layer_id,
            mode=mode,
            conv_shift=conv_shift,
            headscale=headscale,
            delta_rule=delta_rule,
            lucid=lucid,
            lucid_chunk_size=lucid_chunk_size,
            lucid_self_reg=lucid_self_reg,
            temperature=temperature,
            dtype=dtype,
        )
        self.ffn = RWKV6ChannelMix(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            layer_id=layer_id,
            mode=mode,
            dtype=dtype,
        )

        if dropout > 0.0:
            self.drop0 = nn.Dropout(p=dropout)
            self.drop1 = nn.Dropout(p=dropout)
        else:
            self.drop0 = nn.Identity()
            self.drop1 = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.ln0(x)
        att_out, new_state = self.att(self.ln1(x), state=state)
        x = self.drop0(x + att_out)
        x = self.drop1(x + self.ffn(self.ln2(x)))
        return x, new_state
