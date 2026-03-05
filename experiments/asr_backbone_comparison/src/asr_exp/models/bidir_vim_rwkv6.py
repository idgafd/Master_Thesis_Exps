"""Bidirectional RWKV v6 encoder — Vim-style flip-and-sum approach.

Each layer has two separate RWKV6 blocks with independent parameters:
  - Forward block processes x in natural order
  - Backward block processes x.flip() (reversed sequence)
  - Outputs are summed (divided by 2) and projected

This is the same strategy as Vision Mamba (Vim) v2 (Zhu et al. 2024),
applied to RWKV-6 blocks instead of Mamba SSM blocks.

Unlike LION, there is no correction term — token i's self-contribution
is counted in both directions. The approach is heuristic but simple.

Not compatible with carry-state inference (requires full sequence).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from asr_exp.models.components import SinusoidalPE


class BidirVimRWKV6Encoder(nn.Module):
    """Bidirectional RWKV-6 encoder using Vim-style dual scan.

    Two independent RWKV6LayerBlock stacks: one forward, one backward.
    Requires RWKV-block installed (see setup_rwkv.sh).
    d_model must be divisible by 32.
    """

    supports_carry_state = False

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        head_size: int = 64,
        dtype: torch.dtype = torch.float32,
        tmix_backend: str = "auto",
    ):
        super().__init__()
        assert d_model % 32 == 0, f"d_model must be divisible by 32, got {d_model}"
        assert d_model % head_size == 0, (
            f"d_model ({d_model}) must be divisible by head_size ({head_size})"
        )

        from rwkv_block.v6_finch.block.rwkv6_block_config_map import RWKV6BlockConfigMap
        from rwkv_block.v6_finch.block.rwkv6_layer_block import RWKV6LayerBlock

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_head = d_model // head_size
        self.head_size = head_size

        self.pos_enc = SinusoidalPE(d_model, max_len=8000)

        self.layers_fwd = nn.ModuleList()
        self.layers_bwd = nn.ModuleList()
        for i in range(n_layers):
            cfg_fwd = RWKV6BlockConfigMap(
                num_hidden_layers=n_layers,
                hidden_size=d_model,
                head_size=head_size,
                dropout_rate=dropout,
                layer_id=i,
                tmix_backend=tmix_backend,
                dtype=dtype,
            )
            cfg_bwd = RWKV6BlockConfigMap(
                num_hidden_layers=n_layers,
                hidden_size=d_model,
                head_size=head_size,
                dropout_rate=dropout,
                layer_id=i,
                tmix_backend=tmix_backend,
                dtype=dtype,
            )
            self.layers_fwd.append(RWKV6LayerBlock(cfg_fwd))
            self.layers_bwd.append(RWKV6LayerBlock(cfg_bwd))

        print(
            f"Bidirectional RWKV-6 (Vim) encoder initialized "
            f"({n_layers} layers, d_model={d_model}, head_size={head_size})"
        )

    def _init_state(self, batch_size: int, device: torch.device):
        return [
            (
                torch.zeros(batch_size, self.d_model, device=device),
                torch.zeros(
                    batch_size, self.n_head, self.head_size, self.head_size,
                    dtype=torch.float32, device=device,
                ),
                torch.zeros(batch_size, self.d_model, device=device),
            )
            for _ in range(self.n_layers)
        ]

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state=None,
    ) -> Tuple[torch.Tensor, None]:
        x = self.pos_enc(x)
        B, T, _ = x.shape

        # Padding mask
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()  # (B, T, 1)

        state_fwd = self._init_state(B, x.device)
        state_bwd = self._init_state(B, x.device)

        x_fwd = x
        x_bwd = x.flip([1])  # reverse time dimension

        for i in range(self.n_layers):
            x_fwd, state_fwd[i] = self.layers_fwd[i](x_fwd, state_fwd[i])
            x_bwd, state_bwd[i] = self.layers_bwd[i](x_bwd, state_bwd[i])

        # Flip backward output back and combine
        x = (x_fwd + x_bwd.flip([1])) * 0.5

        # Zero out padding
        x = x * mask_f

        return x, None
