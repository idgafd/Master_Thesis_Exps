"""RWKV v6 (Finch) encoder backbone."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class RWKV6Encoder(nn.Module):
    """RWKV v6 (Finch) blocks as a sequence encoder.

    Requires RWKV-block installed from the patched local clone (see setup_rwkv.sh).
    d_model must be divisible by 32.
    WKV state is always kept in float32 regardless of weight dtype.
    """

    supports_carry_state = True

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

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            cfg = RWKV6BlockConfigMap(
                num_hidden_layers=n_layers,
                hidden_size=d_model,
                head_size=head_size,
                dropout_rate=dropout,
                layer_id=i,
                tmix_backend=tmix_backend,
                dtype=dtype,
            )
            self.layers.append(RWKV6LayerBlock(cfg))

        print(
            f"RWKV-6 encoder initialized "
            f"({n_layers} layers, d_model={d_model}, head_size={head_size})"
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        state: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        if state is not None:
            return self._forward_carry(x, state)
        return self._forward_stateless(x)

    def _forward_stateless(self, x: torch.Tensor):
        state = self.init_state(x.shape[0], x.device)
        for i, layer in enumerate(self.layers):
            x, state[i] = layer(x, state[i])
        return x, None

    def _forward_carry(self, x: torch.Tensor, state: List):
        new_state = []
        for i, layer in enumerate(self.layers):
            x, new_layer_state = layer(x, state[i])
            new_state.append(new_layer_state)
        return x, new_state

    def init_state(self, batch_size: int, device: torch.device) -> List:
        """Per-layer state: (tmix_shift, wkv, cmix_shift)."""
        return [
            (
                torch.zeros(batch_size, self.d_model, device=device),
                torch.zeros(
                    batch_size,
                    self.n_head,
                    self.head_size,
                    self.head_size,
                    dtype=torch.float32,
                    device=device,
                ),
                torch.zeros(batch_size, self.d_model, device=device),
            )
            for _ in range(self.n_layers)
        ]
