"""RWKV v7 (Goose) encoder backbone."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class RWKV7Encoder(nn.Module):
    """RWKV v7 (Goose) blocks as a sequence encoder.

    Requires RWKV-block installed from the patched local clone (see setup_rwkv.sh).
    d_model must be divisible by 32.

    Note: v7 passes a value-residual tensor (v_first) from layer 0 to all
    deeper layers. Its shape is [B, T, d_model] — tied to the chunk length —
    so it resets every forward call and is NOT part of carry-state.

    Note: RWKV7 blocks internally use `torch.device` context managers that
    temporarily set the default device to CUDA. This leaks into the DataLoader
    sampler's `torch.randperm` between batches. We save/restore the default
    device around every forward call to fix this.
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

        from rwkv_block.v7_goose.block.rwkv7_block_config_map import RWKV7BlockConfigMap
        from rwkv_block.v7_goose.block.rwkv7_layer_block import RWKV7LayerBlock

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_head = d_model // head_size
        self.head_size = head_size

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            cfg = RWKV7BlockConfigMap(
                num_hidden_layers=n_layers,
                hidden_size=d_model,
                head_size=head_size,
                dropout_rate=dropout,
                layer_id=i,
                tmix_backend=tmix_backend,
                dtype=dtype,
            )
            block = RWKV7LayerBlock(cfg)
            block.reset_parameters()
            self.layers.append(block)

        print(
            f"RWKV-7 encoder initialized "
            f"({n_layers} layers, d_model={d_model}, head_size={head_size})"
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        state: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        # Save and restore default device to prevent CUDA device leak into
        # DataLoader's torch.randperm calls between batches.
        prev_default = torch.get_default_device()
        try:
            if state is not None:
                return self._forward_carry(x, state)
            return self._forward_stateless(x)
        finally:
            torch.set_default_device(prev_default)

    def _forward_stateless(self, x: torch.Tensor):
        state = self.init_state(x.shape[0], x.device)
        v_first = None
        for i, layer in enumerate(self.layers):
            x, state[i], v_first = layer(x, state[i], v_first)
        return x, None

    def _forward_carry(self, x: torch.Tensor, state: List):
        new_state = []
        v_first = None  # v_first resets each chunk (seq-length-dependent, not recurrent)
        for i, layer in enumerate(self.layers):
            x, new_layer_state, v_first = layer(x, state[i], v_first)
            new_state.append(new_layer_state)
        return x, new_state

    def init_state(self, batch_size: int, device: torch.device) -> List:
        """Per-layer state: (tmix_shift, wkv, cmix_shift) — same layout as v6."""
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
