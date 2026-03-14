"""RWKV v7 (Goose) encoder with initialization fixes for small-model CTC-ASR.

Problems identified in stock RWKV-7 at d_model=256 / 6 layers:

1. Decay init too aggressive (triton backend):
   Stock w0 ∈ [-6.5, -1.5] → exp(-softplus(-w0)-0.5) ∈ [0.0009, 0.111].
   Effective memory ≈ 1 token. RWKV-6 starts at [0.0025, 0.368].
   Fix: shift w0 by +2.0 → decay ∈ [0.007, 0.378], matching RWKV-6.

2. Value residual (v_first) too strong:
   v0 init = 1.0 → sigmoid(1.0) = 0.73 → layers 1-5 copy 73% of layer 0 value.
   Suppresses hierarchical feature learning needed by CTC.
   Fix: set v0 = -5.0 → sigmoid(-5) ≈ 0.007 → effectively disabled.

3. Key scaling halved by in-context learning rate:
   iclr = sigmoid(0) = 0.5, k_a = 1.0 → k *= (1 + (0.5-1)*1) = k * 0.5.
   Fix: set k_a = 0.0 → k *= 1.0 (no scaling at init).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class RWKV7FixedEncoder(nn.Module):
    """RWKV-7 encoder with configurable initialization fixes.

    Args:
        fix_decay:   Shift w0 by +2.0 to get RWKV-6-like decay range.
        fix_vfirst:  Set v0 = -5.0 to disable value residual at init.
        fix_ka:      Set k_a = 0.0 to prevent key magnitude halving.
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
        fix_decay: bool = True,
        fix_vfirst: bool = False,
        fix_ka: bool = False,
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

        # Apply fixes AFTER reset_parameters so we override the stock init
        self._apply_fixes(fix_decay, fix_vfirst, fix_ka)

        fixes = []
        if fix_decay:
            fixes.append("decay(w0+2.0)")
        if fix_vfirst:
            fixes.append("vfirst(v0=-5)")
        if fix_ka:
            fixes.append("ka(k_a=0)")
        fix_str = ", ".join(fixes) if fixes else "none"
        print(
            f"RWKV-7 FIXED encoder initialized "
            f"({n_layers} layers, d_model={d_model}, head_size={head_size}, "
            f"fixes=[{fix_str}])"
        )

    def _apply_fixes(self, fix_decay: bool, fix_vfirst: bool, fix_ka: bool):
        with torch.no_grad():
            for layer in self.layers:
                att = layer.att

                if fix_decay:
                    # Shift w0 by +2.0 to match RWKV-6 decay range under triton
                    # Stock: w0 ∈ [-6.5, -1.5] → triton decay [0.0009, 0.111]
                    # Fixed: w0 ∈ [-4.5,  0.5] → triton decay [0.007,  0.378]
                    att.w0.add_(2.0)

                if fix_vfirst and hasattr(att, "v0"):
                    # Disable value residual: sigmoid(-5) ≈ 0.007
                    att.v0.fill_(-5.0)

                if fix_ka:
                    # Prevent key halving: k_a = 0 → k *= (1 + (iclr-1)*0) = k
                    att.k_a.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        state: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
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
        v_first = None
        for i, layer in enumerate(self.layers):
            x, new_layer_state, v_first = layer(x, state[i], v_first)
            new_state.append(new_layer_state)
        return x, new_state

    def init_state(self, batch_size: int, device: torch.device) -> List:
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
