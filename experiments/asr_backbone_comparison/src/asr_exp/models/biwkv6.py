"""Bi-WKV RWKV-6 encoder — AudioRWKV-style bidirectional architecture.

Implements AudioRWKV's Bi-WKV (Xiong et al. 2025 §3.2) using RWKV-6 blocks.

Per-layer computation:
  1. Forward RWKV-6 block   (causal, left→right):   p_fwd = block_fwd(x)
  2. Backward RWKV-6 block  (causal on flipped seq): p_bwd = flip(block_bwd(flip(x)))
  3a. With ConvShift + gate (use_conv_shift=True, use_gate=True):
        xres   = DWConv1d(x) − x            local context residual
        G      = sigmoid(W_gate(xres))       per-position, per-channel gate
        x_next = G ⊙ p_fwd + (1−G) ⊙ p_bwd
  3b. Plain BiLSTM average (use_conv_shift=False, use_gate=False):
        x_next = 0.5 * (p_fwd + p_bwd)

IMPORTANT — why ConvShift and gate are inseparable in Bi-WKV:
  The RWKV-6 library blocks have their own internal token shift.  Unlike in
  bidir_rwkv6_conv (where ConvShift REPLACES the internal shift), here
  ConvShift sits outside the blocks and produces xres solely to condition
  the fusion gate.  If the gate is disabled, xres is unused — turning
  ConvShift on without the gate is identical to having neither.  Therefore
  the only meaningful Bi-WKV ablations are:
    biwkv6_no_conv_no_gate  (use_conv_shift=False, use_gate=False)
    biwkv6                  (use_conv_shift=True,  use_gate=True)

Key difference from LION (bidir_rwkv6):
  - LION:   O(T²K) full attention matrix, no carry state.
  - Bi-WKV: O(T) two sequential RNN passes per layer, supports carry state.

Inference / carry-state:
  When state is not None only the FORWARD blocks run — backward pass is
  skipped entirely, giving standard causal RWKV-6 streaming inference.
  supports_carry_state = True.

Parameter count:
  n_layers effective bidir layers = 2*n_layers RWKV-6 blocks.
  Use n_layers=3 to match a 6-layer LION model (≈7.74 M params).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from asr_exp.models.bidir_rwkv6_conv import DWConvShift
from asr_exp.models.components import SinusoidalPE


def _make_rwkv6_block(d_model: int, n_layers: int, layer_id: int,
                       head_size: int, dropout: float, dtype: torch.dtype):
    """Instantiate a single RWKV6LayerBlock from the rwkv_block library."""
    from rwkv_block.v6_finch.block.rwkv6_block_config_map import RWKV6BlockConfigMap
    from rwkv_block.v6_finch.block.rwkv6_layer_block import RWKV6LayerBlock

    cfg = RWKV6BlockConfigMap(
        num_hidden_layers=n_layers,
        hidden_size=d_model,
        head_size=head_size,
        dropout_rate=dropout,
        layer_id=layer_id,
        tmix_backend="auto",
        dtype=dtype,
    )
    return RWKV6LayerBlock(cfg)


class BiWKV6Encoder(nn.Module):
    """AudioRWKV-style Bi-WKV using RWKV-6 blocks.

    n_layers: number of effective bidirectional layers.
              Each layer contains 2 RWKV-6 blocks (fwd + bwd).
              Use n_layers = target_param_budget_layers // 2
              (e.g. n_layers=3 to match a 6-layer LION model).

    use_gate: True  → AudioRWKV learned gate (G from xres).
              False → fixed 0.5/0.5 average (BiLSTM baseline).

    supports_carry_state = True:
      In carry-state mode only the forward blocks run.
      The backward blocks are skipped (no bidirectionality at inference).
    """

    supports_carry_state = True

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        head_size: int = 64,
        dtype: torch.dtype = torch.float32,
        use_conv_shift: bool = True,
        use_gate: bool = True,
    ):
        super().__init__()
        assert d_model % 32 == 0, f"d_model must be divisible by 32, got {d_model}"
        assert d_model % head_size == 0
        assert not (use_gate and not use_conv_shift), (
            "use_gate=True requires use_conv_shift=True (gate is conditioned on xres)"
        )

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_head = d_model // head_size
        self.head_size = head_size
        self.use_conv_shift = use_conv_shift
        self.use_gate = use_gate

        # Two RWKV-6 blocks per effective layer (numbered 0..2*n_layers-1 for
        # correct layer-dependent decay initialisation).
        n_rwkv_blocks = 2 * n_layers
        self.layers_fwd = nn.ModuleList([
            _make_rwkv6_block(d_model, n_rwkv_blocks, 2 * i, head_size, dropout, dtype)
            for i in range(n_layers)
        ])
        self.layers_bwd = nn.ModuleList([
            _make_rwkv6_block(d_model, n_rwkv_blocks, 2 * i + 1, head_size, dropout, dtype)
            for i in range(n_layers)
        ])

        # ConvShift + gate (coupled: gate needs xres from conv shift)
        if use_conv_shift:
            self.conv_shifts = nn.ModuleList([DWConvShift(d_model) for _ in range(n_layers)])
        if use_gate:
            self.gates = nn.ModuleList([
                nn.Linear(d_model, d_model, bias=True, dtype=dtype)
                for _ in range(n_layers)
            ])
            for gate in self.gates:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)   # sigmoid(0) = 0.5 → symmetric start

        self.pos_enc = SinusoidalPE(d_model, max_len=8000)

        mode = "ConvShift+gate" if use_gate else "BiLSTM avg"
        print(
            f"Bi-WKV RWKV-6 encoder initialized "
            f"({n_layers} bidir layers = {n_rwkv_blocks} RWKV-6 blocks, "
            f"d_model={d_model}, head_size={head_size}, mode={mode})"
        )

    # ── state helpers ──────────────────────────────────────────────────────────

    def init_state(self, batch_size: int, device: torch.device) -> List:
        """Per-layer forward state: (tmix_shift, wkv, cmix_shift)."""
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

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask_f = (
            (torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))
            .unsqueeze(-1).float()
        )

        if state is not None:
            return self._forward_carry(x, state, mask_f)
        return self._forward_bidir(x, mask_f)

    def _forward_bidir(
        self, x: torch.Tensor, mask_f: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """Full bidirectional training pass: fwd + bwd + gate per layer."""
        B = x.shape[0]
        state_fwd = self.init_state(B, x.device)
        state_bwd = self.init_state(B, x.device)

        for i in range(self.n_layers):
            # Forward pass (causal, left-to-right)
            p_fwd, state_fwd[i] = self.layers_fwd[i](x, state_fwd[i])

            # Backward pass (causal on flipped sequence = right-to-left)
            p_bwd_flipped, state_bwd[i] = self.layers_bwd[i](x.flip(1), state_bwd[i])
            p_bwd = p_bwd_flipped.flip(1)

            if self.use_gate:
                # ConvShift residual conditions the fusion gate
                xres = self.conv_shifts[i](x) - x       # (B, T, D)
                G = torch.sigmoid(self.gates[i](xres))   # per-position, per-channel
                x = G * p_fwd + (1.0 - G) * p_bwd
            else:
                x = 0.5 * (p_fwd + p_bwd)

            x = x * mask_f

        return x, None

    def _forward_carry(
        self, x: torch.Tensor, state: List, mask_f: torch.Tensor
    ) -> Tuple[torch.Tensor, List]:
        """Carry-state inference: forward blocks only (causal, streaming-safe)."""
        new_state = []
        for i in range(self.n_layers):
            x, new_s = self.layers_fwd[i](x, state[i])
            new_state.append(new_s)
            x = x * mask_f
        return x, new_state
