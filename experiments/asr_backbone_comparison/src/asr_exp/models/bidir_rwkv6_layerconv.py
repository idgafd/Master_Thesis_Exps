"""Bidirectional RWKV v6 encoder with Layer-Dependent ConvShift.

Based on bidir_rwkv6_conv.py (run-006's ConvShift, no gate) with one change:

**Layer-Dependent kernel size**: Lower layers use wider kernels (broader
temporal context), upper layers use narrower kernels (local context).
This directly encodes the most consistent empirical finding across runs
005–013: lower layers want broad attention, upper layers want local.

Kernel schedule:  layer 0 → kernel 7, layer 5 → kernel 3
(linear interpolation, always odd).

Everything else (LION full attention matrix, parameter layout, forward loop)
is identical to bidir_rwkv6_conv.py with use_gate=False.
supports_carry_state = False.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr_exp.models.bidir_rwkv6 import _lion_parallel_attention
from asr_exp.models.components import SinusoidalPE


class LayerDWConvShift(nn.Module):
    """Depthwise Conv1d token shift with layer-dependent kernel size.

    Lower layers (layer_idx=0) get wider kernels for broad context.
    Upper layers (layer_idx=n_layers-1) get narrow kernels for local context.

    Kernel schedule: k = max_kernel - (max_kernel - min_kernel) * (layer_idx / (n_layers - 1))
    Default: L0=7, L5=3 for a 6-layer model.

    Initialised so that left/right neighbours get equal weight and center
    weight is zero, generalising the run-006 [0.5, 0, 0.5] init to wider
    kernels (e.g. kernel=7: [1/6, 1/6, 1/6, 0, 1/6, 1/6, 1/6]).
    """

    def __init__(self, d_model: int, layer_idx: int, n_layers: int,
                 max_kernel: int = 7, min_kernel: int = 3):
        super().__init__()
        if n_layers > 1:
            ratio = layer_idx / (n_layers - 1)
        else:
            ratio = 0.0
        kernel_size = int(round(max_kernel - (max_kernel - min_kernel) * ratio))
        # Ensure odd kernel for symmetric padding
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model, bias=False,
        )
        # Init: equal weight on all non-center positions, zero on center
        # This generalises the [0.5, 0, 0.5] init from run-006
        nn.init.zeros_(self.conv.weight)
        n_neighbors = kernel_size - 1  # number of non-center taps
        if n_neighbors > 0:
            w = 1.0 / n_neighbors
            center = kernel_size // 2
            for i in range(kernel_size):
                if i != center:
                    self.conv.weight.data[:, 0, i] = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class BidirRWKV6LayerConvTimeMix(nn.Module):
    """Bidirectional RWKV6 time-mixing with LION attention + layer-dependent ConvShift."""

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        num_hidden_layers: int,
        layer_id: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = 8

        hidden_size_att = hidden_size

        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(num_hidden_layers - 1, 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)
            ddd = torch.ones(1, 1, hidden_size, dtype=dtype)
            for i in range(hidden_size):
                ddd[0, 0, i] = i / hidden_size

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            )
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_DIM = 32
            self.time_maa_w1 = nn.Parameter(
                torch.zeros(hidden_size, D_MIX_DIM * 5, dtype=dtype)
            )
            self.time_maa_w2 = nn.Parameter(
                torch.zeros(5, D_MIX_DIM, hidden_size, dtype=dtype).uniform_(-0.01, 0.01)
            )

            decay_speed = torch.ones(hidden_size_att, dtype=dtype)
            for n in range(hidden_size_att):
                decay_speed[n] = -6 + 5 * (n / (hidden_size_att - 1)) ** (
                    0.7 + 1.3 * ratio_0_to_1
                )
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, hidden_size_att))

            D_DECAY_DIM = 64
            self.time_decay_w1 = nn.Parameter(
                torch.zeros(hidden_size, D_DECAY_DIM, dtype=dtype)
            )
            self.time_decay_w2 = nn.Parameter(
                torch.zeros(D_DECAY_DIM, hidden_size_att, dtype=dtype).uniform_(-0.01, 0.01)
            )

        # Layer-dependent ConvShift (wider kernel at lower layers)
        self.conv_shift = LayerDWConvShift(hidden_size, layer_id, num_hidden_layers)

        self.receptance = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.gate = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.output = nn.Linear(hidden_size_att, hidden_size, bias=False, dtype=dtype)
        self.ln_x = nn.GroupNorm(
            n_head, hidden_size_att, dtype=dtype,
            eps=(1e-5) * (self.head_size_divisor ** 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        H = self.n_head
        K = self.head_size

        # Layer-dependent ConvShift residual
        x_conv = self.conv_shift(x)
        dxprev = x_conv - x

        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, D)

        mw, mk, mv, mr, mg = xxx.unbind(dim=0)
        xw = x + dxprev * (self.time_maa_w + mw)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xr = x + dxprev * (self.time_maa_r + mr)
        xg = x + dxprev * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = w.to(r.dtype)

        r_h = r.view(B, T, H, K).transpose(1, 2)
        k_h = k.view(B, T, H, K).transpose(1, 2)
        v_h = v.view(B, T, H, K).transpose(1, 2)
        w_h = w.view(B, T, H, K).transpose(1, 2)
        w_h = -torch.exp(w_h)

        y = _lion_parallel_attention(r_h, k_h, v_h, w_h).to(r.dtype)
        y = y.transpose(1, 2).reshape(B * T, D)

        y = self.ln_x(y).view(B, T, D)
        y = self.output(y * g)
        return y


class BidirRWKV6LayerConvChannelMix(nn.Module):
    """Bidirectional RWKV6 channel mix with layer-dependent ConvShift."""

    def __init__(self, hidden_size: int, num_hidden_layers: int, layer_id: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        hidden_size_ffn = int((hidden_size * 3.5) // 32 * 32)

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)
            ddd = torch.ones(1, 1, hidden_size, dtype=dtype)
            for i in range(hidden_size):
                ddd[0, 0, i] = i / hidden_size
            self.time_maa_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.conv_shift = LayerDWConvShift(hidden_size, layer_id, num_hidden_layers)

        self.key = nn.Linear(hidden_size, hidden_size_ffn, bias=False, dtype=dtype)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.value = nn.Linear(hidden_size_ffn, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dxprev = self.conv_shift(x) - x

        xk = x + dxprev * self.time_maa_k
        xr = x + dxprev * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class BidirRWKV6LayerConvBlock(nn.Module):
    """Single BidirRWKV6 layer block with layer-dependent ConvShift."""

    def __init__(self, hidden_size: int, n_head: int, head_size: int,
                 num_hidden_layers: int, layer_id: int, dropout: float,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln0 = nn.LayerNorm(hidden_size, dtype=dtype) if layer_id == 0 else nn.Identity()

        self.att = BidirRWKV6LayerConvTimeMix(
            hidden_size, n_head, head_size, num_hidden_layers, layer_id, dtype
        )
        self.ffn = BidirRWKV6LayerConvChannelMix(
            hidden_size, num_hidden_layers, layer_id, dtype
        )

        self.drop0 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.drop1 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln0(x)
        x = self.drop0(x + self.att(self.ln1(x)))
        x = self.drop1(x + self.ffn(self.ln2(x)))
        return x


class BidirRWKV6LayerConvEncoder(nn.Module):
    """Bidirectional RWKV v6 (LION) + layer-dependent ConvShift.

    Lower layers get wider convolutional kernels (broader temporal mixing),
    upper layers get narrower kernels (local mixing). No gate.

    d_model must be divisible by 32.
    supports_carry_state = False (full-sequence LION form).
    """

    supports_carry_state = False

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        head_size: int = 64,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert d_model % 32 == 0, f"d_model must be divisible by 32, got {d_model}"
        assert d_model % head_size == 0, (
            f"d_model ({d_model}) must be divisible by head_size ({head_size})"
        )

        self.d_model = d_model
        self.n_layers = n_layers
        n_head = d_model // head_size

        self.pos_enc = SinusoidalPE(d_model, max_len=8000)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                BidirRWKV6LayerConvBlock(
                    d_model, n_head, head_size, n_layers, i, dropout, dtype
                )
            )

        # Print kernel schedule for verification
        kernels = [layer.att.conv_shift.kernel_size for layer in self.layers]
        print(
            f"BidirRWKV6LayerConv encoder initialized "
            f"({n_layers} layers, d_model={d_model}, head_size={head_size}, "
            f"kernel_schedule={kernels})"
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state=None,
    ) -> Tuple[torch.Tensor, None]:
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()

        for layer in self.layers:
            x = layer(x) * mask_f

        return x, None
