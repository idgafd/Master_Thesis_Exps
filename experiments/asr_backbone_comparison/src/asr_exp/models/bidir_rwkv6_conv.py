"""Bidirectional RWKV v6 encoder with learned ConvShift token mixing.

Based on bidir_rwkv6.py (LION parallel form) with two changes motivated by
AudioRWKV (Xiong et al. 2025):

1. **ConvShift** (AudioRWKV §3.1): The fixed bidirectional token shift
   `(x[t-1] + x[t+1]) / 2 - x[t]` is replaced by a learned depthwise
   Conv1d (kernel=3, per-channel) over the time axis.  This is a temporal
   approximation of AudioRWKV's 2D DWConv: in their work the 2D grid
   (freq × time) is still explicit, while here ConvSubsampling has already
   collapsed frequency into d_model.  The learned kernel can still capture
   asymmetric temporal context (e.g. weight past vs future differently per
   channel) and acts as a soft local attention over the 3-frame window.
   Initialised to reproduce the original symmetric shift [0.5, 0, 0.5].

2. **xres-conditioned gate** (AudioRWKV §3.2, optional, enabled by default):
   The ConvShift residual `xres = conv_shift(x) - x` summarises local
   context deviation.  We use it to derive a per-position sigmoid gate that
   modulates the time-mix output, analogous to AudioRWKV's fusion gate G.
   In the LION form there is no separate forward/backward output stream to
   fuse, so the gate instead scales the combined bidirectional output before
   the GroupNorm step.  This lets the model learn when to trust vs suppress
   the bidirectional recurrence based on local context.

Everything else (LION full attention matrix, parameter layout, forward loop)
is identical to bidir_rwkv6.py.  Supports_carry_state = False (same as
bidir_rwkv6).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr_exp.models.bidir_rwkv6 import _lion_parallel_attention
from asr_exp.models.components import SinusoidalPE


class DWConvShift(nn.Module):
    """Depthwise Conv1d (kernel=3) token shift over the time axis.

    Replaces the fixed bidirectional token shift `(x[t-1] + x[t+1]) / 2`.
    Operates on `(B, T, D)` — transposed internally for Conv1d convention.

    Initialised with weight [0.5, 0, 0.5] per channel so that on epoch-0
    the behaviour is identical to the fixed symmetric shift.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False
        )
        # Init to reproduce (x[t-1] + x[t+1]) / 2
        nn.init.zeros_(self.conv.weight)
        self.conv.weight.data[:, 0, 0] = 0.5  # left neighbour
        self.conv.weight.data[:, 0, 2] = 0.5  # right neighbour

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        # Conv1d expects (B, D, T)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class BidirRWKV6ConvTimeMix(nn.Module):
    """Bidirectional RWKV6 time-mixing with LION parallel attention + ConvShift.

    Identical to BidirRWKV6TimeMix except:
    - token shift is a learned DWConv1d (DWConvShift) instead of fixed average
    - optional xres-conditioned gate on the output (use_gate=True)
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        num_hidden_layers: int,
        layer_id: int,
        dtype: torch.dtype = torch.float32,
        use_gate: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = 8
        self.use_gate = use_gate

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

        # Learned ConvShift (replaces fixed bidirectional token shift)
        self.conv_shift = DWConvShift(hidden_size)

        self.receptance = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.gate = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.output = nn.Linear(hidden_size_att, hidden_size, bias=False, dtype=dtype)
        self.ln_x = nn.GroupNorm(
            n_head, hidden_size_att, dtype=dtype,
            eps=(1e-5) * (self.head_size_divisor ** 2),
        )

        # xres-conditioned gate (AudioRWKV-style): gates bidirectional output
        # using the local context residual from the ConvShift.
        # Initialised near 1 (sigmoid(3) ≈ 0.95) so early training is stable.
        if use_gate:
            self.xres_gate = nn.Linear(hidden_size, hidden_size_att, bias=True, dtype=dtype)
            nn.init.zeros_(self.xres_gate.weight)
            nn.init.constant_(self.xres_gate.bias, 3.0)   # sigmoid(3) ≈ 0.95

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        H = self.n_head
        K = self.head_size

        # Learned ConvShift residual: deviation of local conv context from x
        x_conv = self.conv_shift(x)     # (B, T, D) — learned 3-tap per-channel filter
        dxprev = x_conv - x             # deviation from identity (= shift residual)

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

        # Apply xres-conditioned gate before GroupNorm (AudioRWKV §3.2 analogue)
        if self.use_gate:
            gate_val = torch.sigmoid(self.xres_gate(dxprev))   # (B, T, D)
            y = y * gate_val.reshape(B * T, D)

        y = self.ln_x(y).view(B, T, D)
        y = self.output(y * g)
        return y


class BidirRWKV6ConvChannelMix(nn.Module):
    """Bidirectional RWKV6 channel mix with learned ConvShift."""

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

        self.conv_shift = DWConvShift(hidden_size)

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


class BidirRWKV6ConvBlock(nn.Module):
    """Single BidirRWKV6 layer block with ConvShift and optional xres gate."""

    def __init__(self, hidden_size: int, n_head: int, head_size: int,
                 num_hidden_layers: int, layer_id: int, dropout: float,
                 dtype: torch.dtype = torch.float32, use_gate: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln0 = nn.LayerNorm(hidden_size, dtype=dtype) if layer_id == 0 else nn.Identity()

        self.att = BidirRWKV6ConvTimeMix(
            hidden_size, n_head, head_size, num_hidden_layers, layer_id, dtype, use_gate
        )
        self.ffn = BidirRWKV6ConvChannelMix(
            hidden_size, num_hidden_layers, layer_id, dtype
        )

        self.drop0 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.drop1 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln0(x)
        x = self.drop0(x + self.att(self.ln1(x)))
        x = self.drop1(x + self.ffn(self.ln2(x)))
        return x


class BidirRWKV6ConvEncoder(nn.Module):
    """Bidirectional RWKV v6 (LION) + learned ConvShift + xres gate.

    Ablation variants (controlled by constructor flags):
      - use_conv_shift=True,  use_gate=True  → full AudioRWKV-inspired model (default)
      - use_conv_shift=True,  use_gate=False → ConvShift only
      - use_conv_shift=False, use_gate=True  → gate only (not very useful alone)

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
        use_gate: bool = True,
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
                BidirRWKV6ConvBlock(
                    d_model, n_head, head_size, n_layers, i, dropout, dtype, use_gate
                )
            )

        print(
            f"BidirRWKV6Conv encoder initialized "
            f"({n_layers} layers, d_model={d_model}, head_size={head_size}, "
            f"use_gate={use_gate})"
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
