"""Bidirectional RWKV v6 with multi-scale decay — LION parallel form.

Two approaches to break the one-scale-per-dimension limitation of standard LION:

Exp E — Dual-decay:
    Each dimension gets two exponential decays mixed per-dimension:
        Y = α · LION(Q,K,V,w_fast) + (1-α) · LION(Q,K,V,w_slow)
    w_slow = w_fast * sigmoid(slow_scale_h)  per head
    α learned per dimension, initialized near 1 (baseline-dominant at start)
    Compute: 2× LION attention. Extra params: n_heads + d_model per layer.

Exp F — Per-head decay scaling:
    Multiplicatively scale each head's decay spectrum:
        w_h = w_h * exp(head_bias_h)
    head_bias_h > 0 → faster decay (more local)
    head_bias_h < 0 → slower decay (more global)
    Compute: identical to baseline. Extra params: n_heads per layer.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr_exp.models.bidir_rwkv6 import (
    _bidirectional_token_shift,
    _lion_parallel_attention,
    BidirRWKV6ChannelMix,
)
from asr_exp.models.components import SinusoidalPE


# ── Time-mixing with multi-scale decay ────────────────────────────────────────

class BidirRWKV6MultiScaleTimeMix(nn.Module):
    """Bidirectional RWKV6 time-mixing with multi-scale decay options.

    Modes:
      use_dual=False: per-head decay scaling only (Exp F). Zero extra compute.
      use_dual=True:  dual-decay with per-dim mixing (Exp E). 2× LION compute.
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        num_hidden_layers: int,
        layer_id: int,
        use_dual: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = 8
        self.use_dual = use_dual

        hidden_size_att = hidden_size

        # ── Per-head decay scaling (both modes) ──
        # Initialized to 0 → exp(0) = 1 → baseline behavior
        self.head_decay_bias = nn.Parameter(torch.zeros(n_head, dtype=dtype))

        # ── Dual-decay params (Exp E only) ──
        if use_dual:
            # slow_scale: how much to scale the decay for the slow component
            # sigmoid(-0.85) ≈ 0.30 → w_slow = w_fast * 0.30 (3× longer half-life)
            self.slow_scale = nn.Parameter(torch.full((n_head,), -0.85, dtype=dtype))
            # decay_mix: per-dimension mixing weight
            # sigmoid(2.94) ≈ 0.95 → 95% fast + 5% slow at init (near-baseline)
            self.decay_mix = nn.Parameter(torch.full((hidden_size_att,), 2.94, dtype=dtype))

        # ── Standard RWKV-6 params (identical to baseline) ──
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

        self.receptance = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.gate = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.output = nn.Linear(hidden_size_att, hidden_size, bias=False, dtype=dtype)
        self.ln_x = nn.GroupNorm(
            n_head, hidden_size_att, dtype=dtype,
            eps=(1e-5) * (self.head_size_divisor ** 2),
        )

    def get_scale_info(self) -> dict:
        """Return head-decay-bias and optional dual-decay info for logging."""
        info = {
            "head_bias": self.head_decay_bias.detach().cpu().tolist(),
        }
        if self.use_dual:
            info["slow_scale"] = torch.sigmoid(self.slow_scale).detach().cpu().tolist()
            alpha = torch.sigmoid(self.decay_mix).detach().cpu()
            info["alpha_mean"] = float(alpha.mean())
            info["alpha_min"] = float(alpha.min())
            info["alpha_max"] = float(alpha.max())
        return info

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        H = self.n_head
        K = self.head_size

        dxprev = _bidirectional_token_shift(x) - x

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

        # Log-decay with per-head multiplicative scaling:
        #   w_scaled = w * exp(bias): bias>0 → faster, bias<0 → slower
        w_h = -torch.exp(w_h)
        scale = torch.exp(self.head_decay_bias).view(1, H, 1, 1)
        w_fast = w_h * scale

        if self.use_dual:
            # Slow component: w_slow = w_fast * sigmoid(slow_scale), per head
            # sigmoid in (0,1) → w_slow is less negative → slower decay
            s = torch.sigmoid(self.slow_scale).view(1, H, 1, 1)
            w_slow = w_fast * s

            y_fast = _lion_parallel_attention(r_h, k_h, v_h, w_fast)
            y_slow = _lion_parallel_attention(r_h, k_h, v_h, w_slow)

            # Per-dimension mixing: α → [1, H, 1, K]
            alpha = torch.sigmoid(self.decay_mix).view(1, H, 1, K)
            y = alpha * y_fast + (1.0 - alpha) * y_slow
        else:
            y = _lion_parallel_attention(r_h, k_h, v_h, w_fast)

        y = y.to(r.dtype)
        y = y.transpose(1, 2).reshape(B * T, D)
        y = self.ln_x(y).view(B, T, D)
        y = self.output(y * g)
        return y


# ── Block and Encoder ─────────────────────────────────────────────────────────

class BidirRWKV6MultiScaleBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        num_hidden_layers: int,
        layer_id: int,
        dropout: float,
        use_dual: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln0 = nn.LayerNorm(hidden_size, dtype=dtype) if layer_id == 0 else nn.Identity()

        self.att = BidirRWKV6MultiScaleTimeMix(
            hidden_size, n_head, head_size, num_hidden_layers, layer_id,
            use_dual, dtype
        )
        self.ffn = BidirRWKV6ChannelMix(hidden_size, num_hidden_layers, layer_id, dtype)

        self.drop0 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.drop1 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln0(x)
        x = self.drop0(x + self.att(self.ln1(x)))
        x = self.drop1(x + self.ffn(self.ln2(x)))
        return x


class BidirRWKV6MultiScaleEncoder(nn.Module):
    """Bidirectional RWKV v6 encoder with multi-scale decay.

    Exp F (headscale): per-head decay scaling. Zero extra compute, n_heads params/layer.
    Exp E (dual):      dual-decay with mixing. 2× LION compute, (n_heads + d_model) params/layer.

    Both start at baseline behavior (bias=0, alpha≈1) and learn deviations.
    """

    supports_carry_state = False

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        head_size: int = 64,
        use_dual: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert d_model % 32 == 0
        assert d_model % head_size == 0

        self.d_model = d_model
        self.n_layers = n_layers
        self.use_dual = use_dual
        n_head = d_model // head_size

        extra = n_head * n_layers
        if use_dual:
            extra += (n_head + d_model) * n_layers
        mode = "dual-decay" if use_dual else "headscale"

        self.pos_enc = SinusoidalPE(d_model, max_len=8000)
        self.layers = nn.ModuleList([
            BidirRWKV6MultiScaleBlock(
                d_model, n_head, head_size, n_layers, i, dropout,
                use_dual, dtype
            )
            for i in range(n_layers)
        ])

        print(
            f"Bidirectional RWKV-6 MULTISCALE [{mode}] encoder initialized "
            f"({n_layers} layers, d_model={d_model}, head_size={head_size}, "
            f"extra_params={extra})"
        )

    def get_theta_info(self) -> Optional[str]:
        """Return per-layer multi-scale info for logging."""
        parts = []
        for i, layer in enumerate(self.layers):
            info = layer.att.get_scale_info()
            bias_str = ",".join(f"{b:+.3f}" for b in info["head_bias"])
            s = f"L{i}:bias=[{bias_str}]"
            if self.use_dual:
                ss = ",".join(f"{v:.2f}" for v in info["slow_scale"])
                s += f" ss=[{ss}] α={info['alpha_mean']:.3f}[{info['alpha_min']:.2f},{info['alpha_max']:.2f}]"
            parts.append(s)
        return "  scale: " + " | ".join(parts)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state=None,
    ) -> Tuple[torch.Tensor, None]:
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask_f = (
            (torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))
            .unsqueeze(-1).float()
        )
        for layer in self.layers:
            x = layer(x) * mask_f
        return x, None
