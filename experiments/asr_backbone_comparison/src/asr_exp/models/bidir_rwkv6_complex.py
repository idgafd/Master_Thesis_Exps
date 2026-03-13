"""Bidirectional RWKV v6 with complex-valued decay poles — LION parallel form.

Extends the real diagonal decay of standard LION bidirectional attention with
complex poles: lambda_k = r_k * exp(i * theta), creating an oscillating
(non-monotone) locality bias.

Variants implemented:
  Config B  (theta=0.31 fixed):  M[i,j] = r^|i-j| * cos(0.31*|i-j|)   — syllable ~203ms
  Config C  (theta=0.90 fixed):  M[i,j] = r^|i-j| * cos(0.90*|i-j|)   — phoneme  ~70ms
  Config B-cos2 (theta=0.31):    M[i,j] = r^|i-j| * cos²(0.31*|i-j|)  — always non-negative
  Config D  (theta learnable):   theta_l = sigmoid(a_theta_l)*pi per layer, init 0.31
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr_exp.models.bidir_rwkv6 import (
    _bidirectional_token_shift,
    BidirRWKV6ChannelMix,
    _lion_parallel_attention,
)
from asr_exp.models.components import SinusoidalPE


# ── Attention kernels ─────────────────────────────────────────────────────────

def _lion_complex_parallel_attention(
    r: torch.Tensor,   # (B, H, T, K)
    k: torch.Tensor,   # (B, H, T, K)
    v: torch.Tensor,   # (B, H, T, K)
    w: torch.Tensor,   # (B, H, T, K) — log-space magnitude decay (negative)
    theta: float,      # phase angle per step (radians)
) -> torch.Tensor:
    """LION parallel full attention with complex decay poles.

    Effective mask: M[i,j] = r^|i-j| * cos(theta * |i-j|)

    Can be negative — see _lion_cos2_parallel_attention for the non-negative variant.
    Computed via cos(a-b) = cos(a)cos(b) + sin(a)sin(b) → 2 real matmuls per direction.

    Returns: (B, H, T, K) float32.
    """
    B, H, T, K = r.shape

    r = r.float()
    k = k.float()
    v = v.float()
    w = w.float()

    # Magnitude prefix sums (identical to real baseline)
    cs = torch.cumsum(w, dim=2)
    cs_b = cs - w

    shift_f = cs[:, :, T // 2 : T // 2 + 1, :]
    shift_b = cs_b[:, :, T // 2 : T // 2 + 1, :]
    cs_f_s = (cs - shift_f).clamp(-60, 60)
    cs_b_s = (cs_b - shift_b).clamp(-60, 60)

    # Phase prefix sums — constant theta → linear in position
    # theta_F[t] = theta*(t+1),  theta_B[t] = theta*t
    t_idx = torch.arange(T, device=r.device, dtype=torch.float32)
    theta_F = (theta * (t_idx + 1)).view(1, 1, T, 1)
    theta_B = (theta * t_idx).view(1, 1, T, 1)

    cos_F, sin_F = torch.cos(theta_F), torch.sin(theta_F)
    cos_B, sin_B = torch.cos(theta_B), torch.sin(theta_B)

    # Forward: Re(Q*L^F @ K/L^F)^T = (Q*exp(cs)*cos) @ (K*exp(-cs)*cos)^T
    #                               + (Q*exp(cs)*sin) @ (K*exp(-cs)*sin)^T
    exp_f  = torch.exp(cs_f_s)
    exp_nf = torch.exp(-cs_f_s)

    A_fwd = torch.tril(
        (r * exp_f * cos_F) @ (k * exp_nf * cos_F).transpose(-2, -1)
        + (r * exp_f * sin_F) @ (k * exp_nf * sin_F).transpose(-2, -1)
    )

    # Backward: same identity with cs_b, diagonal=1 (strict upper tri)
    exp_b  = torch.exp(cs_b_s)
    exp_nb = torch.exp(-cs_b_s)

    A_bwd = torch.triu(
        (r * exp_nb * cos_B) @ (k * exp_b * cos_B).transpose(-2, -1)
        + (r * exp_nb * sin_B) @ (k * exp_b * sin_B).transpose(-2, -1),
        diagonal=1,
    )

    return (A_fwd + A_bwd) @ v


def _lion_cos2_parallel_attention(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    """LION parallel attention with cos²-decay — always non-negative mask.

    Effective mask: M[i,j] = r^|i-j| * cos²(theta * |i-j|)

    Uses the identity  cos²(x) = (1 + cos(2x)) / 2:
        A_cos2 = 0.5 * (A_real + A_complex(2*theta))

    where A_real is standard LION (no oscillation) and A_complex uses 2*theta.
    This guarantees M[i,j] ≥ 0 everywhere — no forced anti-attention.

    Note: effective resonance period is pi/theta (half that of the cos version
    with the same theta).

    Returns: (B, H, T, K) float32.
    """
    y_real = _lion_parallel_attention(r, k, v, w)
    y_cplx = _lion_complex_parallel_attention(r, k, v, w, 2.0 * theta)
    return 0.5 * (y_real + y_cplx)


# ── Time-mixing block ─────────────────────────────────────────────────────────

class BidirRWKV6ComplexTimeMix(nn.Module):
    """Bidirectional RWKV6 time-mixing with complex or cos²-decay poles.

    Supports:
      learnable_theta=False: fixed scalar theta (Configs B, C, B-cos2)
      learnable_theta=True:  theta_l = sigmoid(a_theta_l)*pi per layer (Config D)
      use_cos2=False: cos decay (can go negative)
      use_cos2=True:  cos² decay (always non-negative)
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        num_hidden_layers: int,
        layer_id: int,
        theta_init: float,
        learnable_theta: bool = False,
        use_cos2: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = 8
        self.use_cos2 = use_cos2
        self.learnable_theta = learnable_theta

        if learnable_theta:
            # Inverse sigmoid: sigmoid(a)*pi = theta_init
            a_init = math.log(theta_init / (math.pi - theta_init))
            self.a_theta = nn.Parameter(torch.tensor(a_init, dtype=dtype))
        else:
            self.a_theta = None
            self._fixed_theta = theta_init

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

        self.receptance = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.gate = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.output = nn.Linear(hidden_size_att, hidden_size, bias=False, dtype=dtype)
        self.ln_x = nn.GroupNorm(
            n_head, hidden_size_att, dtype=dtype,
            eps=(1e-5) * (self.head_size_divisor ** 2),
        )

    def get_theta(self) -> float:
        """Return current effective theta as Python float — for logging only, not forward."""
        if self.a_theta is not None:
            return (
                (torch.sigmoid(self.a_theta) * math.pi)
                .clamp(min=0.05, max=math.pi - 0.01)
                .detach().item()
            )
        return self._fixed_theta

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

        w_h = -torch.exp(w_h)
        w_h = w_h.clamp(max=math.log(0.99))   # r <= 0.99: prevents state divergence

        # Resolve theta — keep as tensor for Config D so a_theta gets gradients.
        # get_theta() is for logging only (detaches). Here we compute inline.
        if self.a_theta is not None:
            theta = (torch.sigmoid(self.a_theta) * math.pi).clamp(min=0.05, max=math.pi - 0.01)
        else:
            theta = self._fixed_theta  # plain float, no gradient needed

        if self.use_cos2:
            y = _lion_cos2_parallel_attention(r_h, k_h, v_h, w_h, theta)
        else:
            y = _lion_complex_parallel_attention(r_h, k_h, v_h, w_h, theta)
        y = y.to(r.dtype)

        y = y.transpose(1, 2).reshape(B * T, D)
        y = self.ln_x(y).view(B, T, D)
        y = self.output(y * g)
        return y


# ── Block and Encoder ─────────────────────────────────────────────────────────

class BidirRWKV6ComplexBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        num_hidden_layers: int,
        layer_id: int,
        dropout: float,
        theta_init: float,
        learnable_theta: bool = False,
        use_cos2: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln0 = nn.LayerNorm(hidden_size, dtype=dtype) if layer_id == 0 else nn.Identity()

        self.att = BidirRWKV6ComplexTimeMix(
            hidden_size, n_head, head_size, num_hidden_layers, layer_id,
            theta_init, learnable_theta, use_cos2, dtype
        )
        self.ffn = BidirRWKV6ChannelMix(hidden_size, num_hidden_layers, layer_id, dtype)

        self.drop0 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.drop1 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln0(x)
        x = self.drop0(x + self.att(self.ln1(x)))
        x = self.drop1(x + self.ffn(self.ln2(x)))
        return x


class BidirRWKV6ComplexEncoder(nn.Module):
    """Bidirectional RWKV v6 encoder with complex or cos²-valued decay poles.

    Configs:
      B  (theta=0.31, learnable=False, cos2=False): oscillating, can go negative
      C  (theta=0.90, learnable=False, cos2=False): same, faster oscillation
      B-cos2 (theta=0.31, learnable=False, cos2=True): always non-negative, 0 extra params
      D  (theta=0.31, learnable=True,  cos2=False): theta per layer learnable, +6 params
    """

    supports_carry_state = False

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        head_size: int = 64,
        theta_init: float = 0.31,
        learnable_theta: bool = False,
        use_cos2: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert d_model % 32 == 0
        assert d_model % head_size == 0

        self.d_model = d_model
        self.n_layers = n_layers
        self.learnable_theta = learnable_theta
        self.use_cos2 = use_cos2
        n_head = d_model // head_size

        effective_theta = theta_init
        resonance_ms = (2 * math.pi / effective_theta) * 10
        cos2_note = " [cos²]" if use_cos2 else ""
        learn_note = " [learnable]" if learnable_theta else f" [fixed]"
        extra_params = n_layers if learnable_theta else 0

        self.pos_enc = SinusoidalPE(d_model, max_len=8000)
        self.layers = nn.ModuleList([
            BidirRWKV6ComplexBlock(
                d_model, n_head, head_size, n_layers, i, dropout,
                theta_init, learnable_theta, use_cos2, dtype
            )
            for i in range(n_layers)
        ])

        print(
            f"Bidirectional RWKV-6 COMPLEX{cos2_note} encoder initialized "
            f"({n_layers} layers, d_model={d_model}, head_size={head_size}, "
            f"theta_init={theta_init:.4f}{learn_note}, "
            f"resonance~{resonance_ms:.0f}ms, extra_params={extra_params})"
        )

    def get_theta_info(self) -> Optional[str]:
        """Return per-layer theta string (only for learnable Config D)."""
        if not self.learnable_theta:
            return None
        thetas = [layer.att.get_theta() for layer in self.layers]
        res = [2 * math.pi / t * 10 for t in thetas]
        theta_str = " ".join(f"L{i}:{t:.3f}({r:.0f}ms)" for i, (t, r) in enumerate(zip(thetas, res)))
        return f"theta: {theta_str}"

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
