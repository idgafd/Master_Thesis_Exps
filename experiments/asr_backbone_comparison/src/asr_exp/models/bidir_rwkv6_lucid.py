"""Bidirectional RWKV-6 (LION) + LUCID preconditioner encoder backbone.

Takes the LION parallel attention form and adds LUCID's key decorrelation
preconditioner:

  Standard LION:   Y = (A_fwd + A_bwd) @ V
  LUCID:           Y = (A_fwd + A_bwd) @ P @ V

where P = (M * exp(temp * K @ K^T))^{-1} decorrelates correlated keys,
reducing noise accumulation in the attention output.

This is the most natural pairing: LION already computes O(T^2) attention
matrices, and LUCID adds a T×T preconditioner — no new asymptotic cost.
For T~200-400 frames (after subsampling), the matrix solve is cheap.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from asr_exp.models.components import SinusoidalPE


def _bidirectional_token_shift(x: Tensor) -> Tensor:
    """Bidirectional token shift: average of left and right neighbors."""
    B, T, D = x.shape
    left = F.pad(x[:, :-1, :], (0, 0, 1, 0))
    right = F.pad(x[:, 1:, :], (0, 0, 0, 1))
    return (left + right) * 0.5


def _lion_lucid_parallel_attention(
    r: Tensor,         # (B, H, T, K)
    k: Tensor,         # (B, H, T, K)
    v: Tensor,         # (B, H, T, K)
    w: Tensor,         # (B, H, T, K) — log-space decay (negative)
    lucid_temp: Tensor, # (1, H, 1, 1) — learnable temperature
) -> Tensor:
    """LION parallel attention with LUCID preconditioner.

    Standard LION attention matrix (bidirectional) with LUCID's
    key decorrelation preconditioner applied before multiplying by V.

    The preconditioner P = (M * exp(temp * K@K^T))^{-1} where:
    - M is derived from the combined attention matrix (provides structure)
    - K@K^T captures key correlations
    - temp controls decorrelation strength (learnable)
    """
    B, H, T, K = r.shape

    r = r.float()
    k = k.float()
    v = v.float()
    w = w.float()

    # ── Standard LION attention ──
    cs = torch.cumsum(w, dim=2)
    cs_b = cs - w

    shift_f = cs[:, :, T // 2 : T // 2 + 1, :]
    shift_b = cs_b[:, :, T // 2 : T // 2 + 1, :]
    cs_f_s = (cs - shift_f).clamp(-60, 60)
    cs_b_s = (cs_b - shift_b).clamp(-60, 60)

    exp_cs_f = torch.exp(cs_f_s)
    A_fwd = torch.tril(
        (r * exp_cs_f) @ (k * torch.exp(-cs_f_s)).transpose(-2, -1)
    )

    exp_cs_b = torch.exp(cs_b_s)
    A_bwd = torch.triu(
        (r * torch.exp(-cs_b_s)) @ (k * exp_cs_b).transpose(-2, -1),
        diagonal=1,
    )

    A_combined = A_fwd + A_bwd  # (B, H, T, T)

    # ── LUCID preconditioner ──
    # L2-normalize keys for stable gram matrix
    k_norm = F.normalize(k, dim=-1, p=2.0)
    K_gram = k_norm @ k_norm.transpose(-2, -1)  # (B, H, T, T) in [-1, 1]

    # Temperature-scaled gram matrix (bounded)
    temp = F.softplus(lucid_temp.float())
    scaled_gram = (temp * K_gram).clamp(-20, 20)

    # Preconditioner: P = (I + exp(temp * K_gram))^{-1}
    precond_matrix = torch.eye(
        T, device=k.device, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0) + torch.exp(scaled_gram)

    # Solve P @ X = V for preconditioned values
    P_v = torch.linalg.solve(precond_matrix, v)  # (B, H, T, K)

    return (A_combined @ P_v).to(r.dtype)


class BidirRWKV6LucidTimeMix(nn.Module):
    """Bidirectional RWKV6 time-mixing with LION + LUCID preconditioner."""

    def __init__(self, hidden_size: int, n_head: int, head_size: int,
                 num_hidden_layers: int, layer_id: int,
                 dtype: torch.dtype = torch.float32):
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

        # Learnable LUCID temperature per head
        self.lucid_temperature = nn.Parameter(torch.ones(1, n_head, 1, 1, dtype=dtype))

        self.receptance = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.gate = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.output = nn.Linear(hidden_size_att, hidden_size, bias=False, dtype=dtype)
        self.ln_x = nn.GroupNorm(
            n_head, hidden_size_att, dtype=dtype,
            eps=(1e-5) * (self.head_size_divisor ** 2),
        )

    def forward(self, x: Tensor) -> Tensor:
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
        w_h = -torch.exp(w.view(B, T, H, K).transpose(1, 2))

        y = _lion_lucid_parallel_attention(
            r_h, k_h, v_h, w_h, self.lucid_temperature
        ).to(r.dtype)

        y = y.transpose(1, 2).reshape(B * T, D)
        y = self.ln_x(y).view(B, T, D)
        y = self.output(y * g)

        return y


class BidirRWKV6LucidChannelMix(nn.Module):
    """Bidirectional RWKV6 channel mix — same as stock LION."""

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

        self.key = nn.Linear(hidden_size, hidden_size_ffn, bias=False, dtype=dtype)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.value = nn.Linear(hidden_size_ffn, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        dxprev = _bidirectional_token_shift(x) - x
        xk = x + dxprev * self.time_maa_k
        xr = x + dxprev * self.time_maa_r
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class BidirRWKV6LucidBlock(nn.Module):
    """Single LION + LUCID layer block."""

    def __init__(self, hidden_size: int, n_head: int, head_size: int,
                 num_hidden_layers: int, layer_id: int, dropout: float,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(hidden_size, dtype=dtype)
        else:
            self.ln0 = nn.Identity()

        self.att = BidirRWKV6LucidTimeMix(
            hidden_size, n_head, head_size, num_hidden_layers, layer_id, dtype
        )
        self.ffn = BidirRWKV6LucidChannelMix(
            hidden_size, num_hidden_layers, layer_id, dtype
        )

        if dropout > 0.0:
            self.drop0 = nn.Dropout(p=dropout)
            self.drop1 = nn.Dropout(p=dropout)
        else:
            self.drop0 = nn.Identity()
            self.drop1 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.ln0(x)
        x = self.drop0(x + self.att(self.ln1(x)))
        x = self.drop1(x + self.ffn(self.ln2(x)))
        return x


class BidirRWKV6LucidEncoder(nn.Module):
    """Bidirectional RWKV-6 (LION) + LUCID preconditioner encoder.

    LION parallel attention with LUCID key decorrelation.
    O(T^2 K + T^3) compute per layer per head — the T^3 from the
    matrix solve is negligible for T < 500 typical in ASR.
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
        assert d_model % 32 == 0
        assert d_model % head_size == 0

        self.d_model = d_model
        self.n_layers = n_layers
        n_head = d_model // head_size

        self.pos_enc = SinusoidalPE(d_model, max_len=8000)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                BidirRWKV6LucidBlock(
                    d_model, n_head, head_size, n_layers, i, dropout, dtype
                )
            )

        print(
            f"Bidirectional RWKV-6+LUCID (LION+LUCID) encoder initialized "
            f"({n_layers} layers, d_model={d_model}, head_size={head_size})"
        )

    def forward(
        self,
        x: Tensor,
        lengths: Tensor,
        state=None,
    ) -> Tuple[Tensor, None]:
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()

        for layer in self.layers:
            x = layer(x) * mask_f

        return x, None
