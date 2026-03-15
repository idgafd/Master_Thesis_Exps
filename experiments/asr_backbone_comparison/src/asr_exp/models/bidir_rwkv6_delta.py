"""Bidirectional RWKV-6 (LION) + Delta Rule encoder backbone.

Takes the LION parallel attention form and adds RWKV-7's delta rule as a
correction term to the attention matrix.

Standard LION:   Y = (A_fwd + A_bwd) @ V
Delta rule:      Y = (A_fwd + A_bwd + A_delta) @ V

The delta rule correction A_delta captures the selective erasure effect:
for each position i, it adjusts the attention weights based on how much
the state would have been modified by the erasure term.

In the parallel form, the rank-1 erasure at each position creates an
additional O(T^2) correction matrix that can be computed efficiently
since LION already pays O(T^2) cost.

New learnable parameters per layer (from RWKV-7):
  - a0, a1, a2: LoRA for in-context learning rate
  - k_k: key normalization scaling
  - k_a: key scaling factor
"""

import math
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


def _lion_delta_parallel_attention(
    r: Tensor,    # (B, H, T, K) — receptance/query
    k: Tensor,    # (B, H, T, K) — key (already ICLR-scaled for writing)
    v: Tensor,    # (B, H, T, K) — value
    w: Tensor,    # (B, H, T, K) — log-space decay (negative)
    kk: Tensor,   # (B, H, T, K) — normalized key for erasure
    iclr: Tensor, # (B, H, T, K) — in-context learning rate
) -> Tensor:
    """LION parallel attention with delta rule correction.

    The delta rule adds a correction to the standard LION attention.
    In the recurrent form, the state update is:
      S_i = diag(w_i) * S_{i-1} + S_{i-1} @ ab_i + v_i @ k_i^T

    The ab_i = (-kk_i)^T @ (kk_i * iclr_i) term creates an additional
    contribution to the attention matrix. For positions j < i (forward):
      delta_correction[i,j] = r_i^T @ [prod_{l=j+1}^{i} G_l] @ ab_j+1..i terms

    We approximate this as a first-order correction:
      A_delta_fwd[i,j] = sum_{m=j+1}^{i} A_fwd[i,m] * erasure_effect[m,j]

    For efficiency, we compute it as a matrix product of the standard
    attention with an erasure matrix.
    """
    B, H, T, K = r.shape

    r = r.float()
    k = k.float()
    v = v.float()
    w = w.float()
    kk = kk.float()
    iclr = iclr.float()

    # ── Standard LION attention (same as bidir_rwkv6.py) ──
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

    # ── Delta rule correction ──
    # The erasure at position m affects all positions j < m (forward)
    # and j > m (backward) by modifying what state they contributed.
    #
    # Erasure matrix: E[m, j] = kk[m]^T @ (kk[m] * iclr[m]) measures
    # how much position m's erasure affects the key at position j.
    # In the parallel form, this becomes a correction to the attention:
    #
    # For forward direction: the erasure at position m reduces the
    # contribution of earlier positions j whose keys are correlated
    # with kk[m]. This is captured by:
    #   A_delta[i,j] ≈ -sum_{m=j+1}^{i} A_fwd_raw[i,m] * (kk[m] @ kk[j])^T * iclr[m]

    # Compute key correlation: how much each position's erasure key
    # correlates with each other position's key
    # kk_corr[m,j] = sum_d kk[m,d] * (kk[j,d] * iclr[m,d])
    # This tells us: position m's erasure removes correlation with j
    kk_corr = (kk * iclr) @ kk.transpose(-2, -1)  # (B, H, T, T)

    # Forward correction: for i > j, the erasure at positions m in (j,i]
    # reduces j's contribution to i's output.
    # We use a cumulative approach: the total erasure effect on j
    # as seen from position i includes all erasures between j+1 and i.
    # Approximate as: A_delta_fwd = -A_fwd * cumulative_erasure_effect
    # where cumulative_erasure_effect accumulates kk_corr along the path.

    # Simplified first-order correction:
    # The correction at (i,j) is proportional to the sum of erasure
    # correlations between j+1 and i, weighted by the decay.
    # This is equivalent to: A_delta ≈ -(A_fwd @ tril(kk_corr, -1))
    kk_corr_causal = torch.tril(kk_corr, diagonal=-1)  # strictly lower tri
    kk_corr_anticausal = torch.triu(kk_corr, diagonal=1)  # strictly upper tri

    # Forward delta correction
    A_delta_fwd = -torch.tril(A_fwd @ kk_corr_causal)

    # Backward delta correction
    A_delta_bwd = -torch.triu(A_bwd @ kk_corr_anticausal, diagonal=1)

    # Combined output
    A_total = A_fwd + A_bwd + A_delta_fwd + A_delta_bwd

    return (A_total @ v).to(r.dtype)


class BidirRWKV6DeltaTimeMix(nn.Module):
    """Bidirectional RWKV6 time-mixing with LION + delta rule."""

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

            # ── RWKV-6 token shift params ──
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            )
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_a = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

            D_MIX_DIM = 32
            self.time_maa_w1 = nn.Parameter(
                torch.zeros(hidden_size, D_MIX_DIM * 6, dtype=dtype)
            )
            self.time_maa_w2 = nn.Parameter(
                torch.zeros(6, D_MIX_DIM, hidden_size, dtype=dtype).uniform_(-0.01, 0.01)
            )

            # ── Decay ──
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

            # ── Delta rule params ──
            D_AAA_LORA = max(1, round(hidden_size_att ** 0.5 * 1.8 / 32)) * 32

            self.a0 = nn.Parameter(torch.zeros(1, 1, hidden_size_att, dtype=dtype))
            self.a1 = nn.Parameter(torch.zeros(hidden_size_att, D_AAA_LORA, dtype=dtype))

            def ortho_init(x, scale):
                gain = math.sqrt(x.shape[0] / x.shape[1]) if x.shape[0] > x.shape[1] else 1
                nn.init.orthogonal_(x, gain=gain * scale)
                return x.to(dtype=dtype)

            self.a2 = nn.Parameter(
                ortho_init(torch.zeros(D_AAA_LORA, hidden_size_att), 0.1)
            )

            self.k_k = nn.Parameter(torch.ones(1, 1, hidden_size_att, dtype=dtype) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, hidden_size_att, dtype=dtype))

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

        # Bidirectional token shift
        dxprev = _bidirectional_token_shift(x) - x

        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 6, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(6, B, T, D)

        mw, mk, mv, mr, mg, ma = xxx.unbind(dim=0)
        xw = x + dxprev * (self.time_maa_w + mw)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xr = x + dxprev * (self.time_maa_r + mr)
        xg = x + dxprev * (self.time_maa_g + mg)
        xa = x + dxprev * (self.time_maa_a + ma)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = w.to(r.dtype)

        # Delta rule parameters
        iclr = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        kk = F.normalize(
            (k * self.k_k).view(B, T, H, -1), dim=-1, p=2.0
        ).view(B, T, D)
        k = k * (1 + (iclr - 1) * self.k_a)

        # Reshape to (B, H, T, K)
        r_h = r.view(B, T, H, K).transpose(1, 2)
        k_h = k.view(B, T, H, K).transpose(1, 2)
        v_h = v.view(B, T, H, K).transpose(1, 2)
        w_h = -torch.exp(w.view(B, T, H, K).transpose(1, 2))
        kk_h = kk.view(B, T, H, K).transpose(1, 2)
        iclr_h = iclr.view(B, T, H, K).transpose(1, 2)

        # LION + delta rule
        y = _lion_delta_parallel_attention(r_h, k_h, v_h, w_h, kk_h, iclr_h).to(r.dtype)

        y = y.transpose(1, 2).reshape(B * T, D)
        y = self.ln_x(y).view(B, T, D)
        y = self.output(y * g)

        return y


class BidirRWKV6DeltaChannelMix(nn.Module):
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


class BidirRWKV6DeltaBlock(nn.Module):
    """Single LION + Delta Rule layer block."""

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

        self.att = BidirRWKV6DeltaTimeMix(
            hidden_size, n_head, head_size, num_hidden_layers, layer_id, dtype
        )
        self.ffn = BidirRWKV6DeltaChannelMix(
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


class BidirRWKV6DeltaEncoder(nn.Module):
    """Bidirectional RWKV-6 (LION) + Delta Rule encoder.

    LION parallel attention with first-order delta rule correction.
    O(T^2 K) compute — same asymptotic cost as standard LION since
    the delta correction is an additional T×T matrix multiplication.
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
                BidirRWKV6DeltaBlock(
                    d_model, n_head, head_size, n_layers, i, dropout, dtype
                )
            )

        print(
            f"Bidirectional RWKV-6+DeltaRule (LION+Delta) encoder initialized "
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
