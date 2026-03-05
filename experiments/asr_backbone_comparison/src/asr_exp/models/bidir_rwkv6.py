"""Bidirectional RWKV v6 encoder backbone — LION parallel form for training.

Implements the LION full linear attention form for RWKV-6 (Theorem B.1, Eq. 127
from Afzal et al. 2025):

  Y = (TRIL[(Q ⊙ L^F)(K ⊙ Inv(L^F))^T] + TRIU[(Q ⊙ L^B)(K ⊙ Inv(L^B))^T]) V

This is the parallel/matrix form — O(T²K) compute with O(1) sequential depth,
fully parallelisable on GPU.  Mathematically equivalent to the bidirectional
RNN form (Eq. 125-126) but dramatically faster for training because every
position is computed simultaneously via matrix products rather than a step-by-
step loop over T timesteps.

Causal RWKV-6 recurrence (for reference):
  S_i = Diag(Λ_i) · S_{i-1} + k_i · v_i^T
  y_i = q_i^T · S_i   (non-scaled)

RWKV-6 uses diagonal (per-head-dimension) input-dependent decay and is
non-scaled, so no SCALE denominator in the output.

Key subtlety: the recurrence applies decay Λ_i *before* adding k_i v_i^T at
position i.  This means the forward decay range from j to i is Π_{l=j+1}^{i}
while the backward range from i to j is Π_{l=i}^{j-1} — an asymmetry that
requires different prefix sums for the two directions:
  log_L_fwd = cumsum(w)            → log_L_fwd[t] = Σ_{l=0}^{t} w[l]
  log_L_bwd = cumsum(w) - w        → log_L_bwd[t] = Σ_{l=0}^{t-1} w[l]

Token-shift is bidirectional (average of left and right neighbours).
Not compatible with carry-state inference (requires full sequence).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr_exp.models.components import SinusoidalPE


def _bidirectional_token_shift(x: torch.Tensor) -> torch.Tensor:
    """Bidirectional token shift: average of left and right neighbors.

    For position i, returns (x[i-1] + x[i+1]) / 2, with zero-padding at boundaries.
    Replaces the causal-only shift (x[i-1] only) used in standard RWKV.
    """
    B, T, D = x.shape
    left = F.pad(x[:, :-1, :], (0, 0, 1, 0))   # x[i-1], zero at i=0
    right = F.pad(x[:, 1:, :], (0, 0, 0, 1))    # x[i+1], zero at i=T-1
    return (left + right) * 0.5


def _lion_parallel_attention(
    r: torch.Tensor,  # (B, H, T, K) — receptance (= q in LION notation)
    k: torch.Tensor,  # (B, H, T, K)
    v: torch.Tensor,  # (B, H, T, K)
    w: torch.Tensor,  # (B, H, T, K) — log-space decay (negative values)
) -> torch.Tensor:
    """LION parallel full attention for bidirectional RWKV-6 (Theorem B.1).

    The RWKV-6 recurrence S_i = Λ_i S_{i-1} + k_i v_i^T applies the decay
    at position i to the *incoming* state.  Unrolling gives:

      Forward coeff(j→i), i > j:  Π_{l=j+1}^{i} Λ_l = exp(cs[i] - cs[j])
      Backward coeff(j→i), i < j: Π_{l=i}^{j-1} Λ_l = exp(cs_b[j] - cs_b[i])

    where  cs  = cumsum(w)  and  cs_b = cumsum(w) - w  (shifted prefix sum,
    because the backward direction excludes the current position's decay on
    the *emitting* side and includes it on the *receiving* side).

    The attention matrix is built via the cumulative-product trick:

      A_fwd = tril( (Q·exp(cs)) @ (K·exp(-cs))^T )      — lower tri + diag
      A_bwd = triu( (Q·exp(-cs_b)) @ (K·exp(cs_b))^T, 1) — strict upper tri
      Y     = (A_fwd + A_bwd) @ V

    The diagonal is counted once (from A_fwd) with coefficient 1, matching
    M_{ii} = 1.

    Numerically stabilised by shifting the prefix sums so exponents stay
    within float32 range (the shifts cancel in the L[i]/L[j] ratios).

    Returns: (B, H, T, K) output in float32.
    """
    B, H, T, K = r.shape

    r = r.float()
    k = k.float()
    v = v.float()
    w = w.float()

    # Prefix sums for forward and backward directions
    cs = torch.cumsum(w, dim=2)  # cs[t] = Σ_{l=0}^{t} w[l]
    cs_b = cs - w                # cs_b[t] = Σ_{l=0}^{t-1} w[l]  (backward shift)

    # Numerical stabilization: shift each prefix sum by its midpoint value
    # (shifts cancel in L[i]/L[j] ratios)
    shift_f = cs[:, :, T // 2 : T // 2 + 1, :]
    shift_b = cs_b[:, :, T // 2 : T // 2 + 1, :]
    cs_f_s = (cs - shift_f).clamp(-60, 60)
    cs_b_s = (cs_b - shift_b).clamp(-60, 60)

    # Forward (lower-triangular including diagonal):
    #   A_fwd[i,j] = Σ_d r[i,d]*k[j,d]*exp(cs[i,d]-cs[j,d])  for i >= j
    exp_cs_f = torch.exp(cs_f_s)
    A_fwd = torch.tril(
        (r * exp_cs_f) @ (k * torch.exp(-cs_f_s)).transpose(-2, -1)
    )

    # Backward (strictly upper-triangular, diagonal already in A_fwd):
    #   A_bwd[i,j] = Σ_d r[i,d]*k[j,d]*exp(cs_b[j,d]-cs_b[i,d])  for i < j
    exp_cs_b = torch.exp(cs_b_s)
    A_bwd = torch.triu(
        (r * torch.exp(-cs_b_s)) @ (k * exp_cs_b).transpose(-2, -1),
        diagonal=1,
    )

    # Combined: diagonal counted once with coefficient 1 (M_{ii} = 1)
    return (A_fwd + A_bwd) @ v  # (B, H, T, K)


class BidirRWKV6TimeMix(nn.Module):
    """Bidirectional RWKV6 time-mixing with LION parallel full attention.

    Same parameter structure as the causal RWKV6TimeMix but:
    - Bidirectional token shift instead of causal
    - Full attention matrix (parallel) instead of sequential RNN recurrence
    - RWKV-6 is non-scaled, outputs are simply forward + backward
    """

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

        # Bidirectional token shift
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

        r = self.receptance(xr)  # q in LION notation
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = w.to(r.dtype)

        # Reshape to (B, H, T, K) for parallel attention
        r_h = r.view(B, T, H, K).transpose(1, 2)
        k_h = k.view(B, T, H, K).transpose(1, 2)
        v_h = v.view(B, T, H, K).transpose(1, 2)
        w_h = w.view(B, T, H, K).transpose(1, 2)

        # Log-decay: actual decay = exp(-exp(w_raw)), so log-decay = -exp(w_raw)
        w_h = -torch.exp(w_h)

        # LION parallel full attention (replaces sequential RNN recurrence)
        y = _lion_parallel_attention(r_h, k_h, v_h, w_h).to(r.dtype)

        # Reshape back to (B*T, D) for GroupNorm
        y = y.transpose(1, 2).reshape(B * T, D)

        # GroupNorm + gate + output projection (same as standard RWKV6)
        y = self.ln_x(y).view(B, T, D)
        y = self.output(y * g)

        return y


class BidirRWKV6ChannelMix(nn.Module):
    """Bidirectional RWKV6 channel mix (FFN with bidirectional token shift)."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dxprev = _bidirectional_token_shift(x) - x

        xk = x + dxprev * self.time_maa_k
        xr = x + dxprev * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class BidirRWKV6Block(nn.Module):
    """Single bidirectional RWKV6 layer block."""

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

        self.att = BidirRWKV6TimeMix(
            hidden_size, n_head, head_size, num_hidden_layers, layer_id, dtype
        )
        self.ffn = BidirRWKV6ChannelMix(
            hidden_size, num_hidden_layers, layer_id, dtype
        )

        if dropout > 0.0:
            self.drop0 = nn.Dropout(p=dropout)
            self.drop1 = nn.Dropout(p=dropout)
        else:
            self.drop0 = nn.Identity()
            self.drop1 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln0(x)
        x = self.drop0(x + self.att(self.ln1(x)))
        x = self.drop1(x + self.ffn(self.ln2(x)))
        return x


class BidirRWKV6Encoder(nn.Module):
    """Bidirectional RWKV v6 encoder using LION parallel form (Afzal et al. 2025).

    Each layer computes the full T×T bidirectional attention matrix via the
    cumulative-product trick (Theorem B.1), giving O(T²K) compute with O(1)
    sequential depth — fully parallelisable for training.

    Token shifts are bidirectional.
    Not compatible with carry-state inference (requires full sequence).
    d_model must be divisible by 32.
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
                BidirRWKV6Block(
                    d_model, n_head, head_size, n_layers, i, dropout, dtype
                )
            )

        print(
            f"Bidirectional RWKV-6 (LION parallel) encoder initialized "
            f"({n_layers} layers, d_model={d_model}, head_size={head_size})"
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
        mask_f = mask.unsqueeze(-1).float()  # (B, T, 1)

        for layer in self.layers:
            x = layer(x) * mask_f

        return x, None
