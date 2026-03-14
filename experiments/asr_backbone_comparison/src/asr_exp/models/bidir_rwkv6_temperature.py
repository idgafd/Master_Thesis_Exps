"""Bidirectional RWKV v6 encoder with per-layer attention temperature.

Based on bidir_rwkv6.py (LION parallel form) with one change:

**Attention temperature**: After computing the LION attention matrix A,
apply element-wise power A^τ (for non-negative entries) to sharpen or
flatten attention distributions. Lower layers use τ ≈ 1.0 (preserve
broad attention), upper layers use τ > 1.0 (sharpen to local focus).

τ is a learnable per-head scalar, initialised with a layer-dependent
schedule: τ_init = 1.0 + layer_idx / (n_layers - 1) (L0: 1.0, L5: 2.0).

The sharpening operates on the *combined* forward+backward attention
matrix before multiplying by V. Row-wise re-normalisation ensures the
attention weights sum to a consistent scale after sharpening.

Everything else (LION full attention, parameter layout, token shift)
is identical to bidir_rwkv6.py.
supports_carry_state = False.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr_exp.models.bidir_rwkv6 import _bidirectional_token_shift
from asr_exp.models.components import SinusoidalPE


def _lion_parallel_attention_with_temperature(
    r: torch.Tensor,  # (B, H, T, K)
    k: torch.Tensor,  # (B, H, T, K)
    v: torch.Tensor,  # (B, H, T, K)
    w: torch.Tensor,  # (B, H, T, K) — log-space decay (negative values)
    tau: torch.Tensor, # (H,) — per-head temperature
) -> torch.Tensor:
    """LION parallel attention with per-head temperature sharpening.

    Same as _lion_parallel_attention but after computing A = A_fwd + A_bwd,
    applies A_sharp = A^τ (element-wise) with per-head τ, then re-normalises
    rows so that sum(A_sharp[i,:]) matches sum(A[i,:]).

    τ > 1 sharpens (concentrates mass on large entries).
    τ = 1 is identity (no change).
    τ < 1 flattens (spreads mass more evenly).
    """
    B, H, T, K = r.shape

    r = r.float()
    k = k.float()
    v = v.float()
    w = w.float()

    # Prefix sums for forward and backward directions
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

    A = A_fwd + A_bwd  # (B, H, T, T)

    # Apply per-head temperature: A_sharp = A^tau
    # LION attention values are non-negative (products of exp-weighted terms),
    # but numerical noise can produce tiny negatives. Clamp to 0.
    A = A.clamp(min=0.0)

    # tau: (H,) → (1, H, 1, 1) for broadcasting
    tau_bcast = tau.float().view(1, H, 1, 1)

    # A^tau — for A=0 and tau>0, result is 0 (safe)
    # Use torch.pow which handles 0^positive correctly
    A_sharp = torch.pow(A + 1e-12, tau_bcast)
    # Zero out where A was zero (the 1e-12 offset would produce nonzero)
    A_sharp = A_sharp * (A > 0).float()

    # Re-normalise rows: scale so row sums match original
    # This preserves the overall magnitude while sharpening the distribution.
    row_sum_orig = A.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    row_sum_sharp = A_sharp.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    A_sharp = A_sharp * (row_sum_orig / row_sum_sharp)

    return A_sharp @ v  # (B, H, T, K)


class BidirRWKV6TemperatureTimeMix(nn.Module):
    """Bidirectional RWKV6 time-mixing with LION attention + per-head temperature."""

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

        # Per-head temperature: init from 1.0 (lower layers) to 2.0 (upper layers)
        # Stored in log-space to ensure tau > 0: tau = exp(log_tau)
        tau_init = 1.0 + ratio_0_to_1  # L0: 1.0, L5: 2.0
        self.log_tau = nn.Parameter(
            torch.full((n_head,), fill_value=float(torch.tensor(tau_init).log()), dtype=dtype)
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

    @property
    def tau(self) -> torch.Tensor:
        """Per-head temperature (always positive)."""
        return self.log_tau.exp()

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

        y = _lion_parallel_attention_with_temperature(
            r_h, k_h, v_h, w_h, self.tau
        ).to(r.dtype)

        y = y.transpose(1, 2).reshape(B * T, D)
        y = self.ln_x(y).view(B, T, D)
        y = self.output(y * g)
        return y


class BidirRWKV6TemperatureBlock(nn.Module):
    """Single BidirRWKV6 layer block with temperature-controlled attention."""

    def __init__(self, hidden_size: int, n_head: int, head_size: int,
                 num_hidden_layers: int, layer_id: int, dropout: float,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln0 = nn.LayerNorm(hidden_size, dtype=dtype) if layer_id == 0 else nn.Identity()

        self.att = BidirRWKV6TemperatureTimeMix(
            hidden_size, n_head, head_size, num_hidden_layers, layer_id, dtype
        )
        # Standard channel mix with bidirectional token shift (no temperature needed in FFN)
        from asr_exp.models.bidir_rwkv6 import BidirRWKV6ChannelMix
        self.ffn = BidirRWKV6ChannelMix(
            hidden_size, num_hidden_layers, layer_id, dtype
        )

        self.drop0 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.drop1 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln0(x)
        x = self.drop0(x + self.att(self.ln1(x)))
        x = self.drop1(x + self.ffn(self.ln2(x)))
        return x


class BidirRWKV6TemperatureEncoder(nn.Module):
    """Bidirectional RWKV v6 (LION) with per-layer attention temperature.

    Lower layers: τ ≈ 1.0 (broad attention preserved).
    Upper layers: τ ≈ 2.0 (attention sharpened toward high-weight entries).

    τ is learnable per head (4 params per layer = 24 total extra params).

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
                BidirRWKV6TemperatureBlock(
                    d_model, n_head, head_size, n_layers, i, dropout, dtype
                )
            )

        # Print tau schedule for verification
        taus = [layer.att.tau.data.tolist() for layer in self.layers]
        tau_summary = [f"L{i}:{t[0]:.2f}" for i, t in enumerate(taus)]
        print(
            f"BidirRWKV6Temperature encoder initialized "
            f"({n_layers} layers, d_model={d_model}, head_size={head_size}, "
            f"tau_init=[{', '.join(tau_summary)}])"
        )

    def get_tau_info(self) -> str:
        """Return a string with per-layer tau values for logging."""
        parts = []
        for i, layer in enumerate(self.layers):
            tau_vals = layer.att.tau.data.tolist()
            tau_str = ",".join(f"{t:.3f}" for t in tau_vals)
            parts.append(f"L{i}:[{tau_str}]")
        return "tau: " + " ".join(parts)

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
