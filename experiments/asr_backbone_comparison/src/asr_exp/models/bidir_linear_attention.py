"""Bidirectional Linear Attention encoder (LION dual-recurrence correction).

Implements the balanced dual-recurrence from the LION recipe:
  - Forward pass  : S_i^F = λ S_{i-1}^F + k_i v_i^T
  - Backward pass : S_i^B = λ S_{i+1}^B + k_i v_i^T
  - Correction    : subtract ½(q_i·k_i) v_i so neither stream double-counts token i
  - Combine       : y_i = (ŷ_i^F + ŷ_i^B) / (c_i^F + c_i^B)

This is mathematically equivalent to full (non-causal) linear attention while
staying in the recurrent / TC^0 complexity class (O(T·D²) per layer).

Feature map: ELU(x)+1  (Katharopoulos et al. 2020, same as linear_attention.py)
Decay      : learnable scalar per head, initialised near 1 via sigmoid(log(9))≈0.9
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr_exp.models.components import SinusoidalPE


class BidirLinearAttentionLayer(nn.Module):
    """Single LION-corrected bidirectional linear attention layer with pre-norm + FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

        # Learnable per-head scalar decay: λ = sigmoid(w), init ≈ 0.9
        self.decay_logit = nn.Parameter(torch.full((n_heads,), math.log(9.0)))

    @staticmethod
    def _elu_feature(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def _recurrence(
        self,
        Q: torch.Tensor,   # (B, H, T, d)
        K: torch.Tensor,   # (B, H, T, d)
        V: torch.Tensor,   # (B, H, T, d)
        decay: torch.Tensor,  # (H,) in [0,1]
        reverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one directional recurrence; returns (Y, C) each (B, H, T, d)."""
        B, H, T, d = Q.shape
        device = Q.device

        S = torch.zeros(B, H, d, d, device=device, dtype=Q.dtype)  # context matrix
        z = torch.zeros(B, H, d, device=device, dtype=Q.dtype)      # normaliser vector

        Y_list = []
        C_list = []

        time_range = range(T - 1, -1, -1) if reverse else range(T)

        for i in time_range:
            k_i = K[:, :, i, :]  # (B, H, d)
            v_i = V[:, :, i, :]  # (B, H, d)
            q_i = Q[:, :, i, :]  # (B, H, d)

            # decay shape: (1, H, 1, 1) and (1, H, 1) for broadcasting
            lam = decay.view(1, H, 1, 1)
            lam_z = decay.view(1, H, 1)

            # State update: S = λS + k⊗v,   z = λz + k
            S = lam * S + torch.einsum("bhd,bhe->bhde", k_i, v_i)
            z = lam_z * z + k_i

            # Raw output before LION correction
            y_raw = torch.einsum("bhd,bhde->bhe", q_i, S)   # (B, H, d)
            c_raw = (q_i * z).sum(dim=-1)                    # (B, H)

            # LION correction: subtract ½(q·k)v  /  ½(q·k)
            qk = (q_i * k_i).sum(dim=-1, keepdim=True)       # (B, H, 1)
            y_corr = y_raw - 0.5 * qk * v_i                  # (B, H, d)
            c_corr = c_raw - 0.5 * qk.squeeze(-1)            # (B, H)

            Y_list.append(y_corr)
            C_list.append(c_corr)

        if reverse:
            Y_list.reverse()
            C_list.reverse()

        Y = torch.stack(Y_list, dim=2)  # (B, H, T, d)
        C = torch.stack(C_list, dim=2)  # (B, H, T)
        return Y, C

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x    : (B, T, D)
        mask : (B, T) bool, True = padding position
        """
        B, T, D = x.shape
        residual = x
        x = self.norm1(x)

        # Project and reshape to (B, H, T, head_dim)
        def proj_reshape(proj):
            return proj(x).view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        Q = self._elu_feature(proj_reshape(self.q_proj))
        K = self._elu_feature(proj_reshape(self.k_proj))
        V = proj_reshape(self.v_proj)

        # Zero out padding positions so they don't contaminate the state
        if mask is not None:
            pad = (~mask).float().view(B, 1, T, 1)  # True=padding → 0
            K = K * pad
            V = V * pad

        decay = torch.sigmoid(self.decay_logit)  # (H,) in (0,1)

        Y_f, C_f = self._recurrence(Q, K, V, decay, reverse=False)
        Y_b, C_b = self._recurrence(Q, K, V, decay, reverse=True)

        # Combine: y_i = (Y_f + Y_b) / (C_f + C_b)
        denom = (C_f + C_b).unsqueeze(-1).clamp(min=1e-6)  # (B, H, T, 1)
        out = (Y_f + Y_b) / denom                          # (B, H, T, d)

        # Reshape back to (B, T, D)
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)
        out = self.dropout(self.o_proj(out))

        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x


class BidirLinearAttentionEncoder(nn.Module):
    """Stack of LION bidirectional linear attention layers.

    Mathematically equivalent to full (non-causal) linear attention.
    Not compatible with streaming / carry-state inference because the backward
    pass requires the full sequence.
    """

    supports_carry_state = False

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.pos_enc = SinusoidalPE(d_model, max_len=8000)
        self.layers = nn.ModuleList(
            [
                BidirLinearAttentionLayer(d_model, n_heads, ffn_dim, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state=None,
    ) -> tuple[torch.Tensor, None]:
        """
        x       : (B, T, D)
        lengths : (B,) valid frame counts
        state   : ignored (bidirectional, no carry state)
        """
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x, None
