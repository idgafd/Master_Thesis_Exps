"""Bidirectional Linear Attention with learnable decay (LION-D, parallel form).

LION-D full linear attention (Eq. 8 from Afzal et al. 2025):
  Y = SCALE( Q K^T ⊙ M ) V

where M_ij = λ^|i-j| is a symmetric decay mask with learnable per-head λ.
This is the parallel/matrix form — mathematically equivalent to the RNN form
with LION correction terms, but fully parallelizable for training.

Compared to linear_attention.py (which is LION-LIT with M=1, no decay),
this adds a learnable distance-based decay that down-weights farther tokens.

Feature map: ELU(x)+1  (same as linear_attention.py for fair comparison)
Decay: learnable scalar per head, λ = sigmoid(w), init ≈ 0.9

Not compatible with carry-state inference (bidirectional).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr_exp.models.components import SinusoidalPE


class BidirLinearAttentionLayer(nn.Module):
    """Single LION-D bidirectional linear attention layer (parallel form)."""

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

        Q = self._elu_feature(proj_reshape(self.q_proj))  # (B, H, T, d)
        K = self._elu_feature(proj_reshape(self.k_proj))  # (B, H, T, d)
        V = proj_reshape(self.v_proj)                      # (B, H, T, d)

        # Zero out padding positions
        if mask is not None:
            pad = (~mask).float().view(B, 1, T, 1)  # True=padding → 0
            K = K * pad
            V = V * pad

        # Build decay mask: M_ij = λ^|i-j|, shape (H, T, T)
        decay = torch.sigmoid(self.decay_logit)  # (H,) in (0,1)
        positions = torch.arange(T, device=x.device)
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()  # (T, T)
        # Compute λ^|i-j| in log-space for stability: exp(|i-j| * log(λ))
        log_decay = torch.log(decay.clamp(min=1e-8))  # (H,)
        M = torch.exp(distances.unsqueeze(0) * log_decay.view(-1, 1, 1))  # (H, T, T)

        # Attention: A = Q K^T ⊙ M, shape (B, H, T, T)
        A = torch.einsum("bhid,bhjd->bhij", Q, K) * M.unsqueeze(0)

        # SCALE: normalize each row by its sum
        row_sum = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, H, T, 1)
        A = A / row_sum

        # Output: A @ V
        out = torch.einsum("bhij,bhjd->bhid", A, V)  # (B, H, T, d)

        # Reshape back to (B, T, D)
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)
        out = self.dropout(self.o_proj(out))

        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x


class BidirLinearAttentionEncoder(nn.Module):
    """Stack of LION-D bidirectional linear attention layers (parallel form).

    Same as linear_attention.py but with a learnable per-head decay mask
    M_ij = λ^|i-j| that down-weights distant tokens.
    Not compatible with streaming / carry-state inference.
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
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x, None
