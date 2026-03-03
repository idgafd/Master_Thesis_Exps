"""Linear Attention encoder backbone (ELU+1 feature map, Katharopoulos et al. 2020)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr_exp.models.components import SinusoidalPE


class LinearAttentionLayer(nn.Module):
    """Single linear attention layer with pre-norm and FFN.

    Complexity O(T·D²) instead of O(T²·D).
    Bidirectional within each chunk; no recurrent state to carry.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
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

    @staticmethod
    def _elu_feature(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x = self.norm1(x)

        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q = self._elu_feature(Q)
        K = self._elu_feature(K)

        if mask is not None:
            m = (~mask).float().unsqueeze(1).unsqueeze(-1)
            K = K * m
            V = V * m

        KV = torch.einsum("bhnd,bhne->bhde", K, V)
        Z = 1.0 / (torch.einsum("bhnd,bhd->bhn", Q, K.sum(dim=2)) + 1e-6)
        out = torch.einsum("bhnd,bhde,bhn->bhne", Q, KV, Z)

        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.dropout(self.o_proj(out))
        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x


class LinearAttentionEncoder(nn.Module):
    """Stack of linear attention layers.

    Bidirectional within chunks — carry-state would require a causal training
    regime, so it is marked unsupported to keep the comparison fair.
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
            [LinearAttentionLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state=None,
    ) -> tuple[torch.Tensor, None]:
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask = (
            torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        )
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x, None
