"""LION-LIT bidirectional Linear Attention encoder.

Mirrors the LION paper's mapping (Afzal et al. 2025, Table 1) for the
Katharopoulos linear attention family: ``LinAtt → LION-LIT`` (no decay,
λ = 1).  Provides the bidirectional analog of ``CausalLinearAttentionLayer``
in ``linear_attn_causal.py`` so the matrix slot (LA, mode=lion) is filled
with a paper-faithful LION wrapper rather than the naive non-balanced
bidir LA in ``blocks.py::LinearAttentionLayer``.

Mathematical form
-----------------

For LION-LIT (no decay), the unified LION kernel
``lion_attention.lion_parallel_attention(r, k, v, w)`` reduces to:

    A_fwd = tril(  phi(Q) @ phi(K)^T )      (lower-triangular, includes diagonal)
    A_bwd = triu(  phi(Q) @ phi(K)^T , 1 )  (strictly upper-triangular)
    Y     = (A_fwd + A_bwd) @ V             (full bidirectional sum)

with ``w = 0`` (zero log-decay) so ``exp(cs) = exp(-cs) = 1`` and the
forward / backward decomposition collapses to the full QK^T matrix
without double counting (A_bwd excludes the diagonal).

Feature map ``phi = elu + 1`` matches the existing causal LA so the
choice of feature map is consistent across modes 5 (causal) and 6 (LION).
No L1 normalisation: matches the unscaled form used by RWKV-6 LION
(``lion_attention.lion_parallel_attention``) and Mamba-2 LION
(``mamba2_kernels.ssd_scan_lion``).  Sticking to the unscaled form
keeps the LION wrapper truly unified across the three architectures —
the same ``A · V`` shape with mechanism-specific decay (none, here).

The optional ``multidil_v2`` pre-mix mirrors the causal layer (Stage 11.1b):
symmetric padding by mode default, ``MAIN_DIL=1`` branch initialised to
center-tap identity, ``alpha_{d>1}=0.01`` per the v2 init-fix
(``MULTIDIL_INIT_FIX_HANDOFF.md``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import SinusoidalPE
from src.models.lion_attention import lion_parallel_attention
from src.models.mechanisms.conv_shift import MultiDilationDWConvShift


def phi_elu1(x: torch.Tensor) -> torch.Tensor:
    """Katharopoulos feature map (matches causal LA)."""
    return F.elu(x) + 1.0


class LIONLinearAttentionLayer(nn.Module):
    """Pre-norm LION-LIT linear attention layer.

    Parallels ``CausalLinearAttentionLayer`` in shape and parameter count
    so the LION matrix slot is param-matched against its causal
    counterpart.  No L1 denominator (LION-LIT unscaled form, consistent
    with RWKV-6 / Mamba-2 LION wrappers in this codebase).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        use_multidil_sym: bool = False,
        multidil_kernel: int = 3,
        multidil_dilations: tuple = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
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

        self.premix: Optional[nn.Module] = None
        if use_multidil_sym:
            self.premix = MultiDilationDWConvShift(
                d_model=d_model,
                kernel_size=multidil_kernel,
                dilations=multidil_dilations,
                padding_mode="symmetric",
                content_conditional=False,
            )
            with torch.no_grad():
                d1 = self.premix.dilations.index(1) if 1 in self.premix.dilations else 0
                b1 = self.premix.branches[d1]
                b1.weight.zero_()
                b1.weight[:, 0, multidil_kernel // 2] = 1.0

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, T, D); key_padding_mask: (B, T), True = pad."""
        B, T, D = x.shape
        H, K = self.n_heads, self.head_dim
        residual = x
        x_n = self.norm1(x)
        if self.premix is not None:
            x_n = self.premix(x_n)

        q = self.q_proj(x_n).view(B, T, H, K).transpose(1, 2)  # (B, H, T, K)
        k = self.k_proj(x_n).view(B, T, H, K).transpose(1, 2)
        v = self.v_proj(x_n).view(B, T, H, K).transpose(1, 2)

        phi_q = phi_elu1(q)
        phi_k = phi_elu1(k)

        if key_padding_mask is not None:
            keep = (~key_padding_mask).to(phi_k.dtype).view(B, 1, T, 1)
            phi_k = phi_k * keep
            v = v * keep

        # LION-LIT: zero log-decay ⇒ unified kernel reduces to bidirectional
        # phi(Q) @ phi(K)^T @ V with the diagonal correction baked in by
        # the tril/triu split (A_bwd excludes the diagonal).
        w = torch.zeros_like(phi_q)
        attn_out = lion_parallel_attention(phi_q, phi_k, v, w)  # (B, H, T, K)

        attn_out = attn_out.to(x.dtype).transpose(1, 2).reshape(B, T, D)
        attn_out = self.dropout(self.o_proj(attn_out))
        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class LIONLinearAttentionEncoder(nn.Module):
    """LION-LIT linear attention encoder; bidirectional, no carry-state.

    Matches the encoder interface used by ``asr_model.py``: forward takes
    ``(x, lengths, state=None)`` and returns ``(output, new_state=None)``.
    """

    supports_carry_state = False

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        use_multidil_sym: bool = False,
        multidil_kernel: int = 3,
        multidil_dilations: tuple = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln0 = nn.LayerNorm(d_model)
        self.pos_enc = SinusoidalPE(d_model, max_len=8000, dropout=dropout)

        self.layers = nn.ModuleList([
            LIONLinearAttentionLayer(
                d_model, n_heads, ffn_dim, dropout,
                use_multidil_sym=use_multidil_sym,
                multidil_kernel=multidil_kernel,
                multidil_dilations=multidil_dilations,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, None]:
        B, T, _ = x.shape
        x = self.ln0(x)
        x = self.pos_enc(x)
        key_padding_mask = ~(
            torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        )
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x, None
