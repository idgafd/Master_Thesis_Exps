"""LION-LIT and LION-S bidirectional Linear Attention encoders.

Two decay variants per the LION paper (Afzal et al. 2025) Table 1:

  - ``decay_mode="lit"`` — LION-LIT (λ=1, no decay).  Natural mapping
    of Katharopoulos LinAtt; default.
  - ``decay_mode="s"``   — LION-S (per-head selective σ-decay, mirrors
    Gated RFA → LION-S).  Used as a control for the "decay is the
    missing piece" hypothesis on LA: causal LA at 50 ep landed test
    0.1879; LA LION-LIT vanilla landed dev ~0.30 / test ~0.30
    (worse — no per-position weighting in the bidirectional sum
    leads to attention smearing on CTC ASR).

Mathematical form (paper Eq. 8 + Table 1 "+ Scaling" entry for LinAtt)
---------------------------------------------------------------------

    Y = SCALE(  phi(Q) @ phi(K)^T  ) @ V

where SCALE divides each row of the attention matrix by its row-sum.
In the bidirectional case (no decay), the row-sum factorises:

    sum_j phi(Q)[t] · phi(K)[j]  =  phi(Q)[t] · sum_j phi(K)[j]

so SCALE reduces to dividing the output by ``phi(Q) · phi(K).sum(j)``.
Matches the L1 denominator pattern used by the existing causal LA
(``linear_attn_causal.py``: ``Z = phi_k.cumsum``) and the naive bidir
LA in ``blocks.py::LinearAttentionLayer``.

Implementation reuses the unified ``lion_attention.lion_parallel_attention``
kernel with ``w = 0`` for the numerator (gives the bidirectional
``phi(Q) @ phi(K)^T @ V`` with diagonal correction baked in by the
tril/triu split), then divides by the L1 denominator.

Why SCALE matters here: RWKV-6 / Mamba-2 LION wrappers run unscaled
because their data-dependent decay (λ < 1, cumulative product) bounds
output magnitudes naturally.  LA has no decay; without SCALE the
``phi(Q) @ phi(K)^T`` row-sum scales as O(T · head_dim), output
magnitudes blow up, training diverges.  Pre-fix LA LION vanilla
landed at dev 0.4551 vs causal LA 0.1879 — fixed by adding SCALE.

Feature map ``phi = elu + 1`` matches the existing causal LA, so mode 5
(causal) and mode 6 (LION) share both feature map and SCALE convention.

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

import math

from src.models.components import SinusoidalPE
from src.models.lion_attention import (
    lion_attention_with_lucid,
    lion_parallel_attention,
)
from src.models.mechanisms.conv_shift import MultiDilationDWConvShift


def phi_elu1(x: torch.Tensor) -> torch.Tensor:
    """Katharopoulos feature map (matches causal LA)."""
    return F.elu(x) + 1.0


class LIONLinearAttentionLayer(nn.Module):
    """Pre-norm LION linear attention layer.

    Parallels ``CausalLinearAttentionLayer`` in shape and parameter count
    so the LION matrix slot is param-matched against its causal
    counterpart.

    ``decay_mode``:
      - ``"lit"``: λ = 1 (no decay).  Bidirectional QK^T with SCALE.
        Denominator factorises as phi(Q)·sum_j phi(K_j) — cheap.
      - ``"s"``:   per-head selective decay λ_t = σ(W_λ · x_t + b_λ),
        broadcast across the K-dim.  Bidirectional cumulative-product
        decay matrix; SCALE denominator does NOT factorise — computed
        via the v-extended trick (concatenate a column of ones to V,
        the extra output column gives the row-sum).  Init: b_λ = +5
        ⇒ σ ≈ 0.993 ⇒ near-LION-LIT at step 0 (zero-regression).
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
        eps: float = 1e-6,
        decay_mode: str = "lit",
        use_lucid: bool = False,
        lucid_chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        if decay_mode not in {"lit", "s"}:
            raise ValueError(f"decay_mode must be 'lit' or 's', got {decay_mode!r}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = eps
        self.decay_mode = decay_mode
        self.use_lucid = use_lucid
        self.lucid_chunk_size = lucid_chunk_size

        if use_lucid:
            # Init so softplus(param) = 1.0 — paper-faithful unit scaling
            # (matches RWKV-6 LION × LUCID and Mamba-2 LUCID-c init).
            self.lucid_temperature = nn.Parameter(
                torch.full((n_heads,), math.log(math.e - 1))
            )

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # LION-S: per-head selective decay λ_t = σ(W_λ x + b_λ).  Init so
        # σ(b_λ) ≈ 1 (b_λ = +5 ⇒ 0.9933) — zero-regression vs LION-LIT.
        if decay_mode == "s":
            self.decay_proj = nn.Linear(d_model, n_heads)
            with torch.no_grad():
                self.decay_proj.weight.zero_()
                self.decay_proj.bias.fill_(5.0)
        else:
            self.decay_proj = None

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

        if self.decay_mode == "lit":
            # LION-LIT numerator: w=0 ⇒ unified kernel reduces to
            # bidirectional phi(Q) @ phi(K)^T @ V with the diagonal
            # correction baked in by the tril/triu split.
            w = torch.zeros_like(phi_q)
            if self.use_lucid:
                # P^{-1} preconditioning on V using phi_k as key source,
                # then bidirectional attention.  P has unit diagonal at any
                # tau; init tau s.t. softplus(tau)=1 matches the paper.
                temp = F.softplus(self.lucid_temperature)
                attn_out = lion_attention_with_lucid(
                    phi_q, phi_k, v, w, temp,
                    chunk_size=self.lucid_chunk_size,
                )
            else:
                attn_out = lion_parallel_attention(phi_q, phi_k, v, w)  # (B, H, T, K)

            # SCALE: row-sum factorises as phi(Q)[t] · sum_j phi(K)[j].
            # The denominator is the row-sum of A (independent of LUCID's
            # value preconditioner), so the same divisor applies whether
            # LUCID is on or off.
            phi_k_sum = phi_k.float().sum(dim=2, keepdim=True)  # (B, H, 1, K)
            denom = (phi_q.float() * phi_k_sum).sum(dim=-1, keepdim=True) + self.eps
            attn_out = attn_out / denom
        else:
            # LION-S: λ_t = σ(W_λ x + b_λ) ∈ (0, 1)^(B, T, H), broadcast
            # to (B, H, T, K).  log-decay w = log(λ); the kernel applies
            # cumprod via cumsum-of-log + exp internally.
            lam = torch.sigmoid(self.decay_proj(x_n))  # (B, T, H)
            log_lam = torch.log(lam.clamp(min=1e-8))   # (B, T, H), <= 0
            w = log_lam.permute(0, 2, 1).unsqueeze(-1).expand(B, H, T, K)

            # SCALE for non-factorising decay: numerator and denominator
            # both go through lion_parallel_attention with the same (phi_q,
            # phi_k, w).  For the denominator we set v = constant column 1
            # so the output is sum_j A[t, j] · 1 = the row-sum (same value
            # across all K dims; take the first column).
            ones_v = torch.ones(B, H, T, K, dtype=v.dtype, device=v.device)
            if key_padding_mask is not None:
                ones_v = ones_v * keep
            if self.use_lucid:
                # LION-S × LUCID: precondition V via the unit-diagonal
                # preconditioner before the bidirectional attention.
                # SCALE denominator stays unconditioned on LUCID — it is
                # the row-sum of A and depends only on (phi_q, phi_k, w),
                # not on the value tensor.
                temp = F.softplus(self.lucid_temperature)
                attn_num = lion_attention_with_lucid(
                    phi_q, phi_k, v, w, temp,
                    chunk_size=self.lucid_chunk_size,
                )
            else:
                attn_num = lion_parallel_attention(phi_q, phi_k, v, w)
            attn_den = lion_parallel_attention(phi_q, phi_k, ones_v, w)
            denom = attn_den[..., :1].clamp(min=self.eps)  # (B, H, T, 1)
            attn_out = attn_num / denom

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
        decay_mode: str = "lit",
        use_lucid: bool = False,
        lucid_chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.decay_mode = decay_mode

        self.ln0 = nn.LayerNorm(d_model)
        self.pos_enc = SinusoidalPE(d_model, max_len=8000, dropout=dropout)

        self.layers = nn.ModuleList([
            LIONLinearAttentionLayer(
                d_model, n_heads, ffn_dim, dropout,
                use_multidil_sym=use_multidil_sym,
                multidil_kernel=multidil_kernel,
                multidil_dilations=multidil_dilations,
                decay_mode=decay_mode,
                use_lucid=use_lucid,
                lucid_chunk_size=lucid_chunk_size,
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
