"""Causal Linear Attention — Katharopoulos recurrent form with L1 denominator.

Stage 11.0a baseline for the causal architecture transfer study.

Per STAGE10_PLAN.md §6 Stage 11.0a:

    S_t = S_{t-1} + phi(k_t) v_t^T            (per head)
    z_t = z_{t-1} + phi(k_t)                  (per head)
    o_t = phi(q_t)^T S_t / (phi(q_t)^T z_t + eps)

with phi(x) = elu(x) + 1 (Katharopoulos 2020).  Causal: S_t, z_t depend
only on positions s <= t.

The existing ``blocks.py::LinearAttentionLayer`` is parallel bidirectional
without the explicit L1 denominator — not the object this baseline measures.

Two compute paths:

    forward_parallel — cumsum over T. Training-path. Supports state carry.
    forward_recurrent — explicit t-loop. Used by equivalence tests and as a
                        reference for single-step streaming.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import SinusoidalPE
from src.models.mechanisms.conv_shift import MultiDilationDWConvShift


def phi_elu1(x: torch.Tensor) -> torch.Tensor:
    """Katharopoulos feature map: elu(x) + 1, non-negative."""
    return F.elu(x) + 1.0


class _SymmetricDWConvShiftLA(nn.Module):
    """Stage 11.5c — single-dilation symmetric depthwise Conv1d pre-mix.

    Plain ``nn.Conv1d`` with ``padding=kernel_size // 2`` (symmetric for
    odd kernels); init ``[0.5, 0, 0.5]`` matching the Stage-3 DWConvShift
    convention — bidirectional token-shift average at step 0.  No
    per-layer alpha scalar, no multi-dilation: isolates the
    symmetric-padding effect cleanly.  Kept local to this file so the
    shared ``conv_shift.py`` isn't modified (per the multidil-fix
    handoff scope boundary).
    """

    def __init__(self, d_model: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model, bias=False,
        )
        with torch.no_grad():
            self.conv.weight.zero_()
            if kernel_size == 3:
                self.conv.weight[:, 0, 0] = 0.5
                self.conv.weight[:, 0, -1] = 0.5
            else:
                # Generic: init to center-tap identity for even kernels.
                self.conv.weight[:, 0, kernel_size // 2] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class CausalLinearAttentionLayer(nn.Module):
    """Pre-norm causal LA layer with matched-shape FFN.

    Matches the layer shell of the bidirectional variant (norm1/attn/res →
    norm2/ffn/res) so param counts line up with existing baselines modulo
    the architectural difference (no d_inner expansion).
    """

    # Class-level opt-in diagnostics hook.  When set to a callable, the
    # parallel forward invokes it with (layer, {phi_q, phi_k, S, Z, den,
    # key_padding_mask}) after computing intermediates but before returning.
    # Leave None in production — the None check is all the fast path pays.
    _diag_hook = None

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        eps: float = 1e-6,
        use_multidil_sym: bool = False,
        multidil_kernel: int = 3,
        multidil_dilations: tuple = (1, 2, 4, 8),
        use_convshift_sym: bool = False,
        convshift_sym_kernel: int = 3,
        use_lucid: bool = False,
        lucid_chunk_size: int = 64,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = eps

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

        # Stage 11.1b — optional symmetric multi-dilation pre-mix applied
        # after norm1 and before Q/K/V projections.  With alpha_1=1 and the
        # single-dilation branch initialised to center-tap identity, the
        # layer is bit-exact vs vanilla linear_attn_causal at init.
        #
        # Stage 11.5c — alternatively, a plain single-dilation symmetric
        # DWConv pre-mix (no alpha, no multi-dilation) to isolate the
        # symmetric-padding effect from the multi-dilation axis which is
        # inert under the init-gradient trap on MultiDilationDWConvShift.
        # The two flags are mutually exclusive.
        if use_multidil_sym and use_convshift_sym:
            raise ValueError(
                "use_multidil_sym and use_convshift_sym are mutually exclusive"
            )
        self.premix: Optional[nn.Module] = None
        if use_multidil_sym:
            self.premix = MultiDilationDWConvShift(
                d_model=d_model,
                kernel_size=multidil_kernel,
                dilations=multidil_dilations,
                padding_mode="symmetric",
                content_conditional=False,
            )
            # Override branch_1 init to center-tap identity so the pre-mix
            # is identity at step 0 — zero-regression vs vanilla LA.
            with torch.no_grad():
                d1 = self.premix.dilations.index(1) if 1 in self.premix.dilations else 0
                b1 = self.premix.branches[d1]
                b1.weight.zero_()
                b1.weight[:, 0, multidil_kernel // 2] = 1.0
        elif use_convshift_sym:
            self.premix = _SymmetricDWConvShiftLA(
                d_model=d_model, kernel_size=convshift_sym_kernel,
            )

        # P9 — LUCID preconditioner applied to values before they enter the
        # causal accumulator.  Uses the same `_apply_lucid_recurrent`
        # implementation as RWKV-6; decorrelates V against K's chunk-local
        # Gram matrix so that the running sum accumulates less-correlated
        # content.  Natural home for LUCID given LA's explicit attention
        # structure.  Zero-regression: tau zero-init ⇒ P ≈ I ⇒ v' ≈ v to
        # within clamp + regulariser precision.
        self.use_lucid = use_lucid
        self.lucid_chunk_size = lucid_chunk_size
        if use_lucid:
            self.lucid_temperature = nn.Parameter(torch.zeros(n_heads))

    # ----------------------------------------------------------------------
    # Parallel (cumsum) path — training + chunked inference with state carry
    # ----------------------------------------------------------------------

    def forward_parallel(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """x: (B, T, D); key_padding_mask: (B, T), True=pad."""
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

        # Stage 11 P9 — LUCID preconditioner on values (before accumulation).
        # Uses raw K (not phi(K)) for the Gram matrix, matching the paper's
        # formulation and RWKV-6's implementation.  With lucid_temperature = 0
        # at init, P reduces to the unit-diagonal identity-ish within clamp
        # precision, leaving v nearly unchanged (zero-regression within
        # regulariser tolerance).
        if self.use_lucid:
            from src.models.rwkv6_time_mix import _apply_lucid_recurrent
            temp = F.softplus(self.lucid_temperature)
            v = _apply_lucid_recurrent(
                k, v, temp, chunk_size=self.lucid_chunk_size,
            )

        if key_padding_mask is not None:
            keep = (~key_padding_mask).to(phi_k.dtype).view(B, 1, T, 1)
            phi_k = phi_k * keep
            v = v * keep

        # Outer products phi(k) v^T: (B, H, T, K, K)
        kv = torch.einsum("bhtk,bhtv->bhtkv", phi_k, v)

        # Causal cumsum: S_t = S_{t-1} + kv_t, z_t = z_{t-1} + phi_k_t.
        S = kv.cumsum(dim=2)
        Z = phi_k.cumsum(dim=2)

        if state is not None:
            S_prev = state["S"].to(S.dtype).unsqueeze(2)  # (B, H, 1, K, K)
            z_prev = state["z"].to(Z.dtype).unsqueeze(2)  # (B, H, 1, K)
            S = S + S_prev
            Z = Z + z_prev

        num = torch.einsum("bhtk,bhtkv->bhtv", phi_q, S)
        den = torch.einsum("bhtk,bhtk->bht", phi_q, Z) + self.eps
        attn_out = num / den.unsqueeze(-1)  # (B, H, T, K)

        if CausalLinearAttentionLayer._diag_hook is not None:
            CausalLinearAttentionLayer._diag_hook(
                self,
                {
                    "phi_q": phi_q,
                    "phi_k": phi_k,
                    "S": S,
                    "Z": Z,
                    "den": den,
                    "key_padding_mask": key_padding_mask,
                },
            )

        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.dropout(self.o_proj(attn_out))
        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))

        new_state: Optional[dict] = None
        if state is not None:
            # Tail state at the last valid position. For simplicity and to
            # match Mamba-2's carry discipline, use the final (padded-zeroed)
            # entry and detach — chunks trained independently, state is a
            # streaming artefact.
            new_state = {
                "S": S[:, :, -1].detach(),
                "z": Z[:, :, -1].detach(),
            }
        return x, new_state

    # ----------------------------------------------------------------------
    # Recurrent path — reference implementation for tests and streaming step
    # ----------------------------------------------------------------------

    def forward_recurrent(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Explicit t-loop — slower, used to verify forward_parallel."""
        B, T, D = x.shape
        H, K = self.n_heads, self.head_dim
        residual = x
        x_n = self.norm1(x)
        if self.premix is not None:
            x_n = self.premix(x_n)

        q = self.q_proj(x_n).view(B, T, H, K).transpose(1, 2)
        k = self.k_proj(x_n).view(B, T, H, K).transpose(1, 2)
        v = self.v_proj(x_n).view(B, T, H, K).transpose(1, 2)
        phi_q = phi_elu1(q)
        phi_k = phi_elu1(k)

        if state is not None:
            S = state["S"].to(phi_q.dtype).clone()
            Z = state["z"].to(phi_q.dtype).clone()
        else:
            S = torch.zeros(B, H, K, K, device=x.device, dtype=phi_q.dtype)
            Z = torch.zeros(B, H, K, device=x.device, dtype=phi_q.dtype)

        outs = []
        for t in range(T):
            if key_padding_mask is not None:
                keep_t = (~key_padding_mask[:, t]).to(phi_q.dtype).view(B, 1, 1)
                phi_k_t = phi_k[:, :, t] * keep_t
                v_t = v[:, :, t] * keep_t.view(B, 1, 1)
            else:
                phi_k_t = phi_k[:, :, t]
                v_t = v[:, :, t]
            S = S + torch.einsum("bhk,bhv->bhkv", phi_k_t, v_t)
            Z = Z + phi_k_t
            num_t = torch.einsum("bhk,bhkv->bhv", phi_q[:, :, t], S)
            den_t = torch.einsum("bhk,bhk->bh", phi_q[:, :, t], Z) + self.eps
            outs.append(num_t / den_t.unsqueeze(-1))
        attn_out = torch.stack(outs, dim=2)  # (B, H, T, K)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.dropout(self.o_proj(attn_out))
        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))

        new_state: Optional[dict] = None
        if state is not None:
            new_state = {"S": S.detach(), "z": Z.detach()}
        return x, new_state

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        return self.forward_parallel(x, key_padding_mask=key_padding_mask, state=state)


class CausalLinearAttentionEncoder(nn.Module):
    """Stack of causal LA layers; matches Mamba2Encoder / CausalTransformer
    signature so asr_model.py / training/ need no changes."""

    supports_carry_state = True

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
        use_convshift_sym: bool = False,
        convshift_sym_kernel: int = 3,
        use_lucid: bool = False,
        lucid_chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Matches Mamba-2 / causal-transformer: ln0 on layer-0 input, then PE.
        # PE is included because LA has no inherent positional bias beyond
        # t-ordered cumsum; Mamba-2 / causal-transformer in this repo also
        # include PE, so the baselines share this.
        self.ln0 = nn.LayerNorm(d_model)
        self.pos_enc = SinusoidalPE(d_model, max_len=8000, dropout=dropout)

        self.layers = nn.ModuleList([
            CausalLinearAttentionLayer(
                d_model, n_heads, ffn_dim, dropout,
                use_multidil_sym=use_multidil_sym,
                multidil_kernel=multidil_kernel,
                multidil_dilations=multidil_dilations,
                use_convshift_sym=use_convshift_sym,
                convshift_sym_kernel=convshift_sym_kernel,
                use_lucid=use_lucid,
                lucid_chunk_size=lucid_chunk_size,
            )
            for _ in range(n_layers)
        ])

    def init_state(self, batch_size: int, device: torch.device) -> dict:
        K = self.head_dim
        H = self.n_heads
        return {
            "layers": [
                {
                    "S": torch.zeros(batch_size, H, K, K, device=device, dtype=torch.float32),
                    "z": torch.zeros(batch_size, H, K, device=device, dtype=torch.float32),
                }
                for _ in range(self.n_layers)
            ],
            "offset": 0,
        }

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        B, T, _ = x.shape
        offset = state["offset"] if state is not None else 0
        layer_states = state["layers"] if state is not None else None

        x = self.ln0(x)
        x = self.pos_enc(x, offset=offset)

        key_padding_mask = ~(
            torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        )

        new_layer_states: Optional[list] = [] if state is not None else None
        for i, layer in enumerate(self.layers):
            ls = layer_states[i] if layer_states is not None else None
            x, ns = layer(x, key_padding_mask=key_padding_mask, state=ls)
            if new_layer_states is not None:
                new_layer_states.append(ns)

        new_state: Optional[dict] = None
        if new_layer_states is not None:
            new_state = {"layers": new_layer_states, "offset": offset + T}
        return x, new_state
