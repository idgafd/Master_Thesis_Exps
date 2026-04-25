"""Stage 11.2b — Linear Attention + RSE (block-complex transition) + viscosity.

Extends ``CausalLinearAttentionLayer`` with a block-complex state transition
matching the Stage-3 RSE mechanism on RWKV-6, ported to the LA recurrence.

Mechanism (per STAGE10_PLAN.md §6 Stage 11.2b):

    * Pair adjacent k/q channels into complex pairs:
      k_c[b] = k[2b] + i k[2b+1], q_c[b] = q[2b] + i q[2b+1].
    * Complex state per (head, block, value-channel):
          c_t[b, v] = z_t[b] * c_{t-1}[b, v] + k_c[t, b] * v[t, v]
      where z_t = exp(-lambda_eff + i theta_t),
      lambda_eff = lambda_base + eta * theta_t^2  (Stage-5 viscosity).
    * Output: y[t, v] = Re( sum_b conj(q_c[t, b]) * c_t[b, v] ).
    * No explicit L1 denominator — the exponential decay replaces the
      normalisation role played by the denominator in vanilla LA.

Zero-regression contract: with theta_base = 0, LoRA weights = 0, and
eta = 0 at init, the mechanism reduces to a per-(head, block) real-decay
linear recurrence.  This is NOT bit-exact vs vanilla LA (which has no
decay); RSE on LA deliberately adds bounded decay as part of the
mechanism — decay per §6 is what replaces the L1 denominator.

Computation follows RWKV-6's chunked complex scan in
``rwkv6_time_mix.py::_forward_recurrent_rse``: within a chunk,
cumlog_z = cumsum(log z), pairwise diffs give A[t,s], masked to causal
triangle, outer products with v form the intra-chunk contribution;
inter-chunk state is carried complex.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import SinusoidalPE
from src.models.linear_attn_causal import CausalLinearAttentionLayer, phi_elu1


class CausalLinearAttentionRSELayer(CausalLinearAttentionLayer):
    """LA layer with block-complex SO(2) × R+ transition and optional viscosity.

    Projection shapes are inherited from ``CausalLinearAttentionLayer`` —
    only the scan and the RSE parameters are new.
    """

    # Disable the base class's _diag_hook path (uses different intermediates).
    _diag_hook = None

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        eps: float = 1e-6,
        rse_theta_init_scale: float = math.pi / 16,
        rse_theta_clip: float = math.pi / 2,
        rse_theta_lora_dim: int = 48,
        rse_viscosity: bool = True,
        rse_lambda_init_range: Tuple[float, float] = (0.5, 6.0),
        rse_chunk_size: int = 64,
        use_multidil_sym: bool = False,
    ) -> None:
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            eps=eps,
            use_multidil_sym=use_multidil_sym,
        )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"RSE requires even head_dim for 2x2 blocks; got {self.head_dim}"
            )
        self.n_blocks = self.head_dim // 2
        self.theta_clip = rse_theta_clip
        self.rse_viscosity = rse_viscosity
        self.chunk_size = rse_chunk_size

        # ── Per-(head, block) base rotation angle + LoRA ──────────────────
        theta_init = torch.empty(n_heads, self.n_blocks).uniform_(
            -rse_theta_init_scale, rse_theta_init_scale
        )
        self.theta_base = nn.Parameter(theta_init)  # (H, Bk)
        # Zero-init LoRA: down-proj W1=0, up-proj W2~U(-0.01, 0.01).
        # At init LoRA contribution = tanh(x @ W1) @ W2 = 0.
        self.theta_w1 = nn.Parameter(torch.zeros(d_model, rse_theta_lora_dim))
        self.theta_w2 = nn.Parameter(
            torch.empty(rse_theta_lora_dim, n_heads * self.n_blocks).uniform_(
                -0.01, 0.01
            )
        )

        # ── Per-(head, block) base log-decay ──────────────────────────────
        # lambda_base > 0 gives decay e^{-lambda_base} ∈ (0, 1) per step.
        # Uniform over rse_lambda_init_range matches Stage-3 RSE's range
        # (roughly 0.5 to 6 ≈ e^{-0.5}..e^{-6} = 0.6..0.0025 per step).
        # Unlike RWKV-6 which reuses its native decay, LA has none — we
        # introduce decay as part of the mechanism.
        lo, hi = rse_lambda_init_range
        lambda_init = torch.empty(n_heads, self.n_blocks).uniform_(lo, hi)
        self.lambda_base = nn.Parameter(lambda_init)

        # ── Stage 5 viscosity coupling ────────────────────────────────────
        # eta_{h,b} scalar per head × block, zero-init ⇒ identical to pure
        # RSE at init; lambda_eff = lambda_base + eta * theta_t^2 couples
        # rotation and decay (Rayleigh dissipation).
        if rse_viscosity:
            self.viscosity_eta = nn.Parameter(torch.zeros(n_heads, self.n_blocks))

    # ----------------------------------------------------------------------
    # RSE chunked scan (parallel within chunks, serial between chunks)
    # ----------------------------------------------------------------------

    def _compute_theta(self, x_n: torch.Tensor) -> torch.Tensor:
        """theta_t = clip(theta_base + LoRA(x_n), -clip, +clip).

        x_n: (B, T, D).  Returns (B, H, T, Bk).
        """
        B, T, _ = x_n.shape
        lora = torch.tanh(x_n @ self.theta_w1) @ self.theta_w2  # (B, T, H*Bk)
        lora = lora.view(B, T, self.n_heads, self.n_blocks)     # (B, T, H, Bk)
        theta = self.theta_base.view(1, 1, self.n_heads, self.n_blocks) + lora
        theta = torch.clamp(theta, -self.theta_clip, self.theta_clip)
        # Permute to (B, H, T, Bk) for the scan.
        return theta.permute(0, 2, 1, 3).contiguous()

    def forward_parallel(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """x: (B, T, D); key_padding_mask: (B, T), True=pad."""
        B, T, D = x.shape
        H, K = self.n_heads, self.head_dim
        Bk = self.n_blocks
        residual = x
        x_n = self.norm1(x)
        if self.premix is not None:
            x_n = self.premix(x_n)

        # Projections — q / k / v all (B, H, T, K).
        q = self.q_proj(x_n).view(B, T, H, K).transpose(1, 2)
        k = self.k_proj(x_n).view(B, T, H, K).transpose(1, 2)
        v = self.v_proj(x_n).view(B, T, H, K).transpose(1, 2)
        phi_k = phi_elu1(k)
        # Query side: no phi needed — we read out via inner product on the
        # complex state.  Keeping q linear (Proposal A §3 convention).

        # Mask pad positions out of the k/v contributions.
        if key_padding_mask is not None:
            keep = (~key_padding_mask).to(phi_k.dtype).view(B, 1, T, 1)
            phi_k = phi_k * keep
            v = v * keep

        # Pair K dim into complex pairs (Bk complex channels per head).
        q_pairs = q.float().view(B, H, T, Bk, 2)
        k_pairs = phi_k.float().view(B, H, T, Bk, 2)
        q_c = torch.complex(q_pairs[..., 0], q_pairs[..., 1])  # (B, H, T, Bk)
        k_c = torch.complex(k_pairs[..., 0], k_pairs[..., 1])  # (B, H, T, Bk)
        v_f = v.float()                                         # (B, H, T, K=Bk*2)

        # Per-step theta_t (clipped) and lambda_eff.
        theta = self._compute_theta(x_n)                        # (B, H, T, Bk)
        lambda_base_hb = self.lambda_base.view(1, H, 1, Bk)     # (1, H, 1, Bk)
        lambda_eff = lambda_base_hb.expand(B, H, T, Bk).float()
        if self.rse_viscosity:
            lambda_eff = lambda_eff + (
                self.viscosity_eta.view(1, H, 1, Bk).float() * theta.float() ** 2
            )
        # log_z = -lambda_eff + i theta
        log_z = torch.complex(-lambda_eff, theta.float())       # (B, H, T, Bk)

        # Initial complex state.
        if state is not None:
            c_state = state["c"].to(torch.complex64)
        else:
            c_state = torch.zeros(B, H, Bk, K, dtype=torch.complex64, device=x.device)

        out = torch.zeros(B, H, T, K, dtype=torch.float32, device=x.device)

        cur = 0
        while cur < T:
            tc = min(self.chunk_size, T - cur)
            log_z_c = log_z[:, :, cur:cur + tc]                 # (B, H, tc, Bk)
            k_c_c = k_c[:, :, cur:cur + tc]                     # (B, H, tc, Bk)
            q_c_c = q_c[:, :, cur:cur + tc]                     # (B, H, tc, Bk)
            v_c = v_f[:, :, cur:cur + tc]                       # (B, H, tc, K)

            # Cumulative log-z (inclusive).
            cumlog = log_z_c.cumsum(dim=2)                      # (B, H, tc, Bk)

            # Within-chunk attention:
            # A[t, s, b] = exp( cumlog[t] - cumlog[s] ) for s <= t, 0 otherwise.
            diff = cumlog.unsqueeze(3) - cumlog.unsqueeze(2)    # (B, H, tc, tc, Bk)
            mask = torch.tril(
                torch.ones(tc, tc, device=x.device, dtype=torch.bool)
            ).view(1, 1, tc, tc, 1)
            # Clamp real part before exp to avoid -inf, then zero the upper
            # triangle exactly via torch.where (matches RWKV-6 Stage 5 fix).
            real_part = diff.real.masked_fill(~mask, -60.0)
            A_raw = torch.exp(torch.complex(real_part, diff.imag))
            A = torch.where(mask, A_raw, torch.zeros_like(A_raw))  # (B, H, tc, tc, Bk)

            # Intra-chunk S[t, b, v] = sum_s A[t,s,b] * k_c[s, b] * v[s, v].
            scaled_k = A * k_c_c.unsqueeze(2)                    # (B, H, tc, tc, Bk)
            v_c_complex = v_c.to(torch.complex64)
            S_intra = torch.einsum(
                "bhtsk,bhsv->bhtkv", scaled_k, v_c_complex
            )                                                   # (B, H, tc, Bk, K)

            # Inter-chunk: prior state contribution exp(cumlog[t]) * c_state.
            decay_to_t = torch.exp(cumlog)                      # (B, H, tc, Bk)
            prior_contrib = decay_to_t.unsqueeze(-1) * c_state.unsqueeze(2)
            S_total = prior_contrib + S_intra                   # (B, H, tc, Bk, K)

            # Readout y[t, v] = Re( sum_b conj(q_c[t, b]) * S_total[t, b, v] ).
            r_contract = q_c_c.conj()
            y_chunk = torch.einsum(
                "bhtk,bhtkv->bhtv", r_contract, S_total
            ).real                                              # (B, H, tc, K)

            out[:, :, cur:cur + tc] = y_chunk
            c_state = S_total[:, :, -1]                         # (B, H, Bk, K)
            cur += tc

        attn_out = out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.dropout(self.o_proj(attn_out))
        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))

        new_state: Optional[dict] = None
        if state is not None:
            new_state = {"c": c_state.detach()}
        return x, new_state

    # Keep forward_recurrent inherited from parent unusable for RSE mode;
    # hide it behind an explicit error since the math differs.
    def forward_recurrent(self, *args, **kwargs):
        raise NotImplementedError(
            "CausalLinearAttentionRSELayer uses a dedicated chunked complex "
            "scan; forward_recurrent is not ported to the RSE path."
        )


class CausalLinearAttentionRSEEncoder(nn.Module):
    """Encoder stack using RSE layers; matches the signature of the other
    encoder classes in this repo."""

    supports_carry_state = True

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        rse_theta_init_scale: float = math.pi / 16,
        rse_theta_clip: float = math.pi / 2,
        rse_theta_lora_dim: int = 48,
        rse_viscosity: bool = True,
        rse_lambda_init_range: Tuple[float, float] = (0.5, 6.0),
        use_multidil_sym: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln0 = nn.LayerNorm(d_model)
        self.pos_enc = SinusoidalPE(d_model, max_len=8000, dropout=dropout)

        self.layers = nn.ModuleList([
            CausalLinearAttentionRSELayer(
                d_model, n_heads, ffn_dim, dropout,
                rse_theta_init_scale=rse_theta_init_scale,
                rse_theta_clip=rse_theta_clip,
                rse_theta_lora_dim=rse_theta_lora_dim,
                rse_viscosity=rse_viscosity,
                rse_lambda_init_range=rse_lambda_init_range,
                use_multidil_sym=use_multidil_sym,
            )
            for _ in range(n_layers)
        ])

    def init_state(self, batch_size: int, device: torch.device) -> dict:
        K = self.head_dim
        Bk = K // 2
        H = self.n_heads
        return {
            "layers": [
                {
                    "c": torch.zeros(
                        batch_size, H, Bk, K, dtype=torch.complex64, device=device,
                    ),
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
