"""Stage 11.2a — Mamba-2 + RSE (block-complex SSD transition) + viscosity.

Ports the Stage-3 RSE mechanism (block-complex SO(2) × R+ transition) into
Mamba-2's selective-Δt state-space scan.  Per §6 Stage 11.2a of
STAGE10_PLAN.md.

Mechanism:

    * Pair adjacent d_state channels into complex pairs:
      B_c[..., b] = B[..., 2b] + i B[..., 2b+1],   similarly for C.
    * The continuous A becomes per-(head, block) COMPLEX:
          A_cont[h, b] = -lambda_base[h, b] + i theta_base[h, b]
      with theta data-dependent via LoRA and lambda coupled by viscosity:
          theta_t[h, b] = clip( theta_base[h, b] + LoRA(x_t)[h, b] )
          lambda_eff_t[h, b] = lambda_base[h, b] + eta[h, b] * theta_t[h, b]^2
          A_cont_t[h, b] = -lambda_eff_t[h, b] + i theta_t[h, b]
    * Discretise by dt:  dA_t[h, b] = exp(dt_t[h] * A_cont_t[h, b]).  (Note
      dt still per-head scalar; the block extension is only on A.)
    * Complex state  h_t[B, H, P, Bk]:
          h_t[b] = dA_t[h, b] * h_{t-1}[b] + dt_t[h] * B_c[t, h, b] * x_t[h, p]
    * Output:  y_t[h, p] = Re( sum_b C_c[t, h, b] * h_t[h, p, b] )
      Then the usual D * x_t skip and z gating / out_proj.

Zero-regression-at-init-ish contract: theta_base ~ U(-pi/16, pi/16) (small,
symmetry-breaking); LoRA weights zero-init ⇒ no data-dependent theta at
step 0.  lambda_base drawn from the same [1, 16] range as Mamba-2's A_log
so the effective decay at init matches vanilla Mamba-2 in distribution.
This is not bit-exact vs vanilla Mamba-2 because the scalar A_cont per head
is replaced by a complex A_cont per (head, block); the mechanism
deliberately replaces the scalar decay with block-complex dynamics.

Computation: pure-PyTorch chunked complex scan, same skeleton as
``linear_attn_rse.py`` adapted for Mamba-2's (B, L, H, P) X and (B, L, H, N)
B/C.  No Triton kernel.  Supports ``mode='recurrent'`` only.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import SinusoidalPE
from src.models.mamba2_block import Mamba2Block


class Mamba2RSEBlock(nn.Module):
    """Mamba-2 block with RSE-extended complex SSD scan + viscosity.

    Reuses Mamba-2's projections (in_proj, conv1d, out_proj, RMSNorm, gate,
    D skip) from the parent ``Mamba2Block``; only replaces the SSD scan.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        headdim: int = 64,
        expand: int = 2,
        ngroups: int = 1,
        chunk_size: int = 64,
        bias: bool = False,
        conv_bias: bool = True,
        dtype: torch.dtype = torch.float32,
        # RSE / viscosity params
        rse_theta_init_scale: float = math.pi / 16,
        rse_theta_clip: float = math.pi / 2,
        rse_theta_lora_dim: int = 48,
        rse_viscosity: bool = True,
        A_init_range: Tuple[float, float] = (1.0, 16.0),
        mode: str = "recurrent",
    ):
        super().__init__()
        if d_state % 2 != 0:
            raise ValueError(
                f"RSE requires even d_state for 2x2 blocks; got {d_state}"
            )
        if mode not in ("recurrent", "lion"):
            raise ValueError(
                f"Mamba2RSEBlock supports mode='recurrent' or 'lion'; got {mode!r}"
            )
        self.mode = mode
        self.chunk_size = chunk_size

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim
        self.expand = expand
        self.d_inner = expand * d_model
        if self.d_inner % headdim != 0:
            raise ValueError(
                f"d_inner ({self.d_inner}) must be divisible by headdim ({headdim})"
            )
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups
        if self.nheads % ngroups != 0:
            raise ValueError(
                f"nheads ({self.nheads}) must be divisible by ngroups ({ngroups})"
            )
        self._heads_per_group = self.nheads // ngroups
        self.n_blocks = d_state // 2  # Bk

        # ── projections (same as vanilla) ───────────────────────────────
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias, dtype=dtype)

        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.conv_dim = conv_dim
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim, out_channels=conv_dim,
            kernel_size=d_conv, groups=conv_dim, padding=d_conv - 1,
            bias=conv_bias, dtype=dtype,
        )

        # dt bias (softplus) — same as vanilla.
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt.to(dtype))
        self.dt_bias._no_weight_decay = True

        # ── RSE: per-(head, block) base lambda + theta + LoRA + eta ─────
        # lambda_base drawn from the same range as vanilla's A_log so the
        # per-block decays spread across [e^{-1}, e^{-16}] at init.
        lo, hi = A_init_range
        lambda_init = torch.empty(self.nheads, self.n_blocks).uniform_(lo, hi)
        self.lambda_base = nn.Parameter(lambda_init.to(dtype))
        self.lambda_base._no_weight_decay = True

        theta_init = torch.empty(self.nheads, self.n_blocks).uniform_(
            -rse_theta_init_scale, rse_theta_init_scale
        )
        self.theta_base = nn.Parameter(theta_init.to(dtype))
        self.theta_w1 = nn.Parameter(
            torch.zeros(d_model, rse_theta_lora_dim, dtype=dtype)
        )
        self.theta_w2 = nn.Parameter(
            torch.empty(
                rse_theta_lora_dim, self.nheads * self.n_blocks, dtype=dtype
            ).uniform_(-0.01, 0.01)
        )
        self.theta_clip = rse_theta_clip

        self.rse_viscosity = rse_viscosity
        if rse_viscosity:
            self.viscosity_eta = nn.Parameter(
                torch.zeros(self.nheads, self.n_blocks, dtype=dtype)
            )

        # D skip — per-head real scalar, same as vanilla.
        self.D = nn.Parameter(torch.ones(self.nheads, dtype=dtype))
        self.D._no_weight_decay = True

        # Output stack (same as vanilla).
        self.norm = nn.RMSNorm(self.d_inner, dtype=dtype)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, dtype=dtype)

    # ------------------------------------------------------------------
    # Carry-state helpers
    # ------------------------------------------------------------------

    @property
    def supports_carry_state(self) -> bool:
        return True

    def init_state(self, batch_size: int, device: torch.device) -> dict:
        """Complex SSM state: (B, H, P, Bk) complex."""
        return {
            "conv": torch.zeros(
                batch_size, self.conv_dim, self.d_conv - 1,
                device=device, dtype=torch.float32,
            ),
            "ssm": torch.zeros(
                batch_size, self.nheads, self.headdim, self.n_blocks,
                device=device, dtype=torch.complex64,
            ),
        }

    # ------------------------------------------------------------------
    # RSE chunked complex SSD scan
    # ------------------------------------------------------------------

    def _compute_theta(self, x: torch.Tensor) -> torch.Tensor:
        """theta_t = clip(theta_base + LoRA(x), -clip, +clip).

        x: (B, L, D).  Returns (B, L, H, Bk) real.
        """
        B, L, _ = x.shape
        lora = torch.tanh(x @ self.theta_w1) @ self.theta_w2  # (B, L, H*Bk)
        lora = lora.view(B, L, self.nheads, self.n_blocks)
        theta = self.theta_base.view(1, 1, self.nheads, self.n_blocks) + lora
        return torch.clamp(theta, -self.theta_clip, self.theta_clip)

    def _rse_scan(
        self,
        x_heads: torch.Tensor,    # (B, L, H, P)
        dt: torch.Tensor,         # (B, L, H)
        B_c: torch.Tensor,        # (B, L, H, Bk) complex
        C_c: torch.Tensor,        # (B, L, H, Bk) complex
        theta: torch.Tensor,      # (B, L, H, Bk) real (clipped)
        ssm_state: Optional[torch.Tensor],  # (B, H, P, Bk) complex or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunked complex SSD scan.  Returns (y_heads, final_state).

        y_heads: (B, L, H, P) real
        final_state: (B, H, P, Bk) complex
        """
        Bsz, L, H, P = x_heads.shape
        Bk = self.n_blocks
        device = x_heads.device

        # Continuous A per step: -lambda_eff + i * theta.
        lambda_base = self.lambda_base.view(1, 1, H, Bk).float()
        if self.rse_viscosity:
            lambda_eff = lambda_base + (
                self.viscosity_eta.view(1, 1, H, Bk).float() * theta.float() ** 2
            )
        else:
            lambda_eff = lambda_base.expand(Bsz, L, H, Bk)
        A_cont_c = torch.complex(-lambda_eff.float(), theta.float())  # (B, L, H, Bk)

        # Discretised log-z per step: dt * A_cont  — dt per-head scalar broadcast
        # over Bk.
        log_z = A_cont_c * dt.float().unsqueeze(-1)              # (B, L, H, Bk)

        # Discretised values: dt * x per (B, L, H, P).
        x_disc = (x_heads.float() * dt.float().unsqueeze(-1))    # (B, L, H, P)

        # Initial state.
        if ssm_state is None:
            state = torch.zeros(
                Bsz, H, P, Bk, dtype=torch.complex64, device=device
            )
        else:
            state = ssm_state.to(torch.complex64)

        y_out = torch.zeros(Bsz, L, H, P, dtype=torch.float32, device=device)

        cur = 0
        while cur < L:
            tc = min(self.chunk_size, L - cur)
            log_z_c = log_z[:, cur:cur + tc]                      # (B, tc, H, Bk)
            B_cc = B_c[:, cur:cur + tc]                           # (B, tc, H, Bk)
            C_cc = C_c[:, cur:cur + tc]                           # (B, tc, H, Bk)
            x_cc = x_disc[:, cur:cur + tc]                        # (B, tc, H, P)

            # Cumulative log-z (inclusive) along T.
            cumlog = log_z_c.cumsum(dim=1)                        # (B, tc, H, Bk)

            # Within-chunk attention: A[t, s, h, b] = exp(cumlog[t] - cumlog[s])
            # for s <= t, else 0.
            diff = cumlog.unsqueeze(2) - cumlog.unsqueeze(1)      # (B, tc, tc, H, Bk)
            mask = torch.tril(
                torch.ones(tc, tc, device=device, dtype=torch.bool)
            ).view(1, tc, tc, 1, 1)
            real_part = diff.real.masked_fill(~mask, -60.0)
            A_raw = torch.exp(torch.complex(real_part, diff.imag))
            A_att = torch.where(mask, A_raw, torch.zeros_like(A_raw))  # (B,tc,tc,H,Bk)

            # Intra-chunk S[t, h, p, b] =
            #   sum_s A[t,s,h,b] * B_c[s,h,b] * x_disc[s,h,p]
            # Do this in two steps to keep memory reasonable.
            scaled_B = A_att * B_cc.unsqueeze(2)                  # (B, tc, tc, H, Bk)
            x_cc_complex = x_cc.to(torch.complex64)               # (B, tc, H, P)
            S_intra = torch.einsum(
                "btshk,bshp->bthpk", scaled_B, x_cc_complex
            )                                                     # (B, tc, H, P, Bk)

            # Inter-chunk: prior state contribution.
            decay_to_t = torch.exp(cumlog)                        # (B, tc, H, Bk)
            prior_contrib = decay_to_t.unsqueeze(3) * state.unsqueeze(1)
            # state: (B, H, P, Bk); unsqueeze→(B,1,H,P,Bk).  decay: (B,tc,H,Bk)
            # decay.unsqueeze(3): (B, tc, H, 1, Bk) — broadcast against (B, 1, H, P, Bk).
            # Product shape: (B, tc, H, P, Bk).  Good.

            S_total = prior_contrib + S_intra                     # (B, tc, H, P, Bk)

            # Readout y[t, h, p] = Re( sum_b C_c[t, h, b] * S_total[t, h, p, b] )
            y_chunk = torch.einsum(
                "bthk,bthpk->bthp", C_cc, S_total
            ).real                                                # (B, tc, H, P)

            y_out[:, cur:cur + tc] = y_chunk
            state = S_total[:, -1]                                # (B, H, P, Bk)
            cur += tc

        return y_out, state

    # ------------------------------------------------------------------
    # LION bidirectional RSE scan
    # ------------------------------------------------------------------

    def _rse_scan_lion(
        self,
        x_heads: torch.Tensor,    # (B, L, H, P)
        dt: torch.Tensor,         # (B, L, H)
        B_c: torch.Tensor,        # (B, L, H, Bk) complex
        C_c: torch.Tensor,        # (B, L, H, Bk) complex
        theta: torch.Tensor,      # (B, L, H, Bk) real (clipped)
    ) -> torch.Tensor:
        """Bidirectional LION-style RSE scan for Mamba-2.

        Same per-step complex dynamics as ``_rse_scan`` (log_z = dt · A_cont,
        viscosity-coupled λ_eff), but the within-T attention is built as the
        Hermitian-symmetric LION matrix:

            A_fwd[t, s, h, b] = exp(cs[t, h, b] - cs[s, h, b])              for t ≥ s
            A_bwd[t, s, h, b] = exp(conj(cs_b[s, h, b] - cs_b[t, h, b]))    for t < s

        Output ``y[t, h, p] = Re(Σ_s Σ_b A[t, s, h, b] · B_c[s, h, b]
                                   · x_disc[s, h, p] · C_c[t, h, b])``.

        Memory bounded by chunking over the pair-block index Bk.
        """
        Bsz, L, H, P = x_heads.shape
        Bk = self.n_blocks
        device = x_heads.device

        lambda_base = self.lambda_base.view(1, 1, H, Bk).float()
        if self.rse_viscosity:
            lambda_eff = lambda_base + (
                self.viscosity_eta.view(1, 1, H, Bk).float() * theta.float() ** 2
            )
        else:
            lambda_eff = lambda_base.expand(Bsz, L, H, Bk)
        A_cont_c = torch.complex(-lambda_eff.float(), theta.float())            # (B, L, H, Bk)
        log_z = A_cont_c * dt.float().unsqueeze(-1)                             # (B, L, H, Bk)
        x_disc = (x_heads.float() * dt.float().unsqueeze(-1)).to(torch.complex64)  # (B, L, H, P)

        cs = log_z.cumsum(dim=1)                                                # (B, L, H, Bk)
        cs_b = cs - log_z

        mid = L // 2
        shift_f_re = cs[:, mid:mid + 1].real
        shift_b_re = cs_b[:, mid:mid + 1].real
        cs_s = torch.complex(cs.real - shift_f_re, cs.imag)
        cs_bs = torch.complex(cs_b.real - shift_b_re, cs_b.imag)

        fwd_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
        bwd_mask = ~fwd_mask
        fwd_mask_b = fwd_mask.view(1, L, L, 1, 1)
        bwd_mask_b = bwd_mask.view(1, L, L, 1, 1)

        y = x_heads.new_zeros(Bsz, L, H, P)

        pair_chunk = 8  # match RWKV-6 LION × RSE (caps the (L, L, p) tensor size)
        for b0 in range(0, Bk, pair_chunk):
            b1 = min(b0 + pair_chunk, Bk)
            cs_s_c = cs_s[..., b0:b1]                                            # (B, L, H, p)
            cs_bs_c = cs_bs[..., b0:b1]
            B_cc = B_c[..., b0:b1]                                               # (B, L, H, p)
            C_cc = C_c[..., b0:b1]

            diff_f = cs_s_c.unsqueeze(2) - cs_s_c.unsqueeze(1)                   # (B, L, L, H, p)
            diff_b = cs_bs_c.unsqueeze(1) - cs_bs_c.unsqueeze(2)                 # diff_b[t, s] = cs_bs[s] − cs_bs[t]
            diff_b_conj = torch.complex(diff_b.real, -diff_b.imag)

            real_f = diff_f.real.clamp(-60.0, 60.0)
            real_b = diff_b_conj.real.clamp(-60.0, 60.0)
            A_fwd = torch.exp(torch.complex(real_f, diff_f.imag))
            A_bwd = torch.exp(torch.complex(real_b, diff_b_conj.imag))
            A = torch.where(fwd_mask_b, A_fwd, torch.zeros_like(A_fwd)) \
                + torch.where(bwd_mask_b, A_bwd, torch.zeros_like(A_bwd))        # (B, L, L, H, p)

            scaled_B = A * B_cc.unsqueeze(2)                                     # (B, L, L, H, p)
            S = torch.einsum("btshk,bshp->bthpk", scaled_B, x_disc)              # (B, L, H, P, p)
            y_chunk = torch.einsum("bthk,bthpk->bthp", C_cc, S).real             # (B, L, H, P)
            y = y + y_chunk

        return y

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,                  # (B, L, d_model)
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        Bsz, L, D = x.shape

        # Project and split.
        zxbcdt = self.in_proj(x)
        z, xBC, dt_raw = torch.split(
            zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1,
        )
        dt = F.softplus(dt_raw + self.dt_bias)                    # (B, L, H)

        # Depthwise conv + optional carry.
        xBC_bdt = xBC.transpose(1, 2)
        prefix_len = 0
        if state is not None and state.get("conv") is not None:
            conv_prefix = state["conv"].to(xBC_bdt.dtype)
            xBC_bdt = torch.cat([conv_prefix, xBC_bdt], dim=-1)
            prefix_len = conv_prefix.size(-1)
        conv_out_all = self.conv1d(xBC_bdt)
        if prefix_len > 0:
            conv_out = conv_out_all[..., prefix_len:prefix_len + L]
        else:
            conv_out = conv_out_all[..., :L]
        xBC_conv = F.silu(conv_out).transpose(1, 2)               # (B, L, conv_dim)

        tail_len = self.d_conv - 1
        if L >= tail_len:
            new_conv_state = xBC[:, -tail_len:, :].transpose(1, 2).contiguous()
        else:
            pad = torch.zeros(
                Bsz, tail_len - L, xBC.size(-1),
                device=xBC.device, dtype=xBC.dtype,
            )
            new_conv_state = torch.cat([pad, xBC], dim=1).transpose(1, 2).contiguous()

        # Split into x, B, C.
        x_ssm, B_ssm, C_ssm = torch.split(
            xBC_conv,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        x_heads = x_ssm.view(Bsz, L, self.nheads, self.headdim)
        B_g = B_ssm.view(Bsz, L, self.ngroups, self.d_state)
        C_g = C_ssm.view(Bsz, L, self.ngroups, self.d_state)
        if self._heads_per_group == 1:
            B_h = B_g
            C_h = C_g
        else:
            B_h = B_g.repeat_interleave(self._heads_per_group, dim=2)
            C_h = C_g.repeat_interleave(self._heads_per_group, dim=2)
        # B_h, C_h: (B, L, H, N=d_state).  Pair into Bk complex along N.
        B_h_pairs = B_h.float().view(Bsz, L, self.nheads, self.n_blocks, 2)
        C_h_pairs = C_h.float().view(Bsz, L, self.nheads, self.n_blocks, 2)
        B_c = torch.complex(B_h_pairs[..., 0], B_h_pairs[..., 1])  # (B, L, H, Bk)
        C_c = torch.complex(C_h_pairs[..., 0], C_h_pairs[..., 1])

        # theta_t via LoRA.  Use pre-conv x (the block input) as the driver,
        # matching Mamba-2 convention of using x for dt_raw.
        theta = self._compute_theta(x)                             # (B, L, H, Bk)

        ssm_state = state.get("ssm") if state is not None else None
        if self.mode == "recurrent":
            y_heads, new_ssm = self._rse_scan(
                x_heads=x_heads, dt=dt, B_c=B_c, C_c=C_c,
                theta=theta, ssm_state=ssm_state,
            )
        else:  # mode == "lion"
            # Bidirectional T×T complex attention.  No carry-state — LION
            # mode is bidirectional and sees the whole sequence.
            from torch.utils.checkpoint import checkpoint
            if self.training:
                y_heads = checkpoint(
                    self._rse_scan_lion,
                    x_heads, dt, B_c, C_c, theta,
                    use_reentrant=False,
                )
            else:
                y_heads = self._rse_scan_lion(x_heads, dt, B_c, C_c, theta)
            new_ssm = None

        # D skip (per-head scalar, real).
        y_heads = y_heads + self.D.view(1, 1, self.nheads, 1) * x_heads

        # Flatten heads and finalise.
        y = y_heads.reshape(Bsz, L, self.d_inner)
        y = self.norm(y) * self.act(z)
        out = self.out_proj(y)

        new_state = {"conv": new_conv_state, "ssm": new_ssm}
        return out, new_state


class Mamba2RSEEncoderLayer(nn.Module):
    """Single layer: LN → Mamba2RSE → drop → LN → FFN → drop."""

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        headdim: int = 64,
        expand: int = 2,
        ngroups: int = 1,
        chunk_size: int = 64,
        dropout: float = 0.1,
        layer_id: int = 0,
        dtype: torch.dtype = torch.float32,
        rse_viscosity: bool = True,
        mode: str = "recurrent",
        rse_theta_init_scale: float = math.pi / 16,
        rse_theta_clip: float = math.pi / 2,
        rse_theta_lora_dim: int = 48,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, dtype=dtype)
        self.ln2 = nn.LayerNorm(d_model, dtype=dtype)
        self.ln0 = nn.LayerNorm(d_model, dtype=dtype) if layer_id == 0 else nn.Identity()

        self.mamba = Mamba2RSEBlock(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            headdim=headdim, expand=expand, ngroups=ngroups,
            chunk_size=chunk_size, dtype=dtype,
            rse_viscosity=rse_viscosity,
            rse_theta_init_scale=rse_theta_init_scale,
            rse_theta_clip=rse_theta_clip,
            rse_theta_lora_dim=rse_theta_lora_dim,
            mode=mode,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model, dtype=dtype),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        x = self.ln0(x)
        mamba_out, new_state = self.mamba(self.ln1(x), state=state)
        x = x + self.drop(mamba_out)
        x = x + self.ffn(self.ln2(x))
        return x, new_state


class Mamba2RSEEncoder(nn.Module):
    """Mamba-2 + RSE encoder.  Mirrors ``Mamba2Encoder`` signature."""

    supports_carry_state = True

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        ffn_dim: int = 896,
        d_state: int = 64,
        d_conv: int = 4,
        headdim: int = 64,
        expand: int = 2,
        ngroups: int = 1,
        chunk_size: int = 64,
        dtype: torch.dtype = torch.float32,
        rse_viscosity: bool = True,
        mode: str = "recurrent",
        rse_per_layer_overrides: Optional[list] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.supports_carry_state = (mode == "recurrent")

        self.pos_enc = SinusoidalPE(d_model, max_len=8000, dropout=dropout)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            override = (
                rse_per_layer_overrides[i] if rse_per_layer_overrides is not None
                else {}
            )
            self.layers.append(Mamba2RSEEncoderLayer(
                d_model=d_model, ffn_dim=ffn_dim, d_state=d_state,
                d_conv=d_conv, headdim=headdim, expand=expand, ngroups=ngroups,
                chunk_size=chunk_size, dropout=dropout, layer_id=i, dtype=dtype,
                rse_viscosity=rse_viscosity,
                mode=mode,
                rse_theta_init_scale=override.get("theta_init_scale", math.pi / 16),
                rse_theta_clip=override.get("theta_clip", math.pi / 2),
                rse_theta_lora_dim=override.get("theta_lora_dim", 48),
            ))

    def init_state(self, batch_size: int, device: torch.device) -> dict:
        return {
            "layers": [
                layer.mamba.init_state(batch_size, device) for layer in self.layers
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

        x = self.pos_enc(x, offset=offset)
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()

        new_layer_states: Optional[list] = [] if state is not None else None
        for i, layer in enumerate(self.layers):
            ls = layer_states[i] if layer_states is not None else None
            x, ns = layer(x, state=ls)
            x = x * mask_f
            if new_layer_states is not None:
                new_layer_states.append(ns)

        new_state: Optional[dict] = None
        if new_layer_states is not None:
            new_state = {"layers": new_layer_states, "offset": offset + T}
        return x, new_state
