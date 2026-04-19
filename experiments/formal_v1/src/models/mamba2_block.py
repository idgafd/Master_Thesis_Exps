"""Mamba-2 SSM block with pluggable scan kernels.

Mirrors the role of ``rwkv6_time_mix.py`` for RWKV-6: owns the learnable
parameters (in_proj, conv1d, dt_bias, A_log, D, norm, out_proj) and dispatches
on ``self.mode`` to a scan kernel from ``mamba2_kernels.py``.

Supported modes:

    "recurrent"   — Mamba-2 SSD causal scan (default).  Carry-state capable.
    "lion"        — LION full-attention bidirectional (parameter-free bidir).
                    No carry-state (sees the whole sequence at once).
    "lion_chunk"  — LION chunkwise bidirectional.  No carry-state.

Projections follow Mamba-2 (``mamba_ssm/modules/mamba2_simple.py``): a single
``in_proj`` outputs ``[z, x, B, C, dt]`` concatenated; the conv1d is applied to
the ``xBC`` slab; then x / B / C are split and fed to the SSM kernel; the
output is RMS-normed, gated by ``silu(z)``, and projected back to ``d_model``.

Reference: Dao & Gu, "Transformers are SSMs" (2024); Afzal et al., LION (2025).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mamba2_kernels import (
    ssd_scan_causal,
    ssd_scan_lion,
    ssd_scan_lion_chunk,
)


_MODES_WITH_CARRY = {"recurrent"}


class Mamba2Block(nn.Module):
    """Mamba-2 block with mode dispatch.

    Args:
        d_model:    model dimension (input / output).
        d_state:    SSM state dimension (N).
        d_conv:     depthwise conv kernel width.
        headdim:    per-head value dimension (P).  ``d_inner`` must divide.
        expand:     expansion factor; d_inner = expand * d_model.
        ngroups:    number of B/C groups.  Heads per group = nheads / ngroups.
        chunk_size: chunk length used by ``recurrent`` and ``lion_chunk`` modes.
        mode:       one of {"recurrent", "lion", "lion_chunk"}.
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
        mode: str = "recurrent",
        A_init_range: Tuple[float, float] = (1.0, 16.0),
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if mode not in {"recurrent", "lion", "lion_chunk"}:
            raise ValueError(f"unknown mode: {mode!r}")
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

        # ── projections ──────────────────────────────────────────────────
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias, dtype=dtype)

        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.conv_dim = conv_dim
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            bias=conv_bias,
            dtype=dtype,
        )

        # ── dt bias (softplus range [dt_min, dt_max]) ────────────────────
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt.to(dtype))
        self.dt_bias._no_weight_decay = True

        # ── A (continuous, negative) ────────────────────────────────────
        if not (A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]):
            raise ValueError("A_init_range must be positive and non-decreasing")
        A = torch.empty(self.nheads).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A).to(dtype))
        self.A_log._no_weight_decay = True

        # ── D skip ──────────────────────────────────────────────────────
        self.D = nn.Parameter(torch.ones(self.nheads, dtype=dtype))
        self.D._no_weight_decay = True

        # ── output stack ────────────────────────────────────────────────
        # RMSNorm requires torch >= 2.4.  Caller is on torch 2.7.
        self.norm = nn.RMSNorm(self.d_inner, dtype=dtype)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, dtype=dtype)

    # ------------------------------------------------------------------
    # Carry-state helpers (recurrent mode only)
    # ------------------------------------------------------------------

    @property
    def supports_carry_state(self) -> bool:
        return self.mode in _MODES_WITH_CARRY

    def init_state(self, batch_size: int, device: torch.device) -> dict:
        """Zero-initialised carry state for this block.

        Returns a dict with:
            conv : (B, conv_dim, d_conv - 1)
            ssm  : (B, nheads, headdim, d_state)
        """
        return {
            "conv": torch.zeros(
                batch_size, self.conv_dim, self.d_conv - 1,
                device=device, dtype=torch.float32,
            ),
            "ssm": torch.zeros(
                batch_size, self.nheads, self.headdim, self.d_state,
                device=device, dtype=torch.float32,
            ),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Forward over a full chunk of tokens.

        x: (B, L, d_model).  For ``mode != 'recurrent'`` ``state`` must be None.
        """
        if state is not None and not self.supports_carry_state:
            raise RuntimeError(
                f"mode={self.mode!r} does not support carry-state"
            )

        B, L, D = x.shape

        # Project and split.
        zxbcdt = self.in_proj(x)
        z, xBC, dt_raw = torch.split(
            zxbcdt,
            [self.d_inner, self.conv_dim, self.nheads],
            dim=-1,
        )
        dt = F.softplus(dt_raw + self.dt_bias)                     # (B, L, H)

        # Depthwise conv with optional carry-state history.
        xBC_bdt = xBC.transpose(1, 2)                              # (B, conv_dim, L)
        prefix_len = 0
        if state is not None and state.get("conv") is not None:
            conv_prefix = state["conv"].to(xBC_bdt.dtype)          # (B, conv_dim, d_conv-1)
            xBC_bdt = torch.cat([conv_prefix, xBC_bdt], dim=-1)
            prefix_len = conv_prefix.size(-1)

        conv_out_all = self.conv1d(xBC_bdt)
        if prefix_len > 0:
            conv_out = conv_out_all[..., prefix_len : prefix_len + L]
        else:
            conv_out = conv_out_all[..., :L]
        xBC_conv = F.silu(conv_out).transpose(1, 2)                # (B, L, conv_dim)

        # New conv state — last (d_conv - 1) frames of the pre-conv xBC.
        tail_len = self.d_conv - 1
        if L >= tail_len:
            new_conv_state = xBC[:, -tail_len:, :].transpose(1, 2).contiguous()
        else:
            pad = torch.zeros(
                B, tail_len - L, xBC.size(-1),
                device=xBC.device, dtype=xBC.dtype,
            )
            new_conv_state = torch.cat([pad, xBC], dim=1).transpose(1, 2).contiguous()

        # Split xBC_conv into x_ssm, B_ssm, C_ssm.
        x_ssm, B_ssm, C_ssm = torch.split(
            xBC_conv,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        x_heads = x_ssm.view(B, L, self.nheads, self.headdim)
        B_g = B_ssm.view(B, L, self.ngroups, self.d_state)
        C_g = C_ssm.view(B, L, self.ngroups, self.d_state)
        if self._heads_per_group == 1:
            B_h = B_g
            C_h = C_g
        else:
            B_h = B_g.repeat_interleave(self._heads_per_group, dim=2)
            C_h = C_g.repeat_interleave(self._heads_per_group, dim=2)
        # B_h, C_h: (B, L, H, N)

        A_cont = -torch.exp(self.A_log.float())                    # (H,)

        ssm_state = state.get("ssm") if state is not None else None

        # Dispatch to kernel.
        if self.mode == "recurrent":
            y_heads, new_ssm = ssd_scan_causal(
                x_heads, dt, A_cont, B_h, C_h,
                chunk_size=self.chunk_size, state=ssm_state,
            )
        elif self.mode == "lion":
            y_heads = ssd_scan_lion(x_heads, dt, A_cont, B_h, C_h)
            new_ssm = None
        else:  # "lion_chunk"
            y_heads = ssd_scan_lion_chunk(
                x_heads, dt, A_cont, B_h, C_h, chunk_size=self.chunk_size,
            )
            new_ssm = None

        # D skip connection (per-head scalar).
        y_heads = y_heads + self.D.view(1, 1, self.nheads, 1) * x_heads

        # Flatten heads and finalise.
        y = y_heads.reshape(B, L, self.d_inner)
        y = self.norm(y) * self.act(z)
        out = self.out_proj(y)

        new_state = {"conv": new_conv_state, "ssm": new_ssm}
        return out, new_state

    # ------------------------------------------------------------------
    # Single-step inference
    # ------------------------------------------------------------------

    def step(
        self,
        x: torch.Tensor,           # (B, 1, D)
        conv_state: torch.Tensor,  # (B, conv_dim, d_conv - 1)
        ssm_state: torch.Tensor,   # (B, nheads, headdim, d_state)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-token inference.  Only valid in ``recurrent`` mode."""
        if not self.supports_carry_state:
            raise RuntimeError(
                f"step() not supported for mode={self.mode!r}"
            )

        B = x.size(0)
        zxbcdt = self.in_proj(x.squeeze(1))
        z, xBC, dt_raw = torch.split(
            zxbcdt,
            [self.d_inner, self.conv_dim, self.nheads],
            dim=-1,
        )
        dt = F.softplus(dt_raw + self.dt_bias)                     # (B, H)

        # Conv step: prepend stored conv_state, then take weighted sum.
        # full has shape (B, conv_dim, d_conv); the conv weight is (conv_dim, 1, d_conv).
        full = torch.cat([conv_state.to(xBC.dtype), xBC.unsqueeze(-1)], dim=-1)
        w = self.conv1d.weight.squeeze(1)                          # (conv_dim, d_conv)
        xBC_conv = (full * w).sum(-1)
        if self.conv1d.bias is not None:
            xBC_conv = xBC_conv + self.conv1d.bias
        xBC_conv = F.silu(xBC_conv)                                # (B, conv_dim)
        new_conv_state = full[:, :, 1:]

        x_ssm, B_ssm, C_ssm = torch.split(
            xBC_conv,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        x_heads = x_ssm.view(B, self.nheads, self.headdim).float()
        B_g = B_ssm.view(B, self.ngroups, self.d_state).float()
        C_g = C_ssm.view(B, self.ngroups, self.d_state).float()
        if self._heads_per_group == 1:
            B_h = B_g
            C_h = C_g
        else:
            B_h = B_g.repeat_interleave(self._heads_per_group, dim=1)
            C_h = C_g.repeat_interleave(self._heads_per_group, dim=1)
        # B_h, C_h: (B, H, N)

        A_cont = -torch.exp(self.A_log.float())                    # (H,)
        dA = torch.exp(dt * A_cont)                                # (B, H)
        dB = dt.unsqueeze(-1) * B_h                                # (B, H, N)

        # h_new = dA * h_old + dB ⊗ x
        ssm_state = (
            ssm_state * dA.unsqueeze(-1).unsqueeze(-1)
            + x_heads.unsqueeze(-1) * dB.unsqueeze(-2)
        )
        y_heads = (ssm_state * C_h.unsqueeze(-2)).sum(-1)          # (B, H, P)
        y_heads = y_heads + self.D.view(1, self.nheads, 1) * x_heads

        y = y_heads.reshape(B, self.d_inner).to(x.dtype)
        y = self.norm(y) * self.act(z)
        out = self.out_proj(y).unsqueeze(1)
        return out, new_conv_state, ssm_state
