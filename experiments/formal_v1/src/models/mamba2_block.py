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
    ssd_scan_causal_novelty,
    ssd_scan_lion,
    ssd_scan_lion_chunk,
)


_MODES_WITH_CARRY = {"recurrent"}


class SymmetricDWConv1d(nn.Module):
    """Stage 11.5b — single-dilation symmetric depthwise Conv1d.

    Plain ``nn.Conv1d`` with manual symmetric padding; no multi-dilation,
    no per-layer alpha scalar.  For even kernels the padding split is
    ``(k//2 - 1 ... actually floor(k-1 / 2), ceil(k-1 / 2))`` — left
    receives the fewer taps.

    Isolates the *symmetric-padding* effect from the multi-dilation axis
    (which is inert due to a multiplicative-zero init trap on
    ``MultiDilationDWConv1d``; see MULTIDIL_INIT_FIX_HANDOFF).  Output
    length equals input length.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=0, groups=channels, bias=bias, dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, N) -> (B, C, N)."""
        pad_total = self.kernel_size - 1
        left = pad_total // 2
        right = pad_total - left
        x_p = F.pad(x, (left, right))
        return self.conv(x_p)


class MultiDilationDWConv1d(nn.Module):
    """Stage 11.1a — channel-first multi-dilation depthwise Conv1d drop-in
    replacement for Mamba-2's internal xBC conv.

    Per §6 Stage 11.1a spec: parallel DWConv1d branches at dilations
    {1, 2, 4, 8}, symmetric padding per branch, per-layer learnable alpha_d
    with alpha_1 = 1 and alpha_{2,4,8} = 0 at init.  At init the module
    reduces to a SINGLE-DILATION k=4 SYMMETRIC DWConv on xBC — not
    bit-exact vs vanilla Mamba-2 (which uses causal padding) because the
    "sym" variant deliberately uses symmetric padding, matching the
    RWKV-6 Stage 10.3-sym zero-regression contract.

    Interface matches what Mamba2Block expects from ``self.conv1d``: input
    ``(B, C, N)`` returns ``(B, C, N)`` (same length, not N+3 like vanilla
    with padding=3 — the length difference is absorbed by the
    ``conv_out_all[..., prefix_len : prefix_len + L]`` slice downstream,
    which works for either output length as long as L + prefix_len covers
    the slice range).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        dilations: Tuple[int, ...] = (1, 2, 4, 8),
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        fixed_init: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = tuple(int(d) for d in dilations)

        self.branches = nn.ModuleList([
            nn.Conv1d(
                channels, channels,
                kernel_size=kernel_size,
                padding=0,
                groups=channels,
                bias=bias,
                dtype=dtype,
            )
            for _ in self.dilations
        ])

        # ───────── FIXED INIT (gradient-trap safe, Option B) ─────────
        # Pre-fix bug mirrored the MultiDilationDWConvShift trap: α_{d>1}=0
        # AND branch_{d>1}.weight=0 gave a zero-zero product for both
        # ∂L/∂α_d and ∂L/∂branch_d.weight → no SGD signal for any d>1
        # branch.  Option-B fix: α_{d>1}=0.01 + branch_{d>1}.weight ~
        # N(0, 0.01²).  Matches ``src/models/mechanisms/conv_shift.py``
        # as of commit ``3af846d``.  The ``fixed_init=False`` toggle is
        # kept only for bit-exact reproducibility of the original
        # broken-init runs (11.1a ``mamba2_convshift_multidil_symmetric``,
        # pre-v2).
        NON_MAIN_ALPHA = 0.01 if fixed_init else 0.0
        NON_MAIN_WEIGHT_STD = 0.01 if fixed_init else 0.0

        d1 = self.dilations.index(1) if 1 in self.dilations else 0
        alphas = torch.zeros(len(self.dilations), dtype=dtype)
        for i, d in enumerate(self.dilations):
            alphas[i] = 1.0 if i == d1 else NON_MAIN_ALPHA
        self.alpha = nn.Parameter(alphas)

        # Branches: main branch keeps PyTorch default Kaiming init.  For d>1,
        # weights init small random (so ∂L/∂α_d flows when α_d is nonzero
        # AND ∂L/∂branch_d.weight flows when α_d is nonzero).  Biases zeroed.
        with torch.no_grad():
            for idx, d in enumerate(self.dilations):
                if idx == d1:
                    continue
                if fixed_init:
                    self.branches[idx].weight.normal_(mean=0.0, std=NON_MAIN_WEIGHT_STD)
                else:
                    self.branches[idx].weight.zero_()
                if self.branches[idx].bias is not None:
                    self.branches[idx].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, N) -> (B, C, N).  Symmetric padding per branch.

        For even kernel sizes the split is biased by one tap (left
        receives floor(pad/2), right receives ceil(pad/2)) — matches
        ``MultiDilationDWConvShift`` in ``mechanisms/conv_shift.py``.
        """
        out = None
        for idx, (d, branch) in enumerate(zip(self.dilations, self.branches)):
            pad_total = (self.kernel_size - 1) * d
            left = pad_total // 2
            right = pad_total - left
            x_p = F.pad(x, (left, right))
            y = F.conv1d(
                x_p, branch.weight,
                bias=branch.bias,
                stride=1, padding=0,
                dilation=d, groups=self.channels,
            )
            contrib = self.alpha[idx] * y
            out = contrib if out is None else out + contrib
        return out


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
        use_multidil_sym: bool = False,
        multidil_dilations: Tuple[int, ...] = (1, 2, 4, 8),
        use_convshift_sym: bool = False,
        use_lucid: bool = False,
        lucid_key_source: str = "B",
        lucid_decay_aware: bool = False,
        use_novelty_gate: bool = False,
        novelty_gamma_fixed: Optional[float] = None,
        use_householder: bool = False,
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
        self.use_multidil_sym = use_multidil_sym
        self.use_convshift_sym = use_convshift_sym
        if use_multidil_sym and use_convshift_sym:
            raise ValueError(
                "use_multidil_sym and use_convshift_sym are mutually exclusive"
            )
        if use_multidil_sym:
            self.conv1d = MultiDilationDWConv1d(
                channels=conv_dim,
                kernel_size=d_conv,
                dilations=multidil_dilations,
                bias=conv_bias,
                dtype=dtype,
            )
        elif use_convshift_sym:
            # Stage 11.5b — single-dilation symmetric DWConv; isolates
            # padding-direction effect from multi-dilation.
            self.conv1d = SymmetricDWConv1d(
                channels=conv_dim,
                kernel_size=d_conv,
                bias=conv_bias,
                dtype=dtype,
            )
        else:
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

        # ── LUCID preconditioner on the chunked SSD dual form ───────────
        # Per-head temperature initialised so that softplus(raw) = 1.0
        # exactly: raw = log(e - 1) ≈ 0.5413.  Matches the paper's unit
        # scaling and the RWKV-6 LUCID implementation (rwkv6_time_mix.py).
        # At tau = 1.0 with chunk_size=64 and d_state=64, off-diagonal P
        # entries are exp(-sqrt(d_state)) ≈ exp(-8) ≈ 3e-4 → P ≈ I within
        # fp32 noise, so X_c_precond ≈ X_c at init (~0.03% off, not 20%).
        # Earlier zero-init bug gave softplus(0) = 0.693, which attenuated
        # the DC component of X_c by ~20% per chunk and caused a first-epoch
        # loss deficit — diagnosed and fixed 2026-04-23.
        self.use_lucid = use_lucid
        self.lucid_key_source = lucid_key_source
        # D-LUCID (decay-aware LUCID, v6 additive-decay-penalty).
        # Adds a per-pair decay-distance penalty to LUCID's exponent:
        #   scaled_ij = τ_h · (G_ij/√N − √N) − γ_h · |cs[i] − cs[j]|
        # γ_h ≥ 0 via softplus(γ_raw − 5); γ_raw init 0 ⇒ γ ≈ 0.007 at
        # init (near-vanilla LUCID) while still on a smooth gradient
        # path Adam can climb.  γ = 0 exactly recovers LUCID; Δcs = 0
        # (zero-decay within-chunk) also recovers LUCID for any γ.
        #
        # The shift of 5 is deliberate: softplus directly (γ_raw init 0)
        # would give γ ≈ 0.69, already in the engagement regime and too
        # far from vanilla LUCID at init.  Shift 5 gives γ ≈ 0.007, close
        # to zero but with a non-vanishing gradient path — σ(−5) ≈ 0.007,
        # the same order the scientist used successfully on lucid_temperature.
        self.lucid_decay_aware = lucid_decay_aware
        self._dlucid_gamma_shift = 5.0
        if lucid_decay_aware and not use_lucid:
            raise ValueError(
                "lucid_decay_aware=True requires use_lucid=True"
            )
        if use_lucid:
            if mode != "recurrent":
                raise ValueError(
                    f"LUCID is only defined on the recurrent SSD dual form; "
                    f"got mode={mode!r}. LION modes don't apply."
                )
            if lucid_key_source not in ("B", "C"):
                raise ValueError(
                    f"lucid_key_source must be 'B' or 'C', got "
                    f"{lucid_key_source!r}"
                )
            self.lucid_temperature = nn.Parameter(
                torch.full(
                    (self.nheads,),
                    math.log(math.e - 1),  # softplus(log(e-1)) = 1.0 exactly
                    dtype=dtype,
                )
            )
            if lucid_decay_aware:
                # Per-head γ_raw; γ = softplus(γ_raw − 5).  Init 0 gives
                # γ ≈ 0.007 at step 0 — near-vanilla LUCID, non-saturated.
                self.dlucid_gamma_raw = nn.Parameter(
                    torch.zeros(self.nheads, dtype=dtype)
                )

        # ── Householder inter-chunk state rotation ──────────────────────
        # Generalised partial Householder applied at chunk boundaries in
        # the SSD scan's inter-chunk propagation:
        #     H_h = I − 2(1 − α_h) · u_h · u_hᵀ       α_h ∈ [0, 1]
        # Per-head parameters (H·N for u_raw, H for α_raw).  α = 1 gives
        # bit-exact vanilla Mamba-2 (H = I).  α ∈ [0, 1] guarantees
        # operator norm ≤ 1 → stability preserved.
        #
        # Parameterisation chosen so init α ≈ 1 (so the backbone starts
        # at vanilla Mamba-2 and SGD engages Householder as useful):
        #   α_raw init = 5 ⇒ α = sigmoid(5) ≈ 0.9933 ⇒ (1 − α) ≈ 0.007.
        # u_raw is initialised with small random values and is unit-
        # normalised at use, giving a well-defined reflection direction
        # that can move freely during training.
        self.use_householder = use_householder
        if use_householder and mode != "recurrent":
            raise ValueError(
                f"Householder is only defined on the recurrent SSD scan; "
                f"got mode={mode!r}"
            )
        if use_householder:
            self.householder_u_raw = nn.Parameter(
                torch.randn(self.nheads, d_state, dtype=dtype) * 0.01
            )
            self.householder_alpha_raw = nn.Parameter(
                torch.full((self.nheads,), 5.0, dtype=dtype)
            )

        # ── Write-novelty gate (per-chunk Σ variant) ────────────────────
        # ω_t = 1 / (1 + γ_h / (B_t^T Σ_c^{-1} B_t + ε)).
        #
        # Parameterisation: γ_h = softplus(γ_raw − NOVELTY_SHIFT).
        # Init γ_raw = 0 → γ = softplus(-5) ≈ 6.7e-3 → ω ≈ 0.993 at
        # init (≈ 0.7% off vanilla — within the "near-identity" envelope
        # but deliberately NOT bit-exact, to keep the gradient path
        # trainable by Adam).  The naive γ_raw=-30 init would give
        # bit-exact vanilla at step 0 but σ(-30) ≈ 9e-14 collapses the
        # softplus chain-rule scale to ~1e-13 — γ_raw sees an update of
        # ~5e-5/step through Adam's ε floor and never moves off init.
        # Shifted parameterisation: σ(0) = 0.5 on the γ_raw side gives
        # responsiveness comparable to lucid_temperature (τ_raw init
        # 0.54, σ(0.54)≈0.63).  Only (H,) parameters added.
        # Two modes:
        #  - Trainable (novelty_gamma_fixed=None, default): γ =
        #    softplus(γ_raw - shift) with γ_raw as parameter, init 0 →
        #    γ ≈ 6.7e-3; SGD lifts γ during training.
        #  - Fixed ablation (novelty_gamma_fixed=<value>): γ is a buffer,
        #    no gradient, applied directly (no softplus).  Forces gate
        #    into engagement regardless of SGD preference — isolates
        #    "mechanism productive at fixed γ?" from "mechanism
        #    reachable via training?".
        self.use_novelty_gate = use_novelty_gate
        self._novelty_shift = 5.0
        self._novelty_trainable = use_novelty_gate and novelty_gamma_fixed is None
        if use_novelty_gate:
            if mode != "recurrent":
                raise ValueError(
                    f"novelty-gate is only defined on the recurrent SSD "
                    f"dual form; got mode={mode!r}."
                )
            if novelty_gamma_fixed is None:
                self.novelty_gamma_raw = nn.Parameter(
                    torch.zeros(self.nheads, dtype=dtype)
                )
            else:
                if novelty_gamma_fixed < 0:
                    raise ValueError(
                        f"novelty_gamma_fixed must be non-negative, got "
                        f"{novelty_gamma_fixed}"
                    )
                self.register_buffer(
                    "novelty_gamma_buf",
                    torch.full(
                        (self.nheads,), float(novelty_gamma_fixed),
                        dtype=dtype,
                    ),
                )

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
        st = {
            "conv": torch.zeros(
                batch_size, self.conv_dim, self.d_conv - 1,
                device=device, dtype=torch.float32,
            ),
            "ssm": torch.zeros(
                batch_size, self.nheads, self.headdim, self.d_state,
                device=device, dtype=torch.float32,
            ),
        }
        if self.use_novelty_gate:
            # Σ_0 = 0; the eps_reg · I regulariser at inversion time
            # handles the first-chunk case (matches "no keys seen yet").
            st["sigma"] = torch.zeros(
                batch_size, self.nheads, self.d_state, self.d_state,
                device=device, dtype=torch.float32,
            )
        return st

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

        # Dispatch to kernel.  LUCID temperature passed only when enabled;
        # ssd_scan_causal applies the chunk-local preconditioner on X_c
        # using B-correlation before the intra/inter einsums.
        lucid_temp = (
            F.softplus(self.lucid_temperature) if self.use_lucid else None
        )
        lucid_decay_penalty = (
            F.softplus(self.dlucid_gamma_raw - self._dlucid_gamma_shift)
            if self.lucid_decay_aware else None
        )
        if self.use_householder:
            # Normalise u_raw to a unit direction per head; sigmoid α_raw
            # into [0, 1].  Broadcasting handles the inner product in the
            # kernel; u must be a clean unit vector for the spec to hold.
            u_norm = F.normalize(self.householder_u_raw, dim=-1)
            alpha = torch.sigmoid(self.householder_alpha_raw)
            householder_u = u_norm
            householder_alpha = alpha
        else:
            householder_u = None
            householder_alpha = None
        new_sigma: Optional[torch.Tensor] = None
        if self.mode == "recurrent":
            if self.use_novelty_gate:
                sigma_in = (
                    state.get("sigma") if state is not None else None
                )
                if self._novelty_trainable:
                    novelty_gamma = F.softplus(
                        self.novelty_gamma_raw - self._novelty_shift
                    )
                else:
                    novelty_gamma = self.novelty_gamma_buf
                y_heads, new_ssm, new_sigma = ssd_scan_causal_novelty(
                    x_heads, dt, A_cont, B_h, C_h,
                    chunk_size=self.chunk_size, state=ssm_state,
                    sigma_state=sigma_in,
                    novelty_gamma=novelty_gamma,
                    lucid_temp=lucid_temp,
                    lucid_key_source=self.lucid_key_source,
                    lucid_decay_aware=self.lucid_decay_aware,
                    lucid_decay_penalty=lucid_decay_penalty,
                )
            else:
                y_heads, new_ssm = ssd_scan_causal(
                    x_heads, dt, A_cont, B_h, C_h,
                    chunk_size=self.chunk_size, state=ssm_state,
                    lucid_temp=lucid_temp,
                    lucid_key_source=self.lucid_key_source,
                    lucid_decay_aware=self.lucid_decay_aware,
                    lucid_decay_penalty=lucid_decay_penalty,
                    householder_u=householder_u,
                    householder_alpha=householder_alpha,
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
        if self.use_novelty_gate:
            new_state["sigma"] = new_sigma
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
        if self.use_multidil_sym:
            # step() reads self.conv1d.weight directly, which assumes an
            # nn.Conv1d.  Multidil uses a ModuleList of branches; single-
            # token streaming inference for multidil would need per-branch
            # state tracking.  Chunked eval via forward() handles state
            # correctly through prefix concatenation, so we only disallow
            # the single-token path here.
            raise NotImplementedError(
                "step() is not implemented for multidil_sym; use the "
                "chunked forward() path for state-carry evaluation."
            )
        if self.use_convshift_sym:
            raise NotImplementedError(
                "step() is not implemented for convshift_sym; use the "
                "chunked forward() path for state-carry evaluation."
            )
        if self.use_lucid or self.lucid_decay_aware:
            raise NotImplementedError(
                "step() is not implemented for LUCID; the preconditioner "
                "is chunk-local, and single-token streaming would need a "
                "per-token update of the chunk-local Gram matrix.  Use "
                "the chunked forward() path for state-carry evaluation."
            )
        if self.use_novelty_gate:
            raise NotImplementedError(
                "step() is not implemented for the novelty gate; the "
                "running Σ is maintained at chunk granularity.  Use the "
                "chunked forward() path for state-carry evaluation."
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
