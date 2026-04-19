"""Unified RWKV-6 TimeMix block — supports recurrent, LION, and bidir_serial modes.

Self-contained implementation (no external rwkv-block dependency).
All mechanism flags (conv_shift, headscale, delta_rule, lucid, temperature, rse)
are compositional and can be combined freely.

Reference: RWKV-6 (Peng et al. 2024), LION (Afzal et al. 2025), RSE (ProposalA, 2026)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lion_attention import (
    lion_parallel_attention,
    lion_attention_with_delta,
    lion_attention_with_lucid,
)


def _bidirectional_token_shift(x: torch.Tensor) -> torch.Tensor:
    """Bidirectional shift: (x[t-1] + x[t+1]) / 2, zero-padded."""
    left = F.pad(x[:, :-1, :], (0, 0, 1, 0))
    right = F.pad(x[:, 1:, :], (0, 0, 0, 1))
    return (left + right) * 0.5


def _causal_token_shift(x: torch.Tensor) -> torch.Tensor:
    """Causal shift: x[t-1], zero at t=0."""
    return F.pad(x[:, :-1, :], (0, 0, 1, 0))


class RWKV6TimeMix(nn.Module):
    """Unified RWKV-6 time-mixing block.

    Modes:
        "recurrent"    — causal chunked WKV, carry-state capable
        "lion"         — LION parallel full T*T attention, bidirectional
        "bidir_serial" — forward + backward recurrent, merged

    Mechanism flags (all composable):
        conv_shift:  learned DWConv1d replacing fixed token shift
        headscale:   per-head learnable decay bias
        delta_rule:  selective state erasure (causal-only on LION)
        lucid:       LUCID preconditioner on attention output
        temperature: per-head learnable attention temperature
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        num_hidden_layers: int,
        layer_id: int,
        mode: str = "lion",
        conv_shift: bool = False,
        headscale: bool = False,
        delta_rule: bool = False,
        lucid: bool = False,
        lucid_chunk_size: Optional[int] = None,
        lucid_self_reg: bool = False,
        temperature: bool = False,
        discretization: str = "zoh",
        discretization_init: str = "zoh",
        drop_u: bool = False,
        rse: bool = False,
        # θ_init scale: acoustic prior says useful angular frequencies for
        # syllable/formant-envelope dynamics are 1–10 Hz. At 100-fps frames
        # that maps to per-step θ in [0.06, 0.6] rad. U(-π/16, π/16) ≈ ±0.2
        # gives the right initial spread without saturating phase.
        rse_theta_init_scale: float = math.pi / 16,
        # θ_clip: bounds learned per-step rotation. π/4 ≈ 0.78 rad/step caps
        # at ~12 Hz at 100 fps — well above formant-envelope rates, leaves
        # head-room without aliasing into per-step phase wraps.
        rse_theta_clip: float = math.pi / 4,
        # LoRA dim for the θ projection (default 32 matches Stage 3 RSE).
        rse_theta_lora_dim: int = 32,
        rse_n_scales: int = 1,
        # ── Stage 5 P²-RSE: paired-pole RSE ──────────────────────────────
        # Two complex poles per block with shared λ, independent θ, and a
        # data-dependent real mixer β. Phase-complementary init: θ^(2) = -θ^(1).
        p2rse: bool = False,
        # Mixer type: "linear" (unconstrained real β — main) or "softmax" (control).
        p2rse_mixer: str = "linear",
        # ── Stage 5 Viscosity coupling (Rayleigh dissipation) ────────────
        # Soft self-regulating phase-coherence bound:
        #     λ_eff = λ_raw + η_{h,b} · θ²
        # A single learnable scalar per (head, block) pair, initialized to 0
        # so the forward pass is bit-identical to the RSE baseline until SGD
        # drives η away from 0.  Physical motivation: Stokes drag / Rayleigh
        # dissipation in a damped harmonic oscillator — high-frequency
        # components decay faster, which is the acoustic prior for formant
        # bandwidth scaling with centre frequency.
        rse_viscosity: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = 8
        self.mode = mode
        self.use_conv_shift = conv_shift
        self.use_headscale = headscale
        self.use_delta_rule = delta_rule
        self.use_lucid = lucid
        self.lucid_chunk_size = lucid_chunk_size
        self.use_lucid_self_reg = lucid_self_reg
        self.use_temperature = temperature
        self.discretization = discretization
        self.discretization_init = discretization_init
        self.drop_u = drop_u
        self.use_rse = rse
        self.rse_theta_clip = rse_theta_clip
        self.rse_n_scales = rse_n_scales
        self.use_p2rse = p2rse
        self.p2rse_mixer = p2rse_mixer
        self.use_viscosity = rse_viscosity
        self.layer_id = layer_id

        if self.use_rse:
            assert head_size % 2 == 0, "RSE requires even head_size (2x2 blocks)"
            assert mode == "recurrent", "RSE currently supports only mode='recurrent'"
            assert rse_n_scales >= 1

        if self.use_p2rse:
            assert self.use_rse, "P²-RSE requires rse=True"
            assert rse_n_scales == 1, "P²-RSE incompatible with multi-rate rse_n_scales > 1"
            assert self.p2rse_mixer in ("linear", "softmax")

        hidden_size_att = hidden_size

        # ── Token shift parameters ───────────────────────────────────────
        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(num_hidden_layers - 1, 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)
            ddd = torch.ones(1, 1, hidden_size, dtype=dtype)
            for i in range(hidden_size):
                ddd[0, 0, i] = i / hidden_size

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            )
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_DIM = 32
            self.time_maa_w1 = nn.Parameter(
                torch.zeros(hidden_size, D_MIX_DIM * 5, dtype=dtype)
            )
            self.time_maa_w2 = nn.Parameter(
                torch.zeros(5, D_MIX_DIM, hidden_size, dtype=dtype).uniform_(-0.01, 0.01)
            )

            # ── Decay parameters ─────────────────────────────────────────
            decay_speed = torch.ones(hidden_size_att, dtype=dtype)
            for n in range(hidden_size_att):
                decay_speed[n] = -6 + 5 * (n / (hidden_size_att - 1)) ** (
                    0.7 + 1.3 * ratio_0_to_1
                )
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, hidden_size_att))

            D_DECAY_DIM = 64
            self.time_decay_w1 = nn.Parameter(
                torch.zeros(hidden_size, D_DECAY_DIM, dtype=dtype)
            )
            self.time_decay_w2 = nn.Parameter(
                torch.zeros(D_DECAY_DIM, hidden_size_att, dtype=dtype).uniform_(-0.01, 0.01)
            )

            # Bonus term (u in RWKV notation, time_faaaa in code)
            tmp = torch.zeros(hidden_size_att, dtype=dtype)
            for n in range(hidden_size_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (hidden_size_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(n_head, head_size))

        # ── Linear projections ───────────────────────────────────────────
        self.receptance = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.gate = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.output = nn.Linear(hidden_size_att, hidden_size, bias=False, dtype=dtype)
        self.ln_x = nn.GroupNorm(
            n_head, hidden_size_att, dtype=dtype,
            eps=(1e-5) * (self.head_size_divisor ** 2),
        )

        # ── Mechanism: ConvShift ─────────────────────────────────────────
        if self.use_conv_shift:
            from src.models.mechanisms.conv_shift import DWConvShift
            self.conv_shift_module = DWConvShift(hidden_size)

        # ── Mechanism: Headscale ─────────────────────────────────────────
        if self.use_headscale:
            # (1, H, 1, K) for broadcasting against w_h of shape (B, H, T, K).
            self.head_decay_bias = nn.Parameter(torch.zeros(1, n_head, 1, head_size))

        # ── Mechanism: Delta Rule ────────────────────────────────────────
        if self.use_delta_rule:
            from src.models.mechanisms.delta_rule import DeltaRuleParams
            self.delta_params = DeltaRuleParams(hidden_size, n_head, head_size, dtype=dtype)

        # ── Mechanism: LUCID ─────────────────────────────────────────────
        if self.use_lucid:
            # Init so softplus(param) = 1.0, matching the paper's unit scaling
            # softplus(x) = 1.0 → x = ln(e - 1) ≈ 0.5413
            self.lucid_temperature = nn.Parameter(
                torch.full((n_head,), math.log(math.e - 1))
            )

        # ── Mechanism: Temperature ───────────────────────────────────────
        if self.use_temperature:
            self.attention_temperature = nn.Parameter(torch.ones(1, n_head, 1, 1))

        # ── Stage 2: Discretization scheme ───────────────────────────────
        # `gen2` adds learnable α₀, α₁ per head. Initialized from `discretization_init`:
        #   "zoh"  → α₀ ≈ 1, α₁ ≈ 0  (start as ZOH, learn to add lookback)
        #   "trap" → α₀ ≈ ½, α₁ ≈ ½  (start as trapezoidal)
        # Param values are pre-softplus. softplus(2) ≈ 2.13, softplus(-3) ≈ 0.05.
        if self.discretization == "gen2":
            if self.discretization_init == "trap":
                init_a0, init_a1 = 0.5413, 0.5413  # softplus(0.5413) ≈ 1.0, then /2 in code
            else:  # "zoh"
                init_a0, init_a1 = 2.0, -3.0
            self.disc_alpha0_raw = nn.Parameter(
                torch.full((n_head,), init_a0, dtype=dtype)
            )
            self.disc_alpha1_raw = nn.Parameter(
                torch.full((n_head,), init_a1, dtype=dtype)
            )

        # ── RSE: rotation parameters (θ) for 2x2 block-diagonal transition ──
        # Each head has B = head_size // 2 blocks. Per block: (lambda, theta).
        # lambda is reused from the existing per-channel decay (averaged over
        # adjacent pairs); theta is a new data-dependent angle via LoRA.
        if self.use_rse:
            n_blocks = head_size // 2
            theta_init = torch.empty(n_head, n_blocks, dtype=dtype).uniform_(
                -rse_theta_init_scale, rse_theta_init_scale
            )
            theta_init_reshaped = theta_init.reshape(1, 1, n_head * n_blocks)
            self.time_theta = nn.Parameter(theta_init_reshaped.clone())
            D_THETA_DIM = rse_theta_lora_dim
            self.time_theta_w1 = nn.Parameter(
                torch.zeros(hidden_size, D_THETA_DIM, dtype=dtype)
            )
            self.time_theta_w2 = nn.Parameter(
                torch.zeros(D_THETA_DIM, n_head * n_blocks, dtype=dtype).uniform_(-0.01, 0.01)
            )

            # ── Stage 5 Viscosity coupling ──────────────────────────────
            # η_{h,b} scalar per head × block, zero-init ⇒ identical to
            # baseline RSE until SGD grows it.  Added to λ (log-decay) as
            #     λ_eff = λ_raw + η · θ²
            # inside _forward_recurrent_rse, before cumulative log-z.
            if self.use_viscosity:
                self.viscosity_eta = nn.Parameter(
                    torch.zeros(n_head, n_blocks, dtype=dtype)
                )

            # ── Stage 5 P²-RSE: second θ pole (phase-complementary init) ──
            # Two complex poles per block with SHARED λ and INDEPENDENT θ.
            # Init: θ^(2)_base = -θ^(1)_base element-wise. LoRA^(2) zero-init,
            # matching Stage-3 RSE convention for the first pole.
            # Output fusion is real data-dependent β_m (linear or softmax).
            if self.use_p2rse:
                self.time_theta_2 = nn.Parameter(-theta_init_reshaped.clone())
                self.time_theta_w1_2 = nn.Parameter(
                    torch.zeros(hidden_size, D_THETA_DIM, dtype=dtype)
                )
                self.time_theta_w2_2 = nn.Parameter(
                    torch.zeros(D_THETA_DIM, n_head * n_blocks, dtype=dtype).uniform_(-0.01, 0.01)
                )
                # β mixer: W_β x_t ∈ R^{2H} (2 scalars per head).
                # Init N(0, 0.01) weights + zero bias ⇒ β_m ≈ 0 at init.
                # Under softmax this yields β ≈ (½, ½); under linear it yields
                # near-zero passthrough that grows freely under training.
                self.beta_mixer = nn.Linear(hidden_size, 2 * n_head, bias=True, dtype=dtype)
                nn.init.normal_(self.beta_mixer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.beta_mixer.bias)

            # ── Multi-Rate RSE: extra scales 1..M-1 with independent (λ, θ) ──
            # Each extra scale gets its own per-block (lambda, theta) LoRA pair.
            # Scale 0 reuses the existing time_decay path (averaged into Bk blocks)
            # and the time_theta path defined just above.
            if rse_n_scales > 1:
                D_LAM_DIM = 32
                D_THE_DIM = 32
                self.rse_extra_lambda_base = nn.ParameterList()
                self.rse_extra_lambda_w1 = nn.ParameterList()
                self.rse_extra_lambda_w2 = nn.ParameterList()
                self.rse_extra_theta_base = nn.ParameterList()
                self.rse_extra_theta_w1 = nn.ParameterList()
                self.rse_extra_theta_w2 = nn.ParameterList()
                for m in range(1, rse_n_scales):
                    # Per-block λ_base — slow-decay bias for higher scales
                    # so they tend to act as longer-context channels.
                    # softplus(λ_base) ≈ 0.5 / (m+1)  →  pre-softplus value
                    target = 0.5 / (m + 1)
                    init_lam = math.log(math.exp(target) - 1.0)  # inverse softplus
                    self.rse_extra_lambda_base.append(nn.Parameter(
                        torch.full((n_head, n_blocks), init_lam, dtype=dtype)
                    ))
                    self.rse_extra_lambda_w1.append(nn.Parameter(
                        torch.zeros(hidden_size, D_LAM_DIM, dtype=dtype)
                    ))
                    self.rse_extra_lambda_w2.append(nn.Parameter(
                        torch.zeros(D_LAM_DIM, n_head * n_blocks, dtype=dtype).uniform_(-0.01, 0.01)
                    ))
                    # Per-block θ_base — same warm-init range as scale 0
                    theta_init_m = torch.empty(n_head, n_blocks, dtype=dtype).uniform_(
                        -rse_theta_init_scale, rse_theta_init_scale
                    )
                    self.rse_extra_theta_base.append(nn.Parameter(theta_init_m))
                    self.rse_extra_theta_w1.append(nn.Parameter(
                        torch.zeros(hidden_size, D_THE_DIM, dtype=dtype)
                    ))
                    self.rse_extra_theta_w2.append(nn.Parameter(
                        torch.zeros(D_THE_DIM, n_head * n_blocks, dtype=dtype).uniform_(-0.01, 0.01)
                    ))

                # Query-conditional softmax mixer over scales: β_t = softmax(W_β x_t)
                # Initialized small so the model starts ~uniform across scales.
                self.rse_mixer = nn.Linear(
                    hidden_size, n_head * rse_n_scales, bias=True, dtype=dtype
                )
                nn.init.zeros_(self.rse_mixer.weight)
                nn.init.zeros_(self.rse_mixer.bias)

    def _token_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute dxprev based on mode and conv_shift setting."""
        if self.use_conv_shift:
            return self.conv_shift_module(x) - x
        if self.mode in ("lion", "bidir_serial"):
            return _bidirectional_token_shift(x) - x
        return _causal_token_shift(x) - x

    def _compute_rkv_gw(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared computation: token shift -> r, k, v, g, w."""
        B, T, D = x.size()

        dxprev = self._token_shift(x)

        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, D)

        mw, mk, mv, mr, mg = xxx.unbind(dim=0)
        xw = x + dxprev * (self.time_maa_w + mw)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xr = x + dxprev * (self.time_maa_r + mr)
        xg = x + dxprev * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = w.to(r.dtype)

        if self.use_rse:
            # Reuse the xw mixing path for theta (the rotation degree of freedom
            # is informationally analogous to decay — both modulate the transition).
            theta_lora = torch.tanh(xw @ self.time_theta_w1) @ self.time_theta_w2
            theta = self.rse_theta_clip * torch.tanh(self.time_theta + theta_lora)
            theta = theta.to(r.dtype)

            if self.use_p2rse:
                # Second pole's theta — independent LoRA, same xw / clip.
                theta_lora_2 = torch.tanh(xw @ self.time_theta_w1_2) @ self.time_theta_w2_2
                theta_2 = self.rse_theta_clip * torch.tanh(self.time_theta_2 + theta_lora_2)
                theta_2 = theta_2.to(r.dtype)
                return r, k, v, g, w, theta, theta_2

            return r, k, v, g, w, theta

        return r, k, v, g, w

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.size()
        H = self.n_head
        K = self.head_size

        if self.use_p2rse:
            r, k, v, g, w, theta, theta_2 = self._compute_rkv_gw(x)
        elif self.use_rse:
            r, k, v, g, w, theta = self._compute_rkv_gw(x)
            theta_2 = None
        else:
            r, k, v, g, w = self._compute_rkv_gw(x)
            theta = None
            theta_2 = None

        # Reshape to (B, H, T, K)
        r_h = r.view(B, T, H, K).transpose(1, 2)
        k_h = k.view(B, T, H, K).transpose(1, 2)
        v_h = v.view(B, T, H, K).transpose(1, 2)
        w_h = w.view(B, T, H, K).transpose(1, 2)

        # Log-decay: actual decay = exp(-exp(w_raw))
        w_h = -torch.exp(w_h)

        # Apply headscale
        if self.use_headscale:
            w_h = w_h + self.head_decay_bias.to(w_h.dtype)

        # Apply temperature
        if self.use_temperature:
            r_h = r_h * self.attention_temperature

        # Dispatch to mode-specific attention
        new_state = None
        if self.use_p2rse:
            # P²-RSE: two complex poles per block, unconstrained real mixer.
            n_blocks = K // 2
            theta_h_1 = theta.view(B, T, H, n_blocks).transpose(1, 2)
            theta_h_2 = theta_2.view(B, T, H, n_blocks).transpose(1, 2)
            y, new_state = self._forward_recurrent_p2rse(
                r_h, k_h, v_h, w_h, theta_h_1, theta_h_2, x, state
            )
        elif self.use_rse:
            # theta arrives as (B, T, H*Bk); reshape to (B, H, T, Bk)
            n_blocks = K // 2
            theta_h = theta.view(B, T, H, n_blocks).transpose(1, 2)
            if self.rse_n_scales == 1:
                y, new_state = self._forward_recurrent_rse(
                    r_h, k_h, v_h, w_h, theta_h, state
                )
            else:
                y, new_state = self._forward_recurrent_rse_multi(
                    r_h, k_h, v_h, w_h, theta_h, x, state
                )
        elif self.mode == "lion":
            y = self._forward_lion(r_h, k_h, v_h, w_h)
        elif self.mode == "recurrent":
            y, new_state = self._forward_recurrent(r_h, k_h, v_h, w_h, state)
        elif self.mode == "bidir_serial":
            y = self._forward_bidir_serial(r_h, k_h, v_h, w_h)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        y = y.to(r.dtype)

        # Reshape back and apply GroupNorm + gate + output
        y = y.transpose(1, 2).reshape(B * T, D)
        y = self.ln_x(y).view(B, T, D)
        y = self.output(y * g)

        return y, new_state

    def _forward_lion(
        self,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """LION parallel full T*T attention."""
        if self.use_delta_rule:
            kk, iclr = self.delta_params.compute_kk_iclr(
                k, k.shape[0], k.shape[2], self.n_head, self.head_size
            )
            return lion_attention_with_delta(r, k, v, w, kk, iclr)

        if self.use_lucid:
            temp = F.softplus(self.lucid_temperature)
            return lion_attention_with_lucid(
                r, k, v, w, temp, chunk_size=self.lucid_chunk_size
            )

        return lion_parallel_attention(r, k, v, w)

    def _forward_recurrent(
        self,
        r: torch.Tensor,  # (B, H, T, K)
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,  # already log-decay (negative)
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunked parallel WKV with carry state (SmerkyG algorithm).

        Dispatches on `self.discretization`:
          "zoh"      — standard RWKV-6 update (with bonus `u`)
          "trap"     — trapezoidal (½, ½) with current decay W
          "trap_var" — trapezoidal with geometric-mean decay W̃ = sqrt(W_t W_{t-1})
          "gen2"     — learnable α₀, α₁ per head (preserves bonus `u`)
          "ab3"      — Adams-Bashforth-3 (decay clamped for stability)
        """
        B, H, T, K = r.shape

        if state is None:
            wkv_state = torch.zeros(B, H, K, K, dtype=torch.float32, device=r.device)
        else:
            wkv_state = state.float()

        # Bonus term — ablated for trap/trap_var/ab3 (would conflate with α₀)
        if self.drop_u:
            u = torch.zeros(1, H, 1, K, dtype=r.dtype, device=r.device)
        else:
            u = self.time_faaaa.view(1, H, 1, K).to(r.dtype)

        # ── LUCID self-reg (legacy path) ─────────────────────────────────
        if self.use_lucid_self_reg:
            y, wkv_state = _chunked_wkv(r, k, v, w, u, wkv_state, self_reg=True)
            return y, wkv_state

        # ── LUCID preconditioner (legacy path) ───────────────────────────
        if self.use_lucid:
            temp = F.softplus(self.lucid_temperature)
            v_precond = _apply_lucid_recurrent(k, v, temp, self.lucid_chunk_size or 64)
            y, wkv_state = _chunked_wkv(r, k, v_precond, w, u, wkv_state)
            return y, wkv_state

        # ── Stage 2 discretization variants ──────────────────────────────
        if self.discretization == "zoh":
            y, wkv_state = _chunked_wkv(r, k, v, w, u, wkv_state)
            return y, wkv_state

        if self.discretization == "gen2":
            sp0 = F.softplus(self.disc_alpha0_raw).view(1, H, 1, 1).to(r.dtype)
            sp1 = F.softplus(self.disc_alpha1_raw).view(1, H, 1, 1).to(r.dtype)
            denom = sp0 + sp1 + 1e-6
            alpha_0, alpha_1 = sp0 / denom, sp1 / denom
            y, wkv_state = _multistep_wkv(
                r, k, v, w, u, wkv_state,
                alphas=(alpha_0, alpha_1),
                var_decay=True,
            )
            return y, wkv_state

        if self.discretization == "trap":
            y, wkv_state = _multistep_wkv(
                r, k, v, w, u, wkv_state,
                alphas=(0.5, 0.5), var_decay=False,
            )
            return y, wkv_state

        if self.discretization == "trap_var":
            y, wkv_state = _multistep_wkv(
                r, k, v, w, u, wkv_state,
                alphas=(0.5, 0.5), var_decay=True,
            )
            return y, wkv_state

        if self.discretization == "ab3":
            # Adams-Bashforth-3 — explicit, conditionally stable.
            # Clamp decay so step size Δ = -log(W) ≤ 0.7 → W ≥ exp(-0.7) ≈ 0.5.
            w_clamped = w.clamp(min=-0.7)
            y, wkv_state = _multistep_wkv(
                r, k, v, w_clamped, u, wkv_state,
                alphas=(23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0),
                var_decay=True,
            )
            return y, wkv_state

        raise ValueError(f"Unknown discretization: {self.discretization}")

    def _forward_recurrent_rse(
        self,
        r: torch.Tensor,      # (B, H, T, K)
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,      # log-decay (negative), per-channel (K)
        theta: torch.Tensor,  # (B, H, T, Bk) — rotation angle per 2x2 block
        state: Optional[torch.Tensor] = None,
        apply_bonus: bool = True,   # P²-RSE calls this twice and adds u externally
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunked RSE scan via complex-number reformulation.

        SO(2)×R+ acting on a 2-vector is multiplication by a complex scalar
        z_t = exp(-lambda_t + i*theta_t).  Pack each row-pair (S[2b], S[2b+1])
        into a complex c[b] and the recurrence S_t = G_t S_{t-1} + k v^T
        becomes c_t = z_t * c_{t-1} + k_c_t * v_t  (k_c_t = k[2b] + i*k[2b+1]).
        Readout y_t = r^T S_t = sum_b Re(conj(r_c_t[b]) * c_t[b, :]).

        Within a chunk of size T_c, the attention coefficient
            A[t, s, b] = z_t z_{t-1} ... z_{s+1} = exp(cumlog_z[t] - cumlog_z[s])
        is a complex scalar and the chunk is computed in parallel.  Inter-chunk
        state c is carried serially.  Bonus term `u` (real, current-step
        shortcut) is preserved per Proposal A §3.5.
        """
        B, H, T, K = r.shape
        Bk = K // 2
        device, in_dtype = r.device, r.dtype
        chunk_size = 64

        # Per-block log-decay (average of channel pairs).  RSE trades the
        # within-block decay asymmetry for the rotation angle.
        log_decay_block = w.view(B, H, T, Bk, 2).mean(dim=-1).float()  # (B,H,T,Bk)

        # ── Stage 5 viscosity coupling: λ_eff = λ_raw + η_{h,b} · θ² ──
        # log_decay is -λ (since λ>0 maps to log(decay)<0), so we SUBTRACT
        # η · θ² from log_decay (equivalent to adding it to λ).  This
        # implements Rayleigh dissipation: high-frequency rotations decay
        # faster, self-regulating phase coherence without a hard clip.
        if self.use_viscosity:
            theta_sq = theta.float() ** 2                                # (B,H,T,Bk)
            log_decay_block = log_decay_block - self.viscosity_eta.view(1, H, 1, Bk).float() * theta_sq

        # log z = -lambda + i*theta  (lambda = -log_decay > 0 since w is negative)
        log_z = torch.complex(log_decay_block, theta.float())          # (B,H,T,Bk)

        # Pack r and k into complex pairs along K dim
        r_pairs = r.float().view(B, H, T, Bk, 2)
        k_pairs = k.float().view(B, H, T, Bk, 2)
        r_c = torch.complex(r_pairs[..., 0], r_pairs[..., 1])          # (B,H,T,Bk)
        k_c = torch.complex(k_pairs[..., 0], k_pairs[..., 1])          # (B,H,T,Bk)
        v_f = v.float()                                                # (B,H,T,K)

        # Initial state as complex
        if state is None:
            c_state = torch.zeros(B, H, Bk, K, dtype=torch.complex64, device=device)
        else:
            S0 = state.float().view(B, H, Bk, 2, K)
            c_state = torch.complex(S0[..., 0, :], S0[..., 1, :])      # (B,H,Bk,K)

        # Bonus term `u` (real, per-channel).  `apply_bonus=False` forces skip
        # so P²-RSE can run the scan twice and add `u` exactly once externally.
        use_bonus_here = apply_bonus and not self.drop_u
        u_hk = self.time_faaaa.view(1, H, 1, K).float() if use_bonus_here else None

        out = torch.zeros(B, H, T, K, dtype=torch.float32, device=device)

        cur = 0
        while cur < T:
            tc = min(chunk_size, T - cur)
            log_z_c = log_z[:, :, cur:cur + tc]                        # (B,H,tc,Bk)
            k_c_c = k_c[:, :, cur:cur + tc]                            # (B,H,tc,Bk)
            r_c_c = r_c[:, :, cur:cur + tc]                            # (B,H,tc,Bk)
            v_c = v_f[:, :, cur:cur + tc]                              # (B,H,tc,K)

            # Cumulative log-z (inclusive): cum[t] = sum_{i<=t} log_z[i]
            cumlog = log_z_c.cumsum(dim=2)                             # (B,H,tc,Bk)

            # Within-chunk attention A[t, s, b] = exp(cumlog[t] - cumlog[s])
            #   for s == t: factor 1 (current-step direct write)
            #   for s <  t: product G_{s+1}..G_t
            #   for s >  t: zero (masked via torch.where after exp)
            diff = cumlog.unsqueeze(3) - cumlog.unsqueeze(2)           # (B,H,tc,tc,Bk)
            mask = torch.tril(
                torch.ones(tc, tc, device=device, dtype=torch.bool)
            ).view(1, 1, tc, tc, 1)
            # Stage 5 stability fix: clamp real part for safe exp, then zero
            # the upper triangle exactly with torch.where (replaces the
            # previous -60 clamp that leaked e^{-60} ≈ 1e-26 residuals).
            real_part = diff.real.masked_fill(~mask, -60.0)            # safe for exp
            A_raw = torch.exp(torch.complex(real_part, diff.imag))     # (B,H,tc,tc,Bk)
            A = torch.where(mask, A_raw, torch.zeros_like(A_raw))      # exact zero above

            # S_intra[t, b, c] = sum_s A[t,s,b] * k_c[s,b] * v[s,c]
            scaled_k = A * k_c_c.unsqueeze(2)                          # (B,H,tc,tc,Bk)
            v_c_complex = v_c.to(torch.complex64)
            S_intra = torch.einsum(
                'bhtsk,bhsc->bhtkc', scaled_k, v_c_complex
            )                                                          # (B,H,tc,Bk,K)

            # Inter-chunk: prior state contribution = exp(cumlog[t]) * c_state
            decay_to_t = torch.exp(cumlog)                             # (B,H,tc,Bk)
            prior_contrib = decay_to_t.unsqueeze(-1) * c_state.unsqueeze(2)  # (B,H,tc,Bk,K)

            S_total = prior_contrib + S_intra                          # (B,H,tc,Bk,K)

            # Readout: y[t,c] = Re( sum_b conj(r_c[t,b]) * S_total[t,b,c] )
            y_chunk = torch.einsum(
                'bhtk,bhtkc->bhtc', r_c_c.conj(), S_total
            ).real                                                     # (B,H,tc,K)

            # Bonus current-step shortcut (real)
            if u_hk is not None:
                r_t_r = r.float()[:, :, cur:cur + tc]
                k_t_r = k.float()[:, :, cur:cur + tc]
                scalar = (r_t_r * u_hk[:, :, 0:1] * k_t_r).sum(dim=-1, keepdim=True)
                y_chunk = y_chunk + scalar * v_c

            out[:, :, cur:cur + tc] = y_chunk

            # Carry state forward = S_total at last index of chunk
            c_state = S_total[:, :, -1]                                # (B,H,Bk,K)

            cur += tc

        # Convert final state back to real (B,H,K,K) for caller
        final_state = torch.zeros(B, H, K, K, dtype=torch.float32, device=device)
        final_view = final_state.view(B, H, Bk, 2, K)
        final_view[..., 0, :] = c_state.real
        final_view[..., 1, :] = c_state.imag

        return out.to(in_dtype), final_state

    def _forward_recurrent_p2rse(
        self,
        r: torch.Tensor,       # (B, H, T, K)
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,       # shared log-decay (negative), per-channel (K)
        theta_1: torch.Tensor, # (B, H, T, Bk) — mode 1 rotation
        theta_2: torch.Tensor, # (B, H, T, Bk) — mode 2 rotation (init = -theta_1)
        x_in: torch.Tensor,    # (B, T, D) — original input for β mixer
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage-5 P²-RSE: paired-pole state evolution with real β mixer.

        Runs the complex chunked RSE scan twice — once per mode (θ^(1), θ^(2)),
        shared λ/k/v — and fuses outputs with a data-dependent real mixer.
        Current-step bonus u is applied once at the end (both internal calls
        use `apply_bonus=False`).

        Mixer variants (self.p2rse_mixer):
          "linear"  — β = W_β x ∈ R^{2H}, unconstrained (primary).
          "softmax" — β = softmax(W_β x), convex (control).

        Phase-complementary init (θ^(2) = -θ^(1)) breaks the mode-exchange
        saddle at step 0 without expressivity constraints on the learned
        manifold.  State is (2, B, H, K, K) — two independent complex states.
        """
        B, H, T, K = r.shape
        device = r.device

        if state is None:
            s1_in, s2_in = None, None
        else:
            # Packed as (2, B, H, K, K); split per mode.
            s1_in = state[0].contiguous()
            s2_in = state[1].contiguous()

        # ── Two parallel RSE scans — bonus skipped (added once at end) ──
        y1, s1_out = self._forward_recurrent_rse(
            r, k, v, w, theta_1, s1_in, apply_bonus=False
        )
        y2, s2_out = self._forward_recurrent_rse(
            r, k, v, w, theta_2, s2_in, apply_bonus=False
        )

        # ── β mixer ────────────────────────────────────────────────────
        # (B, T, D) → (B, T, 2H) → (B, H, T, 2)
        beta_logits = self.beta_mixer(x_in).view(B, T, H, 2).transpose(1, 2)
        if self.p2rse_mixer == "softmax":
            beta = F.softmax(beta_logits, dim=-1)
        else:  # "linear" — unconstrained
            beta = beta_logits
        beta = beta.to(r.dtype)

        # β_m shape (B, H, T, 1) → broadcasts over K
        beta_1 = beta[..., 0:1]
        beta_2 = beta[..., 1:2]
        y = beta_1 * y1 + beta_2 * y2                                  # (B, H, T, K)

        # ── Bonus current-step shortcut (applied once, as in single-mode RSE) ──
        if not self.drop_u:
            u_hk = self.time_faaaa.view(1, H, 1, K).to(r.dtype)
            scalar = (r * u_hk * k).sum(dim=-1, keepdim=True)          # (B, H, T, 1)
            y = y + scalar * v                                         # (B, H, T, K)

        # ── Pack states as (2, B, H, K, K) ──
        new_state = torch.stack([s1_out, s2_out], dim=0)

        return y, new_state

    def _forward_recurrent_rse_multi(
        self,
        r: torch.Tensor,         # (B, H, T, K)
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,         # log-decay (negative), per-channel (K) — for scale 0
        theta_scale0: torch.Tensor,  # (B, H, T, Bk) — for scale 0
        x_in: torch.Tensor,      # (B, T, D) — original input for mixer
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-Rate RSE: M parallel rotation-decay scans with query-conditional mixer.

        Each scale m has independent (lambda_m, theta_m) per-block parameters
        and a fully separate complex state.  Outputs are mixed by
        β_t = softmax(W_β x_t) ∈ Δ^{M-1} (per head).

        State carried: (M, B, H, K, K) for streaming.
        """
        B, H, T, K = r.shape
        Bk = K // 2
        M = self.rse_n_scales
        device = r.device

        # ── Compute (log_decay, theta) per scale ──────────────────────────
        # Scale 0 — reuse channel-w averaged into Bk blocks, and theta from caller.
        log_dec_per_scale = [w.view(B, H, T, Bk, 2).mean(dim=-1).float()]   # (B,H,T,Bk)
        theta_per_scale = [theta_scale0.float()]                            # (B,H,T,Bk)

        # Scales 1..M-1 — independent LoRA-derived (lambda, theta).
        for idx, m in enumerate(range(1, M)):
            lam_lora = torch.tanh(x_in @ self.rse_extra_lambda_w1[idx]) @ self.rse_extra_lambda_w2[idx]
            lam_base = self.rse_extra_lambda_base[idx].view(1, 1, H * Bk)
            lam_pre = lam_base + lam_lora                                   # (B, T, H*Bk)
            # softplus → positive lambda; negate for log_decay (negative real part)
            lam_pos = F.softplus(lam_pre).view(B, T, H, Bk).transpose(1, 2)  # (B,H,T,Bk)
            log_dec_per_scale.append(-lam_pos)                              # (B,H,T,Bk)

            the_lora = torch.tanh(x_in @ self.rse_extra_theta_w1[idx]) @ self.rse_extra_theta_w2[idx]
            the_base = self.rse_extra_theta_base[idx].view(1, 1, H * Bk)
            theta_m = self.rse_theta_clip * torch.tanh(the_base + the_lora)
            theta_per_scale.append(theta_m.view(B, T, H, Bk).transpose(1, 2))  # (B,H,T,Bk)

        # ── Mixer: β_t = softmax(W_β x_t), per-head over scales ───────────
        # Shape (B, T, H*M) → (B, H, T, M)
        beta_logits = self.rse_mixer(x_in).view(B, T, H, M).transpose(1, 2)
        beta = F.softmax(beta_logits, dim=-1)                               # (B,H,T,M)

        # ── Initial state per scale ──────────────────────────────────────
        if state is None:
            states_in = [None] * M
        else:
            # state is (M, B, H, K, K); split per scale
            states_in = [state[m].contiguous() for m in range(M)]

        # ── Run M parallel scans (independent state, shared k/v) ─────────
        ys, states_out = [], []
        for m in range(M):
            log_dec_m = log_dec_per_scale[m]          # (B,H,T,Bk) negative
            theta_m = theta_per_scale[m]              # (B,H,T,Bk)
            # Reconstruct a "w_per_channel" view by repeating each block decay
            # across its 2 channels.  _forward_recurrent_rse averages it back.
            w_m = log_dec_m.unsqueeze(-1).expand(B, H, T, Bk, 2).reshape(B, H, T, K)
            y_m, s_m = self._forward_recurrent_rse(
                r, k, v, w_m, theta_m, states_in[m]
            )
            ys.append(y_m)
            states_out.append(s_m)

        # ── Mix outputs by β ─────────────────────────────────────────────
        # ys[m] is (B,H,T,K); beta[..., m] is (B,H,T)
        y_stack = torch.stack(ys, dim=-1)            # (B,H,T,K,M)
        y = (y_stack * beta.unsqueeze(-2)).sum(dim=-1)  # (B,H,T,K)

        # Pack final states into (M, B, H, K, K)
        new_state = torch.stack(states_out, dim=0)

        return y, new_state

    def _forward_bidir_serial(
        self,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """Bidirectional serial: forward recurrent + backward recurrent, merged."""
        B, H, T, K = r.shape

        # Forward pass
        y_fwd, _ = self._forward_recurrent(r, k, v, w, state=None)

        # Backward pass (flip time, run forward, flip back)
        r_b = r.flip(2)
        k_b = k.flip(2)
        v_b = v.flip(2)
        w_b = w.flip(2)
        y_bwd, _ = self._forward_recurrent(r_b, k_b, v_b, w_b, state=None)
        y_bwd = y_bwd.flip(2)

        return y_fwd + y_bwd


def _apply_lucid_recurrent(
    k: torch.Tensor,   # (B, H, T, K)
    v: torch.Tensor,   # (B, H, T, K)
    temp: torch.Tensor, # (H,) — per-head temperature (already softplus'd)
    chunk_size: int = 64,
) -> torch.Tensor:
    """Paper-faithful LUCID preconditioner for recurrent RWKV-6.

    Applies the LUCID preconditioner within fixed-size chunks:
        P = exp(K_RN @ K_RN^T / sqrt(d) - sqrt(d))  (unit diagonal)
        Solve P · Y = V for Y

    This preconditions the values BEFORE they enter the WKV state,
    matching the paper's formulation: O = A · P^{-1} · V.

    The inter-chunk recurrent state dynamics are unchanged.
    """
    B, H, T, K = k.shape
    sqrt_d = K ** 0.5

    # Reshape temp for broadcasting: (H,) -> (1, H, 1, 1)
    if temp.dim() == 1:
        temp = temp.view(1, H, 1, 1)

    v_out = torch.zeros_like(v)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        k_c = k[:, :, start:end, :]  # (B, H, cs, K)
        v_c = v[:, :, start:end, :]

        # RMSNorm: k_rn = sqrt(d) * k / ||k||_2
        k_rn = sqrt_d * F.normalize(k_c, dim=-1)

        # Gram matrix: diagonal = d
        gram = k_rn @ k_rn.transpose(-2, -1)  # (B, H, cs, cs)

        # Paper: exp(gram / sqrt(d) - sqrt(d)) — unit diagonal
        scaled = (temp * (gram / sqrt_d - sqrt_d)).clamp(-30, 30)
        P = torch.exp(scaled.float())  # (B, H, cs, cs)

        # Regularize for numerical stability (ensures P is always invertible)
        P = P + 1e-6 * torch.eye(end - start, device=P.device, dtype=P.dtype)

        # Solve P · Y = V for preconditioned values
        v_out[:, :, start:end, :] = torch.linalg.solve(P, v_c.float()).to(v.dtype)

    return v_out


def _chunked_wkv(
    r: torch.Tensor,   # (B, H, T, K)
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,   # log-decay (negative)
    u: torch.Tensor,   # (1, H, 1, K) bonus term
    wkv_state: torch.Tensor,  # (B, H, K, K)
    self_reg: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked parallel WKV with carry-state (SmerkyG algorithm).

    Processes T tokens using decreasing chunk sizes for GPU-parallel
    intra-chunk attention with sequential inter-chunk state updates.

    If self_reg=True, applies the RKHS delta rule self-regulation:
    the inter-chunk state update accumulates prediction errors instead
    of raw key-value products, suppressing redundant updates.
    """
    B, H, T, K = r.shape

    processed = 0
    remaining = T
    out_parts = []
    state = wkv_state

    for chunk_len in [128, 16, 2, 1]:
        while remaining >= chunk_len:
            mul = remaining // chunk_len
            seg_len = chunk_len * mul

            out, state = _wkv_subchunk(
                r[:, :, processed:processed + seg_len],
                k[:, :, processed:processed + seg_len],
                v[:, :, processed:processed + seg_len],
                w[:, :, processed:processed + seg_len],
                u, state,
                chunk_len=chunk_len,
                self_reg=self_reg,
            )
            out_parts.append(out)
            processed += seg_len
            remaining -= seg_len

    return torch.cat(out_parts, dim=2), state


def _wkv_subchunk(
    r: torch.Tensor,   # (B, H, L, K) — L must be exact multiple of chunk_len
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,   # log-decay (negative)
    u: torch.Tensor,   # (1, H, 1, K)
    wkv_state: torch.Tensor,  # (B, H, K, K)
    chunk_len: int = 128,
    self_reg: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parallel intra-chunk + recurrent inter-chunk WKV.

    Direct port of RWKVx060_subchunk_torch_inner from RWKV-block.

    If self_reg=True, applies the RKHS delta rule (erase-then-write):
      Standard:  S_new = decay * S + K^T @ V
      Self-reg:  S_new = decay * S + K^T @ (V - S @ K_norm^T)
    The state only accumulates prediction errors, suppressing
    redundant updates when the state already encodes the association.
    """
    B, H, L, K = k.shape
    V = v.size(-1)
    T = chunk_len

    # Single token: simple step
    if L == 1:
        kv = k.mT @ v
        if self_reg:
            k_norm = F.normalize(k, dim=-1)
            prediction = (wkv_state @ k_norm.mT).mT  # (B,H,1,V)
            error = v - prediction
            kv = k.mT @ error
        out = r @ (wkv_state + u.mT * kv)
        wkv_state = torch.exp(w).mT * wkv_state + kv
        return out, wkv_state

    assert L % T == 0
    N = L // T

    # Numerical stability: clamp decay factor
    precision_min_val = 0.005
    precision_dtype = torch.float64 if T > 24 else torch.float32
    # w is log-decay (negative): decay = exp(w) in (0, 1)
    w_decay = torch.exp(w).clamp(min=precision_min_val)

    # Cumulative log-decay
    w_log = w_decay.float().log()
    wc_log = w_log.view(w.size(0), H, N, T, K)
    wc_log_cum = wc_log.cumsum(dim=-2)
    shifted_wc_log_cum = F.pad(wc_log_cum, (0, 0, 1, -1))

    # Pre-compute decay weights
    ws = wc_log.sum(dim=-2, keepdim=True)
    w_inter = ws - wc_log_cum
    w_intra = wc_log_cum - wc_log

    ws_list = list(ws.mT.exp().to(r.dtype).unbind(dim=-3))
    w_inter = w_inter.exp().to(r.dtype)
    w_intra = w_intra.exp().to(r.dtype)

    # Reshape to chunks
    r = r.view(B, H, N, T, K)
    k = k.view(B, H, N, T, K)
    v = v.view(B, H, N, T, V)
    u_c = u.unsqueeze(2).to(r.dtype)

    # Parallel intra-chunk attention
    wc_log_offset = shifted_wc_log_cum[..., T // 2:T // 2 + 1, :]
    r_decay = (shifted_wc_log_cum - wc_log_offset).to(precision_dtype).exp()
    k_inv_decay = (wc_log_offset - wc_log_cum).to(precision_dtype).exp()
    a = ((r * r_decay) @ (k * k_inv_decay).mT).to(r.dtype).tril(-1)
    # Add bonus term on diagonal
    a = a + torch.einsum('bhntk,bhntk->bhnt', r, u_c * k).diag_embed()
    out = a @ v

    if not self_reg:
        # ── Standard state update ──────────────────────────────────
        wkv = (k * w_inter).mT @ v
        wkv_list = list(wkv.unbind(dim=-3))

        states = []
        for i in range(N):
            states.append(wkv_state)
            wkv_state = wkv_state * ws_list[i] + wkv_list[i]
        states = torch.stack(states, dim=2)
    else:
        # ── Self-regulating state update (RKHS delta rule) ─────────
        # S_new = decay * S + K^T @ (V - S @ K_norm^T)
        # The state suppresses updates for associations it already knows.
        states = []
        for i in range(N):
            states.append(wkv_state)
            k_i = k[:, :, i]         # (B, H, T, K)
            v_i = v[:, :, i]         # (B, H, T, V)
            w_inter_i = w_inter[:, :, i]  # (B, H, T, K)

            # What the state predicts for these keys
            k_norm_i = F.normalize(k_i.float(), dim=-1)
            # state @ k_norm^T → (B,H,K,K) @ (B,H,K,T) = (B,H,K,T)
            prediction = (wkv_state @ k_norm_i.mT).mT.to(v_i.dtype)  # (B,H,T,V)

            # Prediction error: only accumulate what's new
            error = v_i - prediction  # (B, H, T, V)

            # State update with error instead of raw values
            wkv_i = (k_i * w_inter_i).mT @ error
            wkv_state = wkv_state * ws_list[i] + wkv_i.float()
        states = torch.stack(states, dim=2)

    # Apply state to output
    out = out + (r * w_intra) @ states
    out = out.view(B, H, L, V)

    return out, wkv_state


# ────────────────────────────────────────────────────────────────────────
# Stage 2: Generalized linear-multistep state update
# ────────────────────────────────────────────────────────────────────────
#
# S_t = W_t S_{t-1} + Σ_i α_i · W̃_t^{i/p} ⊙ b_{t-i}     where b_t = k_t v_t^T
#
# Implemented as `p` parallel calls to the existing chunked WKV scan,
# exploiting linearity of the recurrence in the drive:
#
#   S = Σ_i S^(i),   where each S^(i) is driven by   α_i · W̃_t^{i/p} ⊙ b_{t-i}
#
# Each shifted drive becomes a re-scaled (k_lag, v_lag) pair fed through
# the SAME _chunked_wkv with the SAME `w` decay schedule.
#
# Output: y = Σ_i y^(i), state = Σ_i S^(i)_T  (linearity).
#
# Carry-state: this implementation re-runs the lookback drive from a
# zero state and discards the small boundary error at the chunk seam
# (k_{-1}, k_{-2} from the previous chunk). For long-utterance training
# this is exact within a forward pass; for streaming inference it
# introduces a one-token transient at each chunk boundary. A cleaner
# carry-state implementation that propagates the per-scan substate is
# tracked in the discretization study's follow-up work.


def _shift_lag(x: torch.Tensor, k: int) -> torch.Tensor:
    """Shift along time axis (dim=2) by k steps, zero-pad the front."""
    if k <= 0:
        return x
    return F.pad(x[:, :, :-k, :], (0, 0, k, 0))


def _multistep_wkv(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    wkv_state: torch.Tensor,
    alphas,
    var_decay: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear-multistep WKV: sum of `p` parallel chunked scans.

    Args:
        r, k, v: (B, H, T, K) — receptance, key, value
        w:       (B, H, T, K) — log-space decay (negative)
        u:       (1, H, 1, K) — bonus term applied to current step at readout
        wkv_state: (B, H, K, K) — carry-state input
        alphas:  tuple of length p — coefficients (α₀, α₁, [α₂, ...]).
                 Each is either a Python float OR a tensor broadcastable
                 to (B, H, T, K) — supports per-head learnable α (gen2).
        var_decay: if True use geometric-mean decay W̃_t = sqrt(W_t W_{t-1});
                   if False use the current decay W_t for the lookback.

    Returns: (y, new_state) of shapes (B, H, T, K), (B, H, K, K).
    """
    p = len(alphas)
    B, H, T, K = k.shape

    if var_decay:
        w_lag1 = _shift_lag(w, 1)
        w_geo1 = (w + w_lag1) * 0.5  # log-space geometric mean for lag-1
    else:
        w_geo1 = w  # use current decay for the lookback weighting

    # Scan 0 — current step. α₀ scales the drive (k v^T), bonus `u` retained.
    a0 = alphas[0]
    if isinstance(a0, float):
        k0 = k * a0
    else:
        k0 = k * a0.to(k.dtype)
    y, state_total = _chunked_wkv(r, k0, v, w, u, wkv_state)

    # Lookback scans — α_i and W̃^i absorbed into the shifted k_lag.
    # Lookback drives carry no bonus term (u was a current-step extra weight).
    zero_u = torch.zeros_like(u)
    zero_state = torch.zeros_like(wkv_state)
    for i in range(1, p):
        a_i = alphas[i]
        k_lag = _shift_lag(k, i)
        v_lag = _shift_lag(v, i)
        # W̃^i along K dim — exp in log-space then accumulate
        if i == 1:
            decay_factor = torch.exp(w_geo1.float()).to(k.dtype)
        else:
            # Higher-order lookback: composed geometric-mean decay over `i` steps.
            # For AB3 this is W̃² ≈ exp((w_t + w_{t-1} + w_{t-2})/2 · 2/i).
            # We approximate as i × w_geo1 (sufficient for stability-bounded AB3
            # since the additional approximation error is dominated by truncation).
            decay_factor = torch.exp((w_geo1.float() * i)).to(k.dtype)
        if isinstance(a_i, float):
            scale = decay_factor * a_i
        else:
            scale = decay_factor * a_i.to(k.dtype)
        k_b = k_lag * scale
        y_i, s_i = _chunked_wkv(r, k_b, v_lag, w, zero_u, zero_state.clone())
        y = y + y_i
        state_total = state_total + s_i

    return y, state_total
