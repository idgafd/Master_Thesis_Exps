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


# ── Stage 10.1 — Static precompute for chunked log-linear scan ──────────
# Per-chunk-size local-level masks.  `mask[chunk_len][ell, tau, sigma] = 1`
# iff source position sigma belongs to local bucket level ell at the time
# target position tau reads inside the chunk (tau, sigma ∈ [0, C)).
#
# Relies on two facts:
#   1. Chunk length C = 2^J and chunk starts at multiples of C ⇒
#      levels 0..J-1 are empty on entry (prior cascade at position `start`
#      fires up to level ≥ J).
#   2. Intra-chunk cascade events stay below level J (the chunk-end
#      collapse at position `start + C` is handled separately, propagating
#      local levels 0..J-1 into global level J or higher).
_LOGLINEAR_LEVEL_MASKS: dict = {}


def _build_local_level_masks(C: int, L_max: int) -> torch.Tensor:
    """(J, C, C) bool tensor; J = min(log2(C), L_max). Empty if J = 0."""
    import math as _math
    assert C >= 1 and (C & (C - 1)) == 0, f"chunk_len must be power of 2: {C}"
    J = int(_math.log2(C))
    J = min(J, L_max)
    if J == 0:
        return torch.zeros(0, C, C, dtype=torch.bool)
    mask = torch.zeros(J, C, C, dtype=torch.bool)
    buckets_py: list[list[int]] = [[] for _ in range(J)]
    for tau in range(C):
        # Readout at tau sees σ ∈ buckets[0..J-1].
        for ell in range(J):
            if buckets_py[ell]:
                for sigma in buckets_py[ell]:
                    mask[ell, tau, sigma] = True
        # Push tau into bucket 0 (post-readout).
        buckets_py[0].append(tau)
        # Intra-chunk cascade: cap at ell < J (chunk-end cascade is separate).
        ell = 1
        while ell < J and ((tau + 1) % (1 << ell)) == 0:
            buckets_py[ell].extend(buckets_py[ell - 1])
            buckets_py[ell - 1] = []
            ell += 1
    return mask


def _get_level_masks(chunk_len: int, L_max: int, device: torch.device) -> torch.Tensor:
    """Cached, device-specific level masks."""
    key = (chunk_len, L_max, str(device))
    if key not in _LOGLINEAR_LEVEL_MASKS:
        _LOGLINEAR_LEVEL_MASKS[key] = _build_local_level_masks(chunk_len, L_max).to(device)
    return _LOGLINEAR_LEVEL_MASKS[key]


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
        delta_warmstart: bool = False,  # TODO_DELTA_RULE §5 H1 — init a0 at -5
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
        # ── Stage 5 Phase 2b — Independent-λ P²-RSE ──────────────────────
        # Relaxes the shared-λ constraint of Phase-1 P²-RSE. Each pole gets
        # its own decay LoRA (time_decay_*_2) in addition to the already-
        # independent θ LoRA. See src/models/p2rse_indep_lambda.py and
        # TODO_FUTURE_IDEAS.md §Stage-5 deferred for rationale.
        # Requires p2rse=True.  Init: pole-2 base λ = clone of pole-1;
        # pole-2 LoRA zero → indep-λ path reduces exactly to shared-λ P²-RSE
        # at t=0 (zero-regression-at-init contract).
        p2rse_indep_lambda: bool = False,
        # ── Stage 5 Phase 2b-ext — Independent drive-side (k, v) per pole ─
        # On top of Phase 2b: pole 2 gets rank-`p2rse_kv_lora_dim` LoRA
        # deltas on BOTH the key and value projections, so the two poles
        # can read from different subspaces of the input rather than only
        # differing in their transition (λ, θ). Motivated by
        # STAGE5_RESULTS §6.4(2): physical formants read from different
        # cochlear-spectrum regions, not identical ones.  Requires
        # p2rse_indep_lambda=True.  Zero-regression-at-init: LoRA down-
        # projections init zero ⇒ k_2=k, v_2=v until SGD grows them.
        p2rse_indep_kv: bool = False,
        p2rse_kv_lora_dim: int = 32,
        # ── Stage 6 EXPRESSIVENESS paper adaptations ─────────────────────
        # Per-head RMSNorm replacing GroupNorm on the WKV output — paper's
        # §4.5 finding that any vector norm approximates the softmax
        # denominator G_t (L2/RMS/LayerNorm all tied; GroupNorm's cross-
        # token statistic is an outlier we can remove).
        use_rmsnorm: bool = False,
        # Diagonal n=2 Taylor branch: run a parallel WKV scan on (k⊙k, r⊙r)
        # with decay^2 (i.e. log-decay ×2) and zero bonus. β_hadamard is
        # learnable per-head, zero-init. This is the Hadamard approximation
        # of the paper's (k⊗k, r⊗r) Kronecker lift — no cross-channel terms.
        use_hadamard_n2: bool = False,
        # Full Kronecker n=2 tail: k⊗k, r⊗r giving K² features per head;
        # decay w⊕w at pair (i,j) = w_i+w_j (log). β_qtail is zero-init.
        # Caller should enable this only at select (top) layers for cost.
        use_qtail: bool = False,
        # Learnable per-head decay coupling γ on the Kronecker branch.
        #   w_pair[i,j] = γ_h · (w_i + w_j)
        # γ=1.0: current qtail (product of per-channel decays — bit-exact).
        # γ=0.5: geometric mean (Kronecker branch decays as slowly as linear).
        # γ=0.0: paper's undecayed accumulator (H_t^{(n)} = H_{t-1}^{(n)} + …).
        # γ>1.0: Kronecker branch decays faster than linear (short-term role).
        # Paper's Taylor derivation imposes NO decay on order-n state, so γ
        # is principled: SGD picks where on the spectrum to sit.  Init γ=1.0
        # ⇒ bit-exact reduction to vanilla qtail at t=0.  Requires use_qtail.
        use_qtail_gamma: bool = False,
        # R2 — Data-dependent β mixer for the Kronecker branch.
        #   β_{q,t}(x) = β_static + (W_β · x_t)    (per-head, per-token)
        # W_β is a Linear(hidden_size, n_head) layer with zero-init weight
        # AND zero-init bias, so at t=0 the data-dependent term is zero and
        # β_{q,t} = β_static (which is also zero-init per self.beta_qtail).
        # Gives the Kronecker branch a selective gating mechanism analogous
        # to Mamba-2's selective scan — β can fire on tokens where cross-
        # channel features matter (coarticulation boundaries, formant
        # transitions) and suppress elsewhere.  Requires use_qtail.
        use_qtail_dbeta: bool = False,
        # Low-rank Kronecker — project r,k to K' << K per head, then full
        # Kronecker on K'² features instead of K² = 4096.  Tests whether
        # qtail's gain survives under Eckart–Young truncation.  If yes,
        # enables Kronecker at all layers and scale-up to larger models.
        # qtail_lr_rank: projected per-head dim K'. K'=16 gives K'²=256,
        # a 16× memory reduction vs full qtail. Decay: per-head mean of w,
        # doubled for the Kronecker lift (matches the average behaviour of
        # w_i + w_j averaged across channel pairs).
        use_qtail_lowrank: bool = False,
        qtail_lr_rank: int = 16,
        # H1 — Per-(head, channel-pair) β allocation on the Kronecker branch.
        # Hypothesis: a per-head scalar β has to absorb both "is this
        # mechanism useful here" and "which channel-pairs matter" — too
        # coarse for SGD to allocate the lift cleanly.  This flag adds
        # `beta_qtail_per_pair` of shape (n_head, K2) where K2 = K'² (low-
        # rank) or K² (full), init 1.0, applied as an element-wise scale on
        # k_kron (gates per-pair contributions to the lifted state).  Zero-
        # regression-at-init still holds via the outer `beta_qtail` per-head
        # scalar (zero-init); β_pp gets gradient once β_qtail moves, same
        # γ-style mobility pattern used in qtail_gamma.  Requires use_qtail.
        use_qtail_beta_per_pair: bool = False,
        # H2 — Init value for `qtail_gamma`.  Default 1.0 ⇒ bit-exact reduction
        # to the no-gamma baseline at t=0 (current behaviour).  Set to 0.0 to
        # test the "undecayed Kronecker accumulator" hypothesis: with γ=0 the
        # lifted state has no decay (paper's literal Taylor formulation).
        # Only meaningful when use_qtail_gamma=True.
        qtail_gamma_init: float = 1.0,
        # ── Stage 7A (A1′) — Data-dependent readout phase φ_{t,h,b} ─────
        # Adds a learnable per-(token, head, block) phase that rotates the
        # complex readout contraction before .real collapses it:
        #     y_t = Σ_b Re( exp(-iφ_{t,h,b}) · conj(r_c_{t,h,b}) · c_{t,h,b} )
        # Implementation: φ(x) = tanh(W_φ · x_in / C) · C with C=π (soft-
        # clipped full-circle).  W_φ and b_φ are zero-init ⇒ φ=0 at t=0 ⇒
        # readout is bit-identical to `rwkv6_rse_*_viscosity` anchor at
        # init.  Motivated by the Stage-7 diagnostic (STAGE7_DIAGNOSTICS.md
        # §D2): per-block |Im|/|Re| ≈ 1 and global ratio reaches 0.70 at
        # L5 — static per-head-block phase cannot reshape this, but a
        # per-token phase can.  Requires rse=True.
        use_data_dep_readphase: bool = False,
        # Clip on |φ| via tanh.  π gives full-circle coverage; smaller
        # values stabilise training but cap the recoverable gauge.
        readphase_clip: float = math.pi,
        # ── Stage 8 T2 — non-normal RSE in polar parameterisation ───────
        # Per-block transition extends from e^{-λ} R(θ) to the 4-DOF form
        #     G = e^{-λ} · R(ψ)^T diag(e^ρ, e^{-ρ}) R(ψ) · R(θ)
        # = rotation-decay × anisotropic-symmetric × rotation.
        # Four parameters per (head, block, token): λ, θ, ρ, ψ.
        # ρ = 0 ⇒ P = I ⇒ exact RSE reduction.  Requires rse=True.
        # See STAGE8_PLAN.md §4 for motivation and stability analysis.
        use_nonnormal_rse: bool = False,
        # Stability clip: |ρ| ≤ κ · softplus(λ̃_block). κ < 1 keeps ρ
        # strictly subordinate to damping; conservative vs the exact
        # envelope ρ² < λ² + θ² which requires stable Jordan-boundary
        # avoidance at runtime.
        nonnormal_rho_kappa: float = 0.6,
        # LoRA rank for the ρ and ψ data-dependent projections.
        nonnormal_lora_dim: int = 32,
        # ── Stage 9 — Sparse edge-layer specialist transition ───────────
        # Adds a per-(layer, head) gate `sparse_nn_gate`, zero-init, which
        # multiplies the realised ρ and ψ. SGD picks which (ℓ, h) slots
        # activate the non-normal extension.  At init g=0 ⇒ ρ_eff=ψ_eff=0
        # ⇒ exact RSE+viscosity reduction.  Requires use_nonnormal_rse=True.
        # Motivation: STAGE8_RESULTS §3 — T2's D9 showed bimodal per-head
        # specialisation (L0 head 3 at |ρ|=0.46; other heads ≈ 0.06). The
        # dense T2 form spread freedom uniformly; Stage 9 makes sparsity
        # structural via the gate.
        use_sparse_nonnormal_rse: bool = False,
        # Hard edge-layer-only variant: freeze gate on middle layers so
        # only L0 and L_{n-1} can learn to activate.  Option B from
        # STAGE9_PLAN §2.2.  Default False = Option A (learned sparsity).
        sparse_nn_edge_only: bool = False,
        # Stage 9 Fix 3 — drop the token-dependent ψ LoRA, keep only a
        # static ψ_base per (head, block).  Simplifies identifiability
        # (ψ becomes a slow-timescale direction, not an extra token-level
        # freedom); trims front-end compute too.  Requires use_nonnormal_rse.
        nonnormal_psi_static: bool = False,
        # ── Stage 10.1 — Log-Linear RWKV-6 (Fenwick bucket readout) ──────
        # L bucket states partition the WKV prefix at log-scale.  Per-token
        # per-scale mixer λ_t^(ℓ) = 1 + W_λ^(2) tanh(W_λ^(1) · x_shifted);
        # W_λ^(1) zero-init ⇒ λ ≡ 1 ⇒ Σ_ℓ λ · r^T · S^(ℓ) = r^T · S = vanilla
        # RWKV-6 readout bit-exact at t=0.  Requires mode='recurrent' and not
        # composed with RSE/P²-RSE/non-normal paths (kept orthogonal for
        # Stage 10.1 attribution; Stage 10.7 composition is a separate run).
        use_loglinear: bool = False,
        loglinear_levels: int = 10,
        # ── Stage 10.2 — M²RNN sparing-use (single-layer non-linear state) ─
        # Parallel branch Z=tanh(S·W + kv^T), gated forget-update, added to
        # the RWKV readout with scalar λ_h (zero-init ⇒ bit-exact).  Active
        # at one layer only (default: the top layer of a 6-layer stack).
        use_m2rnn: bool = False,
        m2rnn_layer: int = 5,
        # ── Stage 10.3 — Multi-dilation ConvShift ───────────────────────
        # Replaces the single-dilation DWConvShift with parallel dilated
        # branches (1, 2, 4, 8) and learnable per-layer α_d. Requires
        # conv_shift=True.
        use_conv_shift_multidilation: bool = False,
        # Padding mode for the multi-dilation ConvShift: "auto" chooses
        # causal in mode=recurrent, symmetric otherwise. "symmetric" forces
        # symmetric even in recurrent (10.3-sym apples-to-apples control).
        conv_shift_multidil_padding_mode: str = "auto",
        # CB-3 — content-conditional α_d on multi-dilation ConvShift.
        conv_shift_multidil_content_conditional: bool = False,
        # ── Stage 10.5 — Cayley-orthogonal transition ────────────────────
        # G_t = exp(-λ_t) · O_t where O_t = (I - A_t)(I + A_t)^{-1} and
        # A_t = U_t V_t^T - V_t U_t^T (rank-2·cayley_rank skew). At
        # U=V=0 init, A=0 ⇒ O=I ⇒ scan reduces to vanilla RWKV-6 bit-exact.
        # cayley_rank=1 keeps param parity; higher ranks break parity.
        use_cayley_orthogonal: bool = False,
        cayley_rank: int = 1,
        cayley_lora_dim: int = 32,
        # ── Stage 10.6 — PoM polynomial value-lift ──────────────────────
        # v̂_t = v_t + Σ_{p=2..k} γ_p ⊙ (W_h x_t)^⊙p; γ=0 at init ⇒ v̂=v.
        # The WKV state update is unchanged; only v is lifted.
        use_pom_vlift: bool = False,
        pom_order: int = 2,
        pom_expansion: int = 64,
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
        self.delta_warmstart = delta_warmstart
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
        self.p2rse_indep_lambda = p2rse_indep_lambda
        self.p2rse_indep_kv = p2rse_indep_kv
        self.p2rse_kv_lora_dim = p2rse_kv_lora_dim
        self.use_rmsnorm = use_rmsnorm
        self.use_hadamard_n2 = use_hadamard_n2
        self.use_qtail = use_qtail
        self.use_qtail_gamma = use_qtail_gamma
        self.use_qtail_dbeta = use_qtail_dbeta
        self.use_qtail_lowrank = use_qtail_lowrank
        self.qtail_lr_rank = qtail_lr_rank
        self.use_qtail_beta_per_pair = use_qtail_beta_per_pair
        self.qtail_gamma_init = qtail_gamma_init
        self.use_data_dep_readphase = use_data_dep_readphase
        self.readphase_clip = readphase_clip
        self.use_nonnormal_rse = use_nonnormal_rse
        self.nonnormal_rho_kappa = nonnormal_rho_kappa
        self.use_sparse_nonnormal_rse = use_sparse_nonnormal_rse
        self.sparse_nn_edge_only = sparse_nn_edge_only
        self.nonnormal_psi_static = nonnormal_psi_static
        self.use_loglinear = use_loglinear
        self.loglinear_levels = loglinear_levels
        self.use_m2rnn = use_m2rnn
        self.m2rnn_layer = m2rnn_layer
        self.m2rnn_active = use_m2rnn and (layer_id == m2rnn_layer)
        self.use_conv_shift_multidilation = use_conv_shift_multidilation
        self.use_cayley_orthogonal = use_cayley_orthogonal
        self.cayley_rank = cayley_rank
        self.cayley_lora_dim = cayley_lora_dim
        self.use_pom_vlift = use_pom_vlift
        self.pom_order = pom_order
        self.pom_expansion = pom_expansion
        self.layer_id = layer_id

        if self.use_rse:
            assert head_size % 2 == 0, "RSE requires even head_size (2x2 blocks)"
            assert mode == "recurrent", "RSE currently supports only mode='recurrent'"
            assert rse_n_scales >= 1

        if self.use_p2rse:
            assert self.use_rse, "P²-RSE requires rse=True"
            assert rse_n_scales == 1, "P²-RSE incompatible with multi-rate rse_n_scales > 1"
            assert self.p2rse_mixer in ("linear", "softmax")

        if self.use_data_dep_readphase:
            assert self.use_rse, "Data-dependent readout phase requires rse=True"

        if self.use_nonnormal_rse:
            assert self.use_rse, "Non-normal RSE requires rse=True"
            assert mode == "recurrent", "Non-normal RSE requires mode='recurrent'"
            assert rse_n_scales == 1, "Non-normal RSE incompatible with multi-rate scales > 1"
            assert not self.use_p2rse, "Non-normal RSE not composed with P²-RSE (Stage-8 scope)"

        if self.use_sparse_nonnormal_rse:
            assert self.use_nonnormal_rse, "Sparse nonnormal_rse requires use_nonnormal_rse=True"

        if self.use_loglinear:
            assert mode == "recurrent", "Log-Linear RWKV-6 requires mode='recurrent'"
            # Stage 10.1 is orthogonal to RSE / P²-RSE / non-normal paths;
            # Stage 10.7 composition re-enables the RSE×Log-Linear combo with
            # its own complex-bucket reformulation (not this code path).
            assert not self.use_rse, (
                "use_loglinear=True is not composed with RSE in Stage 10.1 "
                "(use the Stage 10.7 `rwkv6_loglinear_rse_strong_viscosity` path)"
            )
            assert self.loglinear_levels >= 1

        if self.use_m2rnn:
            assert mode == "recurrent", "M²RNN sparing-use requires mode='recurrent'"
            assert not self.use_rse, (
                "Stage 10.2 M²RNN is orthogonal to RSE (Family C vs Family B) "
                "per STAGE10_PLAN §6; keep attribution clean."
            )
            assert 0 <= m2rnn_layer < num_hidden_layers, (
                f"m2rnn_layer={m2rnn_layer} out of range for {num_hidden_layers} layers"
            )

        if self.use_conv_shift_multidilation:
            assert self.use_conv_shift, (
                "use_conv_shift_multidilation requires conv_shift=True"
            )

        if self.use_cayley_orthogonal:
            assert mode == "recurrent", "Cayley-orthogonal requires mode='recurrent'"
            assert not self.use_rse, (
                "Stage 10.5 Cayley is a separate Family-B branch; "
                "keep orthogonal to RSE for attribution."
            )
            assert not self.use_loglinear and not self.use_m2rnn, (
                "Cayley-orthogonal is orthogonal to 10.1/10.2 for clean attribution."
            )
            assert cayley_rank >= 1

        if self.use_pom_vlift:
            assert pom_order >= 2, "pom_order < 2 is a no-op"
            assert pom_expansion >= 1

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
        # Post-WKV normalization: either the standard RWKV GroupNorm (per-head
        # group across the hidden-size-att axis, computed over B*T) or a
        # per-head RMSNorm as recommended by the EXPRESSIVENESS paper §4.5.
        self._rmsnorm_eps = (1e-5) * (self.head_size_divisor ** 2)
        if self.use_rmsnorm:
            # Per-head-per-channel learnable scale; no bias (RMSNorm convention).
            self.rmsnorm_scale = nn.Parameter(torch.ones(n_head, head_size, dtype=dtype))
        else:
            self.ln_x = nn.GroupNorm(
                n_head, hidden_size_att, dtype=dtype,
                eps=self._rmsnorm_eps,
            )

        # ── Paper's n=2 Taylor branches (zero-regression at init) ────────
        if self.use_hadamard_n2:
            self.beta_hadamard = nn.Parameter(torch.zeros(n_head, dtype=dtype))
        if self.use_qtail:
            self.beta_qtail = nn.Parameter(torch.zeros(n_head, dtype=dtype))
            # Learnable per-head decay coupling γ on the Kronecker-lifted
            # pair-decay w_pair[i,j] = γ · (w_i + w_j). Init `qtail_gamma_init`
            # (default 1.0 ⇒ bit-exact reduction; 0.0 ⇒ undecayed accumulator
            # H2 hypothesis).  Requires use_qtail.
            if self.use_qtail_gamma:
                self.qtail_gamma = nn.Parameter(
                    torch.full((n_head,), float(self.qtail_gamma_init), dtype=dtype)
                )
            # R2 — Data-dependent β projector: Linear(hidden, n_head).
            # Zero-init weights AND bias ⇒ data-dep term adds exactly 0 at
            # t=0.  Combined with static beta_qtail (zero-init), effective
            # β is 0 at t=0 → bit-exact reduction to baseline (Kronecker
            # branch gated off). SGD then grows both components.
            if self.use_qtail_dbeta:
                self.beta_qtail_proj = nn.Linear(hidden_size, n_head, bias=True, dtype=dtype)
                nn.init.zeros_(self.beta_qtail_proj.weight)
                nn.init.zeros_(self.beta_qtail_proj.bias)
            # Low-rank Kronecker — per-head K → K' projections.
            # Two separate projections for r and k (may learn different
            # subspaces useful for the Kronecker interaction vs selfdot).
            # Default init: Kaiming uniform — the Kronecker branch is
            # gated by β=0 at init anyway, so projection init doesn't
            # affect zero-regression.
            if self.use_qtail_lowrank:
                r = self.qtail_lr_rank
                self.qtail_lr_proj_r = nn.Parameter(torch.empty(n_head, head_size, r, dtype=dtype))
                self.qtail_lr_proj_k = nn.Parameter(torch.empty(n_head, head_size, r, dtype=dtype))
                nn.init.kaiming_uniform_(self.qtail_lr_proj_r, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.qtail_lr_proj_k, a=math.sqrt(5))
            # H1 — per-(head, lifted-pair) β allocation, init 1.0.
            # Lifted dim K2 = K'² for low-rank or K² for full Kronecker.
            # Acts as element-wise scale on k_kron in the lifted state update;
            # outer β_qtail (zero-init) still gates the whole branch off at t=0.
            if self.use_qtail_beta_per_pair:
                K2_pp = (self.qtail_lr_rank ** 2) if self.use_qtail_lowrank else (head_size ** 2)
                self.beta_qtail_per_pair = nn.Parameter(
                    torch.ones(n_head, K2_pp, dtype=dtype)
                )

        # ── Mechanism: ConvShift ─────────────────────────────────────────
        if self.use_conv_shift:
            if self.use_conv_shift_multidilation:
                from src.models.mechanisms.conv_shift import MultiDilationDWConvShift
                # Padding selection: "auto" = causal-in-recurrent (plan default),
                # "symmetric" / "causal" override for controls.
                if conv_shift_multidil_padding_mode == "auto":
                    pad_mode = "causal" if mode == "recurrent" else "symmetric"
                else:
                    assert conv_shift_multidil_padding_mode in ("causal", "symmetric"), (
                        f"bad conv_shift_multidil_padding_mode: {conv_shift_multidil_padding_mode}"
                    )
                    pad_mode = conv_shift_multidil_padding_mode
                self.conv_shift_module = MultiDilationDWConvShift(
                    hidden_size,
                    kernel_size=3,
                    dilations=(1, 2, 4, 8),
                    padding_mode=pad_mode,
                    content_conditional=conv_shift_multidil_content_conditional,
                )
            else:
                from src.models.mechanisms.conv_shift import DWConvShift
                self.conv_shift_module = DWConvShift(hidden_size)

        # ── Mechanism: Headscale ─────────────────────────────────────────
        if self.use_headscale:
            # (1, H, 1, K) for broadcasting against w_h of shape (B, H, T, K).
            self.head_decay_bias = nn.Parameter(torch.zeros(1, n_head, 1, head_size))

        # ── Mechanism: Delta Rule ────────────────────────────────────────
        if self.use_delta_rule:
            from src.models.mechanisms.delta_rule import DeltaRuleParams
            self.delta_params = DeltaRuleParams(
                hidden_size, n_head, head_size,
                warmstart=self.delta_warmstart,
                dtype=dtype,
            )
            # Stage-8 T1 — recurrent-path hard gate for exact zero-init
            # contract.  Multiplies the iclr (erasure strength) inside the
            # sequential recurrent delta scan.  Zero-init ⇒ rank-1 erase
            # is inactive at step 0 ⇒ output bit-identical to vanilla
            # RWKV-6 recurrent.  SGD grows per-head `delta_recurrent_gate`
            # only if the mechanism is productive. This parameter is
            # unused on the LION path (which uses delta_params alone).
            self.delta_recurrent_gate = nn.Parameter(
                torch.zeros(n_head, dtype=dtype)
            )

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

        # ── Stage 10.1 Log-Linear — λ-mixer LoRA (zero-init W1, small W2) ──
        # λ_t^(ℓ) = 1 + (W_λ^(2) · tanh(W_λ^(1) x_shift))_{ℓ}  per head.
        # At init W_λ^(1) = 0 ⇒ λ ≡ 1 ⇒ Σ_ℓ λ^(ℓ) · r^T S^(ℓ) = r^T S.
        if self.use_loglinear:
            L = self.loglinear_levels
            D_LOGLIN_DIM = 32
            self.loglinear_lam_w1 = nn.Parameter(
                torch.zeros(hidden_size, D_LOGLIN_DIM, dtype=dtype)
            )
            self.loglinear_lam_w2 = nn.Parameter(
                torch.zeros(D_LOGLIN_DIM, n_head * L, dtype=dtype).uniform_(-0.01, 0.01)
            )

        # ── Stage 10.2 M²RNN — single-layer parallel non-linear branch ─────
        # Active at layer `m2rnn_layer` only.  At init:
        #   W_h = I_K per head (cosmetic; irrelevant while λ=0 gates the branch off)
        #   λ_h = 0  ⇒  bit-exact vanilla RWKV-6 readout
        #   W_f = 0  ⇒  f_t = sigmoid(0) = 0.5 (forget-update at midpoint)
        if self.m2rnn_active:
            K = head_size
            # (H, K, K): start each head at I_K
            eye = torch.eye(K, dtype=dtype).unsqueeze(0).expand(n_head, K, K).clone()
            self.m2rnn_W = nn.Parameter(eye)
            # λ_h: (H,)
            self.m2rnn_lambda = nn.Parameter(torch.zeros(n_head, dtype=dtype))
            # Paper-faithful forget gate (Mishra et al., arXiv:2603.14360):
            #     z_{t,h} = W_f x_t + β_h        (pre-activation)
            #     f_{t,h} = 1 / (1 + exp(z))^{α_h}        with α_h > 0
            # α_h must be positive for f ∈ (0, 1]; we keep it positive by
            # reparameterising α = softplus(α_raw) + ε.  At init α_raw = ln(e-1)
            # so softplus(α_raw) = 1.0, matching the paper's default α_h = 1
            # (which reduces f to sigmoid(-z)).
            self.m2rnn_forget_proj = nn.Linear(hidden_size, n_head, bias=False, dtype=dtype)
            nn.init.zeros_(self.m2rnn_forget_proj.weight)
            self.m2rnn_forget_alpha_raw = nn.Parameter(
                torch.full((n_head,), math.log(math.e - 1.0), dtype=dtype)
            )
            self.m2rnn_forget_beta = nn.Parameter(torch.zeros(n_head, dtype=dtype))

        # ── Stage 10.5 Cayley-orthogonal — rank-`cayley_rank` U, V LoRAs ──
        # Static base U, V ∈ (H, K) per rank index; zero-init.
        # Data-dependent LoRA with zero-init down-projection so U=V=0 at init
        # regardless of the xw stream content.  Matches T2's deployment shape
        # (dense per-token) for apples-to-apples diagnostic comparison.
        if self.use_cayley_orthogonal:
            R = self.cayley_rank
            K = head_size
            D_CAY = self.cayley_lora_dim
            # Base per-head rotation axes (static, zero-init)
            self.cayley_U_base = nn.Parameter(
                torch.zeros(R, n_head, K, dtype=dtype)
            )
            self.cayley_V_base = nn.Parameter(
                torch.zeros(R, n_head, K, dtype=dtype)
            )
            # LoRA pairs (data-dependent, zero-init down-projection)
            self.cayley_U_w1 = nn.Parameter(
                torch.zeros(hidden_size, D_CAY, dtype=dtype)
            )
            self.cayley_U_w2 = nn.Parameter(
                torch.zeros(D_CAY, R * n_head * K, dtype=dtype).uniform_(-0.01, 0.01)
            )
            self.cayley_V_w1 = nn.Parameter(
                torch.zeros(hidden_size, D_CAY, dtype=dtype)
            )
            self.cayley_V_w2 = nn.Parameter(
                torch.zeros(D_CAY, R * n_head * K, dtype=dtype).uniform_(-0.01, 0.01)
            )

        # ── Stage 10.6 PoM polynomial value-lift — thin config ────────────
        # v̂_t = v_t + γ_2 ⊙ (W_h x_t)^⊙2  (k=2 thin variant)
        # W_h expands input to pom_expansion dims; then element-wise square;
        # γ_2 ∈ R^{hidden_size_att} is zero-init so v̂=v at t=0.
        # Up-projection back to hidden_size_att via a second linear.
        if self.use_pom_vlift:
            D_POM = self.pom_expansion
            self.pom_W_h = nn.Linear(hidden_size, D_POM, bias=False, dtype=dtype)
            # Up-projection from D_POM back to hidden_size_att for element-wise
            # gamma-weighted addition to v.
            self.pom_W_up = nn.Linear(D_POM, hidden_size_att, bias=False, dtype=dtype)
            # γ per order (for k=2 we have γ_2 only); zero-init for bit-exact v.
            self.pom_gamma = nn.Parameter(
                torch.zeros(self.pom_order - 1, hidden_size_att, dtype=dtype)
            )

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

            # ── Stage 7A (A1′) — data-dependent readout phase parameters ─
            # W_φ, b_φ zero-init ⇒ φ(x) ≡ 0 ⇒ rotation matrix is I at t=0.
            if self.use_data_dep_readphase:
                self.readphase_proj = nn.Linear(
                    hidden_size, n_head * n_blocks, bias=True, dtype=dtype
                )
                nn.init.zeros_(self.readphase_proj.weight)
                nn.init.zeros_(self.readphase_proj.bias)

            # ── Stage 8 T2 — non-normal RSE polar-form parameters ──────
            # ρ: anisotropy magnitude.  ψ: anisotropy axis (direction).
            #   ρ_base ≡ 0, ρ LoRA weights ≡ 0  ⇒  ρ(x) ≡ 0  ⇒  P = I
            #   ψ_base ~ U(0, 2π) — arbitrary direction at init because
            #     ψ has no effect when ρ = 0 (P ≡ I). Re-rolled per SGD
            #     once ρ > 0 starts to matter.
            # Stability-related parameters:
            #   μ: viscosity coupling for ρ², symmetric with η on θ².
            if self.use_nonnormal_rse:
                # Base magnitudes — zero-init for exact RSE reduction
                self.nonnormal_rho_base = nn.Parameter(
                    torch.zeros(n_head, n_blocks, dtype=dtype)
                )
                # Direction base — arbitrary init (irrelevant at ρ = 0)
                self.nonnormal_psi_base = nn.Parameter(
                    torch.empty(n_head, n_blocks, dtype=dtype).uniform_(0.0, 2 * math.pi)
                )
                # Data-dependent LoRAs (rank nonnormal_lora_dim, zero-init)
                D_NN = nonnormal_lora_dim
                self.nonnormal_rho_w1 = nn.Parameter(
                    torch.zeros(hidden_size, D_NN, dtype=dtype)
                )
                self.nonnormal_rho_w2 = nn.Parameter(
                    torch.zeros(D_NN, n_head * n_blocks, dtype=dtype).uniform_(-0.01, 0.01)
                )
                # ψ LoRA is skipped under Stage-9 static-ψ flag: ψ stays
                # as a per-(head, block) direction parameter, constant
                # across tokens.  Fewer DOF, cleaner identifiability.
                if not self.nonnormal_psi_static:
                    self.nonnormal_psi_w1 = nn.Parameter(
                        torch.zeros(hidden_size, D_NN, dtype=dtype)
                    )
                    self.nonnormal_psi_w2 = nn.Parameter(
                        torch.zeros(D_NN, n_head * n_blocks, dtype=dtype).uniform_(-0.01, 0.01)
                    )
                # Viscosity coupling for ρ² (symmetric with η on θ²)
                self.nonnormal_mu = nn.Parameter(
                    torch.zeros(n_head, n_blocks, dtype=dtype)
                )

                # Stage 9 — per-head sparse gate.
                # g_h = sigmoid(raw_h) ∈ (0, 1).  Nonnegative, bounded;
                # avoids the signed-scalar redundancy where the gate and
                # ρ-sign would both carry the direction of anisotropy.
                # Applied to ρ only (not ψ): ρ=0 already kills non-normality
                # via sinh(0)=0 regardless of ψ, so gating ψ is redundant.
                # Init raw=0 ⇒ g=0.5 (neutral).  Exact regression at init
                # holds via ρ_raw zero-init (tanh(0)=0 ⇒ ρ_eff=0), NOT via
                # gate=0 — a cleaner identifiability story.
                if self.use_sparse_nonnormal_rse:
                    self.sparse_nn_gate_raw = nn.Parameter(
                        torch.zeros(n_head, dtype=dtype)
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

                # ── Phase 2b: Independent-λ P²-RSE ──────────────────────
                # Second pole gets its OWN decay LoRA (base + rank-D_DECAY_DIM
                # projection).  Init:
                #   base: clone of self.time_decay → same λ schedule at t=0
                #   LoRA weights: zero → no contribution at t=0
                # Zero-regression-at-init: indep-λ path reduces EXACTLY to
                # shared-λ P²-RSE until SGD moves the new LoRA.  Symmetry is
                # broken immediately by the phase-complementary θ^(2)
                # initialisation (above) and by SGD thereafter.
                if self.p2rse_indep_lambda:
                    self.time_decay_2 = nn.Parameter(self.time_decay.detach().clone())
                    self.time_decay_w1_2 = nn.Parameter(
                        torch.zeros(hidden_size, D_DECAY_DIM, dtype=dtype)
                    )
                    self.time_decay_w2_2 = nn.Parameter(
                        torch.zeros(D_DECAY_DIM, hidden_size_att, dtype=dtype).uniform_(-0.01, 0.01)
                    )

                # ── Phase 2b-ext: Independent drive-side (k, v) per pole ───
                # Rank-D_KV LoRA deltas on top of the shared key/value.
                # down-init 0 ⇒ pole-2 drive == pole-1 drive at t=0.
                # Stage-3 RSE convention: up matrix uniform(-0.01, 0.01).
                if self.p2rse_indep_kv:
                    D_KV = self.p2rse_kv_lora_dim
                    # key LoRA pair for pole 2 (delta from self.key)
                    self.key_lora_a_2 = nn.Parameter(
                        torch.zeros(hidden_size, D_KV, dtype=dtype)
                    )
                    self.key_lora_b_2 = nn.Parameter(
                        torch.zeros(D_KV, hidden_size_att, dtype=dtype).uniform_(-0.01, 0.01)
                    )
                    # value LoRA pair for pole 2 (delta from self.value)
                    self.value_lora_a_2 = nn.Parameter(
                        torch.zeros(hidden_size, D_KV, dtype=dtype)
                    )
                    self.value_lora_b_2 = nn.Parameter(
                        torch.zeros(D_KV, hidden_size_att, dtype=dtype).uniform_(-0.01, 0.01)
                    )

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

        # Stage 10.6 PoM value-lift: v̂ = v + Σ_p γ_p ⊙ W_up(h(W_h x)^⊙p).
        # γ=0 at init ⇒ v̂ = v bit-exact.  Applied before the WKV scan so
        # every existing scan path (vanilla, RSE, p2rse, …) consumes the
        # lifted value without further changes.
        if self.use_pom_vlift:
            z = self.pom_W_h(x)                                              # (B, T, D_pom)
            acc = torch.zeros_like(v)
            for p_idx in range(self.pom_order - 1):
                p = p_idx + 2  # p = 2, 3, ...
                z_p = z ** p                                                 # (B, T, D_pom)
                up = self.pom_W_up(z_p)                                      # (B, T, hidden_size_att)
                acc = acc + self.pom_gamma[p_idx].view(1, 1, -1) * up
            v = v + acc.to(v.dtype)

        w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = w.to(r.dtype)

        if self.use_rse:
            # Reuse the xw mixing path for theta (the rotation degree of freedom
            # is informationally analogous to decay — both modulate the transition).
            theta_lora = torch.tanh(xw @ self.time_theta_w1) @ self.time_theta_w2
            theta = self.rse_theta_clip * torch.tanh(self.time_theta + theta_lora)
            theta = theta.to(r.dtype)

            # Stage 8 T2 — non-normal RSE ρ, ψ data-dependent projections.
            # Computed from the SAME xw stream as θ (one token-shift pass for
            # all LoRAs).  Returned as raw tensors (B, T, H*Bk); the stability
            # clip (κ · softplus(λ̃)) and reshape to (B, H, T, Bk) are applied
            # in forward() where n_blocks is available.
            if self.use_nonnormal_rse:
                rho_lora = torch.tanh(xw @ self.nonnormal_rho_w1) @ self.nonnormal_rho_w2
                rho_raw = self.nonnormal_rho_base.view(1, 1, -1) + rho_lora       # (B, T, H*Bk)

                # Stage 9 Fix 3 — static ψ option: emit ψ_base broadcast
                # across (B, T) instead of computing a token-dependent LoRA.
                if self.nonnormal_psi_static:
                    psi_raw = self.nonnormal_psi_base.view(1, 1, -1).expand(
                        xw.shape[0], xw.shape[1], -1
                    )
                else:
                    psi_lora = torch.tanh(xw @ self.nonnormal_psi_w1) @ self.nonnormal_psi_w2
                    psi_raw = self.nonnormal_psi_base.view(1, 1, -1) + psi_lora  # (B, T, H*Bk)

                rho_raw = rho_raw.to(r.dtype)
                psi_raw = psi_raw.to(r.dtype)
                return r, k, v, g, w, theta, rho_raw, psi_raw

            if self.use_p2rse:
                # Second pole's theta — independent LoRA, same xw / clip.
                theta_lora_2 = torch.tanh(xw @ self.time_theta_w1_2) @ self.time_theta_w2_2
                theta_2 = self.rse_theta_clip * torch.tanh(self.time_theta_2 + theta_lora_2)
                theta_2 = theta_2.to(r.dtype)
                # ── Phase 2b: Independent-λ — also return second w ──
                # Mirrors the w computation above, using the pole-2 LoRA.
                # At init, time_decay_2 is a clone of time_decay and the
                # LoRA weights are zero ⇒ w_2 == w at t=0 (zero-regression
                # contract; indep-λ reduces to shared-λ until SGD moves it).
                if self.p2rse_indep_lambda:
                    w_2 = self.time_decay_2 + torch.tanh(xw @ self.time_decay_w1_2) @ self.time_decay_w2_2
                    w_2 = w_2.to(r.dtype)

                    # ── Phase 2b-ext: Independent drive-side (k, v) ────
                    # Pole-2 key/value = shared + LoRA delta. The LoRA uses
                    # the same token-shifted input (xk, xv) as the shared
                    # projection to preserve the Stage-3 token-shift semantics.
                    # At init, key_lora_a_2 = value_lora_a_2 = 0 ⇒ pole-2
                    # drive matches shared drive exactly.
                    if self.p2rse_indep_kv:
                        k_2 = k + (xk @ self.key_lora_a_2) @ self.key_lora_b_2
                        v_2 = v + (xv @ self.value_lora_a_2) @ self.value_lora_b_2
                        k_2 = k_2.to(r.dtype)
                        v_2 = v_2.to(r.dtype)
                        return r, k, v, g, w, theta, theta_2, w_2, k_2, v_2
                    return r, k, v, g, w, theta, theta_2, w_2
                return r, k, v, g, w, theta, theta_2

            return r, k, v, g, w, theta

        # Non-RSE path also returns xw — consumed by Stage 10.1 loglinear
        # λ-mixer (paper-spec decay-side stream).  RSE variants don't need
        # it separately because their LoRAs already run off xw internally.
        return r, k, v, g, w, xw

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.size()
        H = self.n_head
        K = self.head_size

        if self.use_p2rse:
            if self.p2rse_indep_lambda:
                if self.p2rse_indep_kv:
                    # Phase 2b-ext — 10-tuple: indep λ + indep drive
                    (r, k, v, g, w, theta, theta_2, w_2, k_2, v_2) = self._compute_rkv_gw(x)
                else:
                    # Phase 2b — 8-tuple: indep λ only
                    r, k, v, g, w, theta, theta_2, w_2 = self._compute_rkv_gw(x)
                    k_2 = None
                    v_2 = None
            else:
                r, k, v, g, w, theta, theta_2 = self._compute_rkv_gw(x)
                w_2 = None
                k_2 = None
                v_2 = None
        elif self.use_rse:
            if self.use_nonnormal_rse:
                r, k, v, g, w, theta, rho_raw, psi_raw = self._compute_rkv_gw(x)
            else:
                r, k, v, g, w, theta = self._compute_rkv_gw(x)
                rho_raw = None
                psi_raw = None
            theta_2 = None
        else:
            r, k, v, g, w, xw = self._compute_rkv_gw(x)
            theta = None
            theta_2 = None
            rho_raw = None
            psi_raw = None

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
            if self.p2rse_indep_lambda:
                # Phase 2b — each pole carries its own log-decay.
                # Compute pole-2 log-decay tensor the same way the shared
                # path does for pole 1 (negate exp of raw w_2 LoRA output).
                w_h_2 = w_2.view(B, T, H, K).transpose(1, 2)
                w_h_2 = -torch.exp(w_h_2)
                if self.use_headscale:
                    w_h_2 = w_h_2 + self.head_decay_bias.to(w_h_2.dtype)

                # Phase 2b-ext: reshape independent pole-2 k/v if present.
                if k_2 is not None:
                    k_h_2 = k_2.view(B, T, H, K).transpose(1, 2)
                    v_h_2 = v_2.view(B, T, H, K).transpose(1, 2)
                else:
                    k_h_2 = None
                    v_h_2 = None

                y, new_state = self._forward_recurrent_p2rse_indeplam(
                    r_h, k_h, v_h, w_h, w_h_2, theta_h_1, theta_h_2, x, state,
                    k_h_2=k_h_2, v_h_2=v_h_2,
                )
            else:
                y, new_state = self._forward_recurrent_p2rse(
                    r_h, k_h, v_h, w_h, theta_h_1, theta_h_2, x, state
                )
        elif self.use_rse:
            # theta arrives as (B, T, H*Bk); reshape to (B, H, T, Bk)
            n_blocks = K // 2
            theta_h = theta.view(B, T, H, n_blocks).transpose(1, 2)
            # Stage 7A (A1′) — data-dependent readout phase tensor.
            # φ = clip · tanh(W_φ x + b_φ / clip).  Zero-init W_φ, b_φ ⇒ φ≡0.
            phi_h = None
            if self.use_data_dep_readphase:
                phi_raw = self.readphase_proj(x)                 # (B, T, H*Bk)
                phi = self.readphase_clip * torch.tanh(
                    phi_raw / self.readphase_clip
                )
                phi_h = phi.view(B, T, H, n_blocks).transpose(1, 2)  # (B,H,T,Bk)

            if self.use_nonnormal_rse:
                # Stage 8 T2 — ρ and ψ come from _compute_rkv_gw (one xw pass
                # shared with θ and w LoRAs).  Apply stability clip and reshape.
                # Stability clip: |ρ| ≤ κ · softplus(λ̃_block).  λ̃ is the raw
                # (pre-exp) decay averaged over the channel pair per block.
                lam_raw_block = w.view(B, T, H, n_blocks, 2).mean(dim=-1).float()  # (B,T,H,Bk)
                lam_soft = F.softplus(lam_raw_block)
                rho_bthb = rho_raw.view(B, T, H, n_blocks).float()
                rho_h_pre = (self.nonnormal_rho_kappa * lam_soft * torch.tanh(rho_bthb))
                psi_bthb = psi_raw.view(B, T, H, n_blocks).float()

                # Stage 9 — per-head sparse gate on ρ amplitude only.
                # g_{ℓ,h} = sigmoid(raw) ∈ (0, 1).  Applied to ρ only; ψ
                # un-gated because non-normality already vanishes when
                # ρ=0.  Zero-regression at init holds via ρ_raw LoRA
                # zero-init (tanh(ρ_raw)=0 ⇒ ρ_eff=0).
                if self.use_sparse_nonnormal_rse:
                    gate = torch.sigmoid(
                        self.sparse_nn_gate_raw
                    ).view(1, 1, H, 1).float()                              # (1,1,H,1)
                    rho_h_pre = rho_h_pre * gate

                rho_h = rho_h_pre.permute(0, 2, 1, 3).contiguous()     # (B, H, T, Bk)
                psi_h = psi_bthb.permute(0, 2, 1, 3).contiguous()      # (B, H, T, Bk)

                # Extend viscosity: λ_eff = λ + η θ² + μ ρ².  η already
                # absorbed inside _forward_recurrent_rse; for the non-normal
                # scan we apply μ·ρ² directly here by modifying w_h.
                if self.use_viscosity:
                    # Also apply the η·θ² coupling for consistency with the
                    # RSE anchor (it's added inside _forward_recurrent_rse).
                    # For nonnormal scan we fold both into w_h before the scan.
                    theta_sq = theta_h.float() ** 2                         # (B,H,T,Bk)
                    eta_bk = self.viscosity_eta.view(1, H, 1, n_blocks).float()
                    mu_bk = self.nonnormal_mu.view(1, H, 1, n_blocks).float()
                    visc_contrib_block = (
                        eta_bk * theta_sq + mu_bk * rho_h.float() ** 2
                    )                                                       # (B,H,T,Bk)
                    # Distribute block-level contribution back onto channel pairs
                    # (both channels in a block get the same viscosity add).
                    visc_contrib_chan = visc_contrib_block.unsqueeze(-1).expand(
                        -1, -1, -1, -1, 2
                    ).reshape(B, H, T, K)
                    w_h_eff = w_h.float() - visc_contrib_chan               # λ_eff = λ + visc in log-space
                    w_h_eff = w_h_eff.to(w_h.dtype)
                else:
                    w_h_eff = w_h

                # Bonus u (reuse the existing time_faaaa convention)
                u_bonus = self.time_faaaa.view(1, H, 1, K).to(r_h.dtype)
                y, new_state = _recurrent_nonnormal_rse_scan(
                    r_h, k_h, v_h, w_h_eff, theta_h, rho_h, psi_h, u_bonus,
                    state=state, apply_bonus=not self.drop_u,
                )
            elif self.rse_n_scales == 1:
                y, new_state = self._forward_recurrent_rse(
                    r_h, k_h, v_h, w_h, theta_h, state, phi=phi_h
                )
            else:
                y, new_state = self._forward_recurrent_rse_multi(
                    r_h, k_h, v_h, w_h, theta_h, x, state
                )
        elif self.mode == "lion":
            y = self._forward_lion(r_h, k_h, v_h, w_h)
        elif self.mode == "recurrent":
            # R2 — precompute data-dep β tensor if the flag is active.
            # β_proj(x) returns (B, T, H); transpose to (B, H, T) for the
            # scan's time-major layout.
            beta_tok = None
            if self.use_qtail_dbeta:
                beta_tok = self.beta_qtail_proj(x).transpose(1, 2)  # (B, H, T)
            if self.use_loglinear:
                # Stage 10.1 — Log-Linear Fenwick bucket readout.
                # λ-mixer reads the decay-side stream x^(w) per Paper-1 spec,
                # plumbed explicitly from _compute_rkv_gw's non-RSE return.
                y, new_state = self._forward_recurrent_loglinear(
                    r_h, k_h, v_h, w_h, xw, state
                )
            elif self.use_cayley_orthogonal:
                # Stage 10.5 — Cayley-orthogonal transition (rank-`cayley_rank`).
                # U, V come from zero-init-LoRA on the xw stream, so at t=0
                # the scan is bit-exact vanilla RWKV-6 (A=0 ⇒ O=I).
                # rank-1 uses the chunked affine-scan fast path; higher ranks
                # fall back to the generic sequential Woodbury path.
                if self.cayley_rank == 1:
                    y, new_state = self._forward_recurrent_cayley_rank1_chunked(
                        r_h, k_h, v_h, w_h, xw, state
                    )
                else:
                    y, new_state = self._forward_recurrent_cayley(
                        r_h, k_h, v_h, w_h, xw, state
                    )
            else:
                y, new_state = self._forward_recurrent(r_h, k_h, v_h, w_h, state, beta_tok=beta_tok)
            # Stage 10.2 — M²RNN parallel branch at the configured layer only.
            if self.m2rnn_active:
                y = y + self._m2rnn_branch(r_h, k_h, v_h, x)
        elif self.mode == "bidir_serial":
            y = self._forward_bidir_serial(r_h, k_h, v_h, w_h)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        y = y.to(r.dtype)

        # Reshape back and apply post-WKV norm (GroupNorm or per-head RMSNorm),
        # then gate and project.  y is (B, H, T, K) coming from the scan.
        if self.use_rmsnorm:
            # per-head L2/RMS norm across the head dim K
            y_htk = y.transpose(1, 2).contiguous()                     # (B, T, H, K)
            rms = y_htk.pow(2).mean(dim=-1, keepdim=True).add(self._rmsnorm_eps).rsqrt()
            y_htk = y_htk * rms * self.rmsnorm_scale.to(y_htk.dtype).view(1, 1, H, K)
            y = y_htk.reshape(B, T, D)
        else:
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
        beta_tok: Optional[torch.Tensor] = None,  # R2: data-dep β_{q,t}, (B, H, T)
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
            # ── Stage 8 T1 — recurrent delta rank-1 erase ────────────────
            # Gated by use_delta_rule (set via backbone-name substring) and
            # mutually exclusive with LUCID / LUCID-self-reg / n=2 Taylor
            # branches (mixing these would confound attribution).
            if self.use_delta_rule:
                kk, iclr = self.delta_params.compute_kk_iclr(
                    k, k.shape[0], k.shape[2], self.n_head, self.head_size
                )
                y, wkv_state = _recurrent_delta_scan(
                    r, k, v, w, u, kk, iclr, self.delta_recurrent_gate, wkv_state
                )
                return y, wkv_state

            y, wkv_state = _chunked_wkv(r, k, v, w, u, wkv_state)

            # ── Paper's n=2 Taylor branches — additive, β init 0 ─────────
            # `use_hadamard_n2` and `use_qtail` are mutually exclusive at a
            # given layer; if both are set somehow, run both independently.
            if self.use_hadamard_n2 or self.use_qtail:
                y = y + self._paper_n2_branch(r, k, v, w, B, H, T, K, beta_tok=beta_tok)

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

    def _forward_recurrent_loglinear(
        self,
        r: torch.Tensor,       # (B, H, T, K)
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,       # log-decay (negative); actual per-step decay = exp(w)
        x_in: torch.Tensor,    # (B, T, D) — feeds the λ-LoRA
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 10.1 — Chunked Fenwick bucket scan.

        Exact decomposition (not an approximation) based on Fenwick-alignment
        of the cascade: on entry to a chunk of length C = 2^J starting at a
        multiple of C, levels 0..J-1 are provably empty (the prior cascade
        fired up to level ≥ J). Inside the chunk, local kv contributions
        only populate levels 0..J-1, while the carry in levels ≥ J only
        decays. At chunk end, all local levels collapse into level
        J + v₂(chunk_index).

        Schedule mirrors `_chunked_wkv`: [128, 16, 2, 1]. Each chunk_len is
        a power of 2; consecutive chunk sizes are divisors of their
        predecessors, so Fenwick alignment is preserved as we descend the
        schedule.

        Matches `_forward_recurrent_loglinear_seq` within fp32 numerical
        precision. State input is ignored (carry-state disabled for
        loglinear via encoder.supports_carry_state).
        """
        import math as _math

        B, H, T, K = r.shape
        L = self.loglinear_levels
        device, in_dtype = r.device, r.dtype

        # λ (B, H, T, L) from decay-side xw stream.
        lam_lora = torch.tanh(x_in @ self.loglinear_lam_w1) @ self.loglinear_lam_w2
        lam = 1.0 + lam_lora.view(B, T, H, L).permute(0, 2, 1, 3).float()   # (B, H, T, L)

        # Buckets: (L, B, H, K, K).  Zero-init (state ignored).
        buckets = torch.zeros(L, B, H, K, K, dtype=torch.float32, device=device)

        # Keep (1, H, 1, K) shape so it broadcasts over both batch and time axes
        # when multiplied with rc / kc of shape (B, H, C, K).
        u_bhTK = self.time_faaaa.view(1, H, 1, K).float() if not self.drop_u else None

        out_parts = []
        processed = 0
        remaining = T

        # Cast working tensors to fp32 once (w is already fp on -exp(...))
        r_f = r.float()
        k_f = k.float()
        v_f = v.float()
        w_f = w.float()

        for chunk_len in [128, 16, 2, 1]:
            # Assert chunk_len is a power of 2.
            assert chunk_len & (chunk_len - 1) == 0
            num_chunks = remaining // chunk_len
            if num_chunks == 0:
                continue
            J = int(_math.log2(chunk_len))  # number of local levels

            # Precomputed level masks (cached per chunk_len + device).
            if J > 0:
                J_eff = min(J, L)
                level_masks = _get_level_masks(chunk_len, L_max=J_eff, device=device)  # (J_eff, C, C) bool
                # Pre-build tril mask for σ < τ (strict).
                tril_mask = torch.tril(
                    torch.ones(chunk_len, chunk_len, device=device, dtype=torch.bool),
                    diagonal=-1,
                )  # (C, C)

            for _ in range(num_chunks):
                start = processed
                end = processed + chunk_len
                k_chunk_idx = end // chunk_len   # 1-based

                # Per-chunk slices (fp32)
                rc = r_f[:, :, start:end]                                  # (B, H, C, K)
                kc = k_f[:, :, start:end]
                vc = v_f[:, :, start:end]
                wc = w_f[:, :, start:end]
                lamc = lam[:, :, start:end]                                # (B, H, C, L)

                # Prefix decays (per-channel).
                wc_cum = wc.cumsum(dim=2)                                  # (B, H, C, K) inclusive
                wc_prev = F.pad(wc_cum[:, :, :-1], (0, 0, 1, 0))           # (B, H, C, K) exclusive

                # ── 1. Carry contribution: levels ≥ J ──
                # r_carry[τ] = r[τ] ⊙ exp(wc_prev[τ]) — decay-adjusted r
                r_carry = rc * wc_prev.exp()                               # (B, H, C, K)
                # per_level_carry[ell][τ, v] = r_carry[τ] · buckets[ell][:, v]
                # shape: (L-J, B, H, C, K)
                if J < L:
                    buckets_carry = buckets[J:]                             # (L-J, B, H, K, K)
                    # One einsum for all carry levels.
                    per_level_carry = torch.einsum(
                        'bhtk,lbhkv->lbhtv', r_carry, buckets_carry
                    )                                                      # (L-J, B, H, C, K)
                    lam_carry = lamc[:, :, :, J:L]                         # (B, H, C, L-J)
                    # Weighted sum over carry levels.
                    y_carry = torch.einsum(
                        'bhtl,lbhtv->bhtv', lam_carry, per_level_carry
                    )                                                      # (B, H, C, K)
                else:
                    y_carry = torch.zeros(B, H, chunk_len, K, dtype=torch.float32, device=device)

                # ── 2. Local contribution: levels 0..J-1 ──
                # Intra-chunk attention A[τ, σ, i] = exp(wc_prev[τ, i] - wc_cum[σ, i])
                # for σ < τ (strict lower triangular), else 0.
                if J > 0:
                    # Stability: the upper triangle (σ ≥ τ) has +positive diffs
                    # that can overflow exp() for long chunks, producing
                    # inf × 0 = NaN after masking. Clamp to -60 BEFORE exp,
                    # then torch.where to zero exactly (mirrors the fix in
                    # _forward_recurrent_rse line 1337).
                    diff = wc_prev.unsqueeze(3) - wc_cum.unsqueeze(2)       # (B, H, C, C, K)
                    mask4d = tril_mask.view(1, 1, chunk_len, chunk_len, 1)
                    safe_diff = diff.masked_fill(~mask4d, -60.0)
                    A = torch.exp(safe_diff)
                    A = torch.where(mask4d, A, torch.zeros_like(A))

                    # Combined level-weighted mask: λ̃[τ, σ] = Σ_ℓ λ^(ℓ)[τ] · level_masks[ℓ, τ, σ]
                    # shape (B, H, C, C).  Exploits the fact that level_masks
                    # are a partition of {σ < τ}, so this collapses the
                    # per-level loop into one einsum.
                    lam_local = lamc[:, :, :, :J_eff]                      # (B, H, C, J_eff)
                    lam_weighted_mask = torch.einsum(
                        'bhtl,lts->bhts', lam_local, level_masks.to(A.dtype)
                    )                                                      # (B, H, C, C)

                    # β[τ, σ] = Σ_i r[τ, i] · A[τ, σ, i] · k[σ, i]  — per-channel contraction.
                    # Then y_local[τ, v] = Σ_σ λ̃[τ, σ] · β[τ, σ] · v[σ, v].
                    # Fuse into one einsum:
                    #   y_local[τ, v] = Σ_σ λ̃[τ, σ] · Σ_i r[τ, i]·A[τ, σ, i]·k[σ, i] · v[σ, v]
                    #                = Σ_σ Σ_i λ̃[τ, σ] · r[τ, i]·A[τ, σ, i]·k[σ, i] · v[σ, v]
                    # Intermediate β_scaled = λ̃ ⊙ β.
                    beta = torch.einsum(
                        'bhti,bhsi,bhtsi->bhts', rc, kc, A
                    )                                                      # (B, H, C, C)
                    beta_scaled = beta * lam_weighted_mask
                    y_local = torch.einsum(
                        'bhts,bhsv->bhtv', beta_scaled, vc
                    )                                                      # (B, H, C, K)
                else:
                    y_local = torch.zeros_like(y_carry)

                # ── 3. Bonus (current-step shortcut) ──
                if u_bhTK is not None:
                    bonus = (rc * u_bhTK * kc).sum(-1, keepdim=True) * vc
                else:
                    bonus = 0.0

                out_parts.append(y_carry + y_local + bonus)

                # ── 4. Chunk-end update + collapse ──
                # Decay all buckets by total_log = wc_cum[:, :, -1] (per-channel).
                total_log = wc_cum[:, :, -1]                               # (B, H, K)
                buckets = buckets * total_log.exp().unsqueeze(0).unsqueeze(-1)   # (L, B, H, K, K)

                # Local J-collapse: contributions of local σ at chunk end.
                # kv[σ] at chunk end has been decayed by exp(Σ_{u=σ+1..C-1} wc[u])
                # = exp(total_log - wc_cum[σ]).
                decay_to_end = (total_log.unsqueeze(2) - wc_cum).exp()     # (B, H, C, K)
                kc_scaled = kc * decay_to_end                              # (B, H, C, K)
                local_J = torch.einsum('bhck,bhcv->bhkv', kc_scaled, vc)   # (B, H, K, K)

                # Target level: ℓ_max = min(L-1, J + v₂(k_chunk_idx)).
                v2 = 0
                kk = k_chunk_idx
                while (kk & 1) == 0 and kk > 0:
                    v2 += 1
                    kk >>= 1
                ell_max = min(L - 1, J + v2)

                # Merge: levels 0..ell_max collapse into ell_max.
                # Levels 0..J-1 were empty on entry; after decay they're still
                # empty (zero times anything = zero).  So only J..ell_max carry
                # non-zero content that participates.
                if ell_max >= J:
                    merged = local_J
                    if ell_max >= J:
                        merged = merged + buckets[J:ell_max + 1].sum(dim=0)
                    # Assemble new buckets tensor.
                    new_buckets = torch.zeros_like(buckets)
                    new_buckets[ell_max] = merged
                    if ell_max + 1 < L:
                        new_buckets[ell_max + 1:] = buckets[ell_max + 1:]
                    buckets = new_buckets
                else:
                    # ell_max < J: shouldn't happen because ell_max ≥ J always.
                    raise RuntimeError(f"unexpected cascade: ell_max={ell_max}, J={J}")

                processed += chunk_len
                remaining -= chunk_len

        out = torch.cat(out_parts, dim=2)                                  # (B, H, T, K)
        final_state = buckets.sum(dim=0).contiguous()
        return out.to(in_dtype), final_state

    def _forward_recurrent_loglinear_seq(
        self,
        r: torch.Tensor,       # (B, H, T, K)
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,       # log-decay (negative); actual per-step decay = exp(w)
        x_in: torch.Tensor,    # (B, T, D) — feeds the λ-LoRA
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 10.1 — Log-Linear RWKV-6 Fenwick bucket scan (SEQUENTIAL REFERENCE).

        Maintains L bucket states {S^(ℓ)} that partition the WKV prefix at
        log-scale. At step t:
          1. All buckets decay by the per-channel factor exp(w_t).
          2. Contribution k_t ⊗ v_t enters bucket 0.
          3. Fenwick cascade: for ℓ = 1..L-1, if (t+1) is a multiple of 2^ℓ,
             bucket[ℓ] ← bucket[ℓ] + bucket[ℓ-1]; bucket[ℓ-1] ← 0.
        The partition identity Σ_ℓ S^(ℓ)_t = S_t (vanilla WKV state) holds at
        every t. Readout:
            y_t = Σ_ℓ λ_t^(ℓ) · r_t^T · S^(ℓ)_t  +  bonus
        with λ_t^(ℓ) = 1 + (W_λ^(2) · tanh(W_λ^(1) · x_t))_{h, ℓ}, W_λ^(1)=0
        at init ⇒ λ ≡ 1 ⇒ Σ_ℓ λ · r^T · S^(ℓ) = r^T · S = vanilla RWKV-6
        readout bit-exact at t=0.

        Implementation note: sequential Python loop over T; adequate for dry-run
        profiling. A chunked / Triton port is the first Reviewer target.
        """
        B, H, T, K = r.shape
        L = self.loglinear_levels
        device, in_dtype = r.device, r.dtype

        # λ tensor (B, H, T, L), 1.0 + zero-init LoRA at t=0.
        lam_lora = torch.tanh(x_in @ self.loglinear_lam_w1) @ self.loglinear_lam_w2
        lam = 1.0 + lam_lora.view(B, T, H, L).permute(0, 2, 1, 3)        # (B, H, T, L)
        lam = lam.float()

        # Per-step real decay factor (actual, in (0, 1)).
        decay_step = torch.exp(w.float())                                  # (B, H, T, K)

        # Buckets packed as (L, B, H, K, K). A per-bucket indicator mask lets
        # us push kv_t into bucket 0 via broadcast-add (1 kernel) rather than
        # torch.cat. Cascade events are rare enough per step that we branch.
        buckets = torch.zeros(L, B, H, K, K, dtype=torch.float32, device=device)
        bucket0_mask = torch.zeros(L, 1, 1, 1, 1, dtype=torch.float32, device=device)
        bucket0_mask[0] = 1.0  # constant indicator: "add to bucket 0 only"

        u = self.time_faaaa.view(1, H, 1, K).float() if not self.drop_u else None
        u_bh = u[:, :, 0] if u is not None else None                       # (1, H, K)

        out = torch.zeros(B, H, T, K, dtype=torch.float32, device=device)

        for t in range(T):
            k_t = k[:, :, t].float()                                        # (B, H, K)
            v_t = v[:, :, t].float()
            r_t = r[:, :, t].float()
            d_t = decay_step[:, :, t]                                       # (B, H, K)
            kv_t = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)                    # (B, H, K, K)

            # 1. Readout FIRST (from S_prev), single einsum over L buckets.
            per_ell = torch.einsum('bhi,lbhij->lbhj', r_t, buckets)         # (L, B, H, K)
            lam_t = lam[:, :, t]                                            # (B, H, L)
            # Weighted sum over ℓ: y_t_j = Σ_ℓ λ^(ℓ) · per_ell[ℓ]_j
            y_t = (lam_t.permute(2, 0, 1).unsqueeze(-1) * per_ell).sum(dim=0)  # (B, H, K)

            if u_bh is not None:
                bonus = (r_t * u_bh * k_t).sum(-1, keepdim=True) * v_t
                y_t = y_t + bonus

            out[:, :, t] = y_t

            # 2. Update: decay (1 broadcast multiply) + push kv into bucket 0
            #    (1 broadcast add via indicator mask — no torch.cat).
            buckets = buckets * d_t.unsqueeze(0).unsqueeze(-1)              # (L,B,H,K,K)
            buckets = buckets + bucket0_mask * kv_t.unsqueeze(0)

            # 3. Fenwick cascade at (t+1) boundaries. The highest ℓ firing at
            #    step (t+1) absorbs buckets 0..ℓ-1; those lower slots reset.
            #    Rare event (≤ 1 per step); we build the new buckets tensor
            #    only when it fires, using torch.cat only in that path.
            t_plus_1 = t + 1
            ell_max = 0
            for ell in range(1, L):
                period = 1 << ell
                if t_plus_1 < period:
                    break
                if t_plus_1 % period == 0:
                    ell_max = ell
            if ell_max > 0:
                absorb = buckets[:ell_max].sum(dim=0)                       # (B, H, K, K)
                new_top = buckets[ell_max] + absorb                         # receiving slot
                zeros_below = torch.zeros(
                    ell_max, B, H, K, K, dtype=torch.float32, device=device
                )
                rest = buckets[ell_max + 1:]
                buckets = torch.cat(
                    [zeros_below, new_top.unsqueeze(0), rest], dim=0
                )

        # Aggregate S = Σ_ℓ S^(ℓ) as the carry-state so the interface matches
        # vanilla _forward_recurrent. Kept for shape-compat only — the encoder
        # advertises supports_carry_state=False for loglinear, so this value
        # is not consumed by the chunked-eval harness.
        final_state = buckets.sum(dim=0).contiguous()
        return out.to(in_dtype), final_state

    def _m2rnn_branch(
        self,
        r: torch.Tensor,   # (B, H, T, K)
        k: torch.Tensor,
        v: torch.Tensor,
        x_in: torch.Tensor,  # (B, T, D) — feeds the forget-gate projection
    ) -> torch.Tensor:
        """Stage 10.2 — M²RNN parallel branch readout increment.

        Runs only when self.m2rnn_active is True (layer_id == m2rnn_layer).

        Paper-faithful recurrence (Mishra et al. arXiv:2603.14360):
            Z_t   = tanh(S_{t-1} · W + k_t ⊗ v_t)
            z     = W_f · x_t + β_h                        # (B, T, H)
            f_t   = (1 + exp(z))^{-α_h}                    # (B, T, H) ∈ (0,1]
            S_t   = f_t · S_{t-1} + (1 - f_t) · Z_t
            y_add = λ_h · r_t^T · S_t                      # (B, H, T, K)

        At init: λ_h = 0 ⇒ y_add ≡ 0 regardless of S, hence no effect on
        the host layer's output. W = I per head (irrelevant while λ=0).
        α_h = 1, β_h = 0, W_f = 0 ⇒ f_t = 0.5 everywhere at init (neutral
        forget). Exact zero-regression contract holds via λ_h = 0.
        """
        B, H, T, K = r.shape
        device = r.device

        # State per (B, H): matrix in R^{K×K}. fp32 for stability.
        S = torch.zeros(B, H, K, K, dtype=torch.float32, device=device)

        # Paper gate: f_t = (1 + exp(z))^{-α}, z = W_f x + β, with α > 0.
        # α = softplus(α_raw) + ε keeps α strictly positive so f ∈ (0, 1];
        # without this constraint SGD could drive α < 0 and push f above 1,
        # breaking the forget-gate semantics.
        z_pre = self.m2rnn_forget_proj(x_in) + self.m2rnn_forget_beta.view(1, 1, -1)  # (B, T, H)
        alpha_pos = F.softplus(self.m2rnn_forget_alpha_raw).view(1, 1, -1) + 1e-4
        # Stable compute: log f = -α · softplus(z), f = exp(log f) ∈ (0, 1].
        log_f = -alpha_pos * F.softplus(z_pre)
        f_probs = torch.exp(log_f).float()                                  # (B, T, H) ∈ (0, 1]

        # W per head: (H, K, K). Broadcast over batch in einsum.
        W_h = self.m2rnn_W.float()                                         # (H, K, K)

        out = torch.zeros(B, H, T, K, dtype=torch.float32, device=device)
        lam_h = self.m2rnn_lambda.view(1, H, 1).float()                    # (1, H, 1)

        for t in range(T):
            k_t = k[:, :, t].float()                                       # (B, H, K)
            v_t = v[:, :, t].float()
            kv_t = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)                   # (B, H, K, K)
            # SW: (B, H, K, K) = einsum over col axis (j): SW_{b,h,i,k} = Σ_j S_{b,h,i,j} W_{h,j,k}
            SW = torch.einsum('bhij,hjk->bhik', S, W_h)
            Z = torch.tanh(SW + kv_t)
            f_t = f_probs[:, t].unsqueeze(-1).unsqueeze(-1)                # (B, H, 1, 1)
            S = f_t * S + (1.0 - f_t) * Z

            r_t = r[:, :, t].float()
            y_t = torch.einsum('bhi,bhij->bhj', r_t, S)                    # (B, H, K)
            out[:, :, t] = lam_h * y_t

        return out.to(r.dtype)

    def _forward_recurrent_cayley_rank1_chunked(
        self,
        r: torch.Tensor,       # (B, H, T, K)
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,       # log-decay (negative); actual per-step decay = exp(w)
        x_in: torch.Tensor,    # (B, T, D) — drives the U/V LoRAs
        state: Optional[torch.Tensor] = None,
        *,
        chunk_size: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 10.5 (rank-1 fast path) — chunked affine scan.

        The Cayley recurrence
            S_t = D_t O_t S_{t-1} + k_t v_t^T
                = A_t S_{t-1} + U_t        where   A_t = D_t O_t,  U_t = k_t v_t^T
        is an affine scan associative under
            (A_b, U_b) ∘ (A_a, U_a) = (A_b A_a, A_b U_a + U_b).
        We reuse the existing ``_delta_affine_prefix_scan`` Hillis–Steele
        primitive to batch the within-chunk scan; chunks are processed
        sequentially with a carry state.  Build A_t directly from the
        rank-1 closed form (avoids materialising O_t):

            a = u·u, b = u·v, c = v·v,   Δ = 1 − b² + a·c
            O  = I − (2c/Δ) uu^T − (2(1−b)/Δ) uv^T + (2(1+b)/Δ) vu^T − (2a/Δ) vv^T
            A_t = D_t − (2c/Δ)(D_t u)u^T − (2(1−b)/Δ)(D_t u)v^T
                       + (2(1+b)/Δ)(D_t v)u^T − (2a/Δ)(D_t v)v^T

        Zero-regression at init (U=V=0 ⇒ A=0 ⇒ O=I) ⇒ A_t = D_t ⇒ vanilla
        RWKV-6 bit-exact.  Exact — not an approximation — modulo fp32
        accumulation noise from the prefix scan's different operation order.
        """
        B, H, T, K = r.shape
        device, in_dtype = r.device, r.dtype

        # ── Build U, V per step (rank-1 only here) ──
        U_flat = torch.tanh(x_in @ self.cayley_U_w1) @ self.cayley_U_w2     # (B, T, H·K)
        V_flat = torch.tanh(x_in @ self.cayley_V_w1) @ self.cayley_V_w2
        u_all = U_flat.view(B, T, H, K).permute(0, 2, 1, 3).contiguous().float()  # (B, H, T, K)
        v_all = V_flat.view(B, T, H, K).permute(0, 2, 1, 3).contiguous().float()
        u_all = u_all + self.cayley_U_base.view(1, H, 1, K).float()
        v_all = v_all + self.cayley_V_base.view(1, H, 1, K).float()

        r_f = r.float()
        k_f = k.float()
        v_f = v.float()
        w_f = w.float()
        decay_full = torch.exp(w_f)                                         # (B, H, T, K)

        # State init
        if state is None:
            S_carry = torch.zeros(B, H, K, K, dtype=torch.float32, device=device)
        else:
            S_carry = state.float()

        # Bonus scalar precomputed once for the full sequence: (B, H, T).
        # bonus_t = (r_t ⊙ u_faaaa ⊙ k_t).sum_channels
        if not self.drop_u:
            u_bonus = self.time_faaaa.view(1, H, 1, K).float()
            bonus_all = (r_f * u_bonus * k_f).sum(dim=-1)                   # (B, H, T)
        else:
            bonus_all = None

        out_parts = []
        cur = 0

        while cur < T:
            tc = min(chunk_size, T - cur)

            # Slice chunk-local tensors.
            rc = r_f[:, :, cur:cur + tc]                                    # (B, H, tc, K)
            kc = k_f[:, :, cur:cur + tc]
            vc = v_f[:, :, cur:cur + tc]
            dc = decay_full[:, :, cur:cur + tc]                             # (B, H, tc, K)
            uc = u_all[:, :, cur:cur + tc]                                  # (B, H, tc, K)
            vcc = v_all[:, :, cur:cur + tc]                                 # cayley "v" (NOT RWKV v)

            # Rank-1 invariants per step: (B, H, tc, 1).
            a = (uc  * uc ).sum(dim=-1, keepdim=True)
            b = (uc  * vcc).sum(dim=-1, keepdim=True)
            c = (vcc * vcc).sum(dim=-1, keepdim=True)
            delta = 1.0 - b * b + a * c

            coef_uu = -2.0 * c / delta                                      # (B, H, tc, 1)
            coef_uv = -2.0 * (1.0 - b) / delta
            coef_vu =  2.0 * (1.0 + b) / delta
            coef_vv = -2.0 * a / delta

            du = dc * uc                                                    # D_t ⊙ u  (B, H, tc, K)
            dv = dc * vcc                                                   # D_t ⊙ v

            # Build A_c = D_t O_t per step, shape (B, H, tc, K, K).
            # Chunk-local only — never materialise full-sequence A.
            A_c = torch.diag_embed(dc)                                      # (B, H, tc, K, K)
            A_c = A_c + coef_uu.unsqueeze(-1) * (du.unsqueeze(-1)  * uc.unsqueeze(-2))
            A_c = A_c + coef_uv.unsqueeze(-1) * (du.unsqueeze(-1)  * vcc.unsqueeze(-2))
            A_c = A_c + coef_vu.unsqueeze(-1) * (dv.unsqueeze(-1)  * uc.unsqueeze(-2))
            A_c = A_c + coef_vv.unsqueeze(-1) * (dv.unsqueeze(-1)  * vcc.unsqueeze(-2))

            # U_c = k_t v_t^T per step, shape (B, H, tc, K, K).
            U_c = kc.unsqueeze(-1) * vc.unsqueeze(-2)

            # Existing exact affine prefix helper — operates on (A, U) pairs.
            A_p, U_p = _delta_affine_prefix_scan(A_c, U_c)                  # (B, H, tc, K, K)

            # Recover in-chunk states: S[t] = A_p[t] · S_carry + U_p[t].
            AS = torch.einsum('bhtij,bhjv->bhtiv', A_p, S_carry)
            S_chunk = AS + U_p                                              # (B, H, tc, K, K)

            # Readout uses S_prev (shifted): first entry from carry, rest from S_chunk.
            S_prev = torch.cat(
                [S_carry.unsqueeze(2), S_chunk[:, :, :-1]], dim=2
            )                                                               # (B, H, tc, K, K)
            y_state = torch.einsum('bhtk,bhtkv->bhtv', rc, S_prev)          # (B, H, tc, K)

            if bonus_all is not None:
                bonus = bonus_all[:, :, cur:cur + tc].unsqueeze(-1) * vc     # (B, H, tc, K)
                y_chunk = y_state + bonus
            else:
                y_chunk = y_state

            out_parts.append(y_chunk)
            S_carry = S_chunk[:, :, -1]
            cur += tc

        out = torch.cat(out_parts, dim=2)                                   # (B, H, T, K)
        return out.to(in_dtype), S_carry.contiguous()

    def _forward_recurrent_cayley(
        self,
        r: torch.Tensor,       # (B, H, T, K)
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,       # log-decay (negative); actual per-step decay = exp(w)
        x_in: torch.Tensor,    # (B, T, D) — drives the U/V LoRAs
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 10.5 — Cayley-orthogonal transition.

        G_t = exp(-λ_t) · O_t where
            O_t = (I - A_t)(I + A_t)^{-1},
            A_t = Σ_r (U_t^(r) V_t^(r)^T − V_t^(r) U_t^(r)^T)  (skew, rank ≤ 2·cayley_rank).
        State update:
            S_t = diag(exp(w_t)) · O_t · S_{t-1} + k_t v_t^T.

        Readout uses S_prev (vanilla convention):
            y_t = r_t^T S_{t-1} + bonus_t.

        U=V=0 at init (zero-init LoRA down-projection + zero-init base) ⇒ A=0
        ⇒ O=I ⇒ scan reduces to vanilla RWKV-6 bit-exact.

        Implementation strategy:

        Generic rank-R uses the **Woodbury identity** with direct closed-form
        ``O = I − 2 M₁ (I + M₂ᵀ M₁)⁻¹ M₂ᵀ`` (saves one A-pass vs the naive
        ``(I − A)(I + A)⁻¹ S`` evaluation). Inner 2R×2R solve is the only
        non-batched matrix op per step.

        **Rank-1 fast path** (`cayley_rank == 1`) specialises further with
        scalar invariants ``a = u·u, b = u·v, c = v·v`` and rank-1 outer
        corrections:
            p = uᵀ S,   q = vᵀ S
            Δ = 1 − b² + a·c
            α = ((1 − b) q + c p) / Δ
            β = (−a q + (1 + b) p) / Δ
            O·S = S − 2 u⊗α + 2 v⊗β
        Drops kernel count from ~15 to ~8 per step and avoids any inner
        solve; activations saved per step are pure K-vectors, not K×K.

        Sequential Python loop over T (same structure as 10.1 sequential).
        A Triton fused-scan port is the next optimisation lever if the
        rank-1 path is still too slow after `torch.compile`.
        """
        B, H, T, K = r.shape
        R = self.cayley_rank
        device, in_dtype = r.device, r.dtype

        # Compute U, V tensors of shape (R, B, H, T, K) from the xw stream.
        U_flat = torch.tanh(x_in @ self.cayley_U_w1) @ self.cayley_U_w2     # (B, T, R·H·K)
        V_flat = torch.tanh(x_in @ self.cayley_V_w1) @ self.cayley_V_w2
        U = U_flat.view(B, T, R, H, K).permute(2, 0, 3, 1, 4).contiguous()  # (R, B, H, T, K)
        V = V_flat.view(B, T, R, H, K).permute(2, 0, 3, 1, 4).contiguous()
        U = U + self.cayley_U_base.view(R, 1, H, 1, K)
        V = V + self.cayley_V_base.view(R, 1, H, 1, K)

        # State init
        if state is None:
            S = torch.zeros(B, H, K, K, dtype=torch.float32, device=device)
        else:
            S = state.float()

        u_bhTK = self.time_faaaa.view(1, H, 1, K).float() if not self.drop_u else None
        decay_step = torch.exp(w.float())                                   # (B, H, T, K)

        # Pre-cast & permute time-major for efficient per-step slicing.
        r_f = r.float()
        k_f = k.float()
        v_f = v.float()

        out = torch.zeros(B, H, T, K, dtype=torch.float32, device=device)

        # ── Rank-1 fast path ───────────────────────────────────────────
        if R == 1:
            U1 = U[0]                                                       # (B, H, T, K)
            V1 = V[0]

            for t in range(T):
                # Readout from S_prev (vanilla convention) + bonus.
                r_t = r_f[:, :, t]                                          # (B, H, K)
                k_t = k_f[:, :, t]
                v_t = v_f[:, :, t]
                y_t = torch.einsum('bhi,bhij->bhj', r_t, S)
                if u_bhTK is not None:
                    bonus = (r_t * u_bhTK[:, :, 0] * k_t).sum(-1, keepdim=True) * v_t
                    y_t = y_t + bonus
                out[:, :, t] = y_t

                # Rank-1 Cayley vectors for this step.
                u_t  = U1[:, :, t]                                          # (B, H, K)
                vc_t = V1[:, :, t]                                          # (B, H, K)  (Cayley "v"; NOT the RWKV v)

                # Scalar invariants (B, H, 1).
                a = (u_t  * u_t ).sum(-1, keepdim=True)
                b = (u_t  * vc_t).sum(-1, keepdim=True)
                c = (vc_t * vc_t).sum(-1, keepdim=True)
                delta = 1.0 - b * b + a * c                                 # (B, H, 1)

                # Row projections (B, H, K): p = uᵀ S,  q = vᵀ S.
                p = torch.einsum('bhi,bhij->bhj', u_t,  S)
                q = torch.einsum('bhi,bhij->bhj', vc_t, S)

                # Correction factors (B, H, K).
                alpha = ((1.0 - b) * q + c * p) / delta
                beta  = (-a * q + (1.0 + b) * p) / delta

                # O·S = S − 2 u⊗α + 2 v⊗β (both rank-1 outer products).
                OS = (
                    S
                    - 2.0 * u_t.unsqueeze(-1)  * alpha.unsqueeze(-2)
                    + 2.0 * vc_t.unsqueeze(-1) * beta.unsqueeze(-2)
                )

                # State update: S_t = diag(exp(w_t)) · OS + k_t v_t^T.
                d_t  = decay_step[:, :, t]                                  # (B, H, K)
                kv_t = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)                # (B, H, K, K)
                S = d_t.unsqueeze(-1) * OS + kv_t

            return out.to(in_dtype), S.contiguous()

        # ── Generic rank-R Woodbury path (cayley_rank > 1) ─────────────
        two_R = 2 * R
        I_inner = torch.eye(two_R, dtype=torch.float32, device=device).view(1, 1, two_R, two_R)

        for t in range(T):
            # Readout from S_prev (vanilla convention) + bonus.
            r_t = r_f[:, :, t]                                              # (B, H, K)
            k_t = k_f[:, :, t]
            v_t = v_f[:, :, t]
            y_t = torch.einsum('bhi,bhij->bhj', r_t, S)
            if u_bhTK is not None:
                bonus = (r_t * u_bhTK[:, :, 0] * k_t).sum(-1, keepdim=True) * v_t
                y_t = y_t + bonus
            out[:, :, t] = y_t

            U_t = U[:, :, :, t]                                             # (R, B, H, K)
            V_t = V[:, :, :, t]
            M1 = torch.stack([U_t, -V_t], dim=-1).permute(1, 2, 3, 0, 4).reshape(B, H, K, two_R)
            M2 = torch.stack([V_t,  U_t], dim=-1).permute(1, 2, 3, 0, 4).reshape(B, H, K, two_R)

            T_inner = torch.einsum('bhki,bhkj->bhij', M2, M1)
            IpT_inv = torch.linalg.inv(I_inner + T_inner)

            # Direct O·S = S − 2 M1 (I + M2ᵀ M1)⁻¹ M2ᵀ S  (saves one A-pass).
            M2T_S = torch.einsum('bhki,bhkj->bhij', M2, S)                  # (B, H, 2R, K)
            inner = torch.einsum('bhij,bhjk->bhik', IpT_inv, M2T_S)         # (B, H, 2R, K)
            corr  = torch.einsum('bhki,bhij->bhkj', M1, inner)              # (B, H, K, K)
            OS = S - 2.0 * corr

            d_t  = decay_step[:, :, t]
            kv_t = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
            S = d_t.unsqueeze(-1) * OS + kv_t

        return out.to(in_dtype), S.contiguous()

    def _paper_n2_branch(
        self,
        r: torch.Tensor,   # (B, H, T, K)
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,   # log-decay (negative), (B, H, T, K)
        B: int, H: int, T: int, K: int,
        beta_tok: Optional[torch.Tensor] = None,  # R2: data-dep β, (B, H, T)
    ) -> torch.Tensor:
        """Adds the second-order Taylor term of softmax attention.

        Two variants, controlled by flags set at construction time:
          use_hadamard_n2: diagonal lift (k⊙k, r⊙r). State dim = K per head,
                           same shape as the linear branch. Cheap control.
          use_qtail:       Kronecker lift (k⊗k, r⊗r). State dim = K² per head.
                           Expensive but implements the paper's Eq. 3/7 exactly.

        Both are gated by a zero-init per-head β so the branch is inert at
        step 0 (same zero-regression contract as viscosity / P²-RSE).  The
        scan reuses `_chunked_wkv` with no new kernel. Bonus `u` is set to
        zero for the n=2 branch since the paper's Taylor derivation has no
        current-step boost analog for n≥2.
        """
        device, in_dtype = r.device, r.dtype
        y_total = torch.zeros_like(r)

        if self.use_hadamard_n2:
            # r⊙r, k⊙k — element-wise, diagonal of the full Kronecker lift.
            # Decay on the n=2 state is the diagonal of w⊗w, i.e. 2·w (log-space).
            r2 = r * r
            k2 = k * k
            w2_log = 2.0 * w
            u2 = torch.zeros(1, H, 1, K, dtype=in_dtype, device=device)
            s2_init = torch.zeros(B, H, K, K, dtype=torch.float32, device=device)
            y2, _ = _chunked_wkv(r2, k2, v, w2_log, u2, s2_init)
            beta_h = self.beta_hadamard.view(1, H, 1, 1).to(in_dtype)
            y_total = y_total + beta_h * y2.to(in_dtype)

        if self.use_qtail:
            # Full Kronecker (default): r⊗r, k⊗k ∈ R^{K²}.
            # Low-rank variant (use_qtail_lowrank=True): project r, k to
            # K'=qtail_lr_rank per head first, then full Kronecker on K'².
            if self.use_qtail_lowrank:
                Kp = self.qtail_lr_rank
                K2 = Kp * Kp
                # Per-head projection: (B,H,T,K) × (H,K,K') → (B,H,T,K')
                r_lr = torch.einsum('bhtk,hkp->bhtp', r, self.qtail_lr_proj_r)
                k_lr = torch.einsum('bhtk,hkp->bhtp', k, self.qtail_lr_proj_k)
                r_kron = (r_lr.unsqueeze(-1) * r_lr.unsqueeze(-2)).reshape(B, H, T, K2)
                k_kron = (k_lr.unsqueeze(-1) * k_lr.unsqueeze(-2)).reshape(B, H, T, K2)
                # Decay: scalar per-head (mean of w across channels),
                # doubled for Kronecker (matches the average of w_i + w_j
                # across channel pairs). Safe, preserves w ≤ 0 constraint.
                w_mean = w.mean(dim=-1, keepdim=True)                    # (B,H,T,1)
                w_pair_effective = (2.0 * w_mean).expand(B, H, T, K2)    # (B,H,T,K²)
                # γ coupling (if active) scales the decay strength
                if self.use_qtail_gamma:
                    gamma = self.qtail_gamma.view(1, H, 1, 1).to(w_pair_effective.dtype)
                    w_pair_effective = gamma * w_pair_effective
                w_kron = w_pair_effective
            else:
                # Full Kronecker: r⊗r, k⊗k ∈ R^{K²}. Decay at lifted index
                # (i,j) = w_i + w_j.  Zero bonus.  State shape: (B, H, K², K).
                # Memory cost per layer at (B=8,H=4,K=64,T=500,fp32):
                #   r,k lifted:  8·4·500·4096·4 ≈ 260 MB each
                #   state:       8·4·4096·64·4  ≈  32 MB
                # Enabled only at the top 2 of 6 layers for cost containment.
                K2 = K * K
                r_out = r.unsqueeze(-1) * r.unsqueeze(-2)              # (B,H,T,K,K)
                r_kron = r_out.reshape(B, H, T, K2)
                k_out = k.unsqueeze(-1) * k.unsqueeze(-2)
                k_kron = k_out.reshape(B, H, T, K2)
                # Decay at lifted pair (i,j).  Base: w_i + w_j (product of decays).
                # If qtail_gamma is active, scale by learnable per-head γ.
                w_pair = w.unsqueeze(-1) + w.unsqueeze(-2)              # (B,H,T,K,K)
                if self.use_qtail_gamma:
                    gamma = self.qtail_gamma.view(1, H, 1, 1, 1).to(w_pair.dtype)
                    w_pair = gamma * w_pair
                w_kron = w_pair.reshape(B, H, T, K2)
            # H1 — per-(head, lifted-pair) β multiplies k_kron element-wise.
            # Init β_pp = 1.0 makes this a no-op at t=0; outer β_qtail (zero-
            # init) still gates the branch off, so zero-regression holds.
            # Once SGD grows β_qtail, β_pp gets gradient and starts allocating
            # the lift to specific channel-pairs (the H1 hypothesis).
            if self.use_qtail_beta_per_pair:
                beta_pp = self.beta_qtail_per_pair.view(1, H, 1, K2).to(in_dtype)
                k_kron = k_kron * beta_pp
            u_kron = torch.zeros(1, H, 1, K2, dtype=in_dtype, device=device)
            s_kron_init = torch.zeros(B, H, K2, K, dtype=torch.float32, device=device)
            y_q, _ = _chunked_wkv(r_kron, k_kron, v, w_kron, u_kron, s_kron_init)
            # Effective β: static per-head scalar (+ data-dep per-token if R2 active)
            # R2: beta_tok is (B, H, T); reshape to (B, H, T, 1) for broadcast.
            # Both components zero-init ⇒ effective β = 0 at t=0 (zero-regression).
            beta_q = self.beta_qtail.view(1, H, 1, 1).to(in_dtype)
            if self.use_qtail_dbeta and beta_tok is not None:
                beta_q = beta_q + beta_tok.unsqueeze(-1).to(in_dtype)
            y_total = y_total + beta_q * y_q.to(in_dtype)

        return y_total

    def _forward_recurrent_rse(
        self,
        r: torch.Tensor,      # (B, H, T, K)
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,      # log-decay (negative), per-channel (K)
        theta: torch.Tensor,  # (B, H, T, Bk) — rotation angle per 2x2 block
        state: Optional[torch.Tensor] = None,
        apply_bonus: bool = True,   # P²-RSE calls this twice and adds u externally
        phi: Optional[torch.Tensor] = None,  # Stage-7A A1′: readout phase (B,H,T,Bk)
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

            # Readout: y[t,c] = Re( sum_b exp(-iφ_{t,b}) · conj(r_c[t,b]) · S_total[t,b,c] )
            # Stage 7A (A1′): learnable per-(token, head, block) phase φ
            # rotates the readout to recover quadrature content that is
            # otherwise discarded by `.real`.  φ=None reproduces the anchor.
            if phi is not None:
                phi_c = phi[:, :, cur:cur + tc]                        # (B,H,tc,Bk)
                # exp(-iφ), as complex tensor
                rot = torch.polar(
                    torch.ones_like(phi_c.float()), -phi_c.float()
                )                                                      # (B,H,tc,Bk) complex64
                r_contract = r_c_c.conj() * rot                        # (B,H,tc,Bk)
            else:
                r_contract = r_c_c.conj()
            y_chunk = torch.einsum(
                'bhtk,bhtkc->bhtc', r_contract, S_total
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

    def _forward_recurrent_p2rse_indeplam(
        self,
        r: torch.Tensor,       # (B, H, T, K)
        k: torch.Tensor,
        v: torch.Tensor,
        w_1: torch.Tensor,     # (B, H, T, K)  log-decay for pole 1
        w_2: torch.Tensor,     # (B, H, T, K)  log-decay for pole 2 (independent)
        theta_1: torch.Tensor, # (B, H, T, Bk) rotation for pole 1
        theta_2: torch.Tensor, # (B, H, T, Bk) rotation for pole 2
        x_in: torch.Tensor,    # (B, T, D)  for β mixer
        state: Optional[torch.Tensor] = None,
        k_h_2: Optional[torch.Tensor] = None,  # (B, H, T, K) Phase 2b-ext pole-2 key
        v_h_2: Optional[torch.Tensor] = None,  # (B, H, T, K) Phase 2b-ext pole-2 value
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Phase 2b — Independent-λ P²-RSE scan.

        Differences from ``_forward_recurrent_p2rse`` (the shared-λ Phase-1
        variant):
          * Accepts separate ``w_1`` and ``w_2`` — each pole has its own
            per-channel log-decay tensor derived from an independent LoRA.
          * Routes through ``p2rse_indep_lambda_scan`` which uses the
            real-arithmetic ``rse_viscosity_scan`` kernel for both poles —
            avoids the complex64 Tensor-Core fallback (~2.5–3× faster).
          * Viscosity coupling η is shared across both poles, preserving
            the Phase-3 zero-η-at-init contract.

        Phase 2b-ext: if ``k_h_2`` and ``v_h_2`` are provided, pole 2 uses
        its own drive-side (k, v) projections (shared + LoRA delta).

        Zero-regression-at-init: at t=0, ``w_2 == w_1`` (pole-2 decay LoRA
        zero) AND ``k_h_2 == k, v_h_2 == v`` (pole-2 drive LoRA zero), so
        output matches Phase-1 P²-RSE exactly until SGD moves the LoRAs.
        """
        from src.models.p2rse_indep_lambda import p2rse_indep_lambda_scan

        B, H, T, K = r.shape

        # β mixer logits — matches Phase-1 shape and convention
        beta_logits = self.beta_mixer(x_in).view(B, T, H, 2).transpose(1, 2)

        # Viscosity coupling — shared across poles
        eta = self.viscosity_eta if self.use_viscosity else None

        # Bonus u — applied once inside the helper (after mixing)
        u = None if self.drop_u else self.time_faaaa.view(H, K)

        y, new_state = p2rse_indep_lambda_scan(
            r=r, k=k, v=v,
            w_1=w_1, w_2=w_2,
            theta_1=theta_1, theta_2=theta_2,
            beta_logits=beta_logits,
            mixer_type=self.p2rse_mixer,
            eta=eta,
            u=u,
            state=state,
            k_2=k_h_2,
            v_2=v_h_2,
        )
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


def _recurrent_nonnormal_rse_scan(
    r: torch.Tensor,       # (B, H, T, K)
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,       # log-decay (negative), (B, H, T, K)
    theta: torch.Tensor,   # (B, H, T, Bk) rotation angle per 2x2 block
    rho: torch.Tensor,     # (B, H, T, Bk) anisotropy magnitude per block
    psi: torch.Tensor,     # (B, H, T, Bk) anisotropy axis per block
    u: torch.Tensor,       # (1, H, 1, K) bonus per-channel (real)
    state: Optional[torch.Tensor] = None,
    apply_bonus: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stage-8 T2 — Sequential scan for non-normal polar-RSE transition.

    Per-block 2×2 transition (polar form):
        G_{t,b} = e^{-λ} · R(ψ)^T · diag(e^ρ, e^{-ρ}) · R(ψ) · R(θ)
                = e^{-λ} · P(ρ, ψ) · R(θ)
    where P(ρ, ψ) is symmetric positive-definite with eigenvalues e^{±ρ}
    along rotated axes. ρ = 0 ⇒ P = I ⇒ G = e^{-λ} R(θ) = exact RSE.

    State convention (per block): (2, K_v) — each block carries a
    2-vector × K_v matrix.  Update:
        S_{t,b} = G_{t,b} · S_{t-1,b} + k_{t,b} v_t^T
    where k_{t,b} = (k_{t, 2b}, k_{t, 2b+1}) is the 2-vector slice of
    the key at block b, and v_t is shared across blocks.

    Readout (per token):
        y_t[:,c] = Σ_b r_{t,b}^T · S_{t,b, :, c]
    where r_{t,b} = (r_{t, 2b}, r_{t, 2b+1}).

    Bonus (current-step real shortcut, preserving existing convention):
        y_t[:,c] += (r_t ⊙ u)^T k_t · v_t[c]
    This is the same scalar bonus as _chunked_wkv, kept identical.

    Sequential within the full sequence (Option A in STAGE8_PLAN §4.5):
    each step is O(1) GPU work; T sequential steps total.
    Zero-regression contract: ρ ≡ 0 ⇒ P = I ⇒ G = e^{-λ} R(θ) = exact RSE,
    and the state evolves identically to the RSE scan (modulo FP order).
    """
    B, H, T, K = r.shape
    Bk = K // 2
    device = r.device
    dtype = r.dtype

    # Per-block log-decay (average of channel pairs) — identical to
    # _forward_recurrent_rse convention.
    log_decay_block = w.view(B, H, T, Bk, 2).float().mean(dim=-1)       # (B, H, T, Bk)

    # Analytic per-block 2x2 transition (no matrix_exp).
    # P entries (symmetric 2x2):
    #   P00 = cosh(ρ) + sinh(ρ) cos(2ψ)
    #   P01 = P10 = sinh(ρ) sin(2ψ)
    #   P11 = cosh(ρ) - sinh(ρ) cos(2ψ)
    # G = exp(-λ) · P · R(θ)
    #   R(θ) = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
    cos_t = torch.cos(theta.float())                                    # (B, H, T, Bk)
    sin_t = torch.sin(theta.float())
    cosh_r = torch.cosh(rho.float())
    sinh_r = torch.sinh(rho.float())
    cos_2p = torch.cos(2.0 * psi.float())
    sin_2p = torch.sin(2.0 * psi.float())

    P00 = cosh_r + sinh_r * cos_2p
    P01 = sinh_r * sin_2p        # = P10
    P11 = cosh_r - sinh_r * cos_2p

    # G = exp(-λ) · [P · R(θ)]
    #   [P · R(θ)]_{00} = P00 cos - P01 sin ⟵ actually need (P·R)_ij
    # Let's do it row-by-row. P · R(θ):
    #   row 0:  [P00 cos + P01 sin,  -P00 sin + P01 cos]
    #   row 1:  [P01 cos + P11 sin,  -P01 sin + P11 cos]
    decay = torch.exp(log_decay_block)                                  # (B, H, T, Bk)
    G00 = decay * (P00 * cos_t + P01 * sin_t)
    G01 = decay * (-P00 * sin_t + P01 * cos_t)
    G10 = decay * (P01 * cos_t + P11 * sin_t)
    G11 = decay * (-P01 * sin_t + P11 * cos_t)

    # ── Chunked affine associative scan (Stage-8 optimization) ──────
    # The pair (G, U) with U_t = k_t v_t^T composes associatively:
    #   (G_b, U_b) ⊗ (G_a, U_a) = (G_b G_a, G_b U_a + U_b)
    # So the within-chunk prefix is computable via Hillis–Steele doubling
    # in O(log T_c) parallel passes.  Inter-chunk we carry the chunk-end
    # state sequentially.  This recovers the cumlog-style parallelism of
    # the current RSE scan, extended to the non-commutative 2×2 case.

    chunk_size = 64

    # Cast inputs once
    k_f = k.float()
    r_f = r.float()
    v_f = v.float()
    r_2 = r_f.view(B, H, T, Bk, 2)                                      # (B,H,T,Bk,2)

    # Stack G into (B, H, T, Bk, 2, 2) then permute block dim in
    G_stack = torch.stack([
        torch.stack([G00, G01], dim=-1),
        torch.stack([G10, G11], dim=-1),
    ], dim=-2)                                                           # (B,H,T,Bk,2,2)
    # Rearrange to (B, H, Bk, T, 2, 2) so the T-axis is adjacent to matrix dims
    G_full = G_stack.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Build U_t = k_{t,b} v_t^T per block: shape (B, H, Bk, T, 2, K)
    k_2 = k_f.view(B, H, T, Bk, 2)                                       # (B,H,T,Bk,2)
    # (B,H,T,Bk,2) × (B,H,T,1,1,K) → (B,H,T,Bk,2,K), then permute block in
    U_full = (k_2.unsqueeze(-1) * v_f.view(B, H, T, 1, 1, K))            # (B,H,T,Bk,2,K)
    U_full = U_full.permute(0, 1, 3, 2, 4, 5).contiguous()               # (B,H,Bk,T,2,K)

    # Bonus precomputation (matches the sequential reference exactly)
    use_bonus_here = apply_bonus and (u is not None)
    if use_bonus_here:
        u_k = u.float().view(1, H, 1, K).expand(B, H, T, K)
        bonus_scalar_all = (r_f * u_k * k_f).sum(dim=-1)                 # (B,H,T)

    # Carry state — packed (B, H, Bk, 2, K).
    if state is None:
        S_carry = torch.zeros(B, H, Bk, 2, K, device=device, dtype=torch.float32)
    else:
        S_carry = state.float().view(B, H, Bk, 2, K)

    out_chunks: List[torch.Tensor] = []
    cur = 0
    while cur < T:
        tc = min(chunk_size, T - cur)
        G_c = G_full[:, :, :, cur:cur + tc].contiguous()                 # (B,H,Bk,tc,2,2)
        U_c = U_full[:, :, :, cur:cur + tc].contiguous()                 # (B,H,Bk,tc,2,K)

        # Prefix scan:  G_p[t] = G_c[t] ⊗ G_c[t-1] ⊗ … ⊗ G_c[0]
        #               U_p[t] = Σ_s (G_c[t]…G_c[s+1]) U_c[s]
        G_p, U_p = _affine_prefix_scan(G_c, U_c)

        # State inside chunk:  S_t = G_p[t] · S_carry + U_p[t]
        # G_p:(B,H,Bk,tc,2,2) × S_carry:(B,H,Bk,2,K) → (B,H,Bk,tc,2,K)
        GS = torch.einsum('bhptij,bhpjk->bhptik', G_p, S_carry)
        S_chunk = GS + U_p                                               # (B,H,Bk,tc,2,K)

        # Readout per token: y[t,c] = Σ_b (r_b[0] S[b,0,c] + r_b[1] S[b,1,c])
        # Rearrange to (B, H, tc, Bk, 2, K) for the readout contraction
        S_btc = S_chunk.permute(0, 1, 3, 2, 4, 5).contiguous()           # (B,H,tc,Bk,2,K)
        r_c = r_2[:, :, cur:cur + tc]                                    # (B,H,tc,Bk,2)
        y_chunk = (r_c.unsqueeze(-1) * S_btc).sum(dim=(3, 4))            # (B,H,tc,K)

        if use_bonus_here:
            v_c = v_f[:, :, cur:cur + tc]
            y_chunk = y_chunk + bonus_scalar_all[:, :, cur:cur + tc].unsqueeze(-1) * v_c

        out_chunks.append(y_chunk)

        # Inter-chunk carry: final state is at the last position in the chunk
        S_carry = S_btc[:, :, -1]                                        # (B,H,Bk,2,K)

        cur += tc

    out = torch.cat(out_chunks, dim=2)                                   # (B,H,T,K)

    # Pack final state back to (B, H, K, K) real — match RSE convention
    final_state = torch.zeros(B, H, K, K, device=device, dtype=torch.float32)
    final_view = final_state.view(B, H, Bk, 2, K)
    final_view[..., 0, :] = S_carry[..., 0, :]
    final_view[..., 1, :] = S_carry[..., 1, :]

    return out.to(dtype), final_state


def _affine_prefix_scan(
    G: torch.Tensor,   # (B, H, Bk, T_c, 2, 2)
    U: torch.Tensor,   # (B, H, Bk, T_c, 2, K)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hillis–Steele parallel prefix scan over affine pairs (G, U) with
    composition  (G_b, U_b) ⊗ (G_a, U_a) = (G_b G_a, G_b U_a + U_b).

    Returns (G_prefix, U_prefix) of the same shapes, where
        G_prefix[t] = G[t] ⊗ G[t-1] ⊗ … ⊗ G[0]
        U_prefix[t] = U[t] + G[t] U[t-1] + G[t] G[t-1] U[t-2] + …

    Uses ⌈log₂ T_c⌉ doubling passes; each pass is a batched einsum + where.
    For T_c=64: 6 passes.  Exact (not truncated) — associativity of the
    affine composition makes this equivalent to the sequential recurrence.
    """
    T_c = G.shape[3]
    t_idx = torch.arange(T_c, device=G.device)

    d = 1
    while d < T_c:
        # Shift G and U by d steps along the time axis.
        # Positions [0, d) receive zero; position t receives value at t-d.
        # F.pad with (…, d, 0) pads the T-dim (third-from-last for G, same
        # for U) at the FRONT with d entries of zero.
        G_shift = F.pad(G[:, :, :, : T_c - d], (0, 0, 0, 0, d, 0))
        U_shift = F.pad(U[:, :, :, : T_c - d], (0, 0, 0, 0, d, 0))

        # Combine: at position t, if t ≥ d we have a "previous" value.
        #   G_new[t] = G[t] · G_shift[t]      (= G[t] · G[t-d])
        #   U_new[t] = G[t] · U_shift[t] + U[t]  (= G[t] · U[t-d] + U[t])
        G_cand = torch.einsum('bhptij,bhptjk->bhptik', G, G_shift)
        U_cand = torch.einsum('bhptij,bhptjk->bhptik', G, U_shift) + U

        # For t < d, keep original (G_shift / U_shift were zero, so G_cand
        # would be zero not G[t]; must mask back).
        mask = (t_idx >= d).view(1, 1, 1, T_c, 1, 1)
        G = torch.where(mask, G_cand, G)
        U = torch.where(mask, U_cand, U)

        d *= 2

    return G, U


def _recurrent_delta_scan(
    r: torch.Tensor,      # (B, H, T, K)
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,      # log-decay (negative), (B, H, T, K)
    u: torch.Tensor,      # (1, H, 1, K) bonus term
    kk: torch.Tensor,     # normalized key for erase direction, (B, H, T, K)
    iclr: torch.Tensor,   # per-channel erasure strength, (B, H, T, K)
    gate: torch.Tensor,   # per-head hard gate, shape (H,), zero-init
    wkv_state: torch.Tensor,  # (B, H, K, K)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sequential RWKV-6 WKV scan with rank-1 delta erase — Stage-8 T1.

    Per-step update (state convention: S (K, V), rows=key, cols=value):
        kv_t    = k_t v_t^T                                        # (K, V)
        y_t     = r_t^T (S_{t-1} + u ⊙ kv_t)                        # readout + bonus
        S_dec   = diag(exp(w_t)) · S_{t-1}                         # per-row decay
        β_t     = gate_h · iclr_t  ∈ [0, 2·gate]                   # effective erase
        S_erase = S_dec - β_t (kk_t · kk_t^T) · S_dec              # rank-1 erase
                = S_dec - (β_t ⊙ kk_t) (kk_t^T · S_dec)
        S_t     = S_erase + kv_t                                   # write

    Zero-regression contract: when `gate ≡ 0`, β_t ≡ 0 ⇒ S_erase = S_dec,
    and the update reduces bit-exactly to vanilla RWKV-6 recurrent WKV.
    """
    B, H, T, K = r.shape
    V = v.size(-1)
    device = r.device
    dtype = r.dtype

    if wkv_state is None:
        S_carry = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)
    else:
        S_carry = wkv_state.float()

    # ── Chunked affine associative scan (Stage-8 delta optimization) ──
    # The pair (A_t, U_t) with
    #   A_t = (I - (β_t ⊙ kk_t) kk_t^T) · diag(w_t)
    #   U_t = k_t v_t^T
    # composes associatively: (A_b, U_b) ⊗ (A_a, U_a) = (A_b A_a, A_b U_a + U_b).
    # Hillis–Steele prefix scan within chunks, serial state carry across.

    chunk_size = 64

    r_f = r.float()
    k_f = k.float()
    v_f = v.float()
    w_decay = torch.exp(w.float())                           # (B, H, T, K)
    kk_f = kk.float()
    gate_hk = gate.view(1, H, 1, 1).float()
    beta_eff = gate_hk * iclr.float()                        # (B, H, T, K)

    # Precompute per-token A_t as K×K.  We have
    #   A_t[i, j] = δ_{ij} · w_t[j] − (β_t[i]·kk_t[i]) · (kk_t[j]·w_t[j])
    # Memory: (B, H, T, K, K).
    # NOTE: we build A implicitly without storing a K×K diagonal, by
    # A_t_diag_part = diag_embed(w_t), rank-1 = (β⊙kk)_i · (kk⊙w)_j.
    # Allowed: K = 64, so K×K per token = 4096 floats. Tc=64 chunk ⇒
    # ~40 MB per layer — fine.
    diag_w = torch.diag_embed(w_decay)                       # (B, H, T, K, K)
    rank1_left = (beta_eff * kk_f).unsqueeze(-1)             # (B, H, T, K, 1)
    rank1_right = (kk_f * w_decay).unsqueeze(-2)             # (B, H, T, 1, K)
    A_full = diag_w - rank1_left * rank1_right               # (B, H, T, K, K)

    # U_t = k_t v_t^T
    U_full = k_f.unsqueeze(-1) * v_f.unsqueeze(-2)           # (B, H, T, K, V)

    # Bonus precomputation (preserved current-step shortcut)
    u_bcast = u.float().view(1, H, 1, K).expand(B, H, T, K)
    bonus_scalar_all = (r_f * u_bcast * k_f).sum(dim=-1)     # (B, H, T)

    out_chunks: List[torch.Tensor] = []
    cur = 0
    while cur < T:
        tc = min(chunk_size, T - cur)
        A_c = A_full[:, :, cur:cur + tc].contiguous()        # (B, H, tc, K, K)
        U_c = U_full[:, :, cur:cur + tc].contiguous()        # (B, H, tc, K, V)

        # Prefix scan
        A_p, U_p = _delta_affine_prefix_scan(A_c, U_c)

        # State inside chunk: S_t = A_p[t] · S_carry + U_p[t]
        AS = torch.einsum('bhtij,bhjv->bhtiv', A_p, S_carry)  # (B, H, tc, K, V)
        S_chunk = AS + U_p                                    # (B, H, tc, K, V)

        # Readout (includes bonus): y_t = r_t · (S_{t-1} + u ⊙ kv_t)
        # Note: the sequential reference reads r_t @ S_{t-1} PLUS bonus.
        # Since S_chunk[t] already includes A_p[t]·S_carry + U_p[t], and
        # S_{t-1} for a delta sequential is "S AFTER t-1 update but BEFORE
        # t decay-erase-write" — same as S_chunk[t-1] for t≥1, or S_carry
        # for t=0.  We therefore build a shifted S:
        #   S_prev[t] = S_chunk[t-1] for t≥1 ; S_prev[0] = S_carry
        S_prev = torch.cat([
            S_carry.unsqueeze(2),                            # (B, H, 1, K, V)
            S_chunk[:, :, :-1],                              # (B, H, tc-1, K, V)
        ], dim=2)                                            # (B, H, tc, K, V)
        r_c = r_f[:, :, cur:cur + tc]                        # (B, H, tc, K)
        y_state = torch.einsum('bhtk,bhtkv->bhtv', r_c, S_prev)
        v_c = v_f[:, :, cur:cur + tc]
        y_bonus = bonus_scalar_all[:, :, cur:cur + tc].unsqueeze(-1) * v_c
        y_chunk = y_state + y_bonus                          # (B, H, tc, V)

        out_chunks.append(y_chunk)

        # Carry: final state after this chunk
        S_carry = S_chunk[:, :, -1]                          # (B, H, K, V)

        cur += tc

    out = torch.cat(out_chunks, dim=2)                       # (B, H, T, V)
    return out.to(dtype), S_carry


def _delta_affine_prefix_scan(
    A: torch.Tensor,   # (B, H, T_c, K, K)
    U: torch.Tensor,   # (B, H, T_c, K, V)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hillis–Steele parallel prefix scan over delta affine pairs.
    Same recurrence as the non-normal-RSE variant but with K×K A matrices.
    """
    T_c = A.shape[2]
    t_idx = torch.arange(T_c, device=A.device)

    d = 1
    while d < T_c:
        A_shift = F.pad(A[:, :, : T_c - d], (0, 0, 0, 0, d, 0))
        U_shift = F.pad(U[:, :, : T_c - d], (0, 0, 0, 0, d, 0))
        A_cand = torch.einsum('bhtij,bhtjk->bhtik', A, A_shift)
        U_cand = torch.einsum('bhtij,bhtjk->bhtik', A, U_shift) + U
        mask = (t_idx >= d).view(1, 1, T_c, 1, 1)
        A = torch.where(mask, A_cand, A)
        U = torch.where(mask, U_cand, U)
        d *= 2
    return A, U


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
