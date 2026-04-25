"""Encoder factory — single entry point for all encoder types."""

import torch.nn as nn

from src.config import ExperimentConfig


def build_encoder(cfg: ExperimentConfig) -> nn.Module:
    """Build encoder based on config backbone and mechanism flags."""
    backbone = cfg.backbone

    if backbone == "transformer":
        from src.models.transformer import TransformerEncoder
        return TransformerEncoder(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
        )

    if backbone == "transformer_causal":
        from src.models.transformer_causal import CausalTransformerEncoder
        return CausalTransformerEncoder(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
        )

    # Stage 11.0a — causal Katharopoulos Linear Attention with explicit L1
    # denominator.  Distinct from the parallel bidirectional ELU+1 layer in
    # `blocks.py::LinearAttentionLayer` (no denominator, no causality).
    # Stage 11.1b — same backbone + symmetric multi-dilation pre-mix before
    # Q/K/V.  Zero-regression at init: pre-mix branch_1 is center-tap
    # identity, alpha_{2,4,8}=0 ⇒ vanilla linear_attn_causal bit-exact.
    # Stage 11.1b / 11.5c / P3 v2 — the LA family registered here.
    # ``*_v2`` routes through the same LA multidil path as the pre-v2
    # variant; the ``MultiDilationDWConvShift`` class in conv_shift.py
    # was fixed universally (commit ``3af846d``), so the only difference
    # is the output directory name for distinguishability.
    _la_multidil_backbones = {
        "linear_attn_convshift_multidil_symmetric",
        "linear_attn_convshift_multidil_symmetric_v2",
    }
    # P9 / P10 — LUCID on LA.  `linear_attn_lucid` is the axis-2 cross-arch
    # transfer (LUCID alone on LA).  `linear_attn_lucid_convshift_multidil_symmetric_v2`
    # is the conditional P10 (LUCID × multidil_v2 on LA).
    _la_lucid_backbones = {
        "linear_attn_lucid",
        "linear_attn_lucid_convshift_multidil_symmetric_v2",
    }
    if backbone in (
        "linear_attn_causal",
        "linear_attn_convshift_symmetric",
    ) or backbone in _la_multidil_backbones or backbone in _la_lucid_backbones:
        from src.models.linear_attn_causal import CausalLinearAttentionEncoder
        use_multidil = (
            backbone in _la_multidil_backbones
            or backbone == "linear_attn_lucid_convshift_multidil_symmetric_v2"
        )
        return CausalLinearAttentionEncoder(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            use_multidil_sym=use_multidil,
            use_convshift_sym=(backbone == "linear_attn_convshift_symmetric"),
            use_lucid=(backbone in _la_lucid_backbones),
        )

    # ── LA LION family (LION-LIT bidirectional) ───────────────────────────
    # `linear_attn_lion`                                — vanilla LION-LIT LA
    # `linear_attn_lion_convshift_multidil_symmetric_v2` — + multidil_v2 pre-mix
    # See linear_attn_lion.py.  Maps Katharopoulos LA → LION-LIT (no decay)
    # per Afzal et al. 2025 Table 1; the unified lion_attention kernel with
    # w=0 implements the bidirectional balanced form.
    _la_lion_multidil_backbones = {
        "linear_attn_lion_convshift_multidil_symmetric_v2",
    }
    if backbone == "linear_attn_lion" or backbone in _la_lion_multidil_backbones:
        from src.models.linear_attn_lion import LIONLinearAttentionEncoder
        return LIONLinearAttentionEncoder(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            use_multidil_sym=(backbone in _la_lion_multidil_backbones),
        )

    # Stage 11.2b — Linear Attention + block-complex RSE transition + viscosity.
    # The chunked complex scan replaces the cumsum-based LA forward;
    # exponential decay subsumes the L1-denominator role on this path.
    if backbone == "linear_attn_rse_strong_viscosity":
        from src.models.linear_attn_rse import CausalLinearAttentionRSEEncoder
        return CausalLinearAttentionRSEEncoder(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            rse_viscosity=True,
        )

    if backbone == "mamba":
        from src.models.mamba_encoder import MambaEncoder
        return MambaEncoder(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            ffn_dim=cfg.ffn_dim,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
        )

    if backbone == "mamba_cuda":
        from src.models.mamba_cuda_encoder import MambaCudaEncoder
        return MambaCudaEncoder(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            ffn_dim=cfg.ffn_dim,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
        )

    if backbone == "mamba_bidir":
        from src.models.mamba_encoder import BidirMambaEncoder
        return BidirMambaEncoder(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            ffn_dim=cfg.ffn_dim,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
        )

    # ── Mamba-2 family (LION-compatible bidirectional via `mode`) ──────────
    # Naming: `mamba2` (causal), `mamba2_lion` (full bidir attention),
    # `mamba2_lion_chunk` (chunkwise bidir, long sequences),
    # `mamba2_convshift_multidil_symmetric` (Stage 11.1a — replaces the
    # internal xBC short DWConv with parallel dilated {1,2,4,8} branches
    # under symmetric padding, per §6 Stage 11.1a).
    # Stage 11.5b — Mamba-2 with a single-dilation SYMMETRIC DWConv
    # replacing the native causal Conv1d.  Isolates padding-direction
    # effect from multi-dilation (which is inert on the existing
    # `mamba2_convshift_multidil_symmetric` due to an init-gradient trap
    # in MultiDilationDWConv1d).
    # ``*_v2`` variants route through the same Mamba-2 multidil path as
    # their pre-v2 sibling; the Option-B init fix in
    # ``MultiDilationDWConv1d.__init__`` (mamba2_block.py) applies
    # universally, so the ``_v2`` naming is for output-directory
    # distinguishability relative to the pre-fix 11.1a result.
    _mamba2_backbones = {
        # (mode, multidil, convshift_sym, lucid, lucid_key, dlucid, novelty, γ_fixed, householder)
        "mamba2":                                         ("recurrent", False, False, False, "B", False, False, None, False),
        "mamba2_lion":                                    ("lion",      False, False, False, "B", False, False, None, False),
        "mamba2_lion_chunk":                              ("lion_chunk", False, False, False, "B", False, False, None, False),
        "mamba2_convshift_multidil_symmetric":            ("recurrent", True,  False, False, "B", False, False, None, False),
        "mamba2_convshift_multidil_symmetric_v2":         ("recurrent", True,  False, False, "B", False, False, None, False),
        "mamba2_convshift_symmetric":                     ("recurrent", False, True,  False, "B", False, False, None, False),
        # LUCID on Mamba-2 SSD dual form.  B-correlation preconditioner
        # applied to X_c within each chunk.
        "mamba2_lucid":                                   ("recurrent", False, False, True,  "B", False, False, None, False),
        "mamba2_lucid_convshift_multidil_symmetric_v2":   ("recurrent", True,  False, True,  "B", False, False, None, False),
        # C-side variant — query-analog correlation.
        "mamba2_lucid_c":                                 ("recurrent", False, False, True,  "C", False, False, None, False),
        # D-LUCID (v6, decay-aware via additive γ·Δcs penalty on the LUCID
        # exponent).  See `_apply_lucid_mamba2_chunked_decay_aware`.  γ = 0
        # reduces bit-exactly to LUCID.
        "mamba2_dlucid":                                  ("recurrent", False, False, True,  "B", True,  False, None, False),
        "mamba2_dlucid_c":                                ("recurrent", False, False, True,  "C", True,  False, None, False),
        "mamba2_dlucid_convshift_multidil_symmetric_v2":  ("recurrent", True,  False, True,  "B", True,  False, None, False),
        # Write-novelty gate (chunked Σ variant).
        "mamba2_novelty_gate":                            ("recurrent", False, False, False, "B", False, True,  None, False),
        "mamba2_lucid_novelty":                           ("recurrent", False, False, True,  "B", False, True,  None, False),
        "mamba2_novelty_fixed_g05":                       ("recurrent", False, False, False, "B", False, True,  0.5,  False),
        # Generalised partial Householder on inter-chunk state transition.
        # Replaces the diagonal-only inter-chunk propagation with
        #     s_{c+1} = D_c · H_h · s_c + produced_c,
        #     H_h = I − 2(1 − α_h) u_h u_hᵀ,  α_h ∈ [0, 1] (via sigmoid).
        # α_raw init = 5 ⇒ α ≈ 0.993 ⇒ H ≈ I at step 0 ⇒ near-vanilla
        # Mamba-2 at init, Adam-engageable afterwards.  Replaces the
        # vectorised inter-chunk einsum with a per-chunk loop — nC ~ 5–10
        # extra matmuls of tiny (N×N) matrices per head per batch ≈ negligible.
        "mamba2_householder":                             ("recurrent", False, False, False, "B", False, False, None, True),
    }
    if backbone in _mamba2_backbones:
        from src.models.mamba2_encoder import Mamba2Encoder
        (mode_for_backbone, use_multidil, use_convshift_sym, use_lucid,
         lucid_key_source, lucid_decay_aware, use_novelty_gate,
         novelty_gamma_fixed, use_householder) = _mamba2_backbones[backbone]
        return Mamba2Encoder(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            ffn_dim=cfg.ffn_dim,
            d_state=cfg.mamba2_d_state,
            d_conv=cfg.mamba_d_conv,
            headdim=cfg.mamba2_headdim,
            expand=cfg.mamba_expand,
            ngroups=cfg.mamba2_ngroups,
            chunk_size=cfg.mamba2_chunk_size,
            mode=mode_for_backbone,
            use_multidil_sym=use_multidil,
            use_convshift_sym=use_convshift_sym,
            use_lucid=use_lucid,
            lucid_key_source=lucid_key_source,
            lucid_decay_aware=lucid_decay_aware,
            use_novelty_gate=use_novelty_gate,
            novelty_gamma_fixed=novelty_gamma_fixed,
            use_householder=use_householder,
        )

    # Stage 11.2a — Mamba-2 + RSE (block-complex SSD transition) + viscosity.
    # Per §6 Stage 11.2a: pure-PyTorch chunked complex SSD scan replaces the
    # vanilla ssd_scan_causal; rest of the Mamba-2 stack unchanged.
    if backbone == "mamba2_rse_strong_viscosity":
        from src.models.mamba2_rse import Mamba2RSEEncoder
        return Mamba2RSEEncoder(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            ffn_dim=cfg.ffn_dim,
            d_state=cfg.mamba2_d_state,
            d_conv=cfg.mamba_d_conv,
            headdim=cfg.mamba2_headdim,
            expand=cfg.mamba_expand,
            ngroups=cfg.mamba2_ngroups,
            chunk_size=cfg.mamba2_chunk_size,
            rse_viscosity=True,
        )

    # All RWKV-6 variants (rwkv6, lion, and all mechanism combinations)
    # Determine mode from backbone name or explicit config
    mode_map = {
        "rwkv6": "recurrent",
        "rwkv6_lucid": "recurrent",
        "rwkv6_lucid_sr": "recurrent",
        "rwkv6_delta": "recurrent",
        # TODO_DELTA_RULE Tier-1 diagnostic — a0 init -5 so β ≈ 0 at t=0
        "rwkv6_delta_warmstart": "recurrent",
        # Stage 8 T1 — recurrent delta rank-1 erase, ACTUALLY WIRED on the
        # recurrent path (the historical rwkv6_delta_warmstart run at
        # commit 3aebd56 did not branch on use_delta_rule in recurrent;
        # the 0.1260 plateau is an implementation artifact, not evidence).
        # Re-run with the fixed wiring, hard-gated for exact zero-init.
        "rwkv6_delta_warmstart_fixed": "recurrent",
        "rwkv6_lucid_delta": "recurrent",
        "rwkv6_headscale": "recurrent",
        # Stage 2 discretization variants (causal)
        "rwkv6_trap": "recurrent",
        "rwkv6_trap_var": "recurrent",
        "rwkv6_gen2": "recurrent",
        "rwkv6_gen2_zoh_init": "recurrent",
        "rwkv6_gen2_trap_init": "recurrent",
        "rwkv6_ab3": "recurrent",
        "rwkv6_convshift_trap": "recurrent",
        # Stage 11.5a — plain symmetric single-dilation DWConvShift on
        # default (zoh) discretisation; isolates padding-direction effect
        # from multi-dilation (which is inert due to an init-gradient
        # trap on MultiDilationDWConvShift; see MULTIDIL_INIT_FIX_HANDOFF).
        "rwkv6_convshift_symmetric": "recurrent",
        # Stage 3 RSE variants (causal recurrent only)
        "rwkv6_rse": "recurrent",
        "rwkv6_rse_convshift": "recurrent",
        "rwkv6_rse_headscale": "recurrent",
        "rwkv6_rse_convshift_headscale": "recurrent",
        "rwkv6_rse_m2": "recurrent",
        "rwkv6_rse_m2_convshift": "recurrent",
        "rwkv6_rse_m4": "recurrent",
        # Stage 4 (refined) — per-layer depth-graded rotation budget
        "rwkv6_rse_depth": "recurrent",
        "rwkv6_rse_strong": "recurrent",
        # Stage 5 — Paired-Pole RSE (P²-RSE): 2 complex poles / block, real β mixer
        "rwkv6_p2rse": "recurrent",            # unconstrained linear β (Exp A)
        "rwkv6_p2rse_softmax": "recurrent",    # convex softmax β (Exp B, control)
        # Phase 2 — P²-RSE × Stage-4 budget refinements (orthogonal stacking)
        "rwkv6_p2rse_strong": "recurrent",     # P²-RSE + uniform large budget (π/2, LoRA 48)
        "rwkv6_p2rse_depth":  "recurrent",     # P²-RSE + depth-graded budget (π/8 → π/4 → π/2)
        # Phase 3 — Viscosity coupling (Rayleigh dissipation, soft self-regulating clip)
        "rwkv6_rse_strong_viscosity": "recurrent",   # Stage-4 strong budget + λ_eff = λ + η·θ²
        "rwkv6_rse_depth_viscosity":  "recurrent",   # Stage-4 depth-graded budget + viscosity
        # Diagnostic control for Phase 2b: shared-λ P²-RSE + strong + viscosity
        # (the composition that Phase 2b added indep-λ on top of — never tested
        # alone before, needed to attribute Phase 2b's regression).
        "rwkv6_p2rse_strong_viscosity": "recurrent",
        # Stage 7A (A1′) — data-dependent readout phase on the RSE anchor.
        # Adds a zero-init Linear producing φ_{t,h,b}; the complex readout
        # contracts exp(-iφ)·conj(r_c) with S_total before .real, recovering
        # quadrature content identified by STAGE7_DIAGNOSTICS §D2.
        "rwkv6_rse_dphi_viscosity": "recurrent",
        # Stage 7A (A1′-no-viscosity control) for isolating φ alone
        # without the viscosity composition (optional, not main).
        "rwkv6_rse_dphi": "recurrent",
        # Stage 8 T2 — non-normal RSE in polar parameterisation.
        # G = e^{-λ} R(ψ)^T diag(e^ρ, e^{-ρ}) R(ψ) R(θ) per block, with ρ
        # zero-init (exact RSE reduction).  Viscosity variant is primary;
        # no-viscosity control is optional for isolation.
        "rwkv6_nonnormal_rse_viscosity": "recurrent",
        "rwkv6_nonnormal_rse": "recurrent",
        # Stage 9 — sparse edge-layer specialist transition.
        # Adds per-(layer, head) gate g_{ℓ,h} zero-init multiplying ρ and ψ.
        # At init g=0 ⇒ exact RSE+viscosity reduction per (ℓ, h).
        # κ tightened to 0.4 for safer spectral envelope.
        # Option A (learned sparsity): "rwkv6_sparse_nonnormal_rse_viscosity"
        # Option B (hard edge-only):   "rwkv6_sparse_nonnormal_rse_edge_only_viscosity"
        "rwkv6_sparse_nonnormal_rse_viscosity": "recurrent",
        "rwkv6_sparse_nonnormal_rse_edge_only_viscosity": "recurrent",
        # Phase 2b — Independent-λ P²-RSE (each pole has its own decay LoRA)
        "rwkv6_p2rse_indeplam_strong_viscosity": "recurrent",  # indep-λ + strong θ budget + viscosity
        "rwkv6_p2rse_indeplam_depth_viscosity":  "recurrent",  # indep-λ + depth-graded θ + viscosity
        # Phase 2b-ext — Independent drive-side (k, v) on top of Phase 2b
        "rwkv6_p2rse_indeplam_extkv_strong_viscosity": "recurrent",  # + rank-32 k/v LoRA per pole
        "rwkv6_p2rse_indeplam_extkv_depth_viscosity":  "recurrent",
        # Stage 6 — EXPRESSIVENESS paper (Mongaras & Larson 2025) adaptations on pure RWKV-6
        "rwkv6_rmsnorm":     "recurrent",   # GroupNorm → per-head RMSNorm readout (paper §4.5)
        "rwkv6_hadamard_n2": "recurrent",   # + diagonal n=2 Taylor branch (no cross-channels)
        "rwkv6_qtail":       "recurrent",   # + Kronecker n=2 tail at top 2 layers (cross-channels)
        # Stage 6.5 — Refinement of the single Kronecker mechanism.
        # rwkv6_qtail_gamma: learnable per-head γ decay coupling on the
        # Kronecker branch (pair-decay = γ·(w_i+w_j)). Paper's Taylor
        # derivation imposes no decay; γ parameterizes the whole spectrum.
        "rwkv6_qtail_gamma": "recurrent",
        # R2 — qtail-γ + data-dependent β_{q,t} per token per head.
        # Makes the Kronecker-branch gating selective (Mamba-2 analog).
        "rwkv6_qtail_gamma_dbeta": "recurrent",
        # Low-rank Kronecker — project r,k to K'=16 per head, Kronecker K'²=256.
        # Eckart-Young truncation test; enables all-layer Kronecker if effect survives.
        "rwkv6_qtail_lowrank": "recurrent",
        "rwkv6_qtail_gamma_lowrank": "recurrent",
        "rwkv6_qtail_gamma_dbeta_lowrank": "recurrent",
        # All-layer lowrank — Kronecker branch at ALL 6 layers (not just top 2).
        # Made feasible by the K'=16 memory reduction.  Tests whether the
        # Kronecker mechanism helps at shallow layers too, or only at the
        # depth where γ had the strongest specialisation (L5).
        "rwkv6_qtail_lowrank_all": "recurrent",
        "rwkv6_qtail_gamma_lowrank_all": "recurrent",
        "rwkv6_qtail_gamma_dbeta_lowrank_all": "recurrent",
        # Stage 10.1 — Log-Linear RWKV-6 (Family A, structural multi-scale)
        "rwkv6_loglinear": "recurrent",
        # Stage 10.2 — M²RNN sparing-use (Family C, non-linear state at L5)
        "rwkv6_m2rnn_sparse": "recurrent",
        # Stage 10.3 — Multi-dilation ConvShift (Family A input-side)
        "rwkv6_convshift_multidil": "recurrent",
        # Stage 10.3-sym — Symmetric-padding multi-dilation control, resolves
        # the causality-vs-dilation confound against `rwkv6_convshift_trap`.
        "rwkv6_convshift_multidil_symmetric": "recurrent",
        # _v2 suffix = reruns with the init-gradient-trap fix in
        # MultiDilationDWConvShift (α_{d>1}=0.01, branch_{d>1}.weight ~ N(0, 0.01)).
        # Code path is identical; the suffix is purely for output-directory
        # naming and result accounting. See MULTIDIL_INIT_FIX_HANDOFF.md.
        # NOTE: mamba2_* and linear_attn_* _v2 variants dispatch through
        # their own architecture-specific branches above; the entries here
        # are RWKV-6 only.
        "rwkv6_convshift_multidil_symmetric_v2": "recurrent",
        "rwkv6_rse_convshift_multidil_symmetric_v2": "recurrent",
        "rwkv6_convshift_multidil_symmetric_gated_v2": "recurrent",
        "rwkv6_qtail_lowrank_all_convshift_multidil_symmetric_v2": "recurrent",
        # H1 — per-(head, channel-pair) β allocation on the Kronecker branch.
        # `betapp` substring → use_qtail_beta_per_pair=True.  Tests whether
        # the Family-D nulls were β-allocation-limited rather than capacity-
        # limited.  No `gamma` substring ⇒ no γ coupling (clean isolation).
        "rwkv6_qtail_lowrank_all_betapp_convshift_multidil_symmetric_v2": "recurrent",
        # H2 — γ=0 init on the Kronecker decay coupling (undecayed accumulator).
        # `gamma0` substring → use_qtail_gamma=True with qtail_gamma_init=0.0.
        # Tests whether cross-channel state needs longer memory than per-
        # channel decay implies (paper's literal Taylor formulation).
        "rwkv6_qtail_lowrank_all_gamma0_convshift_multidil_symmetric_v2": "recurrent",
        # Stage 11 P7 + P8 — LUCID × multidil_v2 composition (P7 MARGINAL+
        # triggered P8 per STAGE11_AGENT_QUEUE decision tree):
        "rwkv6_lucid_convshift_multidil_symmetric_v2": "recurrent",
        "rwkv6_lucid_rse_convshift_multidil_symmetric_v2": "recurrent",
        # CB-1 — Composition of RSE (Stage 3 transition-side) × multidil_sym
        # (Stage 10.3-sym input-side). Tests whether input-side RF expansion
        # and transition-side rotation are orthogonal gains over `convshift_trap`.
        "rwkv6_rse_convshift_multidil_symmetric": "recurrent",
        # CB-5a — frontend_v2 lean (~413K frontend params, -1.5M vs v1).
        # Tests structural change at reduced capacity.
        "rwkv6_frontend_v2": "recurrent",
        # CB-5b — frontend_v2 matched (~1.94M, param-neutral vs v1).
        # Isolates structural change from parameter-count effect.
        "rwkv6_frontend_v2_matched": "recurrent",
        # CB-7 — Cross-channel × temporal composition (post-CB-1 pivot).
        # qtail_lowrank_all (channel-side Kronecker) × multidil_sym (input-side
        # temporal). The cheapest genuinely-orthogonal composition with
        # existing infrastructure; tests whether temporal and channel-side
        # gains compose additively after CB-1 falsified temporal × temporal.
        "rwkv6_qtail_lowrank_all_convshift_multidil_symmetric": "recurrent",
        # CB-3 — Content-conditional α_d on multi-dilation ConvShift.
        # Each frame selects its own dilation mix via softmax(W_α x + b_d).
        # Genuinely new expressivity axis vs the per-layer scalar α_d.
        "rwkv6_convshift_multidil_symmetric_gated": "recurrent",
        "rwkv6_convshift_multidil_gated": "recurrent",
        # Stage 10.4 — Avey partial-embedding ChannelMix bypass (Family D)
        "rwkv6_chanmix_bypass": "recurrent",
        # Stage 10.5 — Cayley-orthogonal (NCGRU-style, rank-1)
        "rwkv6_orthogonal": "recurrent",
        # Stage 10.6 — PoM polynomial value-lift (thin)
        "rwkv6_pom_vlift": "recurrent",
        # Stage 10.7 — Log-Linear × RSE composition (conditional on 10.1 ≥ MARGINAL)
        "rwkv6_loglinear_rse_strong_viscosity": "recurrent",
        # Existing LION variants
        "lion": "lion",
        "lion_convshift": "lion",
        "lion_lucid": "lion",
        "lion_lucid_chunked": "lion",
        "lion_delta": "lion",
        "lion_convshift_delta": "lion",
        "lion_headscale": "lion",
        "lion_convshift_headscale": "lion",
        "lion_temperature": "lion",
        # Stage 2 transfer to LION — route through bidir_serial so the
        # discretization fires on both forward and backward recurrent sweeps.
        # (Pure LION uses parallel T×T attention, which has no recurrence
        #  and therefore no discretization concept to ablate.)
        "lion_trap": "bidir_serial",
        "lion_convshift_trap": "bidir_serial",
        "bidir_serial": "bidir_serial",
    }

    mode = mode_map.get(backbone, cfg.rwkv_mode)

    # Derive mechanism flags from backbone name (override config)
    conv_shift = "convshift" in backbone or cfg.conv_shift
    headscale = "headscale" in backbone or cfg.headscale
    delta_rule = "delta" in backbone or cfg.delta_rule
    # TODO_DELTA_RULE H1 — a0 = -5 init. Triggered by "warmstart" substring.
    delta_warmstart = delta_rule and "warmstart" in backbone
    lucid = "lucid" in backbone or cfg.lucid
    lucid_self_reg = "lucid_sr" in backbone or cfg.lucid_self_reg
    temperature = "temperature" in backbone or cfg.temperature
    lucid_chunk_size = 64 if "chunked" in backbone else cfg.lucid_chunk_size

    import re
    import math as _math

    # Stage 2 discretization — derive from backbone name (substring match).
    # Order matters: gen2 wins over trap (so "gen2_trap_init" stays gen2).
    if "ab3" in backbone:
        discretization = "ab3"
    elif "gen2" in backbone:
        discretization = "gen2"
    elif "trap_var" in backbone:
        discretization = "trap_var"
    elif "trap" in backbone:
        discretization = "trap"
    else:
        discretization = cfg.discretization

    if "trap_init" in backbone:
        discretization_init = "trap"
    elif "zoh_init" in backbone:
        discretization_init = "zoh"
    else:
        discretization_init = cfg.discretization_init

    # Trap/AB3 ablate the bonus `u` (it would conflate readout α₀ with state α₀).
    # gen2 keeps `u` because it learns α₀ on top.
    drop_u = discretization in ("trap", "trap_var", "ab3") or cfg.discretization_drop_u

    # Stage 3: RSE (rotational state evolution) — substring trigger on backbone name.
    # P²-RSE also requires rse=True (it sits on top of the single-pole RSE infrastructure).
    rse = (
        ("rse" in backbone.split("_"))
        or ("p2rse" in backbone)
        or cfg.rse
        or cfg.p2rse
    )
    # Multi-Rate RSE: substring "m{N}" in backbone name overrides cfg.rse_n_scales.
    rse_n_scales = cfg.rse_n_scales
    m_match = re.search(r"_m(\d+)(?:_|$)", backbone)
    if m_match:
        rse_n_scales = int(m_match.group(1))

    # Stage 5 viscosity coupling: triggered by substring "viscosity" in backbone
    # or explicit cfg.rse_viscosity.  Orthogonal to p2rse — can be combined.
    rse_viscosity = ("viscosity" in backbone) or getattr(cfg, "rse_viscosity", False)

    # Stage 7A (A1′) — data-dependent readout phase, triggered by "dphi"
    # substring.  Requires rse=True (enforced in RWKV6TimeMix __init__).
    use_data_dep_readphase = "dphi" in backbone

    # Stage 8 T2 — non-normal RSE, triggered by "nonnormal_rse" substring.
    # Requires rse=True (enforced in RWKV6TimeMix __init__).
    use_nonnormal_rse = "nonnormal_rse" in backbone

    # Stage 9 — sparse edge-layer specialist transition.
    # Triggered by "sparse_nonnormal_rse" substring. Requires use_nonnormal_rse.
    # Option B via "edge_only" substring; otherwise learned sparsity (Option A).
    use_sparse_nonnormal_rse = "sparse_nonnormal_rse" in backbone
    sparse_nn_edge_only = use_sparse_nonnormal_rse and "edge_only" in backbone
    # Stage-9 variants tighten κ from 0.6 → 0.4 per STAGE9_PLAN §2.3.
    nonnormal_rho_kappa_override = 0.4 if use_sparse_nonnormal_rse else None
    # Stage-9 Fix 3 — static ψ (no token-dependent ψ LoRA) for cleaner
    # identifiability.  Does not apply to dense T2 backbone.
    nonnormal_psi_static = use_sparse_nonnormal_rse

    # Optional clip override, triggered by substring "phiclip{N}" meaning
    # the clip is π/N.  Example: `rwkv6_rse_dphi_phiclip2_viscosity` → π/2.
    # Absent ⇒ None ⇒ time-mix default (π, full circle).
    readphase_clip = None
    phiclip_match = re.search(r"phiclip(\d+)", backbone)
    if phiclip_match:
        readphase_clip = _math.pi / int(phiclip_match.group(1))

    # Stage 6 — EXPRESSIVENESS paper adaptations.  Name-substring triggers:
    #   rwkv6_rmsnorm     → GroupNorm → per-head RMSNorm only
    #   rwkv6_hadamard_n2 → RMSNorm + diagonal (k⊙k, r⊙r) second-order branch
    #   rwkv6_qtail       → RMSNorm + Kronecker (k⊗k, r⊗r) tail at top 2 layers
    use_rmsnorm = (
        "rmsnorm" in backbone
        or "hadamard_n2" in backbone
        or "qtail" in backbone
    )
    use_hadamard_n2 = "hadamard_n2" in backbone
    use_qtail = "qtail" in backbone
    # Stage-6.5 Kronecker refinement: learnable per-head γ decay coupling
    # triggered by substring "gamma" in a qtail-family backbone name.
    use_qtail_gamma = use_qtail and "gamma" in backbone
    # R2 — data-dep β, triggered by substring "dbeta" (requires qtail).
    use_qtail_dbeta = use_qtail and "dbeta" in backbone
    # Low-rank Kronecker — triggered by substring "lowrank" (requires qtail).
    use_qtail_lowrank = use_qtail and "lowrank" in backbone
    # All-layer qtail — substring "lowrank_all" enables Kronecker at every
    # layer (not just top-2).  Only applicable for lowrank because full
    # K²=4096 Kronecker at 6 layers wouldn't fit memory.
    qtail_all_layers = use_qtail and "lowrank_all" in backbone
    # H1 — per-(head, channel-pair) β allocation, triggered by "betapp".
    use_qtail_beta_per_pair = use_qtail and "betapp" in backbone
    # H2 — γ=0 init for the Kronecker decay coupling, triggered by "gamma0".
    # Forces use_qtail_gamma=True (γ machinery must exist for init to apply)
    # and overrides the default 1.0 init to 0.0 (undecayed accumulator).
    if use_qtail and "gamma0" in backbone:
        use_qtail_gamma = True
        qtail_gamma_init = 0.0
    else:
        qtail_gamma_init = 1.0

    # ── Stage 10 — new mechanism families ────────────────────────────────
    # 10.1 Log-Linear RWKV-6 (Fenwick bucket readout).
    use_loglinear = "loglinear" in backbone or getattr(cfg, "use_loglinear", False)
    loglinear_levels = getattr(cfg, "loglinear_levels", 10)
    # 10.2 M²RNN sparing-use (non-linear state at a single layer).
    use_m2rnn = "m2rnn" in backbone or getattr(cfg, "use_m2rnn", False)
    m2rnn_layer = getattr(cfg, "m2rnn_layer", 5)
    # 10.3 Multi-dilation ConvShift — substring "multidil" on top of ConvShift.
    # Also force conv_shift=True so the token-shift path uses the learned conv.
    use_conv_shift_multidilation = (
        "multidil" in backbone or getattr(cfg, "use_conv_shift_multidilation", False)
    )
    if use_conv_shift_multidilation:
        conv_shift = True
    # 10.3-sym: symmetric-padding multidilation variant, for the
    # causality-vs-dilation apples-to-apples control against `_convshift_trap`.
    # "auto" = mode-based default (causal for recurrent, symmetric for lion).
    if use_conv_shift_multidilation and "multidil_symmetric" in backbone:
        conv_shift_multidil_padding_mode = "symmetric"
    else:
        conv_shift_multidil_padding_mode = "auto"
    # CB-3: content-conditional α_d — substring "gated" on top of multidil.
    conv_shift_multidil_content_conditional = (
        use_conv_shift_multidilation and "gated" in backbone
    ) or getattr(cfg, "conv_shift_multidil_content_conditional", False)
    # 10.4 ChannelMix bypass (Avey partial-embedding).
    use_chanmix_bypass = (
        "chanmix_bypass" in backbone or getattr(cfg, "use_chanmix_bypass", False)
    )
    # 10.5 Cayley-orthogonal transition (rank-1 by default).
    use_cayley_orthogonal = (
        "orthogonal" in backbone or getattr(cfg, "use_cayley_orthogonal", False)
    )
    cayley_rank = getattr(cfg, "cayley_rank", 1)
    # 10.6 PoM polynomial value-lift (thin).
    use_pom_vlift = (
        "pom_vlift" in backbone or getattr(cfg, "use_pom_vlift", False)
    )
    pom_order = getattr(cfg, "pom_order", 2)
    pom_expansion = getattr(cfg, "pom_expansion", 64)

    # Stage 5: P²-RSE flags
    p2rse = ("p2rse" in backbone) or cfg.p2rse
    # Phase 2b: independent-λ paired-pole variant (extra decay LoRA per pole).
    # Triggered by substring "indeplam" in backbone name. Requires p2rse.
    p2rse_indep_lambda = "indeplam" in backbone
    # Phase 2b-ext: independent drive-side (k, v) per pole via rank-32 LoRA delta.
    # Triggered by substring "extkv" in backbone name. Requires indeplam.
    p2rse_indep_kv = "extkv" in backbone
    if "p2rse_softmax" in backbone:
        p2rse_mixer = "softmax"
    elif backbone in ("rwkv6_p2rse_strong", "rwkv6_p2rse_depth"):
        # Phase-2 variants: use the Phase-1-winning softmax mixer by default.
        # Linear β under-performed at 0.1250 vs softmax 0.1220; we're stacking
        # the 2-pole mechanism onto the Stage-4 budget, so we pick the best
        # mixer rather than re-ablating it.
        p2rse_mixer = "softmax"
    elif p2rse_indep_lambda:
        # Phase 2b: inherit the Phase-1-winning softmax mixer — this is a
        # compositional stack, we don't re-ablate the mixer choice.
        p2rse_mixer = "softmax"
    elif p2rse:
        p2rse_mixer = "linear"
    else:
        p2rse_mixer = cfg.p2rse_mixer

    # Stage 4 — depth-graded rotation budget per layer.
    # Diagnosed in scripts/rse_theta_diagnostic.py: deeper layers learn larger
    # data-dependent rotation contributions; uniform per-layer budget under-
    # serves L4–L5. Override schedule applies only when backbone name asks.
    rse_per_layer_overrides = None
    if backbone in ("rwkv6_rse_depth", "rwkv6_p2rse_depth", "rwkv6_rse_depth_viscosity", "rwkv6_p2rse_indeplam_depth_viscosity", "rwkv6_p2rse_indeplam_extkv_depth_viscosity"):
        # Depth-graded rotation budget L0..L5.  Shared between single-pole
        # (Stage-4), paired-pole (Stage-5 Phase-2), and viscosity-coupled
        # (Stage-5 Phase-3) variants.  The viscosity refinement applies a
        # soft self-regulating clip on TOP of this schedule.
        rse_per_layer_overrides = [
            {"theta_clip": _math.pi / 8, "theta_init_scale": _math.pi / 32, "theta_lora_dim": 16},
            {"theta_clip": _math.pi / 8, "theta_init_scale": _math.pi / 32, "theta_lora_dim": 16},
            {"theta_clip": _math.pi / 4, "theta_init_scale": _math.pi / 16, "theta_lora_dim": 32},
            {"theta_clip": _math.pi / 4, "theta_init_scale": _math.pi / 16, "theta_lora_dim": 32},
            {"theta_clip": _math.pi / 2, "theta_init_scale": _math.pi / 8,  "theta_lora_dim": 48},
            {"theta_clip": _math.pi / 2, "theta_init_scale": _math.pi / 8,  "theta_lora_dim": 48},
        ]
    elif backbone in ("rwkv6_rse_strong", "rwkv6_p2rse_strong", "rwkv6_rse_strong_viscosity", "rwkv6_p2rse_strong_viscosity", "rwkv6_p2rse_indeplam_strong_viscosity", "rwkv6_p2rse_indeplam_extkv_strong_viscosity", "rwkv6_rse_dphi_viscosity", "rwkv6_rse_dphi", "rwkv6_nonnormal_rse_viscosity", "rwkv6_nonnormal_rse", "rwkv6_sparse_nonnormal_rse_viscosity", "rwkv6_sparse_nonnormal_rse_edge_only_viscosity", "rwkv6_loglinear_rse_strong_viscosity"):
        # Uniform but expanded budget: doubles clip and LoRA dim, keeps init small.
        # Shared between Stage-4 (rse_strong), Phase-2 (p2rse_strong),
        # Phase-3 (rse_strong_viscosity), and Stage-7A (rse_dphi_viscosity).
        rse_per_layer_overrides = [
            {"theta_clip": _math.pi / 2, "theta_init_scale": _math.pi / 16, "theta_lora_dim": 48}
        ] * cfg.n_layers

    from src.models.rwkv6_encoder import RWKV6Encoder
    return RWKV6Encoder(
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        head_size=cfg.head_size,
        mode=mode,
        conv_shift=conv_shift,
        headscale=headscale,
        delta_rule=delta_rule,
        delta_warmstart=delta_warmstart,
        lucid=lucid,
        lucid_chunk_size=lucid_chunk_size,
        lucid_self_reg=lucid_self_reg,
        temperature=temperature,
        discretization=discretization,
        discretization_init=discretization_init,
        drop_u=drop_u,
        rse=rse,
        rse_n_scales=rse_n_scales,
        rse_per_layer_overrides=rse_per_layer_overrides,
        p2rse=p2rse,
        p2rse_mixer=p2rse_mixer,
        rse_viscosity=rse_viscosity,
        p2rse_indep_lambda=p2rse_indep_lambda,
        p2rse_indep_kv=p2rse_indep_kv,
        use_rmsnorm=use_rmsnorm,
        use_hadamard_n2=use_hadamard_n2,
        use_qtail=use_qtail,
        use_qtail_gamma=use_qtail_gamma,
        use_qtail_dbeta=use_qtail_dbeta,
        use_qtail_lowrank=use_qtail_lowrank,
        qtail_top_k=(cfg.n_layers if qtail_all_layers else 2),
        use_qtail_beta_per_pair=use_qtail_beta_per_pair,
        qtail_gamma_init=qtail_gamma_init,
        use_data_dep_readphase=use_data_dep_readphase,
        readphase_clip=readphase_clip,
        use_nonnormal_rse=use_nonnormal_rse,
        nonnormal_rho_kappa=nonnormal_rho_kappa_override,
        use_sparse_nonnormal_rse=use_sparse_nonnormal_rse,
        sparse_nn_edge_only=sparse_nn_edge_only,
        nonnormal_psi_static=nonnormal_psi_static,
        use_loglinear=use_loglinear,
        loglinear_levels=loglinear_levels,
        use_m2rnn=use_m2rnn,
        m2rnn_layer=m2rnn_layer,
        use_conv_shift_multidilation=use_conv_shift_multidilation,
        conv_shift_multidil_padding_mode=conv_shift_multidil_padding_mode,
        conv_shift_multidil_content_conditional=conv_shift_multidil_content_conditional,
        use_chanmix_bypass=use_chanmix_bypass,
        use_cayley_orthogonal=use_cayley_orthogonal,
        cayley_rank=cayley_rank,
        use_pom_vlift=use_pom_vlift,
        pom_order=pom_order,
        pom_expansion=pom_expansion,
    )
