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
    # `mamba2_lion_chunk` (chunkwise bidir, long sequences).
    if backbone in ("mamba2", "mamba2_lion", "mamba2_lion_chunk"):
        from src.models.mamba2_encoder import Mamba2Encoder
        mode_for_backbone = {
            "mamba2": "recurrent",
            "mamba2_lion": "lion",
            "mamba2_lion_chunk": "lion_chunk",
        }[backbone]
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
        )

    # All RWKV-6 variants (rwkv6, lion, and all mechanism combinations)
    # Determine mode from backbone name or explicit config
    mode_map = {
        "rwkv6": "recurrent",
        "rwkv6_lucid": "recurrent",
        "rwkv6_lucid_sr": "recurrent",
        "rwkv6_delta": "recurrent",
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
    lucid = "lucid" in backbone or cfg.lucid
    lucid_self_reg = "lucid_sr" in backbone or cfg.lucid_self_reg
    temperature = "temperature" in backbone or cfg.temperature
    lucid_chunk_size = 64 if "chunked" in backbone else cfg.lucid_chunk_size

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
    import re
    m_match = re.search(r"_m(\d+)(?:_|$)", backbone)
    if m_match:
        rse_n_scales = int(m_match.group(1))

    # Stage 5 viscosity coupling: triggered by substring "viscosity" in backbone
    # or explicit cfg.rse_viscosity.  Orthogonal to p2rse — can be combined.
    rse_viscosity = ("viscosity" in backbone) or getattr(cfg, "rse_viscosity", False)

    # Stage 5: P²-RSE flags
    p2rse = ("p2rse" in backbone) or cfg.p2rse
    if "p2rse_softmax" in backbone:
        p2rse_mixer = "softmax"
    elif backbone in ("rwkv6_p2rse_strong", "rwkv6_p2rse_depth"):
        # Phase-2 variants: use the Phase-1-winning softmax mixer by default.
        # Linear β under-performed at 0.1250 vs softmax 0.1220; we're stacking
        # the 2-pole mechanism onto the Stage-4 budget, so we pick the best
        # mixer rather than re-ablating it.
        p2rse_mixer = "softmax"
    elif p2rse:
        p2rse_mixer = "linear"
    else:
        p2rse_mixer = cfg.p2rse_mixer

    # Stage 4 — depth-graded rotation budget per layer.
    # Diagnosed in scripts/rse_theta_diagnostic.py: deeper layers learn larger
    # data-dependent rotation contributions; uniform per-layer budget under-
    # serves L4–L5. Override schedule applies only when backbone name asks.
    import math as _math
    rse_per_layer_overrides = None
    if backbone in ("rwkv6_rse_depth", "rwkv6_p2rse_depth", "rwkv6_rse_depth_viscosity"):
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
    elif backbone in ("rwkv6_rse_strong", "rwkv6_p2rse_strong", "rwkv6_rse_strong_viscosity"):
        # Uniform but expanded budget: doubles clip and LoRA dim, keeps init small.
        # Shared between Stage-4 (rse_strong), Phase-2 (p2rse_strong),
        # and Phase-3 (rse_strong_viscosity) variants.
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
    )
