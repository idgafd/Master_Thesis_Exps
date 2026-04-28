"""Encoder factory for synthetics_v1.

Handles the §3 single-mechanism MQAR cohort plus the existing reduced cohort:

    Transformer:  transformer, transformer_causal
    RWKV-6:       rwkv6, rwkv6_lucid, rwkv6_delta, rwkv6_lucid_delta,
                  rwkv6_decay_coupled_delta,
                  rwkv6_convshift_multidil_symmetric_v2,
                  rwkv6_rse_strong_viscosity
    Mamba:        mamba
    Mamba-2:      mamba2, mamba2_lucid_c,
                  mamba2_convshift_multidil_symmetric_v2,
                  mamba2_rse_strong_viscosity
    LA:           linear_attn_causal, linear_attn_lucid,
                  linear_attn_convshift_multidil_symmetric_v2,
                  linear_attn_rse_strong_viscosity

Mechanism flags are derived from canonical formal_v1 backbone names (no
substring magic on most paths — explicit dispatch keeps the matrix
predictable).  Every encoder honours the contract:

    forward(x: (B, T, D), lengths: (B,), state=None) -> ((B, T, D), state)
"""

from __future__ import annotations

import math
import torch.nn as nn

from src.config import SyntheticsConfig


SUPPORTED_BACKBONES = {
    # Transformer
    "transformer",
    "transformer_causal",
    # RWKV-6
    "rwkv6",
    "rwkv6_lucid",
    "rwkv6_delta",
    "rwkv6_decay_coupled_delta",
    "rwkv6_convshift_multidil_symmetric_v2",
    "rwkv6_rse_strong_viscosity",
    # Mamba / Mamba-2
    "mamba",
    "mamba2",
    "mamba2_lucid_c",
    "mamba2_convshift_multidil_symmetric_v2",
    "mamba2_rse_strong_viscosity",
    "mamba2_rse_depth_viscosity",
    # Linear Attention
    "linear_attn_causal",
    "linear_attn_lucid",
    "linear_attn_convshift_multidil_symmetric_v2",
    "linear_attn_rse_strong_viscosity",
    "linear_attn_rse_depth_viscosity",
    # RWKV-6 RSE-depth-viscosity (causal mode — formal_v1 probe variant)
    "rwkv6_rse_depth_viscosity",
}


def _depth_graded_rse_overrides(n_layers: int) -> list[dict]:
    """Depth-graded θ-clip schedule: π/8 (L0–L1) → π/4 (L2–L3) → π/2 (L4–L5).

    Mirrors the `rwkv6_rse_depth_viscosity` schedule in formal_v1 encoder.py
    lines 752–759 (Stage-4).  For n_layers != 6, we tile the 3-band schedule
    proportionally; for the standard 6-layer setup this produces exactly
    [π/8, π/8, π/4, π/4, π/2, π/2].
    """
    bands = [
        {"theta_clip": math.pi / 8, "theta_init_scale": math.pi / 32, "theta_lora_dim": 16},
        {"theta_clip": math.pi / 8, "theta_init_scale": math.pi / 32, "theta_lora_dim": 16},
        {"theta_clip": math.pi / 4, "theta_init_scale": math.pi / 16, "theta_lora_dim": 32},
        {"theta_clip": math.pi / 4, "theta_init_scale": math.pi / 16, "theta_lora_dim": 32},
        {"theta_clip": math.pi / 2, "theta_init_scale": math.pi / 8,  "theta_lora_dim": 48},
        {"theta_clip": math.pi / 2, "theta_init_scale": math.pi / 8,  "theta_lora_dim": 48},
    ]
    if n_layers == 6:
        return bands
    # Tile by ratio for non-6 layer counts.
    out = []
    for i in range(n_layers):
        idx = min(int(i * 6 / n_layers), 5)
        out.append(bands[idx])
    return out


def build_encoder(cfg: SyntheticsConfig) -> nn.Module:
    """Construct the encoder module for `cfg.backbone`."""
    backbone = cfg.backbone

    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Supported in synthetics_v1: {sorted(SUPPORTED_BACKBONES)}."
        )

    # ── Transformer family ────────────────────────────────────────────────
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

    # ── Mamba (S6) ────────────────────────────────────────────────────────
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

    # ── Mamba-2 family ────────────────────────────────────────────────────
    # Vanilla + 2 single mechanisms (multidil_v2, lucid_c).  RSE is a separate
    # encoder class (`Mamba2RSEEncoder`) handled below.
    _mamba2_flag_table = {
        # backbone:                                          (multidil, lucid, lucid_key)
        "mamba2":                                            (False, False, "B"),
        "mamba2_convshift_multidil_symmetric_v2":            (True,  False, "B"),
        "mamba2_lucid_c":                                    (False, True,  "C"),
    }
    if backbone in _mamba2_flag_table:
        from src.models.mamba2_encoder import Mamba2Encoder
        use_multidil, use_lucid, lucid_key_source = _mamba2_flag_table[backbone]
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
            mode="recurrent",
            use_multidil_sym=use_multidil,
            use_lucid=use_lucid,
            lucid_key_source=lucid_key_source,
        )

    if backbone in ("mamba2_rse_strong_viscosity", "mamba2_rse_depth_viscosity"):
        from src.models.mamba2_rse import Mamba2RSEEncoder
        rse_per_layer_overrides = (
            _depth_graded_rse_overrides(cfg.n_layers)
            if backbone == "mamba2_rse_depth_viscosity"
            else None
        )
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
            rse_per_layer_overrides=rse_per_layer_overrides,
        )

    # ── Linear Attention family ───────────────────────────────────────────
    _la_causal_backbones = {
        "linear_attn_causal",
        "linear_attn_lucid",
        "linear_attn_convshift_multidil_symmetric_v2",
    }
    if backbone in _la_causal_backbones:
        from src.models.linear_attn_causal import CausalLinearAttentionEncoder
        return CausalLinearAttentionEncoder(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            use_multidil_sym=(backbone == "linear_attn_convshift_multidil_symmetric_v2"),
            use_convshift_sym=False,
            use_lucid=(backbone == "linear_attn_lucid"),
        )

    if backbone in ("linear_attn_rse_strong_viscosity", "linear_attn_rse_depth_viscosity"):
        from src.models.linear_attn_rse import CausalLinearAttentionRSEEncoder
        rse_per_layer_overrides = (
            _depth_graded_rse_overrides(cfg.n_layers)
            if backbone == "linear_attn_rse_depth_viscosity"
            else None
        )
        return CausalLinearAttentionRSEEncoder(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            rse_viscosity=True,
            use_multidil_sym=False,
            rse_per_layer_overrides=rse_per_layer_overrides,
        )

    # ── RWKV-6 family ─────────────────────────────────────────────────────
    delta_rule = "delta" in backbone
    lucid = "lucid" in backbone
    use_decay_coupled_delta = "decay_coupled" in backbone
    use_conv_shift_multidilation = "multidil" in backbone
    rse = backbone in ("rwkv6_rse_strong_viscosity", "rwkv6_rse_depth_viscosity")
    rse_viscosity = "viscosity" in backbone

    rse_per_layer_overrides = None
    if backbone == "rwkv6_rse_strong_viscosity":
        # Stage-4 strong-budget schedule (uniform π/2 with LoRA 48), shared with
        # `rwkv6_rse_strong_viscosity` in formal_v1 — see encoder.py:760-766.
        rse_per_layer_overrides = [
            {"theta_clip": math.pi / 2, "theta_init_scale": math.pi / 16, "theta_lora_dim": 48}
        ] * cfg.n_layers
    elif backbone == "rwkv6_rse_depth_viscosity":
        rse_per_layer_overrides = _depth_graded_rse_overrides(cfg.n_layers)

    from src.models.rwkv6_encoder import RWKV6Encoder
    return RWKV6Encoder(
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        head_size=cfg.head_size,
        mode="recurrent",
        conv_shift=use_conv_shift_multidilation,
        headscale=False,
        delta_rule=delta_rule,
        delta_warmstart=delta_rule,
        use_decay_coupled_delta=use_decay_coupled_delta,
        decay_coupled_delta_p_init=1.0,
        lucid=lucid,
        lucid_chunk_size=None,
        lucid_self_reg=False,
        temperature=False,
        discretization="zoh",
        discretization_init="zoh",
        drop_u=False,
        rse=rse,
        rse_n_scales=1,
        rse_per_layer_overrides=rse_per_layer_overrides,
        p2rse=False,
        p2rse_mixer="linear",
        rse_viscosity=rse_viscosity,
        p2rse_indep_lambda=False,
        p2rse_indep_kv=False,
        use_rmsnorm=False,
        use_hadamard_n2=False,
        use_qtail=False,
        use_qtail_gamma=False,
        use_qtail_dbeta=False,
        use_qtail_lowrank=False,
        qtail_top_k=2,
        use_data_dep_readphase=False,
        readphase_clip=None,
        use_nonnormal_rse=False,
        nonnormal_rho_kappa=None,
        use_sparse_nonnormal_rse=False,
        sparse_nn_edge_only=False,
        nonnormal_psi_static=False,
        use_loglinear=False,
        loglinear_levels=10,
        use_m2rnn=False,
        m2rnn_layer=5,
        use_conv_shift_multidilation=use_conv_shift_multidilation,
        conv_shift_multidil_padding_mode="symmetric" if "symmetric" in backbone else "auto",
        conv_shift_multidil_content_conditional=False,
        use_chanmix_bypass=False,
        use_cayley_orthogonal=False,
        cayley_rank=1,
        use_pom_vlift=False,
        pom_order=2,
        pom_expansion=64,
    )
