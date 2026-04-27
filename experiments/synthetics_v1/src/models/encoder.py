"""Slim encoder factory for synthetics_v1.

Handles the 8 backbones in the reduced cohort (PLAN.md §4.0):

    transformer, transformer_causal, rwkv6, rwkv6_lucid, rwkv6_delta,
    rwkv6_lucid_delta, mamba, mamba2

Mechanism flags (lucid, delta_rule) are derived from substring matching on the
backbone name — same convention as `formal_v1/src/models/encoder.py`. Every
encoder honours the contract:

    forward(x: (B, T, D), lengths: (B,), state=None) -> ((B, T, D), state)

To extend (e.g. add LION variants for Cohort B), follow the patterns below;
do NOT copy the entire 500-line formal_v1 dispatcher unless we actually need
the full mechanism matrix.
"""

from __future__ import annotations

import torch.nn as nn

from src.config import SyntheticsConfig


SUPPORTED_BACKBONES = {
    "transformer",
    "transformer_causal",
    "rwkv6",
    "rwkv6_lucid",
    "rwkv6_delta",
    # Stage 12 — Decay-Coupled Delta (per-head learnable γ_t = α_t^{p_h}
    # weighting on the rank-1 erase direction).  See
    # formal_v1/STAGE12_DECAY_COUPLED_DELTA.md §4.2 for the MQAR cohort spec.
    # `rwkv6_delta` above is the negative control on the same path.
    "rwkv6_decay_coupled_delta",
    "mamba",
    "mamba2",
}


def build_encoder(cfg: SyntheticsConfig) -> nn.Module:
    """Construct the encoder module for `cfg.backbone`."""
    backbone = cfg.backbone

    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Supported in synthetics_v1: {sorted(SUPPORTED_BACKBONES)}. "
            "If you need more variants, extend this dispatcher; do not "
            "blindly copy formal_v1's 500-line encoder.py without the "
            "matching mechanism config fields."
        )

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

    if backbone == "mamba2":
        from src.models.mamba2_encoder import Mamba2Encoder
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
        )

    # All remaining: RWKV-6 family, mode="recurrent" (causal). Mechanism flags
    # follow formal_v1's substring convention.
    delta_rule = "delta" in backbone
    lucid = "lucid" in backbone
    # Stage 12 — Decay-Coupled Delta (per-head learnable γ = α^{p_h} on the
    # rank-1 erase direction). Substring "decay_coupled" → flag.  Implies
    # delta_rule + a0=-8 warmstart inside RWKV6TimeMix.
    use_decay_coupled_delta = "decay_coupled" in backbone

    from src.models.rwkv6_encoder import RWKV6Encoder
    return RWKV6Encoder(
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        head_size=cfg.head_size,
        mode="recurrent",
        conv_shift=False,
        headscale=False,
        delta_rule=delta_rule,
        # warmstart=True keeps the delta branch ~off at init (iclr ≈ 0.013,
        # a0_init=-5) so SGD can grow it where useful. The False default
        # fires delta at full strength (iclr ≈ 1.76 at t=0) and empirically
        # destroys the randomly-initialised wkv_state before useful
        # associations form — see mechanisms/delta_rule.py:36-46 comment.
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
        rse=False,
        rse_n_scales=1,
        rse_per_layer_overrides=None,
        p2rse=False,
        p2rse_mixer="linear",
        rse_viscosity=False,
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
        use_conv_shift_multidilation=False,
        conv_shift_multidil_padding_mode="auto",
        conv_shift_multidil_content_conditional=False,
        use_chanmix_bypass=False,
        use_cayley_orthogonal=False,
        cayley_rank=1,
        use_pom_vlift=False,
        pom_order=2,
        pom_expansion=64,
    )
