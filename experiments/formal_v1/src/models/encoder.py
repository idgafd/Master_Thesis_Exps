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

    # All RWKV-6 variants (rwkv6, lion, and all mechanism combinations)
    # Determine mode from backbone name or explicit config
    mode_map = {
        "rwkv6": "recurrent",
        "rwkv6_lucid": "recurrent",
        "rwkv6_delta": "recurrent",
        "rwkv6_lucid_delta": "recurrent",
        "lion": "lion",
        "lion_convshift": "lion",
        "lion_lucid": "lion",
        "lion_lucid_chunked": "lion",
        "lion_delta": "lion",
        "lion_headscale": "lion",
        "lion_convshift_headscale": "lion",
        "lion_temperature": "lion",
        "bidir_serial": "bidir_serial",
    }

    mode = mode_map.get(backbone, cfg.rwkv_mode)

    # Derive mechanism flags from backbone name (override config)
    conv_shift = "convshift" in backbone or cfg.conv_shift
    headscale = "headscale" in backbone or cfg.headscale
    delta_rule = "delta" in backbone or cfg.delta_rule
    lucid = "lucid" in backbone or cfg.lucid
    temperature = "temperature" in backbone or cfg.temperature
    lucid_chunk_size = 64 if "chunked" in backbone else cfg.lucid_chunk_size

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
        temperature=temperature,
    )
