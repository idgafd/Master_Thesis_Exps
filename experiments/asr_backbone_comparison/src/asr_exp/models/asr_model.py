"""Top-level ASR model: ConvSubsampling + Encoder + CTCHead."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr_exp.config import ExperimentConfig
from asr_exp.models.components import ConvSubsampling, CTCHead


class ASRModel(nn.Module):
    """CTC-based ASR model with swappable encoder backbone.

    Architecture: ConvSubsampling (4× downsample) → Encoder → CTCHead.
    All five backbone types share the same frontend and output head.
    """

    BACKBONE_TYPES = {
        "transformer", "linear_attention", "bidir_linear_attention",
        "mamba", "rwkv6", "rwkv7",
        "bidir_rwkv6", "bidir_rwkv6_conv", "bidir_rwkv6_conv_nogate",
        "bidir_rwkv6_cplx_b", "bidir_rwkv6_cplx_c",
        "bidir_rwkv6_cplx_b_cos2", "bidir_rwkv6_cplx_d", "bidir_rwkv6_cplx_d_cos2",
        "bidir_rwkv6_headscale", "bidir_rwkv6_dual", "bidir_rwkv6_gaussian",
        "biwkv6", "biwkv6_no_conv_no_gate",
        "bidir_vim_rwkv6", "bidir_vim_mamba",
    }

    def __init__(self, backbone_type: str, vocab_size: int, cfg: ExperimentConfig):
        super().__init__()
        if backbone_type not in self.BACKBONE_TYPES:
            raise ValueError(
                f"Unknown backbone: {backbone_type!r}. "
                f"Must be one of {sorted(self.BACKBONE_TYPES)}"
            )
        self.backbone_type = backbone_type
        self.frontend = ConvSubsampling(cfg.n_mels, cfg.d_model, cfg.conv_channels)
        self.encoder = self._build_encoder(backbone_type, cfg)
        self.ctc_head = CTCHead(cfg.d_model, vocab_size)

    @staticmethod
    def _build_encoder(backbone_type: str, cfg: ExperimentConfig) -> nn.Module:
        ffn_dim = cfg.d_model * cfg.ffn_mult

        if backbone_type == "transformer":
            from asr_exp.models.transformer import TransformerEncoder
            return TransformerEncoder(cfg.d_model, cfg.n_heads, cfg.n_layers, ffn_dim, cfg.dropout)

        if backbone_type == "linear_attention":
            from asr_exp.models.linear_attention import LinearAttentionEncoder
            return LinearAttentionEncoder(cfg.d_model, cfg.n_heads, cfg.n_layers, ffn_dim, cfg.dropout)

        if backbone_type == "bidir_linear_attention":
            from asr_exp.models.bidir_linear_attention import BidirLinearAttentionEncoder
            return BidirLinearAttentionEncoder(cfg.d_model, cfg.n_heads, cfg.n_layers, ffn_dim, cfg.dropout)

        if backbone_type == "mamba":
            from asr_exp.models.mamba import MambaEncoder
            return MambaEncoder(cfg.d_model, cfg.n_layers, cfg.dropout)

        if backbone_type == "rwkv6":
            from asr_exp.models.rwkv6 import RWKV6Encoder
            return RWKV6Encoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size)

        if backbone_type == "rwkv7":
            from asr_exp.models.rwkv7 import RWKV7Encoder
            return RWKV7Encoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size)

        if backbone_type == "bidir_rwkv6":
            from asr_exp.models.bidir_rwkv6 import BidirRWKV6Encoder
            return BidirRWKV6Encoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size)

        if backbone_type == "bidir_rwkv6_conv":
            from asr_exp.models.bidir_rwkv6_conv import BidirRWKV6ConvEncoder
            return BidirRWKV6ConvEncoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size, use_gate=True)

        if backbone_type == "bidir_rwkv6_conv_nogate":
            from asr_exp.models.bidir_rwkv6_conv import BidirRWKV6ConvEncoder
            return BidirRWKV6ConvEncoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size, use_gate=False)

        if backbone_type == "bidir_rwkv6_cplx_b":
            from asr_exp.models.bidir_rwkv6_complex import BidirRWKV6ComplexEncoder
            return BidirRWKV6ComplexEncoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size, theta_init=0.31)

        if backbone_type == "bidir_rwkv6_cplx_c":
            from asr_exp.models.bidir_rwkv6_complex import BidirRWKV6ComplexEncoder
            return BidirRWKV6ComplexEncoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size, theta_init=0.90)

        if backbone_type == "bidir_rwkv6_cplx_b_cos2":
            from asr_exp.models.bidir_rwkv6_complex import BidirRWKV6ComplexEncoder
            return BidirRWKV6ComplexEncoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size, theta_init=0.31, use_cos2=True)

        if backbone_type == "bidir_rwkv6_cplx_d":
            from asr_exp.models.bidir_rwkv6_complex import BidirRWKV6ComplexEncoder
            return BidirRWKV6ComplexEncoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size, theta_init=0.31, learnable_theta=True)

        if backbone_type == "bidir_rwkv6_cplx_d_cos2":
            from asr_exp.models.bidir_rwkv6_complex import BidirRWKV6ComplexEncoder
            return BidirRWKV6ComplexEncoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size, theta_init=0.31, learnable_theta=True, use_cos2=True)

        if backbone_type == "bidir_rwkv6_headscale":
            from asr_exp.models.bidir_rwkv6_multiscale import BidirRWKV6MultiScaleEncoder
            return BidirRWKV6MultiScaleEncoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size, use_dual=False)

        if backbone_type == "bidir_rwkv6_dual":
            from asr_exp.models.bidir_rwkv6_multiscale import BidirRWKV6MultiScaleEncoder
            return BidirRWKV6MultiScaleEncoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size, use_dual=True)

        if backbone_type == "bidir_rwkv6_gaussian":
            from asr_exp.models.bidir_rwkv6_gaussian import BidirRWKV6GaussianEncoder
            return BidirRWKV6GaussianEncoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size)

        if backbone_type == "biwkv6":
            from asr_exp.models.biwkv6 import BiWKV6Encoder
            return BiWKV6Encoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size, use_conv_shift=True, use_gate=True)

        if backbone_type == "biwkv6_no_conv_no_gate":
            from asr_exp.models.biwkv6 import BiWKV6Encoder
            return BiWKV6Encoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size, use_conv_shift=False, use_gate=False)

        if backbone_type == "bidir_vim_rwkv6":
            from asr_exp.models.bidir_vim_rwkv6 import BidirVimRWKV6Encoder
            return BidirVimRWKV6Encoder(cfg.d_model, cfg.n_layers, cfg.dropout, head_size=cfg.head_size)

        if backbone_type == "bidir_vim_mamba":
            from asr_exp.models.bidir_vim_mamba import BidirVimMambaEncoder
            return BidirVimMambaEncoder(cfg.d_model, cfg.n_layers, cfg.dropout)

        raise ValueError(backbone_type)  # unreachable

    @property
    def supports_carry_state(self) -> bool:
        return self.encoder.supports_carry_state

    def forward(
        self,
        mels: torch.Tensor,
        mel_lengths: torch.Tensor,
        state=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List]]:
        """
        mels:        (B, n_mels, T)
        mel_lengths: (B,)
        state:       encoder carry-state (None → stateless / reset mode)

        Returns: log_probs (B, T', vocab_size), output_lengths (B,), new_state
        """
        x, lengths = self.frontend(mels, mel_lengths)
        x, new_state = self.encoder(x, lengths, state=state)
        logits = self.ctc_head(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, lengths, new_state
