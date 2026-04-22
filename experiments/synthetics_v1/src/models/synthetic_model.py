"""Top-level synthetics model: TokenEmbedding → Encoder → LMHead.

Mirrors the structure of `formal_v1.src.models.asr_model.ASRModel`, but with
the audio frontend replaced by a token embedding and the CTC head replaced
by an LM-style projection.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.config import SyntheticsConfig
from src.models.encoder import build_encoder


class LMHead(nn.Module):
    """LayerNorm + Linear projection to vocabulary logits."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class SyntheticModel(nn.Module):
    """nn.Embedding → encoder → LMHead.

    The encoder is built from `formal_v1`'s leaf backbones via our slim
    `src.models.encoder.build_encoder` dispatcher. Every encoder honours the
    `(B, T, D), lengths -> (B, T, D), state` contract so this wrapper is
    backbone-agnostic.
    """

    def __init__(self, cfg: SyntheticsConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.encoder = build_encoder(cfg)
        self.lm_head = LMHead(cfg.d_model, cfg.vocab_size)

        # Same init scale convention as formal_v1's CTCHead path: small
        # embedding init keeps the encoder's residual stream well-scaled.
        nn.init.normal_(self.embed.weight, mean=0.0, std=cfg.d_model ** -0.5)

    @property
    def supports_carry_state(self) -> bool:
        return getattr(self.encoder, "supports_carry_state", False)

    def forward(
        self,
        input_ids: torch.Tensor,                      # (B, T) int64
        lengths: torch.Tensor,                        # (B,) int64
        state: Optional[object] = None,
    ) -> Tuple[torch.Tensor, Optional[object]]:
        """Returns: logits (B, T, V), new_state."""
        x = self.embed(input_ids)                     # (B, T, D)
        x, new_state = self.encoder(x, lengths, state=state)
        logits = self.lm_head(x)                      # (B, T, V)
        return logits, new_state
