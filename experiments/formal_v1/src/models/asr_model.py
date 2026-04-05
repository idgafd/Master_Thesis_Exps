"""Top-level ASR model: ConvSubsampling -> Encoder -> CTCHead."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ExperimentConfig
from src.models.components import ConvSubsampling, CTCHead
from src.models.encoder import build_encoder


class ASRModel(nn.Module):
    """CTC-based ASR model with swappable encoder backbone."""

    def __init__(self, vocab_size: int, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.frontend = ConvSubsampling(cfg.n_mels, cfg.d_model, cfg.conv_channels)
        self.encoder = build_encoder(cfg)
        self.ctc_head = CTCHead(cfg.d_model, vocab_size)

    @property
    def supports_carry_state(self) -> bool:
        return getattr(self.encoder, "supports_carry_state", False)

    def forward(
        self,
        mels: torch.Tensor,
        mel_lengths: torch.Tensor,
        state: Optional[object] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[object]]:
        """
        mels:        (B, n_mels, T)
        mel_lengths: (B,)
        state:       encoder carry-state (None = stateless)

        Returns: log_probs (B, T', vocab), output_lengths (B,), new_state
        """
        x, lengths = self.frontend(mels, mel_lengths)
        x, new_state = self.encoder(x, lengths, state=state)
        logits = self.ctc_head(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, lengths, new_state
