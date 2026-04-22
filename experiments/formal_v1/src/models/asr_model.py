"""Top-level ASR model: ConvSubsampling -> Encoder -> CTCHead."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ExperimentConfig
from src.models.components import ConvSubsampling, ConvSubsamplingV2, CTCHead
from src.models.encoder import build_encoder


class ASRModel(nn.Module):
    """CTC-based ASR model with swappable encoder backbone."""

    def __init__(self, vocab_size: int, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        # Frontend selection: explicit cfg flag OR "frontend_v2" in backbone name.
        # Variant: "_matched" suffix → param-neutral control (CB-5b),
        # otherwise lean (CB-5a).
        use_v2 = (
            getattr(cfg, "use_frontend_v2", False)
            or "frontend_v2" in cfg.backbone
        )
        if use_v2:
            # Abort sentinel: touch this file to fail-fast on frontend_v2
            # runs (e.g. while the design is being iterated).  The first
            # CB-5 attempt non-converged at CER ~0.86; halting further runs
            # until the design is re-validated.
            import os
            _abort_sentinel = os.path.expanduser(
                "~/Master_Thesis_Exps/experiments/formal_v1/outputs/.abort_frontend_v2"
            )
            if os.path.exists(_abort_sentinel):
                raise RuntimeError(
                    f"frontend_v2 aborted via sentinel file {_abort_sentinel}. "
                    "Remove the sentinel to re-enable."
                )
            variant = "matched" if "matched" in cfg.backbone else "lean"
            self.frontend = ConvSubsamplingV2(cfg.n_mels, cfg.d_model, variant=variant)
        else:
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
