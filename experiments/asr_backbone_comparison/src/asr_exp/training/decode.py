"""CTC decoding and error rate computation."""

from typing import List

import torch
from jiwer import wer as compute_wer

from asr_exp.data.vocab import CharVocab


def greedy_ctc_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    vocab: CharVocab,
) -> List[str]:
    """Greedy CTC decoding: argmax → collapse repeats → remove blanks.

    Args:
        log_probs: (B, T, V)
        lengths:   (B,) valid frame counts
        vocab:     CharVocab with blank at index 0

    Returns: list of decoded strings, length B
    """
    predictions = log_probs.argmax(dim=-1)  # (B, T)
    decoded = []
    for i in range(predictions.size(0)):
        seq = predictions[i, : lengths[i]].tolist()
        collapsed: List[int] = []
        prev = -1
        for s in seq:
            if s != prev:
                collapsed.append(s)
            prev = s
        collapsed = [c for c in collapsed if c != 0]  # remove blank
        decoded.append(vocab.decode(collapsed))
    return decoded


def compute_cer(hypotheses: List[str], references: List[str]) -> float:
    """Character error rate computed via word-level WER on character sequences."""
    total_chars = 0
    total_errors = 0
    for hyp, ref in zip(hypotheses, references):
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        if not ref_chars:
            total_errors += len(hyp_chars)
            continue
        r = " ".join(ref_chars)
        h = " ".join(hyp_chars) if hyp_chars else ""
        total_chars += len(ref_chars)
        total_errors += compute_wer(r, h) * len(ref_chars)
    return total_errors / max(total_chars, 1)
