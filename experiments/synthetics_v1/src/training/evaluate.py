"""Evaluation utilities for MQAR.

Two metrics, both restricted to query-target positions (where targets != -100):

- **per_query_acc**: fraction of query positions whose argmax matches the
  target value. This is the literature-standard "MQAR accuracy".
- **per_seq_acc**:   fraction of sequences in which ALL queries are correct.
  This is the stricter Log-Linear / RWKV-7 reporting metric (≥99 % threshold).

Loss reported is the masked cross-entropy (ignore_index=-100).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.tasks.mqar import IGNORE_INDEX


@dataclass
class EvalResult:
    loss: float
    per_query_acc: float
    per_seq_acc: float
    n_examples: int


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
) -> EvalResult:
    """Run `model` over `loader`; return aggregated metrics.

    Assumes `loader` yields `_Batch` objects from `src.data.dataset`.
    """
    model.eval()

    total_loss = 0.0
    total_query_correct = 0
    total_query_count = 0
    total_seq_correct = 0
    total_seq_count = 0

    for batch in loader:
        input_ids = batch.input_ids.to(device, non_blocking=True)
        targets = batch.targets.to(device, non_blocking=True)
        lengths = batch.lengths.to(device, non_blocking=True)

        logits, _ = model(input_ids, lengths)              # (B, T, V)

        # Loss — masked cross-entropy.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        n_query_tokens = (targets != IGNORE_INDEX).sum().item()
        total_loss += loss.item()
        total_query_count += n_query_tokens

        # Accuracy.
        preds = logits.argmax(dim=-1)                       # (B, T)
        is_query = targets != IGNORE_INDEX                  # (B, T)
        correct = (preds == targets) & is_query             # (B, T)
        total_query_correct += correct.sum().item()

        # Per-sequence: a sequence is correct iff ALL its queries are correct.
        # `is_query.sum(1) > 0` masks out any all-padding rows (none in MQAR
        # but defensive).
        per_seq_query_count = is_query.sum(dim=1)
        per_seq_correct = correct.sum(dim=1)
        seq_ok = (per_seq_correct == per_seq_query_count) & (per_seq_query_count > 0)
        total_seq_correct += seq_ok.sum().item()
        total_seq_count += input_ids.size(0)

    model.train()

    avg_loss = total_loss / max(total_query_count, 1)
    return EvalResult(
        loss=avg_loss,
        per_query_acc=total_query_correct / max(total_query_count, 1),
        per_seq_acc=total_seq_correct / max(total_seq_count, 1),
        n_examples=total_seq_count,
    )
