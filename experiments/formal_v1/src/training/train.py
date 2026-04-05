"""Training loop."""

import logging

import torch
import torch.nn as nn

from src.config import ExperimentConfig

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    spec_aug,
    cfg: ExperimentConfig,
    epoch: int,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns average CTC loss."""
    model.train()
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    total_loss = 0.0
    n_batches = 0

    for batch_idx, (mels, targets, mel_lengths, target_lengths) in enumerate(dataloader):
        mels = mels.to(device)
        targets = targets.to(device)
        mel_lengths = mel_lengths.to(device)
        target_lengths = target_lengths.to(device)

        if spec_aug is not None:
            mels = spec_aug(mels)

        log_probs, output_lengths, _ = model(mels, mel_lengths)
        log_probs_ctc = log_probs.permute(1, 0, 2)
        output_lengths = torch.clamp(output_lengths, max=log_probs.size(1))

        loss = ctc_loss_fn(log_probs_ctc, targets, output_lengths, target_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Skipping batch {batch_idx} (loss={loss.item():.4f})")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 50 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    return total_loss / max(n_batches, 1)
