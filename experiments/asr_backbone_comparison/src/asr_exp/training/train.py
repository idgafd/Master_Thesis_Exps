"""Training loop and learning-rate scheduler."""

import torch
import torch.nn as nn

from asr_exp.config import ExperimentConfig


class WarmupScheduler:
    """Linear warmup followed by an arbitrary base scheduler."""

    def __init__(self, optimizer, scheduler, warmup_steps: int):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self) -> None:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / self.warmup_steps
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * scale
        else:
            self.scheduler.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler: WarmupScheduler,
    spec_aug,
    cfg: ExperimentConfig,
    epoch: int,
    device: torch.device,
) -> float:
    """Run one training epoch.

    Returns: average CTC loss over all batches.
    """
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
        log_probs_ctc = log_probs.permute(1, 0, 2)  # (T, B, V)
        output_lengths = torch.clamp(output_lengths, max=log_probs.size(1))

        loss = ctc_loss_fn(log_probs_ctc, targets, output_lengths, target_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  !!! Skipping batch {batch_idx} (loss={loss.item():.4f})")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 50 == 0:
            print(
                f"  Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    return total_loss / max(n_batches, 1)
