"""Training loop with MetricLogger integration."""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
import torch.nn as nn

from src.config import ExperimentConfig
from src.training.metrics import MetricLogger

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
    metric_logger: Optional[MetricLogger] = None,
    global_step_offset: int = 0,
    max_steps: Optional[int] = None,
) -> dict:
    """Run one training epoch.

    Returns a dict with:
        train_loss         — mean CTC loss over the epoch
        train_loss_std     — per-batch loss stddev
        grad_norm_mean     — mean post-clip grad norm
        tokens_per_sec     — mean throughput over the epoch
        last_step          — the last global step index (for resume)
    """
    model.train()
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    losses: list[float] = []
    grad_norms: list[float] = []
    throughputs: list[float] = []
    global_step = global_step_offset

    for batch_idx, (mels, targets, mel_lens, tgt_lens) in enumerate(dataloader):
        if max_steps is not None and batch_idx >= max_steps:
            # Truncate to the synced step count under DDP so all ranks
            # finish the epoch with the same number of optimizer steps.
            break
        mels = mels.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mel_lens = mel_lens.to(device, non_blocking=True)
        tgt_lens = tgt_lens.to(device, non_blocking=True)

        if spec_aug is not None:
            mels = spec_aug(mels)

        step_t0 = time.time()

        log_probs, out_lens, _ = model(mels, mel_lens)
        log_probs_ctc = log_probs.permute(1, 0, 2)
        out_lens = torch.clamp(out_lens, max=log_probs.size(1))

        loss = ctc_loss_fn(log_probs_ctc, targets, out_lens, tgt_lens)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Skipping batch {batch_idx} (loss={loss.item():.4f})")
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient norms (raw = pre-clip, clipped = post-clip, same tensor
        # post-clip due to in-place semantics of clip_grad_norm_)
        grad_norm_raw = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        ).item()
        # Post-clip norm equals min(raw, clip). We report raw so the plot
        # shows when clipping fires.
        grad_norm_post = min(grad_norm_raw, cfg.grad_clip)

        optimizer.step()
        scheduler.step()

        step_dt = time.time() - step_t0
        global_step += 1
        losses.append(loss.item())
        grad_norms.append(grad_norm_post)

        # throughput in input tokens (mel frames) per second
        n_tokens = int(mel_lens.sum().item())
        tps = n_tokens / max(step_dt, 1e-6)
        throughputs.append(tps)

        if metric_logger is not None:
            metric_logger.log_step({
                "step": global_step,
                "epoch": epoch,
                "train_loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm_post,
                "grad_norm_raw": grad_norm_raw,
                "tokens_per_sec": tps,
                "batch_duration_sec": step_dt,
                "gpu_mem_gb": _gpu_mem_gb(device),
            })

        if batch_idx % 50 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    n = max(len(losses), 1)
    mean_loss = sum(losses) / n
    var = sum((x - mean_loss) ** 2 for x in losses) / n
    return {
        "train_loss": mean_loss,
        "train_loss_std": var ** 0.5,
        "grad_norm_mean": sum(grad_norms) / max(len(grad_norms), 1),
        "tokens_per_sec": sum(throughputs) / max(len(throughputs), 1),
        "last_step": global_step,
    }


def _gpu_mem_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1e9
