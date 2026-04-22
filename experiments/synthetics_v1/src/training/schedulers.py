"""Learning-rate scheduler — linear warmup + cosine decay.

Mirrors `formal_v1/src/training/schedulers.py:WarmupCosineScheduler` exactly.
We keep a local copy rather than symlinking the formal_v1 file so that the
synthetics package has no implicit cross-project dependency on the formal_v1
training module (which itself imports `from src.config import ExperimentConfig`).
"""

from __future__ import annotations

import math


class WarmupCosineScheduler:
    """Linear warmup over `warmup_steps`, then cosine decay to `eta_min`."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        scale = self._compute_scale()
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def _compute_scale(self) -> float:
        if self.step_count <= self.warmup_steps:
            return self.step_count / max(self.warmup_steps, 1)
        progress = (self.step_count - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        progress = min(progress, 1.0)
        min_scale = (
            self.eta_min / self.base_lrs[0] if self.base_lrs[0] > 0 else 0.0
        )
        return min_scale + (1.0 - min_scale) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]
