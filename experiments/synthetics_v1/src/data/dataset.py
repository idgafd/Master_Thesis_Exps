"""Dataset wrappers around the MQAR generator.

Train data is an `IterableDataset` that regenerates fresh examples each
epoch from a per-epoch seed (cheap — see `COST_ESTIMATE.md §1`).
Eval data is a fixed-seed in-memory tensor pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from src.tasks.mqar import MQARSpec, generate_mqar_batch, make_eval_set


@dataclass
class _Batch:
    input_ids: torch.Tensor   # (B, T) int64
    targets: torch.Tensor     # (B, T) int64, -100 = ignore
    lengths: torch.Tensor     # (B,) int64 — every entry == T (fixed-length task)


def _collate_fixed_length(
    items: list[tuple[torch.Tensor, torch.Tensor]]
) -> _Batch:
    """Stack pre-batched (input_ids, targets) pairs into a single _Batch.

    The MQARTrainDataset already yields full batches, so `items` has length 1.
    """
    assert len(items) == 1, "MQARTrainDataset yields one batch per __next__"
    input_ids, targets = items[0]
    B, T = input_ids.shape
    lengths = torch.full((B,), T, dtype=torch.long)
    return _Batch(input_ids=input_ids, targets=targets, lengths=lengths)


class MQARTrainDataset(IterableDataset):
    """Yields (input_ids, targets) batches from the MQAR generator.

    Each call to `__iter__` re-seeds from `(base_seed, epoch)` so that:
      - within one epoch, the data stream is reproducible;
      - across epochs, the model sees fresh examples.

    `set_epoch(e)` should be called by the training loop before each epoch
    (mirrors the DistributedSampler convention).
    """

    def __init__(
        self,
        spec: MQARSpec,
        steps_per_epoch: int,
        batch_size: int,
        base_seed: int = 0,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.base_seed = base_seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        n_workers = worker_info.num_workers if worker_info is not None else 1

        # Per-worker seed: distinct, reproducible.
        seed = self.base_seed * 1_000_003 + self._epoch * 1_009 + worker_id
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        # Round steps to be divisible across workers so each worker yields
        # the same number of batches.
        my_steps = self.steps_per_epoch // n_workers
        for _ in range(my_steps):
            yield generate_mqar_batch(
                self.batch_size, self.spec, generator=g, device="cpu"
            )


def build_train_loader(
    spec: MQARSpec,
    steps_per_epoch: int,
    batch_size: int,
    base_seed: int,
    num_workers: int = 2,
) -> DataLoader:
    ds = MQARTrainDataset(
        spec, steps_per_epoch, batch_size, base_seed=base_seed
    )
    return DataLoader(
        ds,
        batch_size=None,                 # ds already yields batches
        num_workers=num_workers,
        collate_fn=_collate_fixed_length,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def build_eval_loader(
    spec: MQARSpec,
    n_examples: int,
    batch_size: int,
    seed: int = 0,
) -> DataLoader:
    """Pre-materialised eval loader — same seed every epoch."""
    input_ids, targets = make_eval_set(n_examples, spec, seed=seed, device="cpu")
    ds = TensorDataset(input_ids, targets)

    def _collate(items: list[tuple[torch.Tensor, torch.Tensor]]) -> _Batch:
        ids = torch.stack([it[0] for it in items], dim=0)
        tgt = torch.stack([it[1] for it in items], dim=0)
        B, T = ids.shape
        return _Batch(
            input_ids=ids,
            targets=tgt,
            lengths=torch.full((B,), T, dtype=torch.long),
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
        pin_memory=True,
    )
