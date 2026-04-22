"""Trivial token vocabulary for synthetic tasks.

We do not need to map characters to ids — MQAR uses raw int tokens directly.
This module exists only so that downstream code has a consistent place to
look up `pad_id`, `vocab_size`, etc.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenVocab:
    vocab_size: int
    pad_id: int = 0  # also used as the MQAR query placeholder

    def __post_init__(self) -> None:
        if self.pad_id < 0 or self.pad_id >= self.vocab_size:
            raise ValueError(
                f"pad_id={self.pad_id} out of range for vocab_size={self.vocab_size}"
            )
