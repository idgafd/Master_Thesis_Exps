"""MQAR (Multi-Query Associative Recall) data generator.

Follows the convention used by Zoology (Arora et al. 2024) and reproduced in
RWKV-7 §7.3 (Peng et al. 2025) and Log-Linear Attention §4.1 (Guo, Yang
et al. 2025).

Sequence layout (one example, length T, K key-value pairs, Q queries):

    [k_1, v_1, k_2, v_2, ..., k_K, v_K, <distractors>, q_1, ?, q_2, ?, ...]

- Keys are sampled without replacement from the key half of the vocabulary.
- Each key has exactly one associated value (sampled from the value half).
- All KV pair tokens are placed in the first 2K positions (paired layout).
- The next (T - 2K - 2Q) positions are distractors — random tokens drawn
  from the union of key and value alphabets.
- The final 2Q positions hold (query_key, target_value_placeholder) pairs.
- The model must predict the value at each placeholder position.

Targets:
- All positions are -100 (ignore_index) EXCEPT the placeholder positions,
  which carry the correct value token id.

This layout is deliberately simple: it exercises associative recall over a
contiguous-then-distracted layout. For a more aggressive variant (queries
interleaved among pairs), see `_layout` in the literature; we hold this for
a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


IGNORE_INDEX = -100


@dataclass(frozen=True)
class MQARSpec:
    """Frozen MQAR configuration for one (T, K, Q) setting."""

    seq_len: int
    n_kv_pairs: int
    n_queries: int
    vocab_size: int = 8192
    key_vocab_size: int = 4096
    value_vocab_size: int = 4096

    def __post_init__(self) -> None:
        if self.key_vocab_size + self.value_vocab_size > self.vocab_size:
            raise ValueError(
                f"key+value alphabets ({self.key_vocab_size}+{self.value_vocab_size}) "
                f"exceed vocab_size ({self.vocab_size})"
            )
        min_seq_len = 2 * self.n_kv_pairs + 2 * self.n_queries
        if self.seq_len < min_seq_len:
            raise ValueError(
                f"seq_len={self.seq_len} too short for "
                f"K={self.n_kv_pairs}, Q={self.n_queries}: needs ≥ {min_seq_len}"
            )
        if self.n_queries > self.n_kv_pairs:
            raise ValueError(
                f"n_queries={self.n_queries} > n_kv_pairs={self.n_kv_pairs}: "
                "every query must reference a presented key"
            )

    @property
    def key_id_offset(self) -> int:
        return 0

    @property
    def value_id_offset(self) -> int:
        return self.key_vocab_size


def generate_mqar_batch(
    batch_size: int,
    spec: MQARSpec,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of MQAR examples.

    Returns:
        input_ids: (B, T) int64 tensor on `device`.
        targets:   (B, T) int64 tensor; -100 everywhere except query placeholders.
    """
    B = batch_size
    T = spec.seq_len
    K = spec.n_kv_pairs
    Q = spec.n_queries
    K_vocab = spec.key_vocab_size
    V_vocab = spec.value_vocab_size
    K_off = spec.key_id_offset
    V_off = spec.value_id_offset

    g = generator
    dev = torch.device(device)

    # ── 1. Sample K distinct key indices per example (no key reused). ────
    # randperm-per-row is the simplest correct way; argsort of random scores
    # is the vectorised equivalent.
    key_scores = torch.rand(B, K_vocab, generator=g, device=dev)
    key_indices = key_scores.argsort(dim=1)[:, :K]              # (B, K)
    key_tokens = key_indices + K_off                             # (B, K)

    # ── 2. Sample one value per key (with replacement across keys). ──────
    value_indices = torch.randint(
        0, V_vocab, (B, K), generator=g, device=dev
    )
    value_tokens = value_indices + V_off                         # (B, K)

    # ── 3. Sample Q query indices (without replacement) into the K pairs. ─
    query_scores = torch.rand(B, K, generator=g, device=dev)
    query_pair_idx = query_scores.argsort(dim=1)[:, :Q]          # (B, Q)
    query_keys = torch.gather(key_tokens, 1, query_pair_idx)     # (B, Q)
    query_targets = torch.gather(value_tokens, 1, query_pair_idx)  # (B, Q)

    # ── 4. Sample distractor tokens for the middle filler region. ────────
    n_distractors = T - 2 * K - 2 * Q
    if n_distractors > 0:
        # Distractors drawn uniformly from the joint key+value alphabet.
        distractor_tokens = torch.randint(
            0, K_vocab + V_vocab, (B, n_distractors), generator=g, device=dev
        )
    else:
        distractor_tokens = torch.empty(B, 0, dtype=torch.long, device=dev)

    # ── 5. Assemble the input sequence. ──────────────────────────────────
    # Layout: [k1 v1 k2 v2 ... kK vK | distractors | q1 0 q2 0 ... qQ 0]
    pair_block = torch.empty(B, 2 * K, dtype=torch.long, device=dev)
    pair_block[:, 0::2] = key_tokens
    pair_block[:, 1::2] = value_tokens

    # Query placeholders use token id 0 (PAD); they are not scored as input,
    # the model only needs to PREDICT the value token at the position AFTER
    # each query key. We follow the Zoology convention of putting a marker
    # token (here: 0) right after each query key as the prediction position.
    query_block = torch.empty(B, 2 * Q, dtype=torch.long, device=dev)
    query_block[:, 0::2] = query_keys
    query_block[:, 1::2] = 0  # placeholder; loss is computed at this index

    input_ids = torch.cat([pair_block, distractor_tokens, query_block], dim=1)
    assert input_ids.shape == (B, T), \
        f"layout bug: got {input_ids.shape}, expected ({B}, {T})"

    # ── 6. Build targets — -100 everywhere except at placeholder positions. ─
    targets = torch.full((B, T), IGNORE_INDEX, dtype=torch.long, device=dev)
    placeholder_offset = 2 * K + n_distractors  # absolute index of 1st placeholder
    placeholder_indices = placeholder_offset + 1 + 2 * torch.arange(
        Q, device=dev
    )                                               # 1, 3, 5, ... within query_block
    targets[:, placeholder_indices] = query_targets

    return input_ids, targets


def make_eval_set(
    n_examples: int,
    spec: MQARSpec,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise a fixed eval set deterministically from `seed`.

    Generated in chunks of 4096 to keep peak memory reasonable.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    chunks_in: list[torch.Tensor] = []
    chunks_tgt: list[torch.Tensor] = []
    remaining = n_examples
    chunk = 4096
    while remaining > 0:
        b = min(chunk, remaining)
        ids, tgt = generate_mqar_batch(b, spec, generator=g, device=device)
        chunks_in.append(ids)
        chunks_tgt.append(tgt)
        remaining -= b
    return torch.cat(chunks_in, dim=0), torch.cat(chunks_tgt, dim=0)
