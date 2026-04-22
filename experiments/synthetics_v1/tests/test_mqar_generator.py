"""Correctness tests for the MQAR generator.

Run with:
    cd experiments/synthetics_v1
    uv run python -m pytest tests/test_mqar_generator.py -v
"""

from __future__ import annotations

import torch

from src.tasks.mqar import IGNORE_INDEX, MQARSpec, generate_mqar_batch, make_eval_set


def _spec(T=64, K=16, Q=None, V=8192) -> MQARSpec:
    return MQARSpec(
        seq_len=T,
        n_kv_pairs=K,
        n_queries=Q if Q is not None else K,
        vocab_size=V,
        key_vocab_size=V // 2,
        value_vocab_size=V // 2,
    )


def test_shape_and_dtype():
    spec = _spec(T=64, K=16)
    g = torch.Generator().manual_seed(0)
    ids, tgt = generate_mqar_batch(8, spec, generator=g)
    assert ids.shape == (8, 64)
    assert tgt.shape == (8, 64)
    assert ids.dtype == torch.long
    assert tgt.dtype == torch.long


def test_targets_only_at_query_positions():
    spec = _spec(T=64, K=16, Q=16)
    g = torch.Generator().manual_seed(0)
    ids, tgt = generate_mqar_batch(32, spec, generator=g)
    n_queries_per_row = (tgt != IGNORE_INDEX).sum(dim=1)
    assert torch.all(n_queries_per_row == 16), \
        f"expected 16 query targets per row, got {n_queries_per_row.tolist()}"


def test_query_targets_match_presented_pairs():
    """Every query position must demand a value that was actually presented
    earlier in the same sequence, paired with the matching key.

    Under the Zoology-style layout the query key sits AT the target position
    (model predicts v_q as the next token, standard autoregressive LM).
    """
    spec = _spec(T=128, K=32)
    g = torch.Generator().manual_seed(0)
    ids, tgt = generate_mqar_batch(16, spec, generator=g)

    K_off = spec.key_id_offset
    V_off = spec.value_id_offset

    for b in range(ids.size(0)):
        # Reconstruct the KV pair table from the first 2K positions.
        kv = {}
        for i in range(spec.n_kv_pairs):
            k = ids[b, 2 * i].item()
            v = ids[b, 2 * i + 1].item()
            assert K_off <= k < K_off + spec.key_vocab_size, \
                f"key at pos {2*i} out of range: {k}"
            assert V_off <= v < V_off + spec.value_vocab_size, \
                f"value at pos {2*i+1} out of range: {v}"
            assert k not in kv, f"duplicate key in pairs: {k}"
            kv[k] = v

        # In the new layout, the query key is the INPUT token at the target
        # position, and the value appears as the NEXT input token (= LM target).
        query_positions = (tgt[b] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        for qp in query_positions.tolist():
            query_key = ids[b, qp].item()
            target_val = tgt[b, qp].item()
            # Sanity: the value also appears as the next input token.
            assert ids[b, qp + 1].item() == target_val, \
                f"next-token contract broken at pos {qp}"
            assert query_key in kv, \
                f"query at pos {qp} references key {query_key} not in pair table"
            assert kv[query_key] == target_val, \
                f"query target mismatch at pos {qp}: " \
                f"key={query_key} expected_v={kv[query_key]} got={target_val}"


def test_keys_are_distinct():
    """No duplicate keys across the K presented pairs."""
    spec = _spec(T=256, K=64)
    g = torch.Generator().manual_seed(0)
    ids, _ = generate_mqar_batch(64, spec, generator=g)
    pair_keys = ids[:, 0::2][:, : spec.n_kv_pairs]   # (B, K)
    for b in range(ids.size(0)):
        assert pair_keys[b].unique().numel() == spec.n_kv_pairs, \
            "duplicate keys in presented pairs"


def test_reproducible_under_seed():
    spec = _spec(T=128, K=32)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a1, t1 = generate_mqar_batch(16, spec, generator=g1)
    a2, t2 = generate_mqar_batch(16, spec, generator=g2)
    assert torch.equal(a1, a2)
    assert torch.equal(t1, t2)


def test_different_seeds_diverge():
    spec = _spec(T=128, K=32)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(1)
    a1, _ = generate_mqar_batch(8, spec, generator=g1)
    a2, _ = generate_mqar_batch(8, spec, generator=g2)
    assert not torch.equal(a1, a2)


def test_eval_set_materialises_to_n():
    spec = _spec(T=64, K=16)
    ids, tgt = make_eval_set(1000, spec, seed=0)
    assert ids.shape == (1000, 64)
    assert tgt.shape == (1000, 64)
    # And reproducible:
    ids2, tgt2 = make_eval_set(1000, spec, seed=0)
    assert torch.equal(ids, ids2)
    assert torch.equal(tgt, tgt2)


def test_chance_baseline_is_low():
    """Sanity: random argmax over the value alphabet should score ≈ 1/V_v."""
    spec = _spec(T=64, K=16)
    ids, tgt = generate_mqar_batch(256, spec, generator=torch.Generator().manual_seed(0))
    # Random predictions over the value sub-alphabet:
    rand_preds = torch.randint(spec.value_id_offset,
                               spec.value_id_offset + spec.value_vocab_size,
                               tgt.shape)
    is_query = tgt != IGNORE_INDEX
    correct = (rand_preds == tgt) & is_query
    acc = correct.sum().item() / max(is_query.sum().item(), 1)
    # Expected: 1/4096 ≈ 0.00024. Allow generous slack — just verify <1 %.
    assert acc < 0.01, f"chance baseline suspiciously high: {acc}"


def test_min_seq_len_validation():
    """Short seq_len with too many pairs+queries should raise."""
    import pytest
    with pytest.raises(ValueError):
        MQARSpec(seq_len=10, n_kv_pairs=8, n_queries=8)
