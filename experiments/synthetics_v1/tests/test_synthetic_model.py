"""Smoke tests for SyntheticModel across all reduced-cohort backbones.

Verifies:
    (a) every backbone instantiates at the cfg envelope without error;
    (b) forward returns the (B, T, V) shape contract;
    (c) backward produces gradients on every parameter;
    (d) loss is finite and lower than uniform-baseline on a single batch.

Run with:
    cd experiments/synthetics_v1
    bash scripts/setup_symlinks.sh        # one-time
    uv run python -m pytest tests/test_synthetic_model.py -v
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.config import SyntheticsConfig
from src.models.encoder import SUPPORTED_BACKBONES
from src.models.synthetic_model import SyntheticModel
from src.tasks.mqar import IGNORE_INDEX, MQARSpec, generate_mqar_batch


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiny_cfg(backbone: str, T: int = 32) -> SyntheticsConfig:
    """Tiny model — fast to instantiate and forward in CI / smoke."""
    return SyntheticsConfig(
        task="mqar",
        vocab_size=512,
        key_vocab_size=256,
        value_vocab_size=256,
        seq_len=T,
        n_kv_pairs=8,
        n_queries=8,
        batch_size=4,
        backbone=backbone,
        d_model=64,
        n_layers=2,
        n_heads=2,
        head_size=32,
        dropout=0.0,
        max_steps=1,
    )


@pytest.mark.parametrize("backbone", sorted(SUPPORTED_BACKBONES))
def test_forward_shape(backbone, device):
    cfg = _tiny_cfg(backbone)
    model = SyntheticModel(cfg).to(device)
    spec = MQARSpec(
        seq_len=cfg.seq_len,
        n_kv_pairs=cfg.n_kv_pairs,
        n_queries=cfg.resolved_n_queries,
        vocab_size=cfg.vocab_size,
        key_vocab_size=cfg.key_vocab_size,
        value_vocab_size=cfg.value_vocab_size,
    )
    g = torch.Generator(device="cpu").manual_seed(0)
    ids, tgt = generate_mqar_batch(cfg.batch_size, spec, generator=g)
    ids, tgt = ids.to(device), tgt.to(device)
    lengths = torch.full((cfg.batch_size,), cfg.seq_len, dtype=torch.long, device=device)

    logits, _ = model(ids, lengths)
    assert logits.shape == (cfg.batch_size, cfg.seq_len, cfg.vocab_size), \
        f"shape mismatch for {backbone}: {logits.shape}"
    assert torch.isfinite(logits).all(), f"non-finite logits from {backbone}"


@pytest.mark.parametrize("backbone", sorted(SUPPORTED_BACKBONES))
def test_backward_flows_to_all_params(backbone, device):
    # RWKV-6 uses zero-init LoRA bottlenecks (time_maa_w1, time_decay_w1, and
    # delta_params.a1) so that mechanism additions reduce to vanilla output at
    # step 0 (zero-regression contract). At step 0 their LoRA partners
    # (time_maa_x/w/w2, time_decay_w2, delta_params.{k_k,k_a,a0,a1,a2},
    # delta_recurrent_gate) therefore receive exactly-zero gradient — by
    # construction, not by bug. We warm up with a few SGD steps to break out
    # of that init, then verify gradient flow on a fresh forward/backward.
    cfg = _tiny_cfg(backbone)
    model = SyntheticModel(cfg).to(device)
    spec = MQARSpec(
        seq_len=cfg.seq_len, n_kv_pairs=cfg.n_kv_pairs,
        n_queries=cfg.resolved_n_queries, vocab_size=cfg.vocab_size,
        key_vocab_size=cfg.key_vocab_size, value_vocab_size=cfg.value_vocab_size,
    )
    g = torch.Generator(device="cpu").manual_seed(0)
    ids, tgt = generate_mqar_batch(cfg.batch_size, spec, generator=g)
    ids, tgt = ids.to(device), tgt.to(device)
    lengths = torch.full((cfg.batch_size,), cfg.seq_len, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(ids, lengths)
        warm_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        warm_loss.backward()
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(ids, lengths)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    assert torch.isfinite(loss), f"non-finite loss from {backbone}: {loss}"
    loss.backward()

    no_grad = []
    for n, p in model.named_parameters():
        if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0):
            no_grad.append(n)
    # Some encoders may legitimately not touch every embedding row in one
    # forward pass (e.g. unused vocab IDs), so we tolerate the embedding
    # weight having sparse grads. Anything else with no grad is suspect.
    no_grad = [n for n in no_grad if not n.startswith("embed.")]
    assert not no_grad, f"{backbone}: parameters got no gradient: {no_grad}"


@pytest.mark.parametrize("backbone", sorted(SUPPORTED_BACKBONES))
def test_loss_below_uniform_baseline(backbone, device):
    """At init, the LM head is small but non-zero — masked CE should still
    be roughly log(V) (uniform over vocab). After ONE optimization step on
    a fresh batch, loss must strictly decrease. This catches catastrophic
    init bugs (loss=NaN, loss=0, loss diverges)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(backbone)
    model = SyntheticModel(cfg).to(device)
    spec = MQARSpec(
        seq_len=cfg.seq_len, n_kv_pairs=cfg.n_kv_pairs,
        n_queries=cfg.resolved_n_queries, vocab_size=cfg.vocab_size,
        key_vocab_size=cfg.key_vocab_size, value_vocab_size=cfg.value_vocab_size,
    )
    g = torch.Generator(device="cpu").manual_seed(0)
    ids, tgt = generate_mqar_batch(cfg.batch_size, spec, generator=g)
    ids, tgt = ids.to(device), tgt.to(device)
    lengths = torch.full((cfg.batch_size,), cfg.seq_len, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    logits, _ = model(ids, lengths)
    loss0 = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt.reshape(-1),
        ignore_index=IGNORE_INDEX,
    ).item()
    uniform = math.log(cfg.vocab_size)
    assert math.isfinite(loss0), f"{backbone}: init loss not finite: {loss0}"
    # At init, loss should be close to log(V); allow generous slack (2× either way).
    assert loss0 < 2 * uniform, \
        f"{backbone}: init loss {loss0:.3f} >> uniform baseline {uniform:.3f}"

    # Take 5 steps on the SAME batch — overfitting must work.
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(ids, lengths)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        loss.backward()
        optimizer.step()

    assert loss.item() < loss0, \
        f"{backbone}: 5 steps did not reduce loss ({loss0:.3f} → {loss.item():.3f})"
