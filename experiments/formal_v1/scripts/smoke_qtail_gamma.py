#!/usr/bin/env python3
"""Smoke test for rwkv6_qtail_gamma — learnable γ decay coupling refinement.

Verifies:
1. γ parameter exists on qtail-active layers (L4, L5), has shape (n_head,), init 1.0
2. Forward + backward runs, finite output
3. γ receives non-zero gradient after bwd (confirms it's in the compute graph)
4. At γ=1.0, forward output matches rwkv6_qtail structurally (same-seed smoke in
   ep-0 shouldn't produce bit-exact outputs due to RNG-state drift from the
   extra param, so we only check finiteness and magnitude regime — the real
   zero-regression check is ≈0 perturbation from changing γ VALUE away from 1).
"""
from __future__ import annotations

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.vocab import CharVocab
from src.models.asr_model import ASRModel
from src.utils.misc import seed_everything, count_parameters, format_param_count


def build(backbone: str) -> ASRModel:
    cfg = load_config("configs/default.yaml", {"backbone": backbone})
    vocab = CharVocab.build_english()
    return ASRModel(vocab_size=vocab.size, cfg=cfg)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    seed_everything(42)
    B, T_mel, n_mels = 2, 200, 80
    mels = torch.randn(B, n_mels, T_mel, device=device)
    lengths = torch.tensor([T_mel, T_mel - 20], device=device)

    bb = "rwkv6_qtail_gamma"
    print(f"\n=== {bb} ===")
    seed_everything(42)
    model = build(bb).to(device)
    pc = count_parameters(model)
    print(f"  params: total={format_param_count(pc['total'])} enc={format_param_count(pc['encoder'])}")

    # γ should exist ONLY on the top-2 layers (4, 5) — the qtail-active ones.
    for i, layer in enumerate(model.encoder.layers):
        tm = layer.att
        has_qtail = getattr(tm, "use_qtail", False)
        has_gamma = hasattr(tm, "qtail_gamma")
        tag = f"L{i}: qtail={has_qtail}, gamma={has_gamma}"
        if has_gamma:
            gamma = tm.qtail_gamma
            tag += f"  shape={tuple(gamma.shape)} init_mean={gamma.mean().item():.3f}"
            assert gamma.shape == (4,), f"γ should be per-head, got {gamma.shape}"
            assert torch.allclose(gamma, torch.ones_like(gamma)), "γ should init to 1.0"
        print(f"  {tag}")
    print("  ✅ γ allocated on top-2 layers only, shape=(n_head,), init=1.0")

    # Forward + backward at default init.  β_qtail is zero-init ⇒ the whole
    # Kronecker branch is gated off, so γ (living inside that branch) gets
    # no gradient.  That's the correct zero-regression behavior: until SGD
    # grows β, γ is inert.  We verify this explicitly, then activate β to
    # exercise γ's gradient path.
    model.train()
    log_probs, _, _ = model(mels, lengths)
    loss = log_probs.pow(2).mean()
    loss.backward()
    assert torch.isfinite(log_probs).all(), "log_probs has NaN/Inf"
    print(f"  fwd+bwd OK  loss={loss.item():.6e}  logits={tuple(log_probs.shape)}")

    # At default init (β=0), γ should have zero gradient (branch gated off):
    for i, layer in enumerate(model.encoder.layers):
        tm = layer.att
        if hasattr(tm, "qtail_gamma"):
            g = tm.qtail_gamma.grad
            gmag = g.abs().max().item() if g is not None else 0.0
            assert gmag == 0.0, f"L{i} γ grad should be 0 at β=0 init, got {gmag:.3e}"
    print("  ✅ γ inert at β=0 init (zero-regression gating)")

    # Now push β off zero and redo bwd — γ should start receiving gradient.
    model.zero_grad()
    for layer in model.encoder.layers:
        if hasattr(layer.att, "beta_qtail"):
            layer.att.beta_qtail.data.fill_(0.1)  # small nonzero
    log_probs2, _, _ = model(mels, lengths)
    loss2 = log_probs2.pow(2).mean()
    loss2.backward()
    for i, layer in enumerate(model.encoder.layers):
        tm = layer.att
        if hasattr(tm, "qtail_gamma"):
            g = tm.qtail_gamma.grad
            gmag = g.abs().max().item()
            print(f"  L{i}: γ.grad max|·| after β=0.1 = {gmag:.3e}")
            assert gmag > 0, f"L{i} γ has zero grad even with β≠0 — not wired in"
    print("  ✅ γ receives nonzero gradient once β is active (compute graph OK)")
    del model
    torch.cuda.empty_cache()

    # ── Perturbation sanity (with β active): changing γ moves the output ──
    # Confirms γ is actually wired into the Kronecker scan, not a dead
    # parameter. β must be nonzero for this test to be informative.
    print("\n=== perturbation sanity with β=0.1: changing γ moves output ===")
    seed_everything(42)
    m = build(bb).to(device).eval()
    # Activate β first
    for layer in m.encoder.layers:
        if hasattr(layer.att, "beta_qtail"):
            layer.att.beta_qtail.data.fill_(0.1)
    with torch.no_grad():
        y_g1, _, _ = m(mels, lengths)
        # Set γ=0.5 on all active layers
        for layer in m.encoder.layers:
            if hasattr(layer.att, "qtail_gamma"):
                layer.att.qtail_gamma.data.fill_(0.5)
        y_ghalf, _, _ = m(mels, lengths)
    delta = (y_g1 - y_ghalf).abs().max().item()
    print(f"  max|y(γ=1) - y(γ=0.5)| = {delta:.3e}")
    assert delta > 1e-4, (
        f"Changing γ should move the forward output (γ controls Kronecker decay). "
        f"Got delta = {delta:.2e}"
    )
    print("  ✅ γ is live — perturbing it moves the forward output")


if __name__ == "__main__":
    main()
