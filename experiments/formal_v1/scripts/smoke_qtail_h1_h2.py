#!/usr/bin/env python3
"""Smoke test for H1 (per-pair β) and H2 (γ=0 init) qtail variants.

For each new backbone, verifies:

  1. Build OK and the new params live where they should (qtail-active layers).
  2. Param shapes / inits match the contract.
  3. Forward output is BIT-EXACT identical to the matching baseline backbone
     at default init (since β_qtail = 0 gates the whole Kronecker branch off
     for both variants — outer scalar β still controls everything).
  4. After perturbing β_qtail off zero, the new params receive nonzero gradient
     (i.e. they are wired into the compute graph, not dead).

Baselines:
  H1 (rwkv6_qtail_lowrank_all_betapp_convshift_multidil_symmetric_v2)
    matches rwkv6_qtail_lowrank_all_convshift_multidil_symmetric_v2 at init.
  H2 (rwkv6_qtail_lowrank_all_gamma0_convshift_multidil_symmetric_v2)
    matches the same baseline at init (β=0 makes γ irrelevant).
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

    BASELINE = "rwkv6_qtail_lowrank_all_convshift_multidil_symmetric_v2"
    H1 = "rwkv6_qtail_lowrank_all_betapp_convshift_multidil_symmetric_v2"
    H2 = "rwkv6_qtail_lowrank_all_gamma0_convshift_multidil_symmetric_v2"

    seed_everything(42)
    B, T_mel, n_mels = 2, 200, 80
    mels = torch.randn(B, n_mels, T_mel, device=device)
    lengths = torch.tensor([T_mel, T_mel - 20], device=device)

    # ── reference forward ────────────────────────────────────────────────
    seed_everything(42)
    base = build(BASELINE).to(device).eval()
    pc = count_parameters(base)
    print(f"\n[{BASELINE}]  total={format_param_count(pc['total'])}")
    with torch.no_grad():
        y_base, _, _ = base(mels, lengths)
    del base
    torch.cuda.empty_cache()

    # ── H1 ───────────────────────────────────────────────────────────────
    print(f"\n=== H1: {H1} ===")
    seed_everything(42)
    m1 = build(H1).to(device)
    pc = count_parameters(m1)
    print(f"  params: total={format_param_count(pc['total'])} enc={format_param_count(pc['encoder'])}")

    # All 6 layers must have qtail (lowrank_all) AND beta_qtail_per_pair.
    H = m1.encoder.layers[0].att.n_head
    Kp = m1.encoder.layers[0].att.qtail_lr_rank
    K2_pp = Kp * Kp
    for i, layer in enumerate(m1.encoder.layers):
        tm = layer.att
        assert getattr(tm, "use_qtail", False), f"L{i} qtail off"
        assert hasattr(tm, "beta_qtail_per_pair"), f"L{i} missing beta_qtail_per_pair"
        bp = tm.beta_qtail_per_pair
        assert bp.shape == (H, K2_pp), f"L{i} β_pp shape {tuple(bp.shape)} != ({H},{K2_pp})"
        assert torch.all(bp == 1.0), f"L{i} β_pp not init 1.0"
        assert torch.all(tm.beta_qtail == 0.0), f"L{i} β_qtail not zero-init"
        assert not getattr(tm, "use_qtail_gamma", False), f"L{i} γ unexpectedly on"
    print(f"  ✅ β_pp on all {len(m1.encoder.layers)} layers, shape ({H},{K2_pp})=({H},{Kp**2}), init 1.0")

    # Bit-exact equality with baseline at default init (β=0 gates branch off).
    m1.eval()
    with torch.no_grad():
        y1, _, _ = m1(mels, lengths)
    delta = (y_base - y1).abs().max().item()
    print(f"  max|y_base − y_H1| at init = {delta:.3e}")
    assert delta == 0.0, f"H1 not bit-exact with baseline at init: max|Δ|={delta:.3e}"
    print("  ✅ bit-exact reduction to baseline at β=0")

    # Gradient flow through β_pp once β_qtail is nonzero.
    m1.train()
    for layer in m1.encoder.layers:
        layer.att.beta_qtail.data.fill_(0.1)
    m1.zero_grad()
    log_probs, _, _ = m1(mels, lengths)
    loss = log_probs.pow(2).mean()
    loss.backward()
    for i, layer in enumerate(m1.encoder.layers):
        bp = layer.att.beta_qtail_per_pair
        gmag = bp.grad.abs().max().item() if bp.grad is not None else 0.0
        assert gmag > 0, f"L{i} β_pp.grad = 0 even with β_qtail=0.1"
    print("  ✅ β_pp receives nonzero gradient after β_qtail perturbed")
    del m1
    torch.cuda.empty_cache()

    # ── H2 ───────────────────────────────────────────────────────────────
    print(f"\n=== H2: {H2} ===")
    seed_everything(42)
    m2 = build(H2).to(device)
    pc = count_parameters(m2)
    print(f"  params: total={format_param_count(pc['total'])} enc={format_param_count(pc['encoder'])}")

    for i, layer in enumerate(m2.encoder.layers):
        tm = layer.att
        assert getattr(tm, "use_qtail", False), f"L{i} qtail off"
        assert getattr(tm, "use_qtail_gamma", False), f"L{i} γ off"
        gamma = tm.qtail_gamma
        assert gamma.shape == (H,), f"L{i} γ shape {tuple(gamma.shape)}"
        assert torch.all(gamma == 0.0), f"L{i} γ not init 0.0 (got {gamma.tolist()})"
        assert torch.all(tm.beta_qtail == 0.0), f"L{i} β_qtail not zero-init"
        assert not getattr(tm, "use_qtail_beta_per_pair", False), f"L{i} β_pp unexpectedly on"
    print(f"  ✅ γ allocated on all {len(m2.encoder.layers)} layers, shape ({H},), init 0.0")

    m2.eval()
    with torch.no_grad():
        y2, _, _ = m2(mels, lengths)
    delta = (y_base - y2).abs().max().item()
    print(f"  max|y_base − y_H2| at init = {delta:.3e}")
    assert delta == 0.0, f"H2 not bit-exact with baseline at init: max|Δ|={delta:.3e}"
    print("  ✅ bit-exact reduction to baseline at β=0 (γ irrelevant when β=0)")

    # Gradient flow through γ once β_qtail is nonzero.
    m2.train()
    for layer in m2.encoder.layers:
        layer.att.beta_qtail.data.fill_(0.1)
    m2.zero_grad()
    log_probs, _, _ = m2(mels, lengths)
    loss = log_probs.pow(2).mean()
    loss.backward()
    for i, layer in enumerate(m2.encoder.layers):
        g = layer.att.qtail_gamma.grad
        gmag = g.abs().max().item() if g is not None else 0.0
        assert gmag > 0, f"L{i} γ.grad = 0 even with β_qtail=0.1"
    print("  ✅ γ receives nonzero gradient after β_qtail perturbed")

    print("\n=== all H1/H2 smoke checks pass ===")


if __name__ == "__main__":
    main()
