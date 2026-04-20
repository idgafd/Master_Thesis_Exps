#!/usr/bin/env python3
"""Smoke test for Stage 6.5 refinements — R2 and low-rank Kronecker.

Tests two new backbones:
  1. rwkv6_qtail_gamma_dbeta — R2: data-dep β on top of qtail-γ
  2. rwkv6_qtail_lowrank     — project r,k to K'=16, full Kronecker on K'²=256

Both must:
  * Build and pass fwd+bwd cleanly
  * Preserve zero-regression-at-init (β=0 gates the Kronecker branch off)
  * Have the new params initialized as the contract specifies
  * Receive gradients through the new params once β is perturbed non-zero
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

    # ── R2 backbone: qtail-γ + data-dep β ──
    bb = "rwkv6_qtail_gamma_dbeta"
    print(f"\n=== {bb} ===")
    seed_everything(42)
    model = build(bb).to(device)
    pc = count_parameters(model)
    print(f"  params: total={format_param_count(pc['total'])} enc={format_param_count(pc['encoder'])}")

    tm5 = model.encoder.layers[5].att
    assert hasattr(tm5, "beta_qtail"), "beta_qtail missing at L5"
    assert hasattr(tm5, "qtail_gamma"), "qtail_gamma missing at L5"
    assert hasattr(tm5, "beta_qtail_proj"), "beta_qtail_proj missing at L5"
    assert torch.all(tm5.beta_qtail == 0.0)
    assert torch.all(tm5.qtail_gamma == 1.0)
    assert torch.all(tm5.beta_qtail_proj.weight == 0.0)
    assert torch.all(tm5.beta_qtail_proj.bias == 0.0)
    tm0 = model.encoder.layers[0].att
    assert not hasattr(tm0, "beta_qtail_proj"), "dbeta should be gated to qtail-active layers only"
    print("  ✅ R2 params: beta_qtail=0, γ=1, W_β=0, b_β=0, only on qtail-active layers")

    model.train()
    log_probs, _, _ = model(mels, lengths)
    loss = log_probs.pow(2).mean()
    loss.backward()
    assert torch.isfinite(log_probs).all()
    print(f"  fwd+bwd OK  loss={loss.item():.6e}")

    # Zero-regression: beta_qtail_proj output is 0 at init, so even perturbing
    # it would have no effect unless β_static is also nonzero. Let's verify.
    with torch.no_grad():
        y_a, _, _ = model.eval().forward(mels, lengths)
        # Perturb beta_qtail_proj.bias — data-dep β becomes nonzero, should change output
        tm5.beta_qtail_proj.bias.data.fill_(0.5)  # σ-free linear β, nonzero now
        y_b, _, _ = model(mels, lengths)
    delta = (y_a - y_b).abs().max().item()
    print(f"  perturbation sanity: max|Δy| after β_proj.bias=0.5 = {delta:.2e}")
    assert delta > 1e-4, "Setting β_proj.bias nonzero should move the output"
    print("  ✅ data-dep β path is live (perturbation moves output)")
    del model
    torch.cuda.empty_cache()

    # ── low-rank Kronecker ──
    bb = "rwkv6_qtail_lowrank"
    print(f"\n=== {bb} ===")
    seed_everything(42)
    model = build(bb).to(device)
    pc = count_parameters(model)
    print(f"  params: total={format_param_count(pc['total'])} enc={format_param_count(pc['encoder'])}")

    tm5 = model.encoder.layers[5].att
    assert hasattr(tm5, "qtail_lr_proj_r"), "qtail_lr_proj_r missing at L5"
    assert hasattr(tm5, "qtail_lr_proj_k"), "qtail_lr_proj_k missing at L5"
    assert tm5.qtail_lr_proj_r.shape == (tm5.n_head, tm5.head_size, 16), \
        f"qtail_lr_proj_r shape = {tm5.qtail_lr_proj_r.shape}"
    assert tm5.qtail_lr_proj_k.shape == (tm5.n_head, tm5.head_size, 16)
    assert tm5.qtail_lr_rank == 16
    tm0 = model.encoder.layers[0].att
    assert not hasattr(tm0, "qtail_lr_proj_r")
    print(f"  ✅ lowrank params: shapes {tuple(tm5.qtail_lr_proj_r.shape)}, K'=16")

    model.train()
    log_probs, _, _ = model(mels, lengths)
    loss = log_probs.pow(2).mean()
    loss.backward()
    assert torch.isfinite(log_probs).all()
    print(f"  fwd+bwd OK  loss={loss.item():.6e}")

    # Perturb β to check gradient flow through projections
    tm5.beta_qtail.data.fill_(0.1)
    model.zero_grad()
    log_probs, _, _ = model(mels, lengths)
    loss = log_probs.pow(2).mean()
    loss.backward()
    assert tm5.qtail_lr_proj_r.grad is not None and tm5.qtail_lr_proj_r.grad.abs().max() > 0, \
        "qtail_lr_proj_r has no gradient"
    assert tm5.qtail_lr_proj_k.grad is not None and tm5.qtail_lr_proj_k.grad.abs().max() > 0, \
        "qtail_lr_proj_k has no gradient"
    print("  ✅ gradients flow through qtail_lr_proj_r and qtail_lr_proj_k")

    print("\n=== all smoke checks pass ===")


if __name__ == "__main__":
    main()
