#!/usr/bin/env python3
"""Smoke test for Stage-6 EXPRESSIVENESS-paper backbones on pure causal RWKV-6.

Builds each of (rwkv6, rwkv6_rmsnorm, rwkv6_hadamard_n2, rwkv6_qtail) and runs
a small forward + backward pass to catch shape / nan / param-count issues
before launching full 30-epoch runs.

Zero-regression-at-init check: with β_hadamard == 0 and β_qtail == 0, the
`hadamard_n2` and `qtail` outputs must match `rwkv6_rmsnorm` exactly for a
given input.
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


def forward_backward(model: ASRModel, mels: torch.Tensor, lengths: torch.Tensor):
    model.train()
    log_probs, out_lengths, _ = model(mels, lengths)
    loss = log_probs.pow(2).mean()
    loss.backward()
    return loss.item(), log_probs.shape


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    seed_everything(42)
    B, T_mel, n_mels = 2, 200, 80
    mels = torch.randn(B, n_mels, T_mel, device=device)
    lengths = torch.tensor([T_mel, T_mel - 20], device=device)

    backbones = ["rwkv6", "rwkv6_rmsnorm", "rwkv6_hadamard_n2", "rwkv6_qtail"]
    results = {}
    for bb in backbones:
        print(f"\n=== {bb} ===")
        seed_everything(42)
        model = build(bb).to(device)
        pc = count_parameters(model)
        print(
            f"  params: total={format_param_count(pc['total'])} "
            f"enc={format_param_count(pc['encoder'])}"
        )
        try:
            loss, shape = forward_backward(model, mels, lengths)
            print(f"  fwd+bwd OK  loss={loss:.6e}  logits={tuple(shape)}")
            results[bb] = {"params": pc["total"], "loss": loss, "shape": shape}
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            raise
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Zero-regression-at-init: for fixed seed, the forward pass on
    # hadamard_n2 / qtail should match rmsnorm exactly at β = 0.
    print("\n=== zero-regression-at-init check ===")
    seed_everything(42)
    rms = build("rwkv6_rmsnorm").to(device).eval()
    seed_everything(42)
    had = build("rwkv6_hadamard_n2").to(device).eval()
    seed_everything(42)
    qtl = build("rwkv6_qtail").to(device).eval()

    with torch.no_grad():
        y_rms, _, _ = rms(mels, lengths)
        y_had, _, _ = had(mels, lengths)
        y_qtl, _, _ = qtl(mels, lengths)

    d_had = (y_rms - y_had).abs().max().item()
    d_qtl = (y_rms - y_qtl).abs().max().item()
    print(f"  max|rmsnorm - hadamard_n2| = {d_had:.2e}  (should be ~0)")
    print(f"  max|rmsnorm - qtail|       = {d_qtl:.2e}  (should be ~0)")
    if d_had > 1e-5 or d_qtl > 1e-5:
        raise AssertionError("zero-regression-at-init violated")
    print("  ✅ both quadratic branches reduce exactly to rwkv6_rmsnorm at init")


if __name__ == "__main__":
    main()
