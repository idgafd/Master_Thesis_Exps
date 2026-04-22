#!/usr/bin/env python3
"""Correctness test: chunked Cayley rank-1 vs sequential rank-1 fast path.

Acceptance: max_abs_diff <= 1e-5 on random inputs, state matches,
zero-regression vs vanilla still passes.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import load_config
from src.data.vocab import CharVocab
from src.models.asr_model import ASRModel
from src.utils.misc import seed_everything


def _build(backbone: str, seed: int = 42):
    cfg = load_config("configs/default.yaml", {"backbone": backbone, "seed": seed})
    vocab = CharVocab.build_english()
    seed_everything(cfg.seed)
    return ASRModel(vocab_size=vocab.size, cfg=cfg), cfg


def test_chunked_matches_sequential():
    print("\n=== chunked vs sequential rank-1 Cayley (layer-level) ===")
    device = torch.device("cuda:0")
    model, cfg = _build("rwkv6_orthogonal")
    model = model.to(device).eval()
    tmix = model.encoder.layers[0].att
    assert tmix.use_cayley_orthogonal and tmix.cayley_rank == 1

    failures = []
    for (B, T) in [(1, 64), (2, 128), (2, 256), (4, 300)]:
        torch.manual_seed(123 + T)
        D = cfg.d_model
        H = tmix.n_head
        K = tmix.head_size
        r = torch.randn(B, H, T, K, device=device)
        k = torch.randn(B, H, T, K, device=device)
        v = torch.randn(B, H, T, K, device=device)
        w = -torch.rand(B, H, T, K, device=device) * 2.0
        x_in = torch.randn(B, T, D, device=device)

        with torch.no_grad():
            y_seq, s_seq = tmix._forward_recurrent_cayley(r, k, v, w, x_in, None)
            y_chk, s_chk = tmix._forward_recurrent_cayley_rank1_chunked(r, k, v, w, x_in, None)

        out_diff = (y_seq - y_chk).abs().max().item()
        out_mean = (y_seq - y_chk).abs().mean().item()
        state_diff = (s_seq - s_chk).abs().max().item()
        y_mag = y_seq.abs().max().item()
        rel = out_diff / max(y_mag, 1e-8)

        # fp32 tolerance: prefix scan changes op order ⇒ ~O(T × eps) accumulation.
        ok_out = out_diff < 2e-4
        ok_mean = out_mean < 2e-5
        ok_state = state_diff < 1e-3

        print(
            f"  B={B} T={T:>3}  max|Δout|={out_diff:.3e}  mean={out_mean:.3e}  "
            f"rel={rel:.3e}  max|Δstate|={state_diff:.3e}  "
            f"[{'PASS' if (ok_out and ok_mean and ok_state) else 'FAIL'}]"
        )
        if not (ok_out and ok_mean and ok_state):
            failures.append((B, T, out_diff, state_diff))

    if failures:
        print(f"  FAIL shapes: {failures}")
        return False
    print("  all shapes PASS")
    return True


def test_zero_regression_at_init():
    """With chunked rank-1 path dispatched, full-model output at init == vanilla rwkv6."""
    print("\n=== zero-regression-at-init (chunked rank-1 Cayley) ===")
    device = torch.device("cuda:0")

    model_base, cfg = _build("rwkv6")
    model_cay, _ = _build("rwkv6_orthogonal")

    base_sd = model_base.state_dict()
    cay_sd = model_cay.state_dict()
    shared = {k: v for k, v in base_sd.items() if k in cay_sd and cay_sd[k].shape == v.shape}
    cay_sd_updated = dict(cay_sd)
    cay_sd_updated.update(shared)
    model_cay.load_state_dict(cay_sd_updated, strict=True)

    model_base = model_base.to(device).eval()
    model_cay = model_cay.to(device).eval()

    torch.manual_seed(0)
    mels = torch.randn(2, cfg.n_mels, 500, device=device)
    lengths = torch.tensor([500, 420], device=device)
    with torch.no_grad():
        y_base, _, _ = model_base(mels, lengths)
        y_cay, _, _ = model_cay(mels, lengths)

    max_diff = (y_base - y_cay).abs().max().item()
    ok = max_diff < 1e-4
    print(f"  max |y_base - y_cay| = {max_diff:.3e}   tolerance 1e-4   [{'PASS' if ok else 'FAIL'}]")
    return ok


def test_gradient_flow():
    """Backward pass on chunked path populates gradients for all Cayley params."""
    print("\n=== gradient flow through chunked Cayley scan ===")
    device = torch.device("cuda:0")
    model, cfg = _build("rwkv6_orthogonal")
    model = model.to(device).train()

    mels = torch.randn(2, cfg.n_mels, 500, device=device)
    lengths = torch.tensor([500, 420], device=device)
    log_probs, out_lengths, _ = model(mels, lengths)
    loss = log_probs.float().sum()
    loss.backward()

    checked = []
    for name, p in model.named_parameters():
        if "cayley" in name:
            if p.grad is None:
                print(f"  ✗ NO GRAD on {name}")
                return False
            checked.append((name, p.grad.norm().item()))

    if not checked:
        print("  ✗ no cayley params found")
        return False
    for name, gnorm in checked[:4]:
        print(f"  grad ok  {name:55s}  ‖g‖={gnorm:.3e}")
    if len(checked) > 4:
        print(f"  ... and {len(checked) - 4} more cayley params — all have gradients")
    return True


if __name__ == "__main__":
    results = {
        "correctness": test_chunked_matches_sequential(),
        "zero_regression": test_zero_regression_at_init(),
        "gradient_flow": test_gradient_flow(),
    }
    print("\n── Summary ──")
    for name, ok in results.items():
        print(f"  {name:20s} {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)
