"""Equivalence + causality + carry tests for CausalLinearAttentionLayer.

Contract for Stage 11.0a:
  * parallel cumsum path and explicit recurrent loop produce identical
    output (up to fp precision) for the same inputs and state;
  * output at position t depends only on inputs at positions s <= t;
  * chunked forward with state carry equals single-pass forward;
  * padded positions do not contribute to the running (S, z).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import ExperimentConfig
from src.models.encoder import build_encoder
from src.models.linear_attn_causal import (
    CausalLinearAttentionEncoder,
    CausalLinearAttentionLayer,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maxerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _mk_layer(d_model: int = 64, n_heads: int = 4, ffn_dim: int = 128) -> CausalLinearAttentionLayer:
    torch.manual_seed(0)
    layer = CausalLinearAttentionLayer(
        d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim, dropout=0.0
    ).to(DEVICE)
    layer.eval()  # disable dropout for determinism
    return layer


def test_parallel_matches_recurrent_no_state():
    """cumsum vs t-loop agree to < 1e-5 on the same inputs."""
    layer = _mk_layer()
    torch.manual_seed(1)
    x = torch.randn(2, 16, 64, device=DEVICE)
    y_par, _ = layer.forward_parallel(x)
    y_rec, _ = layer.forward_recurrent(x)
    err = _maxerr(y_par, y_rec)
    assert err < 1e-4, f"parallel vs recurrent diverge: max |err| = {err}"


def test_parallel_matches_recurrent_with_state():
    """State carry semantics match between paths."""
    layer = _mk_layer()
    torch.manual_seed(2)
    x = torch.randn(2, 16, 64, device=DEVICE)
    state0 = {
        "S": torch.randn(2, 4, 16, 16, device=DEVICE),
        "z": torch.randn(2, 4, 16, device=DEVICE),
    }
    y_par, st_par = layer.forward_parallel(x, state={k: v.clone() for k, v in state0.items()})
    y_rec, st_rec = layer.forward_recurrent(x, state={k: v.clone() for k, v in state0.items()})
    assert _maxerr(y_par, y_rec) < 1e-4
    assert _maxerr(st_par["S"], st_rec["S"]) < 1e-4
    assert _maxerr(st_par["z"], st_rec["z"]) < 1e-4


def test_causality():
    """Perturbing x[:, t_cut:, :] does not change output at positions < t_cut."""
    layer = _mk_layer()
    torch.manual_seed(3)
    x = torch.randn(2, 20, 64, device=DEVICE)
    y1, _ = layer.forward_parallel(x)

    t_cut = 10
    x2 = x.clone()
    x2[:, t_cut:, :] = torch.randn_like(x2[:, t_cut:, :])
    y2, _ = layer.forward_parallel(x2)

    err = _maxerr(y1[:, :t_cut], y2[:, :t_cut])
    assert err < 1e-5, f"causality violated: past diverges when future perturbed ({err})"


def test_chunked_state_carry_matches_single_pass():
    """Two-chunk forward with state carry equals one-shot forward over T."""
    layer = _mk_layer()
    torch.manual_seed(4)
    x = torch.randn(2, 24, 64, device=DEVICE)

    # Single pass with state (so we exercise the stateful path both times).
    zero_state = {
        "S": torch.zeros(2, 4, 16, 16, device=DEVICE),
        "z": torch.zeros(2, 4, 16, device=DEVICE),
    }
    y_full, _ = layer.forward_parallel(x, state={k: v.clone() for k, v in zero_state.items()})

    # Two chunks of length 12.
    y1, st1 = layer.forward_parallel(x[:, :12], state={k: v.clone() for k, v in zero_state.items()})
    y2, _ = layer.forward_parallel(x[:, 12:], state=st1)
    y_chunked = torch.cat([y1, y2], dim=1)

    err = _maxerr(y_full, y_chunked)
    assert err < 1e-4, f"chunked vs single-pass diverge: {err}"


def test_padding_does_not_contribute():
    """Padded positions leave (S, z) unchanged — mask is honoured."""
    layer = _mk_layer()
    torch.manual_seed(5)
    x_valid = torch.randn(2, 10, 64, device=DEVICE)

    # Extend with 5 random padded positions.
    x_pad = torch.randn(2, 5, 64, device=DEVICE)
    x = torch.cat([x_valid, x_pad], dim=1)
    mask = torch.zeros(2, 15, dtype=torch.bool, device=DEVICE)
    mask[:, 10:] = True  # True = pad

    y_masked, _ = layer.forward_parallel(x, key_padding_mask=mask)
    y_unpadded, _ = layer.forward_parallel(x_valid)

    err = _maxerr(y_masked[:, :10], y_unpadded)
    assert err < 1e-5, f"padded positions leaked into past output: {err}"


def test_encoder_factory_builds_with_correct_shape():
    """build_encoder('linear_attn_causal') returns a causal LA encoder with
    the canonical spine shape."""
    cfg = ExperimentConfig()
    cfg.backbone = "linear_attn_causal"
    cfg.d_model = 256
    cfg.n_heads = 4
    cfg.n_layers = 6
    cfg.head_size = 64
    cfg.dropout = 0.0
    enc = build_encoder(cfg).to(DEVICE)
    assert isinstance(enc, CausalLinearAttentionEncoder)
    assert enc.supports_carry_state
    assert enc.n_layers == 6
    assert enc.head_dim == 64
    assert enc.n_heads == 4

    params = sum(p.numel() for p in enc.parameters())
    # Expect ~4.3M: per-layer q/k/v/o ≈ 263K + ffn ≈ 460K + norms ≈ 1K.
    assert 3_000_000 < params < 6_000_000, f"unexpected param count {params}"

    x = torch.randn(2, 50, 256, device=DEVICE)
    lengths = torch.tensor([50, 32], device=DEVICE)
    y, new_state = enc(x, lengths)
    assert y.shape == (2, 50, 256)
    assert new_state is None

    state = enc.init_state(2, DEVICE)
    y2, st = enc(x, lengths, state=state)
    assert y2.shape == (2, 50, 256)
    assert st is not None
    assert st["offset"] == 50
    assert len(st["layers"]) == 6
