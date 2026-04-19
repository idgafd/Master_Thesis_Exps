"""Smoke + consistency tests for Mamba2Block and Mamba2Encoder."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.models.mamba2_block import Mamba2Block
from src.models.mamba2_encoder import Mamba2Encoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maxerr(a, b):
    return (a - b).abs().max().item()


def test_block_forward_all_modes():
    torch.manual_seed(0)
    x = torch.randn(2, 128, 256, device=DEVICE)
    for mode in ["recurrent", "lion", "lion_chunk"]:
        blk = Mamba2Block(
            d_model=256, d_state=64, d_conv=4, headdim=64,
            expand=2, ngroups=1, chunk_size=64, mode=mode,
        ).to(DEVICE)
        y, state = blk(x, state=None)
        assert y.shape == (2, 128, 256), f"{mode} wrong shape: {y.shape}"
        assert torch.isfinite(y).all(), f"{mode} produced NaN/inf"
        print(f"  ✓ mode={mode:<11s}  y=(2,128,256)  mean={y.mean().item():+.3f}"
              f"  std={y.std().item():.3f}")


def test_block_step_matches_forward():
    """In recurrent mode, token-by-token step() must match a single full forward."""
    torch.manual_seed(1)
    L = 32
    x = torch.randn(2, L, 256, device=DEVICE)
    blk = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64,
        expand=2, ngroups=1, chunk_size=16, mode="recurrent",
    ).to(DEVICE)
    blk.eval()

    with torch.no_grad():
        y_full, _ = blk(x, state=None)

        state = blk.init_state(2, DEVICE)
        conv_s = state["conv"]
        ssm_s = state["ssm"]
        y_step_list = []
        for t in range(L):
            y_t, conv_s, ssm_s = blk.step(x[:, t:t + 1], conv_s, ssm_s)
            y_step_list.append(y_t)
        y_step = torch.cat(y_step_list, dim=1)

    err = _maxerr(y_step, y_full)
    rel = err / (y_full.abs().max().item() + 1e-8)
    print(f"  step vs forward:  max |Δ| = {err:.2e}  (rel {rel:.1e})")
    assert rel < 1e-3, f"step/forward mismatch: rel={rel}"


def test_block_carry_state_consistency():
    """Split sequence in two; carry state across.  Must equal single-shot forward."""
    torch.manual_seed(2)
    L = 128
    x = torch.randn(2, L, 256, device=DEVICE)
    blk = Mamba2Block(
        d_model=256, d_state=64, d_conv=4, headdim=64,
        expand=2, ngroups=1, chunk_size=32, mode="recurrent",
    ).to(DEVICE).eval()

    with torch.no_grad():
        y_full, _ = blk(x, state=None)

        state = blk.init_state(2, DEVICE)
        y1, state = blk(x[:, :L // 2], state=state)
        y2, _ = blk(x[:, L // 2:], state=state)
        y_cat = torch.cat([y1, y2], dim=1)

    err = _maxerr(y_cat, y_full)
    rel = err / (y_full.abs().max().item() + 1e-8)
    print(f"  split-chunk carry vs full:  max |Δ| = {err:.2e}  (rel {rel:.1e})")
    assert rel < 5e-3, f"carry-state mismatch: rel={rel}"


def test_lion_modes_agree():
    """lion (full) and lion_chunk must produce the same output."""
    torch.manual_seed(3)
    x = torch.randn(2, 128, 256, device=DEVICE)
    blk_f = Mamba2Block(d_model=256, d_state=64, headdim=64, ngroups=1,
                        mode="lion").to(DEVICE).eval()
    blk_c = Mamba2Block(d_model=256, d_state=64, headdim=64, ngroups=1,
                        mode="lion_chunk", chunk_size=32).to(DEVICE).eval()
    # Share weights so outputs can be compared.
    blk_c.load_state_dict(blk_f.state_dict())
    with torch.no_grad():
        y_full, _ = blk_f(x)
        y_chunk, _ = blk_c(x)
    err = _maxerr(y_full, y_chunk) / (y_full.abs().max().item() + 1e-8)
    print(f"  lion vs lion_chunk:  rel Δ = {err:.1e}")
    assert err < 1e-3


def test_encoder_shapes():
    """Encoder works end-to-end in every mode."""
    torch.manual_seed(4)
    x = torch.randn(2, 128, 256, device=DEVICE)
    lengths = torch.tensor([128, 96], device=DEVICE)
    for mode in ["recurrent", "lion", "lion_chunk"]:
        enc = Mamba2Encoder(
            d_model=256, n_layers=2, dropout=0.1, ffn_dim=896,
            d_state=64, headdim=64, ngroups=1, chunk_size=32, mode=mode,
        ).to(DEVICE).train()
        y, _ = enc(x, lengths)
        assert y.shape == (2, 128, 256)
        # Masked positions should be zero.
        masked_zero = y[1, 96:].abs().max().item()
        assert masked_zero < 1e-5, f"{mode} masked tail not zero: {masked_zero}"
        print(f"  ✓ encoder mode={mode:<11s}  "
              f"params={sum(p.numel() for p in enc.parameters())/1e6:.2f}M")


def test_bidir_no_param_doubling():
    """A bidirectional encoder must have the same parameter count as unidirectional."""
    enc_uni = Mamba2Encoder(d_model=256, n_layers=6, dropout=0.1, ffn_dim=896,
                            mode="recurrent").to(DEVICE)
    enc_lion = Mamba2Encoder(d_model=256, n_layers=6, dropout=0.1, ffn_dim=896,
                             mode="lion").to(DEVICE)
    n_uni = sum(p.numel() for p in enc_uni.parameters())
    n_lion = sum(p.numel() for p in enc_lion.parameters())
    print(f"  params: causal={n_uni/1e6:.3f}M  lion-bidir={n_lion/1e6:.3f}M")
    assert n_uni == n_lion, (
        f"bidirectional encoder should have identical param count, "
        f"got {n_uni} vs {n_lion}"
    )


if __name__ == "__main__":
    tests = [
        test_block_forward_all_modes,
        test_block_step_matches_forward,
        test_block_carry_state_consistency,
        test_lion_modes_agree,
        test_encoder_shapes,
        test_bidir_no_param_doubling,
    ]
    n_pass = 0
    for t in tests:
        name = t.__name__
        print(f"· {name}")
        try:
            t()
            n_pass += 1
        except AssertionError as e:
            print(f"    FAIL: {e}")
        except Exception as e:
            print(f"    {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{n_pass}/{len(tests)} passed")
