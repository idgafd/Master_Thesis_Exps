#!/usr/bin/env python3
"""Stage-7 pre-registration diagnostic on the best causal RSE checkpoint.

Three questions, each a go/no-go gate for Stage 7A (gauge-complete RSE):

    D1  Rotation-budget saturation.
        Does θ saturate at ±clip = ±π/2, or does it sit inside the budget?
        If not saturated → the "strong" budget is not binding, and *adding
        more rotation* is not the mechanism that would help.

    D2  Quadrature amplitude at readout.
        The current readout is `Re(conj(r_c) * S_total)`.  A learnable
        readout phase φ_{h,b} can only recover content that currently
        lives in `Im(conj(r_c) * S_total)`.  If the imaginary part is
        already negligible, Stage 7A has no quadrature content to harvest.

    D3  λ-θ coupling (viscosity engagement).
        Phase-3 viscosity parameterises λ_eff = λ + η·θ².  Is η actually
        large enough at convergence to couple the two?  Is the predicted
        correlation between θ² and λ_eff present in realised data?

Target checkpoint
-----------------

The Stage-5 anchor `rwkv6_rse_strong_viscosity` (dev 0.1185 / test 0.1177)
did not save .pt checkpoints.  We use `rwkv6_p2rse_strong_viscosity`
(dev 0.1190 / test 0.1196, tied within σ) as the mechanism-level proxy.
Both share identical θ/λ/viscosity parameterisation; the p2rse variant
adds a second pole with shared λ.  Mechanism-level statistics of θ, λ,
quadrature amplitude are expected to be invariant between the two.

Output
------

  outputs/stage7_diagnostics/
    rse_diag_summary.json    — all numbers, machine-readable
    rse_diag_theta.png       — per-layer θ histograms (both poles)
    rse_diag_quadrature.png  — per-layer |Im|/|Re| readout histograms
    rse_diag_viscosity.png   — η values + θ²-vs-λ scatter per layer

Run:
  cd experiments/formal_v1
  uv run python scripts/stage7_rse_diagnostics.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config
from src.data.vocab import CharVocab
from src.data.dataset import ASRDataset, DurationBatchSampler, collate_fn
from src.data.librispeech import load_librispeech
from src.models.asr_model import ASRModel
from src.training.checkpoint import load_checkpoint
from torch.utils.data import DataLoader


DEFAULT_RUN_DIR = ROOT / "outputs" / "p2rse_sv_rwkv6_p2rse_strong_viscosity_seed42"

THETA_CLIP = math.pi / 2        # strong budget
N_UTTERANCES_DEFAULT = 120      # enough for stable tail quantiles, cheap
SATURATION_THRESHOLD = 0.95     # |θ|/clip > this counts as "saturated"


# ─────────────────────────────────────────────────────────────────────────
# Patch helpers — capture per-layer intermediates without modifying model.
# ─────────────────────────────────────────────────────────────────────────

def install_capture_hooks(encoder, storage: dict):
    """Monkey-patch each RSE-enabled layer's time-mix methods to capture
    θ (both poles), log-decay w, η·θ² viscosity term, and the per-block
    complex readout product before `.real` collapses it.

    Returns a list of (layer_id, attr_name, original_fn) undo tuples.
    """
    undo = []

    for layer_id, layer in enumerate(encoder.layers):
        att = layer.att
        if not att.use_rse:
            continue
        storage[layer_id] = {
            "theta_1": [],      # (B, T, H, Bk)
            "theta_2": [],      # p2rse — (B, T, H, Bk)
            "w_h": [],          # (B, T, H, K) log-decay (negative)
            # readout diagnostics — captured per chunk then concatenated
            "re_abs": [],       # |Re(conj(r_c) * S_total)| summed over blocks — (B, H, T, K)
            "im_abs": [],       # |Im(conj(r_c) * S_total)| summed over blocks — (B, H, T, K)
            "re_blockwise": [], # |Re(conj(r_c) * S_total)| per block (for quadrature ratio)
            "im_blockwise": [], # |Im(conj(r_c) * S_total)| per block
            "re_blockwise_postrot": [],  # A1′: post-rotation |Re| per block
            "im_blockwise_postrot": [],  # A1′: post-rotation |Im| per block
            "state_re_mag": [], # mean |Re(c_state)| per block at end of chunk
            "state_im_mag": [], # mean |Im(c_state)| per block at end of chunk
            "phi_t": [],        # A1′: realised φ(x) per token per (head, block)
        }

        # ── Hook #1: _compute_rkv_gw — captures θ and w before the scan. ──
        orig_rkv = att._compute_rkv_gw
        def _make_rkv_hook(lid, att_ref, orig_fn):
            def hook(x):
                out = orig_fn(x)
                # Returns variable-length tuple depending on mode; unpack
                # common prefix: r, k, v, g, w  (+ theta, [theta_2, [w_2, [k_2, v_2]]])
                w = out[4]
                theta = out[5] if len(out) > 5 else None
                theta_2 = out[6] if len(out) > 6 else None
                B = x.shape[0]; T = x.shape[1]
                H = att_ref.n_head; K = att_ref.head_size; Bk = K // 2
                storage[lid]["w_h"].append(
                    w.detach().float().view(B, T, H, K).cpu()
                )
                if theta is not None:
                    storage[lid]["theta_1"].append(
                        theta.detach().float().view(B, T, H, Bk).cpu()
                    )
                if theta_2 is not None:
                    storage[lid]["theta_2"].append(
                        theta_2.detach().float().view(B, T, H, Bk).cpu()
                    )
                return out
            return hook
        att._compute_rkv_gw = _make_rkv_hook(layer_id, att, orig_rkv)
        undo.append((layer_id, "_compute_rkv_gw", orig_rkv))

        # ── Hook #1b: post-hook on readphase_proj → record realised φ. ──
        if getattr(att, "use_data_dep_readphase", False):
            clip = float(att.readphase_clip)
            def _make_phi_hook(lid):
                def phi_hook(module, inputs, output):
                    # output: (B, T, H*Bk)
                    phi = (clip * torch.tanh(output / clip)).detach().float().cpu()
                    storage[lid]["phi_t"].append(phi)
                return phi_hook
            handle = att.readphase_proj.register_forward_hook(_make_phi_hook(layer_id))
            # Undo via callable closure
            undo.append((layer_id, "__phi_hook_handle__", handle))

        # ── Hook #2: _forward_recurrent_rse — captures the per-chunk
        #            complex readout contraction *before* .real.        ──
        orig_rse = att._forward_recurrent_rse
        def _make_rse_hook(lid, att_ref, orig_fn):
            def hook(r, k, v, w, theta, state=None, apply_bonus=True, phi=None):
                """Reimplementation mirrors the original scan but also
                stashes complex diagnostics.  Returns exactly (out, state).
                The control flow here duplicates the body of
                rwkv6_time_mix._forward_recurrent_rse so the diagnostic
                computation can piggyback without cost.

                Note: `phi` is accepted for API compatibility with the
                A1′ readout, but the diagnostic intentionally measures
                the PRE-rotation complex readout — what `.real` would
                discard without the gauge fix.  If phi is not None we
                also measure the POST-rotation readout and report both.
                """
                B, H, T, K = r.shape
                Bk = K // 2
                device, in_dtype = r.device, r.dtype
                chunk_size = 64

                log_decay_block = w.view(B, H, T, Bk, 2).mean(dim=-1).float()
                if att_ref.use_viscosity:
                    theta_sq = theta.float() ** 2
                    log_decay_block = log_decay_block - att_ref.viscosity_eta.view(
                        1, H, 1, Bk
                    ).float() * theta_sq
                log_z = torch.complex(log_decay_block, theta.float())

                r_pairs = r.float().view(B, H, T, Bk, 2)
                k_pairs = k.float().view(B, H, T, Bk, 2)
                r_c = torch.complex(r_pairs[..., 0], r_pairs[..., 1])
                k_c = torch.complex(k_pairs[..., 0], k_pairs[..., 1])
                v_f = v.float()

                if state is None:
                    c_state = torch.zeros(B, H, Bk, K, dtype=torch.complex64, device=device)
                else:
                    S0 = state.float().view(B, H, Bk, 2, K)
                    c_state = torch.complex(S0[..., 0, :], S0[..., 1, :])

                use_bonus_here = apply_bonus and not att_ref.drop_u
                u_hk = att_ref.time_faaaa.view(1, H, 1, K).float() if use_bonus_here else None

                out = torch.zeros(B, H, T, K, dtype=torch.float32, device=device)

                # Diagnostic accumulators (full sequence, concatenated after).
                re_abs_full = torch.zeros(B, H, T, K, dtype=torch.float32, device=device)
                im_abs_full = torch.zeros(B, H, T, K, dtype=torch.float32, device=device)
                # Per-block diagnostic: mean over (B, T, K) of |Re| and |Im| per-block.
                re_block_acc = torch.zeros(H, Bk, dtype=torch.float64, device=device)
                im_block_acc = torch.zeros(H, Bk, dtype=torch.float64, device=device)
                # A1′: also track post-rotation quadrature for comparison.
                re_block_acc_pr = torch.zeros(H, Bk, dtype=torch.float64, device=device)
                im_block_acc_pr = torch.zeros(H, Bk, dtype=torch.float64, device=device)
                block_counts = 0

                cur = 0
                while cur < T:
                    tc = min(chunk_size, T - cur)
                    log_z_c = log_z[:, :, cur:cur + tc]
                    k_c_c = k_c[:, :, cur:cur + tc]
                    r_c_c = r_c[:, :, cur:cur + tc]
                    v_c = v_f[:, :, cur:cur + tc]

                    cumlog = log_z_c.cumsum(dim=2)
                    diff = cumlog.unsqueeze(3) - cumlog.unsqueeze(2)
                    mask = torch.tril(
                        torch.ones(tc, tc, device=device, dtype=torch.bool)
                    ).view(1, 1, tc, tc, 1)
                    real_part = diff.real.masked_fill(~mask, -60.0)
                    A_raw = torch.exp(torch.complex(real_part, diff.imag))
                    A = torch.where(mask, A_raw, torch.zeros_like(A_raw))

                    scaled_k = A * k_c_c.unsqueeze(2)
                    v_c_complex = v_c.to(torch.complex64)
                    S_intra = torch.einsum(
                        'bhtsk,bhsc->bhtkc', scaled_k, v_c_complex
                    )

                    decay_to_t = torch.exp(cumlog)
                    prior_contrib = decay_to_t.unsqueeze(-1) * c_state.unsqueeze(2)

                    S_total = prior_contrib + S_intra

                    # ── Complex readout pre-collapse, shape (B, H, tc, K) ──
                    # Always measure the PRE-rotation version: this is what
                    # an anchor model would read out, and what quantifies the
                    # quadrature content available for gauge recovery.
                    y_complex_pre = torch.einsum(
                        'bhtk,bhtkc->bhtc', r_c_c.conj(), S_total
                    )

                    # Per-block pre-rotation (conj(r) alone): (B, H, tc, Bk, K)
                    per_block_pre = r_c_c.conj().unsqueeze(-1) * S_total
                    re_block_acc += per_block_pre.real.abs().mean(dim=(0, 2, 4)).to(torch.float64)
                    im_block_acc += per_block_pre.imag.abs().mean(dim=(0, 2, 4)).to(torch.float64)

                    # ── If the model is A1′, also compute POST-rotation ──
                    # conj(r) · exp(-iφ) · S_total — what the A1′ readout
                    # actually uses.  For anchor (phi=None) this equals pre.
                    if phi is not None:
                        phi_c = phi[:, :, cur:cur + tc].float()           # (B,H,tc,Bk)
                        rot = torch.polar(torch.ones_like(phi_c), -phi_c)  # exp(-iφ)
                        r_contract = r_c_c.conj() * rot
                        per_block_post = r_contract.unsqueeze(-1) * S_total
                        re_block_acc_pr += per_block_post.real.abs().mean(dim=(0, 2, 4)).to(torch.float64)
                        im_block_acc_pr += per_block_post.imag.abs().mean(dim=(0, 2, 4)).to(torch.float64)
                        y_complex = torch.einsum('bhtk,bhtkc->bhtc', r_contract, S_total)
                    else:
                        # anchor — post-rotation is identical to pre
                        re_block_acc_pr += per_block_pre.real.abs().mean(dim=(0, 2, 4)).to(torch.float64)
                        im_block_acc_pr += per_block_pre.imag.abs().mean(dim=(0, 2, 4)).to(torch.float64)
                        y_complex = y_complex_pre

                    y_re = y_complex.real
                    y_im = y_complex.imag

                    block_counts += 1

                    y_chunk = y_re

                    if u_hk is not None:
                        r_t_r = r.float()[:, :, cur:cur + tc]
                        k_t_r = k.float()[:, :, cur:cur + tc]
                        scalar = (r_t_r * u_hk[:, :, 0:1] * k_t_r).sum(dim=-1, keepdim=True)
                        y_chunk = y_chunk + scalar * v_c

                    out[:, :, cur:cur + tc] = y_chunk
                    re_abs_full[:, :, cur:cur + tc] = y_re.abs()
                    im_abs_full[:, :, cur:cur + tc] = y_im.abs()

                    c_state = S_total[:, :, -1]
                    cur += tc

                # Record diagnostics for this call.  Reduce spatial dims
                # now so we keep memory small.
                storage[lid]["re_abs"].append(
                    re_abs_full.detach().float().mean(dim=(0, 2, 3)).cpu()  # (H,)
                )
                storage[lid]["im_abs"].append(
                    im_abs_full.detach().float().mean(dim=(0, 2, 3)).cpu()  # (H,)
                )
                # Per-block (H, Bk) means across this forward call
                if block_counts > 0:
                    storage[lid]["re_blockwise"].append(
                        (re_block_acc / block_counts).cpu().float()  # (H, Bk)
                    )
                    storage[lid]["im_blockwise"].append(
                        (im_block_acc / block_counts).cpu().float()
                    )
                    storage[lid]["re_blockwise_postrot"].append(
                        (re_block_acc_pr / block_counts).cpu().float()
                    )
                    storage[lid]["im_blockwise_postrot"].append(
                        (im_block_acc_pr / block_counts).cpu().float()
                    )
                # Final chunk's |c_state| (H, Bk)
                storage[lid]["state_re_mag"].append(
                    c_state.real.abs().mean(dim=(0, 3)).detach().float().cpu()
                )
                storage[lid]["state_im_mag"].append(
                    c_state.imag.abs().mean(dim=(0, 3)).detach().float().cpu()
                )

                final_state = torch.zeros(B, H, K, K, dtype=torch.float32, device=device)
                final_view = final_state.view(B, H, Bk, 2, K)
                final_view[..., 0, :] = c_state.real
                final_view[..., 1, :] = c_state.imag

                return out.to(in_dtype), final_state
            return hook

        att._forward_recurrent_rse = _make_rse_hook(layer_id, att, orig_rse)
        undo.append((layer_id, "_forward_recurrent_rse", orig_rse))

    return undo


def remove_hooks(encoder, undo):
    for layer_id, attr, orig in undo:
        if attr == "__phi_hook_handle__":
            # torch.nn.Module.register_forward_hook returns a RemovableHandle
            orig.remove()
        else:
            setattr(encoder.layers[layer_id].att, attr, orig)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Where to write the diagnostic JSON and PNGs. "
                             "Defaults to <run-dir>/stage7_diagnostics/.")
    parser.add_argument("--n-utt", type=int, default=N_UTTERANCES_DEFAULT)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt = run_dir / "best_model.pt"
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "stage7_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(run_dir / "config.yaml"), {})
    vocab = CharVocab.build_english()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"Run dir  : {run_dir}")
    print(f"Out dir  : {out_dir}")
    print(f"Loading  : {ckpt} …")
    model = ASRModel(vocab_size=vocab.size, cfg=cfg).to(device)
    state = load_checkpoint(ckpt, model=model, map_location=device, restore_rng=False)
    model.eval()
    print(f"  ckpt epoch={state.get('epoch')}  best_cer={state.get('best_cer'):.4f}")

    encoder = model.encoder
    n_layers = encoder.n_layers

    # ── 0. A1′ static parameter mobility ──────────────────────────────
    # If the model has data-dependent readout phase, measure how far W_φ, b_φ
    # moved from their zero init.  This is the single most informative
    # diagnostic for the ambiguous-band outcome: did SGD use the mechanism?
    static_phi = {}
    att0 = encoder.layers[0].att
    has_dphi = getattr(att0, "use_data_dep_readphase", False)
    if has_dphi:
        print("\n" + "=" * 76)
        print("STATIC (A1′): W_φ, b_φ parameter mobility (init was all-zero)")
        print("=" * 76)
        print(f"{'Layer':<6} | {'|W_φ|_F':<10} | {'|W_φ|_∞':<10} | {'|b_φ|_2':<10} | "
              f"{'|b_φ|_∞':<10}")
        print("-" * 76)
        for i, layer in enumerate(encoder.layers):
            att = layer.att
            if not getattr(att, "use_data_dep_readphase", False):
                continue
            W = att.readphase_proj.weight.detach().float().cpu()
            b = att.readphase_proj.bias.detach().float().cpu()
            W_fnorm = W.norm().item()
            W_inf = W.abs().max().item()
            b_l2 = b.norm().item()
            b_inf = b.abs().max().item()
            print(f"L{i:<5}| {W_fnorm:.4e} | {W_inf:.4e} | {b_l2:.4e} | {b_inf:.4e}")
            static_phi[i] = {
                "W_fnorm": W_fnorm, "W_inf": W_inf,
                "b_l2": b_l2, "b_inf": b_inf,
                "readphase_clip": float(att.readphase_clip),
            }

    # ── 1. Static viscosity inspection ────────────────────────────────
    print("\n" + "=" * 76)
    print("STATIC: per-layer viscosity η")
    print("=" * 76)
    print(f"{'Layer':<6} | {'mean η':<10} | {'std η':<10} | {'max η':<10} | {'Bk':<4}")
    print("-" * 76)
    static_eta = {}
    for i, layer in enumerate(encoder.layers):
        att = layer.att
        if not att.use_rse:
            continue
        if not att.use_viscosity:
            continue
        eta = att.viscosity_eta.detach().float().cpu()  # (H, Bk)
        mean = eta.mean().item()
        std = eta.std().item()
        max_abs = eta.abs().max().item()
        Bk = eta.shape[1]
        print(f"L{i:<5}| {mean:+.4f}   | {std:.4f}    | {max_abs:.4f}    | {Bk}")
        static_eta[i] = {
            "mean": mean, "std": std, "max_abs": max_abs,
            "per_head_mean": eta.mean(dim=1).tolist(),
            "per_head_max": eta.abs().max(dim=1).values.tolist(),
        }

    # ── 2. Dynamic capture over dev batches ────────────────────────────
    storage: dict = {}
    undo = install_capture_hooks(encoder, storage)

    try:
        print("\nLoading dev-clean utterances…")
        dev_entries = load_librispeech("dev", cfg.data_cache_dir, cfg.min_audio_sec, cfg.max_audio_sec)
        dev_entries = dev_entries[:args.n_utt]
        dev_ds = ASRDataset(dev_entries, vocab, cfg)
        dev_loader = DataLoader(
            dev_ds,
            batch_sampler=DurationBatchSampler(dev_entries, cfg.batch_max_seconds, False, cfg.seed),
            collate_fn=collate_fn,
            num_workers=2,
        )

        print("Running forward pass with capture hooks…")
        n_processed = 0
        with torch.no_grad():
            for batch in dev_loader:
                # collate_fn returns (mels, targets, mel_lengths, target_lengths)
                mels, _targets, mel_lens, _tlens = batch
                mels = mels.to(device); mel_lens = mel_lens.to(device)
                try:
                    _ = model(mels, mel_lens)
                except Exception as e:
                    print(f"  batch failed: {e}")
                    continue
                n_processed += mels.size(0)
                if n_processed >= args.n_utt:
                    break
        print(f"  processed {n_processed} utterances")
    finally:
        remove_hooks(encoder, undo)

    # ── 3. Summarise D1: rotation-budget saturation ────────────────────
    print("\n" + "=" * 76)
    print(f"D1  θ rotation-budget saturation (strong clip = ±π/2 ≈ {THETA_CLIP:.3f})")
    print("=" * 76)
    print(f"{'Layer':<6} | {'pole':<4} | {'mean θ':<9} | {'std θ':<8} | "
          f"{'|θ|/clip p50':<12} | {'p95':<7} | {'p99':<7} | {'sat>0.95':<9}")
    print("-" * 76)
    d1_summary = {}
    for lid in sorted(storage.keys()):
        d1_summary[lid] = {}
        for pole_idx, key in enumerate(["theta_1", "theta_2"]):
            if not storage[lid][key]:
                continue
            all_theta = torch.cat([t.reshape(-1) for t in storage[lid][key]]).numpy()
            abs_over_clip = np.abs(all_theta) / THETA_CLIP
            p50 = float(np.percentile(abs_over_clip, 50))
            p95 = float(np.percentile(abs_over_clip, 95))
            p99 = float(np.percentile(abs_over_clip, 99))
            frac_sat = float(np.mean(abs_over_clip > SATURATION_THRESHOLD))
            mean_ = float(all_theta.mean())
            std_ = float(all_theta.std())
            pole_label = f"p{pole_idx + 1}"
            print(f"L{lid:<5}| {pole_label:<4} | {mean_:+.4f}  | {std_:.4f}   | "
                  f"{p50:.3f}        | {p95:.3f}   | {p99:.3f}   | {frac_sat:.4f}")
            d1_summary[lid][pole_label] = {
                "mean": mean_, "std": std_,
                "p50_abs_over_clip": p50,
                "p95_abs_over_clip": p95,
                "p99_abs_over_clip": p99,
                "fraction_saturated": frac_sat,
                "n_samples": int(all_theta.size),
            }

    # ── 4. Summarise D2: quadrature amplitude at readout ───────────────
    print("\n" + "=" * 76)
    print("D2  Readout quadrature amplitude  ρ = |Im| / |Re|  at  conj(r_c) * S_total")
    print("=" * 76)
    print(f"{'Layer':<6} | {'mean |Re|':<11} | {'mean |Im|':<11} | "
          f"{'Im/Re (global)':<15} | {'blk p50':<8} | {'blk p95':<8}")
    print("-" * 76)
    d2_summary = {}
    for lid in sorted(storage.keys()):
        re_abs_list = storage[lid]["re_abs"]
        im_abs_list = storage[lid]["im_abs"]
        if not re_abs_list:
            continue
        # Per-head means across forward calls
        re_mean = torch.stack(re_abs_list).mean(dim=0).numpy()  # (H,)
        im_mean = torch.stack(im_abs_list).mean(dim=0).numpy()
        re_global = float(re_mean.mean())
        im_global = float(im_mean.mean())
        ratio_global = im_global / (re_global + 1e-12)

        # Per-block ratios (pre-rotation — same for anchor and A1′)
        re_blocks = torch.stack(storage[lid]["re_blockwise"]).mean(dim=0)  # (H, Bk)
        im_blocks = torch.stack(storage[lid]["im_blockwise"]).mean(dim=0)
        blk_ratio = (im_blocks / (re_blocks + 1e-12)).numpy().reshape(-1)
        blk_p50 = float(np.percentile(blk_ratio, 50))
        blk_p95 = float(np.percentile(blk_ratio, 95))

        # Per-block ratios POST-rotation (A1′: what survives the learned φ).
        re_blocks_pr = torch.stack(storage[lid]["re_blockwise_postrot"]).mean(dim=0)
        im_blocks_pr = torch.stack(storage[lid]["im_blockwise_postrot"]).mean(dim=0)
        blk_ratio_pr = (im_blocks_pr / (re_blocks_pr + 1e-12)).numpy().reshape(-1)
        blk_p50_pr = float(np.percentile(blk_ratio_pr, 50))
        blk_p95_pr = float(np.percentile(blk_ratio_pr, 95))
        im_post_global = float(im_blocks_pr.mean())
        re_post_global = float(re_blocks_pr.mean())
        ratio_post_global = im_post_global / (re_post_global + 1e-12)

        # State magnitude diagnostic
        state_re = torch.stack(storage[lid]["state_re_mag"]).mean(dim=0)
        state_im = torch.stack(storage[lid]["state_im_mag"]).mean(dim=0)
        state_ratio = (state_im / (state_re + 1e-12)).mean().item()

        print(f"L{lid:<5}| {re_global:.4f}      | {im_global:.4f}      | "
              f"{ratio_global:.4f}          | {blk_p50:.3f}    | {blk_p95:.3f}")
        d2_summary[lid] = {
            "re_mean_global": re_global,
            "im_mean_global": im_global,
            "im_over_re_global": ratio_global,
            "im_over_re_blockwise_p50": blk_p50,
            "im_over_re_blockwise_p95": blk_p95,
            "state_im_over_re_mean": state_ratio,
            "per_head_re": re_mean.tolist(),
            "per_head_im": im_mean.tolist(),
            # POST-rotation — for anchor, same as pre.  For A1′, shrinkage
            # of this vs pre is the mechanism-level signature that the
            # learned readout phase is doing its gauge-alignment job.
            "im_over_re_global_postrot": ratio_post_global,
            "im_over_re_blockwise_p50_postrot": blk_p50_pr,
            "im_over_re_blockwise_p95_postrot": blk_p95_pr,
        }

    # ── 5. Summarise D3: λ-θ coupling ──────────────────────────────────
    print("\n" + "=" * 76)
    print("D3  Viscosity coupling:  λ_eff = λ_raw + η·θ²  — realised correlation")
    print("=" * 76)
    print(f"{'Layer':<6} | {'corr(θ², λ_eff)':<17} | {'mean η·θ²':<12} | "
          f"{'η·θ² / λ_raw':<14}")
    print("-" * 76)
    d3_summary = {}
    for lid in sorted(storage.keys()):
        if not storage[lid]["theta_1"] or not storage[lid]["w_h"]:
            continue
        # Batches have variable T, so flatten each (B, T, H, Bk) / (B, T, H, K)
        # before concatenating.  We keep (per-cell) shape for lam/theta so the
        # correlation is measured across (batch, time, head, block) cells.
        att = encoder.layers[lid].att
        H = att.n_head; K = att.head_size; Bk = K // 2
        eta = att.viscosity_eta.detach().float().cpu().view(1, 1, H, Bk)        # (1,1,H,Bk)

        ts_chunks = []
        lam_raw_chunks = []
        visc_chunks = []
        for theta_b, w_b in zip(storage[lid]["theta_1"], storage[lid]["w_h"]):
            theta_b = theta_b.float()                                           # (B,T,H,Bk)
            w_b = w_b.float()                                                   # (B,T,H,K)
            w_log = -torch.exp(w_b)                                             # actual log-decay
            lam_pair = w_log.view(w_b.shape[0], w_b.shape[1], H, Bk, 2).mean(dim=-1)
            lam_pos = -lam_pair                                                 # positive magnitude
            visc = eta * theta_b ** 2                                           # (B,T,H,Bk)
            ts_chunks.append((theta_b ** 2).flatten())
            lam_raw_chunks.append(lam_pos.flatten())
            visc_chunks.append(visc.flatten())
        ts = torch.cat(ts_chunks).numpy()
        ls_raw = torch.cat(lam_raw_chunks).numpy()
        vs = torch.cat(visc_chunks).numpy()
        ls = ls_raw + vs                                                        # λ_eff
        if ts.std() > 0 and ls.std() > 0:
            corr = float(np.corrcoef(ts, ls)[0, 1])
        else:
            corr = float("nan")
        if ts.std() > 0 and ls_raw.std() > 0:
            corr_raw = float(np.corrcoef(ts, ls_raw)[0, 1])
        else:
            corr_raw = float("nan")
        mean_visc = float(vs.mean())
        mean_lam_raw = float(ls_raw.mean())
        frac_of_lam = mean_visc / (mean_lam_raw + 1e-12)
        print(f"L{lid:<5}| {corr:+.4f}            | {mean_visc:.6f}    | {frac_of_lam:.4f}")
        d3_summary[lid] = {
            "corr_thetasq_lameff": corr,
            "corr_thetasq_lamraw": corr_raw,
            "mean_viscosity_contrib": mean_visc,
            "mean_lam_raw": mean_lam_raw,
            "visc_fraction_of_lam": frac_of_lam,
        }

    # ── 5b. Summarise D4: A1′ realised φ(x) distribution ──────────────
    d4_summary = {}
    if has_dphi:
        print("\n" + "=" * 76)
        print("D4  A1′ — realised φ(x) at readout (post-clip)")
        print("=" * 76)
        print(f"{'Layer':<6} | {'mean |φ|':<10} | {'std φ':<10} | "
              f"{'p95 |φ|':<10} | {'p99 |φ|':<10} | {'frac ≥ 0.9·clip':<16}")
        print("-" * 76)
        for lid in sorted(storage.keys()):
            if not storage[lid]["phi_t"]:
                continue
            all_phi = torch.cat([t.reshape(-1) for t in storage[lid]["phi_t"]]).numpy()
            clip = static_phi.get(lid, {}).get("readphase_clip", math.pi)
            mean_abs = float(np.mean(np.abs(all_phi)))
            std_ = float(all_phi.std())
            p95 = float(np.percentile(np.abs(all_phi), 95))
            p99 = float(np.percentile(np.abs(all_phi), 99))
            frac_sat = float(np.mean(np.abs(all_phi) > 0.9 * clip))
            print(f"L{lid:<5}| {mean_abs:.4f}    | {std_:.4f}    | "
                  f"{p95:.4f}    | {p99:.4f}    | {frac_sat:.4f}")
            d4_summary[lid] = {
                "mean_abs": mean_abs, "std": std_,
                "p95_abs": p95, "p99_abs": p99,
                "fraction_near_clip": frac_sat,
                "clip": clip, "n_samples": int(all_phi.size),
            }

    # ── 6. Decision summary ────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("STAGE-7A GO / NO-GO SUMMARY")
    print("=" * 76)

    # Heuristic: take max across layers/poles for each property.
    max_sat = max(
        (p["fraction_saturated"] for ld in d1_summary.values() for p in ld.values()),
        default=0.0,
    )
    max_im_re = max(
        (ld["im_over_re_global"] for ld in d2_summary.values()),
        default=0.0,
    )
    max_blkp95 = max(
        (ld["im_over_re_blockwise_p95"] for ld in d2_summary.values()),
        default=0.0,
    )
    max_visc = max(
        (ld["visc_fraction_of_lam"] for ld in d3_summary.values()),
        default=0.0,
    )

    print(f"  max fraction θ saturated             : {max_sat:.4f}  "
          f"{'[LOW — strong budget NOT binding]' if max_sat < 0.03 else '[SATURATED — budget binding]'}")
    print(f"  max global |Im|/|Re| at readout      : {max_im_re:.4f}  "
          f"{'[SMALL — gauge already absorbed by r_proj]' if max_im_re < 0.05 else '[SUBSTANTIAL — gauge recoverable]'}")
    print(f"  max per-block p95 |Im|/|Re|          : {max_blkp95:.4f}  "
          f"{'[per-block rotations average out]' if max_blkp95 < 0.1 else '[per-block phase misalignment]'}")
    print(f"  max η·θ² fraction of λ_raw            : {max_visc:.4f}  "
          f"{'[viscosity barely engaged]' if max_visc < 0.02 else '[viscosity materially engaged]'}")

    # ── 7. Plots ───────────────────────────────────────────────────────
    # 7.1 θ histograms
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=False)
    for lid in range(n_layers):
        if lid not in storage or not storage[lid]["theta_1"]:
            continue
        ax = axes.flatten()[lid]
        all_t1 = torch.cat([t.reshape(-1) for t in storage[lid]["theta_1"]]).numpy()
        ax.hist(all_t1, bins=80, color="C0", alpha=0.55, density=True,
                range=(-THETA_CLIP, THETA_CLIP), label="pole 1")
        if storage[lid]["theta_2"]:
            all_t2 = torch.cat([t.reshape(-1) for t in storage[lid]["theta_2"]]).numpy()
            ax.hist(all_t2, bins=80, color="C3", alpha=0.45, density=True,
                    range=(-THETA_CLIP, THETA_CLIP), label="pole 2")
        ax.axvline(-THETA_CLIP, color="r", lw=0.8, ls=":")
        ax.axvline(+THETA_CLIP, color="r", lw=0.8, ls=":")
        ax.set_title(f"L{lid}")
        ax.set_xlabel("θ (rad)")
        if lid == 0:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"D1 — θ distribution per layer "
                 f"(strong clip ±π/2) — {run_dir.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "rse_diag_theta.png", dpi=120)
    plt.close(fig)

    # 7.2 Quadrature histograms
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=False)
    for lid in range(n_layers):
        if lid not in storage or not storage[lid]["re_blockwise"]:
            continue
        ax = axes.flatten()[lid]
        re_blocks = torch.stack(storage[lid]["re_blockwise"]).mean(dim=0).numpy()
        im_blocks = torch.stack(storage[lid]["im_blockwise"]).mean(dim=0).numpy()
        ratio = (im_blocks / (re_blocks + 1e-12)).reshape(-1)
        ax.hist(ratio, bins=40, color="C2", alpha=0.7)
        ax.axvline(0.05, color="r", lw=0.8, ls=":", label="0.05 threshold")
        ax.set_title(f"L{lid}: p95={np.percentile(ratio, 95):.3f}")
        ax.set_xlabel("per-block |Im|/|Re|")
        if lid == 0:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("D2 — per-block quadrature ratio at readout (the content Stage 7A could recover)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "rse_diag_quadrature.png", dpi=120)
    plt.close(fig)

    # 7.3 Viscosity scatter
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for lid in range(n_layers):
        if lid not in storage or not storage[lid]["theta_1"] or not storage[lid]["w_h"]:
            continue
        ax = axes.flatten()[lid]
        att = encoder.layers[lid].att
        H = att.n_head; K = att.head_size; Bk = K // 2
        eta = att.viscosity_eta.detach().float().cpu().view(1, 1, H, Bk)
        ts_chunks = []; ls_chunks = []
        for theta_b, w_b in zip(storage[lid]["theta_1"], storage[lid]["w_h"]):
            theta_b = theta_b.float(); w_b = w_b.float()
            lam_pos = -(-torch.exp(w_b)).view(w_b.shape[0], w_b.shape[1], H, Bk, 2).mean(dim=-1)
            visc = eta * theta_b ** 2
            ts_chunks.append((theta_b ** 2).flatten())
            ls_chunks.append((lam_pos + visc).flatten())
        ts = torch.cat(ts_chunks).numpy()
        ls = torch.cat(ls_chunks).numpy()
        n = min(20_000, ts.size)
        idx = np.random.default_rng(0).choice(ts.size, size=n, replace=False)
        ax.scatter(ts[idx], ls[idx], s=0.3, alpha=0.3)
        ax.set_xlabel("θ²")
        ax.set_ylabel("λ_eff")
        corr = d3_summary.get(lid, {}).get("corr_thetasq_lameff", 0.0)
        ax.set_title(f"L{lid}: corr={corr:+.3f}")
    fig.suptitle("D3 — viscosity coupling: θ² vs λ_eff", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "rse_diag_viscosity.png", dpi=120)
    plt.close(fig)

    # 7.4 A1′ realised φ histograms (if applicable)
    if has_dphi and any(storage[lid].get("phi_t") for lid in storage):
        fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=False)
        for lid in range(n_layers):
            if lid not in storage or not storage[lid].get("phi_t"):
                continue
            ax = axes.flatten()[lid]
            all_phi = torch.cat([t.reshape(-1) for t in storage[lid]["phi_t"]]).numpy()
            clip = static_phi.get(lid, {}).get("readphase_clip", math.pi)
            ax.hist(all_phi, bins=80, alpha=0.7, density=True,
                    range=(-clip, clip))
            ax.axvline(-clip, color="r", lw=0.8, ls=":")
            ax.axvline(+clip, color="r", lw=0.8, ls=":")
            ax.set_title(f"L{lid}: std={all_phi.std():.3f}")
            ax.set_xlabel("φ (rad)")
        fig.suptitle("D4 — realised φ(x) per layer (readout phase, post-clip)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "rse_diag_phi.png", dpi=120)
        plt.close(fig)

    # ── 8. Dump JSON ────────────────────────────────────────────────────
    summary = {
        "checkpoint": str(ckpt),
        "run_dir": str(run_dir),
        "backbone": cfg.backbone,
        "has_dphi": bool(has_dphi),
        "best_cer": state.get("best_cer"),
        "epoch": state.get("epoch"),
        "n_utterances": n_processed,
        "strong_theta_clip": THETA_CLIP,
        "static_viscosity_eta": static_eta,
        "static_readphase": static_phi,
        "D1_theta_saturation": d1_summary,
        "D2_quadrature_amplitude": d2_summary,
        "D3_lambda_theta_coupling": d3_summary,
        "D4_realised_phi": d4_summary,
        "aggregate": {
            "max_fraction_theta_saturated": max_sat,
            "max_global_im_over_re": max_im_re,
            "max_blockwise_p95_im_over_re": max_blkp95,
            "max_viscosity_fraction_of_lam": max_visc,
        },
    }
    with open(out_dir / "rse_diag_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {out_dir / 'rse_diag_summary.json'}")
    print(f"       {out_dir / 'rse_diag_theta.png'}")
    print(f"       {out_dir / 'rse_diag_quadrature.png'}")
    print(f"       {out_dir / 'rse_diag_viscosity.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
