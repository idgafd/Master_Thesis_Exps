#!/usr/bin/env python3
"""Stage-8 mechanism-level diagnostic — T1 (delta) and T2 (nonnormal_rse).

Runs the pre-registered D5–D9 diagnostic probes on a trained checkpoint
and outputs the exact decision-tree leaf (no interpretation).

For T1 (`rwkv6_delta_warmstart_fixed`):
  - D5_delta  : per-head `delta_recurrent_gate` mobility (init zero)
  - D6_delta  : realised β_eff(x) = g · iclr(x) distribution on dev
  - D7_delta  : relative erase magnitude ‖β·kk·(kk^T·S)‖_F / ‖S‖_F

For T2 (`rwkv6_nonnormal_rse_viscosity`):
  - D5_nn : `nonnormal_rho_base`, LoRA, `nonnormal_mu` mobility
  - D6_nn : realised ρ(x), ψ(x) distribution per layer
  - D7_nn : non-normality score  ‖G^T G - G G^T‖_F / ‖G‖_F²  per layer
  - D8_nn : spectral radius max|eig(G)| distribution (stability envelope)
  - D9_nn : per-head mean |ρ| — cross-head specialisation pattern

Output: prints the diagnostic table, writes JSON, prints ONE decision-tree
leaf per the pre-registered protocol in STAGE8_PLAN §3.4 / §4.7.

Run:
    uv run python scripts/stage8_diagnostics.py --run-dir outputs/s8_t2_...
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
import torch.nn.functional as F

from src.config import load_config
from src.data.vocab import CharVocab
from src.data.dataset import ASRDataset, DurationBatchSampler, collate_fn
from src.data.librispeech import load_librispeech
from src.models.asr_model import ASRModel
from src.training.checkpoint import load_checkpoint
from torch.utils.data import DataLoader


N_UTT_DEFAULT = 120
SIGMA_SEED = 0.0014


# ─────────────────────────────────────────────────────────────────────────
# Capture hooks (store inputs to _compute_rkv_gw paths per layer)
# ─────────────────────────────────────────────────────────────────────────

def install_capture_hooks(encoder, storage: dict, backbone_kind: str):
    """For each RSE-enabled / delta-enabled layer, hook _compute_rkv_gw to
    capture r, k, v, w (raw pre-exp log decay), and if applicable theta,
    rho_raw, psi_raw (non-normal) OR iclr, kk (delta via DeltaRuleParams).
    """
    undo = []
    for lid, layer in enumerate(encoder.layers):
        att = layer.att
        storage[lid] = {
            "r": [], "k": [], "v": [], "w": [],
            "theta": [], "rho_raw": [], "psi_raw": [],
            "kk": [], "iclr": [],
        }

        if backbone_kind == "nonnormal":
            orig = att._compute_rkv_gw
            def _make_nn(lid_, att_, orig_):
                def hook(x):
                    out = orig_(x)
                    # (r, k, v, g, w, theta, rho_raw, psi_raw) when nonnormal
                    if len(out) == 8:
                        r, k, v, g, w, theta, rho_raw, psi_raw = out
                        B = x.shape[0]; T = x.shape[1]
                        H = att_.n_head; K = att_.head_size; Bk = K // 2
                        storage[lid_]["r"].append(r.detach().float().view(B, T, H, K).cpu())
                        storage[lid_]["k"].append(k.detach().float().view(B, T, H, K).cpu())
                        storage[lid_]["v"].append(v.detach().float().view(B, T, H, K).cpu())
                        storage[lid_]["w"].append(w.detach().float().view(B, T, H, K).cpu())
                        storage[lid_]["theta"].append(theta.detach().float().view(B, T, H, Bk).cpu())
                        storage[lid_]["rho_raw"].append(rho_raw.detach().float().view(B, T, H, Bk).cpu())
                        storage[lid_]["psi_raw"].append(psi_raw.detach().float().view(B, T, H, Bk).cpu())
                    return out
                return hook
            att._compute_rkv_gw = _make_nn(lid, att, orig)
            undo.append((lid, "_compute_rkv_gw", orig))

        elif backbone_kind == "delta":
            orig = att._compute_rkv_gw
            def _make_d(lid_, att_, orig_):
                def hook(x):
                    out = orig_(x)
                    r, k, v, g, w = out[0], out[1], out[2], out[3], out[4]
                    B, T = x.shape[0], x.shape[1]
                    H = att_.n_head; K = att_.head_size
                    # Compute kk, iclr via delta_params on the captured k
                    kk, iclr = att_.delta_params.compute_kk_iclr(
                        k.view(B, T, H, K).transpose(1, 2),  # (B,H,T,K)
                        B, T, H, K,
                    )
                    storage[lid_]["r"].append(r.detach().float().view(B, T, H, K).cpu())
                    storage[lid_]["k"].append(k.detach().float().view(B, T, H, K).cpu())
                    storage[lid_]["v"].append(v.detach().float().view(B, T, H, K).cpu())
                    storage[lid_]["w"].append(w.detach().float().view(B, T, H, K).cpu())
                    storage[lid_]["kk"].append(kk.detach().float().cpu())
                    storage[lid_]["iclr"].append(iclr.detach().float().cpu())
                    return out
                return hook
            att._compute_rkv_gw = _make_d(lid, att, orig)
            undo.append((lid, "_compute_rkv_gw", orig))
    return undo


def remove_hooks(encoder, undo):
    for lid, attr, orig in undo:
        setattr(encoder.layers[lid].att, attr, orig)


# ─────────────────────────────────────────────────────────────────────────
# D7: non-normality score for G from (λ, θ, ρ, ψ)
# ─────────────────────────────────────────────────────────────────────────

def compute_G_per_block(w_raw: torch.Tensor, theta: torch.Tensor,
                       rho: torch.Tensor, psi: torch.Tensor):
    """Reconstruct per-block 2×2 transition matrices G from polar-form
    parameters. Returns G shape (..., 2, 2) with all leading batch dims.

    w_raw is the raw (pre -exp -neg) per-channel LoRA output; we first
    transform to log_decay_block (mean over channel pair) and then build
    G = exp(-λ) P(ρ, ψ) R(θ).
    """
    # w_raw shape (..., K). Per-block log-decay.
    # The actual log-decay used in the scan is -exp(w_raw) (= w_h in code).
    # After the viscosity subtraction λ_eff = λ + η θ² + μ ρ² we'd need to
    # add that back; this function is for G at the layer's "raw" decay
    # WITHOUT viscosity — consistent across anchor vs T2 comparisons.
    K = w_raw.shape[-1]
    Bk = K // 2
    lam_log = -torch.exp(w_raw)                         # negative; log-decay
    lam_block_log = lam_log.view(*w_raw.shape[:-1], Bk, 2).mean(dim=-1)  # (..., Bk)
    decay = torch.exp(lam_block_log)                    # (..., Bk), in (0, 1]

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    cosh_r = torch.cosh(rho)
    sinh_r = torch.sinh(rho)
    cos_2p = torch.cos(2.0 * psi)
    sin_2p = torch.sin(2.0 * psi)

    P00 = cosh_r + sinh_r * cos_2p
    P01 = sinh_r * sin_2p
    P11 = cosh_r - sinh_r * cos_2p

    G00 = decay * (P00 * cos_t + P01 * sin_t)
    G01 = decay * (-P00 * sin_t + P01 * cos_t)
    G10 = decay * (P01 * cos_t + P11 * sin_t)
    G11 = decay * (-P01 * sin_t + P11 * cos_t)

    # Stack into (..., Bk, 2, 2)
    G = torch.stack([
        torch.stack([G00, G01], dim=-1),
        torch.stack([G10, G11], dim=-1),
    ], dim=-2)
    return G


def non_normality_and_spectrum(G: torch.Tensor):
    """Given G shape (..., 2, 2), compute for each:
      - non-normality score: ‖G^T G - G G^T‖_F / ‖G‖_F²
      - spectral radius max|eig(G)|
    Returns (score, radius) tensors of shape G.shape[:-2].
    """
    Gt = G.transpose(-1, -2)
    AtA = torch.einsum('...ij,...jk->...ik', Gt, G)
    AAt = torch.einsum('...ij,...jk->...ik', G, Gt)
    comm = AtA - AAt
    # Frobenius norms
    comm_fnorm_sq = (comm ** 2).sum(dim=(-1, -2))
    g_fnorm_sq = (G ** 2).sum(dim=(-1, -2))
    score = torch.sqrt(comm_fnorm_sq) / (g_fnorm_sq + 1e-12)

    # Spectral radius for 2x2: eigenvalues are (tr ± sqrt(tr² - 4 det))/2
    tr = G[..., 0, 0] + G[..., 1, 1]
    det = G[..., 0, 0] * G[..., 1, 1] - G[..., 0, 1] * G[..., 1, 0]
    disc = tr * tr - 4.0 * det
    # Complex if disc < 0: eigenvalues = (tr ± i·sqrt(-disc))/2, magnitude = sqrt(det)
    # Real if disc ≥ 0: eigenvalues = (tr ± sqrt(disc))/2
    real_branch = disc >= 0
    eig_real_hi = (tr + torch.sqrt(disc.clamp(min=0))) / 2.0
    eig_real_lo = (tr - torch.sqrt(disc.clamp(min=0))) / 2.0
    mag_real = torch.maximum(eig_real_hi.abs(), eig_real_lo.abs())
    mag_complex = torch.sqrt(det.abs())
    radius = torch.where(real_branch, mag_real, mag_complex)
    return score, radius


# ─────────────────────────────────────────────────────────────────────────
# Main per-backbone diagnostic functions
# ─────────────────────────────────────────────────────────────────────────

def diagnostic_nonnormal(model, storage, cfg, state_info, out_dir: Path):
    encoder = model.encoder
    n_layers = encoder.n_layers
    kappa = encoder.layers[0].att.nonnormal_rho_kappa

    print("\n" + "=" * 78)
    print("D5_nn — static parameter mobility (init zero for ρ_base/LoRA/μ)")
    print("=" * 78)
    print(f"{'Layer':<6} | {'|ρ_base|_F':<11} | {'|ρ_LoRA|_F':<11} | "
          f"{'|ψ_LoRA|_F':<11} | {'|μ|_F':<9} | {'mean μ':<10}")
    print("-" * 78)
    d5 = {}
    for i, layer in enumerate(encoder.layers):
        att = layer.att
        if not getattr(att, "use_nonnormal_rse", False):
            continue
        rb = att.nonnormal_rho_base.detach().float().cpu()
        rw1 = att.nonnormal_rho_w1.detach().float().cpu()
        rw2 = att.nonnormal_rho_w2.detach().float().cpu()
        mu = att.nonnormal_mu.detach().float().cpu()

        rb_f = rb.norm().item()
        rlora_f = (rw1.norm() * rw2.norm()).item()
        # Stage-9 static-ψ path has no ψ LoRA
        if hasattr(att, "nonnormal_psi_w1"):
            pw1 = att.nonnormal_psi_w1.detach().float().cpu()
            pw2 = att.nonnormal_psi_w2.detach().float().cpu()
            plora_f = (pw1.norm() * pw2.norm()).item()
        else:
            plora_f = float("nan")
        mu_f = mu.norm().item()
        mu_mean = mu.mean().item()
        print(f"L{i:<5}| {rb_f:.3e} | {rlora_f:.3e} | {plora_f:.3e} | "
              f"{mu_f:.3e} | {mu_mean:+.3e}")
        d5[i] = {
            "rho_base_fnorm": rb_f,
            "rho_lora_combined_fnorm": rlora_f,
            "psi_lora_combined_fnorm": plora_f,
            "mu_fnorm": mu_f,
            "mu_mean": mu_mean,
        }

    print("\n" + "=" * 78)
    print(f"D6_nn — realised ρ(x), ψ(x) per layer  (κ = {kappa:.2f})")
    print("=" * 78)
    print(f"{'Layer':<6} | {'mean|ρ|':<9} | {'p95|ρ|':<9} | "
          f"{'p99|ρ|':<9} | {'frac>0.9κ·λ':<12} | {'std ψ':<9}")
    print("-" * 78)
    d6 = {}
    for lid in sorted(storage.keys()):
        if not storage[lid]["rho_raw"]:
            continue
        att = encoder.layers[lid].att
        H = att.n_head; K = att.head_size; Bk = K // 2
        # Compute realised ρ = κ · softplus(λ̃_block) · tanh(ρ_raw)
        rho_chunks = []
        psi_chunks = []
        rho_over_bound_chunks = []
        for w_b, rho_b, psi_b in zip(
            storage[lid]["w"], storage[lid]["rho_raw"], storage[lid]["psi_raw"]
        ):
            # w_b: (N, T, H, K). Per-block λ̃ = mean over pair.
            lam_block = w_b.view(*w_b.shape[:-1], Bk, 2).mean(dim=-1)
            lam_soft = F.softplus(lam_block)
            rho_real = kappa * lam_soft * torch.tanh(rho_b.float())
            rho_chunks.append(rho_real.flatten())
            # Fraction saturating near clip boundary
            bound = kappa * lam_soft
            rho_over_bound_chunks.append((rho_real.abs() / (bound + 1e-12)).flatten())
            psi_chunks.append(psi_b.float().flatten())
        rho_all = torch.cat(rho_chunks).numpy()
        psi_all = torch.cat(psi_chunks).numpy()
        rob_all = torch.cat(rho_over_bound_chunks).numpy()
        mean_abs = float(np.mean(np.abs(rho_all)))
        p95 = float(np.percentile(np.abs(rho_all), 95))
        p99 = float(np.percentile(np.abs(rho_all), 99))
        frac_near = float(np.mean(rob_all > 0.9))
        std_psi = float(np.std(psi_all))
        print(f"L{lid:<5}| {mean_abs:.4f}   | {p95:.4f}   | {p99:.4f}   | "
              f"{frac_near:.4f}      | {std_psi:.3f}")
        d6[lid] = {
            "mean_abs_rho": mean_abs, "p95_abs_rho": p95, "p99_abs_rho": p99,
            "fraction_near_clip": frac_near, "std_psi": std_psi,
        }

    print("\n" + "=" * 78)
    print("D7_nn — non-normality score  ‖G^T G − G G^T‖_F / ‖G‖_F²  per layer")
    print("=" * 78)
    print(f"{'Layer':<6} | {'mean':<9} | {'p50':<9} | {'p95':<9} | {'p99':<9} | "
          f"{'max':<9}")
    print("-" * 78)
    d7 = {}
    for lid in sorted(storage.keys()):
        if not storage[lid]["rho_raw"]:
            continue
        att = encoder.layers[lid].att
        H = att.n_head; K = att.head_size; Bk = K // 2
        # Reconstruct G per (B, T, H, Bk) and compute score
        scores_all = []
        radii_all = []
        for w_b, rho_b, psi_b, th_b in zip(
            storage[lid]["w"], storage[lid]["rho_raw"], storage[lid]["psi_raw"],
            storage[lid]["theta"],
        ):
            # w_b is raw LoRA output (pre -exp); G wants the per-channel
            # log-decay which is -exp(w_raw).  compute_G_per_block handles.
            lam_block = (-torch.exp(w_b.float())).view(*w_b.shape[:-1], Bk, 2).mean(dim=-1)
            lam_soft = F.softplus(w_b.view(*w_b.shape[:-1], Bk, 2).mean(dim=-1).float())
            rho_real = kappa * lam_soft * torch.tanh(rho_b.float())
            G = compute_G_from_parts(lam_block, th_b.float(), rho_real, psi_b.float())
            score, radius = non_normality_and_spectrum(G)
            scores_all.append(score.flatten())
            radii_all.append(radius.flatten())
        s = torch.cat(scores_all).numpy()
        print(f"L{lid:<5}| {s.mean():.4f}   | {np.median(s):.4f}   | "
              f"{np.percentile(s, 95):.4f}   | {np.percentile(s, 99):.4f}   | {s.max():.4f}")
        d7[lid] = {
            "mean": float(s.mean()), "p50": float(np.median(s)),
            "p95": float(np.percentile(s, 95)),
            "p99": float(np.percentile(s, 99)),
            "max": float(s.max()),
        }

    print("\n" + "=" * 78)
    print("D8_nn — spectral radius max|eig(G)| per layer  (must stay < 1)")
    print("=" * 78)
    print(f"{'Layer':<6} | {'mean':<9} | {'p95':<9} | {'p99':<9} | {'max':<9} | "
          f"{'frac > 0.99':<12}")
    print("-" * 78)
    d8 = {}
    for lid in sorted(storage.keys()):
        if not storage[lid]["rho_raw"]:
            continue
        H = encoder.layers[lid].att.n_head
        K = encoder.layers[lid].att.head_size
        Bk = K // 2
        radii_all = []
        for w_b, rho_b, psi_b, th_b in zip(
            storage[lid]["w"], storage[lid]["rho_raw"], storage[lid]["psi_raw"],
            storage[lid]["theta"],
        ):
            lam_block = (-torch.exp(w_b.float())).view(*w_b.shape[:-1], Bk, 2).mean(dim=-1)
            lam_soft = F.softplus(w_b.view(*w_b.shape[:-1], Bk, 2).mean(dim=-1).float())
            rho_real = kappa * lam_soft * torch.tanh(rho_b.float())
            G = compute_G_from_parts(lam_block, th_b.float(), rho_real, psi_b.float())
            _, radius = non_normality_and_spectrum(G)
            radii_all.append(radius.flatten())
        rr = torch.cat(radii_all).numpy()
        frac_unstable = float(np.mean(rr > 0.99))
        print(f"L{lid:<5}| {rr.mean():.4f}   | {np.percentile(rr, 95):.4f}   | "
              f"{np.percentile(rr, 99):.4f}   | {rr.max():.4f}   | {frac_unstable:.4f}")
        d8[lid] = {
            "mean": float(rr.mean()),
            "p95": float(np.percentile(rr, 95)),
            "p99": float(np.percentile(rr, 99)),
            "max": float(rr.max()),
            "frac_gt_099": frac_unstable,
        }

    print("\n" + "=" * 78)
    print("D9_nn — per-head mean |ρ|  (cross-head specialisation)")
    print("=" * 78)
    print(f"{'Layer':<6} | {'per-head mean |ρ|':<40} | spread (max − min)")
    print("-" * 78)
    d9 = {}
    for lid in sorted(storage.keys()):
        if not storage[lid]["rho_raw"]:
            continue
        att = encoder.layers[lid].att
        H = att.n_head; K = att.head_size; Bk = K // 2
        per_head = []
        for w_b, rho_b in zip(storage[lid]["w"], storage[lid]["rho_raw"]):
            lam_block = w_b.view(*w_b.shape[:-1], Bk, 2).mean(dim=-1).float()
            lam_soft = F.softplus(lam_block)
            rho_real = kappa * lam_soft * torch.tanh(rho_b.float())
            # Shape (N, T, H, Bk) — reduce over (N, T, Bk) keeping H
            per_head.append(rho_real.abs().mean(dim=(0, 1, 3)))
        per_head_stack = torch.stack(per_head).mean(dim=0).numpy()
        spread = per_head_stack.max() - per_head_stack.min()
        ph_str = "[" + ", ".join(f"{v:.4f}" for v in per_head_stack) + "]"
        print(f"L{lid:<5}| {ph_str:<40} | {spread:.4f}")
        d9[lid] = {"per_head_mean_abs_rho": per_head_stack.tolist(),
                   "spread": float(spread)}

    # D10_nn — Stage-9 sparse gate values (if present)
    d10 = {}
    has_gate = any(hasattr(l.att, "sparse_nn_gate_raw") for l in encoder.layers)
    if has_gate:
        print("\n" + "=" * 78)
        print("D10_nn — Stage-9 per-(layer, head) sparse gate g = σ(raw)")
        print("=" * 78)
        print(f"{'Layer':<6} | {'per-head σ(raw)':<40} | {'min':<7} | {'max':<7}")
        print("-" * 78)
        for i, layer in enumerate(encoder.layers):
            att = layer.att
            if not hasattr(att, "sparse_nn_gate_raw"):
                continue
            raw = att.sparse_nn_gate_raw.detach().float().cpu()
            g = torch.sigmoid(raw).numpy()
            g_str = "[" + ", ".join(f"{v:.3f}" for v in g) + "]"
            print(f"L{i:<5}| {g_str:<40} | {g.min():.3f}  | {g.max():.3f}")
            d10[i] = {
                "gate_raw": raw.tolist(),
                "gate_sigmoid": g.tolist(),
                "gate_min": float(g.min()),
                "gate_max": float(g.max()),
                "gate_spread": float(g.max() - g.min()),
            }

    return {"D5_nn": d5, "D6_nn": d6, "D7_nn": d7, "D8_nn": d8, "D9_nn": d9,
            "D10_nn_gate": d10, "kappa": kappa}


def compute_G_from_parts(lam_block_log, theta, rho, psi):
    """lam_block_log is the log of decay (i.e. -λ for λ>0), shape (..., Bk)."""
    decay = torch.exp(lam_block_log)
    cos_t = torch.cos(theta); sin_t = torch.sin(theta)
    cosh_r = torch.cosh(rho); sinh_r = torch.sinh(rho)
    cos_2p = torch.cos(2.0 * psi); sin_2p = torch.sin(2.0 * psi)
    P00 = cosh_r + sinh_r * cos_2p
    P01 = sinh_r * sin_2p
    P11 = cosh_r - sinh_r * cos_2p
    G00 = decay * (P00 * cos_t + P01 * sin_t)
    G01 = decay * (-P00 * sin_t + P01 * cos_t)
    G10 = decay * (P01 * cos_t + P11 * sin_t)
    G11 = decay * (-P01 * sin_t + P11 * cos_t)
    G = torch.stack([
        torch.stack([G00, G01], dim=-1),
        torch.stack([G10, G11], dim=-1),
    ], dim=-2)
    return G


def diagnostic_delta(model, storage, cfg, state_info, out_dir: Path):
    encoder = model.encoder

    print("\n" + "=" * 78)
    print("D5_delta — `delta_recurrent_gate` mobility (zero init)")
    print("=" * 78)
    print(f"{'Layer':<6} | {'per-head g_δ':<40} | {'|g|_max':<9} | {'|g|_mean':<10}")
    print("-" * 78)
    d5 = {}
    for i, layer in enumerate(encoder.layers):
        att = layer.att
        if not getattr(att, "use_delta_rule", False):
            continue
        g = att.delta_recurrent_gate.detach().float().cpu()
        g_str = "[" + ", ".join(f"{v:+.4f}" for v in g) + "]"
        print(f"L{i:<5}| {g_str:<40} | {g.abs().max().item():.4e} | "
              f"{g.abs().mean().item():.4e}")
        d5[i] = {"per_head": g.tolist(),
                 "max_abs": float(g.abs().max()),
                 "mean_abs": float(g.abs().mean())}

    print("\n" + "=" * 78)
    print("D6_delta — realised β_eff(x) = g · iclr(x) distribution on dev")
    print("=" * 78)
    print(f"{'Layer':<6} | {'mean |β|':<10} | {'p95|β|':<10} | {'p99|β|':<10} | "
          f"{'max|β|':<10}")
    print("-" * 78)
    d6 = {}
    for lid in sorted(storage.keys()):
        if not storage[lid]["iclr"]:
            continue
        att = encoder.layers[lid].att
        g = att.delta_recurrent_gate.detach().float().cpu()
        H = att.n_head
        betas = []
        for iclr_b in storage[lid]["iclr"]:
            # iclr shape (B, H, T, K). g shape (H,). β = g·iclr broadcast.
            beta = g.view(1, H, 1, 1) * iclr_b.float()
            betas.append(beta.flatten())
        b_all = torch.cat(betas).numpy()
        print(f"L{lid:<5}| {np.mean(np.abs(b_all)):.4e} | "
              f"{np.percentile(np.abs(b_all), 95):.4e} | "
              f"{np.percentile(np.abs(b_all), 99):.4e} | {np.abs(b_all).max():.4e}")
        d6[lid] = {
            "mean_abs": float(np.mean(np.abs(b_all))),
            "p95_abs": float(np.percentile(np.abs(b_all), 95)),
            "p99_abs": float(np.percentile(np.abs(b_all), 99)),
            "max_abs": float(np.abs(b_all).max()),
        }

    print("\n" + "=" * 78)
    print("D7_delta — relative erase magnitude  ‖β kk (kk^T S)‖_F / ‖S‖_F")
    print("  (captured at steady-state via a second forward with state probes)")
    print("=" * 78)
    print("  (simplified: we report ‖β kk‖_F · ‖kk‖_2 as an upper bound proxy)")
    print(f"{'Layer':<6} | {'bound':<10}")
    print("-" * 78)
    d7 = {}
    for lid in sorted(storage.keys()):
        if not (storage[lid]["kk"] and storage[lid]["iclr"]):
            continue
        att = encoder.layers[lid].att
        g = att.delta_recurrent_gate.detach().float().cpu()
        H = att.n_head
        # β·kk shape (B, H, T, K); ||β·kk|| per token
        upper_bounds = []
        for kk_b, iclr_b in zip(storage[lid]["kk"], storage[lid]["iclr"]):
            beta = g.view(1, H, 1, 1) * iclr_b.float()
            bkk = beta * kk_b.float()
            # Per-token: ||β·kk||_2 · ||kk||_2 bounds the erase's operator norm
            bnorm = bkk.norm(dim=-1)          # (B, H, T)
            knorm = kk_b.float().norm(dim=-1)  # (B, H, T) ≈ 1 (normalised)
            upper_bounds.append((bnorm * knorm).flatten())
        ub = torch.cat(upper_bounds).numpy()
        print(f"L{lid:<5}| {ub.mean():.4e}")
        d7[lid] = {"mean_upper_bound": float(ub.mean())}

    return {"D5_delta": d5, "D6_delta": d6, "D7_delta": d7}


# ─────────────────────────────────────────────────────────────────────────
# Decision trees (the only place we interpret)
# ─────────────────────────────────────────────────────────────────────────

def decide_nonnormal(best_cer, diag):
    """Per the tree: ≤0.1174 break/signal; 0.1174-0.1210 ambiguous (diag decides);
    >0.1210 regression (diag refines)."""
    # Aggregate diagnostic signals
    max_mobility = max(
        (d["rho_lora_combined_fnorm"] for d in diag["D5_nn"].values()), default=0.0
    )
    max_rho_p95 = max(
        (d["p95_abs_rho"] for d in diag["D6_nn"].values()), default=0.0
    )
    max_nonnorm = max((d["mean"] for d in diag["D7_nn"].values()), default=0.0)
    max_spec = max((d["max"] for d in diag["D8_nn"].values()), default=0.0)
    max_frac_unstable = max((d["frac_gt_099"] for d in diag["D8_nn"].values()), default=0.0)

    # Decision:
    if best_cer <= 0.1161:
        leaf = "BREAK — replicate on 3 seeds"
    elif best_cer <= 0.1174:
        leaf = "SIGNAL — replicate on 3 seeds"
    elif best_cer <= 0.1210:
        # Ambiguous band — diagnostic decides
        if max_mobility > 0.5 and max_nonnorm > 0.1:
            leaf = ("AMBIGUOUS ENGAGED — mechanism moved (|LoRA_ρ|_F={:.2f}, "
                    "non-normality={:.3f}) but no CER break at this scale. "
                    "Mechanism-level finding, not next main run.").format(max_mobility, max_nonnorm)
        elif max_mobility < 0.1:
            leaf = ("AMBIGUOUS NO-MOBILITY — SGD did not engage ρ LoRA "
                    "(|LoRA_ρ|_F={:.3f}). One retry at κ=1.0 justified.").format(max_mobility)
        else:
            leaf = ("AMBIGUOUS MIXED — partial mobility; inspect per-layer D5/D7. "
                    "|LoRA_ρ|={:.2f}, non-normality={:.3f}.").format(max_mobility, max_nonnorm)
    else:  # > 0.1210
        if max_frac_unstable > 0.01 or max_spec > 1.02:
            leaf = ("REGRESSION STABILITY — spectral radius max={:.3f}, "
                    "{:.1%} of G have |λ|>0.99. Fix stability, rerun.").format(max_spec, max_frac_unstable)
        elif max_nonnorm > 0.5:
            leaf = ("REGRESSION HIGH NON-NORMALITY — non-normality={:.3f} "
                    "seems to destabilise. Reduce κ and retry.").format(max_nonnorm)
        else:
            leaf = ("REGRESSION REJECT FAMILY — stable but worse-than-anchor CER. "
                    "Reject the 2×2 polar-Lie family at this scale.")
    return leaf, {
        "max_rho_lora_fnorm": max_mobility,
        "max_p95_abs_rho": max_rho_p95,
        "max_non_normality_mean": max_nonnorm,
        "max_spectral_radius": max_spec,
        "max_frac_spectral_unstable": max_frac_unstable,
    }


def decide_delta(best_cer, diag):
    """≤0.1230 signal; 0.1230-0.1286 ambiguous; >0.1286 close-axis."""
    max_g_mobility = max((d["max_abs"] for d in diag["D5_delta"].values()), default=0.0)
    max_beta = max((d["p95_abs"] for d in diag["D6_delta"].values()), default=0.0)

    if best_cer <= 0.1230:
        leaf = "SIGNAL — recurrent delta helps; replicate."
    elif best_cer <= 0.1286:
        if max_g_mobility < 1e-3:
            leaf = ("AMBIGUOUS NO-MOBILITY — g_δ stayed near zero (max={:.2e}); "
                    "mechanism was not engaged. One retry at higher init or "
                    "constraint relaxation justified.").format(max_g_mobility)
        else:
            leaf = ("AMBIGUOUS ENGAGED — g_δ moved (max={:.3f}), β_eff p95={:.3f}; "
                    "mechanism engaged, no CER break. Mechanism-level null.").format(
                    max_g_mobility, max_beta)
    else:
        if max_g_mobility < 1e-3:
            leaf = ("CLOSED but NOT ENGAGED — worse-than-vanilla with g_δ≈0. "
                    "Mechanism was inactive; investigate wiring.")
        else:
            leaf = ("CLOSE DELTA AXIS — stable, engaged, worse. "
                    "g_δ max={:.3f}, β p95={:.3f}. Not productive at this scale.").format(
                    max_g_mobility, max_beta)
    return leaf, {
        "max_g_delta_mobility": max_g_mobility,
        "max_beta_eff_p95": max_beta,
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--n-utt", type=int, default=N_UTT_DEFAULT)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "stage8_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "best_model.pt"

    cfg = load_config(str(run_dir / "config.yaml"), {})
    vocab = CharVocab.build_english()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    backbone = cfg.backbone
    print(f"Backbone: {backbone}")
    print(f"Checkpoint: {ckpt}")

    if "nonnormal_rse" in backbone:
        backbone_kind = "nonnormal"
    elif "delta" in backbone:
        backbone_kind = "delta"
    else:
        raise SystemExit(f"Backbone '{backbone}' not recognised for Stage-8 diagnostic.")

    model = ASRModel(vocab_size=vocab.size, cfg=cfg).to(device)
    state = load_checkpoint(ckpt, model=model, map_location=device, restore_rng=False)
    model.eval()
    best_cer = float(state.get("best_cer"))
    print(f"Epoch: {state.get('epoch')}  best dev CER: {best_cer:.4f}")

    storage: dict = {}
    undo = install_capture_hooks(model.encoder, storage, backbone_kind)

    try:
        dev_entries = load_librispeech("dev", cfg.data_cache_dir,
                                       cfg.min_audio_sec, cfg.max_audio_sec)[:args.n_utt]
        dev_ds = ASRDataset(dev_entries, vocab, cfg)
        dev_loader = DataLoader(
            dev_ds,
            batch_sampler=DurationBatchSampler(dev_entries, cfg.batch_max_seconds, False, cfg.seed),
            collate_fn=collate_fn, num_workers=2,
        )
        n_processed = 0
        with torch.no_grad():
            for batch in dev_loader:
                mels, _, mel_lens, _ = batch
                mels = mels.to(device); mel_lens = mel_lens.to(device)
                try:
                    _ = model(mels, mel_lens)
                except Exception as e:
                    print(f"batch failed: {e}"); continue
                n_processed += mels.size(0)
                if n_processed >= args.n_utt: break
        print(f"Processed {n_processed} utterances")
    finally:
        remove_hooks(model.encoder, undo)

    if backbone_kind == "nonnormal":
        diag = diagnostic_nonnormal(model, storage, cfg, state, out_dir)
        leaf, aggregates = decide_nonnormal(best_cer, diag)
    else:
        diag = diagnostic_delta(model, storage, cfg, state, out_dir)
        leaf, aggregates = decide_delta(best_cer, diag)

    print("\n" + "=" * 78)
    print("DECISION-TREE LEAF")
    print("=" * 78)
    print(leaf)
    print("=" * 78)

    summary = {
        "backbone": backbone,
        "checkpoint": str(ckpt),
        "best_cer": best_cer,
        "epoch": state.get("epoch"),
        "n_utterances": n_processed,
        "diagnostic": diag,
        "aggregates": aggregates,
        "decision_tree_leaf": leaf,
    }
    with open(out_dir / "stage8_diag_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {out_dir / 'stage8_diag_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
