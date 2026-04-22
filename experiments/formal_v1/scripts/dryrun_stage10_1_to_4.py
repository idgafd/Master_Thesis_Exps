#!/usr/bin/env python3
"""Phase-3 dry-run profile for Stage 10.1–10.4.

For each of the four new backbones:
  1. Build the model.
  2. Zero-regression-at-init check: full-model forward output == baseline
     (rwkv6 vanilla) output on identical weights (i.e., the new mechanism is
     inert at step 0). Uses max-absolute-difference per §1.1 of STAGE10_PLAN.
  3. Forward / backward sanity on a dummy batch at the plan's canonical
     profiling shape (B=10, T_mel=1200).
  4. Peak VRAM on each of 2 GPUs.
  5. Throughput: 10 iterations, report mean + std ms/iter.
  6. Project per-epoch and full 30-epoch wallclock from the step time and the
     LibriSpeech clean-100 dataset size, using the same steps/epoch estimate
     the training runner uses.

Does NOT start full training. Halts after profiling per Phase-3 instruction.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.config import ExperimentConfig, load_config
from src.data.vocab import CharVocab
from src.models.asr_model import ASRModel
from src.utils.misc import seed_everything, count_parameters

# ── Canonical profile shape from STAGE10_PLAN §7.2 ────────────────────────
# B=10, T_mel=1200 is the timing reference column in the §9.1 master table.
PROFILE_B = 10
PROFILE_T_MEL = 1200

# ── Dataset size for 30-epoch projection ──────────────────────────────────
# LibriSpeech train-clean-100: 28 539 utterances at cfg.max_audio_sec=20, mean
# ≈ 12.3 s. With batch_max_seconds=300, mean batch ≈ 24 utterances → roughly
# 1190 steps/epoch. We read from the existing history.csv of `rwkv6_rse_strong_viscosity`
# if present; otherwise fall back to this estimate.
FALLBACK_STEPS_PER_EPOCH = 1190
EPOCHS_PLANNED = 30


def _infer_steps_per_epoch() -> int:
    """Look for an existing run's steps/epoch; fall back to a known estimate."""
    for run_dir in (
        "outputs/rwkv6_rse_strong_viscosity_seed42",
        "outputs/rwkv6_seed42",
        "outputs/exp02_rwkv6_seed42",
    ):
        csv = os.path.join(run_dir, "history.csv")
        if os.path.exists(csv):
            try:
                import pandas as pd
                df = pd.read_csv(csv)
                if "last_step" in df.columns and len(df) >= 2:
                    s = int(df["last_step"].iloc[1] - df["last_step"].iloc[0])
                    if s > 0:
                        return s
            except Exception:
                pass
    return FALLBACK_STEPS_PER_EPOCH


def _build(backbone: str, seed: int = 42) -> tuple[ASRModel, ExperimentConfig]:
    cfg = load_config("configs/default.yaml", {"backbone": backbone, "seed": seed})
    vocab = CharVocab.build_english()
    seed_everything(cfg.seed)
    model = ASRModel(vocab_size=vocab.size, cfg=cfg)
    return model, cfg


def _dummy_batch(cfg: ExperimentConfig, device, B: int = PROFILE_B, T_mel: int = PROFILE_T_MEL):
    mels = torch.randn(B, cfg.n_mels, T_mel, device=device)
    # One full-length utterance, rest slightly shorter to exercise masking.
    lengths = torch.full((B,), T_mel, device=device, dtype=torch.long)
    lengths[1:] = torch.randint(int(0.6 * T_mel), T_mel + 1, (B - 1,), device=device)
    return mels, lengths


def _compute_loss(log_probs, out_lengths, B: int, vocab_size: int):
    """Dummy CTC loss against random targets — only used to exercise backward."""
    device = log_probs.device
    T_out = log_probs.size(1)
    # Random targets of length T_out // 3 per sample, ensuring targets ≤ outputs.
    target_lengths = torch.full((B,), max(1, T_out // 3), device=device, dtype=torch.long)
    targets = torch.randint(
        1, vocab_size, (int(target_lengths.sum().item()),), device=device, dtype=torch.long
    )
    log_probs_tf = log_probs.transpose(0, 1)  # (T, B, V) for CTC
    loss = nn.functional.ctc_loss(
        log_probs_tf, targets, out_lengths, target_lengths,
        blank=0, reduction="mean", zero_infinity=True,
    )
    return loss


def _zero_regression_check(backbone: str, baseline_bb: str = "rwkv6", seed: int = 42):
    """Full-model at init: new backbone output ≈ baseline output on identical weights."""
    print(f"  ── zero-regression-at-init check: {backbone} vs {baseline_bb} ──")
    device = torch.device("cuda:0")

    # Build both with the same seed; then copy all shared param values from
    # baseline into the new backbone so their RNGs diverge only on new params.
    mdl_base, cfg_base = _build(baseline_bb, seed=seed)
    mdl_new, cfg_new = _build(backbone, seed=seed)

    # Copy shared state_dict entries from base → new. Ignore new params (they
    # are expected to be zero-init by design: λ-LoRA, m2rnn-λ, α-bypass, etc.)
    base_sd = mdl_base.state_dict()
    new_sd = mdl_new.state_dict()
    shared = {k: v for k, v in base_sd.items() if k in new_sd and new_sd[k].shape == v.shape}
    missing_in_new = [k for k in base_sd if k not in new_sd]
    new_only = [k for k in new_sd if k not in base_sd]
    new_sd_updated = dict(new_sd)
    new_sd_updated.update(shared)
    mdl_new.load_state_dict(new_sd_updated, strict=True)

    print(f"    shared state_dict entries: {len(shared):,}; base-only: {len(missing_in_new)}; new-only: {len(new_only)}")

    mdl_base = mdl_base.to(device).eval()
    mdl_new = mdl_new.to(device).eval()

    torch.manual_seed(0)
    mels, lengths = _dummy_batch(cfg_base, device, B=2, T_mel=500)
    with torch.no_grad():
        y_base, _, _ = mdl_base(mels, lengths)
        y_new, _, _ = mdl_new(mels, lengths)

    # Only compare positions within the shorter of the two output lengths.
    # Output lengths from the convolutional frontend are deterministic per
    # input length, so y_base and y_new have the same shape here.
    diff = (y_base - y_new).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"    max |diff| = {max_diff:.3e}   mean |diff| = {mean_diff:.3e}")

    # Tolerance: fp32 CTC log-softmax on a 6-layer net accumulates ~1e-4
    # worst-case. Accept up to 1e-4 as "bit-exact within numerical noise".
    tol = 1e-4
    status = "PASS" if max_diff < tol else "FAIL"
    print(f"    {status}  (tolerance: {tol:.0e})")

    del mdl_base, mdl_new
    torch.cuda.empty_cache()
    return max_diff


def _impulse_response_check():
    """Stage 10.3 — validate the actual impulse response of MultiDilationDWConvShift.

    At init with α = [1, 0, 0, 0] and the dilation-1 branch weights, the
    module's output on an impulse x[t] = δ_{t,t0} must match:
      - causal mode:    0.5·δ_{t,t0} + 0.5·δ_{t, t0+1}   (output at t0, t0+1 = 0.5 each)
      - symmetric mode: 0.5·δ_{t, t0-1} + 0.5·δ_{t, t0+1}

    This validates the kernel's indexing end-to-end, not just the weight slots.
    """
    print("  ── impulse-response check: MultiDilationDWConvShift (causal + symmetric) ──")
    import torch as _t
    from src.models.mechanisms.conv_shift import MultiDilationDWConvShift

    failures = []
    D, T = 8, 20
    t0 = 5
    for mode in ("causal", "symmetric"):
        mod = MultiDilationDWConvShift(D, kernel_size=3, dilations=(1, 2, 4, 8), padding_mode=mode).eval()
        x = _t.zeros(1, T, D)
        x[0, t0, :] = 1.0  # impulse at t = t0, all channels
        with _t.no_grad():
            y = mod(x).squeeze(0)[:, 0]  # (T,) — first channel (identical for all in DW conv)

        y_np = y.tolist()
        if mode == "causal":
            # Expected: y[t0] = 0.5, y[t0+1] = 0.5, rest 0.
            exp = [0.5 if t in (t0, t0 + 1) else 0.0 for t in range(T)]
        else:
            # Expected: y[t0-1] = 0.5, y[t0+1] = 0.5, rest 0.
            exp = [0.5 if t in (t0 - 1, t0 + 1) else 0.0 for t in range(T)]

        nonzero = [(i, y_np[i]) for i in range(T) if abs(y_np[i]) > 1e-6]
        expected_nonzero = [(i, exp[i]) for i in range(T) if abs(exp[i]) > 1e-6]
        diff_max = max(abs(a - b) for a, b in zip(y_np, exp))

        ok = diff_max < 1e-6
        status = "PASS" if ok else "FAIL"
        print(f"    {mode:10s}  actual nonzero: {nonzero}  expected: {expected_nonzero}  diff: {diff_max:.2e}  [{status}]")
        if not ok:
            failures.append(mode)

    if failures:
        print(f"    FAIL modes: {failures}")
    else:
        print(f"    all modes PASS — impulse response matches plan semantics")


def _structural_convshift_multidil_check():
    """Stage 10.3 — verify α-init is (1, 0, 0, 0) and only branch 0 has nonzero weights.

    The plan's contract is 'reduces to single-dilation causal ConvShift at init',
    NOT 'reduces to vanilla RWKV-6' — so a direct output match against `rwkv6`
    is not meaningful (vanilla has no ConvShift path).  Instead assert the
    structural invariant at the module level AND run an impulse-response test.
    """
    print("  ── structural check: rwkv6_convshift_multidil init ──")
    mdl, cfg = _build("rwkv6_convshift_multidil")
    fail = False
    for i, layer in enumerate(mdl.encoder.layers):
        cm = layer.att.conv_shift_module
        alpha = cm.alpha.detach()
        # α_1 (dilation=1 at index 0 of (1,2,4,8)) must be 1.0; others 0.
        if not (alpha[0].item() == 1.0 and (alpha[1:].abs() < 1e-7).all()):
            print(f"    L{i}: α init wrong → {alpha.tolist()}")
            fail = True
        # Only branch 0 has nonzero weights.
        for b_idx, branch in enumerate(cm.branches):
            w = branch.weight.detach()
            wmax = w.abs().max().item()
            if b_idx == 0:
                if wmax < 1e-6:
                    print(f"    L{i}: branch 0 weights are ~0 → wmax={wmax}")
                    fail = True
            else:
                if wmax > 1e-7:
                    print(f"    L{i}: branch {b_idx} (dilation={cm.dilations[b_idx]}) has nonzero weights at init")
                    fail = True
    print(f"    {'FAIL' if fail else 'PASS'}  (structural contract: only dilation-1 branch active at init)")
    # Now the end-to-end impulse test (doesn't depend on backbone build).
    _impulse_response_check()


def _profile_backbone(backbone: str, n_iters: int = 10, warmup: int = 2):
    print(f"\n══════ {backbone} ══════")
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    print(f"  Visible GPUs: {len(devices)}")
    for d in devices:
        print(f"    gpu{d.index}: {torch.cuda.get_device_name(d)}")

    # Parameter accounting on cpu first
    mdl, cfg = _build(backbone, seed=42)
    pc = count_parameters(mdl)
    print(f"  Params: total={pc['total']:,}  encoder={pc['encoder']:,}")
    del mdl
    torch.cuda.empty_cache()

    # Zero-regression check. 10.3 / 10.3-sym have a different baseline
    # (reduces to single-dilation ConvShift, not vanilla RWKV-6) — use a
    # structural + impulse-response check.
    if backbone in ("rwkv6_convshift_multidil", "rwkv6_convshift_multidil_symmetric"):
        _structural_convshift_multidil_check()
    elif backbone != "rwkv6":
        _zero_regression_check(backbone)

    # Per-GPU forward/backward profile
    results_per_gpu = []
    for gpu_idx, device in enumerate(devices):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        mdl, cfg = _build(backbone, seed=42)
        mdl = mdl.to(device).train()
        mels, lengths = _dummy_batch(cfg, device)
        vocab_size = mdl.ctc_head.proj.out_features

        print(f"\n  ── gpu{gpu_idx} forward/backward sanity (B={PROFILE_B}, T_mel={PROFILE_T_MEL}) ──")
        # Single warmup + correctness
        try:
            log_probs, out_lengths, _ = mdl(mels, lengths)
            loss = _compute_loss(log_probs, out_lengths, PROFILE_B, vocab_size)
            loss.backward()
            print(f"    fwd ok  output shape: {tuple(log_probs.shape)}  loss: {loss.item():.3f}")
        except Exception as e:
            print(f"    FAILED: {e!r}")
            del mdl
            torch.cuda.empty_cache()
            continue

        # Warmup iterations (CUDA kernel compilation, caching, etc.)
        for _ in range(warmup):
            mdl.zero_grad(set_to_none=True)
            log_probs, out_lengths, _ = mdl(mels, lengths)
            loss = _compute_loss(log_probs, out_lengths, PROFILE_B, vocab_size)
            loss.backward()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        # Timed iterations
        print(f"  ── gpu{gpu_idx} throughput ({n_iters} iter) ──")
        step_times_ms = []
        t0_total = time.time()
        for it in range(n_iters):
            mdl.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            t0 = time.time()
            log_probs, out_lengths, _ = mdl(mels, lengths)
            loss = _compute_loss(log_probs, out_lengths, PROFILE_B, vocab_size)
            loss.backward()
            torch.cuda.synchronize(device)
            step_times_ms.append((time.time() - t0) * 1000)
        total_sec = time.time() - t0_total

        import statistics
        mean_ms = statistics.mean(step_times_ms)
        stdev_ms = statistics.stdev(step_times_ms) if len(step_times_ms) > 1 else 0.0
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"    mean: {mean_ms:7.1f} ms/iter   stdev: {stdev_ms:.1f}   total: {total_sec:.1f}s")
        print(f"    peak VRAM gpu{gpu_idx}: {peak_mem_gb:.2f} GB")

        results_per_gpu.append({
            "gpu": gpu_idx,
            "mean_ms": mean_ms,
            "stdev_ms": stdev_ms,
            "peak_mem_gb": peak_mem_gb,
            "n_iters": n_iters,
        })

        del mdl
        torch.cuda.empty_cache()

    return {
        "backbone": backbone,
        "params": pc,
        "per_gpu": results_per_gpu,
    }


def _project_wallclock(mean_ms: float, steps_per_epoch: int, n_epochs: int = EPOCHS_PLANNED):
    """Project per-epoch and total time from ms/iter."""
    sec_per_epoch = mean_ms * steps_per_epoch / 1000.0
    total_sec = sec_per_epoch * n_epochs
    return sec_per_epoch, total_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+",
                        default=["rwkv6", "rwkv6_loglinear", "rwkv6_m2rnn_sparse",
                                 "rwkv6_convshift_multidil", "rwkv6_chanmix_bypass"])
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", default="outputs/stage10_dryrun.json")
    args = parser.parse_args()

    print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  gpu{i}: {torch.cuda.get_device_name(i)}  mem: {torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB")

    steps_per_epoch = _infer_steps_per_epoch()
    print(f"\nSteps/epoch: {steps_per_epoch}   Planned epochs: {EPOCHS_PLANNED}\n")

    all_results = []
    for bb in args.backbones:
        try:
            result = _profile_backbone(bb, n_iters=args.n_iters, warmup=args.warmup)
        except Exception as e:
            print(f"\n{bb} FAILED: {e!r}")
            import traceback
            traceback.print_exc()
            continue
        all_results.append(result)

    # Summary table with projections
    print("\n\n╔══════════════════════════════════════════════════════════════════════════════════╗")
    print("║  Stage 10.1–10.4 dry-run summary (B=10, T=1200 mels, 2× RTX PRO 6000 Blackwell)  ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════╝")
    print(f"{'Backbone':<32} {'enc_params':>10} {'ms/iter':>10} {'VRAM GB':>8} {'sec/ep':>8} {'30ep (h)':>10}")
    print("-" * 82)

    # Pull baseline ms for speedup ratios
    baseline_ms = None
    for res in all_results:
        if res["backbone"] == "rwkv6" and res["per_gpu"]:
            baseline_ms = res["per_gpu"][0]["mean_ms"]

    for res in all_results:
        bb = res["backbone"]
        params = res["params"]["encoder"]
        if not res["per_gpu"]:
            print(f"{bb:<32} {params:>10,}   [PROFILE FAILED]")
            continue
        # Take gpu0 as the canonical timing (matches plan convention).
        gpu0 = res["per_gpu"][0]
        sec_per_epoch, total_sec = _project_wallclock(gpu0["mean_ms"], steps_per_epoch)
        ratio = f" ({gpu0['mean_ms']/baseline_ms:.1f}×)" if baseline_ms else ""
        print(
            f"{bb:<32} {params:>10,} {gpu0['mean_ms']:>10.1f} {gpu0['peak_mem_gb']:>8.2f} "
            f"{sec_per_epoch:>8.0f} {total_sec/3600:>10.2f}{ratio}"
        )

    # Save JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "profile_shape": {"B": PROFILE_B, "T_mel": PROFILE_T_MEL},
            "steps_per_epoch": steps_per_epoch,
            "n_epochs_planned": EPOCHS_PLANNED,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved JSON results to {args.output}")
    print("\nPhase 3 complete — HALTING per Phase-3 instruction. Awaiting Optimization Reviewer audit.")


if __name__ == "__main__":
    main()
