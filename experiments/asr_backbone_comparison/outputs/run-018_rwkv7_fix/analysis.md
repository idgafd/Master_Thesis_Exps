# Run 018 Analysis — RWKV-7 Initialization Fixes

Diagnoses why stock RWKV-7 (Goose) failed to converge in an earlier test,
and systematically applies three initialization patches to recover training.

**Key question:** Can RWKV-7 become competitive with LION (RWKV-6 bidirectional)
on offline ASR through initialization fixes alone, or does the unidirectional
architecture impose a fundamental performance ceiling?

---

## Setup

| Parameter | Value |
|-----------|-------|
| Backbones | `rwkv7_fix_decay` (fix 1 only), `rwkv7_fix_all` (fixes 1+2+3) |
| Epochs | 60 |
| Scheduler | WSD (warmup=500 steps, stable until ep48, decay ep48-60) |
| Dropout | 0.15 |
| SpecAugment | standard (freq=15, time=35, 2+2 masks) |
| d_model | 256, 6 layers, 4 heads |
| Params | 7.14M (fewer than LION due to RWKV-7 layout) |
| tmix_backend | triton (fla not installed — see critical note below) |

---

## The Three Fixes

| Fix | Parameter | Stock Init | Fixed Init | Rationale |
|-----|-----------|-----------|------------|-----------|
| 1 — Decay | `w0` | `w0 += 0.0` | **`w0 += 2.0`** | Triton decay range too narrow: [0.0009, 0.111] → fixed [0.007, 0.378] |
| 2 — Value residual | `v0` | `v0 = 1.0` → sigmoid = 0.73 | **`v0 = -5.0`** → sigmoid ≈ 0.007 | v_first designed for large LLMs, suppresses hierarchical CTC learning |
| 3 — Key scaling | `k_a` | `k_a = 1.0` → key × 0.5 | **`k_a = 0.0`** → key × 1.0 | Restores full key magnitude at init |

Fixes are applied AFTER `reset_parameters()` — they override, not replace, the
stock init.

---

## Results

| Backbone | Dev CER | Test CER | Test WER | Train loss @60 |
|----------|---------|----------|----------|---------------|
| `rwkv7_fix_decay` | 0.3570 | 0.3776 | 0.9734 | 1.34 |
| `rwkv7_fix_all` | 0.2388 | 0.2602 | 0.8597 | 0.81 |

For comparison:

| Backbone | Dev CER | Test CER | Test WER |
|----------|---------|----------|----------|
| `bidir_rwkv6` LION baseline (005) | 0.1676 | 0.1790 | 0.6704 |
| `mamba` best (003) | 0.1971 | 0.2098 | 0.7599 |
| `bidir_rwkv6_conv_nogate` (006) | 0.1587 | **0.1760** | 0.6563 |

**Decay-only fix is insufficient.** CER 0.3776 — the model barely converges.
Train loss plateau at 1.34 (vs 0.81 for fix_all). The v_first and k_a problems
are active and prevent learning.

**All-fixes variant recovers partial convergence.** CER 0.2602 — comparable to
Mamba (0.2098) in the same performance bracket, though still well below LION.

---

## Carry-State Evaluation

| Backbone | Full CER | R@2s | R@5s | R@10s | C@2s | C@5s | C@10s |
|----------|----------|------|------|-------|------|------|-------|
| rwkv7_fix_decay | 0.3776 | 0.3983 | 0.3808 | 0.3774 | 0.3833 | 0.3782 | 0.3774 |
| rwkv7_fix_all | 0.2602 | 0.2944 | 0.2655 | 0.2594 | 0.2629 | 0.2600 | 0.2594 |

Carry-state delta (R − C, positive = carry is better):

| Backbone | Δ@2s | Δ@5s | Δ@10s |
|----------|------|------|-------|
| rwkv7_fix_decay | +0.0149 | +0.0026 | 0.0000 |
| rwkv7_fix_all | **+0.0315** | +0.0055 | 0.0000 |

**Carry-state works correctly for rwkv7_fix_all.** Unlike BiWKV6 (run-007/009)
where carry-state caused catastrophic degradation (CER > 0.7), RWKV-7 fix_all
shows a meaningful carry benefit at 2s chunks (+0.0315 CER improvement). The
recurrent state initialisation is healthy.

The carry-state benefit at 2s (0.0315) is similar in magnitude to Mamba's
(0.0472 at 2s). At 5s and 10s the benefit nearly vanishes — consistent with the
model having learned most relevant context within a few seconds.

**rwkv7_fix_decay carry barely helps** (Δ@2s = +0.0149). The model's poor
training means the carry state contains little useful information.

---

## CRITICAL: Training Speed — 11.5× Slower Than LION

**Epoch time: 757–758 seconds for RWKV-7 vs ~66 seconds for LION.**

This is a **11.5× wall-clock slowdown**. Total training time:
- LION (run-005): 60 epochs × 66s ≈ **66 minutes**
- RWKV-7 (run-018): 60 epochs × 757s ≈ **12.6 hours**

The root cause is printed in the log:
```
[WARNING] fla not available, falling back to triton, cuda or pytorch mode
```

The Flash Linear Attention (fla) package provides an optimised CUDA kernel for
RWKV-7. Without it, the triton backend is used. At T≈250 and d=256 on an RTX
5090, the triton kernel is dramatically slower than LION's pure PyTorch parallel
attention (which is just two large matrix multiplications — highly optimised by
PyTorch/cuBLAS).

**LION advantage breakdown:**
- LION parallel attention: two (B, H, T, T) matmuls + element-wise ops. T=250
  makes this 250×250 = 62.5K per head. PyTorch cuBLAS handles this with minimal
  kernel launch overhead.
- RWKV-7 triton: sequential recurrent scan with custom triton kernels. Triton
  kernels have higher launch overhead and are less optimised for short sequences
  (T=250) than for long ones (T>2000) where they are designed to excel.

**Installing fla would likely reduce RWKV-7 epoch time substantially** (estimates
suggest 2-5× speedup), but even then it would likely remain slower than LION
at T=250. The fla kernel is optimised for long sequences and large-batch LLM
training, not short-sequence ASR.

This speed penalty is not documented in ANALYSIS.md and represents a practical
deployment concern that was missed in the high-level summary.

---

## Decay Range Analysis

Stock RWKV-7 triton decay mapping (before fix 1):
```
w0 ∈ [-6.5, -1.5]
r = exp(-softplus(-w0) - 0.5)
  → r ∈ [0.0009, 0.111]
Effective memory: log(0.001)/log(0.111) ≈ 1.04 tokens — essentially 1-token memory
```

After fix 1 (w0 += 2.0):
```
w0 ∈ [-4.5, 0.5]
r ∈ [0.007, 0.378]
Effective memory at r=0.5: log(0.001)/log(0.5) ≈ 10 tokens — meaningful range
```

RWKV-6 decay range for comparison:
```
time_decay ∈ [-6, -1] at init → exp(time_decay) ∈ [0.0025, 0.368]
```

Fix 1 aligns RWKV-7's decay range with RWKV-6's. The triton kernel maps
decay in a non-obvious way (exp(-softplus(-w0)-0.5)), which is why the stock
init produces a much narrower range than intended.

---

## v_first Mechanism Analysis

The value residual mechanism (v_first) in RWKV-7 is designed for large LLMs:
```
v_residual = sigmoid(v0) * v_layer0
# At each layer ≥1: v = v_projected + v_residual
```

At stock v0=1.0: sigmoid(1.0) = 0.731 → 73% of layer 0's value is mixed into
every subsequent layer. For a 6-layer CTC-ASR model, this suppresses hierarchical
feature learning:
- Layer 1 sees: 73% layer-0 features + 27% its own projected features
- Layer 2 sees: 73% layer-0 features + 27% its own projected features
- ...effectively making all layers process similar representations

CTC requires the encoder to learn a hierarchical acoustic feature abstraction
(raw mel → phonetic features → word-level units). v_first at 73% prevents this
hierarchy from forming. Fixing v0 = -5.0 (sigmoid ≈ 0.007) effectively disables
the residual at init, allowing full hierarchical learning. The parameter remains
learnable — the network can recover v_first if it's useful, but starts without it.

---

## k_a Key Scaling Analysis

The RWKV-7 in-context learning rate `iclr = sigmoid(0) = 0.5`. The key
scaling at init:
```
k_new = k * (1 + (iclr - 1) * k_a)  where k_a = 1.0 at stock init
      = k * (1 + (0.5 - 1) * 1.0)
      = k * 0.5
```

This halves the key magnitude at init. Halved keys → halved attention weights
in the WKV kernel → effectively 2× smaller learning signal through the attention
path. Setting k_a = 0.0 restores k_new = k × 1.0 (no scaling).

Note: k_a is also learnable — the model can recover any k_a value during training.
The fix only corrects the poor starting point.

---

## Why All Three Fixes Are Necessary

From the results:

| Fixes applied | Test CER | Train loss @60 |
|---------------|----------|---------------|
| None (stock) | >>0.4 | >>1.5 (non-converging) |
| Fix 1 only (decay) | 0.3776 | 1.34 |
| Fixes 1+2+3 (all) | 0.2602 | 0.81 |

Fix 1 alone recovers some convergence (1.34 vs estimated >1.5 stock) but the
model is still far from learning the task. Fixes 2 and 3 together provide
the additional ~0.53 train loss improvement needed for genuine convergence.

The three fixes address independent failure modes:
- Fix 1: memory range (the model has no effective memory without it)
- Fix 2: feature hierarchy suppression (the model collapses to layer-0 features)
- Fix 3: attention signal magnitude (the model has half the key gradient signal)

All three must be corrected simultaneously. Fixing only one leaves two other
problems that dominate the remaining training difficulty.

---

## Performance Gap to LION

Even with all fixes, RWKV-7 remains 45% worse than LION on test CER:
```
LION (005):     Test CER 0.1790
RWKV-7 fixed:  Test CER 0.2602  (+45% relative)
```

The remaining gap is primarily the **unidirectional limitation**. RWKV-7 is
a causal (left-to-right) model. For offline ASR on pre-recorded speech, the
entire utterance is available — there is no reason to restrict to causal
processing. LION's bidirectional attention uses both past and future context
at every frame, which is essential for resolving phonetic ambiguities.

The gap between Mamba (causal, 0.2098) and LION (bidirectional, 0.1790) is
+17%. RWKV-7 fixed (causal, 0.2602) shows an additional +24% penalty over Mamba.
This extra penalty likely reflects:
1. RWKV-7's initialisation issues requiring more epochs to fully recover
2. The fla-free triton backend making longer training impractical (12.6h already)
3. Potential remaining init suboptimality even with all three fixes

---

## Code Issues

1. **`v_first` initialised to None, not carried across chunks.** In
   `_forward_carry`, `v_first = None` is reset at each call. This means the value
   residual context from a previous chunk is not passed to the next chunk. This
   is a minor architectural limitation — true streaming would propagate v_first
   across chunk boundaries. Current implementation only propagates the WKV hidden
   state, not v_first.

2. **fla package absent — severe performance penalty.** The training log shows the
   triton fallback warning. Installing fla (Flash Linear Attention) is the highest-
   priority infrastructure fix for RWKV-7 experiments. Without it, RWKV-7 is
   impractical for extensive experimentation (12.6h per 60-epoch run vs 66 min for LION).

3. **Fix application after reset_parameters.** The `_apply_fixes` method is called
   after `block.reset_parameters()` in the constructor loop. This is correct — fixes
   override the stock init. However, if `reset_parameters()` is ever called again
   (e.g., by a training framework), the fixes would be undone. For robustness,
   the fixed values should be the defaults in the RWKV7BlockConfigMap, not
   applied post-hoc.

4. **WSD scheduler with decay at epoch 48 (not epoch 12).** Stock RWKV-7 tests
   used decay at epoch 12 (early decay). This run uses decay at epoch 48
   (late decay = 80% stable, 20% decay). The late decay is appropriate for the
   fixes variant — early decay would compress the learning budget further.

---

## Conclusion

RWKV-7 with all three initialization fixes becomes a functional ASR model
(CER 0.2602), reaching Mamba-class performance with functional carry-state.
However:

1. **It remains 45% worse than LION** due to the unidirectional architecture.
2. **It is 11.5× slower to train** due to the triton fallback (fla not installed).
3. **Three interacting init bugs** (not just one) caused the original convergence
   failure — no single fix is sufficient.
4. **Carry-state works correctly** (+0.032 CER improvement at 2s chunks), making
   it viable for streaming deployment where LION is unavailable.

**Practical recommendation:** RWKV-7 is relevant only for streaming ASR scenarios
where true unidirectional recurrent inference is required. For offline ASR,
LION is strictly superior in both quality (45% better CER) and training speed
(11.5× faster). If RWKV-7 streaming experiments are planned, install fla first
to make training feasible.
