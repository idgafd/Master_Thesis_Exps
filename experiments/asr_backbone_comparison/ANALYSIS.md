# Experiment Analysis — Ideas, Problems, Results, Conclusions

---

## Dataset & Setup Constraints

**Dataset:** Common Voice Ukrainian 24.0 — 35h scripted speech, T≈250 frames after 4× subsampling (10s utterance → 250 frames).

**What this means for architecture choices:**

- **Sequences are short.** T=250 makes the full LION attention matrix 250×250 = 62.5K entries per head. O(T²) is not a bottleneck. Sparse attention, linear-attention approximations, and chunked streaming are all motivated by *long sequences* — not relevant here.
- **Data is limited (35h).** Complex architectures with many parameters tend to overfit. The consistent dev→test gap in this project (+0.01 to +0.017 CER across all models) is primarily a data-size effect, not architecture overfitting. More data would widen the margin between modifications that work on dev vs test.
- **Scripted Ukrainian speech** has relatively consistent prosody and vocabulary. Broad attention strategies work well because speakers don't produce long-distance dependency surprises.
- **CER ~0.17–0.21 is the current operating range.** At this performance level, architectural improvements tend to yield 1–5% relative gains. Anything above 5% is large for this setup.

---

## The Central Problem

All runs after 005 are attacking the same core question: **can we give the LION (bidir RWKV-6) attention mechanism a richer temporal structure without hurting stability?**

LION's decay is a monotone exponential per dimension: $A_{ij} \propto r_d^{|i-j|}$, where $r_d \in (0,1)$ is initialised once and partially learned. Every layer gets roughly the same decay profile. The gradient consistently signals that this is suboptimal:

- **Lower layers want broader attention** — they integrate raw acoustic features and benefit from long-range context.
- **Upper layers want sharper, more local attention** — they process higher-level patterns and benefit from focusing on nearby frames.

This layer-depth bias was independently confirmed by three different mechanisms (runs 011, 012, 013) and is the main conceptual finding of the series.

---

## Runs 003–004 — Mamba Baseline

**Goal:** Establish whether Mamba (SSM with selective state) is competitive with LION.

**Setup:** Mamba 7.70M, 6 layers. Ablation on WSD scheduler decay timing (epoch 12 vs epoch 46). Scheduler ablation motivated by observed dev-loss spike in an earlier default run.

**Results:** CER 0.2098 / 0.2125. WSD timing had negligible effect.

**Conclusion:** Mamba underperforms LION by ~17%. The selective gating mechanism does not compensate for the absence of full bidirectional context. **Drop Mamba as primary architecture.**

**Carry-state note:** Mamba carry-state gives +4.7% CER improvement at 2s chunks (reset 0.2593 → carry 0.2121). This is the only architecture where carry-state genuinely works.

---

## Run 005 — LION Baseline

**Goal:** Compare bidirectional RWKV-6 (LION parallel form) against bidir linear attention.

**Setup:** Both trained from scratch with cosine+warmup, identical hyperparameters.

**Results:**
- `bidir_linear_attention` (LION-D ELU+1 feature map): CER 0.2044
- `bidir_rwkv6` (LION parallel): CER 0.1790

**Conclusion:** LION parallel form with learned decay is significantly better than fixed feature-map linear attention. The learned exponential decay kernels carry information that ELU+1 discards. **LION is the base architecture for all subsequent runs.**

---

## Run 006 — ConvShift + Gate Ablation

**Goal:** Test AudioRWKV-style token mixing (1-D depthwise ConvShift) and an Xres gate on top of LION.

**ConvShift:** Replaces token-shift (fixed `[0.5, 0, 0.5]` mix) with a learned 1-D depthwise convolution. Adds ~7K parameters. Improves token mixing before QKV projection.

**Formulation:** $x'_t = \text{depthconv}(x_{t-1:t+1})$ replacing the fixed $x'_t = 0.5 x_{t-1} + 0.5 x_t$.

**Results:**
- NoGate (ConvShift only): CER **0.1760** — best result in entire project
- With gate (Xres-conditioned): CER 0.1813 — gate hurts

**Conclusion:** ConvShift is a cheap, reliable improvement (+1.7% relative CER). The gate adds parameters without benefit — the model doesn't need residual gating on top of LION's existing gating mechanism. **ConvShift is a confirmed addition; gate is dropped.**

---

## Run 007 — BiWKV6 (Serial Bidirectional / Streaming)

**Goal:** Test whether a streaming-capable bidirectional architecture (serial forward+backward passes) can match parallel LION.

**Setup:** Serial fusion architecture: forward RWKV-6 pass, then backward pass, concatenated output. Designed to be deployable for streaming with carry-state.

**Results:** CER 0.2201–0.2211. Carry-state: **CER degraded to 0.71** (worse than reset at 0.22).

**Why carry-state failed:** The serial bidir architecture was evaluated with uninitialised carry state — the backward pass's initial hidden state during inference is not the same as during training. At 6 layers this blows up. The carry-state evaluation is technically broken for BiWKV6 at this depth.

**Conclusion:** Serial bidir is 4% worse than parallel LION and carry-state inference is unusable without explicit carry-state training. **BiWKV6 is not a current priority. Revisit only if streaming deployment is a hard requirement.**

---

## Runs 008–009 — Depth Scaling

**Goal:** Can more capacity (12 layers, 100 epochs) close the gap?

**Results:**
- LION 12L, 100ep: CER 0.1816 — worse than 6L despite 1.75× more params and longer training
- BiWKV6 6L, 100ep: CER 0.1894 — better than 6L/60ep (0.2201), still far behind LION

**Conclusion:** Depth alone does not help LION at this data scale. The model saturates at 6 layers for 35h Ukrainian data. BiWKV6 benefits more from longer training but remains inferior. **Do not pursue deeper LION. More data or better decay structure is the path forward.**

---

## Runs 010–013 — Decay Mechanism Exploration (Connected Series)

These four runs form a single investigation. The shared question: **can we improve LION's temporal attention structure beyond monotone exponential decay?**

### The Hypothesis

Replace the real-valued monotone decay $r^{|i-j|}$ with a modulated form that allows the model to learn richer temporal relationships.

### What Was Tested

**Run 010 — Complex decay poles (fixed θ):**

$A_{ij} \propto r^{|i-j|} \cdot \cos(\theta |i-j|)$

with θ=0.31 (~syllable scale) and θ=0.90 (~phoneme scale).

CER: 0.2140 (θ=0.31), 0.2322 (θ=0.90). Both much worse than baseline.

**Why it failed:** Cosine produces negative attention weights at half-period distances. Negative attention means the model actively suppresses information from specific distances. This is structurally incompatible with ASR — there are no distances from which information should always be suppressed.

**Run 011 — Learnable θ + cos² (non-negative):**

$A_{ij} \propto r^{|i-j|} \cdot \cos^2(\theta_l |i-j|)$

(a) Learnable θ per layer (cplx_d): CER 0.2107 — still worse.
(b) Fixed θ=0.31 with cos² instead of cos: CER 0.1955 — much better but still below baseline.

**Key finding:** Learnable θ spontaneously differentiated into a multi-scale hierarchy:
- Layer 0: θ≈0.18 (slow, ~350ms)
- Layers 2,3: θ≈0.30 (~210ms)
- Layers 4,5: θ≈0.40 (fast, ~157ms)

The gradient actively prefers different temporal scales at different depths. But the cos/cos² multiplicative form is the wrong vehicle.

**Run 012 — D-cos² (learnable θ + cos²) + Headscale:**

D-cos²: CER 0.1977 — worse than fixed-θ cos² (0.1955). The learnable θ with cos² doesn't stack with the non-negative improvement.

Headscale: Per-head learnable scalar bias on log-decay. $w_h' = w_h \cdot e^{\beta_h}$. 24 parameters. CER **0.1839**. Best dev CER in the entire project (0.1660).

**Headscale bias trajectory:**
- L0–L1: biases near zero, one head slightly global (β≈−0.10)
- L2–L5: all heads positive (β=+0.15 to +0.37), i.e. **faster decay = more local**
- Trend is monotone with depth: mean β per layer goes −0.02 → −0.01 → +0.19 → +0.17 → +0.22 → +0.24

This independently confirms the multi-scale temporal hierarchy found in Config D.

**Run 013 — Gaussian mask + Dual-decay:**

Gaussian: $A_{ij} \propto r^{|i-j|} \cdot \exp\left(-\frac{(|i-j| - \mu_h)^2}{2\sigma_h^2}\right)$. CER 0.1861. σ stayed large (~600ms flat window) — all heads preferred broad attention over peaked attention. **μ→0 for all heads.**

Dual-decay: $A_{ij} \propto \alpha r_{\text{fast}}^{|i-j|} + (1-\alpha) r_{\text{slow}}^{|i-j|}$. CER 0.1875. α stayed near 0.95 (mostly fast) — no significant use of the slow component.

### Cross-Run Generalisation (010–013)

Three independent mechanisms all converge on the same finding:

> **The LION attention mechanism wants lower layers to attend broadly and upper layers to attend locally. This is a robust structural preference, not an artefact of any single parameterisation.**

Evidence:
1. Config D theta trajectory (run 011): slow θ in early layers, fast θ in upper layers
2. Headscale bias trajectory (run 012): negative/zero bias in L0–L1, positive and growing bias in L2–L5
3. Dual-decay α (run 013): α didn't differentiate by layer — but σ of Gaussian stayed large, meaning no peaked attention was found useful

The layer-depth bias is confirmed. The modulation forms tested (cos, cos², Gaussian peak, dual-decay) all failed to provide net benefit despite capturing real gradient signal.

**Why they all fail:** The modulation adds structured constraints on top of a decay mechanism that already works well. The cost of the constraint (limited expressiveness, harder optimisation landscape) outweighs the benefit of the added structure. The simplest useful form is headscale — scale the rate per head, no structural constraint.

### What to Drop from This Direction

- **Cosine decay (runs 010, 011):** Negative weights are always harmful. Drop entirely.
- **Cos² decay (runs 011, 012):** Non-negative but still periodic — unnecessary structure. Drop.
- **D-cos² combined (run 012):** Learnable θ doesn't add to cos². Drop.
- **Dual-decay (run 013):** α never differentiates; adds 2× compute for marginal gain. Only viable if you have architectural reasons to want bi-phasic decay (e.g. streaming, longer sequences). Not now.
- **Gaussian peak (run 013):** μ→0 across all heads — the network doesn't want peaked attention at non-zero distances. The "non-monotone attention" hypothesis is **not supported** for this dataset and sequence length.

### What to Keep / Resolve

- **Headscale:** Dev CER 0.1660, test CER 0.1839. Biases not converged at 60 epochs. Try 100 epochs. Low cost to stack with ConvShift. **Carry forward.**
- **Multi-scale depth hypothesis:** Confirmed by three mechanisms. Could be explicitly encoded via layer-dependent decay init (steeper decay for upper layers) rather than learned. Worth testing as a zero-parameter architectural prior.

---

## Runs 014–015 — LayerConv + Temperature (Standard Regularisation)

**Goal:** Encode the multi-scale depth hypothesis directly into two orthogonal mechanisms: (a) layer-dependent convolution kernel sizes, (b) per-head attention temperature scaling.

### Run 014 — Layer-Dependent ConvShift

**Idea:** Replace the uniform kernel=3 ConvShift (run 006) with layer-dependent kernels: lower layers get kernel=7 (broad mixing), upper layers get kernel=3 (local mixing).

**Results:** Dev CER 0.1574, Test CER 0.1768, Test WER 0.6548.

### Run 015 — Attention Temperature

**Idea:** Per-head learnable τ initialised as linear ramp L0:1.0 → L5:2.0. Higher τ sharpens attention in upper layers.

**Results:** Dev CER 0.1606, Test CER 0.1792, Test WER 0.6681.

Learned τ at epoch 60: L0:[1.03,1.01,1.17,1.28] → L5:[1.84,1.81,1.90,1.77]. The model preserved the initialised hierarchy — confirming upper layers genuinely prefer sharper attention.

### Analysis: Dev-Test Gap Problem

Both models achieved excellent dev CER (0.1574, 0.1606) but test CER barely improved over baselines:

| Model | Dev CER | Test CER | Dev-Test Gap |
|-------|---------|----------|-------------|
| bidir_rwkv6 (005) | 0.1912 | 0.1790 | −0.012 (test better) |
| bidir_rwkv6_temperature (015) | 0.1606 | 0.1792 | +0.019 |
| bidir_rwkv6_conv_nogate (006) | 0.1587 | 0.1760 | +0.017 |
| bidir_rwkv6_layerconv (014) | 0.1574 | 0.1768 | +0.019 |

The baseline has test *better* than dev (healthy generalisation). All enhanced models show dev 0.017–0.019 better than test — they overfit the dev distribution. The large dev improvements (~0.03 CER) are ~80% overfitting.

However, the *ideas* are sound: τ learned the expected depth-varying hierarchy, and layerconv matched the best ConvShift result. The problem is training regularisation, not architecture.

---

## Runs 016–017 — Strong Regularisation Experiment (Failed)

**Goal:** Test whether heavier regularisation closes the dev-test gap and lets temperature/layerconv generalise their dev gains to test.

**Changes from standard training:**
- Dropout: 0.15 → 0.25
- SpecAugment freq_mask_param: 15 → 27
- SpecAugment time_mask_param: 35 → 70
- num_time_masks: 2 → 5
- Epochs: 60 → 80, patience: 15 → 20

### Results

| Run | Backbone | Dev CER | Test CER | Test WER | Train loss @80 |
|-----|----------|---------|----------|----------|---------------|
| 016 | `bidir_rwkv6` | 0.2546 | 0.2779 | 0.8733 | 1.73 |
| 016 | `bidir_rwkv6_temperature` | 0.2649 | 0.2874 | 0.8924 | 1.74 |
| 017 | `bidir_rwkv6_conv_nogate` | 0.2977 | 0.3189 | 0.9132 | 1.83 |
| 017 | `bidir_rwkv6_layerconv` | 0.2978 | 0.3205 | 0.9020 | 1.83 |

**Complete failure.** All models severely underfitted. CER roughly doubled compared to standard regularisation.

### Why It Failed

1. **Train loss at epoch 80 is 1.7–1.8** vs ~0.52 in original runs. The models could not learn the training data. The heavy SpecAugment (time_mask_param=70 with 5 masks) was destroying too much signal — for utterances averaging 4–5 seconds, five 70-frame masks can blank out most of the input.
2. **Dropout 0.25 on a 256-dim model with only 7.7M params** is too aggressive. The model is already parameter-efficient; it needs to use its capacity, not have it randomly zeroed.
3. **Dev-test gap is now small** (0.2546 → 0.2779 = 0.023) — the regularisation succeeded in *preventing overfitting*, but at the cost of preventing *learning*.
4. **Conv models (run-017) suffered more** than non-conv (run-016). The ConvShift's small kernels are especially sensitive to SpecAugment blanking — the conv sees masked regions as real patterns and learns to suppress signal.

### Conclusion

The regularisation settings from the literature (SpecAugment LD policy: freq=27, time=100, 2+10 masks) assume models with 10–100× more parameters and 10–100× more data. Our 7.7M model on 35h data cannot tolerate this level of augmentation. The original settings (dropout=0.15, freq=15, time=35, 2+2 masks) were already near-optimal for this scale.

**The dev-test gap is a data-size problem, not a regularisation problem.** With 35h of training data, models with added expressiveness (temperature, layerconv) learn dev-specific patterns that don't transfer. More aggressive regularisation doesn't fix this — it just prevents learning entirely.

---

## Run 018 — RWKV-7 Init Fixes

**Goal:** Diagnose why stock RWKV-7 converges poorly in the baseline run. The hypothesis: the triton backend's decay initialisation maps to a much narrower range than RWKV-6, effectively giving the model 1-token memory.

**Three fixes tested:**
1. **Decay init (w0 += 2.0):** Shifts the sigmoid-mapped decay from [0.0009, 0.111] to [0.007, 0.378], matching RWKV-6's effective range.
2. **v_first disable (v0 = −5.0):** Sets the value residual gate to sigmoid(−5) ≈ 0.007, effectively disabling v_first which is designed for large-scale LLMs.
3. **k_a disable (k_a = 0.0):** Removes initial key halving (stock init makes k *= 0.5 via k_a), restoring full key magnitude.

**Setup:** WSD scheduler (decay at epoch 48), 60 epochs, d=256, 6 layers. Two variants: fix_decay (fix 1 only) vs fix_all (fixes 1+2+3).

### Results

| Variant | Dev CER | Test CER | Test WER | Train loss @60 |
|---------|---------|----------|----------|---------------|
| `rwkv7_fix_decay` | 0.3570 | 0.3776 | 0.9734 | 1.34 |
| `rwkv7_fix_all` | 0.2388 | 0.2602 | 0.8597 | 0.81 |

**Decay-only fix is insufficient.** CER 0.3776 — the model still barely converges. Train loss plateaus at 1.34 (vs 0.81 for fix_all), indicating v_first and k_a cause optimisation problems at this scale.

**All-fixes variant recovers partial convergence.** CER 0.2602 — comparable to Mamba (0.2098) but still well below LION (0.1790). The remaining gap likely reflects RWKV-7's unidirectional nature: without bidirectional context, it hits the same ceiling as Mamba.

**Carry-state works correctly.** Unlike BiWKV6 (carry CER >0.7, broken), RWKV-7 fix_all shows carry-state improvement at 2s chunks: reset 0.2944 → carry 0.2629 (Δ=+0.032). This is better than Mamba's carry benefit (+0.047 at 2s) in absolute terms but similar in relative terms. The recurrent state initialisation is healthy.

### Conclusion

RWKV-7's poor convergence was caused by three interacting init problems, not just decay. Fixing all three brings it to Mamba-level performance with functional carry-state. However, it remains ~45% worse than LION (0.2602 vs 0.1790) due to the fundamental unidirectional limitation. **RWKV-7 is not competitive with LION for offline ASR but could be relevant for streaming if bidirectional processing is unavailable.**

---

## Run 019 — Plain RWKV-6 / RWKV-7 at 100 Epochs (Cosine, Default SpecAugment)

**Goal:** Re-check the plain unidirectional RWKV encoders under the default ASR
recipe, but with longer training. The important constraints were: no bidir
RWKV-6 variants, default SpecAugment retained, cosine scheduler, 100 epochs.

### Results

| Backbone | Best Ep | Best Dev CER | Test CER | Test WER |
|----------|---------|--------------|----------|----------|
| `rwkv6` | 87/100 | 0.2163 | 0.2371 | 0.8017 |
| `rwkv7` | 83/100 | 0.3822 | 0.4000 | 0.9808 |

Carry-state delta (reset − carry):

| Backbone | Δ@2s | Δ@5s | Δ@10s |
|----------|------|------|-------|
| `rwkv6` | +0.0149 | -0.0320 | -0.0416 |
| `rwkv7` | -0.0270 | -0.0395 | -0.0411 |

### RWKV-6 Interpretation

The important RWKV-6 result is **not** the absolute CER value by itself, but
the mismatch between dev and test movement.

Inside the 100-epoch run:

- dev CER at epoch 60: **0.2286**
- best dev CER at epoch 87: **0.2163**
- final test CER: **0.2371**

Relative to the earlier plain-RWKV6 result you already had (~**0.2355** test
CER at 60 epochs), the 100-epoch rerun lands in essentially the same place on
test. So extra optimization improves the in-distribution proxy (dev) but does
not improve real held-out test performance.

This strongly suggests that plain RWKV-6 is hitting a **generalisation ceiling**
on this 35h setup, not a simple optimisation ceiling. “Data ceiling” is the
right first explanation, though the more precise statement is:

> The current plain-RWKV6 recipe can still fit train/dev better with more
> epochs, but that extra fit does not transfer to the test split.

That can be caused by both limited data and dev-test mismatch. The practical
conclusion is the same: **more epochs alone are not the lever that changes the
test ceiling for plain RWKV-6.**

### RWKV-7 Interpretation

Stock RWKV-7 remains poor even with 100 epochs:

- `rwkv7` 100ep cosine: **0.4000** test CER
- `rwkv7_fix_decay` 60ep WSD (run 018): **0.3776**
- `rwkv7_fix_all` 60ep WSD (run 018): **0.2602**

Longer training does **not** substitute for the init fixes. The stock RWKV-7
problem is not “needs more epochs”; it is “starts in a bad operating regime.”
Its carry-state also degrades chunked CER at every tested window, which is the
opposite of the healthy behaviour seen in `rwkv7_fix_all`.

### Short Answer — Why RWKV-7 Stock Mechanisms Degrade at This Scale

The stock RWKV-7 defaults encode assumptions from larger autoregressive models,
and those assumptions are harmful in this 6-layer, 256-dim, 35-hour ASR regime:

1. `v_first` is too dominant and suppresses hierarchical feature formation.
2. `k_a` weakens the already-small attention signal at init.
3. Stock decay init is too narrow, so the model starts with too little usable memory.
4. Even after repairs, RWKV-7 stays unidirectional, so it still cannot close the offline-ASR gap to stronger RWKV-6 variants.

The “fixes” are therefore not generic small-scale boosters; they are **RWKV-7-specific repairs**.
Without them, RWKV-7 degrades. With them, it becomes usable, but still not a
best-model candidate for this dataset.

### Best Model Choice For Now

The selection rule is now clearer:

1. **Best offline model overall:** keep `bidir_rwkv6_conv_nogate` (run 006, CER **0.1760**).
2. **Best plain recurrent encoder:** use `rwkv6`, not `rwkv7`.
3. **If RWKV-7 is revisited:** start from `rwkv7_fix_all`, never from stock `rwkv7`.

---

## Ideas Evaluated but Not Applicable Now

### Option C — Top-k Sparse Attention

**Idea:** Explicitly enforce sparsity: keep only the top-K attention weights per query, zero the rest.

$\alpha_{ij} = \frac{A_{ij}}{\sum_{k \in \text{top-}K(i)} A_{ik}} \cdot \mathbb{1}[j \in \text{top-}K(i)]$

**Why it won't work here:**
1. T=250 frames. The LION matrix is 62.5K entries per head — already tiny. Sparsity has no efficiency motivation.
2. Run 013 shows σ stays at ~600ms (nearly flat Gaussian). The network wants *broad* attention, not sparse. Top-k would forcibly restrict this.
3. Non-differentiability requires straight-through estimators or relaxed sorting — significant engineering complexity.
4. Run 013 Gaussian result (μ→0) directly contradicts the sparsity motivation: if the network wanted to focus on specific distances, we'd see it. It doesn't.

**When it becomes relevant:** T > 2000 (long-form audio), where O(T²) LION is a bottleneck. Not this project.

### Option D — Exponential Feature Maps (FAVOR+/Performer)

**Idea:** Replace the SiLU-based feature map φ in linear attention with Random Fourier Features that better approximate exp(q·k):

$\phi(x) = \frac{e^{|x|^2/2}}{\sqrt{m}} \left[\exp(\omega_1^\top x), \ldots, \exp(\omega_m^\top x)\right], \quad \omega_i \sim \mathcal{N}(0,I)$

Approximation error: $O(1/\sqrt{m})$. At m=256, error ≈ 6%.

**Why it's misaligned:**
1. Our best model is `bidir_rwkv6` — which computes the **exact** T×T attention matrix, not a feature-map approximation. There is no approximation error to fix.
2. `bidir_linear_attention` uses ELU+1 features and scored CER 0.2044 (run-005). FAVOR+ would improve this baseline only, still well below RWKV-6 (0.179).
3. At T=250, exact O(T²) runs in ~65s/epoch. No speed motivation for O(T·m) approximation.

**If ever relevant:** Scale to T > 2000 where exact LION becomes slow. Apply to `bidir_linear_attention`, compare ELU+1 vs FAVOR+. Don't apply to `bidir_rwkv6`.

---

## Summary: What to Do, What to Drop

### Keep and extend
| Item | Why | Action |
|------|-----|--------|
| LION + ConvShift (nogate) | Best absolute CER (0.1760), confirmed useful | Baseline for future runs |
| Headscale | Cheapest useful head-level modification (24 params, dev 0.1660) | Add to ConvShift, try 100 epochs |
| Multi-scale depth prior | Confirmed by 5 mechanisms (runs 011, 012, 013, 014, 015) | Already encoded in layerconv + temperature |
| Plain RWKV-6 | Best simple recurrent encoder; 100ep confirms the test ceiling but keeps it ahead of RWKV-7 | Keep only if unidirectional / carry-state encoder is required |

### Drop
| Item | Why |
|------|-----|
| Mamba | 17% worse than LION, no carry-state benefit at inference |
| Stock RWKV-7 | 100 epochs still gives 0.4000 CER; longer training does not replace the init fixes |
| BiWKV6 (unless streaming needed) | 4% worse than LION, carry-state broken without special training |
| Complex decay (cos, cos², D-cos²) | All worse; structural cost exceeds benefit |
| Dual-decay | α never differentiates; 2× compute for marginal gain (0.1875 vs 0.1790) |
| Gaussian peak modulation | μ→0: non-monotone attention is not needed at T=250 |
| Top-k sparse attention | T too short; contradicts empirical broad-attention finding |
| FAVOR+ feature maps | Wrong model target; no speed motivation at T=250 |
| Aggressive regularisation (runs 016–017) | dropout=0.25 + heavy SpecAugment causes severe underfitting at 7.7M/35h scale |

### Explore
| Item | Why | Condition |
|------|-----|-----------|
| Stacking ConvShift + headscale | Both confirmed independently; combined effect untested | Unconditional |
| Longer training for headscale | Biases not converged at 60ep | Unconditional |
| Moderate regularisation tuning | Runs 016–017 went too far; try dropout=0.20, time_mask=50, 3 masks as middle ground | If dev-test gap remains a concern |
| Stacking temperature + ConvShift | Temperature learned correct hierarchy; combine with conv for two orthogonal mechanisms | Unconditional |
| Increased data / Ukrainian corpora | Dev-test gap is fundamentally data-limited; more data would unlock further gains | Long-term |
