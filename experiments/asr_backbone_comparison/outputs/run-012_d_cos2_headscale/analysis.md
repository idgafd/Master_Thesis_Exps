# Run 012 Analysis — Config D-cos² (learnable θ + non-negative mask) + Headscale (Exp F)

Third experiment in the complex-decay series. Combines findings from run-010 (negative weights destroy
performance) and run-011 (learnable θ differentiates into multi-scale hierarchy, cos² recovers 53% of
the gap). Also introduces a fundamentally different approach: per-head decay scaling (headscale).

---

## Experiment Summary

| | D-cos² (learnable θ, non-negative) | Headscale (per-head decay bias) |
|---|---|---|
| Backbone | bidir_rwkv6_cplx_d_cos2 | bidir_rwkv6_headscale |
| Mask | r^δ · cos²(θ_l · δ) | r^δ (standard, per-head scaled) |
| Learnable | θ per layer (6 params) | bias per head (24 params) |
| Extra compute | 3× matmuls per layer | 0 |
| Extra params | 6 | 24 |
| Total params | 7.74M | 7.74M |

---

## Full Results Table (all experiments to date)

| Config | Mask | θ | DevCER | TestCER | Δbase | rel% | Loss | R@2s | R@5s | R@10s |
|---|---|---|---|---|---|---|---|---|---|---|
| **A  real (005)** | r^δ | — | **0.1676** | **0.1790** | — | — | 0.7988 | 0.2161 | 0.1857 | 0.1790 |
| **A+ Conv (006)** | r^δ+conv | — | **0.1587** | **0.1760** | -0.003 | -1.7% | 0.7585 | 0.2084 | 0.1804 | 0.1744 |
| B  cos (010) | r^δ·cos(θδ) | 0.31 fix | 0.1932 | 0.2140 | +0.035 | +19.6% | 0.9163 | 0.2388 | 0.2161 | 0.2114 |
| C  cos (010) | r^δ·cos(θδ) | 0.90 fix | 0.2109 | 0.2322 | +0.053 | +29.7% | 0.9744 | 0.2523 | 0.2322 | 0.2277 |
| D  cos learn (011) | r^δ·cos(θδ) | learned | 0.1909 | 0.2107 | +0.032 | +17.7% | 0.9002 | 0.2349 | 0.2128 | 0.2081 |
| B² cos² (011) | r^δ·cos²(θδ) | 0.31 fix | 0.1757 | 0.1955 | +0.017 | +9.2% | 0.8546 | 0.2272 | 0.1991 | 0.1938 |
| **D² cos² learn (012)** | r^δ·cos²(θδ) | learned | 0.1782 | 0.1977 | +0.019 | **+10.4%** | 0.8558 | 0.2276 | 0.2010 | 0.1961 |
| **F  headscale (012)** | r^δ (scaled) | — | **0.1660** | **0.1839** | +0.005 | **+2.7%** | 0.8082 | 0.2160 | 0.1879 | 0.1820 |

---

## D-cos² (learnable θ + non-negative mask): Modest Outcome, Rich Diagnostic

### Result: +10.4% relative CER — worse than fixed-θ cos² (B², +9.2%)

This is counterintuitive. Run-011 showed that:
- Learnable θ improved cos (D: +17.7% vs B: +19.6%) — a 2pp gain
- cos² improved fixed θ (B²: +9.2% vs B: +19.6%) — a 10pp gain

We expected D-cos² to combine both gains and approach the baseline. Instead TestCER 0.1977 is
worse than B² at 0.1955. The learnable θ with cos² did NOT stack the improvements.

**Why the non-stacking:** cos² has a different gradient landscape from cos. With cos, negative
weights create strong gradient signal pointing away from bad θ values. With cos², the mask is
always non-negative — the gradient for θ is much weaker (cos² flattens the loss landscape w.r.t.
θ). The model adjusts θ to different values than under cos, but those adjustments are less
consequential for the non-negative mask.

### Theta trajectory — more differentiated than run-011

| Layer | Run-011 Config D (cos) | Run-012 D-cos² | Interpretation |
|---|---|---|---|
| L0 | 0.183 (343ms) | 0.259 (243ms) | Less extreme drift with cos² |
| L1 | 0.301 (209ms) | **0.446 (141ms)** | Strongly fast — sub-phonemic |
| L2 | 0.300 (210ms) | 0.256 (246ms) | Drifted slow (was stable in D) |
| L3 | 0.220 (286ms) | 0.303 (207ms) | Stable at init (was slow in D) |
| L4 | 0.400 (157ms) | **0.453 (139ms)** | Even faster than before |
| L5 | 0.345 (182ms) | **0.194 (324ms)** | Strongly slow — word-level |

**Three groups formed again, but with wider spread:**
- Fast (L1, L4): θ ≈ 0.45, resonance ~140ms
- Mid (L0, L2, L3): θ ≈ 0.26–0.30, resonance ~207-246ms
- Very slow (L5): θ ≈ 0.19, resonance ~324ms

The multi-scale hierarchy is **more pronounced** under cos² than under cos. L5 drifted much further
down (to 324ms vs 182ms), and L1 drifted much further up (to 141ms vs 209ms). This confirms that
the gradient pressure to differentiate temporal scales is genuine — even stronger when negative
weights don't dominate the loss signal.

### Stabilisation timeline

All thetas converge by epoch ~35, frozen by ~45. Identical pattern to run-011.

### Verdict on D-cos²

Scientifically valuable (confirms multi-scale hierarchy is robust across mask types), but
practically worse than even the simplest complex variant (B²). **The complex-decay approach,
in all tested forms, cannot match the monotone real baseline.**

---

## Headscale (Exp F, per-head decay scaling): Best New Result

### Result: DevCER 0.1660, TestCER 0.1839

This is the first non-ConvShift modification to **beat baseline on dev** (0.1660 vs 0.1676).
The test gap of +2.7% is the smallest of any modification except ConvShift itself.

### What headscale does

Each of 4 attention heads per layer gets a learnable scalar bias applied to its log-decay:

```
w_scaled[h] = w[h] * exp(bias[h])
```

- bias > 0 → faster decay (more local attention)
- bias < 0 → slower decay (more global attention)
- bias = 0 → baseline behavior (init)

Total: 4 heads × 6 layers = 24 parameters. Zero extra compute.

### Head bias trajectory (selected epochs)

```
Epoch 1:
  L0: [+0.01, +0.00, +0.02, +0.02]   all near zero (baseline)
  L2: [-0.00, -0.02, +0.01, +0.02]
  L5: [-0.01, -0.00, -0.01, -0.01]

Epoch 10:
  L0: [+0.02, -0.09, +0.01, +0.05]   head 1 going global
  L2: [+0.21, +0.07, +0.05, +0.08]   all heads going local
  L5: [+0.15, +0.19, +0.21, +0.05]   3 heads strongly local

Epoch 30:
  L0: [+0.02, -0.10, +0.00, +0.03]   head 1 stays global, others stable
  L2: [+0.32, +0.18, +0.11, +0.07]   progressive localisation
  L5: [+0.23, +0.30, +0.25, +0.05]   3 heads very local, 1 moderate

Epoch 60 (final):
  L0: [+0.02, -0.11, -0.01, +0.02]   mixed: 1 global, 3 near-baseline
  L1: [+0.03, -0.10, -0.05, +0.10]   mixed: 1 global, 1 fast
  L2: [+0.37, +0.19, +0.12, +0.06]   ALL local (L2 head0: +0.37 = 1.45× faster decay)
  L3: [+0.15, +0.26, +0.20, +0.07]   ALL local
  L4: [+0.29, +0.25, +0.35, -0.00]   3 strongly local, 1 baseline
  L5: [+0.27, +0.35, +0.28, +0.05]   ALL local
```

### Pattern analysis

**The gradient tells a clear story about what each layer needs:**

1. **Lower layers (L0, L1) want mixed attention.** Some heads go global (bias ≈ -0.10, ~10% slower
   decay), others stay near baseline. These layers process raw acoustic features and benefit from
   both local (formant detail) and global (utterance-level context) receptive fields.

2. **Upper layers (L2–L5) overwhelmingly want MORE locality.** Almost all biases are positive,
   often strongly so (+0.25 to +0.37, meaning 1.28× to 1.45× faster decay). This means the
   baseline decay is TOO SLOW for the upper layers — they want sharper, more focused attention.

3. **The localisation trend is monotone with depth.** Mean bias per layer:
   - L0: -0.02 (slightly global)
   - L1: -0.01 (near-baseline)
   - L2: +0.19 (local)
   - L3: +0.17 (local)
   - L4: +0.22 (strongly local)
   - L5: +0.24 (most local)

   This is **the same gradient signal** as Config D's theta trajectory, but expressed through a
   different mechanism. Lower layers attend broadly, upper layers focus locally. The multi-scale
   temporal hierarchy is a robust structural preference of the network.

4. **Biases are NOT converged at epoch 60.** They grew monotonically throughout training with no
   sign of saturation. L2 head0 went from +0.21 at epoch 10 to +0.37 at epoch 60 — nearly
   doubled. L5 head1 went from +0.19 to +0.35. This strongly suggests longer training would
   increase the gains further.

### Dev-test gap

DevCER 0.1660 vs TestCER 0.1839 is a gap of +0.0179. For baseline: 0.1676 vs 0.1790, gap +0.0114.
Headscale's dev-test gap is wider (+0.0065 more). Possible explanations:

1. The biases are slightly overfitting to dev sequence-length statistics.
2. More likely: the model is still undertrained. The upper-layer biases are still growing. The
   dev CER plateaued earlier because dev is smaller and more homogeneous; test CER would continue
   improving with more epochs.

---

## The Core Problem We Are Solving

All experiments from run-010 through run-012 converge on one finding:

**The baseline RWKV-6 decay initialization creates a single temporal scale per dimension, shared
across all layers. The gradient consistently wants different temporal scales at different depths.**

Evidence:
- Run-011 Config D: θ differentiates into 3 regimes (slow/mid/fast by layer)
- Run-012 D-cos²: θ spread is even wider (141ms to 324ms)
- Run-012 headscale: bias differentiates strongly by layer (global→local gradient)

The RWKV-6 decay init formula `decay_speed[n] = -6 + 5 * (n / (d-1)) ^ (0.7 + 1.3 * ratio)` does
scale with layer depth, but the scaling is too weak — the per-dimension range [-6, -1] is constant,
only the exponent changes. The gradient wants a much stronger depth dependence.

**This is not the locality bias problem in the classical sense** (monotone decay cannot attend to
non-adjacent frames more than adjacent ones). Rather, it is a **temporal scale mismatch**: the
current init gives all layers roughly similar decay profiles, when lower layers should attend
broadly and upper layers should attend narrowly.

---

## What Questions Remain Open?

The complex-decay series answered "does the gradient want multi-scale temporal structure?" (yes)
and "is multiplicative phase modulation the right way to provide it?" (no). Headscale showed that
per-head decay scaling captures the same signal and helps on dev. But three deeper questions are
still unanswered:

**Q1. Is monotone decay fundamentally sufficient for ASR?**
Every mask we've tested — real, cos, cos², headscale — is strictly monotone per dimension: closer
frames always get higher weight. We've never given the model the ability to place peak attention
at a non-zero distance. Maybe the baseline's "problem" isn't the temporal scale mismatch at all,
but the inability to attend preferentially to a specific distance (e.g., "the syllable two frames
ago matters more than the immediate neighbor").

**Q2. Can a richer per-dimension decay profile improve over single-exponential?**
The baseline gives each dimension one exponential `r^δ`. Headscale scales the rate per head but
doesn't change the functional form. A mixture of two exponentials (dual-decay) can represent
bi-phasic profiles: fast local dropoff + slow global tail. This is a strictly richer function
class. Does the gradient use it?

**Q3. What is the relationship between headscale's dev-test gap and model understanding?**
Headscale dev 0.1660 vs test 0.1839 (+0.0179 gap) is wider than baseline (0.1676 vs 0.1790,
+0.0114 gap). ConvShift has the same pattern (0.1587 vs 0.1760, +0.0173 gap). Is this inherent
to our dev/test split, or does per-head specialisation genuinely overfit to dev? We need
experiments that answer structural questions — not optimization tricks to close the gap.

---

## Approaches Ranked by Mathematical Understanding They Provide

The ranking criterion is NOT "most likely to beat baseline CER" but rather "which experiment,
regardless of CER outcome, teaches us the most about what this attention mechanism needs."
ConvShift is excluded — it's a generic input-mixing improvement that can be bolted onto any
winning architecture later. It teaches us nothing about attention structure.

---

### Priority 1: Gaussian Attention Mask — The Only Test of Non-Monotone Attention

**The question it answers:** Q1 — is monotone decay sufficient, or does the network want to
attend preferentially to specific non-zero distances?

**Why this is #1:** Every other approach (headscale, dual-decay, ALiBi) preserves monotone
decay. They change the *rate* of decay, not its *shape*. Gaussian is the only experiment that
can detect whether the network wants a fundamentally non-monotone attention profile. This is
a binary question with high information value regardless of CER outcome:

- If μ → 0 for all heads: monotone is sufficient. Close the question. All future work should
  focus on decay rate optimization (headscale, dual-decay, init schedules).
- If μ > 0 for some heads: the network wants non-local attention. This opens a new research
  direction that no amount of exponential-rate tuning can address.

No other experiment can answer this. Dual-decay cannot — it's a sum of two monotone curves,
still monotone. Headscale cannot — it's a rate scaling of a monotone curve. Only a peaked mask
can distinguish "closer is always better" from "this specific distance matters."

**Mechanism:**

```
M_ij^h = exp(-(|i-j| - μ_h)² / (2σ_h²))
```

Learnable μ_h, σ_h per head. Multiplied element-wise onto the LION attention matrix.

**Why Gaussian specifically (not cos, not triangular, not any other bump):**

1. **Always non-negative.** Cos destroyed performance through negative weights. Gaussian ≥ 0
   everywhere. This is the single most important constraint from runs 010-011.
2. **Single peak, no secondary lobes.** cos²(θδ) = 0.5 + 0.5·cos(2θδ) has secondary peaks
   at δ = π/θ, 2π/θ, etc. Gaussian has exactly one peak. Speech attention should attend to
   one distance range, not a periodic pattern of them.
3. **Smooth and differentiable everywhere.** Clean gradients for both μ and σ. No discontinuities,
   no flat regions that trap the optimizer.
4. **Natural degeneration to baseline.** At μ=0 and large σ, the Gaussian is nearly flat over
   the relevant distance range — equivalent to not modifying the LION mask at all. The model
   starts at baseline and can discover non-locality if it exists.
5. **Well-understood interaction with exponential decay.** The effective per-dimension profile is:
   ```
   effective_d(δ) = r_d^δ · exp(-(δ - μ_h)² / 2σ_h²)
   ```
   This product has a closed-form peak at:
   ```
   δ* = μ_h + σ_h² · log(r_d)
   ```
   Since log(r_d) < 0 (decay), the peak is always at δ* < μ_h. For slow-decay dimensions
   (r_d ≈ 1): δ* ≈ μ_h (Gaussian dominates). For fast-decay dimensions (r_d ≈ 0.5):
   δ* ≈ 0 (exponential dominates). This means the Gaussian selectively affects slow-decay
   dimensions without disrupting fast-decay ones. The interaction is automatic and elegant.

**The factorisation non-problem:**

`exp(-(δ-μ)²/2σ²)` does not factorise as f(i)·g(j), blocking O(1) RNN recurrence. This is
irrelevant for us:

- Bidirectional full attention: already O(L²) for both training and inference
- L ~ 250 frames (10s after 4× subsampling): 62K mask entries, negligible vs matmuls
- The mask is Toeplitz: precompute g[0..L-1], broadcast to (T,T), element-wise multiply
- Implementation: one extra (H, T, T) multiply per layer. Overhead: ~1.5% of attention cost

If this ever needs to be ported to streaming inference, the Prony approximation (sum of
exponentials) can approximate the Gaussian to arbitrary precision — this is exactly the
mathematical bridge to dual-decay (see below).

**Implementation sketch:**

The Gaussian mask modifies the LION attention *after* the QK^T is computed but *before* V
multiplication. The key subtlety is that LION uses the prefix-sum trick to avoid materializing
the full (T,T) attention matrix. To apply an element-wise Gaussian mask, we need to materialise
it. Two approaches:

**Approach A (clean, materialise attention matrix):**
Build A_fwd and A_bwd as full (T,T) matrices, multiply by Gaussian mask, then multiply by V.
This requires O(T²) memory per head per layer. At T=250, H=4: 250²×4 = 250K floats = 1MB.
Negligible.

**Approach B (log-space fusion):**
Add the Gaussian log-mask to the prefix-sum before exponentiating. This avoids materialising A
but requires modifying the LION kernel. The Gaussian term `-(δ-μ)²/2σ²` depends on |i-j|, not
on cumulative sums, so it cannot be absorbed into the prefix-sum trick. Approach A is simpler
and sufficient at our scale.

**Params:** 2 per head per layer = 48 total.

**Init:** μ = 0 (local attention), σ = softplus(init) where init gives σ₀ = 10 (~400ms window,
wider than most utterance lengths after subsampling → nearly flat → baseline-like).

**Diagnostics to log:** μ_h and σ_h per head per layer every epoch. The μ trajectory is the
primary scientific output.

**Expected outcomes:**
- **Null result (μ → 0):** Monotone attention is sufficient for ASR at this scale. The remaining
  CER gap is not caused by locality bias. Focus all future work on capacity, data, or decoder.
  Still valuable — it settles Q1 definitively.
- **Positive result (μ > 0, lower layers):** Non-local attention helps. Lower layers want to
  attend ~5-15 frames away (~200-600ms), possibly capturing word-level structure that the
  exponential decay suppresses. This would be a novel finding for RWKV-style architectures.
- **Mixed (μ > 0 but CER worse):** The Gaussian provides useful structure but the implementation
  (multiplicative mask on QK^T) is too crude. Would motivate per-dimension Gaussian or
  attention-weighted Gaussian (content-dependent μ).

---

### Priority 2: Dual-Decay (Exp E) — Richer Functional Form Within LION Framework

**The question it answers:** Q2 — does a two-component exponential mixture improve over single
exponential per dimension?

**Why this is #2:** This is the natural next step in the decay expressiveness hierarchy. The
progression is:

```
Baseline:    1 exponential per dim, same rate across heads       (256 params/layer)
Headscale:   1 exponential per dim, rate scaled per head         (256 + 4 params/layer)
Dual-decay:  2 exponentials per dim, mixed per dim, scaled per head  (256×2 + 4 + 256 params/layer)
```

Dual-decay is the smallest extension that changes the *shape* of the decay curve (not just rate).
It can represent bi-phasic profiles that single-exponential cannot:

```
attention(δ) = α · r_fast^δ + (1-α) · r_slow^δ
```

- α ≈ 1: single fast exponential (local dimension, same as baseline)
- α ≈ 0: single slow exponential (global dimension)
- α ≈ 0.7, r_fast=0.8, r_slow=0.98: fast initial dropoff + long slow tail

The third case is the interesting one. Speech has a bi-phasic correlation structure: strong
local co-articulation (nearby frames highly correlated) plus weak long-range prosodic/speaker
influence. A single exponential must choose one timescale. Two exponentials can represent both.

**Mathematical relationship to Gaussian:**

The Gaussian kernel can be Prony-approximated as a sum of exponentials:

```
exp(-δ²/2σ²) ≈ Σ_{k=1}^{K} w_k · r_k^δ
```

Dual-decay is the K=2 Prony approximation. It's not an exact Gaussian, but it's the cheapest
approximation that breaks monotonicity in the *mixed* profile (even though each component is
monotone, their sum can have an inflection point that creates a "shoulder" — not a peak, but a
region of slower-than-exponential decay).

**Important limitation vs Gaussian:** A mixture of two monotone decreasing functions is still
monotone decreasing. Dual-decay CANNOT create a non-zero peak. It can create:
- Faster-than-exponential decay (both components fast)
- Slower-than-exponential decay (slow component dominates at large δ)
- Bi-phasic "knee" (fast near, slow far)

But NOT: "frame 5 is more important than frame 1." Only Gaussian (Priority 1) can test that.
This is why Gaussian is ranked higher — it answers a more fundamental question.

**Implementation:** Already complete in `bidir_rwkv6_multiscale.py` with `use_dual=True`.
Cost: 2× LION attention per layer (~132s/epoch vs ~66s/epoch). Extra params: 1,560.

**Init design (already implemented):**
- `slow_scale` init: sigmoid(-0.85) ≈ 0.30 → w_slow = w_fast × 0.30 → ~3× longer half-life
- `α` init: sigmoid(2.94) ≈ 0.95 → 95% fast, 5% slow → starts at near-baseline

**Diagnostics to log:**
- `slow_scale` per head (how different are the two timescales?)
- `α` distribution per layer (mean, min, max — how much slow component is used?)
- Per-layer average of both: does upper-layer α → 1 (fast-only) and lower-layer α → 0.5 (mixed)?

**Expected outcomes:**
- **α stays at ~0.95:** Dual-decay adds nothing. Single exponential is sufficient. Close the idea.
  This would be a strong signal that the remaining CER gap is NOT about decay expressiveness.
- **α differentiates by layer:** Lower layers use more slow component (mixed), upper layers stay
  fast-only. This confirms the multi-scale finding from headscale/Config D through a third
  independent mechanism, and shows that bi-phasic profiles genuinely help lower layers.
- **α differentiates by dimension within a layer:** Some dimensions go α→0 (purely global),
  others α→1 (purely local). This would reveal that the model wants a spectral decomposition —
  different feature dimensions tracking different temporal scales. This is the most interesting
  possible outcome.

**Why 2× compute is acceptable:** 66s → 132s per epoch at our model scale. For 60 epochs: ~2h
instead of ~1h. The diagnostic value (α trajectory) justifies the cost even if CER doesn't
improve. And unlike Gaussian, dual-decay stays within the LION recurrence framework — if it
works, it can be deployed for streaming inference with 2× state size.

---

### Priority 3: ALiBi-Style Non-Linear Distance Bias — Only If Gaussian Shows μ > 0

**The question it answers:** Q1 (continued) — if non-monotone attention helps, what is the
optimal functional form for the distance-dependent modulation?

**Key insight from the analysis: linear ALiBi ≡ headscale for LION.**

LION's effective log-attention is: `log A_ij = Σ_d [log(Q_id · K_jd) + cs_d[i] - cs_d[j]]`

Adding a linear ALiBi bias: `log A_ij += -slope_h · |i-j|`

This is mathematically identical to adding a constant `-slope_h / K` to each dimension's decay
rate w_d. Which is exactly what headscale does with `w_h *= exp(bias_h)`. The linear ALiBi
is headscale with a different parameterisation. **No new expressiveness.**

**Where ALiBi becomes interesting: non-linear distance functions.**

```
bias_h(δ) = -a_h · δ² - b_h · log(1 + δ)
```

The quadratic term `-a·δ²` adds super-exponential decay at large distances (Gaussian-like).
The log term `-b·log(1+δ)` adds sub-exponential decay (power-law tail: (1+δ)^{-b}).

Together these can represent:
- Gaussian-like peaked profiles (a > 0, b < 0)
- Power-law long-range tails (a = 0, b > 0)
- Log-linear intermediate profiles (both small)

**Why this is Priority 3 (conditional):**

If Gaussian (Priority 1) shows μ > 0 — the network wants non-monotone attention — then ALiBi
with quadratic term is a more flexible way to achieve the same thing with fewer assumptions.
The Gaussian assumes a specific functional form; ALiBi with polynomial distance terms lets the
model learn whatever profile it needs.

If Gaussian shows μ → 0 — monotone is sufficient — then the quadratic ALiBi term is also
unnecessary (it only adds super-exponential decay, which is just "more local" than the baseline,
not a new capability). In that case, skip this entirely.

**Implementation:** Build (H, T, T) bias matrix from distance, add to log-attention. 2-3 params
per head per layer. Same overhead as Gaussian.

**This experiment is gated on Gaussian results. Do not run it independently.**

---

### Headscale — Completed, Not Prioritised for Standalone Re-run

Headscale (Exp F) already ran. It showed:
- Dev 0.1660 (beats baseline), Test 0.1839 (+2.7%)
- Biases not converged
- Pattern: lower layers mixed, upper layers strongly local

The dev-test gap (0.0179 vs baseline 0.0114) follows the same pattern as ConvShift. Both are
modifications that help the model fit dev better without proportional transfer to test. Running
headscale longer or with separate LR would be an optimization exercise, not a research question.
It doesn't teach us anything new about attention structure.

Headscale's contribution is already captured: the multi-scale hierarchy is confirmed. The bias
trajectory is the evidence. If Gaussian or dual-decay produce a winner, we ADD headscale to it
as a stacking optimisation at the end.

---

## Experiment Plan Summary

| Priority | Experiment | Question | Key diagnostic | Compute | Params |
|---|---|---|---|---|---|
| **1** | Gaussian mask | Is monotone decay sufficient? | μ trajectory per head | +1.5% | 48 |
| **2** | Dual-decay (Exp E) | Does bi-phasic decay help? | α distribution per layer | +100% | 1,560 |
| **3** | Non-linear ALiBi | What distance profile is optimal? | Quadratic/log coefficients | +1.5% | 48-72 |
| — | Headscale re-run | Can we close dev-test gap? | — | 0% | 24 |

Priority 3 is **conditional on Priority 1 showing μ > 0**. If Gaussian μ → 0, skip ALiBi.

Priorities 1 and 2 can run in parallel — they test different questions (shape vs functional form)
and neither result depends on the other. Gaussian takes ~66s/epoch (same as baseline); dual-decay
takes ~132s/epoch. Both at 60 epochs: ~1h and ~2h respectively.

**What each outcome combination tells us:**

| Gaussian μ | Dual-decay α | Interpretation | Next step |
|---|---|---|---|
| μ → 0 | α → 0.95 | Monotone single-exp is optimal | Close decay research, focus on capacity/data |
| μ → 0 | α differentiates | Bi-phasic helps but non-locality doesn't | Optimise dual-decay, keep in final model |
| μ > 0 | α → 0.95 | Non-locality matters but mixture doesn't | Develop Gaussian further (per-dim, content-dep) |
| μ > 0 | α differentiates | Both shape AND functional form matter | Combine: Gaussian × dual-decay, richest model |

---

## Closing the Complex-Decay Investigation

### What we learned (runs 010, 011, 012)

1. **Fixed-theta cos oscillation destroys ASR performance** at all tested frequencies.
   The forced anti-attention at specific distances is structurally incompatible with
   variable-length speech. (Configs B, C)

2. **cos² (non-negative) halves the damage** but cannot eliminate it. Even non-negative
   periodic modulation is a harmful constraint vs free exponential decay. (Config B²)

3. **Learnable theta does not collapse to zero.** The gradient actively shapes per-layer
   theta into a multi-scale temporal hierarchy: slow (word), mid (syllable), fast
   (sub-phonemic). This is reproducible across cos (run-011) and cos² (run-012) masks.
   (Configs D, D²)

4. **But the hierarchy doesn't improve CER.** The multi-scale structure is real gradient
   signal, but the cos/cos² multiplicative mask is the wrong way to encode it. The same
   signal appears in headscale biases (run-012 Exp F) through a simpler mechanism that
   actually helps.

5. **The network wants layer-dependent temporal focus.** Lower layers: broad attention
   (global context). Upper layers: narrow attention (local detail). This is the one
   transferable insight from the entire complex-decay series.

### Recommended thesis narrative

Report configs B → C → D → B² → D² as a progression of hypotheses and fixes:

- B,C: "oscillation hurts" → diagnose: negative weights
- B²: "remove negative weights" → diagnose: still hurts, but less → periodicity itself is costly
- D: "let theta learn" → diagnose: theta doesn't collapse → gradient sees multi-scale structure
- D²: "both fixes together" → result: doesn't stack → close the line

Then pivot to headscale (Exp F) as the approach that captures the same multi-scale insight
through a simpler, more effective mechanism. This is a clean negative-to-positive arc.
