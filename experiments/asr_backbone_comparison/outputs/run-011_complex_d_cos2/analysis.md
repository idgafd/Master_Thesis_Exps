# Run 011 Analysis — Config D (learnable theta) + Config B-cos² (non-negative mask)

Companion to run-010 analysis. This run addresses two hypotheses from the run-010 post-mortem:
1. Does removing negative attention weights (cos²) recover the accuracy lost in Config B?
2. Does the gradient push theta toward zero when it is learnable (Config D)?

---

## Full Results Table (all complex-decay experiments)

| Config | Mask | theta | DevCER | TestCER | Δbase | rel% | Loss | R@2s | R@5s | R@10s |
|---|---|---|---|---|---|---|---|---|---|---|
| **A  real (005)** | r^δ | — | **0.1676** | **0.1790** | — | — | 0.7988 | 0.2161 | 0.1857 | 0.1790 |
| **A+ Conv (006)** | r^δ | — | **0.1587** | **0.1760** | -0.003 | -1.7% | 0.7585 | 0.2084 | 0.1804 | 0.1744 |
| B  cos (010) | r^δ·cos(θδ) | 0.31 fixed | 0.1932 | 0.2140 | +0.035 | +19.6% | 0.9163 | 0.2388 | 0.2161 | 0.2114 |
| C  cos (010) | r^δ·cos(θδ) | 0.90 fixed | 0.2109 | 0.2322 | +0.053 | +29.7% | 0.9744 | 0.2523 | 0.2322 | 0.2277 |
| **D  cos learned (011)** | r^δ·cos(θδ) | learned | 0.1909 | 0.2107 | +0.032 | +17.7% | 0.9002 | 0.2349 | 0.2128 | 0.2081 |
| **B² cos² (011)** | r^δ·cos²(θδ) | 0.31 fixed | **0.1757** | **0.1955** | +0.017 | +9.2% | 0.8546 | 0.2272 | 0.1991 | 0.1938 |

δ = |i-j| (frame distance after 4× subsampling, i.e. 40ms per frame).

---

## Config D: Learned Theta — Key Finding

### Theta did NOT collapse to zero

This is the single most important result. The initial hypothesis was that if the CTC gradient
sees no benefit from oscillation, theta would drift to zero. **It didn't.** Instead, theta
differentiated into three distinct groups:

| Layer | Init | Final | Resonance | Δ from init | Behaviour |
|---|---|---|---|---|---|
| L0 | 0.310 | **0.183** | 343ms | -0.127 | drifted DOWN strongly |
| L1 | 0.310 | **0.301** | 209ms | -0.009 | near-stable |
| L2 | 0.310 | **0.300** | 210ms | -0.010 | near-stable |
| L3 | 0.310 | **0.220** | 286ms | -0.090 | drifted DOWN |
| L4 | 0.310 | **0.400** | 157ms | +0.090 | drifted UP |
| L5 | 0.310 | **0.345** | 182ms | +0.035 | slight UP |

**Three regimes emerged:**
- **Slow oscillation** (L0, L3): θ→0.18–0.22, resonance 280–340ms — approaching word-level
- **Mid oscillation** (L1, L2): θ≈0.30, resonance ~210ms — stable at syllable-level
- **Fast oscillation** (L4, L5): θ→0.35–0.40, resonance 157–182ms — sub-syllabic

The gradient had 6 parameters with complete freedom to push them all to zero. It chose not
to. This means **the complex structure is not noise — the gradient actively shapes it into
a multi-scale temporal hierarchy.**

### But CER is still worse than baseline

Despite the structured theta landscape, Config D gives TestCER 0.2107 vs baseline 0.1790
(+17.7%). The learned thetas reduce the gap from Config B's +19.6% to +17.7% — a marginal
improvement. The oscillation structure is meaningful to the gradient but the cos-mask's
negative weights destroy more information than the multi-scale resonance provides.

### Theta stabilisation timeline

| Epoch | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| 1 | 0.310 | 0.309 | 0.307 | 0.307 | 0.308 | 0.307 |
| 10 | 0.263 | 0.306 | 0.303 | 0.261 | 0.379 | 0.349 |
| 20 | 0.218 | 0.302 | 0.299 | 0.234 | 0.395 | 0.346 |
| 30 | 0.195 | 0.301 | 0.301 | 0.224 | 0.399 | 0.344 |
| 40 | 0.187 | 0.301 | 0.300 | 0.221 | 0.400 | 0.345 |
| 50 | 0.184 | 0.301 | 0.300 | 0.220 | 0.399 | 0.345 |
| 60 | 0.183 | 0.301 | 0.300 | 0.220 | 0.400 | 0.345 |

All thetas stabilise by epoch ~35 and are essentially frozen by epoch 40. The early motion
(epochs 1–15) is the most informative: L0 and L3 drop rapidly, L4 rises rapidly, L1/L2
barely move. This suggests the initial value 0.31 happens to be near-optimal for L1/L2
but wrong for boundary layers.

---

## Config B-cos²: Non-negative mask — partial recovery

### cos² halves the degradation

| Comparison | TestCER | Δ vs baseline |
|---|---|---|
| B (cos, θ=0.31) | 0.2140 | +0.035 (+19.6%) |
| B² (cos², θ=0.31) | 0.1955 | +0.017 (+9.2%) |
| Baseline A | 0.1790 | — |

cos² eliminates negative attention weights. This recovered **53% of the gap** between
Config B and the baseline (from +3.5pp to +1.7pp). The remaining +1.7pp is the pure cost
of forced periodic attention — even non-negative oscillation is a structural constraint
that the model cannot fully compensate for.

### Dev CER is remarkably close to baseline

B² dev CER: 0.1757 vs baseline dev CER: 0.1676 — only +0.008pp gap on dev. The test
gap is larger (+0.017pp), suggesting the cos² mask acts as a mild regularizer on dev
but doesn't transfer fully to test. This could indicate:
- The cos² periodic structure overfits to the specific sequence lengths in the dev set
- Or the remaining oscillation bias (non-negative but still periodic) creates a subtle
  distribution shift between dev and test temporal characteristics

### Loss improvement

B² test loss: 0.8546 vs B test loss: 0.9163 — a 6.7% improvement. Removing negative
weights allows the model to learn much cleaner probability distributions. But baseline
loss is 0.7988, still meaningfully better.

---

## Mathematical Efficiency Analysis

### Computational cost

| Config | Matmuls per layer | Memory overhead | Extra params |
|---|---|---|---|
| A (real) | 2 (fwd + bwd) | 0 | 0 |
| B, C, D (cos) | 4 (2×fwd + 2×bwd) | 0 (same intermediate sizes) | 0 (B,C) / 6 (D) |
| B² (cos²) | 6 (2 real + 4 complex) | 0 | 0 |

Config B/C/D require 2× the attention matmuls vs baseline.
Config B-cos² requires 3× the attention matmuls (real + complex(2θ) averaged).

At this model scale (d_model=256, T~300 after subsampling), the attention matmuls are not
the bottleneck — wall time is comparable across all configs (~66–68s/epoch). At larger
scales the 2–3× matmul cost would become significant.

### Expressiveness analysis

The cos-decay family adds exactly **one structural parameter** (theta) to an already-rich
per-dimension decay system. The baseline LION attention has `d_model = 256` learned decay
dimensions per layer, giving 256 independent locality biases. A single scalar theta
modulates all 256 dimensions identically with a periodic factor.

The information content of this modulation is:
```
cos(θ·δ) = 1 bit of information (one frequency)
256 per-dim decays = 256 independent "frequencies" of exponential decay
```

The oscillation theta provides ~0.4% additional structural information on top of the
existing decay spectrum. This explains why the effect is measurable but small.

### Can per-dimension theta improve this?

Making theta per-dimension (θ_d for d=1..256) would give each dimension its own oscillation
frequency. This adds 256 params per layer (1536 total). The effective mask per dimension
would be:
```
M_d[i,j] = exp(w_d)^|i-j| · cos(θ_d · |i-j|)
```

Each dimension would have an independent decay profile: magnitude `r_d` and frequency `θ_d`.
This is mathematically equivalent to a **complex-valued diagonal SSM**, which is known to
work well in certain settings (S4D, DSS). However:

1. Per-dim cos (not cos²) still has negative weights per dimension
2. Per-dim cos² would be non-negative and equivalent to `0.5·r^δ + 0.5·r^δ·cos(2θ_d·δ)` —
   effectively a two-component mixture of a monotone decay and an oscillating one
3. The RWKV-6 decay mechanism already subsumes much of what per-dim theta would provide:
   the per-dim decay rates create a Fourier-like basis of exponential decay modes

---

## Reassessment: Can Complex Decay Work for ASR?

### Evidence from Config D

The theta trajectory is the most scientifically interesting result in the entire complex-decay
series. It shows that:

1. **The gradient recognises multi-scale temporal structure** in speech — lower layers want
   slower oscillation (word-level), upper layers tolerate faster oscillation (sub-syllabic)
2. **But the cos mask is the wrong way to encode this structure** — negative weights destroy
   the benefit

### Evidence from Config B-cos²

cos² recovers half the gap. A hypothetical per-dimension cos² with learnable θ_d could
potentially close the remaining gap, but at 2-3× compute cost and marginal expected gain.

### Final verdict

**The idea has a valid mathematical intuition but the wrong execution for ASR.**

The core insight — that different temporal scales matter for different layers — is confirmed
by Config D's theta trajectory. But the mechanism (multiplicative phase modulation of the
attention mask) is inferior to alternatives that achieve the same goal:

1. **Multi-scale decay initialization** — simply initializing different heads with different
   decay rate distributions achieves multi-scale attention without oscillation, negative
   weights, or extra compute
2. **ALiBi-style additive position bias** — adds per-head linear bias to attention logits,
   achieving controllable locality without modifying the mask multiplicatively
3. **ConvShift** — run-006 showed that a learned temporal convolution before attention
   provides a simpler, more effective temporal bias (+1.7% CER improvement) than any
   complex-decay variant

---

## Proposed Next Actions

### 1. Config D-cos² (diagnostic — per-layer learnable θ with non-negative mask)

**Why:** Config D showed theta doesn't collapse. Config B-cos² showed non-negative masks are
much less harmful. Their combination would answer: does the learned multi-scale theta
structure from Config D survive when the negative-weight problem is removed?

**Implementation:** `learnable_theta=True, use_cos2=True` — already supported by the codebase.
```python
"bidir_rwkv6_cplx_d_cos2": BidirRWKV6ComplexEncoder(..., theta_init=0.31, learnable_theta=True, use_cos2=True)
```

**Expected outcome:** theta trajectory similar to Config D, CER between B-cos² (0.1955) and
baseline (0.1790). If CER < 0.183 (within 2% of baseline), the multi-scale cos² structure
is a viable attention modification worth further investigation. If CER > 0.190, close the
hypothesis entirely.

### 2. cos² + ConvShift combination

**Why:** ConvShift (run-006) and cos² modify different things — ConvShift modifies the input
mixing before attention, cos² modifies the attention mask shape. They are orthogonal and
could stack.

**Implementation:** Combine `bidir_rwkv6_conv` with `cos²` attention — replace the real LION
attention in the ConvShift encoder with cos² complex attention.

**Expected outcome:** If gains are additive: ~0.1760 (Conv) - 0.005 (cos² structural benefit)
= ~0.171 CER. This would be a new best. If they don't stack, the result will match run-006.

### 3. Multi-scale decay init (no complex arithmetic)

**Why:** Config D proved the gradient wants different temporal scales per layer. We can achieve
this much more cheaply by initializing the real-valued decay parameters with a layer-dependent
schedule — lower layers get slower decay (wider attention), upper layers get faster decay
(sharper attention).

**Implementation:** Modify the baseline `bidir_rwkv6.py` decay initialization:
```python
# Current: decay_speed[n] = -6 + 5 * (n / (d-1)) ** (0.7 + 1.3 * ratio_0_to_1)
# Proposed: scale the range based on layer_id
layer_scale = 0.7 + 0.6 * ratio_0_to_1  # lower layers: wider range, upper: narrower
decay_speed[n] = -6 + 5 * (n / (d-1)) ** (0.7 + 1.3 * ratio_0_to_1) * layer_scale
```

**Expected outcome:** This is the simplest intervention inspired by the Config D finding.
Zero extra params, zero extra compute. Worth trying before anything complex.

### 4. Close the complex-decay investigation line

Regardless of Config D-cos² results, the following should be reported as settled:
- Fixed-theta cos oscillation **hurts** ASR at all tested frequencies (B, C)
- cos² (non-negative) reduces but does not eliminate the degradation
- Learnable theta converges to a multi-scale hierarchy, confirming that the gradient
  recognises temporal structure, but the multiplicative mask form is not the right encoding
- The most impactful temporal modification found so far remains ConvShift (+1.7% from run-006)
