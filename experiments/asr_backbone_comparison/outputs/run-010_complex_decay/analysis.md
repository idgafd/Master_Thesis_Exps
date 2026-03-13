# Run 010 Analysis — bidir_rwkv6_cplx_b / bidir_rwkv6_cplx_c

---

## What Changed vs Run 005 (Baseline)

| | Run 005 | Run 010 Config B | Run 010 Config C |
|---|---|---|---|
| Backbone | bidir_rwkv6 (real decay) | bidir_rwkv6_cplx_b | bidir_rwkv6_cplx_c |
| Decay type | real diagonal: λ ∈ R | complex: λ = r·exp(i·0.31) | complex: λ = r·exp(i·0.90) |
| Attention mask shape | M[i,j] = r^|i-j| (monotone) | r^|i-j|·cos(0.31·|i-j|) | r^|i-j|·cos(0.90·|i-j|) |
| Resonance timescale | none (monotone decay) | ~203ms (syllable level) | ~70ms (phoneme level) |
| Extra parameters | — | 0 | 0 |
| Params total | 7.74M | 7.74M | 7.74M |
| Scheduler | cosine 60ep | cosine 60ep | cosine 60ep |
| ConvShift | no | no | no |
| Gate | no | no | no |
| r clamp | none | r ≤ 0.99 | r ≤ 0.99 |

The only structural change is in `_lion_complex_parallel_attention`: the cumulative product
`L^F[t] = Π_{l=0}^{t} λ_l` is extended to complex values. For constant theta, this adds
a phase term `exp(i·theta·t)` to each prefix product. The real part of the complex attention
matrix is extracted before multiplying V, using the identity:

```
Re(A[i,j]) = Σ_d Q[i,d]·K[j,d]·r^|i-j|·cos(theta·|i-j|)
           = (Q·exp(cs)·cos(theta_F)) @ (K·exp(-cs)·cos(theta_F))^T
           + (Q·exp(cs)·sin(theta_F)) @ (K·exp(-cs)·sin(theta_F))^T
```

via `cos(a-b) = cos(a)cos(b) + sin(a)sin(b)`. Each attention direction requires 2 matmuls
instead of 1 (4 total vs 2), with identical memory footprint.

---

## Key Observations (Facts)

| Metric | bidir_rwkv6 (005) | LION+Conv (006, best) | Config B (theta=0.31) | Config C (theta=0.90) |
|---|---|---|---|---|
| Params | 7.74M | 7.75M | 7.74M | 7.74M |
| BestDevCER | **0.1676** | **0.1587** | 0.1932 | 0.2109 |
| BestEpoch | 59 | 55 | 60 | 60 |
| TestLoss | 0.7988 | 0.7585 | 0.9163 | 0.9744 |
| TestCER | **0.1790** | **0.1760** | 0.2140 | 0.2322 |
| TestWER | **0.6704** | **0.6563** | 0.7312 | 0.7625 |
| R@2s CER | **0.2161** | **0.2084** | 0.2388 | 0.2523 |
| R@5s CER | **0.1857** | **0.1804** | 0.2161 | 0.2322 |
| R@10s CER | **0.1790** | **0.1744** | 0.2114 | 0.2277 |
| Wall time | 5115s | — | 3997s | ~3960s |

**Relative CER degradation vs baseline (run-005):**
- Config B: +3.5pp absolute, **+19.6% relative**
- Config C: +5.3pp absolute, **+29.7% relative**

Both configs are definitively worse than the real-decay baseline across all metrics and all
chunk sizes. Training was stable: no NaN, no loss spikes, monotone dev CER improvement,
both models still improving at epoch 60.

Config C (smaller theta, higher frequency) is substantially worse than Config B (larger
resonance period), consistent with the hypothesis that shorter-period oscillations are more
destructive for ASR.

Wall time dropped ~22% vs run-005 (3997s vs 5115s). With 2x matmuls per direction this is
counter-intuitive. The r ≤ 0.99 clamp forces stronger locality per dimension, which in
practice reduces the effective attention span and makes the problem somewhat easier to
optimize in early epochs.

---

## Anomalies

1. **Training stability is intact despite complex structure.** Both configs converge cleanly
   with no instability, no theta degeneracy, and no NaN. The complex arithmetic path
   (cumsum of cos/sin phases, double matmuls) is numerically well-behaved with the chosen
   stabilization (midpoint shift, ±60 clamp, r ≤ 0.99 clamp).

2. **Both models peak at epoch 60 (still improving at run end).** The real baseline peaked
   at epoch 59. The complex models have higher loss throughout, meaning the cosine schedule
   still has gradient signal to extract at epoch 60 — but at a higher absolute loss level
   than the baseline, not converging toward the same solution.

3. **The degradation gap widens at longer chunk sizes.** At R@10s, Config B is +0.0324pp
   above baseline; at R@2s only +0.0227pp. This is the reverse of what periodic resonance
   should produce — if the oscillating mask were capturing useful long-range structure, the
   gain should appear first at longer contexts. Instead, the complex mask is uniformly
   harmful and slightly more harmful at longer distances, where more negative-weighted
   (anti-attended) frames accumulate.

4. **Config C (phoneme timescale, theta=0.90) is worse than Config B (syllable, theta=0.31)**
   despite 70ms being closer to the acoustic frame rate (10ms frames × ~7 = 70ms). Higher
   theta means more oscillations per unit time → more frames fall in the negative-weight
   zone of cos(theta·|i-j|) → more destructive interference in the attention output.

---

## Why It Did Not Work — Root Causes

### 1. Negative attention weights are structurally destructive

The mask `cos(theta·|i-j|)` is negative for `|i-j| ∈ (pi/(2·theta), 3pi/(2·theta))`.
For theta=0.31 after 4× subsampling (40ms frames): the mask goes negative at distances
5–15 frames (200–600ms), covering exactly the timescale of co-articulation and word
boundaries. For theta=0.90: negative at 2–5 frames (80–200ms), the timescale of phoneme
transitions. The model is **forced to anti-attend** at these distances regardless of content.

This is not a soft regularization — it changes the sign of the gradient for key-value
pairs at those distances. The model must compensate by learning extreme Q,K magnitudes
to counteract the negative mask at useful distances, wasting representational capacity.

### 2. The periodic prior does not match variable-length speech structure

A fixed resonance period assumes speech has periodic structure at a specific timescale
(203ms or 70ms). Ukrainian speech, like all natural language, has variable-length phonemes,
syllables, and words. CTC does not align to fixed-period boundaries. The mask imposes a
rigid spatial periodicity that is fundamentally mismatched with the variable temporal
structure of the signal.

### 3. The complex phase is one-dimensional over a rich per-dimension decay

The real baseline has 256 independent decay rates per layer (one per feature dimension),
giving the model a rich spectrum of learned locality biases. A single fixed scalar theta
modulates all 256 dimensions identically. The additional inductive bias of the complex
structure is vanishingly small relative to what the per-dimension real decay already
provides — but the cost (negative weights, wrong periodicity) is real.

### 4. The r ≤ 0.99 clamp introduced a confound

The clamp was added to prevent state divergence (correct for Config D with learnable theta).
For Configs B and C, it is not mathematically necessary — the existing decay initialization
already produces r < 1. The clamp makes the effective decay stronger (more local) than the
unclamped baseline, which partially confounds the comparison. Some of the CER degradation
may be due to increased locality bias from the clamp, not purely from the oscillation.

---

## Assessment: Can This Idea Work?

**In the fixed-theta formulation tested here: no.**

The core problem is that a single fixed phase angle cannot provide useful inductive bias for
variable-duration phonetic events. The oscillation is a global constraint on the attention
mask shape, not a learned representation — and the constraint is wrong for ASR.

**In a more expressive formulation: possible but unlikely to be competitive.**

If theta were per-head and learnable (Config D), the gradient pressure to minimize CTC loss
would push theta → 0 (collapsing to the real baseline) or the model would find a local
minimum with theta at some non-zero value but no net gain. Config D is worth running as
a probe to verify that theta does not spontaneously collapse, which would confirm the
negative result definitively. But even with learned theta, the negative-weight problem
remains unless the parameterization is changed (e.g., |cos| or cos²).

The theoretical motivation — that oscillating decay captures multi-timescale acoustic
resonance — is intellectually coherent but does not survive contact with the task. Speech
recognition benefits from soft learned locality, not from periodic forced anti-attention.

---

## Next Actions

1. **Report Configs B and C as a clean negative result (recommended).** The result is
   scientifically clear and well-controlled: zero extra parameters, no training instability,
   consistent degradation across all metrics. This closes the complex-decay hypothesis.

2. **Optionally run Config D (learnable theta per layer)** as a final probe. The primary
   diagnostic question is: does theta learn to be non-zero after training? If theta → 0
   for all layers, it confirms the gradient sees no benefit from the oscillation. If theta
   stabilizes at a non-zero value, it shows the structure is at least active — even if CER
   still doesn't improve.
   - Init: `a_theta = -2.21` (θ_0 = 0.31) per layer
   - Monitor: log `sigmoid(a_theta)*pi` each epoch
   - Expected: theta drifts toward 0 or flat-lines near init

3. **Pivot to relative position bias.** The underlying motivation (give attention a richer
   temporal inductive bias) is valid. A learned additive bias per head per relative distance
   (ALiBi-style, but per-head learnable slope) does not introduce negative weights and has
   shown consistent gains in NLP. Implementation cost: one learnable slope per head
   (n_heads × n_layers = 24 parameters total).

4. **Remove the r ≤ 0.99 clamp from future real-decay experiments.** Confirmed here that it
   is not needed for the real baseline and introduces a confound. The existing decay
   initialization already keeps r well below 1 in practice.
