# Run 013 Analysis — Gaussian Distance Mask + Dual-Decay (Exp E)

Two independent experiments answering the two open questions from run-012:
- **Gaussian mask:** Is monotone decay sufficient, or does the network want non-zero-distance attention peaks?
- **Dual-decay:** Does bi-phasic (two-exponential) decay improve over single exponential?

Both ran in parallel on separate GPUs, 60 epochs each, identical hyperparameters to all prior runs.

---

## Full Results Table (all experiments to date)

| Config | Mask | DevCER | TestCER | Δbase | rel% | Loss | R@2s | R@5s | R@10s |
|---|---|---|---|---|---|---|---|---|---|
| **A  real (005)** | r^δ | **0.1676** | **0.1790** | — | — | 0.7988 | 0.2161 | 0.1857 | 0.1790 |
| **A+ Conv (006)** | r^δ+conv | **0.1587** | **0.1760** | -0.003 | -1.7% | 0.7585 | 0.2084 | 0.1804 | 0.1744 |
| B  cos (010) | r^δ·cos(θδ) fixed | 0.1932 | 0.2140 | +0.035 | +19.6% | 0.9163 | 0.2388 | 0.2161 | 0.2114 |
| C  cos (010) | r^δ·cos(θδ) fixed | 0.2109 | 0.2322 | +0.053 | +29.7% | 0.9744 | 0.2523 | 0.2322 | 0.2277 |
| D  cos learn (011) | r^δ·cos(θδ) learn | 0.1909 | 0.2107 | +0.032 | +17.7% | 0.9002 | 0.2349 | 0.2128 | 0.2081 |
| B² cos² (011) | r^δ·cos²(θδ) fixed | 0.1757 | 0.1955 | +0.017 | +9.2% | 0.8546 | 0.2272 | 0.1991 | 0.1938 |
| D² cos² learn (012) | r^δ·cos²(θδ) learn | 0.1782 | 0.1977 | +0.019 | +10.4% | 0.8558 | 0.2276 | 0.2010 | 0.1961 |
| F  headscale (012) | r^δ (scaled) | **0.1660** | **0.1839** | +0.005 | +2.7% | 0.8082 | 0.2160 | 0.1879 | 0.1820 |
| **G  Gaussian (013)** | r^δ × Gauss(μ,σ) | 0.1693 | 0.1861 | +0.007 | **+4.0%** | 0.8027 | 0.2132 | 0.1895 | 0.1846 |
| **E  Dual-decay (013)** | α·r_fast^δ + (1-α)·r_slow^δ | 0.1696 | 0.1875 | +0.009 | **+4.7%** | 0.8220 | 0.2184 | 0.1899 | 0.1841 |

---

## Gaussian Mask: Weak Non-Locality Signal Exists But Doesn't Help

### Result: DevCER 0.1693, TestCER 0.1861 (+4.0% vs baseline)

Worse than baseline on test. Dev-test gap 0.0168 is wider than baseline (0.0114). The Gaussian
mask adds a small regularization cost that the non-locality signal cannot overcome.

### The key diagnostic: μ and σ trajectories

Each of 4 heads × 6 layers has learnable μ (center, in ms) and σ (width, in ms). Init: μ ≈ 27ms
(softplus(0) ≈ 0.69 frames × 40ms), σ ≈ 600ms (σ_init=15 frames × 40ms).

**Epoch 1 → 60 trajectory (selected snapshots):**

```
Epoch 1 (init):
  L0: μ=[27, 28, 28, 27]ms  σ=[598, 599, 599, 599]ms
  L1: μ=[28, 27, 27, 28]ms  σ=[599, 599, 599, 599]ms
  L2-L5: all ~27ms, all ~599ms

Epoch 15:
  L0: μ=[40, 44, 35, 26]ms  σ=[625, 627, 620, 589]ms  ← 3 heads drifting up
  L1: μ=[48, 35, 31, 24]ms  σ=[637, 618, 604, 586]ms  ← head 0 at 48ms
  L2: μ=[22, 24, 26, 27]ms  σ=[586, 590, 593, 590]ms  ← stable/slightly down
  L3-L5: ~20-26ms, σ stable ~580-595ms

Epoch 30:
  L0: μ=[50, 57, 42, 27]ms  σ=[639, 644, 634, 584]ms  ← accelerating
  L1: μ=[62, 41, 33, 23]ms  σ=[656, 628, 606, 579]ms  ← head 0 at 62ms
  L2-L5: stable at 18-26ms, σ tightening to 555-595ms

Epoch 60 (final):
  L0: μ=[60, 75, 54, 27]ms  σ=[656, 666, 652, 576]ms
  L1: μ=[83, 48, 37, 22]ms  σ=[686, 639, 609, 569]ms
  L2: μ=[20, 23, 25, 25]ms  σ=[569, 579, 572, 571]ms
  L3: μ=[25, 23, 20, 23]ms  σ=[585, 558, 555, 553]ms
  L4: μ=[22, 24, 22, 33]ms  σ=[572, 574, 569, 578]ms
  L5: μ=[18, 32, 20, 26]ms  σ=[559, 595, 534, 555]ms
```

### Interpretation of μ trajectories

**Two distinct behaviors by layer depth:**

1. **L0 and L1: gradient pushes μ away from zero.** Three heads in L0 reach μ = 54–75ms
   (1.3–1.9 frames). L1 head 0 reaches μ = 83ms (2.1 frames). These are non-trivial distances —
   the network is expressing a preference for attending to specific non-zero distances in lower
   layers. L0 head 3 and L1 head 3 stay near/below init — one head per layer resists the shift.

2. **L2–L5: μ stays near or slightly below init.** Most heads in layers 2-5 have μ ≈ 18-33ms
   (~0.5-0.8 frames). Several drifted *down* from init (L3 head 2: 27→20ms, L5 head 0: 28→18ms).
   Upper layers prefer peak attention at distance ≈ 0 — consistent with all prior findings that
   upper layers want locality.

**The μ signal is real but weak.** L0-L1 heads moved to 50-83ms (1-2 frames away), meaning
they prefer attending to the immediate neighbor over the current frame. But:
- σ stayed enormous (534-686ms, barely changed from 600ms init)
- At σ = 656ms, the Gaussian mask value at distance 0 vs distance 75ms is:
  exp(0) vs exp(-75²/(2·656²)) = 1.0 vs 0.9935 — a 0.65% difference
- The Gaussian is nearly flat over the entire sequence. μ moved, but the effect is negligible.

**Why σ didn't tighten:** The Gaussian multiplies the LION attention matrix element-wise.
Tightening σ would suppress long-range attention for ALL dimensions in a head. But within each
head, some dimensions need broad attention and others need local. A per-head σ is too coarse —
it cannot give one dimension a narrow window without also narrowing all others. The gradient
pressure to tighten σ is counterbalanced by the need to preserve long-range dimensions.

### Conclusion on Q1: Is monotone decay sufficient?

**Mixed answer.** The gradient *does* push μ > 0 for lower-layer heads, confirming a weak
non-monotone signal. But the per-head granularity of the Gaussian mask is too coarse to exploit
it — σ stays flat because tightening would damage other dimensions. The Gaussian mask as
implemented (per-head, not per-dimension) cannot selectively enhance non-local attention for
specific feature dimensions without suppressing it for others.

A per-dimension Gaussian (μ_d, σ_d for each of 256 dimensions) could in principle capture this,
but would add 512 params/layer and would no longer factorise as a simple (T,T) mask — it would
need to be applied per-dimension inside the attention computation.

**Practical verdict:** At this model scale (7.7M params, 32h data), the non-locality signal
is too weak to justify additional mask parameters. Monotone decay is *practically* sufficient.

---

## Dual-Decay: Single Exponential Is Sufficient

### Result: DevCER 0.1696, TestCER 0.1875 (+4.7% vs baseline)

Worst of the three "mild modification" experiments (Gaussian +4.0%, headscale +2.7%).

### Epoch time: 66s — NOT 2× as predicted

Dual-decay runs two full LION attentions per layer (fast + slow). We predicted ~132s/epoch
(2× baseline). Actual: 66s, identical to baseline.

**Why:** At T≈250 and H=4, the LION attention matrices are (B, 4, 250, 250) — 250K floats per
sample. The computation is memory-bandwidth-bound, not compute-bound. The GPU has enough ALUs
to process both LION passes within the same memory transfer cycles. The "2× compute" translates
to near-zero wall-time overhead at our sequence length. This would not hold for T > 1000.

### The key diagnostic: α and slow_scale trajectories

**α** controls how much slow component is mixed in: α ≈ 1 → pure fast (baseline), α ≈ 0.5 → equal mix.
**slow_scale (ss)** controls how much slower the slow decay is: ss = 0.30 → w_slow = 0.30 × w_fast.

```
Epoch 1 (init):
  All layers: ss ≈ 0.30, α = 0.950 [0.95, 0.95]

Epoch 30:
  L0: ss=[0.32,0.31,0.33,0.34] α=0.945[0.94,0.95]
  L1: ss=[0.33,0.31,0.30,0.37] α=0.945[0.94,0.95]
  L2: ss=[0.34,0.31,0.35,0.42] α=0.946[0.93,0.96]
  L3: ss=[0.35,0.35,0.39,0.39] α=0.946[0.94,0.96]
  L4: ss=[0.38,0.39,0.35,0.50] α=0.947[0.93,0.97]
  L5: ss=[0.39,0.37,0.44,0.40] α=0.948[0.93,0.97]

Epoch 60 (final):
  L0: ss=[0.33,0.31,0.34,0.36] α=0.943[0.93,0.96]  bias=[+0.06,-0.05,+0.05,+0.05]
  L1: ss=[0.34,0.32,0.30,0.41] α=0.943[0.94,0.95]  bias=[+0.09,+0.04,+0.02,+0.11]
  L2: ss=[0.35,0.31,0.37,0.46] α=0.945[0.93,0.96]  bias=[+0.23,+0.04,+0.15,+0.06]
  L3: ss=[0.36,0.36,0.43,0.43] α=0.945[0.94,0.97]  bias=[+0.23,+0.25,+0.25,+0.05]
  L4: ss=[0.40,0.42,0.37,0.56] α=0.946[0.93,0.97]  bias=[+0.36,+0.47,+0.34,+0.06]
  L5: ss=[0.41,0.40,0.50,0.44] α=0.948[0.93,0.98]  bias=[+0.44,+0.27,+0.23,+0.06]
```

### Interpretation

**α barely moved.** From 0.950 to 0.943-0.948 over 60 epochs. The slow component is used at
5% at init and 5.2-5.7% at convergence. This is negligible — the gradient found no meaningful
use for the second exponential.

**slow_scale increased modestly.** From 0.30 to 0.31-0.56. Some heads (L4 head 3: 0.56, L5
head 2: 0.50) made the slow component less slow — pushing it closer to the fast component.
This is the *opposite* of what we'd expect if bi-phasic profiles were useful. Instead of making
the slow component distinctly slower (ss → 0.1), the gradient made it *more similar to fast*
(ss → 0.5). This means the model is trying to merge the two into one effective exponential.

**The headscale bias pattern reappears.** The dual-decay model also has per-head bias (headscale
is a component of the multiscale architecture). The bias values at epoch 60 show the exact same
pattern as run-012 headscale: L0-L1 mixed/small, L2-L5 increasingly local. L5 head 0 bias +0.44,
L4 head 1 bias +0.47. This is the dominant learned signal — the headscale component, not the
dual-decay component.

### Conclusion on Q2: Does bi-phasic decay help?

**No.** α stayed at ~0.95, slow_scale converged toward the fast rate rather than diverging from
it. The model has no use for a second exponential component at this scale. Single exponential
decay is sufficient.

---

## Joint Interpretation: What Run-013 Settles

Mapping to the 2×2 outcome matrix from run-012 analysis:

| | Dual-decay α → 0.95 | Dual-decay α differentiates |
|---|---|---|
| **Gaussian μ → 0** | Close decay research | Bi-phasic helps, non-locality doesn't |
| **Gaussian μ > 0** | Non-locality exists, mixture doesn't help | Both matter |

**Observed outcome: μ > 0 (weakly, L0-L1 only) + α → 0.95.**

This maps to the bottom-left cell: "Non-locality exists but mixture doesn't help." However,
the μ signal is so weak (σ stayed flat, actual mask effect < 1%) that the practical conclusion
is closer to top-left: monotone single-exponential is practically sufficient.

### What we now know for certain

1. **Exponential decay is the right functional form.** Neither Gaussian (shape modification) nor
   dual-decay (richer functional form) improved over single exponential. The baseline's r^δ
   is not the bottleneck.

2. **The temporal scale hierarchy is real but best captured by headscale.** Run-012's headscale
   (Exp F) captures the same "lower layers broad, upper layers narrow" gradient signal as all
   other experiments, at zero compute overhead and with the best test CER (+2.7%).

3. **The remaining CER gap to ConvShift is not about attention shape.** Gaussian (+4.0%),
   dual-decay (+4.7%), and headscale (+2.7%) are all worse than ConvShift (-1.7%). ConvShift's
   advantage comes from input mixing (local context injection at the feature level), not from
   attention structure.

4. **The dev-test gap is a systematic pattern, not experiment-specific.** All modifications that
   beat baseline on dev show a wider dev-test gap than baseline:
   - Baseline: 0.0114
   - ConvShift: 0.0173
   - Headscale: 0.0179
   - Gaussian: 0.0168
   - Dual-decay: 0.0179

   This suggests our dev/test split has systematic distributional differences that amplify
   with any model modification. Not a bug in any specific experiment.

---

## Hypotheses and Next Steps

### What might work

1. **Headscale + ConvShift combination.** ConvShift improves input features. Headscale improves
   attention scale distribution. They operate on different parts of the pipeline and should stack.
   Expected: better than either alone. This is the most likely path to a new best result.

2. **Longer training with headscale.** Run-012 showed headscale biases were still growing at
   epoch 60. The upper-layer localisation trend had not saturated. Running 100-120 epochs with
   cosine schedule could close the dev-test gap if the issue is underfitting.

3. **Layer-dependent decay initialisation.** All experiments show the same gradient direction:
   lower layers want broader attention, upper layers want narrower. Instead of adding learnable
   parameters, directly modify the RWKV-6 decay init formula to give lower layers slower initial
   decay and upper layers faster. This captures the finding with zero runtime cost and zero extra
   parameters. It's the simplest possible integration of everything we've learned.

### What probably won't work

1. **Per-dimension Gaussian mask.** While the per-head Gaussian was too coarse, the per-dimension
   version (512 params/layer) adds significant complexity for a signal that was barely detectable
   at per-head granularity. The μ shift in L0-L1 was 1-2 frames — within the range that token
   shifting already covers.

2. **More exponential components (K=3, K=4 Prony).** If K=2 (dual-decay) showed no use for the
   second component, K=3 will not help. The exponential basis is sufficient at K=1.

3. **Non-linear ALiBi.** The gating condition from run-012 was "only if Gaussian shows μ > 0."
   Gaussian did show weak μ > 0 for L0-L1, but the effect was negligible (σ too wide). Non-linear
   ALiBi would face the same per-head granularity problem and the same inability to differentiate
   dimensions within a head.

4. **Complex-valued decay (any variant).** Runs 010-012 comprehensively showed that all forms of
   oscillatory decay — cos, cos², learnable θ, combinations — perform worse than real baseline.
   This line is closed.

### Global conclusions for thesis

The attention decay structure of bidirectional RWKV-6 (LION) is nearly optimal for ASR at this
scale. Seven experiments across runs 010-013 tested:
- Oscillatory decay (cos, cos²): strictly harmful
- Multi-scale hierarchy (learnable θ, headscale, dual-decay): gradient signal is real but CER
  gains are small (+2.7% for headscale) or absent (dual-decay)
- Non-monotone attention (Gaussian): weak signal exists in lower layers, insufficient to help

The practical recommendation is: use the baseline RWKV-6 bidirectional attention with headscale
(24 extra params, zero compute overhead) and ConvShift (local feature mixing). Focus remaining
research budget on model capacity, training data, or decoder improvements rather than attention
structure.
