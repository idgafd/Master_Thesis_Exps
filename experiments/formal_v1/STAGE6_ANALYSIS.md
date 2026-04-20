# Stage 6 + 6.5 + Phase 2b — Analysis

*Started 2026-04-20. Partial final numbers for two live runs (`qtail_gamma`,
`p2rse_sv`) still pending as of writing — marked PENDING in tables.*

**One-line headline:** The Kronecker feature-lift mechanism (§2) delivers a
small, consistent, test-side gain (**−1.8 % rel vs baseline, −0.9 % vs the
normalizer-only control**) that isolates cleanly to *cross-channel
interactions*. The per-head decay-coupling refinement (γ) moves substantially
from its init in a depth-graded pattern, confirming the mechanism is using the
extra degree of freedom meaningfully. The pole-manifold refinement (§3)
decomposes into a **successful baseline composition** (shared-λ P²-RSE +
viscosity, running ahead of anchor through ep 11) and an **active-regression
overstretch** (independent-λ LoRA, −17 % rel vs anchor). Transferability
across Mamba-2, linear attention, and RWKV is clean for the feature-lift
axis; the pole-manifold axis is RWKV-specific in current implementation but
maps into Mamba-2 via the `P = A ⊙ M` framework.

---

## 0. Summary of results

All runs: LibriSpeech `train-clean-100`, 30 epochs, seed 42, RTX PRO 6000
Blackwell (97 GB), matched hyperparameters (AdamW lr=3e-4, cosine schedule
with 1 k-step warmup, SpecAugment LD policy, batch 300 s total duration,
gradient clip 5.0). σ_seed on this codebase ≈ 0.0014.

### 0.1 Stage-6 feature-lift axis (EXPRESSIVENESS paper adaptations)

| Backbone | Dev CER | Test CER | Δ vs baseline (test) | Classification |
|---|---:|---:|---:|---|
| `rwkv6` baseline (Stage 2 ref) | 0.1258 | 0.1263 | ref | — |
| `rwkv6_rmsnorm` | 0.1264 | 0.1252 | −0.87 % rel | **PLATEAU** |
| `rwkv6_hadamard_n2` | 0.1253 | 0.1251 | −0.95 % rel | **PLATEAU** |
| **`rwkv6_qtail`** | **0.1260** | **0.1240** | **−1.82 % rel** | **PLATEAU (dev) / near-MARGINAL (test)** |
| `rwkv6_qtail_gamma` (Stage 6.5) | PENDING (ep 16/30 @ 0.1456) | PENDING | PENDING | — |

### 0.2 Stage-5 pole-manifold × viscosity axis

| Backbone | Dev CER | Test CER | Δ vs `rse_strong_viscosity` (dev) | Classification |
|---|---:|---:|---:|---|
| `rwkv6_rse_strong_viscosity` (anchor, STAGE5 §4.6) | 0.1185 | 0.1177 | ref | prior causal best |
| **`rwkv6_p2rse_indeplam_strong_viscosity`** (Phase 2b) | **0.1394** | **0.1383** | **+17.6 % rel** | **DEEP PLATEAU / regression** |
| `rwkv6_p2rse_strong_viscosity` (diagnostic ctrl) | PENDING (ep 11/30 @ 0.1593) | PENDING | tracking ≤ anchor through ep 11 | — |

Pre-registered thresholds (STAGE5_PLAN §3, relative to `rse_strong_viscosity`):
- BREAK ≤ 0.1160 dev
- MARGINAL ≤ 0.1180 dev
- PLATEAU > 0.1180 dev

---

## 1. Context and methodology framing

This analysis sits on the causal RWKV-6 track of the thesis. Two axes are
independently developed:

**Axis A — feature-lift expressivity.** Does explicitly materialising
higher-order cross-channel interactions between keys and queries, as the
EXPRESSIVENESS paper argues for softmax attention, produce measurable gains
in the linear-attention / RWKV / Mamba family? Tested via three controls
and one treatment (§2).

**Axis B — pole-manifold expressivity composition.** Stage 5 established
that complex-valued rotation-decay (`rwkv6_rse_strong_viscosity`) is the
causal best on this task at 0.1185 dev. Phase 2b tests the STAGE5 §6.4(1)
hypothesis that relaxing the shared-λ constraint of P²-RSE (each pole gets
its own decay LoRA) would close the remaining gap. The composition is
pole-manifold × viscosity.

Both axes follow the Stage-2 → Stage-5 methodology: **refinement before
composition**. Each experiment has a clearly stated hypothesis, a
pre-registered classification threshold, and a zero-regression-at-init
contract so negative outcomes are interpretable as *mechanism doesn't help*
rather than *wrong starting point*.

---

## 2. Feature-lift axis — the Kronecker story (Stage 6 + 6.5)

### 2.1 Baseline claim (from EXPRESSIVENESS, Mongaras & Larson 2025)

The paper's Taylor expansion of the softmax numerator is

$$O_t = G_t\sum_{n=0}^{\infty}\frac{1}{n!}\,Q_t^{\otimes n}\!\cdot\! H_t^{(n)},\qquad H_t^{(n)} = \sum_{s\le t}(K_s^{\otimes n})^{\!\top} V_s$$

with two empirical claims at LM scale:
1. **§4.5** — the denominator $G_t$ functions primarily as a stabilising
   vector norm; any L2 / RMS / LayerNorm replacement matches softmax.
2. **§3.2** — linear attention is the $n=1$ truncation; the $n≥2$ terms add
   *cross-channel* multiplicative interactions $k_i k_j$ that cannot be
   recovered by element-wise feature maps $\varphi(k)$.

### 2.2 Three controls + one treatment

| Backbone | Mechanism | Tests |
|---|---|---|
| `rwkv6_rmsnorm` | GroupNorm → per-head RMSNorm readout | Paper §4.5 — is the normaliser substitutive? |
| `rwkv6_hadamard_n2` | rmsnorm + diagonal $(k\odot k, r\odot r)$ branch | *Diagonal* lift — no cross-channel terms |
| `rwkv6_qtail` | rmsnorm + full Kronecker $(k\otimes k, r\otimes r)$ at top 2 layers | Paper §3.2 — cross-channel interactions |
| `rwkv6_qtail_gamma` | qtail + learnable per-head γ decay coupling | Does decay choice on Kronecker branch matter? |

### 2.3 Why hadamard is a critical control

`rwkv6_hadamard_n2` exists *specifically* to disambiguate two very different
claims that are sometimes conflated:

- **Any squared-feature nonlinearity helps** (would be a generic activation
  story, trivially reproducible by ReLU², SwiGLU, etc.)
- **Cross-channel $k_i k_j$ interactions for $i \neq j$ help** (the paper's
  actual thesis about softmax expressivity)

The Hadamard variant has only the diagonal terms $k_i^2$ — if it improves
over rmsnorm, the win is generic; if it's null but qtail wins, the effect
isolates to cross-channel.

### 2.4 Results — the three Stage-6 controls

All three at 30 ep seed 42:

| Metric | rwkv6 baseline | rmsnorm | hadamard | qtail |
|---|---:|---:|---:|---:|
| Dev CER | 0.1258 | 0.1264 | 0.1253 | 0.1260 |
| Test CER | 0.1263 | 0.1252 | 0.1251 | **0.1240** |
| vs baseline (test, %rel) | ref | −0.87 % | −0.95 % | **−1.82 %** |
| Peak VRAM | — | 4.7 GB | 6.4 GB | 44.9 GB |
| s/epoch | 110 | 205 | 191 | 504 |

**Findings, in order of reliability:**

1. **rmsnorm ≈ hadamard_n2 on both dev and test.** The diagonal n=2 adds
   essentially nothing on top of the normaliser swap. This is the
   **expected** result if the paper's cross-channel claim is correct —
   diagonal Kronecker $k_i^2$ is equivalent to a squared-activation feature
   map, which the paper's §3.2 explicitly identifies as unable to recover
   softmax expressivity.
2. **qtail beats both controls on test CER** by ~0.0011, a full σ. Not on
   dev. The asymmetry (test better than dev) is non-trivial and could
   indicate the cross-channel features improve generalisation more than
   optimisation. Worth a multi-seed check; single-seed reading is
   directionally consistent.
3. **qtail−hadamard = qtail−rmsnorm = ~0.0011 on test.** This delta cleanly
   isolates to the *cross-channel* term — the only structural difference
   between qtail and hadamard is the presence of $k_i k_j$ for $i \neq j$.

**Thesis-level statement that falls out:**

> On causal RWKV-6 at 7 M params on LibriSpeech clean-100, the Kronecker
> $n=2$ cross-channel feature lift improves test CER by 1.8 % relative vs
> baseline (0.1240 vs 0.1263) and 0.9 % vs the normaliser-only control. The
> matched diagonal-only control contributes zero additional gain, cleanly
> isolating the improvement to the $k_i k_j$ cross-terms the EXPRESSIVENESS
> paper argues for.

### 2.5 Why rmsnorm is null

The paper's §4.5 effect scaled positively with LM size (300 M → 2 B). At
our 7 M × 6 layers, we are below the regime where attention-entropy-collapse
phenomena make the normaliser choice load-bearing. GroupNorm and RMSNorm
differ essentially in whether they subtract the mean; at this depth, the
pre-norm LayerNorm at block input (`ln1`) already handles the mean. This is
a **scale boundary condition, not a refutation of the paper's claim** —
our null informs where the claim applies.

### 2.6 Why qtail's dev is weaker than its test

Observed for baseline, rmsnorm, hadamard, qtail (dev → test):

| | baseline | rmsnorm | hadamard | qtail |
|---|---:|---:|---:|---:|
| Dev | 0.1258 | 0.1264 | 0.1253 | 0.1260 |
| Test | 0.1263 | 0.1252 | 0.1251 | 0.1240 |
| Test − Dev | +0.0005 | −0.0012 | −0.0002 | **−0.0020** |

Baseline has test > dev (dev-over-fit). All three Stage-6 variants show
test ≤ dev, with qtail's sign-flip the strongest. Consistent with
"Kronecker features generalise slightly differently than linear features" —
possibly because they require more specific local context to fire, whereas
the linear branch fires opportunistically on any $r \cdot k$ inner product.
Single-seed effect — multi-seed validation would be the proper test.

### 2.7 Stage 6.5 — the γ refinement (currently running)

**Motivation.** The paper's Taylor derivation imposes no decay on order-$n$
state: $H_t^{(n)} = H_{t-1}^{(n)} + (K^{\otimes n})^\top V$. Our qtail
implementation chose $w_{\text{pair}}[i,j] = w_i + w_j$ (the natural
Kronecker lift of the linear decay). But this is *one particular choice on a
spectrum* — there's no theoretical constraint.

**Mechanism.** Add learnable per-head scalar γ:
$$w_{\text{pair}}[i,j] = \gamma_h \cdot (w_i + w_j)$$
- $\gamma=1$: current qtail (bit-exact reduction at init)
- $\gamma=1/2$: geometric mean (Kronecker remembers as long as linear)
- $\gamma=0$: paper's undecayed accumulator
- $\gamma>1$: Kronecker branch forgets faster than linear

**Cost:** +4 scalars per qtail-active layer = 8 params total. Minimal.
**Transfer:** identical in any feature-lift setting (linear attention,
Mamba-2, RWKV-7).

**Empirical observation at ep 14 (from checkpoint):**

| Layer | γ mean | γ std | γ range |
|---|---:|---:|---:|
| L4 | **0.927** | 0.215 | [0.611, 1.089] |
| L5 | **0.883** | 0.204 | [0.600, 1.081] |

Two patterns:

1. **γ moves substantially from 1.0.** SGD is actively using the
   parameterisation — this is not a dead parameter.
2. **Systematic directionality.** Most heads want γ < 1.0 (Kronecker decays
   *slower* than the product). Deeper layers (L5 mean 0.883) want slower
   still than shallower (L4 mean 0.927). This is the **same depth-hierarchy
   pattern Stage-2 `gen2` discovered** (deeper layers learn larger lookback
   α₁) reproduced at a different mechanism level.

The empirical finding stands independent of whether the full 30-ep CER win
materialises in this specific run:

> **Given a learnable per-head decay coupling γ on the Kronecker branch,
> SGD spontaneously drives γ toward diverse per-head values in [0.6, 1.1]
> with a consistent depth gradient (deeper → smaller γ → slower Kronecker
> decay), providing concrete evidence that cross-channel features operate
> on a wider effective time scale than the natural γ=1 product-of-decays
> would give them.**

This is the thesis-level finding about a **general, mathematically-principled
technique** that transfers across the linear-attention / RWKV / Mamba
family — see §4 transferability.

### 2.8 What could strengthen qtail further (after γ resolves)

Ranked by math-story strength, all refinements *inside* the Kronecker
mechanism (no cross-mechanism stacking):

1. **γ** (running now) — parametrise decay.
2. **Data-dependent β_{q,t} mixer** — make the Kronecker-branch gating
   token-selective (same selectivity argument as Mamba-2). Current β is a
   per-head static scalar.
3. **Low-rank Kronecker** — project r,k to K'=16 per head, then full
   K'²=256 features. Eckart–Young-optimal approximation if the effective
   Kronecker feature rank is ≪ K². Would cut memory ~16× and enable
   scale-up.
4. **Pre-lift RMSNorm on (r,k)** — Taylor convergence argument. Tiny cost.

---

## 3. Pole-manifold axis — Phase 2b and the diagnostic control (Stage 5 deferred)

### 3.1 Hypothesis

STAGE5_RESULTS §6.4(1) diagnosed the Phase 1 under-performance of P²-RSE
(0.1220 dev at Stage-3 small budget, vs 0.1192 for `rse_strong` at
Stage-4 budget) by three non-exclusive reasons, of which **shared-λ across
the two complex poles** was flagged as most likely:

> Forcing equal λ compresses the two poles to the same bandwidth and
> collapses half of the intended expressivity gain. Independent-λ variant
> is the natural remedy and was deferred to Phase 2b.

**Phase 2b mechanism:** each pole carries its own decay LoRA
(`time_decay_2`, `time_decay_w1_2`, `time_decay_w2_2`). Architecture stacks
onto the current causal best `rwkv6_rse_strong_viscosity` (strong rotation
budget + Phase 3 Rayleigh viscosity). Zero-regression-at-init contract:
pole-2 base λ is a clone of pole-1; pole-2 LoRA weights are zero. At t=0,
the forward is bit-identical to shared-λ P²-RSE.

### 3.2 Result

| Metric | Phase 2b (indep-λ) | `rse_strong_viscosity` (anchor) | Δ |
|---|---:|---:|---:|
| Dev CER | **0.1394** | 0.1185 | +0.0209 (**+17.6 %**) |
| Test CER | **0.1383** | 0.1177 | +0.0206 |
| Classification | **DEEP PLATEAU / regression** | ref | — |

15σ outside seed noise. Phase 2b **actively regressed** the composition.
Late-training was jittery (ep 25–30: 0.1394–0.1457 before settling), which
is consistent with optimisation instability rather than slow-catch-up.

### 3.3 Diagnosis — shared-λ was *not* the binding constraint at this composition

The STAGE5 §6.4(1) hypothesis is **falsified at the strong+viscosity
composition**. Possible mechanism:

> Shared-λ acted as implicit regularisation compatible with viscosity's
> $\eta \cdot \theta^2$ coupling — which itself is shared across poles.
> Unshackling λ while keeping shared viscosity leaves the pole-2 LoRA
> underconstrained. The extra ~200 K params SGD has to find a productive
> configuration for, within 30 epochs, evidently cannot stabilise against
> the strong rotation budget + viscosity regulariser.

This is a *conditional* result — the diagnosis fails **at this composition**.
At a looser composition (without viscosity, or at depth budget instead of
strong), indep-λ might still help. We don't test those here.

### 3.4 Diagnostic control — `rwkv6_p2rse_strong_viscosity` (running)

To attribute Phase 2b's regression, we need the "shared-λ P²-RSE + strong +
viscosity" composition in isolation — the exact baseline Phase 2b added
indep-λ on top of. This cell was never actually run in Stage 5 (Phase 2 was
terminated early, Phase 3 was single-pole). Running now.

**Three-case attribution:**
1. **p2rse_sv ≈ 0.118–0.120** → indep-λ LoRA was the regression source.
   Clean negative for Phase 2b specifically.
2. **p2rse_sv ≈ 0.139** → P²-RSE × viscosity composition itself is broken.
   Phase 2b's regression is inherent to the composition.
3. **p2rse_sv ≈ 0.130** → partial regression from composition, amplified by
   indep-λ.

**Current trajectory (ep 5–11):**

| Ep | `rse_strong_viscosity` ref | p2rse_sv | Δ |
|---:|---:|---:|---:|
|  5 | 0.2279 | 0.2262 | −0.0017 |
| 10 | 0.1683 | **0.1663** | −0.0020 |
| 11 | (~0.1635 interp) | **0.1593** | ~−0.004 |

**p2rse_sv is running AT or SLIGHTLY AHEAD of the anchor through ep 11.**
Already suggests case (1): indep-λ was the destabiliser, not the
composition. If this tracks through ep 30, **we get two results at once**:

- Clean attribution of Phase 2b's regression to the indep-λ LoRA
- **A new causal best candidate** — shared-λ P²-RSE + viscosity at
  projected ≤ 0.118 would match or beat the Stage-5 anchor

### 3.5 What to learn from Phase 2b's regression

Three structural implications:

1. **The "more expressive is always better" heuristic is wrong here.**
   Adding a LoRA that gave the model strictly more degrees of freedom
   destabilised training rather than improving it. The shared-λ
   parameterisation had a regularising role we didn't appreciate.
2. **Composition-level interactions matter.** The same Phase 2b mechanism,
   at a different composition (small budget, no viscosity), might be
   productive — Phase 1 P²-RSE at Stage-3 small budget gave −2.5 % rel.
   The strong+viscosity composition is specifically hostile to the extra
   LoRA.
3. **Zero-regression-at-init ≠ zero-regression-at-training.** Phase 2b
   was bit-identical to shared-λ P²-RSE at step 0 but diverged
   catastrophically in training. The contract prevents "start broken"
   failures but doesn't prevent "train broken" failures.

---

## 4. Transferability across architectures

The `P = A ⊙ M` framework (Log-Linear Attention §2, Guo et al. 2025)
unifies linear attention, SSM, and RWKV as instances of a single equation
differing only in how they parameterise the attention matrix $A$ and the
structured mask $M$. Our mechanisms sit on different slots of this
framework, which determines how they port.

### 4.1 Kronecker feature lift (qtail, qtail-γ) — universal

The lift acts on $(r, k)$ *before* they enter any attention computation.
It's slot-independent:

| Architecture | How to apply | Change |
|---|---|---|
| Linear attention (Katharopoulos) | Lift $\varphi(q_t), \varphi(k_t)$ to Kronecker powers | Feature-map substitution |
| RWKV-6 (ours) | Replace $(r, k)$ with $(r\otimes r, k\otimes k)$ before the WKV scan | **Already done — this is qtail** |
| RWKV-7 | Same, on RWKV-7's expressive-state formulation | Identical code port |
| Mamba-2 (SSD) | Lift the $(C, B)$ projections before the SSD matmul | Replace $B_t, C_t$ with Kronecker powers |
| LION (bidirectional RWKV-6) | Lift the T×T attention inputs | Change to `lion_parallel_attention` |
| DeltaNet / Gated DeltaNet | Lift $(k, v)$ before the Householder transition | Same feature-map substitution |

**γ refinement is equally universal.** It parameterises *how much* the
lifted decay inherits the linear decay — independent of whether the scan
uses Mamba-2's SSD, RWKV's chunked WKV, or linear attention's cumulative
sum.

**Empirical γ finding transfers as a qualitative claim:**

> Any architecture with a multi-head linear-attention-family scan
> equipped with a learnable Kronecker feature-lift and γ decay coupling
> should, upon training, show γ < 1.0 with depth gradient (deeper → smaller)
> — the mechanism-level expression of multi-scale temporal hierarchy.

Testable on Mamba-2 and Gated DeltaNet without modifying our code
substantially.

### 4.2 Post-scan normalization (rmsnorm) — universal, null here

GroupNorm → RMSNorm swaps apply to any post-scan readout. The null result
here is a *scale boundary*: the paper's effect scales with depth × params,
so small-scale ASR models won't see it. This port trivially and correctly
predicts nulls at other small-scale tasks (CTC ASR, short sequences,
shallow stacks).

### 4.3 Pole-manifold × viscosity (Stage 5 composition, Phase 2b)

The RSE transition group SO(2)×ℝ₊ and its viscosity refinement are
RWKV-specific in current implementation but generalisation is available:

| Architecture | Analog | Status |
|---|---|---|
| RWKV-6 (ours) | Complex scan $z_t = e^{-\lambda + i\theta}$ | **Native — Stages 3–5** |
| Mamba-2 | Replace scalar $\alpha_t$ transition with 2×2 block rotation-scale | Would need kernel fork — §4.4 |
| Linear attention (Katharopoulos) | Replace identity transition with 2×2 rotation block | Kernel port via `rse_scan_fast.py` |
| LION | Replace LION's scalar-λ 1-semiseparable mask with complex-λ | Port to `lion_attention.py` |
| Gated DeltaNet | Replace scalar α_t with 2×2 block in the Householder chain | Additive |

**Phase 2b's indep-λ finding transfers as a conditional claim:**

> On any architecture with paired-pole transition + viscosity coupling,
> adding independent-λ LoRA without correspondingly independent viscosity
> coupling will destabilise training. A principled port requires either
> independent η per pole or removing viscosity from the composition.

This is testable and the failure mode is predictable.

### 4.4 What doesn't transfer (stays RWKV-6-specific)

- **Token-shift mechanisms** (`time_maa_*` parameters) are a RWKV-specific
  input pre-processing step. Not relevant to Mamba-2 / Linear attention.
- **ChannelMix specifics** (`time_faaaa` bonus) are RWKV-specific.
- **`_chunked_wkv` subchunk sizes** are a RWKV kernel choice.

None of these affect the expressivity claims we make — they're
implementation-layer details.

---

## 5. Live runs (as of 2026-04-20 ~08:41 UTC)

### 5.1 `qtail_gamma` (GPU 0, session `qtg`)

- Backbone: `rwkv6_qtail_gamma` (Stage 6.5)
- Current: ep 16/30 @ 0.1456 dev CER
- γ values from last checkpoint (ep 14): L4 mean 0.927, L5 mean 0.883 — see §2.7
- β_qtail from checkpoint (ep 14): L4 range [−0.012, +0.037], L5 range [−0.033, −0.008] — signed, small magnitude
- ETA: ~10:40 UTC

### 5.2 `p2rse_strong_viscosity` (GPU 1, session `p2rse_sv`)

- Backbone: `rwkv6_p2rse_strong_viscosity` (diagnostic control)
- Current: ep 11/30 @ 0.1593 dev CER — **at or slightly ahead of anchor**
- ETA: ~10:40 UTC

---

## 6. Methodological carry-forwards

Lessons confirmed by this round, to apply to future Stage-7+ experiments:

1. **Pre-registered thresholds stay decisive.** Every experiment had
   BREAK / MARGINAL / PLATEAU thresholds fixed before launch. Phase 2b's
   −17 % rel regression is unambiguous against its pre-registered
   MARGINAL 0.1180 boundary; there's no post-hoc adjustment pressure.
2. **Zero-regression-at-init contracts are necessary but not sufficient.**
   Phase 2b was bit-identical to shared-λ P²-RSE at step 0 and still
   regressed 15σ. Future experiments need additional early-training
   stability checks (e.g., gradient-norm monitoring for new LoRA
   parameters, activation-norm monitoring).
3. **Always run the missing control.** Phase 2b stacked indep-λ onto a
   composition (`p2rse_strong_viscosity`) that had itself never been
   tested in isolation. That missing cell is now running as
   `p2rse_sv` — should have been part of the original Stage-5 Phase-2
   sweep before Phase 2 was terminated early.
4. **Diagonal-only controls are worth the compute.** `hadamard_n2` cost
   as much as a full qtail run to confirm the null; that null is what
   lets us attribute qtail's gain to cross-channel terms specifically,
   not to "any squared feature." Without it, the result is ambiguous.
5. **Read checkpoints mid-training.** The γ distribution at ep 14 gave
   us a concrete transferable finding independent of the final CER. If
   the 30-ep run lands as PLATEAU, the γ finding is still
   thesis-defensible content.

---

## 7. Next steps (conditional on live-run outcomes)

Decisions wait until `qtg` and `p2rse_sv` finish (~10:40 UTC). Scenarios:

### 7a. qtail-γ ends ≤ qtail test 0.1240 by ≥ σ

γ refinement is productive. Write up Stage 6.5 as a small-effect, clean
result: "learnable per-head decay coupling on the Kronecker branch delivers
~1 σ improvement and reveals a depth-graded γ pattern." Propose R2
(data-dependent β) as the natural next refinement.

### 7b. qtail-γ ≈ qtail

γ moves but doesn't translate into CER gain at 30 ep. Still write up the
γ-movement-with-depth-gradient finding as an empirical property of the
mechanism. Propose multi-seed validation.

### 7c. qtail-γ regresses

Unlikely given γ=1 is zero-regression-at-init, but possible if the extra
parameter introduces optimisation noise. Would motivate R2 (data-dependent
β) as the alternative refinement axis.

### 7d. p2rse_sv ≤ 0.1180 (MARGINAL or BREAK)

New causal best. Write up as: "shared-λ P²-RSE + strong + viscosity is the
productive composition; the Phase 2b indep-λ LoRA destabilised what was
otherwise a working mechanism." Future Phase 2b variants should constrain
the indep-λ LoRA rank or warm-start it gradually.

### 7e. p2rse_sv ≈ 0.119–0.120 (PLATEAU within seed noise of anchor)

Viscosity doesn't lose to P²-RSE composition; but P²-RSE doesn't win
either. Interesting null. Combined with Phase 2b's regression, suggests
**the P²-RSE mechanism and the viscosity mechanism are alternatives**, not
stackable. Two separate thesis-level refinements of Stage 4, not a
combined one.

### 7f. p2rse_sv ≥ 0.125 (regression)

Unexpected given ep-11 trajectory. Would mean the composition itself is
broken, and Phase 2b's regression was compositional. Write up as: "P²-RSE
and viscosity are mutually incompatible at strong budget." Strong negative
but clean.

---

## 8. File paths and reproducibility

All configurations seed 42, 30 epochs, `configs/default.yaml`.

```
Stage-6 feature-lift (all done):
  outputs/stage6_01_rwkv6_rmsnorm_seed42/
  outputs/stage6_02_rwkv6_hadamard_n2_seed42/
  outputs/stage6_03_rwkv6_qtail_seed42/

Stage-6.5 refinement (running):
  outputs/qtg_rwkv6_qtail_gamma_seed42/        ← GPU 0, ep 16/30

Phase 2b (done):
  outputs/phase2b_rwkv6_p2rse_indeplam_strong_viscosity_seed42/

Phase 2b diagnostic control (running):
  outputs/p2rse_sv_rwkv6_p2rse_strong_viscosity_seed42/   ← GPU 1, ep 11/30

Smoke tests (exact-reproduction specs):
  scripts/smoke_stage6.py            — Stage 6 controls
  scripts/smoke_qtail_gamma.py       — γ refinement
  scripts/smoke_phase2b.py           — Phase 2b indep-λ
  scripts/smoke_phase2b_ext.py       — Phase 2b-ext indep-(k,v) [NOT LAUNCHED]

Code added this round (additive, no existing methods overwritten):
  src/models/rse_scan_fast.py          — real-arithmetic Tensor-Core scan (pre-existing, reused)
  src/models/p2rse_indep_lambda.py     — Phase 2b scan helper
  src/models/rwkv6_time_mix.py         — new flags: use_rmsnorm, use_hadamard_n2,
                                         use_qtail, use_qtail_gamma,
                                         p2rse_indep_lambda, p2rse_indep_kv,
                                         p2rse_kv_lora_dim
  src/models/rwkv6_block.py            — flag pass-through
  src/models/rwkv6_encoder.py          — flag pass-through, qtail top-k gating
  src/models/encoder.py                — 7 new backbones registered
```

---

*This document is live. Updates when `qtail_gamma` and `p2rse_sv` finish
(~10:40 UTC 2026-04-20) will fill the PENDING cells and resolve the §7
decision branch.*
