# Stage 10 — Honest Summary

RWKV-6 causal sweep beyond the Stages 2–9 chain. Same spine: seed 42,
30 ep, LibriSpeech clean-100, 7 M encoder, d=256, 6 layers, 4 heads,
K=64, σ ≈ 0.0014.

## Final results

| Run | Family | Dev CER | Test CER | Verdict |
|---|---|---:|---:|---|
| `rwkv6` (vanilla reference) | — | 0.1258 | 0.1263 | ref |
| `rwkv6_rse_convshift` (Stage-3 abs-best) | A × B | 0.1145 | 0.1126 | prior win |
| `rwkv6_rse_strong_viscosity` (Stage-5 anchor) | B | 0.1185 | 0.1177 | prior win |
| `rwkv6_convshift_trap` (Stage-2 win) | A | 0.1150 | 0.1150 | prior win |
| `rwkv6_loglinear` (10.1) | A structural | 0.1240 | 0.1226 | PLATEAU |
| `rwkv6_m2rnn_sparse` (10.2) | C | 0.1276 | 0.1264 | tied-vanilla |
| `rwkv6_convshift_multidil` causal (10.3) | A | 0.1229 | 0.1224 | causality penalty |
| **`rwkv6_convshift_multidil_symmetric` (10.3-sym)** | **A** | **0.1153** | **0.1145** | **matches abs-best without RSE** |
| **`rwkv6_convshift_multidil_symmetric_v2` (10.3-sym + init-fix)** | **A** | **0.1013** | **0.1000** | **Paper 7 multi-dilation genuinely engages; see note below** |
| **`rwkv6_rse_convshift_multidil_symmetric_v2` (CB-1 + init-fix)** | **A × B** | **0.0973** | **0.0961** | **RSE × multidil orthogonal composition; first sub-0.10 causal RWKV-6** |
| `rwkv6_chanmix_bypass` (10.4) | D | 0.1251 | 0.1248 | PLATEAU |
| `rwkv6_orthogonal` (10.5, still training at ep 15) | B | 0.1518 (ep 15) | — | regression-track |
| `rwkv6_pom_vlift` (10.6) | D | 0.1254 | 0.1253 | PLATEAU |
| `rwkv6_rse_convshift_multidil_symmetric` (CB-1) | A × B | 0.1169 | 0.1156 | tied 10.3-sym |
| `rwkv6_convshift_multidil_symmetric_gated` (CB-3) | A dense-per-token | 0.1167 | 0.1157 | tied 10.3-sym |
| `rwkv6_qtail_lowrank_all_convshift_multidil_symmetric` (CB-7) | A × D | 0.1159 | 0.1150 | tied 10.3-sym |
| `rwkv6_frontend_v2` (CB-5a lean) | frontend | did not converge | — | failure mode (see §4) |
| `rwkv6_frontend_v2_matched` rescue ep 1 (CB-5b) | frontend | 0.8403 @ ep 1 | — | did not converge |

> *Init-gradient-trap footnote (2026-04-23).* In the broken-init `multidil_sym`
> runs above, dilation branches $d \in \{2, 4, 8\}$ had $\alpha_d = 0$ exactly
> at ep 30 — the mechanism collapsed to single-dilation DWConvShift + per-layer
> scalar. `_v2` reruns carry the init fix (α_{d>1}=0.01, branch_{d>1}.weight
> ~N(0, 0.01)) so gradients reach all branches. Priority-1 `_v2` result: dev
> 0.1013 / test 0.1000 (−0.0140 dev vs broken-init, ~10σ); at ep 30, α₂ exceeds
> α₁ at every layer 1–5, α₈ at layer 5 = 1.23 — full multi-scale engagement with
> a depth gradient. Paper 7's multi-dilation claim does replicate on RWKV-6 ASR.
> Cross-architecture `_v2` reruns (Mamba-2, LA, CB-3, CB-7) pending. CB-1
> `_v2` (above) lands at dev 0.0973 / test 0.0961 — RSE × multidil
> compose orthogonally (multidil α-pattern preserved under RSE; CB-1 is
> P1 −0.004 dev, broken-init CB-1 −0.020 dev ≈ 14σ). See
> `MULTIDIL_INIT_FIX_HANDOFF.md`.

## 1. What this stage established

**One new mechanism clears the ceiling without RSE.** `multidil_sym`
(±8-frame symmetric DWConv with parallel dilations {1,2,4,8}, learned
per-layer α_d) reaches test 0.1145 — ties Stage-3's `rse_convshift`
without the RSE transition-side rotation. This is the only
structurally-new input-side win since Stage 2.

**Causality has a hard additive cost on the input side (+0.008, ≈ 5σ).**
`multidil_sym` vs `multidil_causal` isolates a pure streaming cost. Not a
sequence-modeling property — a signal-processing property.

**The cross-experiment invariant from Stages 8–9 now generalises
across mechanism families and base mechanisms.** Dense per-token
freedom added on top of any working base produces the same
saturation pattern — mechanism engages SGD, CER does not follow.
Demonstrated in Stage 10 on three more axes:
- CB-1 (RSE × multidil_sym): dev 0.1169 — tied 10.3-sym
- CB-3 (content-conditional gated α_d): dev 0.1167 — tied 10.3-sym
- CB-7 (Kronecker feature lift × multidil_sym): dev 0.1159 — tied 10.3-sym

All three engaged; none moved CER. Same pattern as A1′, T1, T2,
S9A, S9B on top of the Stage-5 anchor.

## 2. The real reason the wins worked

Looking back across Stages 2–10, the wins share a single property that
the failures do not. Every mechanism that moved CER added a
**computation the model could not otherwise perform, aligned with a
structural prior of the task**. Every mechanism that did not move CER
added parameter freedom inside a function class the model already
reached.

Concretely:

- **RSE (Stage 3).** Changed the transition Lie group from
  $(\mathbb{R}_+)^K$ to $SO(2)^{K/2} \times (\mathbb{R}_+)^{K/2}$ —
  a strictly larger function class, unreachable from the existing
  projections. Damped-oscillator state aligns with formant dynamics.
- **Budget refinement (Stage 4, `rse_strong` = `rse_depth`).** Did not
  add freedom — **removed a misallocated constraint** from Stage 3's
  uniform π/4 clip. Two different ways of removing the same constraint,
  same outcome.
- **Rayleigh viscosity (Stage 5 Phase 3).** Imposed the physical
  prior $\lambda_{\text{eff}} = \lambda + \eta\,\theta^2$ coupling
  rotation and damping. Zero extra parameters of note; regularisation
  via the physical model, not expanded capacity.
- **ConvShift / multidil_sym (Stage 2, Stage 10.3-sym).** Added
  bidirectional local temporal mixing at the input side — a
  computation the diagonal WKV scan structurally cannot perform. The
  post-subsampling dilation set {1,2,4,8} covers the 40–160 ms window
  that matches phoneme-to-syllable acoustic structure.

The common thread is **function-class extension aligned with a
task-structural prior, not parameter-capacity increase**. Budget
refinement is the one non-extension win; it removed a constraint
the previous mechanism imposed by mistake.

## 3. The real reason the failures failed

Every Stage-2–10 null falls into one of three groups, each describable
structurally:

1. **Reparameterisation-absorbable.** The new parameters lie in a
   subgroup of the old function class; existing projections can
   absorb them. Example: Stage-2 solver variants (`trap`, `trap_var`,
   `gen2`), Stage-7A static readout gauge (shown dead in the D2
   diagnostic).
2. **Same function class, redundant parametrisation.** The new
   mechanism names a subset of what existing parameters can already
   fit. Example: the Stage-6 quadratic-lift cluster (hadamard_n2,
   pom_vlift, qtail with different forms), Stage-10.1 loglinear
   direct-sum readout, Stage-10.4 chanmix_bypass.
3. **Dense per-token freedom with no task-structural content.** The
   mechanism genuinely extends the function class (movable, non-
   absorbable), but its per-token freedom is not matched to a
   structural prior the task exposes. SGD uses the capacity — diagnostic
   probes show mobility, sometimes substantial — but the usage is
   incoherent under the loss. Example: A1′ (data-dep readout phase),
   T1 (recurrent delta), T2 (non-normal RSE dense per-token), S9A/B
   (gated/edge non-normal), CB-3, CB-7 (CB-1 was originally listed
   here; now revised — see note below).

Group 3 is the cross-experiment invariant. It is not "dense freedom
cannot help" — it is "dense freedom cannot help when it is not
aligned with a structural prior of the task signal."

**Important revision post-v2 init fix (2026-04-23).** The original
Stage-10 CB-1 (RSE × multidil) was logged in Group 3 and framed
as falsifying the "RSE and multidil compose orthogonally" (H-orth)
hypothesis. That reading is **overturned by CB-1 v2**: with the
MultiDilationDWConvShift gradient trap fixed, CB-1 v2 lands at
test 0.0961 (vs broken-init CB-1 0.1156, a 14σ drop), **below
both P1 v2 single-mechanism (0.1000) and Stage-3 abs-best
`rse_convshift` (0.1126)**. **RSE and multi-dilation do compose
productively within axis 1** — broken-init CB-1's apparent
"saturation-tie" was a mechanism-suppressed artefact. CB-3 v2
and CB-7 v2 are pending and may revise their listing similarly.
The invariant's actual scope is narrower than originally stated:
it applies to dense per-token freedom on a base where the added
freedom is not matched to a structural prior — A1′, T1, T2, S9A/B
remain cleanly in Group 3. Only the Stage-10 multidil-based
compositions (CB-1/3/7) need revision, and only because the base
mechanism they extend was not actually engaging.

**CB-5 frontend replacement did not converge** at either the lean
(413 K) or matched (1.94 M) configuration, even after the
activation-on-final-stage fix. Ep-1 dev CER 0.84 at matched suggests
the encoder body is initialised against input statistics the vanilla
frontend produces, and an independently-trained replacement frontend
does not fit stably at this spine size. This is an **optimisation
finding**, not a structural refutation of "feature extraction could
matter." A stable replacement would require encoder retraining or
an additive-branch design that preserves v1 activations at init.

## 4. Where the real research attention should go

The thesis question is expressivity and efficiency of linear-time
sequence models. The Stage-10 corpus sharpens the question into a
specific empirical principle:

> **Linear-time sequence mixers improve over baseline when — and only
> when — a mechanism extends the function class to match a structural
> prior the task signal actually carries. Adding dense parametric
> freedom inside an existing function class, or outside without such a
> prior, does not convert into CER gain at 7 M / 30 ep / clean-100.**

This principle is now supported by 25+ single-seed runs across four
mechanism families, two base mechanisms, and two parameterisation
shapes (dense per-token and structural).

Three research directions follow from it, in priority order:

**4.1. Architecture transfer (Stage 11).** The principle predicts
that the two mechanisms that won (RSE-family, ConvShift-family)
should transfer to other linear-time architectures proportional
to how much those architectures lack the computation each adds.

- Linear Attention has **no local temporal mixing** and **no decay
  diversity**. Prediction: `linear_attn_convshift_multidil_symmetric`
  delivers a larger relative gain than on RWKV-6; `linear_attn_rse`
  delivers a large relative gain. Both test the principle.
- Mamba-2 has **native short DWConv** and **selective Δt continuous
  decay**. Prediction: `mamba2_convshift_multidil_symmetric` delivers
  a smaller relative gain (it already has local mixing);
  `mamba2_rse_strong_viscosity` tests whether block-complex
  transition adds over scalar selective decay.
- LION bidirectional recovers the 10.3-sym ↔ 10.3-causal +0.008 penalty
  structurally. `lion_convshift_multidil_symmetric` tests whether the
  input-side win compounds with native bidirectional state.

Stage 11 is the only direction whose outcomes genuinely update
the thesis — it tests the principle under different inductive-bias
baselines. Every outcome is informative: a differential transfer
pattern supports the principle; uniform saturation across architectures
extends the invariant to an architecture-wide ceiling at this scale.

**4.2. New task-structural priors on the transition side.** The
RSE win said "damped oscillators are a useful prior for speech."
The unexplored adjacent prior is **time-coupling of oscillator
modes** — not per-token phase jitter (which is what T2 and A1′
tried and failed), but slowly-varying frequency tracking across
windows. This is closer in spirit to viscosity (physical prior on
how rotation should behave) than to non-normal RSE (freedom to rotate
arbitrarily per token). A specific candidate: frequency-continuity
regularisation on θ via a small finite-difference penalty
$\sum_t (\theta_t - \theta_{t-1})^2$. Zero new parameters.
Tested as a prior, not as a mechanism.

**4.3. Input-side extensions on the same principle.** `multidil_sym`
wins at the phoneme-to-syllable scale (40–160 ms). `wide4`
({1,2,4,8,16}, to ~320 ms) tests whether there is a phrase-boundary
signal RWKV-6's scan also cannot access. If `wide4` > `sym`, it
supports the "missing-computation" reading of why ConvShift works;
if it ties, the ±8-frame scale is where the acoustic prior lives.
Single-axis test; one epoch of work to run.

## 5. Directions that are closed at this scale

The evidence is sufficient to close the following on this spine
(RWKV-6 causal, 7 M, 30 ep, clean-100):

- Higher-order ODE discretisation variants (Stage 2 covers this).
- Feature-side quadratic lifts in any parametrisation (Stage 6 + 10.6).
- Channel-mix bypass / α-gated interpolation (Stage 10.4).
- Dense per-token transition-side freedom in any operator family
  (Stage 7A, 8 T1/T2, 9 A/B, 10.5 orthogonal).
- Structural multi-scale readout via direct-sum decomposition (10.1).
- Sparing-use non-linear state at a single layer (10.2).
- Compositions of dense-per-token mechanisms with `multidil_sym`
  (CB-1, CB-3, CB-7).
- More variants of the above at this scale.

These are closed not because "mechanisms don't work," but because
they fail the function-class-extension-with-task-prior criterion
individually or compositionally.

## 6. Stage 11 baselines — complete

**Stage 11.0b — Mamba-2 causal** — on disk at `outputs/mamba2_seed42/`:
dev 0.1198 / test 0.1192 / WER 0.3615 at 30 ep, seed 42 on the
formal_v1 spine (7.27 M total). Mamba-2 vanilla lands between RWKV-6
vanilla (0.1258) and the Stage-5 RSE anchor (0.1185).

**Stage 11.0a — Linear Attention causal (Katharopoulos with explicit
L1 denominator)** — `outputs/linear_attn_causal_seed42/`, 2026-04-22:
**dev 0.2235 / test 0.2201 / WER 0.6342** at 30 ep, seed 42 (6.26 M
total, 4.35 M encoder). Gap to Mamba-2: **+0.1009 test CER**
(~+85% relative); gap to RWKV-6 vanilla: +0.0938. This is a
substantially larger cross-architecture gap than any mechanism delta
in the Stages 2–10 RWKV-6 discovery chain.

**Diagnostics — deep-layer attention collapse.** `diagnostics.json`
shows that at ep30 the Katharopoulos φ(k) feature map has effectively
died at depth: L2 and L4 have ≈99% of φ(k) components ≤ 1e-4,
the L1 denominator is ε-floored at 65–70% of valid positions at
those depths, and signal reaches the readout primarily via the
residual + FFN paths with the attention mechanism dead beyond L1.
This is a converged state (pattern established by ep10, stable
through ep30), not a training transient. The explicit-denominator
spec exists precisely to rule out denominator-floor failure modes,
and here it documents one.

This baseline datapoint sharpens the Stage 11 differential
predictions: LA's architecture-level deficit is even more dramatic
than "no local mixing / no decay diversity" — it also includes a
deep-layer feature-map collapse that bounded-decay mechanisms (RSE
+ viscosity, Stage 11.2b) should address structurally, while
input-side mechanisms (multidil_sym, Stage 11.1b) can enrich the
shallow-layer signals that reach the residual path without fixing
the deep attention itself.

**Stage 11.1 — complete 2026-04-22** (both 30 ep / seed 42, clean-100):

- **11.1a `mamba2_convshift_multidil_symmetric`** — dev 0.1079 /
  test 0.1055 / WER 0.3217 (7.33 M total; enc 5.41 M). Δ test CER
  vs 11.0b vanilla Mamba-2: **−0.0137 (−11.5% rel, ≈10σ)**.
- **11.1b `linear_attn_convshift_multidil_symmetric`** — dev 0.1977 /
  test 0.1930 / WER 0.5548 (6.28 M total; enc 4.36 M). Δ test CER
  vs 11.0a vanilla LA: **−0.0271 (−12.3% rel, ≈20σ)**.

**Pre-registered differential prediction supported in absolute
CER units.** LA gain (−0.0271) is 1.98× Mamba-2 gain (−0.0137),
matching the §5 Phase III prediction *LA gain > Mamba-2 gain
(LA has no local bias; Mamba-2's native DWConv captures part of
what multidil provides)*. In relative terms both architectures
improved by ≈12%; the absolute-vs-relative gap is itself
informative — Mamba-2's native DWConv did not fully absorb the
local-mixing benefit, both architectures gain proportionally to
their absolute room-to-improve. Halt criterion (§11) does NOT
fire. Stage 11.2+ cleared to proceed on user approval.

**Cross-architecture mechanism-attribution finding (major).**
Post-training diagnostics — *and direct α-parameter inspection of
the Stage 10.3-sym RWKV-6 run itself* — show **α_{2,4,8} = 0 at
every layer on all three architectures where multidil_sym has been
trained**:

- 11.1a Mamba-2: α_1 ∈ [0.89, 1.01] across depth, α_{2,4,8} = 0.
- 11.1b LA: α_1 ∈ [1.17, 1.47], α_{2,4,8} = 0.
- **Stage 10.3-sym on RWKV-6** (the declared Stage-10 win,
  `outputs/rwkv6_convshift_multidil_symmetric_seed42/best_model.pt`,
  dev 0.1153 / test 0.1145): α_1 ∈ [0.77, 2.19] with a clean
  monotonic depth pattern, α_{2,4,8} = 0.

The multi-dilation branches **never engage** on any of the three
architectures. The Stage 11.1 gains — and the Stage 10.3-sym win
itself — decompose into two distinct effects:

- **Add local mixing where none exists** (vanilla LA → LA + k=3
  symmetric DWConv: −0.0271 test CER; vanilla RWKV-6 →
  `convshift_trap` causal k=3 DWConv: −0.0113).
- **Symmetric > causal padding** at the phoneme scale
  (Mamba-2 single-dilation causal → symmetric: −0.0137;
  RWKV-6 multidil_causal → multidil_sym: −0.0079).

The RWKV-6 ablation chain directly confirms this:
`vanilla 0.1263 → convshift_trap 0.1150 (add causal local mixing,
−0.0113) → convshift_multidil 0.1224 (add dilations, still causal
— REGRESSION +0.0074) → multidil_sym 0.1145 (flip to symmetric,
−0.0079)`. Net `multidil_sym` vs single-dilation `convshift_trap`
is **+0.0005 — tied within σ**. Multi-dilation itself is a **null
axis** at this spine.

**Retrospective revision of the Stage-10 second-mechanism
headline.** Stage 10 treated `multidil_sym` as the second
transferable mechanism discovery (with RSE). On the evidence now
in hand, it is not a multi-dilation discovery. The real mechanism
discovery is **symmetric-padded local mixing at the phoneme
window** (post-subsampling ±1 frame ≈ 20–40 ms). This sharpens
rather than negates the STAGE10 §2 principle ("function-class
extension aligned with a task-structural prior") — the prior is
still phoneme-envelope local structure, but the function-class
extension is *the presence of symmetric local mixing*, not
*receptive-field multi-scale*.

A clean single-dilation symmetric DWConv ablation on all three
architectures (`rwkv6_convshift_symmetric` / `mamba2_convshift_symmetric`
/ `linear_attn_convshift_symmetric`) is queued as Stage 11.5,
gated after 11.2. Predicted outcome: matches or beats multidil_sym
within σ on every architecture. If confirmed, the thesis chapter 2
write-up should attribute the second-mechanism win to symmetric
padding, not dilation diversity.

**Additional LA-specific diagnostic** (still mechanism-level
evidence for the LA gain). Beyond the α_d finding, 11.1b's
diagnostics show that the single-dilation symmetric pre-mix
**rescues the middle-layer feature-map collapse observed in
11.0a**: vanilla LA's L2 had 99% of φ(k)≈0 and denominator
ε-floored at 66% of positions; with ConvShift, L2 has 0.1% φ(k)
tiny and 0% ε-floored denominator. Deep layers L3/L5 still
partially degraded but no longer catastrophically. The mechanism
story for LA's ConvShift gain is therefore: local-mixing
injection varies the key features enough to survive the ELU+1
saturation spiral at middle depth. This is independent of the
α_d=0 finding and still holds.

No further Stage-10 RWKV-6-causal compute is scheduled. CB-2
`multidil_wide4` remains available as a cheap single-axis probe
if GPU slack opens during Stage 11 setup, but is not a blocker.

## 7. Stage 11.2 — transition-side complex-pole transfer

**Complete 2026-04-23** (both 30 ep / seed 42 / clean-100):

- **11.2a `mamba2_rse_strong_viscosity`** — dev 0.1210 / test 0.1183 /
  WER 0.3541 (7.42 M total; enc 5.51 M). Δ test CER vs 11.0b vanilla
  Mamba-2: **−0.0009, tied within σ=0.0014 → NULL**.
- **11.2b `linear_attn_rse_strong_viscosity`** — dev 0.1428 /
  test 0.1422 / WER 0.4230 (6.37 M total; enc 4.46 M). Δ test CER
  vs 11.0a vanilla LA: **−0.0779 (−35% rel, ≈55σ) → BREAK**.

**Pre-registered prediction partially revised.** §5 Phase III predicted
"Mamba-2 moderate gain / LA large gain". LA observation is the
predicted BREAK; Mamba-2 observation is NULL, not moderate — the
prediction that Mamba-2's selective Δt "partially approximates
continuous decay but not complex-pole dynamics" is falsified at
this spine: complex-pole dynamics add nothing measurable on top of
selective Δt. LA gain is ~85× Mamba-2 gain in absolute CER — the
strongest architecture-deficit-proportional signal in the Stage-11
corpus so far. Halt criterion still does NOT fire (LA cell well
outside σ). Gap from LA to vanilla Mamba-2 narrows from +0.1009 to
+0.0230 test CER.

**Mechanism engagement on LA** (checkpoint parameter inspection at
ep 30, all three RSE mechanisms engaged meaningfully):

- θ LoRA grew from zero-init to ‖W_1‖ ∈ [3.9, 6.8] across layers;
  θ_base stretched slightly beyond init range to ±0.29.
- **Viscosity η trained to *negative* values with a clear depth
  gradient**: L0 −0.08 → L5 −0.29, |max| 0.64. Negative η means
  λ_eff = λ_base + η·θ² is *smaller* when rotation is larger —
  rotating modes decay *slower*. This is the OPPOSITE SIGN from
  RWKV-6 Stage-5 Rayleigh dissipation (η > 0 there).

  Interpretation: on RWKV-6 the physical prior was "high-frequency
  rotations dissipate faster" (Rayleigh). On LA the failure mode
  is not unbounded accumulation but feature-map collapse; SGD
  preserves rotational signals by reducing their decay, not by
  dissipating them. The depth gradient aligns with 11.0a's
  deep-layer collapse finding — deeper layers need more aggressive
  signal preservation.

**Mechanism engagement on Mamba-2** (11.2a NULL): not inspected in
detail yet; the null result is consistent with Mamba-2's SSM state
dynamics already spanning the block-complex function class via
dt-modulated real decay on a rich (H × N) state, so RSE's explicit
block-complex structure adds no new expressivity.

## 8. Stage 11.1 reinterpretation after the MultiDilationDWConvShift init fix

**v2 init-fix result (other instance, commit `3af846d`,
2026-04-23):** `rwkv6_convshift_multidil_symmetric_v2_seed42/` —
**dev 0.1013 / test 0.1000 / WER 0.3010** (7.76 M total; enc 5.84 M).
**Δ test CER vs v1 broken-init (0.1145): −0.0145, ≈10σ → BREAK**.
Post-training: α_{2,4,8} fully engaged — α_2 > α_1 at L1-L5; α_8
at L5 = 1.23. Depth gradient matches phoneme-to-syllable prior.

**Paper 7 multi-dilation claim replicates on RWKV-6 ASR once the
init trap is released.** The α_{d>1}=0 cross-architecture finding
reported in §6 above, and its retrospective-revision framing of
the Stage 10.3-sym headline, need partial revision:

- **On RWKV-6**, multi-dilation IS a real mechanism once unblocked.
  The broken-init v1 result (test 0.1145) was almost entirely the
  "symmetric-padded local mixing" effect — now confirmed by the v2
  uplift when the dilation axes become gradient-reachable.
- **On Mamba-2 and LA**, the existing multidil_sym runs were under
  the SAME broken init and therefore under-delivered too. Stage
  11.5b (`mamba2_convshift_symmetric`, pure single-dilation,
  test 0.1044) ties 11.1a multidil (test 0.1055) within σ,
  confirming that on Mamba-2 the broken-init multidil was
  effectively single-dilation.

### P2 + P3: cross-architecture multi-dilation v2 (2026-04-23)

Both of the flagged follow-up runs are now complete:

- **P2 `mamba2_convshift_multidil_symmetric_v2`**: dev 0.0982 /
  test **0.0967** / WER 0.2926 (7.33 M total; enc 5.41 M). Δ test
  CER vs 11.1a broken-init (0.1055): **−0.0088 (~6σ)**. Multi-
  dilation genuinely engages on Mamba-2 with the init fix. **This
  is the new single-backbone spine leader at 0.0967 test**,
  overtaking RWKV-6 P1 v2 (0.1000) and within σ of CB-1 v2
  composition (0.0961).
- **P3 `linear_attn_convshift_multidil_symmetric_v2`**: dev 0.1741 /
  test **0.1700** / WER 0.4854 (6.28 M total; enc 4.36 M). Δ test
  CER vs 11.1b broken-init (0.1930): **−0.0230 (~16σ)**. Largest
  absolute multi-dilation gain of the three architectures.

**Pre-registered architecture-deficit-proportional prediction
confirmed monotonically** across the three architectures with
working multi-dilation:

| Architecture | Broken-init test | v2 test | Δ test CER |
|---|---:|---:|---:|
| Mamba-2 (native DWConv, smallest deficit) | 0.1055 | 0.0967 | **−0.0088** |
| RWKV-6 (per-channel decay diversity, middle deficit) | 0.1145 | 0.1000 | **−0.0145** |
| LA (no local bias at all, largest deficit) | 0.1930 | 0.1700 | **−0.0230** |

Ordering LA > RWKV-6 > Mamba-2 in absolute gain, matching the
§5 Phase III pre-registered table. This is the cleanest
architecture-deficit-proportional pattern in the Stage-11 corpus:
each architecture's gain from the same axis-1 mechanism scales
inversely with how much of that mechanism the architecture
already has natively.

**Revised Stage-10 principle.** Symmetric-padded local mixing and
multi-scale receptive field are BOTH real axes: the former is what
the broken-init runs were delivering; the latter is what the v2
fix unlocks. Thesis chapter 2 should present this as a two-part
discovery — the Stage-10 "win" was the first half, the v2 fix
uncovered the second half. On RWKV-6, the single-dilation symmetric
(11.5a, 0.1137) ablation and multidil-v2 (P1, 0.1000) differ by
0.0137, quantifying the multi-scale-axis contribution on that
architecture.

### P5 + P6: compositions on working multi-dilation (2026-04-23)

Tests whether gated-α (CB-3) or Kronecker-feature (CB-7) add on
top of the working multidil substrate. CB-1 v2 (RSE × multidil,
test 0.0961) is the reference.

- **P5 `rwkv6_convshift_multidil_symmetric_gated_v2`**: dev 0.1150 /
  test **0.1136** / WER 0.3427. Δ vs CB-1 v2: **+0.0175 test CER
  → REGRESSION band**. Content-conditional gated α is a null axis
  even with working multidil — ties broken-init CB-3 within σ.
- **P6 `rwkv6_qtail_lowrank_all_convshift_multidil_symmetric_v2`**:
  dev 0.0989 / test **0.0988** / WER 0.2935. Δ vs CB-1 v2: +0.0027
  test CER → **MARGINAL band, tied within ~2σ**. Kronecker feature-
  side × working multidil doesn't add orthogonally above RSE ×
  multidil.

**Composition result**: on working multidil, only axis-1 (RSE ×
multidil_sym, via CB-1 v2 at 0.0961) adds productively. Neither
content-conditional gating (axis-1 dense-per-token) nor Kronecker
feature-side (axis-5) compose additively. RSE × multi-dilation
remains the sole productive composition.

## 9. Stage 11.5 — single-dilation symmetric DWConv ablation

**Complete 2026-04-23** (3 runs serial on GPU 0). Isolates the pure
symmetric-single-dilation effect, clarifying what the broken-init
multidil runs were actually delivering.

- **11.5a `rwkv6_convshift_symmetric`**: dev 0.1151 / test 0.1137 /
  WER 0.3428. **Ties broken-init v1 (10.3-sym, test 0.1145) within
  σ.** Same [0.5, 0, 0.5] init on both sides — clean comparison.
  Confirms v1 was effectively single-dilation. Gap vs v2 fixed-init
  (0.1000): +0.0137, quantifying what the trap cost on RWKV-6.
- **11.5b `mamba2_convshift_symmetric`**: dev 0.1074 / test 0.1044 /
  WER 0.3226 (**same param count as vanilla mamba2** — pure causal→
  symmetric padding swap, no new params). **Ties 11.1a (0.1055)
  within σ.** Same default-Kaiming init on both sides — clean
  comparison. Confirms broken-init Mamba-2 multidil was effectively
  single-dilation.
- **11.5c `linear_attn_convshift_symmetric`**: dev 0.2287 / test
  0.2245 / WER 0.6290. **+0.0315 WORSE than 11.1b (0.1930)**.
  Nominally suggests LA's α_1 scalar helps. **BUT init is not
  matched:** 11.1b's branch_1 was overridden to center-tap identity
  (my code override for bit-exact zero-regression vs vanilla LA),
  while 11.5c inherits `DWConvShift`'s default [0.5, 0, 0.5]
  smoothing init (output at init = 0.5·x[t-1] + 0.5·x[t+1] — no
  x[t] pass-through). Result is confounded by the init difference.
  A clean rerun with identity init would resolve whether α_1 itself
  is load-bearing or whether the smoothing init is simply harmful on
  LA's pure-accumulator architecture. **Flagged for follow-up.**

**Stage 11.5 synthesis.** On RWKV-6 and Mamba-2, single-dilation
symmetric reproduces the broken-init multidil result within σ —
confirming those pre-v2 multidil runs were effectively
"single-dilation + a per-layer α_1 scalar that did nothing."
The Stage-10 "`multidil_sym` as second transferable mechanism"
headline should remain attributed to symmetric-padded local
mixing on the single-dilation substrate; the multi-dilation axis
was inaccessible under the broken init.

**The v2 fix changes the landscape prospectively.** On RWKV-6,
fixed-init multidil (0.1000) beats symmetric-single-dilation
(0.1137) by 0.0137 (~10σ) — i.e., when SGD can access the
multi-dilation branches, it uses them productively. Whether the
same is true on Mamba-2 and LA is the cleanest open
architecture-transfer question. If confirmed, the thesis chapter 2
story is: "Symmetric local mixing is the universal primitive;
multi-dilation extends it productively when gradient-reachable;
earlier claims about multi-dilation being a null axis were an
init-trap artifact."

---

*Result log: `RESULTS.md`. Per-run output directories:
`outputs/rwkv6_*_seed42/` and `outputs/{mamba2,linear_attn}_*_seed42/`.
Plots: `outputs/_stage10_plots/`. Analysis records:
`STAGE10_ANALYSIS.md` (pre-CB-sprint), this file (Stage 10 post-sprint
and Stage 11 in flight), and `MULTIDIL_INIT_FIX_HANDOFF.md` (v2 init
fix context).*
