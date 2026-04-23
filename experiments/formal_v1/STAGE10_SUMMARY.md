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
   (gated/edge non-normal), CB-1, CB-3, CB-7.

Group 3 is the cross-experiment invariant. It is not "dense freedom
cannot help" — it is "dense freedom cannot help when it is not
aligned with a structural prior of the task signal."

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

## 6. Immediate next step

Stage 11.0a — **Linear Attention causal baseline**. Unknown CER on
this spine; the number is itself new information. Then Stage 11.3a
`linear_attn_convshift_multidil_symmetric` — the one experiment whose
outcome genuinely updates the thesis by testing the principle under
a different inductive-bias baseline.

No further Stage-10 RWKV-6-causal compute is scheduled. CB-2
`multidil_wide4` remains available as a cheap single-axis probe
if GPU slack opens during Stage 11 setup, but is not a blocker.

---

*Result log: `RESULTS.md`. Per-run output directories:
`outputs/rwkv6_*_seed42/`. Plots: `outputs/_stage10_plots/`.
Analysis record: `STAGE10_ANALYSIS.md` (pre-CB-sprint) + this file
(post-sprint).*
