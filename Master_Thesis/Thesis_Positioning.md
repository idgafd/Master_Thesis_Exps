# Thesis Positioning — Probe-as-Instrument Frame

*Working positioning document. Not part of the LaTeX deliverable.
Records the framing decisions taken on 2026-04-29 so the writeup of
chapters 1, 3, 4, 5 stays internally consistent. Edit as the framing
sharpens.*

---

## 1. One-paragraph thesis claim

We characterise three mechanism families addressing distinct
expressivity deficits of causal linear-time RNNs — **Multi-Scale
Depthwise Convolution (MSDC)** for short-range temporal hierarchy,
**Chunked Value Decorrelation (CVD)** for associative-memory
interference, and **Damped Harmonic Oscillator (DHO)** for the
real-diagonal-vs-complex-pole transition gap — and we measure their
gains on a $3 \times 2 \times 2$ matrix of architecture (RWKV-6,
Mamba-2, Linear Attention) $\times$ mode (causal, LION
bidirectional) $\times$ scale (7M, 14M parameters) on a probe-grade
ASR pair (LibriSpeech, Common Voice) plus a length-stratified
synthetic axis-2 separator (MQAR at $T \in \{64, 256, 1024\}$). The
empirical pattern across the matrix obeys a single predictive law —
*mechanism-prior alignment*: a mechanism's gain is recoverable from
how much of its targeted deficit the backbone's native primitives
leave uncovered. The contribution is the **transfer-pattern
matrix** plus the law that explains it, not any single CER number.

## 2. Why this frame wins

The thesis is positioned as a **characterisation study**, not a
proposal study. The artifact is the matrix; the mechanisms are the
demonstrations of the framework, not the framework's purpose. Three
consequences follow:

1. **ASR is a probe, not the goal.** The thesis is not "we made
   speech recognition better." It is "we used speech recognition as
   a clean axis-1 instrument for the same reason TIMIT was used for
   phoneme-level studies and ImageNet for vision feature work."
   Treating the dataset as a *measurement instrument* removes the
   most common reviewer attack ("this is ASR-specific"); it also
   removes the obligation to compete with Conformer / Wav2Vec /
   Whisper on absolute CER, because absolute CER is not the claim.

2. **MQAR carries the second axis.** The MQAR cohort at
   $T \in \{64, 256, 1024\}$ is the cleanest available
   axis-2 separator: synthetic, length-scalable, deterministic,
   designed by an independent group (Arora et al. Zoology) for the
   exact mechanism-class differential we test. Its presence in the
   matrix transforms axis 2 from a citation-grade axis ("LUCID is
   said to help with recall") into a measured axis with an
   independently-defined success criterion.

3. **Common Voice carries cross-distribution validation.** The
   same task and metric on a different acoustic distribution
   (speakers, accents, recording conditions) makes any single-CER
   claim falsifiable: a mechanism that helps on LibriSpeech but
   harms on Common Voice would falsify the corresponding axis
   alignment. The CV pilot has 16 cells already on disk; we report
   the $\Delta$ vs LibriSpeech $\Delta$ for every cell, not just
   absolute numbers.

The frame in one sentence: **we are not proposing three speech
recognisers; we are proposing a predictive law about mechanism
transfer in a structural family, and demonstrating it on the
cleanest available probes for the axes that family exercises.**

## 3. Standard reviewer objections this frame defuses

| Objection | Defence under the probe-as-instrument frame |
|---|---|
| "These results are ASR-specific." | The probe is selected *because* it cleanly exercises axis 1. Cross-domain validation of axis 1 is in the literature (5 independent papers, 4 modalities — Vision-RWKV, LiT, AudioRWKV, Paper 7, ours). Axis 2 is exercised independently by MQAR. CV provides cross-distribution within the same axis. |
| "Why not test on a real LM benchmark?" | Aggregate LM benchmarks confound multiple axes and add autoregressive evaluation noise that cannot be separated from mechanism attribution at single-seed budgets. The thesis prioritises clean attribution at the architectural level; LM evaluation is a future-work direction. |
| "You claim three mechanisms but the gain on each is small (3–6σ)." | The claim is the **pattern**, not any single Δ. Three mechanisms × six architecture-mode cells × two scales × two datasets produce ≈70 statistical tests; the deficit-proportional ordering reproduces across all of them. Single-seed individual cells are evidence; the matrix is the artifact. |
| "Why doesn't your framework cover state-tracking?" | State-tracking lives on a complexity-class boundary (NC¹ / PNC¹) that diagonal-class architectures formally cannot cross (Merrill, Petty, Sabharwal 2024; Sarrof, Veitsman, Hahn 2024). Targeting it requires a transition-class mechanism (DeltaNet, DeltaProduct, RWKV-7), which is out of scope for our diagonal-family matrix. The formal results function as scope clarifier, not as theorems we attempt to violate. |
| "How is this different from `Mechanism X` paper?" | We do not propose architecture-specific kernels; we propose three mechanism classes deployed on three different transition primitives, and we measure transfer across that boundary. Each mechanism's individual lineage is honestly cited in the proposed-solution chapter. |

## 4. The three mechanisms — final names and provenance

The mechanism names are the canonical writeup names. They replace
the codebase identifiers (`multidil_v2`, `lucid`, `rse_strong_viscosity`)
in the writeup; the codebase keeps its identifiers for replay.

### Multi-Scale Depthwise Convolution (MSDC) — codebase: `multidil_v2`

Parallel bank of depthwise dilated 1D convolutions on the input,
mixed by learnable per-layer weights:

$$x_\text{mixed}[t] = \sum_{d \in \{1, 2, 4, 8\}} \alpha_d \cdot \mathrm{DWConv}_{k=3, d}(x)[t]$$

**Target axis:** 1 (short-range temporal hierarchy).

**Provenance honesty.** The mechanism class is independently
rediscovered across the linear-time-RNN literature on different
modalities (Q-Shift in Vision-RWKV, depthwise-conv-on-V in LiT,
multi-dilation $\{1, 2, 4, 8\}$ in Paper 7 / Kiruluta et al., 2D
DWSep ConvShift in AudioRWKV). The thesis instantiates the
multi-dilation form on 1D speech across three transition primitives.
Position: *one node in a five-paper cross-domain story*. Strongest
empirically supported claim in the thesis.

### Chunked Value Decorrelation (CVD) — codebase: `lucid` family

Within each chunk of size $T_c$, we compute a key-similarity
preconditioner and decorrelate values:

$$P_c = \exp\!\big(\tau_h \cdot (G_c / \sqrt{d} - \sqrt{d})\big) + \varepsilon I, \qquad G_c = K_c^{RN} (K_c^{RN})^\top$$

$$\tilde V_c = P_c^{-1} V_c, \quad \text{then feed } \tilde V_c \text{ into the unchanged scan}$$

**Target axis:** 2 (associative memory under interference).

**Provenance honesty (CRITICAL — DO NOT WRITE "ADAPTATION").**
LUCID (Duvvuri et al. 2026, arXiv:2602.10410) is a *softmax
attention* mechanism. Their problem is attention noise in
long-context retrieval inside a quadratic-complexity Transformer.
They compute $\mathrm{softmax}(QK^\top/\sqrt{d}) \cdot P^{-1} V$ and
preserve $O(N^2 d)$. Their target task domain is multi-needle long-
context retrieval (MNIAH, BABILong, LongBench).

**CVD is a different mechanism on a different substrate that is
related to LUCID by logic, not by adaptation:**

- LUCID operates on the full $T \times T$ softmax-attention matrix
  inside a Transformer. CVD operates on chunk-local $T_c \times T_c$
  systems inside the recurrence of a linear-time RNN; the full
  $T \times T$ attention matrix is never materialised.
- LUCID retains $O(N^2 d)$. CVD retains the linear-time backbone's
  $O(T \cdot T_c \cdot d)$ envelope at $T_c \ll T$.
- LUCID's deployment context is a softmax attention path. CVD's
  deployment context spans three different transition primitives:
  (i) chunked recurrent for RWKV-6 with the WKV scan, (ii) parallel
  for LION's bidirectional T×T, (iii) SSD-chunk-local for Mamba-2,
  using the SSD dual form's intrinsic chunk-local structure.

**The shared logical move** between LUCID and CVD is the use of a
key-similarity preconditioner to decorrelate values before they
enter the memory operation, motivated by the same observation that
correlated keys cause interference at readout in any
associative-memory implementation. The thesis cites LUCID as
related work that motivates the *principle*, and contributes:
(a) the formulation of value decorrelation on linear-time-RNN
substrates as a class, (b) three architecture-specific deployments
of that class, and (c) the empirical demonstration that the
mechanism transfers asymmetrically — productive on decay-bounded
backbones, harmful on no-decay LION-LIT.

### Damped Harmonic Oscillator (DHO) — codebase: `rse_strong_viscosity` / `rse_depth_viscosity`

Block-complex transition replacing the real diagonal:

$$G_{t,b} = e^{-\lambda_\text{eff}} \cdot R(\theta_{t,b}), \qquad \lambda_\text{eff} = \lambda_\text{raw} + \eta_{h,b} \cdot \theta_{t,b}^2$$

with $R(\theta) \in SO(2)$ a 2×2 rotation, $\theta$ data-dependent
via rank-48 LoRA, and $\eta \cdot \theta^2$ a Rayleigh dissipation
term that damps high-frequency rotations.

**Target axis:** 1 sub-axis (damped-oscillator dynamics).

**Provenance honesty.** Internally derived. The Lie-group-extension
argument (real-diagonal $(\mathbb{R}_+)^K$ → block-complex
$SO(2)^{K/2} \times (\mathbb{R}_+)^{K/2}$ as the natural smallest
extension that can represent formant-class dynamics) and the
Rayleigh-dissipation coupling (a standard physical prior from
damped-oscillator mechanics) are the thesis's contribution. We do
not adapt a prior mechanism; we invoke a function-class extension
plus a physical prior. The negative-bound paper (Merrill et al.
2024 *Illusion of State* + Sarrof et al. 2024 + the diagonal-SSM
state-tracking impossibility result) bracket the formal limits of
where this extension can and cannot reach.

**Two variants** under the same name in the writeup:
- `rse_strong_viscosity` — uniform $\theta$-clip at $\pi/2$.
- `rse_depth_viscosity` — depth-graded $\theta$-clip
  (L0–L1: $\pi/8$, L2–L3: $\pi/4$, L4–L5: $\pi/2$).
The depth-graded variant is preferred on RWKV-6 causal and on the
LION mode of all three architectures; the uniform variant remains
the reference for the cross-architecture causal comparison. The
choice is reported in the chapter where each cell is presented.

## 5. Architecture × mode matrix

| Architecture | Causal mode | LION mode |
|---|---|---|
| RWKV-6 | per-channel data-dependent (WKV) | bidirectional, parallel T×T (Afzal et al. 2025) |
| Mamba-2 | selective Δt-modulated continuous | LION-S (per-head selective σ-decay) |
| Linear Attention | pure accumulator (no decay) | LION-LIT (no decay) **and** LION-S (with σ-decay) — see §6 |

LION provides a single unified bidirectional wrapper extended to
all three causal architectures. The choice of decay parameterisation
in LION mode is a load-bearing methodological decision (see §6).

## 6. LION-LIT vs LION-S — the decay-as-prerequisite finding

Per Afzal et al. 2025 Table 1, the natural mapping from a causal
backbone to its LION form preserves the causal backbone's decay
class. RWKV-6 → LION-S (per-channel data-dependent decay survives);
Mamba-2 → LION-S (per-head selective σ-decay derived from Δt·A);
Linear Attention → LION-LIT (no decay, $\lambda = 1$). LION-LIT is
the "natural" LA-LION variant under the table.

Empirically, LION-LIT vanilla on LA produces test CER 0.295 — far
worse than causal LA (0.188). Adding CVD on top of LION-LIT
*worsens* the result by Δ +0.024. This is the only negative LUCID-
class transfer in the matrix, and it has a clean structural
explanation: LUCID's unit-diagonal preconditioner is built to
correct interference under decay-bounded value accumulation. Without
decay, $\phi(K)^\top$ row-sums grow with $T$, the preconditioner
becomes over-aggressive, and the mechanism inverts in sign.

This is the **decay-as-prerequisite finding**: in the bidirectional
mode, decay is not just a backbone parameter but a structural
prerequisite for the value-decorrelation mechanism to bite. We
support this claim by adding LION-S as a controlled variant on LA
(per-head $\sigma$-decay imported from Afzal et al.'s Gated RFA →
LION-S row), and showing that:

| LA LION variant | vanilla | + MSDC | + CVD | + DHO | + CVD × MSDC | + DHO × MSDC |
|---|---:|---:|---:|---:|---:|---:|
| LION-LIT (no decay) | 0.295 | 0.140 | **0.319 (+0.024 worse)** | **0.104** | — | **0.0961 (lowest LA cell)** |
| LION-S (with σ-decay) | 0.138 | 0.115 | 0.131 | **0.0988 (lowest LA LION-S)** | 0.113 | — |

The LION-LIT vs LION-S divergence is therefore not an
implementation artefact — it is a measured natural experiment that
isolates the causal role of decay in mechanism transfer. CVD on
LION-LIT is harmful, CVD on LION-S converts: decay is the structural
prerequisite. DHO converts on both LION-LIT (BREAK Δ −0.191) and
LION-S (Δ −0.039) because the complex-pole block-SO(2) transition
introduces decay through its own damping channel rather than relying
on the wrapper's. We present this in the LION chapter as the
cleanest structural finding of the bidirectional sweep.

This frames the choice in the writeup: we report **both** LION-LIT
and LION-S for LA, with LION-LIT carried as the falsification
control and LION-S as the primary LA-LION cell. This is honest
about the paper-default choice (LION-LIT per Afzal Table 1), uses
the negative result as evidence rather than as an embarrassment,
and produces a sharper claim than either variant alone.

## 7. Empirical headlines

The LibriSpeech matrix (50-ep matched, single seed, 7M and 14M)
plus the MQAR cohort plus the CV pilot together support these
headline claims, in priority order:

1. **MSDC universal axis-1 with deficit-proportional ordering.**
   LA (Δ −0.047) > RWKV-6 (Δ −0.026) > Mamba-2 (Δ −0.021) at 7M,
   reproducing on 14M. The ordering matches how much native
   axis-1 coverage each backbone already had (LA: none; Mamba-2:
   k=4 DWConv covers part).

2. **DHO BREAK on LA, NULL on Mamba-2 (LibriSpeech), modulated on
   Common Voice.** On LibriSpeech, LA × DHO Δ −0.068 (BREAK) and
   Mamba-2 × DHO Δ −0.0002 (clean predicted NULL — Δt absorbs the
   real-diagonal extension). On Common Voice, the *same* Mamba-2 ×
   DHO-depth cell delivers Δ −0.022 (helpful). The cross-dataset
   divergence is not a contradiction — it is evidence that
   architectural absorption is *task-prior modulated*: Δt absorbs
   DHO's contribution under the LibriSpeech distribution but not
   under Common Voice's broader speaker/accent distribution. This
   becomes a thesis-defining nuance in the synthesis chapter.

3. **CVD asymmetric transfer + decay-as-prerequisite.** Causal:
   productive on RWKV-6 (Δ −0.004), Mamba-2 (Δ −0.008), LA
   (Δ −0.017). LION: marginal on RWKV-6 / Mamba-2, *negative* on
   LION-LIT, *productive* on LION-S. The LION-LIT falsification
   isolates decay as the structural prerequisite (§6).

4. **MQAR T=1024 — causal Transformer FAIL, our linear-time +
   mechanism PASS.** At $T=1024$, $K=256$ multi-query recall, the
   causal Transformer baseline fails to converge (best per-seq
   acc 0.0 after 26k steps, loss flat near initialisation). Every
   linear-time architecture *with either MSDC or CVD* solves it
   in 1k–5k steps. This is the strongest single empirical claim
   in the thesis: it inverts the standard "softmax wins on recall"
   narrative at the length where the comparison becomes binding.
   DHO does not solve MQAR (axis-1 mechanism on an axis-2 task —
   the predicted null), confirming the axis decomposition.

5. **Same-axis composition saturation (P8).** LUCID × DHO ×
   MSDC composition saturates at the matched 50-ep budget,
   matching MSDC alone within noise on RWKV-6 (P7 0.079 vs MSDC
   0.079). This is the cross-experiment-invariant fingerprint:
   beyond a single mechanism per axis, additional same-axis
   freedom does not convert without prior alignment.

6. **Engaged-null catalog (Stages 2–11 supplementary evidence).**
   Seven mechanisms (Delta rule, Cayley orthogonal, M²RNN,
   non-normal RSE, content-conditional $\alpha_d$, readout gauge
   A1′, novelty gate) activate under SGD — diagnostics show
   non-zero parameter mobility — without moving CER, because they
   target axes (2, 3, 5) that ASR's structural prior does not
   exercise. These nulls are not failures; they are scope-evidence
   for the axis decomposition.

## 8. Why ASR specifically — methodology rationale

The probe-as-instrument frame requires an explicit defence of why
ASR was chosen and what the choice costs. The defence has four
elements, presented in the methodology chapter:

1. **Clean, reproducible probe.** Short post-subsampling sequences
   ($T \le 500$), greedy CTC decoding (no autoregressive sampling
   noise), single primary metric (CER), zero-config dataset
   (LibriSpeech HuggingFace). Single-seed 7M / 50-ep runs complete
   in ~2 GPU-h, which is what made the Stages 2–10 mechanism
   discovery chain feasible.

2. **Sharp axis-1 instrument.** Speech has well-defined acoustic
   structure at the phoneme-to-syllable scale (40–160 ms after
   2× subsampling = 4–16 frames). MSDC's dilation set $\{1, 2, 4, 8\}$
   was chosen *because* it covers exactly that range. DHO's
   complex-pole extension matches formant-class dynamics, also
   axis-1 sub-structure. This is the axis ASR is sharpest on; it
   is also the axis the diagonal-class function family can
   express in principle.

3. **Principled scope cost.** ASR cannot characterise axes 2–5 by
   construction. The thesis is explicit about this: every axis-2/3/5
   "engaged null" from Stages 2–10 is not a mechanism failure but
   an axis-mismatch — which is *itself* evidence for the axis
   decomposition. The cost is real and we own it; we do not
   pretend ASR characterises more than it does.

4. **Cross-domain backstop.** The strongest axis-1 claim
   (MSDC-class mechanisms restore short-range mixing in linear-time
   RNNs) is independently confirmed by five papers across four
   modalities. ASR is not the load-bearing observation for axis 1;
   it is the cleanest of five concordant observations. That places
   the thesis as one node in a literature pattern, not as a
   single-domain claim.

The cost of this framing is that we explicitly do not claim SOTA
on LibriSpeech and we openly acknowledge that the mechanism
selection was driven by what ASR can resolve. We accept the cost
because it is what makes the matrix-as-artifact claim defensible.

## 9. Why MQAR specifically — the second probe

MQAR carries the axis-2 evidence by construction. Four reasons:

1. **It exercises axis 2 by design.** Multi-query associative
   recall stresses associative-memory capacity under interference.
   When stored pairs $N$ approach state dimension $d$, an
   architecture that cannot decorrelate similar keys collides in
   its state matrix and recall accuracy drops. This is exactly
   the deficit CVD targets.

2. **It validates CVD where ASR cannot.** CVD's gain on ASR is
   small because ASR doesn't push the state-saturation regime.
   MQAR gives CVD its natural test bed. The MQAR $T=1024$ result
   (causal Transformer fails, linear-time + CVD passes) is the
   sharpest single mechanism-attribution claim in the thesis.

3. **It is the standard axis-2 benchmark in the field.** Based,
   GLA, DeltaNet, BlackMamba, and RWKV-7 all report MQAR numbers
   because the Zoology paper established it as the canonical
   synthetic recall probe for efficient architectures. Reporting
   MQAR places our results in the standard comparison frame the
   linear-time-RNN community already uses.

4. **It is fast and length-scalable.** Synthetic data, unlimited
   examples, minutes per length. The full sweep across three
   lengths × ten backbones is roughly 6 GPU-hours, less than 4 %
   of the total compute budget.

## 10. Why Common Voice — cross-distribution validation

Common Voice plays a third role: cross-distribution validation
within axis 1. Same task, same metric, different acoustic
distribution. Two roles in the writeup:

1. **Falsifier of mechanism gains.** A mechanism that helps on
   LibriSpeech but harms or stays flat on Common Voice fails the
   cross-distribution test for that mechanism. Most cells pass;
   reporting Δ vs LibriSpeech Δ for every cell makes the test
   visible.

2. **Reveals task-prior modulation.** The Mamba-2 × DHO-depth
   cell is the cleanest example: NULL on LibriSpeech, helpful on
   Common Voice. This is not a contradiction with the
   "Δt absorbs DHO" prediction — it is a refinement of it. The
   absorption is not absolute; it depends on what the data
   distribution actually exercises. CV's broader speaker / accent /
   recording variability evidently leaves enough residual
   axis-1-sub-structure for DHO's complex-pole extension to bite,
   while LibriSpeech's narrower audiobook distribution does not.
   This is reported in the synthesis chapter as a refinement of
   the mechanism-prior alignment law: alignment is between
   mechanism and *task-as-exercised-by-data*, not between
   mechanism and *architecture in isolation*.

## 11. Structural exclusions and their honest framing

### Axis 3 — state-tracking, out of scope

State-tracking lives on the formal NC¹ / PNC¹ complexity-class
boundary. Two theorems bracket the axis cleanly:

- *Negative bound:* Diagonal SSMs (and complex-diagonal extensions
  including DHO) cannot track non-Abelian groups at finite
  precision regardless of width. Sarrof, Veitsman, Hahn 2024 +
  the formal-language analysis of single-layer DCD-class SSMs
  used in Merrill, Petty, Sabharwal 2024.
- *Positive bound:* Diagonal-plus-low-rank (DPLR) transitions
  reach $\mathsf{PNC}^1$-completeness. DeltaNet, DeltaProduct,
  RWKV-7 sit in this class.

None of our three mechanisms cross the boundary, by construction.
We *use* the bound as a scope clarifier in §2.5 of the writeup,
and we defer mechanisms that target axis 3 to future work.

### Axes 4 and 5 — long-range and content-adaptive, not led on

Long-range information flow ($T \gg 500$) and content-adaptive
feature-map / channel-side computation are well-populated areas
of the literature (Log-Linear Attention, RWKV-X, HSA, NaLaFormer,
Avey, PoM, Mongaras-Larson Expressiveness, Higher-order Linear
Attention). We cover them in §2.6 of the writeup as part of the
broader axis-decomposition map; the family-D / axis-5 cluster
saturates to a narrow performance band on tasks that do not
reward dense per-channel freedom, which is consistent with our
ASR closed-cells engaged-null evidence.

We do not run these mechanisms on our matrix; their natural homes
are different probes. We retain them in the related-work map and
do not pretend to characterise them.

## 12. Cross-experiment invariant

Across Stages 2–11 of the discovery phase plus the locked 50-ep
matrix, **dense per-token freedom does not convert into measured
gain without prior alignment**. We interpret seven engaged-null
mechanisms (Delta rule T1, NCGRU-Cayley, M²RNN-sparse, non-normal
RSE T2 / S9A / S9B, content-conditional $\alpha_d$ CB-3, readout
gauge A1′, novelty gate) and the family-D quadratic-lift
saturation cluster (qtail, hadamard, PoM, ChannelMix bypass) as
mechanisms that activate under SGD but do not move CER because
ASR's task structure does not reward the axis they extend.

This invariant is a load-bearing claim in the synthesis chapter:
the thesis frames the *positive* mechanism transfer (MSDC, CVD,
DHO) as confirmations of mechanism-prior alignment in one
direction, and the *negative* engaged-null catalogue as
confirmations of the converse. The two together produce a
predictive law, not a one-sided observation.

## 13. Style guidance for the writeup

Maintain across all chapters:

1. **Three mechanism names — Multi-Scale Depthwise Convolution,
   Chunked Value Decorrelation, Damped Harmonic Oscillator —
   capitalised consistently.** Use abbreviated form (MSDC, CVD,
   DHO) only after the first introduction.
2. **Never say "we adapt LUCID."** Always either: "related to
   LUCID by logic," "inspired by the value-decorrelation principle
   of LUCID," or "the same logical move as LUCID applied on a
   different substrate." LUCID is a softmax-attention mechanism;
   our CVD is a linear-time-RNN mechanism that shares its
   key-similarity decorrelation principle.
3. **ASR is "the probe," not "the task."** Single sentence per
   chapter introduction reminds the reader the dataset is an
   instrument.
4. **Pattern over numbers.** Every empirical claim is presented
   as part of the matrix; individual cells are evidence, the
   matrix is the artifact.
5. **Honest provenance per mechanism.** MSDC inherits a
   cross-domain pattern (we are one node); CVD shares a logical
   move with LUCID (we are not adapting); DHO is internally
   derived (we own this).
6. **Engaged-null is not a failure.** Phrase consistently as
   "axis-mismatch evidence" or "scope evidence for the axis
   decomposition."
7. **LION-LIT is a controlled falsifier on LA, not the primary
   reported variant.** LION-S is the primary LA-LION cell; LION-LIT
   is presented in §6-equivalent of the LION chapter as the
   negative control that isolates decay as prerequisite.

## 14. Risks to address before submission

| Risk | Mitigation |
|---|---|
| Single-seed across most cells | Methods chapter frames matrix-as-artifact; phenomenological claim ≈70 statistical tests in single coherent direction. Multi-seed reserved for BREAK-band cells. Reviewer concedes or asks for a specific re-run. |
| 30-ep → 50-ep narrative inversions in `Mechanisms_Overview.md` | Update `Mechanisms_Overview.md` to 50-ep numbers before draft sent to advisor. The CVD-on-LA inversion (30-ep "anomaly" → 50-ep largest) is most pressing. |
| RWKV-6 vanilla NEG scaling at 14M (0.105 → 0.110) | Section 5 paragraph: under matched 50-ep budget, RWKV-6's per-channel data-dep WKV is undertrained at depth 12; Mamba-2's selective Δt extracts more value from the same compute. Architecture-specific compute-efficiency note, not a fundamental property. |
| Mechanism pre-registration burden | Each chapter that claims "predicted null / predicted BREAK" cites the date / commit / document where the prediction was made. `EXPRESSIVITY_AXES.md` axis-mechanism table provides most of the substrate. |

## 15. HLA paper — verdict

Higher-order Linear Attention (Lin et al. 2025, arXiv:2510.27258)
is a polynomial-feature-lift mechanism in the same Family-D /
axis-5 function class as PoM, NaLaFormer, the Mongaras-Larson
Expressiveness paper, and the Avey bypass. Its contribution is
algorithmic (linear-time training scheme for higher-order
streaming statistics), not expressivity-driven. It does not target
axes 1, 2, or 3. It does not evaluate on MQAR or other axis-2
benchmarks.

**Decision.** Cite once in §2.6 as part of the family-D / axis-5
cluster. Do not expand. It strengthens the
"family-D-saturation" claim by adding a 2025-10 instance to the
same axis-5 / value-lift function class.

**Suggested bibliography entry (not yet committed; metadata verified
against the arXiv abstract page on 2026-04-29):**

```bibtex
@misc{zhang2025hla,
    title = {Higher-order Linear Attention},
    author = {Yifan Zhang and Zhen Qin and Quanquan Gu},
    year = {2025},
    eprint = {2510.27258},
    archiveprefix = {arXiv},
    primaryclass = {cs.LG},
    doi = {10.48550/arXiv.2510.27258},
    url = {https://arxiv.org/abs/2510.27258},
    note = {Streaming higher-order tensor interactions via prefix sufficient statistics; algorithmic contribution, polynomial value-lift family}
}
```

## 16. What the writeup chapters look like under this frame

| Chapter | Anchor | Probe-as-instrument framing role |
|---|---|---|
| 1. Introduction | Mechanism-prior alignment as predictive law | Probe choice rationale + matrix-as-artifact thesis |
| 2. Related Work | 8 sections: linear-time alternatives, bidirectional adaptation, cross-modality short-range, associative memory, **state-tracking bounds (axis 3 scope)**, **long-range and content-adaptive (axes 4-5 map)**, **evaluation probes**, **reference implementations** | §2.7 carries the probe defence; §2.5 carries the axis-3 scope clarifier |
| 3. Theoretical Background | Axis decomposition + math of three mechanisms + complexity-class brackets | Defines axes; positions mechanisms on axes; establishes formal limits |
| 4. Proposed Solution | MSDC, CVD, DHO derivations + LION wrapper + decay-as-prerequisite | LION-LIT vs LION-S as controlled natural experiment |
| 5. Experiments and Results | The matrix as a single artifact: ASR (LibriSpeech, CV), MQAR | Each subsection presents one row of the matrix; `Δ vs deficit` plot is the chapter's headline figure |
| 6. Conclusions | Mechanism-prior alignment law as the take-home; cross-experiment invariant; future-work axis 3 | Frame as predictive epistemology, not as performance claim |

---

*End thesis positioning v1 (2026-04-29).*
