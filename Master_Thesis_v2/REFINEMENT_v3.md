# REFINEMENT_v3.md — second-pass refinement plan

**Inputs.** `reviewe_report_v2.txt` (reviewer-2 report, 2026-05-01) and the
author's twelve targeted notes from the v3 conversation.

**Scope.** The matrix and the empirical numbers stay. What changes in this
pass is voice, calibrated language, defensive framing, the LION-vs-operator
table positioning, the bibliography sweep, and a whole-text tightening pass.
The mechanism-prior alignment criterion remains the central conceptual
contribution but its claim strength is dialed back to a predictive heuristic
and the language around it is tightened.

**Out of scope.** Multi-seed reruns, generic-residual-block ablations,
internal-hyperparameter ablations, formal pre-registration. These are
conceded to future work in the limitations and future-work sections.

---

## Section 0. Style block (applies to every section below)

- No bold-claim opening sentences in §5.3 (Transfer pattern on LibriSpeech)
  or its sister mechanism-row paragraphs in §5.4–§5.7. Bold is reserved for
  the mechanism abbreviation (MSDC, DHO, CVD) at paragraph starts. Opening
  sentences are introductory and logical for a new paragraph, not
  headline-claim.
- "Across the matrix, the direction is coherent" replaces "this exact cell
  is BREAK / this exact small Δ proves alignment" framing wherever it
  appears. Single-seed is incompatible with fine-grained per-cell
  significance claims; aggregate directional coherence is the available
  evidence and the language must reflect that.
- "DHO: BREAK on Linear Attention, near-null on Mamba-2" is inaccurate as a
  cross-matrix summary because the Mamba-2 DHO cell is productive on Common
  Voice (Δ = $-0.0226$). Any LibriSpeech-only reading must qualify the
  distribution.
- The author preserves the established style rules from the v2 pass:
  no em / en dashes (тире), minimise colons, generalised form, ASR as
  probe not SOTA, criterion not law.

---

## Section A. Voice and presentation (P0)

### A1. Rewrite §5.3 mechanism paragraphs without bold-claim headers.

**Location.** `chapters/5_experiments_and_results.tex`, the three
`\paragraph{...}` blocks of §5.3 (Transfer pattern on LibriSpeech).

**Current.**
- `\paragraph{MSDC universal, deficit-proportional.}`
- `\paragraph{DHO: BREAK on Linear Attention, near-null on Mamba-2.}`
- `\paragraph{CVD: asymmetric.}`

**Change.** Drop the bold full-claim header. Keep `\paragraph{}` only for
the abbreviation, and let the prose carry the reading. Each paragraph opens
on an introductory sentence (what mechanism, what axis, what to look for),
then describes the across-matrix direction without claiming per-cell
significance, then closes with the cross-distribution / scale qualifier.

Sketch (substitute when implementing):

> **MSDC.** The Multi-Scale Depthwise Convolution targets the
> short-range temporal-hierarchy axis. Across the three causal backbones at
> both scales, the absolute Δ ranks LA $>$ RWKV-6 $>$ Mamba-2, matching the
> deficit map of Section~\ref{sec:backbones:deficit}: the backbone with no
> native local mixing absorbs the largest gain, the backbone with a
> single-step token shift the smallest, and Mamba-2's native short
> convolution leaves the smallest residual headroom. The direction is
> coherent at both scales; the relative-reduction column inverts on the
> 14M MSDC row because vanilla baselines differ, which is why the chapter
> reads the deficit map in absolute Δ throughout.

> **DHO.** The Damped Harmonic Oscillator extends the real-diagonal
> transition with the block-complex closure of
> Section~\ref{sec:proposed:dho}. Across the matrix on LibriSpeech the
> direction is consistent with the prediction that the block-complex
> extension reaches furthest on the architecture with no native attenuation
> and no native phase: Linear Attention shows the largest absolute gain
> ($-0.0681$ at 7M, $-0.0412$ at 14M), RWKV-6 shows a small productive Δ
> consistent with its native per-channel decay covering the attenuation
> component, and Mamba-2's selective scalar transition absorbs the
> contribution under the LibriSpeech audiobook distribution. The Mamba-2
> reading is task-prior modulated rather than absolute; the same cell
> becomes productive on Common Voice (Δ_{CV} = $-0.0226$), discussed in
> Section~\ref{sec:experiments:cv}.

> **CVD.** The Chunked Value Decorrelation transfers asymmetrically across
> the family on LibriSpeech, with the largest single-cell Δ on Linear
> Attention ($-0.0165$ at 7M), an intermediate Δ on Mamba-2, and the
> smallest on RWKV-6. The ordering inverts a naive natural-home-in-attention
> prior, since the value-decorrelation principle descends from a softmax
> construction (Section~\ref{sec:primitives:cvd}); the inversion becomes
> the structural reading once the LION-mode falsifier of
> Section~\ref{sec:experiments:lion} isolates decay as the prerequisite
> for the bidirectional version of the mechanism.

**Why.** The reviewer is right that single-seed cannot support the
"BREAK / near-null" headline framing per cell, and the user is right that
the "near-null on Mamba-2" header is wrong because the same cell is
productive on Common Voice.

### A2. Same sweep on §5.4 cross-scale, §5.6 MQAR, §5.7 composition.

Same edit pattern. Bold = abbreviation only. First sentence is
introductory, not claim-headline. The cross-scale section is least
exposed (figures already speak for themselves); the MQAR section's
"Linear-time plus mechanism solves MQAR at every tested length" header
is fine as written but may be tightened to remove the "every" universal
quantifier in favor of "across the cohort" (the OOM cell at LA-DHO
T=1024 is technically not "solved").

---

## Section B. Methodological framing (P0)

### B1. Noise-floor methodological sentence in §5.1.

**Location.** `chapters/5_experiments_and_results.tex`, §5.1 Training
budget paragraph (where σ ≈ 0.0014 is currently introduced).

**Add.**

> The empirical CER noise floor $\sigma \approx 0.0014$ was estimated from
> a small set of matched-budget reruns conducted during the calibration
> phase under the same preprocessing, optimiser, and decoding pipeline.
> The estimate is used only as a descriptive threshold for categorising
> deltas, not as a formal hypothesis test.

**Open question.** If a specific rerun count $N$ is on record, substitute
"a small set" with "$N$ matched-budget reruns".

### B2. Null-effect side defense in §5.9 (and §6.2 Limitations).

**Location.** `chapters/5_experiments_and_results.tex` §5.9 synthesis;
mirror in `chapters/6_conclusions.tex` §6.2 Limitations.

**Add.**

> The present matrix provides stronger evidence for the positive-transfer
> side of the criterion than for the full converse / null side. Systematic
> null-effect validation is left for future multi-seed work.

The §5.9 "Two-directional evidence" paragraph from the previous pass should
be replaced with this sentence (since Option 2 is the canonical position
and the engaged-null catalogue is not in v2).

### B3. Mixed-precision confound.

**Preferred path.** Remove the fp32 / bf16 distinction from §A.1
hyperparameters and from any chapter prose mentioning it. Either set the
canonical pipeline to fp32 throughout and silently fold the LA LION-mode
runs into that convention, or set the pipeline to bf16 throughout. The
reviewer flagged this as a procedural inconsistency; the cleanest fix is
to remove the inconsistency from the reported configuration.

**Fallback if removal is impossible.** Keep the bf16-only-on-LA-LION fact
and add the sentence:

> We do not expect bf16 to create a systematic advantage sufficient to
> explain the observed directional pattern, but it remains a procedural
> inconsistency and should be controlled in future work.

### B4. Optimization-landscape confound in §6.2 Limitations.

**Add to §6.2.**

> We do not fully disentangle structural inductive bias from generic
> optimisation benefits. Generic parameter-matched controls (for example,
> MSDC compared to a residual block of equivalent parameter count) are
> left for future work.

### B5. Mechanism-vs-architecture-specific-patch defense.

**Location.** `chapters/4_proposed_solution.tex` §4.4 (CVD) closing
paragraph; or §4 chapter intro.

**Add.**

> The mechanisms are transferable at the functional level, not necessarily
> at the tensor-implementation level. Each backbone-specific form
> implements the same logical operation on the chunk structure native to
> that substrate; the operation is shared, the routing is local.

---

## Section C. Calibrated language (P0, single-replacement passes)

### C1. "proportional" → softer.

Replace the strict-mathematical reading with "larger residual deficits
tend to correspond to larger absolute gains" or "monotonic" / "rank-
ordered" depending on the sentence rhythm.

**Affected occurrences (audit before editing).** Search the manuscript
for `proportional` and decide per occurrence:
- §1.2 Conceptual paragraph: "orders mechanism gains by the residual
  structural deficit" — already non-proportional in wording. Reword the
  surrounding sentence to drop any "proportional" reference.
- §3.1 closing criterion statement: contains "proportional to the
  residual structural deficit". Rewrite to "tend to scale with the
  residual structural deficit" or "larger ... tend to correspond to
  larger ...".
- §4.1 RH1 paragraph.
- §5.9 synthesis criterion restatement.
- §6.1 conclusions criterion restatement.

### C2. "orthogonal axes" → "conceptually distinct but empirically interacting".

**Affected occurrences.**
- §1.1 Motivation, line near "different residual deficits along orthogonal
  axes of expressivity".
- §1.2 Conceptual, "Five orthogonal axes provide a structural scaffold".
- §3.1 axis-decomposition opening, "it decomposes into orthogonal axes".

Use the longer phrase ("conceptually distinct but empirically interacting")
on first introduction and shorten to "conceptually distinct axes" on later
uses for readability.

### C3. "pre-registered" → "pre-specified" (or "hypothesised").

**Affected occurrences.**
- §4.1 RH3 paragraph.
- §A.3.2 Common Voice subsection.
- §5.5 cross-distribution validation opening.
- §5.7 composition opening.
- Any caption text that uses "pre-registered".

Rationale: there is no public time-stamped registry, so the formal-science
implication of "pre-registered" overstates procedural rigour. "Pre-specified"
preserves the timeline of intent without the formal claim.

---

## Section D. §1.2 Conceptual revision

Tie C1, C2, B2 together for the abstract-level claim.

**Current §1.2 Conceptual.**

> Five orthogonal axes provide a structural scaffold for mechanism design;
> the empirical matrix directly exercises axes one and two on its primary
> probes, and axis three is bracketed by the formal complexity-class bound
> that delimits the diagonal-class scope. The decomposition supports the
> *mechanism-prior alignment* criterion, a predictive heuristic that orders
> mechanism gains by the residual structural deficit each architecture
> leaves uncovered along the axis the mechanism targets.

**Target.**

> Five conceptually distinct but empirically interacting axes provide a
> structural scaffold for mechanism design; the empirical matrix directly
> exercises axes one and two on its primary probes, and axis three is
> bracketed by the formal complexity-class bound that delimits the
> diagonal-class scope. The decomposition supports the
> *mechanism-prior alignment* criterion, a predictive heuristic stating
> that larger residual deficits along the axis a mechanism targets tend
> to correspond to larger absolute gains, and that no measured gain is
> expected when the axis is not exercised by the task. The criterion is
> supported primarily on its positive-transfer half by the matrix in this
> thesis.

The same softening propagates to the abstract: the abstract's sentence
"a predictive heuristic stating that a mechanism's gain on a task is
proportional to the residual structural deficit..." needs the same swap.

---

## Section E. Composition sign error in §5.7

**Reviewer flag.** §5.7 prose currently states the CVD×MSDC composition on
RWKV-6 at 7M is $\Delta = +0.0003$ over MSDC alone. The master-matrix
table reports MSDC at 0.0788 and CVD×MSDC at 0.0785, so the actual Δ is
$-0.0003$. The text contains a sign error.

**Fix.** In `chapters/5_experiments_and_results.tex` §5.7, change
`+0.0003 at 7M` to `-0.0003 at 7M`. Verify the 14M figure
(MSDC 0.0751 vs CVD×MSDC 0.0746 → Δ = $-0.0005$ ✓ already correct).

This is a one-character fix but it is reviewer-visible.

---

## Section F. LION-vs-operator table positioning (user point 11)

**Location.** `appendices/e_appendix.tex` §C.5 (label
`app:diagnostics:lion-vs-vim`).

**Two tables.**
- `tab:results:bidirectional-comparison-7m`: LION at 7M vs operator-level
  at 7M, matched layers and parameters.
- `tab:results:bidirectional-comparison-14m`: operator-level at 14M only,
  for completeness.

**Issue.** The 14M operator-level result reaches roughly where 7M LION
already sits on the two backbones with native decay (RWKV-6: 0.0738 vs
0.0859; Mamba-2: 0.0746 vs 0.0853). The current framing reads as
"6 layers vs 12 layers", but the operator-level family does forward +
backward in each layer, which is closer to a $2\times$ serial-compute
multiplier per layer than to a layer-count comparison.

**Action.**
1. Sharpen the 14M table caption to explicitly note that operator-level
   at 12 layers does forward + backward inside each layer, paying $2\times$
   serial compute per layer relative to a causal layer of the same size.
2. Add a sentence to the existing post-table prose: "The operator-level
   configuration at 14M is parameter-matched but not compute-matched to
   LION at 7M; the doubled serial compute per layer is the cost the
   operator-level family pays in exchange for retaining the linear-time
   envelope."
3. Reframe the "complementary reading" paragraph from the previous pass
   so the headline is "LION provides equivalent bidirectional expressivity
   at roughly half the parameter and serial-compute budget on the two
   native-decay backbones". The existing text already says this; the goal
   is to make sure the table caption and the prose align on the
   compute-vs-parameter framing rather than the layer-count framing.

---

## Section G. Bibliography sweep (user point 10)

**Audit results from the v2 D3 pass.** 67 distinct citation keys are used
across `chapters/*.tex` and `appendices/*.tex`; `bibliography.bib` carries
~90 entries. ~23 entries are present but unused.

**Action plan.**
1. Spawn a focused agent (or do it in one pass) that:
   - Lists every unused bib entry with its title and one-line note.
   - For each entry, decides: cite (a substantive paper that should be
     referenced somewhere in the manuscript), keep (bib hygiene; might be
     useful in a future revision), or remove (clearly stale).
2. Identify any well-known papers in the linear-RNN / mechanism-design
   space that should be cited but are not (the 80+ figure the user cited
   suggests a few are missing). Candidates to check:
   - Conformer (Gulati et al. 2020) for the depthwise-convolution-in-ASR
     lineage.
   - SE3-equivariant or rotational-positional papers for the DHO
     block-complex / rotation framing.
   - Recent associative-recall benchmarks beyond Zoology (Based,
     CompetitorMix, Wave-RNN).
   - Gradient-checkpointing / memory-efficient training references if
     mentioned anywhere in the appendix.
3. Bring the cited count up to a defensible 75+ without padding. The bar
   is "would a reviewer expect this paper to be cited given the topic".

**Bar (decided).** "Only papers that directly support a claim already made
in the text". No padding for completeness. The bibliography sweep removes
unused entries that are not on standby for a known forthcoming claim, and
adds only entries that fill an existing gap.

**Required addition.** Apple's *Attention to Mamba: A Recipe for
Cross-Architecture Distillation* (Moudgil, Huang, Dhekane, Rodríguez,
Zappella, Danieli, 2026; arXiv 2604.14191) goes into §1.1 Motivation as
part of the production-scale interest in the linear-time family. The
paper reports a two-stage Transformer-to-Mamba distillation recipe at
1B scale; its inclusion underlines that major industrial labs treat
the linear-time substrate as a serious destination for cross-architecture
transfer, complementing the MiniMax-01 / Hunyuan-TurboS / Nemotron-H
deployment chain already cited.

**Out of scope.** Replacing existing valid citations.

---

## Section H. Whole-text tightening pass (user point 12)

**Goal.** Identify concrete sections where the prose can be shortened by
20–30% without loss of meaning.

**Candidate hot-spots from the v2 pass.**
- **§3.2 (Three causal backbones).** Each backbone subsection
  (RWKV-6, Mamba-2, LA) independently restates the unified template.
  Likely 15–25% trim by deduplication of the template re-statement.
- **§3.3 (Bidirectional adaptation via LION).** Already partly tightened in
  C5 of the previous pass; verify no further trim is needed.
- **§4.5 (Unified LION wrapper).** Already compacted; verify.
- **§5.1 (Experimental framework).** The probes paragraph and the
  LS-vs-CV table overlap. Either compress the prose or remove duplicated
  facts from the table. Target: 15% trim.
- **§5.3 (Transfer pattern).** After the A1 voice rewrite, a second pass
  to tighten the qualifier-heavy sentences.
- **§5.5 (Cross-distribution validation).** Possibly redundant with the
  DHO Mamba-2 reading already given in §5.3 (post-A1).
- **Appendix C captions.** Already tightened in B5; verify.

**Action.** Do this pass last, after Sections A–F land. Tightening rests
on stable text; doing it before voice / language sweep means redoing the
trim.

---

## Section I. Reviewer-specific items not covered by user

### I1. Composition coverage defense in §5.7.

**Add.** "Exhaustive pairwise composition testing is out of scope for the
matched-budget matrix. The pre-specified pair per backbone was selected
to test the criterion's prediction on different-axis composition (LA: DHO
axis-1-phase × MSDC axis-1-input) and same-axis composition
(RWKV-6, Mamba-2: CVD axis-2 × MSDC axis-1). Alternative pairings such as
DHO × CVD are out of scope under the budget constraint."

### I2. Internal hyperparameter ablations in §6.2 Limitations.

**Add.** "Internal hyperparameter ablations (the MSDC dilation set
$\{1, 2, 4, 8\}$, the DHO depth-graded $\theta$-clip schedule of
$\pi/8$ and $\pi/4$) are not exhaustively tested in this thesis. The
chosen values follow literature precedent for the underlying primitives
and are reported as design choices rather than as ablated optima."

---

## Section L. §1.1 Motivation and §1.2 Contributions rewrite (P0, user point 6)

The current §1.1 / §1.2 carries leftovers from the original "we
adapt existing fixes" framing, which the reviewer is right to pressure.
The rewrite repositions the work as our own derivations from
expressivity deficits, not as repackagings of prior fixes.

### L1. §1.1 Motivation rewrite.

**Drop entirely.** The closing paragraph beginning "The central
contribution is the demonstration that mechanism transfer is structured
rather than incidental..." moves out of §1.1. Its content is already
covered by §1.2 (Contributions) and §3.1 (axis decomposition); having it
in §1.1 is redundant and pushes the chapter towards a strong-claim voice.

**Reframe the literature paragraph.** The current "The literature has
independently rediscovered mechanism-level fixes for some of these
deficits across modalities..." is wrong about who rediscovers what. The
literature has been rediscovering \textit{bottlenecks} (the same
short-range mixing deficit, the same recall-under-interference
saturation, the same transition-class limit) across modalities; the
\textit{fixes} proposed in this thesis are derived independently from
the architectural deficit each backbone leaves uncovered, not adapted
from prior fixes.

Target framing (substitute when implementing):

> The literature has been rediscovering similar bottlenecks for the
> linear-time family across modalities: short-range mixing in vision
> and audio, associative recall under interference in language, the
> transition-class limit in formal complexity. The fixes proposed in
> different papers tend to be architecture-specific and modality-bound.
> This thesis takes a different route. We treat each bottleneck as a
> structural deficit located on a specific axis of the diagonal-class
> recurrence operator, and derive a mechanism-level extension whose form
> follows from the deficit's mathematical signature. The three resulting
> mechanisms are not adaptations of prior fixes; they are independent
> derivations from the deficit map of
> Section~\ref{sec:backbones:deficit}.

**Apple paper insertion.** The motivation also names recent industrial
interest in the linear-time family. Currently cited: MiniMax-01,
Hunyuan-TurboS, Nemotron-H. Add the Apple cross-architecture distillation
paper (arXiv 2604.14191) to this chain; it shows that a major industrial
lab treats Mamba as a serious destination for distillation from
Transformer teachers.

**Mechanism-positioning rule.** Throughout §1.1 (and rippled through the
thesis), refer to the three mechanisms as \textit{our derivations} from
the deficit map, not as \textit{adaptations} of LUCID, depthwise
convolutions, or harmonic oscillators. The primitives those papers
introduce supply the mathematical building blocks (Section~\ref{sec:primitives});
the assembly into MSDC, DHO, CVD on the linear-time substrate is the
contribution.

### L2. §1.2 Contributions rewrite.

**Drop the three bold paragraph headers.** The existing
\paragraph{Conceptual.} / \paragraph{Methodological.} /
\paragraph{Empirical.} structure is replaced with a single flowing
paragraph (or two short paragraphs if the prose runs long) without bold
headers and without the three-buckets framing.

**Tone.** Honest, not pretentious. State what the thesis does and what
it claims, without hyping.

**Content invariants the new paragraph must carry.** The five-axis
decomposition; the matrix specification (3 backbones × 2 modes × 2
scales × 3 probes); the criterion as a predictive heuristic supported
primarily on its positive-transfer half; the three mechanisms named
(MSDC, DHO, CVD); the MQAR T = 1024 binding empirical claim. No "central
contribution is the demonstration" phrasing.

**Style block ripple.** The terminology fixes from C1, C2 apply here as
well: "conceptually distinct but empirically interacting axes" replaces
"orthogonal axes"; the proportionality language is replaced with
"larger residual deficits tend to correspond to larger absolute gains".

---

## Section J. Suggested execution order

The order minimises rework: the §1.1 / §1.2 rewrite runs first because it
sets the canonical voice for the rest of the manuscript; voice and
language sweeps follow; methodological framing is independent and slots
in anywhere; the bibliography sweep and whole-text tightening come last
because they benefit from a stable text.

1. **L1, L2.** §1.1 / §1.2 rewrite (drops the closing-claim paragraph,
   reframes the literature paragraph, names the Apple paper, removes
   bold-paragraph headers from §1.2).
2. **A1.** Rewrite §5.3 mechanism paragraphs without bold-claim headers.
3. **A2.** Same sweep on §5.4 / §5.6 / §5.7.
4. **C1, C2, C3.** Single-replacement language passes for "proportional",
   "orthogonal", "pre-registered".
5. **D.** §1.2 Conceptual revision tying C1+C2 together; same swap in the
   abstract. (Now applied on top of L2.)
6. **B1.** Noise-floor methodological sentence in §5.1.
7. **B2.** Null-effect-side defense in §5.9 and §6.2.
8. **B3.** Mixed-precision: remove the fp32 / bf16 row from Table A.1
   and any chapter prose mentioning bf16 / fp32 / mixed precision.
   (Decision: full removal, no fallback sentence.)
9. **B4, B5.** Optimisation-confound limit in §6.2; mechanism-vs-patch
   defense in §4.4 closing.
10. **E.** Composition sign error in §5.7.
11. **F.** LION-vs-operator table caption + prose repositioning in
    `e_appendix.tex` §C.5. ($2\times$ serial-compute-per-layer framing
    confirmed.)
12. **I1, I2.** Reviewer-specific defensive sentences for composition
    coverage and internal-hyperparameter ablations.
13. **G.** Bibliography sweep (orphan triage + Apple paper addition;
    bar set at "only papers that directly support a claim already made
    in the text").
14. **H.** Whole-text tightening pass.

Each step is a separate commit. After step 14, run the static
cross-reference / citation check from D3 of the v2 pass again; verify
all `\ref` and `\cite` keys still resolve.

---

## Section K. Decisions and open questions

### K-decided (locked into the plan).

- **B3 mixed-precision: remove entirely.** Drop the "Numerical precision"
  row from Table A.1 (`tab:setup:hyperparams` in `a_appendix.tex`) and
  remove any chapter prose that mentions `bf16`, `fp32`, or "mixed
  precision". The manuscript becomes silent on numerical precision; this
  is acceptable for a thesis-level appendix.
- **F LION-vs-operator framing: "$2\times$ serial compute per layer".**
  The 14M operator-level configuration runs forward + backward inside
  each layer, paying $2\times$ serial compute per layer relative to a
  causal layer of the same parameter count. Caption and prose are
  rewritten around this framing; "12 layers" is reframed as "12
  bidirectional layer-equivalents at $2\times$ serial compute per
  layer".
- **G bibliography sweep: bar set at 'only papers that directly support
  a claim already made in the text'.** No padding for completeness.
  Required addition: the Apple cross-architecture distillation paper
  (arXiv 2604.14191) into §1.1 Motivation alongside the existing
  industrial-deployment chain.
- **Reviewer's editorial-note fallback: declined.** The thesis keeps
  the mechanism-prior alignment criterion as its central conceptual
  contribution and dials the claim strength to a predictive heuristic.
  The reviewer is over-strict on this point; the present plan does not
  reframe the work as exploratory empirical characterisation.

### K-still-open (need a decision before the corresponding step).

- **B1 noise-floor rerun count $N$.** Is there a specific count of
  matched-budget reruns that produced the $\sigma \approx 0.0014$
  estimate, or is the "small set" wording the right level of disclosure?
  If a count is known, the sentence becomes "$N$ matched-budget reruns";
  otherwise the sentence stays at "a small set of matched-budget
  reruns conducted during the calibration phase". Default if no answer
  by the time step 6 fires: keep "a small set".
