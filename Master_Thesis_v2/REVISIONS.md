# Thesis v2 Revision Plan

This document is the working plan for `Master_Thesis_v2`. It is a
fork of the thesis content from `Master_Thesis/` carried into the
restructuring pass. The plan combines the reviewer concerns from
`reviewer_report.txt` (in the parent directory), the framing
decisions in `Thesis_Positioning.md`, the prior `CRITIQUE.md`, and
the additional revisions agreed during the v2 planning session
on 2026-05-01.

## Style rules for v2

These rules apply to all v2 writing.

1. No dashes in prose. Use commas, semicolons, parentheses, or
   sentence rephrasing. Hyphens in compound words are fine
   (linear-time, axis-1, depth-graded).
2. Minimise colons. Avoid colon-built sentences where possible.
   Prefer two short sentences over one long sentence joined by a
   colon.
3. Concise. Avoid overcomplication. Sentences fit one idea each.
4. Generalised final form, not discovery narrative. No Stages
   2-10 history. No rejected variants.
5. Probe-as-instrument framing. ASR is a probe for axis 1, not a
   SOTA target. MQAR is the axis-2 probe. Common Voice is a
   pre-experimental probe for cross-distribution validation.
6. CVD shares logical motivation with LUCID. CVD is not an
   adaptation of LUCID. The substrate, complexity envelope, and
   parameterisation differ.
7. Three canonical mechanism names are MSDC, DHO, CVD. Codebase
   identifiers (multidil_v2, lucid, rse_*) appear nowhere in
   prose.
8. DHO single-variant. The depth-graded variant is canonical.
   Strong-viscosity is not introduced as a separate variant in
   prose.
9. LION-LIT is a controlled falsifier on Linear Attention. LION-S
   is the primary LA-LION configuration. Decay-as-prerequisite is
   the structural finding.
10. Engaged-null framing where applicable, axis-mismatch evidence,
    not failure.

## Priority levels

P0 critical, must land for v2 to be coherent. P1 important,
recommended. P2 polish, time permitting.

---

## Section A. Epistemological reframe (reviewer-driven, P0)

Reviewer concerns from `reviewer_report.txt`. Each item closes
one falsifiability or methodology attack vector.

### A1. Replace "alignment law" with "alignment criterion" (P0)

The word *law* overpromises. Chapter 1 already uses *predictive
criterion*. Seven occurrences remain in chapters 3, 4, 5. Replace
*alignment law* with *alignment criterion* (or *alignment
framework* when the noun refers to the wider scaffold). Targeted
locations are §3.1 closing paragraph, §4.1 (three uses), §5
chapter framing line 8, §5.2 closing paragraph, §5.5 closing
paragraph. The synthesis section title `\section{Synthesis:
mechanism-prior alignment}` already drops the noun and stays as
is.

### A2. Position Common Voice as a pre-experimental hypothesis (P0)

The strongest reviewer attack is that *task-as-exercised-by-data*
is inferred post-hoc from the Mamba-2 $\times$ DHO success on
Common Voice. The defence anchors the broader acoustic
variability of Common Voice in literature as a pre-experimental
fact, then frames the experimental result as confirmation of an
independently established expectation rather than as a
post-hoc rescue. Two edits.

In Apendix A §A.3.2 (Common Voice English 100h), insert two to
three sentences citing the original Common Voice
paper~\cite{ardila2019commonvoice} on speaker, accent, and
recording-condition diversity, and reframe the dataset selection
as deliberately chosen for higher acoustic variance prior to any
DHO experiment.

In Chapter 5 §5.5 (Cross-distribution validation), insert one
sentence at the start noting that the broader acoustic
distribution of Common Voice was a pre-registered probe property
documented in the dataset paper, not an inference from the
Mamba-2 $\times$ DHO result.

### A3. Strengthen the single-seed defence (P0)

Add one paragraph to Chapter 5 §5.1 (Experimental framework,
training budget) and one to Chapter 6 §6.2 (Limitations) stating
that the matched 50-epoch budget across 65 cells exhausts the
available compute, and that the across-matrix directional
coherence provides aggregate statistical confidence that
effectively offsets the variance of any individual cell. Frame
this in the same way that a single correlation across many
configurations is more robust than a single high-precision
measurement on one configuration.

### A4. Falsification protocol as future work (P0)

Reviewer notes that the framework has no pre-experimental metric
for what a dataset exercises. The honest response is to
acknowledge this and propose a falsification protocol as the
next research step. Add to Chapter 6 §6.3 a paragraph stating
that a stricter falsifiability protocol would specify a
numerical prediction for each (architecture, mechanism, dataset)
triple, derived from independently quantified dataset-level
metrics that measure structural axes, and frame the development
of such metrics as the natural next research step. The thesis
does not develop them.

### A5. Probe-as-instrument paragraph in §1.1 (P0)

Insert a short paragraph at the end of Chapter 1 §1.1 (Motivation)
stating what the thesis is not claiming. Do not claim
state-of-the-art on LibriSpeech or Common Voice. Do not propose
three speech recognisers. Do not assert the alignment criterion
holds outside the diagonal-class function family. Speech
recognition is a probe for axis 1, not the application domain of
the contribution.

### A6. Promote the MQAR T = 1024 result in the abstract (P0)

The reviewer praises the MQAR T = 1024 result as the strongest
single empirical claim. The current contributions paragraph
mentions it in the closing sentence. The abstract is still
placeholder template text. When the abstract is written, the
MQAR T = 1024 result is the empirical headline alongside the
alignment criterion claim.

---

## Section B. Structural reorganisation (P0)

User-driven reframe. Aligns the thesis with its own title and
positioning.

### B1. Reframe Motivation and Contributions around the title (P0)

The title is *On Expressiveness and Mechanism Design of Linear
Recurrent Models*. The current Chapter 1 leans on industrial
deployment (MiniMax-01, Hunyuan-TurboS, Nemotron-H, Jamba,
Zamba2, Evo 2) and treats the alignment criterion as the primary
contribution. The original motivation was finding common
bottlenecks across the linear-time recurrent family and
demonstrating the transferability of solutions. The alignment
criterion is a derived analytical framework that explains why
transfer works.

Rewrite §1.1 around two questions, the expressiveness question
(what each backbone leaves uncovered along orthogonal axes) and
the mechanism design question (whether mechanism-level fixes
transfer across the family). Lead with these. Keep the
industrial framing to two sentences supporting the structural
failure modes argument, no more.

Rewrite §1.2 Contributions around three contributions. First,
the conceptual contribution, the axis decomposition of
expressivity for diagonal-class linear-time RNNs and the
alignment criterion as predictive heuristic. Second, the
methodological contribution, the cross-architecture transfer
matrix at matched compute. Third, the empirical contribution,
three transferable mechanisms with the binding empirical claim
that linear-time-plus-mechanism backbones solve MQAR at
$T = 1024$ where the causal Transformer fails. Keep mechanism
names brief, do not enumerate matrix dimensions in the
contributions paragraph.

### B2. Move Research Questions to §4.1 (P0)

Research Questions belong with the problem formulation, not
with the broad motivation. Remove §1.3 from Chapter 1 entirely.
Insert in Chapter 4 §4.1 a structured block with research goal,
research questions, research hypotheses, and goals.

The proposed structure for the §4.1 block is

- **RG (research goal)**, the characterisation of
  expressivity-deficit-driven transferability of mechanism-level
  extensions on diagonal-class linear-time RNNs.
- **RQ1 to RQ4** as currently in §1.3, kept verbatim where the
  framing is still correct.
- **RH1 to RH3** as pre-experimental hypotheses, each with a
  falsifiability marker. The cross-distribution hypothesis in
  particular reads roughly as follows. We hypothesise that
  mechanism gains tested on a single audiobook-source
  distribution under-represent the alignment pattern. A more
  diverse acoustic distribution (Common Voice English) should
  expose mechanism-task interactions that LibriSpeech's narrower
  distribution masks. We pre-register this expectation. If the
  alignment criterion holds, the per-cell $\Delta$ ordering
  should preserve under the broader distribution, with
  selectively absorbed mechanisms (the Mamba-2 $\times$ DHO
  cell under $\Delta t$-absorption is the canonical example)
  becoming productive when the data exercises residual axis-1
  sub-structure unavailable on LibriSpeech.
- **Goals (G1 to G3)** as deliverables. Three mechanisms
  formalised, transfer matrix populated, alignment criterion
  stated and tested.

### B3. Tighten chapter introductions (P1)

Each chapter intro currently lists every subsection with a clause
of its own. The convention is two to three sentences naming what
the chapter does and why. The longest offenders are §3 and §4.
Rewrite the §3 and §4 intros to one short paragraph each.

### B4. Section header simplification (P1)

Selective renames where the title is wordy. The list below
preserves any title that is already short.

| Current | Proposed |
|---|---|
| §3.2 Linear-time recurrence: unified formulation and three backbones | §3.2 Three causal backbones |
| §3.3 Bidirectional adaptation: the parallel form and decay taxonomy | §3.3 Bidirectional adaptation via LION |
| §3.5 Formal expressivity bounds and the diagonal-class scope | §3.5 Formal expressivity bounds |
| §5.3 Cross-architecture transfer pattern on LibriSpeech | §5.3 Transfer pattern on LibriSpeech |
| §5.6 Axis-2 isolation: MQAR length sweep | §5.6 Axis-2 isolation via MQAR |

Other titles stay. Avoid colons in section titles where
practical. Synthesis section title (§5.9) keeps the colon since
it carries semantic role.

### B5. Tighten table captions (P1)

Captions across Chapter 5 and the appendices currently combine
table description, verdict semantics, footnote notes, and
reading guidance. Captions move to two or three sentences
maximum, structural description only. Reading guidance moves to
the prose around the table. Concrete offenders are Table 5.1
(aggregated, twelve lines), Table 5.2 (MQAR cohort, fourteen
lines), and the master matrix tables in Apendix B (renumbered C).

---

## Section C. Content additions (P1)

These items add new content rather than restructure existing.

### C1. LibriSpeech versus Common Voice statistics (P1)

Pre-experimental quantification of the distributional difference.
Two implementations.

Light implementation. Add to §5.1.1 a small table with five or
six rows comparing the two probes on speaker count, accent
diversity (where reported), recording conditions, total hours,
mean utterance duration, vocabulary size. Cite Common Voice and
LibriSpeech papers.

Optional figure in Apendix A. Speaker-count distribution
histogram or audio-duration histogram. Lightweight, does not
require new experiments.

This anchors RH3 in pre-experimental evidence.

### C2. Detailed data and evaluation framework (P1)

Add specifics that the current §5.1 and §A.3 omit.

For Common Voice. Selection protocol explicit. How were
speakers sampled from the validated split. Total source size of
the validated split. Random seed for selection (already in
§A.3.2 manifest hash). Inclusion criteria.

For MQAR. Causal Transformer baseline architecture. Layer count.
Parameter count matched to the linear-time backbones in the
cohort. Dimensionality of the head, FFN. Learning rate and
optimiser. Step budget. These define the fair-comparison
interpretation of the T = 1024 FAIL.

For evaluation framework. Already explained at high level in
§5.1.2. Add one sentence noting that the absence of
language-model rescoring and beam search is deliberate, to
prevent decoder-architecture-bias confounds with the encoder
mechanism attribution.

Long detail moves to Apendix A. High-level rationale stays in
§5.1.

### C3. Linear-time alternatives overview table in §2.1 (P1)

Currently §2.1 surveys around 15 architectures in pure prose. Add
an overview table with five or six categories, four or five
architectures per row, key mechanism, transition class, decay
character. The proposed structure is

| Category | Examples | Mechanism | Transition class |
|---|---|---|---|
| Kernel-feature linear attention | Performer, Linformer, Reformer, Random Feature Attention, Gated RFA | Kernel approximation of softmax | Trivial |
| Recurrent-decay (modern RNNs) | RWKV-4 to 7, RetNet | Bounded recurrent state with decay | Diagonal |
| Selective SSMs | S4, S5, Mamba, Mamba-2, Mamba-3 | Selective state-space duality | Scalar-identity diagonal |
| Diagonal-plus-low-rank | DeltaNet, DeltaProduct, Gated DeltaNet, RWKV-7 | Rank-one Householder updates | Out of diagonal class |
| Long-convolutional | Hyena | Implicit kernels | Convolutional |
| Hybrid / gated | GLA | Linear attention with gating | Diagonal with gating |

Citations to verify or add to bibliography are Performer, Linformer,
Reformer, Random Feature Attention~\cite{peng2021rfa}, RetNet, GLA,
DeltaNet, DeltaProduct~\cite{deltaproduct2025}, Gated DeltaNet.
DeltaProduct is recent and likely missing.

### C4. DeltaProduct citation and expressivity definition (P0)

DeltaProduct positions itself as increasing the expressivity of
DeltaNet. Their definition of *expressivity* is formal, what
classes of state automata an architecture can track, not
benchmark performance.

Add to §3.1 (Axis decomposition) a clarifying sentence at the
opening, *In this thesis, expressivity refers to the formal
sequence-transformation capacity of the recurrent state-update
operator, following the formal-expressivity tradition of Sarrof
et al. and the architectural-capacity framing of Yang et al. on
DeltaNet and DeltaProduct, rather than aggregate benchmark
performance.*

Add DeltaProduct to §3.5 (Bounds) as a DPLR-class extension and
to §6.3 (Future work) as a natural axis-3 extension target.
Verify DeltaProduct is in `bibliography.bib`, add if missing.

### C5. LION wrapper compaction in §4.5 (P1)

§4.5 is currently around 95 lines of prose describing the
decay-class correspondence. The LION paper~\cite{afzal2025linear}
Table 1 shows the same content compactly. Reduce §4.5 prose by
about half. Replace the prose-describing-LION-variants paragraph
with one small table summarising the three $\lambda_k$
formulations (LION-LIT $\lambda_k = 1$, LION-D learnable scalar,
LION-S sigmoid data-dependent), citing Afzal et al. 2025 Table 1.
Keep the existing Table 4.5 mapping our backbones to LION
variants but simplify to three columns, backbone, causal $A_t$
structure, inherited LION variant.

### C6. LION versus operator-level reasoning made explicit (P1)

The Apendix C §C.5 LION versus operator-level table compares
LION and BiWKV/VIM at matched 7M and at 14M operator-only. The
intended reading is not in the prose. Add one paragraph stating
that LION matches operator-level performance at matched 7M
(matched layers, matched parameters) on the two backbones with
native decay, and that operator-level catches up at 14M (twelve
layers, doubled parameters), indicating that the mechanism-level
adaptation provides equivalent expressivity at half the compute
when the underlying causal mechanism carries native decay.

### C7. Verify DHO decay self-sufficiency note in §4.3 (P0)

`CRITIQUE.md` records that a fourth design note for DHO should
be in §4.3 (DHO design notes), stating that the per-block
damping factor $e^{-\lambda_{\mathrm{eff}}}$ makes the
block-complex transition decay-bounded by construction
independent of any external mask. This sets up the §5.4
LION-LIT $\times$ DHO BREAK reading without surprise. Verify
the note exists in the v2 §4.3 (it should, per recent commits
on `thesis`). If absent, add two to three sentences.

### C8. Resolve 7M Mamba-2 $\times$ DHO WER asymmetry (P1)

The 7M Mamba-2 $\times$ DHO row carries a CER override (0.1006,
$\Delta - 0.003$) that reflects the canonical depth-graded run.
The corresponding WER (0.3172) is from the on-disk
strong-viscosity run. Either rerun the depth-graded cell to
obtain a matched WER or add a footnote on this row stating the
source asymmetry.

### C9. Closed-cell engaged-null catalogue decision (P1)

The engaged-null catalogue (formerly Apendix D) was removed in
the previous pass. The catalogue carried converse-direction
evidence for the alignment criterion, mechanisms that engage
under SGD without converting into measured CER reduction.
Without it, the criterion is supported only by
positive-transfer evidence on the 50-epoch matrix. Two options.

Restore a one-page summary as an Apendix B subsection. The
summary frames each engaged-null entry as axis-mismatch evidence
consistent with the criterion. This sharpens falsifiability.

Or drop the converse-direction claim entirely. Replace it with
a single sentence in §5.10 acknowledging that the matrix
evidence is one-directional, and noting the
discovery-phase archive as the source of the converse evidence.

The first option is stronger. The decision is open.

### C10. Noise floor explicit (P1)

The single-seed defence gains force if anchored to a numerical
noise floor between matched configurations. Add one sentence to
§5.1 (Training budget and parameter match) stating the empirical
noise floor at the codebase configuration ($\sigma \approx
0.0014$ on test CER per the discovery phase, if that estimate
survives), and use this to frame the BREAK band ($|\Delta| > 5
\sigma$) versus the marginal band ($|\Delta| \le 2 \sigma$)
consistently across §5.2 to §5.7.

### C11. Composition signature softening in §5.7 (P1)

§5.7 currently claims the composition row populates the
alignment criterion. The empirical evidence is that composition
gain over the strongest single mechanism is small or zero on
most cells, with the LA $\times$ DHO $\times$ MSDC composition
($\Delta - 0.0410$ over MSDC alone at 7M causal) as the strongest
counter-example. Soften the signature claim to *the composition
row provides one productive different-axis stack consistent with
the criterion, with the remaining compositions saturating to
single-mechanism level.*

---

## Section D. Mechanical edits (P1)

### D1. Typographic fixes from reviewer copyediting

- p. iii Quotation page, *so can expand* to *so it can expand*.
- p. iv Declaration, *If you was supported* to *If you were
  supported*. *thank to your donor* to *thank your donor*.
- p. x to xii Abbreviations, Constants, Symbols pages,
  *Keep if if you have* to *Keep if you have* (three
  occurrences).
- p. 1 §1.1, *lay ers* to *layers*.
- p. 7 §2.4, *Based of Arora* to *Based on Arora*.
- p. 11 §3.5, *diagonal-plus-low- rank lift* to
  *diagonal-plus-low-rank lift*.
- p. 15 §3.2.4, *interference- cancellation* to
  *interference-cancellation*.
- p. 31 §4.5 Table 4.5, *back bone* to *backbone*.

### D2. Frontmatter (P0)

`master-thesis-template.tex` carries placeholder content that
must be replaced for v2.

- Title `Thesis Title` to actual title.
- Author `John Smith` to Anastasiia Mazur.
- Supervisor `Dr. James Smith` to actual supervisor.
- Quotation page, replace the Dave Barry quote or remove the
  page entirely.
- Abstract, 250 to 300 words. Names the three mechanisms, the
  matrix, the alignment criterion as a predictive heuristic, and
  the MQAR T = 1024 headline result.
- Acknowledgements, replace placeholder prose. Optional, can be
  removed.
- Abbreviations table, replace example *LAH* with the
  thesis-specific list (MSDC, CVD, DHO, MQAR, LION, CER, WER,
  CTC, SSM, RWKV).
- Constants table, remove. Irrelevant to the thesis.
- Symbols table, remove or replace with the few thesis symbols
  ($T$ sequence length, $d$ state dimension, $T_c$ chunk size).
- Dedication, remove or fill in. Optional.

### D3. Final compile pass (P0)

After substantive revisions land.

- Verify all `\ref{}` cross-references resolve. No question
  marks in the rendered PDF.
- Verify all `\cite{}` citations resolve against
  `bibliography.bib`.
- Verify table of contents, list of figures, list of tables.
- Verify figure paths resolve from the master file.
- Verify appendix lettering matches the included files (a, c,
  e are canonical Apendix A, B, C in v2; b is the alternative
  with the consolidated longtable, kept for reference and not
  included in the master template by default).

---

## Section E. Out of scope, defended decisions (P0 confirmation)

These items are deliberately not changed in v2.

### E1. Multi-seed validation

Computational scope of the matrix at matched compute does not
allow multi-seed runs. Single-seed methodology stays.
Multi-seed validation of the BREAK-band cells is named as
future work in §6.3.

### E2. State-tracking probes

Out of scope by formal complexity-class boundary. The diagonal
class formally cannot recognise non-Abelian permutation
languages. The diagonal-plus-low-rank lift is the natural
extension and is named as future work in §6.3 with
DeltaProduct as the canonical reference.

### E3. SOTA comparisons against Conformer, Wav2Vec, Whisper

Out of scope by the probe-as-instrument framing. The thesis is
not positioning ASR as the application domain. Item A5 above
adds an explicit *what we do not claim* paragraph in §1.1.
No SOTA tables.

### E4. Cross-modality replication of the matrix on vision or text

Out of scope at the thesis budget. Named as future work in
§6.3.

### E5. Theoretical Background before Related Work

The conventional order is Related Work then Theoretical
Background. The axis decomposition of §3.1 builds on
formal-expressivity literature surveyed in §2.5. Reordering
would deprive the reader of the context required to read §3.1.
Order stays.

---

## Section F. Suggested execution order

The order below preserves narrative consistency and minimises
cross-chapter rewrites.

1. C7 verify DHO decay self-sufficiency note in §4.3 (cheap,
   blocks nothing).
2. D1 typographic fixes (mechanical, low risk).
3. A1 terminology unification (alignment law to alignment
   criterion) across chapters 3, 4, 5.
4. B1 reframe Motivation and Contributions in §1.1 and §1.2
   around the title.
5. B2 move Research Questions to §4.1 with RG, RQ, RH, Goals
   block.
6. A5 *what we do not claim* paragraph in §1.1.
7. A2 Common Voice variability citation in §A.3.2 and §5.5.
8. A3 plus C10 single-seed defence and noise floor in §5.1
   and §6.2.
9. A4 falsification protocol paragraph in §6.3.
10. C11 composition signature softening in §5.7.
11. C8 resolve 7M Mamba-2 $\times$ DHO WER asymmetry.
12. C5 LION wrapper compaction in §4.5.
13. C6 LION versus operator-level reasoning explicit in
    Apendix C.
14. C4 DeltaProduct citation and expressivity definition in
    §3.1, §3.5, §6.3.
15. C3 linear-time alternatives overview table in §2.1.
16. C1 LibriSpeech versus Common Voice statistics in §5.1 plus
    Apendix A.
17. C2 detailed data and evaluation framework in §5.1 and §A.3.
18. C9 decide on the closed-cell engaged-null catalogue.
19. B3 tighten chapter introductions.
20. B4 section header simplification.
21. B5 table caption tightening.
22. D2 frontmatter (after substantive revisions stabilise).
23. A6 MQAR result in abstract (during D2).
24. D3 final compile pass.

---

End of `REVISIONS.md` for v2 (initial draft, 2026-05-01).
