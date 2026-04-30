# Critique notes on existing chapters and appendices

*Working document. Not part of the LaTeX deliverable. Records the
review decisions taken on 2026-04-30 during the final-week pass and
the remaining items still to address. Edit as items close.*

---

## Chapter 1 (Introduction, 76 lines)

State: motivation and contributions are solid. Risks:

- §1.3 *Structure Of The Thesis* is a heading with no body. Needs
  five or six short paragraphs, one per chapter.
- Contributions paragraph does not name MQAR or Common Voice as
  probes explicitly. The MQAR T = 1024 result is the single most
  binding empirical claim of the thesis and deserves at least a
  sub-clause in contributions.
- Closing sentence of *Motivation* (line 38, "yet they remain
  fragmented") could be sharpened to name the missing comparison
  precisely: prior work has not been compared on a single substrate
  at matched compute, which makes their relative gains difficult to
  attribute to architectural cause.

## Chapter 2 (Related Work, 449 lines, eight sections)

State: most thorough chapter, all eight sections present. Style is
clean: no em-dashes encountered; LUCID is positioned neutrally as a
softmax-attention mechanism without any "adaptation" framing.

Items to address:

- §2.8 *Reference implementations* references the codebase and
  vendor implementations directly. Per the new Code-Upon-Request
  position, the section is removed and the bibliographic citations
  to vendor papers (RWKV, Mamba, Linear Attention) are kept where
  they already appear in §2.1.
- §2.6 closing sentence cites family-D quadratic-lift cluster
  saturating to "a narrow performance band" without forward
  reference for the empirical evidence. Add a forward reference to
  the cross-experiment invariant section in Chapter 5.

## Chapter 3 (Theoretical Background, 813 lines, five sections)

State: strong. §3.1 axes table is the conceptual scaffold. §3.2
unified recurrence template makes the three backbones comparable.
§3.3 LION + decay-as-prerequisite preview is correctly placed.
§3.4 three primitives is split into one subsection per primitive.
§3.5 formal bounds is a compact scope clarifier.

Items to address:

- §3.1 axes table column header "Task / probe": the axis-1 row
  reads "ASR (LibriSpeech, Common Voice)". The probe-as-instrument
  framing is preserved better if the column reads "Probe" only and
  the entry reads "LibriSpeech, Common Voice". One-line edit.
- §3.4.3 (CVD primitive) closing paragraph: the phrase "shared
  logical motivation rather than direct adaptation" is correct and
  is repeated in §4.4.2; keep one of the two slightly fuller, the
  other shorter.

## Open expansion: DHO decay-self-sufficiency

The framing "DHO is decay-self-sufficient" emerged during
Chapter~5 §5.4 as the structural reading of the LION-LIT $\times$
DHO BREAK cell on Linear Attention. The cell remains productive
($\Delta = -0.1909$) on a no-decay LION mask because the
block-complex transition $G_{t, h, b} = e^{-\lambda_{\mathrm{eff}}}
R(\theta)$ supplies decay through its own damping channel rather
than through the wrapper. To avoid presenting this as a new point
in Chapter~5, expand the framing earlier. Two clean placements:

- §3.4.2 *Block-complex transitions and Rayleigh dissipation*:
  add one sentence after equation~(\ref{eq:block-complex})
  noting that the per-block damping factor $e^{-\lambda}$ makes
  the transition decay-bounded by construction, independent of
  any external mask. This becomes load-bearing in the
  bidirectional setting.
- §4.3 *Damped Harmonic Oscillator*, in the design-notes
  subsection: add a fourth design note labelled "decay
  self-sufficiency" that states the property explicitly and
  forward-references Section~\ref{sec:experiments:lion} for the
  empirical evidence. This is the more visible placement since
  the mechanism's design notes is where readers look for
  structural properties before the empirical chapter.

Recommended placement: §4.3 design note, two to three sentences.
The Chapter~3 sentence is optional; §4.3 is the natural home.

## Chapter 4 (Proposed Solution, 682 lines, six sections)

State: math correct, provenance honest. The user observes that
mathematical clarity could be sharpened. See `CH4_MATH_NOTES.md`
for the detailed sharpening plan; the principal items are:

- Make the per-mechanism subsection structure consistent across
  MSDC, DHO, and CVD. Current structure: MSDC has generic-form +
  design-notes + per-architecture; DHO has the same; CVD has only
  generic-form + provenance + per-architecture (no design-notes).
  Add a CVD design-notes subsection covering initialisation of
  $\tau_{h}$ and the $\varepsilon I$ regulariser as formal
  statements rather than mid-equation prose.
- §4.1 problem formulation overlaps with §3.1 axis decomposition.
  Trim §4.1 to about half its current length; let §3.1 carry the
  axis story.
- §4.2 MSDC initialisation paragraph (lines 142–150) carries the
  multiplicative-gradient-trap design rationale in passive voice.
  Keep the formal initialisation values in the chapter; move the
  trap rationale (one paragraph) into the no-effect engagement
  diagnostic in chapter 5 if the diagnostic is the cleaner home,
  or keep here in shortened form.

Per the Code-Upon-Request stance:

- §4 mechanism tables (MSDC, DHO, CVD per-architecture deployment)
  refer only to algebraic insertion points, no codebase names.
  No edits needed.

## Appendix A (Experimental Setup)

State: solid structure. Items to address per the no-repo-refs
position:

- Drop §A.5 *Reproducibility* per-cell artefacts table (lists
  `config.yaml`, `cli_args.txt`, `git_sha.txt`, `best_model.pt`,
  etc., which are repository artefacts).
- Drop §A.6 *Notes on cells excluded* (references repository
  directories explicitly).
- §A.2 ASR baseline architecture: drop the path
  `experiments/formal_v1/src/models/encoder.py`.
- §A.4 Hardware and software: drop the
  `experiments/formal_v1/`, `experiments/synthetics_v1/`,
  and `experiments/final/` working-area paragraph.
- Add a single Code-Availability sentence: "Implementations of the
  proposed mechanisms and the experimental harness are available
  upon request from the author."

## Appendix B (Mechanism Implementation Details)

Removed entirely per the Code-Upon-Request stance. Mathematical
content of value (preconditioner equation, $\theta$-clip schedule,
LoRA rank schedule, viscosity coefficient) lives in chapter 4.

## Appendix C (Complete Empirical Results)

State: master matrix tables, CV cross-distribution, MQAR length
sweep, chunked-streaming. Items to address:

- Drop the *Cell directory* column from every master-matrix
  table. Replace identifier with the (architecture, mechanism)
  pair already present in adjacent columns. Per the
  no-repo-refs position.
- Update the 7M Mamba-2 DHO row to the canonical depth-graded
  values: dev 0.1020, test 0.1006, $\Delta$ test $-0.003$.
- Update the 14M Mamba-2 DHO row to the canonical depth-graded
  values: dev 0.0783, test 0.0781, $\Delta$ test $-0.0046$.
- §C.4 chunked-streaming has three per-architecture tables.
  Compress to one summary table in the appendix; keep the
  per-architecture detail in the supplementary diagnostics
  appendix (former E, now C in the renumbered scheme).

## Appendix D (Closed-Cell Engaged-Null Catalog)

Removed entirely. Per the user's instruction the thesis does not
report what did not work or the discovery sequence. Mechanisms
that did not transfer to the productive matrix are not reported.

## Appendix E (Supplementary Figures, 79 lines)

Restructured. F11 (per-cell training-curve grid) moves to
chapter 5 as a main figure that demonstrates matched-budget
convergence across the matrix. F8 (MSDC $\alpha$), F9 (DHO
$\eta$ and $\theta$), F10 (CVD $\tau$) remain as appendix
diagnostics. F13 (chunked-streaming evaluation) is added as
appendix material. A new LION versus operator-level
bidirectional comparison table is added with parameter and layer
accounting; the table is appendix material with a one-paragraph
discussion in chapter 5.

F12 (engagement classifier dashboard) is dropped. Its
"engaged-without-gain" quadrant carried the closed-cell evidence
that is no longer reported.

## Renumbered appendix order

After deletions and restructuring:

| Old | New | Title |
|-----|-----|-------|
| A   | A   | Experimental Setup |
| C   | B   | Complete Empirical Results |
| E   | C   | Supplementary Diagnostics and Bidirectional Comparison |

## Caption length (not fixing in current pass)

Several table and figure captions across Chapter~5 and the
appendices are long, multi-paragraph descriptions that combine
the table's structural definition with verdict semantics, footnote
notes, and reading guidance. Examples:

- Table~5.1 (aggregated empirical summary, §5.2): caption now
  spans ~12 lines, including the relative-reduction explanation
  and the deficit-proportional-ordering disclaimer.
- Table~5.2 (MQAR cohort, §5.6): caption explains all five verdict
  classes (PASS, PARTIAL, FAIL, SKIP$^{\dagger}$, OOM), the
  asterisk semantic for the Linear Attention vanilla cell, and
  the Causal Transformer FAIL detail.
- Master-matrix tables in Appendix~B (renumbered C): each carries
  several lines of context.
- LION versus operator-level table in Appendix~C (renumbered E):
  ~7 lines of caption.

The current convention is acceptable for a Master's thesis where
captions can be self-contained, but in a paper-length
deliverable these would be substantially trimmed and most of
the verdict-and-footnote prose would move to the surrounding
section text. Recorded here for awareness; not addressed in the
current writing pass.

## Style sweep items

- Confirm no em-dashes anywhere in chapter 1 to 4, appendices A, C,
  E. Hyphens in compound words (linear-time, axis-1, depth-graded)
  are allowed.
- Confirm no occurrences of "adapts LUCID" or "adaptation of LUCID"
  in any chapter. Acceptable phrasings: "shares the logical move",
  "related by logic", "principle drawn from LUCID applied on a
  different substrate".
- Confirm no codebase paths in chapters or appendices. Bibliographic
  citations to vendor papers (RWKV, Mamba, LION) are kept.

## Bibliography

State: 89 entries, key citations present (afzal2025linear,
duvvuri2026lucid, kiruluta2025breaking, arora2023zoology,
xiong2025audiorwkv, sarrof2024expressive, merrill2024illusion,
peng2024eaglefinch, dao2024transformers, gu2023mamba,
katharopoulos2020transformers). No action required during the
final-week pass beyond a quick `\cite{...}` verification before
submission compile.
