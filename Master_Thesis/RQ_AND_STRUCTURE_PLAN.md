# Research Questions and Thesis Structure plan

*Working planning document. Not part of the LaTeX deliverable.
Records the placement and wording proposed for the Research
Questions block and the Structure of the Thesis paragraph in
Chapter~1. Both are written after Chapter~5 lands, since the
phrasing of each RQ depends on which empirical statements
Chapter~5 actually makes.*

*Created 2026-04-30.*

---

## Research Questions

Four questions cover the matrix's argumentative dimensions. They
are stated as questions; Chapter~5 answers each one with a single
empirical statement.

### Proposed wording

**RQ1 (framework).** Can the expressivity of causal linear-time
recurrent architectures be characterised through an axis
decomposition under which a mechanism's gain on a task is
predictable from the residual structural deficit the architecture
leaves uncovered along the axis the mechanism targets?

**RQ2 (causal matrix).** Do the three proposed mechanisms (MSDC,
DHO, CVD) transfer across the three causal backbones (RWKV-6,
Mamba-2, Linear Attention) in a way that orders with the
architecture-deficit map, and do the three exhibit distinct
transfer signatures rather than a single common pattern?

**RQ3 (bidirectional adaptation).** Does the LION mechanism-level
bidirectional wrapper preserve the transfer pattern observed in
the causal mode, and are there structural prerequisites in the
bidirectional setting that are absent in the causal setting?

**RQ4 (cross-task and cross-distribution).** Does the same
mechanism-prior alignment hold when the matrix is exercised on a
different acoustic distribution (Common Voice) and on a
synthetic axis-2 probe (Multi-Query Associative Recall), and what
modulations does the task-as-exercised-by-data introduce?

### Placement options

**Option A: end of Chapter 1.** RQs follow the *Contributions*
section and precede the *Structure Of The Thesis* paragraph.
Standard placement in dissertation conventions; lets the reader
hold the four questions in mind across all subsequent chapters.

**Option B: §4.1 *Problem formulation*.** RQs lift the existing
problem-formulation paragraphs into four numbered questions and
sit immediately before the mechanism descriptions of §4.2 to §4.4.
Lets the math chapter (Chapter~3) develop the framework before
the questions are stated.

Recommendation: **Option A**. Standard placement, and the four
questions are structurally framework-level rather than mechanism-
level: RQ1 is about axis decomposition (Chapter~3 territory),
RQ2 to RQ4 are about empirical transfer (Chapter~5 territory).
None of them is specifically about a single mechanism, so
placing them in the introduction lets all subsequent chapters
read as answers.

---

## Structure of the Thesis paragraph

The Chapter~1 §1.3 *Structure Of The Thesis* section is currently
a heading without a body. The paragraph states what each chapter
does in one sentence and which research question it addresses.

### Proposed wording

The remainder of the thesis is organised as follows.
Chapter~\ref{ch:related} surveys the architectural and mechanism
literature on which the proposed mechanisms build, the formal
expressivity bounds that delimit the diagonal class, and the
evaluation probes used by the empirical chapter; this chapter
positions the thesis within the wider linear-time recurrent
literature. Chapter~\ref{ch:theoretical} establishes the
theoretical foundation, introduces the axis decomposition that
frames RQ1, formalises the three causal linear-time backbones
within a unified recurrence template, presents the LION parallel
form together with the decay-class taxonomy, and presents the
three mathematical primitives on which the proposed mechanisms
build. Chapter~\ref{ch:proposed} introduces the three proposed
mechanisms (Multi-Scale Depthwise Convolution, Damped Harmonic
Oscillator, Chunked Value Decorrelation), the unified LION
wrapper, and the diagnostic probes used at the parameter level.
Chapter~\ref{ch:experiments} reports the empirical matrix,
answers RQ2, RQ3, and RQ4 across LibriSpeech, Common Voice, and
the Multi-Query Associative Recall benchmark, and synthesises
the mechanism-prior alignment law. Chapter~\ref{ch:conclusions}
summarises the thesis's contributions and identifies the
diagonal-to-diagonal-plus-low-rank extension as the natural
next step for the axis-3 question that the present scope does
not address.

---

## Notes on chapter 1 contributions section

The existing *Contributions* paragraph (lines 46 to 74 of
`chapters/1_introduction.tex`) is solid but does not name MQAR
or Common Voice as probes. After the RQs are added, the
Contributions paragraph can be tightened to one sentence on the
mechanisms, one sentence on the matrix and probes (naming MQAR
and Common Voice explicitly), one sentence on the alignment law
as the predictive criterion. The MQAR $T = 1024$ result is the
single most binding empirical claim of the thesis and deserves
naming in the contributions.

### Proposed Contributions paragraph

The thesis introduces three functional mechanisms that address
documented expressivity deficits of linear-time recurrent
models: a Multi-Scale Depthwise Convolution that addresses the
short-range temporal-hierarchy axis, a Damped Harmonic
Oscillator that extends the recurrent transition class beyond
real-diagonal, and a Chunked Value Decorrelation that targets
associative memory under interference. The mechanisms are
evaluated on a cross-architecture matrix
(RWKV-6, Mamba-2, Linear Attention) across two access modes
(causal, LION bidirectional), two parameter scales (7M, 14M),
and three probes (LibriSpeech, Common Voice for cross-
distribution validation within axis~1, and Multi-Query
Associative Recall for axis~2 isolation at sequence lengths
$T \in \{64, 256, 1024\}$). The resulting evidence supports the
central claim of the thesis: a mechanism's gain on a task is
proportional to the residual structural deficit the architecture
leaves uncovered along the axis the mechanism targets, modulated
by what the task-as-exercised-by-data exercises. Both productive
transfer (deficit-proportional MSDC, BREAK-and-NULL DHO,
asymmetric CVD) and a single counter-example (LION-LIT
$\times$ CVD on Linear Attention, where the absence of decay
inverts the transfer sign) populate the matrix and validate the
mechanism-prior alignment law as a predictive criterion.

---

## Order of writing

1. Chapter~5 lands first, since the RQs and the structure
   paragraph reference its empirical statements by phrasing.
2. RQs are added to Chapter~1 immediately after Chapter~5
   compiles.
3. Structure paragraph is added last; takes one chapter to
   the sentence and is the easiest pass.
4. Contributions paragraph rewrite at the same time as the
   structure paragraph; both edits are local to Chapter~1.
