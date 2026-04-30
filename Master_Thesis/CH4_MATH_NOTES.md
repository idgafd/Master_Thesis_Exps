# Chapter 4 mathematical clarity sharpening notes

*Working planning document. Not part of the LaTeX deliverable.
Records the items to address in Chapter~4 to bring the
mathematical formulation closer to the LION paper's standard
without exceeding the scope of a Master's thesis.*

*Created 2026-04-30. Reference: arXiv:2502.16249, Afzal et al.,
LION: Linear Group RNN for 3D Object Detection in Point Clouds
(used as the calibration target by the user; the thesis matches
the level of mathematical clarity in that paper only partially,
since the present thesis's scope is simpler).*

---

## What the LION paper does well that we should match

1. *One named equation per structural concept.* Linear-attention
   recurrence, causal mask, bidirectional mask, named variants
   (LION-LIT, LION-D, LION-S) each get their own equation
   number. Cross-references stay tight.
2. *Notation defined upon introduction, no forward references.*
   Each symbol that enters an equation is defined at the moment
   of the equation, with a one-line gloss.
3. *Hyperparameter schedules stated in formal form once and
   referenced afterwards.* The LION paper gives the per-variant
   $\lambda_{k}$ formulas in a single boxed section and refers
   to them by name in the deployment discussion.
4. *Initialisation values reported as formal statements rather
   than embedded in the derivation prose.* The LION paper's
   "Initialisation" subsection per variant lets the
   derivation read cleanly without parametric clutter.

## Items to apply in our Chapter~4

### Item~1. Consistent subsection structure across all three mechanisms

Current state: MSDC and DHO have *Generic form* and *Design
notes* subsections; CVD has *Generic form* and *Provenance* but
no *Design notes*. The asymmetry hides the per-mechanism
initialisation and hyperparameter schedule.

Proposed structure for each mechanism:

```
\subsection{Generic form}
\subsection{Initialisation}
\subsection{Hyperparameter schedule}    (where applicable)
\subsection{Per-architecture deployment}
\subsection{Provenance}                 (CVD only; MSDC and DHO
                                         provenance is one
                                         paragraph at the end of
                                         Generic form)
```

Apply to MSDC, DHO, CVD identically. The CVD provenance
subsection is the only special case, since the LUCID
relationship requires a paragraph-length statement.

### Item~2. CVD design notes hoisted out of generic form

Current §4.4.1 generic-form text contains the $\tau_{h}$
softplus link, the architecture-split $\varepsilon$, and the
chunk size $T_{c} = 64$ inline with the equation. These belong
in a *Design notes* or *Initialisation* subsection.

Proposed CVD subsection layout:

- §4.4.1 *Generic form.* Equation~(\ref{eq:cvd-generic}) only,
  with the symbols defined inline as in the LION paper. No
  hyperparameter values inside this subsection.
- §4.4.2 *Initialisation and hyperparameters.* Formal statement
  of $\tau_{\mathrm{raw}, h}$ initialisation per architecture,
  the $\varepsilon$ value per architecture, the chunk size
  $T_{c} = 64$, and the per-head learnable temperature link
  $\tau_{h} = \mathrm{softplus}(\tau_{\mathrm{raw}, h})$.
- §4.4.3 *Provenance: shared logic, distinct substrate.* Current
  §4.4.2 stays as is; just renumber.
- §4.4.4 *Per-architecture deployment.* Current §4.4.3 stays as
  is; just renumber.

### Item~3. Trim §4.1 problem formulation

Current §4.1 *Problem formulation and the experimental matrix*
overlaps with §3.1 *Axis decomposition*. The matrix description,
the law statement, and the probe-list paragraph each repeat
content from Chapter~3. Trim §4.1 to roughly half its current
length:

- One paragraph stating that the matrix is the principal
  artefact and the mechanisms are demonstrations of a predictive
  law that the matrix is built to test (keep verbatim).
- One paragraph defining the matrix as
  $3 \times 2 \times 2$ (architectures, modes, scales) with the
  three probes named (LibriSpeech, Common Voice, MQAR). Drop
  the longer probe descriptions; cite §A.1 to A.3 of the
  appendix.
- One paragraph stating the alignment law in two sides
  (positive and null). Drop the matrix-is-sized-to-support
  paragraph; the experimental chapter shows the support
  directly.

Drop the long paragraph on which mechanism targets which axis;
that information is now contained in the *Generic form*
subsection of each mechanism, which is the natural local home.

### Item~4. MSDC initialisation paragraph

Current §4.2.2 paragraph 2 (lines 142 to 150) reads:

> The mixing weights $\alpha_d$ for $d > 1$ and the convolution
> kernels for $d > 1$ are initialised to a small non-zero value
> rather than to zero, so that the gradient with respect to
> $\alpha_d$ at step zero is non-degenerate. The choice follows
> from the multiplicative form of equation~(\ref{eq:msdc-generic}):
> a strict zero on either of the two factors at any single
> dilation collapses the gradient through that dilation to zero,
> and the operator silently reduces to its $d = 1$ branch.

This is design rationale rather than initialisation statement.
Replace by a formal initialisation block:

> *Initialisation.* The mixing weights are initialised at
> $\alpha_{d = 1}^{(\ell)} = 1$ in every channel of every layer
> and at $\alpha_{d > 1}^{(\ell)} = 10^{-2}$. The branch-1
> convolution kernel is initialised as the centre tap (weights
> $[\tfrac{1}{2}, 0, \tfrac{1}{2}]$, zero bias); branch
> convolutions for $d > 1$ are drawn from
> $\mathcal{N}(0, 10^{-4})$. The non-zero initialisation of the
> $d > 1$ legs prevents the multiplicative product
> $\alpha_{d} \cdot W_{d}$ from collapsing the gradient through
> the dilated branches at step zero, which would otherwise
> reduce the operator to its $d = 1$ branch under stochastic
> gradient descent.

The rationale is preserved, but the initialisation values come
first as a formal block.

### Item~5. Equation cross-references and notation

Current state of equation references in Chapter~4:

- $\mathrm{DWConv}_{k, d}(x)$ is introduced in §3.4.1
  (`eq:multi-resolution`) but reused without a back-reference in
  §4.2.1 (`eq:msdc-generic`). Add a short forward citation:
  "where $\mathrm{DWConv}_{k, d}$ denotes a depthwise dilated
  convolution as in equation~(\ref{eq:multi-resolution}) of
  Section~\ref{sec:primitives:msdc}".
- $R(\theta)$ is introduced in §3.4.2 (`eq:block-complex`) and
  reused in §4.3.1 (`eq:dho-generic`). Same treatment.
- $K_{c}^{\mathrm{RN}}$ is introduced in §3.4.3 (`eq:lucid-rn`)
  and reused in §4.4.1 (`eq:cvd-generic`). Same treatment.

The pattern across the three mechanisms is consistent: the
mathematical primitive lives in §3.4 of Chapter~3, and the
mechanism-specific instantiation in Chapter~4 cites the
primitive's equation by number rather than restating it. The
treatment matches the LION paper's structure of "primitive in
the math section, deployment in the proposed-solution section".

### Item~6. DHO eta-and-theta notation

Current §4.3.1 carries the indices $(t, h, b)$ on each of
$\theta$, $\eta$, $\lambda_{\mathrm{raw}}$, and
$\lambda_{\mathrm{eff}}$. The product is parameter-tensor-clean
but visually heavy. Two micro-edits:

- Suppress the $h$ subscript inside the displayed equation when
  the head is fixed, with one line: "we suppress the head index
  $h$ inside displayed equations and reinstate it when stating
  parameter shapes."
- State the parameter shapes once, in a small paragraph after
  the equation: "The parameter shapes are
  $\eta \in \mathbb{R}^{H \times K/2}$,
  $\lambda_{\mathrm{raw}} \in \mathbb{R}^{H \times K/2}$, and
  $\theta_{t} \in \mathbb{R}^{H \times K/2}$ produced by a
  rank-$r(\ell)$ LoRA projection of the per-token input."

### Item~7a. DHO decay self-sufficiency design note

Add a design note in §4.3 (after the existing three numbered
choices on $\eta$ initialisation, depth-graded $\theta$-clip
schedule, and LoRA rank schedule). The note states that the
per-block damping factor $e^{-\lambda_{\mathrm{eff}, t, h, b}}$
in equation~(\ref{eq:dho-generic}) makes the block-complex
transition decay-bounded by construction, independent of any
external mask, and that the property becomes load-bearing in the
bidirectional setting where the LION wrapper's mask supplies
decay through $\lambda_{k}$ in equation~(\ref{eq:lion-mask}).
DHO is therefore decay-self-sufficient on either LION variant
(LION-LIT or LION-S), unlike the value-decorrelation mechanism
of Section~\ref{sec:proposed:cvd} whose conditioning guarantee
requires a decay-bearing mask.

Two to three sentences. This sets up the empirical payoff in
Chapter~5 §5.4 without surprise: the LION-LIT~$\times$~DHO
BREAK cell is then read as a confirmation of a property
established in Chapter~4, rather than as a new structural
reading introduced in the experimental chapter.

### Item~7. CVD provenance condensation

Current §4.4.2 *Provenance: shared logic, distinct substrate*
runs about forty lines and repeats content from §3.4.3
*Key-similarity preconditioning and value decorrelation*
(which itself runs about thirty lines on LUCID). After Item~3
trims §4.1, condense §4.4.2 to a one-paragraph statement of the
substrate distinction (LUCID is softmax-attention, CVD is
linear-time RNN; LUCID materialises the full $T \times T$
matrix, CVD materialises chunk-local $T_{c} \times T_{c}$ blocks
only; CVD adds the per-head temperature and the
$\varepsilon I$ regulariser; the shared logical move is the use
of a key-similarity preconditioner to decorrelate values before
they enter the memory operation, motivated by the same
observation that correlated keys cause readout interference in
any associative-memory implementation).

The DeltaNet rank-one analogy in current §4.4.2 paragraph 4
(four sentences) is dropped; it appears in §3.4.3 already and
is more naturally home there.

---

## Estimated edit budget

| Item | Lines affected | Edit time |
|------|---------------:|----------:|
| 1 | 0 (structure only) | 5 min planning |
| 2 | ~30 lines, three subsections | 25 min |
| 3 | ~70 lines, single section | 30 min |
| 4 | ~10 lines, one paragraph | 10 min |
| 5 | ~5 lines, three forward citations | 5 min |
| 6 | ~5 lines, two micro-edits | 5 min |
| 7 | ~20 lines, one paragraph | 15 min |
| | **Total** | **~95 min** |

The set of edits is local to Chapter~4 and does not require
re-running any compiles or experiments.

---

## What we are not changing

- The three mechanism names (MSDC, DHO, CVD) and their
  deployment tables stay as written.
- The LION wrapper section (§4.5) stays as written; the
  decay-class mapping table is correctly placed.
- The diagnostic probes section (§4.6) stays as written;
  Chapter~5 picks up the four-cell engagement classifier.
- The provenance honesty for MSDC ("we are one node in a
  cross-domain pattern") and for DHO ("internally derived from
  Lie-group extension plus Rayleigh dissipation") stays as
  written.
