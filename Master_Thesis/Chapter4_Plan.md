# Chapter 4 — Proposed Solution — Working Plan

*Working planning document for the Proposed Solution chapter. Not
part of the LaTeX deliverable. Captures the section structure, the
per-section content scope, the LION-paper-style visualisation
template each mechanism's table will follow, and a separate
validation plan for the agent that will verify the parameters and
algorithmic forms before LaTeX writing begins.*

*Created 2026-04-29.*

> **Status (2026-04-30):** Chapter 4 has been drafted from this plan;
> the LaTeX file lives at `Master_Thesis/chapters/4_proposed_solution.tex`.
> Mechanism implementation parameters were validated against the
> codebase before drafting; the deployment tables in the chapter
> mirror the LION-paper Table 1 visual style.

---

## Style guidance — read before writing or validating

These principles take precedence over any inherited convention from
the discovery-phase documents.

### Generalise, do not reproduce

The chapter presents the three proposed mechanisms in their **final
generalised form** as if each had been arrived at by a clean
top-down derivation. The chapter does **not**:

- recount the discovery-phase narrative (Stages 2–10, multi-version
  iterations, rejected variants);
- present alternative variants of a mechanism that did not survive
  to the final matrix;
- catalogue rejected parameter choices or initial-bug fixes as
  storylines.

Init-corrections and design choices are mentioned **only** as brief
design notes, in passive voice, never as discovery narratives.
Discovery-phase artefacts in `experiments/final/Mechanisms_Overview.md`,
`experiments/formal_v1/MULTIDIL_INIT_FIX_HANDOFF.md`,
`experiments/formal_v1/MAMBA2_LUCID_ADAPTATION.md`, and the per-stage
summaries are read **only** to extract canonical parameters and
algorithmic forms; their structure and prose are not reproduced.

### Single-variant rule for the Damped Harmonic Oscillator

The thesis reports **the depth-graded variant as the canonical and
sole form** of the Damped Harmonic Oscillator mechanism. The
chapter:

- presents the depth-graded $\theta$-clip schedule as the design;
- does **not** mention an alternative uniform-clip variant;
- does **not** present comparison tables of clip choices;
- treats the depth-graded form as if it were the original design.

Internal codebase identifiers `rse_strong_viscosity` and
`rse_depth_viscosity` collapse to the single canonical name DHO in
the writeup. Mentions of "strong" anywhere in the writeup are
removed.

### LION-paper-style visualisation per mechanism

Each mechanism subsection includes a four-column table mirroring
LION Table 1's structure (`afzal2025linear`, page 7), adapted to
this thesis's three-architecture matrix. The table columns are:

| Architecture | Vanilla causal form | Mechanism-augmented form | Mechanism-augmented LION form |
|---|---|---|---|

with rows for RWKV-6, Mamba-2, and Linear Attention. The "Vanilla
causal form" column is reproduced from Section~\ref{sec:backbones}
for visual continuity. The "Mechanism-augmented" columns show the
specific algebraic insertion point for the mechanism on each
backbone and on each mode. Equations are presented in the same
inline style as LION Table 1 (compact `tabular` cells with two or
three lines of math each). The table is the chapter's primary
information-organising device per mechanism.

### What "generic form" means in each subsection

Each mechanism subsection has two layers:

1. **Generic form**: the architecture-agnostic mathematical
   description of the mechanism, expressed as an operator that can
   be inserted at a specific point in the unified recurrence
   template of equation~(\ref{eq:unified}). This layer is
   architecture-independent.
2. **Per-architecture deployment**: the table described above,
   showing how the generic operator is realised on each of the
   three backbones, with implementation-relevant differences made
   visible (chunk granularity, scan kernel, mode parameter).

The generic form leads; the table follows; the prose around the
table reads the table rather than restating its contents.

---

## What this chapter does in the thesis

Chapter~4 presents the three proposed mechanisms as concrete
instantiations of the mathematical primitives introduced in
Section~\ref{sec:primitives}, deployed across the three causal
backbones formalised in Section~\ref{sec:backbones} and across the
LION bidirectional adaptation of Section~\ref{sec:lion}. The
chapter is the place where this thesis's contributions, to the
extent that they are mechanism-shaped, are stated.

Chapter~4 does not present empirical results or transfer patterns;
those live in Chapter~\ref{ch:experiments}. It also does not
revisit the formal expressivity bounds; those were closed in
Section~\ref{sec:bounds}.

---

## Section structure (proposed)

| § | Title | Role |
|---|---|---|
| 4.1 | Problem formulation and the experimental matrix | Defines the matrix as artefact: 3 architectures $\times$ 2 modes $\times$ 2 scales; states mechanism-prior alignment as the predictive law to be tested. |
| 4.2 | Multi-Scale Depthwise Convolution | Generic form + table + per-architecture deployment. |
| 4.3 | Damped Harmonic Oscillator | Generic form + table + per-architecture deployment. Single (depth-graded) variant. |
| 4.4 | Chunked Value Decorrelation | Generic form + table + per-architecture deployment. Honest provenance vs LUCID. |
| 4.5 | Unified LION wrapper | One wrapper across three backbones; decay-class mapping; LION-LIT vs LION-S role on Linear Attention. |
| 4.6 | Diagnostic instrumentation (optional) | Per-mechanism probes; four-cell engagement classifier. Optional pending writeup time. |

---

## §4.1 Problem formulation and the experimental matrix

### Purpose

Frame the chapter (and the experimental chapter that follows) around
the matrix as artefact, and state the mechanism-prior alignment law
as the predictive criterion the matrix tests.

### Content scope

- The matrix is $3 \times 2 \times 2$: architectures $\{$RWKV-6,
  Mamba-2, Linear Attention$\}$ $\times$ modes $\{$causal,
  LION$\}$ $\times$ parameter scales $\{$7M, 14M$\}$.
- One additional pilot dimension on Common Voice for the causal
  cells.
- One synthetic benchmark, MQAR with $T \in \{64, 256, 1024\}$,
  exercising axis~2.
- The mechanism-prior alignment law is stated formally: a
  mechanism's gain on a given task is proportional to the residual
  structural deficit the architecture leaves uncovered along the
  axis the mechanism targets.

### Anti-scope

- No empirical numbers in this section.
- No closed-cell evidence; that lives in the synthesis chapter.

### Citations

| Concept | Bib key |
|---|---|
| LibriSpeech | `panayotov2015librispeech` |
| Common Voice | `ardila2019commonvoice` |
| MQAR / Zoology | `arora2023zoology` |

---

## §4.2 Multi-Scale Depthwise Convolution (MSDC)

### Purpose

Present the input-side multi-scale local-mixing mechanism as a
specific instantiation of the depthwise-convolution-class operator
introduced in Section~\ref{sec:primitives:msdc}.

### Generic form

Multi-Scale Depthwise Convolution operates on the input side of the
backbone, before the time-mix block. For input $x \in \mathbb{R}^{T \times D}$:
\begin{equation}\label{eq:msdc-generic}
\mathrm{MSDC}(x)[t] = \sum_{d \in \mathcal{D}} \alpha_{d} \cdot \mathrm{DWConv}_{k, d}(x)[t],
\end{equation}
where $\mathcal{D} = \{1, 2, 4, 8\}$ is the dilation set, $k = 3$
is the kernel size, $\mathrm{DWConv}_{k, d}$ is a depthwise (channel-
separable) one-dimensional convolution, and
$\alpha_{d} \in \mathbb{R}^{D}$ are per-layer learnable per-channel
mixing weights. The operator is added as a pre-block residual: each
backbone block $f$ becomes $f(\mathrm{MSDC}(x))$.

### Design notes (brief, not a narrative)

- Dilation set $\{1, 2, 4, 8\}$ at $k = 3$ covers receptive fields
  up to 17 frames either side, which after 2x acoustic subsampling
  matches the phoneme-to-syllable scale.
- Mixing weights $\alpha_{d}$ and convolution kernels for $d > 1$
  are initialised to a small non-zero value rather than to zero, in
  order to break the multiplicative gradient trap that otherwise
  silently collapses the operator to a single dilation. See the
  validation plan for the exact value.
- Depthwise structure keeps parameter count under 1 per channel
  per dilation per layer.

### Per-architecture deployment table (LION-paper-style)

This is the chapter's primary visualisation. To populate it:

| Architecture | Vanilla causal form | + MSDC (causal) | + MSDC (LION) |
|---|---|---|---|
| RWKV-6 | from eq. (\ref{eq:rwkv6-state})–(\ref{eq:rwkv6-readout}) | apply MSDC pre-Time-Mix | LION-S wrapper around MSDC-augmented Time-Mix |
| Mamba-2 | from eq. (\ref{eq:mamba2-ssd}) | apply MSDC pre-SSD | LION-S wrapper around MSDC-augmented SSD |
| Linear Attention | from eq. (\ref{eq:la-state})–(\ref{eq:la-output}) | apply MSDC pre-state-update | LION-LIT wrapper around MSDC-augmented kernel |

The mechanism is architecture-agnostic on the input side, so the
table emphasises the consistency of insertion point across the
three rows.

### Anti-scope

- No mention of the v1/v2 init-bug story.
- No recount of cross-domain literature; that is §3.4.1's job.

### Citations

| Concept | Bib key |
|---|---|
| Depthwise dilated convolution lineage (cross-domain) | `kiruluta2025breaking, duan2024visionrwkv, wang2025lit, xiong2025audiorwkv` |

---

## §4.3 Damped Harmonic Oscillator (DHO)

### Purpose

Present the block-complex transition with Rayleigh dissipation as a
specific instantiation of the primitive introduced in
Section~\ref{sec:primitives:dho}, deployed on the diagonal-class
backbones whose transition operator the mechanism extends.

### Generic form

DHO replaces a real-diagonal recurrent transition with a
block-diagonal product of $K/2$ rotation-and-damping blocks. For
each head $h$, block $b$ at step $t$:
\begin{equation}\label{eq:dho-generic}
G_{t,h,b} = e^{-\lambda_{\mathrm{eff}, t, h, b}} \cdot R(\theta_{t, h, b}), \qquad
\lambda_{\mathrm{eff}, t, h, b} = \lambda_{\mathrm{raw}, h, b} + \eta_{h, b} \cdot \theta_{t, h, b}^{2},
\end{equation}
with $R(\theta)$ the SO(2) rotation of equation~(\ref{eq:block-complex}),
$\theta_{t, h, b}$ a data-dependent rotation angle produced by a
low-rank LoRA-style projection of the input, and
$\eta_{h, b} \ge 0$ a per-head, per-block viscosity coefficient
initialised to zero. Validation against the canonical implementation
confirmed $\eta_{h,b}$ shape `(n_heads, n_blocks)` across all three
backbones (validation pass V-4.3.2, 2026-04-29).

The rotation angle is bounded by a depth-graded clip schedule:
\begin{equation}\label{eq:dho-clip}
|\theta_{t, b, \ell}| \le \theta_{\max}(\ell),
\end{equation}
where $\ell$ is the layer index and $\theta_{\max}(\cdot)$ is a
fixed depth-graded schedule that grows from a small clip in early
layers to a larger clip in deeper layers. The schedule is the
mechanism's only canonical clip parameterisation; uniform clips
are not used.

### Design notes (brief)

- The depth-graded schedule reflects the empirical observation that
  deeper layers operate on more abstract phase structure that
  benefits from a wider rotation budget, while shallow layers
  benefit from a tight constraint that biases the operator toward
  near-identity initialisation.
- The viscosity coefficient $\eta_{b}$ is zero-initialised so that
  DHO reduces exactly to plain block-complex RSE at step zero of
  training and engages only as $\eta$ moves under the gradient.
  This makes the mechanism zero-regression-by-construction at init.
- The data-dependent rotation angle is produced by a low-rank
  LoRA-style projection with a depth-graded rank schedule that
  mirrors the depth-graded $\theta$-clip schedule:
  $\{16, 16, 32, 32, 48, 48\}$ for the 6-layer (7M) configuration
  and an analogous schedule for the 12-layer (14M) configuration.
  The rank schedule rises with depth in lockstep with the
  $\theta$-clip schedule, so that the deeper layers, which receive
  the wider rotation budget, also receive the higher-capacity
  LoRA projection.

### Per-architecture deployment table

DHO is a **transition-side** mechanism: it modifies the recurrent
$A_{t}$ operator. The table contrasts the vanilla and
DHO-augmented transition for each architecture:

| Architecture | Vanilla transition $A_{t}$ | + DHO transition |
|---|---|---|
| RWKV-6 | $\mathrm{diag}(w_{t}) \in (\mathbb{R}_{+})^{D/h}$ | block-diagonal product of $K/2$ blocks $G_{t, b}$ replacing pairs of real channels |
| Mamba-2 | $a_{t} I$ scalar identity | block-diagonal closure of pairs of state channels around the scalar $a_{t}$ |
| Linear Attention | $I$ identity (trivial) | block-diagonal product of $K/2$ rotation-and-damping blocks |

The table makes visible that DHO has its strongest structural
impact on the architecture with the most trivial vanilla transition
(Linear Attention), where it introduces an entirely new function-
class capability, and the smallest impact on the architecture
whose vanilla transition already provides per-channel decay
diversity (RWKV-6).

### LION-mode form

DHO transfers naturally to LION because the block-complex transition
is structurally compatible with the symmetric decay mask of the
LION parallel form. The LION-mode deployment uses LION-S on
RWKV-6 and Mamba-2 (where the underlying decay class is data-dep
diagonal) and the controlled LION-S deployment on Linear Attention
(see §4.5 for the rationale).

### Anti-scope

- No mention of a uniform-clip alternative.
- No mention of the "strong" variant.
- No comparative discussion of clip schedules.

### Citations

| Concept | Bib key |
|---|---|
| Lie-group SO(2) | `hall2015lie` |
| Rayleigh dissipation function | `goldstein2002classical` |
| Diagonal SSM class membership | `sarrof2024expressive` |

---

## §4.4 Chunked Value Decorrelation (CVD)

### Purpose

Present the within-chunk value-decorrelation mechanism as a new
operator constructed for linear-time recurrent substrates, sharing
the key-similarity decorrelation principle of LUCID without being
an adaptation of it.

### Generic form

CVD operates within each chunk of size $T_{c}$ and produces
preconditioned values that enter the recurrence in place of the
raw values. For a chunk of keys $K_{c} \in \mathbb{R}^{T_{c} \times d}$
and values $V_{c} \in \mathbb{R}^{T_{c} \times d_{v}}$:
\begin{equation}\label{eq:cvd-generic}
\tilde V_{c} = P_{c}^{-1} V_{c}, \qquad
P_{c} = \exp\!\left( \tau_{h} \cdot \frac{K_{c}^{\mathrm{RN}} (K_{c}^{\mathrm{RN}})^{\top}}{\sqrt{d}} \right) + \varepsilon I,
\end{equation}
where $K_{c}^{\mathrm{RN}}$ is the row-RMS-normalised keys within
the chunk, $\tau_{h}$ is a per-head learnable temperature, and
$\varepsilon I$ is a small regulariser. Validation against the
canonical implementation found $\varepsilon$ to be architecture-split
(validation pass V-4.4.1, 2026-04-29):
$\varepsilon = 10^{-4}$ on the Mamba-2 deployment and
$\varepsilon = 10^{-6}$ on the RWKV-6 and Linear Attention deployments.
The chapter reports the split honestly with a one-sentence
explanation: the larger Mamba-2 value stabilises the chunk-local
solve at trained temperatures around $\tau_{h} \approx 1.5$ where
the SSD chunk-local Gram matrix becomes ill-conditioned; the
smaller RWKV-6 and LA value is paired with an element-wise
singular-fallback in `torch.linalg.solve_ex`.

The per-head temperature is initialised so that
$\mathrm{softplus}(\tau_{\mathrm{raw}, h}) = 1.0$ on RWKV-6, on
the LA-LION deployments, and on the Mamba-2 deployment (init
$\tau_{\mathrm{raw}} = \log(e - 1)$). The Linear Attention
*causal* deployment uses $\mathrm{softplus}(0) = \ln 2 \approx 0.693$
as the initial value (init $\tau_{\mathrm{raw}} = 0$). This minor
cross-backbone difference in the init scheme is footnoted in §4.4
where the trained $\tau_{h}$ values are reported.

The preconditioned values $\tilde V_{c}$ are then consumed by the
unchanged recurrence of the underlying backbone. The full $T \times T$
similarity matrix is never materialised; the operation is
chunk-local with cubic cost $T_{c}^{3}$ per chunk per head from
the linear-system solve, which is amortised across the linear-time
backbone's $T / T_{c}$ chunks.

### Honest provenance vs LUCID

The relationship to LUCID Attention~\cite{duvvuri2026lucid} is
shared logical motivation rather than direct adaptation. LUCID
operates on the full $T \times T$ softmax-attention matrix inside a
quadratic-complexity Transformer; preserves $O(N^{2} d)$ complexity;
and combines the preconditioner with $\mathrm{softmax}(QK^{\top})$
weights in front of the value path. CVD operates inside the
recurrence of a linear-time RNN; never materialises the full
$T \times T$ matrix; uses chunk-local $T_{c} \times T_{c}$ systems
exclusively; introduces a per-head learnable temperature $\tau$ in
place of LUCID's fixed $1 / \sqrt{d}$ scaling; and adds an
$\varepsilon I$ regulariser absent from LUCID. The substrate change
from softmax attention to linear-time recurrent state is the
load-bearing structural difference; the per-head $\tau$ and
$\varepsilon I$ are required for stability of the chunk-local solve
on the linear-time substrate. The chapter states this clearly and
without hedging.

### Per-architecture deployment table

CVD has three architecture-specific deployments because the
substrate structure differs across backbones:

| Architecture | Chunking | Preconditioner source | Solve location |
|---|---|---|---|
| RWKV-6 | within-chunk inside the chunked WKV scan | keys $K_{c}$ produced by the Time-Mix projection | once per chunk, per head, before the WKV scan consumes the values |
| Mamba-2 | within each SSD chunk of the dual form | the chunk-local $\mathbf{C}_{c}$ tensor (the query analog in Mamba-2's scalar-identity SSD) | once per chunk, per head, before the SSD scan consumes the values |
| Linear Attention | within-chunk inside the LION parallel form (when in bidirectional mode); no chunking in causal mode (operates on full state) | $\phi(K_{c})$ in the feature-mapped key sequence | inside the parallel attention path |

The Mamba-2 deployment uses the C-correlation analog rather than
the more obvious B-correlation analog; the validation plan
documents this choice.

### Anti-scope

- No retracted-incompatibility narrative.
- No discovery-phase story of "we initially thought Mamba-2 was
  incompatible".
- No chunking-implementation recipe; that is engineering detail,
  not part of the mechanism description.

### Citations

| Concept | Bib key |
|---|---|
| LUCID prior art | `duvvuri2026lucid` |
| Delta-rule connection | `schlag2021deltanet, yang2024deltanet` |
| Mamba-2 SSD chunk structure | `dao2024transformers` |

---

## §4.5 Unified LION wrapper

### Purpose

Present a single LION wrapper that maps each causal backbone to its
bidirectional parallel form, with an explicit decay-class
correspondence that follows from Section~\ref{sec:lion}.

### Content scope

- The LION code path is realised as a **shared interface
  convention** rather than a single shared wrapper class.
  Validation (V-4.5.1, 2026-04-29) found that each backbone
  carries its own LION deployment:
  - **RWKV-6** uses a `mode` parameter on the existing
    `RWKV6TimeMix` class, with values selecting between
    `recurrent`, `lion`, and `bidir_serial`. The `lion` branch
    dispatches into `lion_attention.lion_parallel_attention`.
  - **Mamba-2** uses a `mode` parameter on `Mamba2Block` (and on
    `Mamba2RSEBlock`), with values `recurrent`, `lion`, or
    `lion_chunk`. The `lion` branch dispatches into
    `mamba2_kernels.ssd_scan_lion`.
  - **Linear Attention** uses a separate encoder class
    `LIONLinearAttentionEncoder` with a `decay_mode` parameter
    selecting between `lit` and `s`. The class is distinct from
    `CausalLinearAttentionEncoder` used for the causal cells.
- The three deployments share a uniform algebraic form, the
  bidirectional parallel attention of Afzal et al. (2025), but each
  is realised by backbone-specific kernels
  (`lion_parallel_attention`, `ssd_scan_lion`, and
  `LIONLinearAttentionLayer.forward_parallel`). The chapter
  describes this as a unified mathematical interface across three
  per-backbone deployments rather than as a single shared
  implementation.
- Per-backbone decay-class mapping:
  - RWKV-6 causal $\to$ LION-S, with the per-channel data-dep
    decay $w_{t}$ inherited.
  - Mamba-2 causal $\to$ LION-S, with the per-token scalar
    data-dep decay $a_{t}$ inherited.
  - Linear Attention causal $\to$ LION-LIT, with $\lambda = 1$
    per the Afzal et al. table-default mapping.
- Linear Attention is additionally tested in a controlled LION-S
  configuration in which a per-token sigmoid decay is added that is
  not present in the underlying causal backbone. This controlled
  configuration is the cleanest available test of decay as a
  structural prerequisite for the value-decorrelation mechanism on
  the bidirectional substrate, and is reported alongside the
  table-default LION-LIT result in Chapter~\ref{ch:experiments}.

### Per-architecture mapping table

| Architecture | Causal decay class | LION decay class (default) | LION decay class (controlled) |
|---|---|---|---|
| RWKV-6 | per-channel data-dep | LION-S | n/a |
| Mamba-2 | scalar per-token data-dep | LION-S | n/a |
| Linear Attention | none ($\lambda = 1$) | LION-LIT | LION-S |

The table reads as: each backbone inherits its causal decay class
into LION mode by the natural mapping per Afzal et al. Linear
Attention's default mapping is LION-LIT; the controlled LION-S
deployment on Linear Attention is reported as a falsification
control isolating the role of decay in bidirectional accumulation.

### Anti-scope

- No comparison with operator-level (Vision-RWKV Bi-WKV) or
  ordering-level (VisualRWKV) bidirectional strategies; that
  comparison was made in §2.2 and is referenced briefly here only.

### Citations

| Concept | Bib key |
|---|---|
| LION framework | `afzal2025linear` |
| Linear Attention causal | `katharopoulos2020transformers` |

---

## §4.6 Diagnostic instrumentation (optional)

Optional section, included if writeup time permits. Covers the
per-mechanism diagnostic probes (parameter mobility, gradient signal,
activation magnitudes, mechanism-specific scalars) and the four-cell
engagement classifier (engaged $\times$ helpful) used in the
experimental chapter. Defer to Chapter~5 if cut.

---

## Order of writing

1. **§4.1** first to set the matrix-as-artefact frame.
2. **§4.2 (MSDC)** second because the mechanism is the simplest of
   the three and serves as the table-template carrier for §4.3 and §4.4.
3. **§4.3 (DHO)** third.
4. **§4.4 (CVD)** fourth, with the LUCID-provenance prose given the
   most attention.
5. **§4.5 (LION wrapper)** fifth.
6. **§4.6** last, optional.

---

## Validation plan — for the working agent

The following validation pass is intended to be executed by an
agent **before** chapter writing begins. The output of this pass is
a structured report that this thesis's author will review; the
chapter is written only after the report is accepted.

### What the agent does

For each item below, the agent locates the canonical value or
algorithmic form in the named internal artefact, extracts the exact
parameter or equation, and reports it in the structured format
specified at the end of this section.

The agent does **not** rewrite or paraphrase the discovery-phase
narrative; only extracts canonical facts.

### Items to validate

#### V-4.2.1 — MSDC dilation set, kernel size, and depthwise structure

- Source: `experiments/formal_v1/MULTIDIL_INIT_FIX_HANDOFF.md` plus
  the codebase backbone identifiers under
  `experiments/formal_v1/src/models/` for any backbone whose name
  contains `convshift_multidil_symmetric_v2`.
- Extract: dilation set ($\mathcal{D}$ in the plan), kernel size
  $k$, whether the convolution is depthwise (channel-separable) or
  not, and where in the backbone the operator is inserted (input
  side, value path, key/value/decay paths).
- Confirm or correct: dilation set $\{1, 2, 4, 8\}$, kernel size
  $k = 3$, depthwise on the input side.

#### V-4.2.2 — MSDC mixing-weight initialisation and bias-break value

- Source: `experiments/formal_v1/MULTIDIL_INIT_FIX_HANDOFF.md`.
- Extract: the initial value of $\alpha_{d}$ for $d > 1$ that
  breaks the multiplicative gradient trap, the corresponding
  initial value of $W_{d > 1}$ for the convolution kernel, and the
  rationale stated in the source.
- Confirm: $\alpha_{d > 1} \approx 10^{-2}$ at init,
  $W_{d > 1} \sim \mathcal{N}(0, 10^{-4})$.

#### V-4.3.1 — DHO depth-graded $\theta$-clip schedule

- Source: `experiments/final/Mechanisms_Overview.md` (the rotation-
  budget paragraph), `experiments/final/STATUS.md` ("decision
  update 2026-04-26") for the layer-by-layer values.
- Extract: the exact $\theta_{\max}(\ell)$ value at each layer index
  for the configurations used in the experimental matrix
  (typically 6 layers at 7M, 12 layers at 14M).
- Confirm or correct: the schedule reported across the matrix is
  a single canonical schedule with no within-matrix variation.

#### V-4.3.2 — DHO LoRA rank for $\theta$, viscosity-coefficient parameterisation

- Source: `experiments/formal_v1/RWKV6_FIX_REVIEW.md` and the
  codebase under `formal_v1/src/models/rwkv6_time_mix.py` (or the
  equivalent files for `mamba2_rse` and `linear_attn_rse`).
- Extract: rank of the LoRA projection that produces $\theta_{t, b}$;
  whether the viscosity coefficient $\eta_{b}$ is per-block,
  per-head, or per-layer; and the initialisation of $\eta$.
- Confirm or correct: LoRA rank 48; $\eta$ zero-initialised.

#### V-4.4.1 — CVD per-head learnable temperature and $\varepsilon$

- Source: codebase implementation of the LUCID variant
  (`formal_v1/src/models/rwkv6_time_mix.py::_apply_lucid_recurrent`,
  `formal_v1/src/models/mamba2_kernels.py::_apply_lucid_mamba2_chunked`,
  and the corresponding LA implementation), plus
  `experiments/formal_v1/MAMBA2_LUCID_ADAPTATION.md` for the
  preconditioner formulation.
- Extract: whether $\tau$ is per-head, per-layer, or scalar; the
  initial value of $\tau$; the value of $\varepsilon$.
- Confirm or correct: per-head learnable $\tau$;
  $\varepsilon = 10^{-4}$.

#### V-4.4.2 — CVD chunk size $T_{c}$ for each backbone

- Source: codebase configurations under `formal_v1/configs/` and
  `experiments/formal_v1/MAMBA2_LUCID_ADAPTATION.md`.
- Extract: the chunk size used by the LUCID-variant deployment on
  RWKV-6, on Mamba-2 (which inherits the SSD chunk size), and on
  the LION parallel form (which operates on full-T or chunked
  bidirectional).
- Confirm or correct: $T_{c} = 64$ for RWKV-6 chunked recurrent,
  $T_{c}$ inherited from SSD for Mamba-2 (verify the SSD chunk-size
  default), $T_{c}$ for LION parallel form (verify whether
  chunked or full-T).

#### V-4.4.3 — CVD on Mamba-2: B-correlation versus C-correlation choice

- Source: `experiments/formal_v1/MAMBA2_LUCID_ADAPTATION.md`.
- Extract: the explicit choice between B-correlation and
  C-correlation in the SSD-chunk-local deployment, and the stated
  rationale for the choice.
- Confirm or correct: C-correlation is reported as the canonical
  form; the rationale is that the C tensor is the query analog in
  the SSD dual form and produces marginally cleaner numbers in
  the experimental matrix.

#### V-4.5.1 — LION wrapper unified mode parameter

- Source: codebase implementation (`formal_v1/src/models/rwkv6_time_mix.py`
  for the canonical wrapper plus `mamba2_kernels.py::ssd_scan_lion`
  and `linear_attn_lion.py::LIONLinearAttentionEncoder`).
- Extract: the actual interface (function signatures, mode
  parameter values), and confirm that the same wrapper interface is
  used across all three backbones.
- Confirm or correct: a single `mode` parameter selects between
  "causal" and "lion" on each backbone; the LION code path is
  shared across the three backbones via a common wrapper.

#### V-4.5.2 — Linear Attention LION-LIT versus LION-S decay mapping

- Source: `experiments/final/Master_Plan.md` §3 plus
  `experiments/final/STATUS.md` LION-S follow-up entries.
- Extract: which LION variant is the table-default for Linear
  Attention; which controlled LION-S configuration is reported
  alongside the default; whether both appear in the experimental
  matrix.
- Confirm or correct: LION-LIT is the table-default for LA;
  LION-S is reported as a controlled experiment isolating the role
  of decay; both appear in the matrix.

### Output format expected from the agent

For each V-* item, the agent returns a block of the form:

```
## V-X.Y.Z — <short title>

### Source consulted
<list of files actually read>

### Extracted parameter / form
<exact value, equation, or algorithmic form>

### Match against plan
[match / mismatch] — <one sentence>

### If mismatch: corrected value or form
<the corrected version>

### Quoted evidence
<one or two short quotes from the source files, with line numbers>
```

The agent does **not** modify any files and does **not** edit
`Chapter4_Plan.md` directly. The structured report is produced as
a standalone output for the thesis author's review.

### Out of scope for the agent

- The agent does not write LaTeX.
- The agent does not produce historical or narrative text.
- The agent does not generate equations beyond what is required to
  report the canonical form already present in the source.
- The agent does not consult external papers; that validation was
  performed during the Chapter~3 pass.

---

## New bib entries that may be needed

Validation pass may identify a citation gap; if so, draft the bib
entry in this file before requesting it be added to
`bibliography.bib`. None are anticipated at planning time; all
relevant citations are already listed per-section above and are
present in `bibliography.bib`.

---

*End Chapter 4 working plan v1 (2026-04-29). Validation pass to be
executed by a separate agent; chapter writing begins only after the
validation report is reviewed and accepted.*
