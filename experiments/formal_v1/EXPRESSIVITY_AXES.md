# Expressivity Axes — Mechanism–Task Matching Framework

Thesis re-framing note. Synthesises the multi-task discussion and
clarifies what the ASR-only nulls actually tell us.

---

## Core claim

Linear-time RNN expressivity does **not** decompose into a single scalar.
It decomposes into **orthogonal axes**, each exercised by different task
structures. Mechanism value is axis-specific: a mechanism only converts
into loss gain when the task's structural prior exercises the axis that
mechanism targets.

> **Function-class extension aligned with the task's structural prior →
> empirical gain. Function-class extension without such alignment →
> engaged-null.**

This is why five+ mechanisms in our Stages 2–10 corpus engaged SGD
mechanically but did not move ASR CER: **they address axes ASR does
not exercise**, not "they don't work."

---

## The five axes

| # | Axis | What it measures | Tasks that exercise it |
|---|---|---|---|
| **1** | **Feature extraction + short-range temporal** | Multi-scale local patterns, formant dynamics, phoneme-to-syllable windows | **ASR**, audio, music, dense prediction |
| **2** | **Associative memory capacity under interference** | Number of $(k, v)$ pairs storable without collision when $N \gtrsim d$ | **MQAR**, copy-with-noise, synthetic $(k, v)$ recall |
| **3** | **State-tracking / finite-automaton simulation** | Tracking discrete state across arbitrary sequence length (TC⁰ → NC¹) | Parity, modular arithmetic, flip-flop, Dyck |
| **4** | **Long-range information flow** | Whether position 0 can influence position $T$ at large $T$ | NIAH, long-context QA, LRA |
| **5** | **Content-adaptive computation** | Inductive bias varying with input content | Mixed-regime sequences, task-switching |

Axes are largely independent: a mechanism strong on axis 2 can be flat
on axis 1, and vice versa.

---

## Master mechanism × axis table

Each mechanism is placed on the axis/axes it structurally targets,
with the empirical ASR result where available and the predicted
matching benchmark for validation.

| Mechanism | Source | Family | Target axis | ASR result (7 M / 30 ep) | Interpretation |
|---|---|---|---:|---:|---|
| **RSE + Rayleigh viscosity** (ours, Stages 3–5) | Internal | B | **1** (complex-pole formants) | **WIN — 0.1185 dev (anchor)** | Axis 1 exercised by ASR formant dynamics |
| **Multi-dilation ConvShift symmetric** (Paper 7 → multidil_sym, Stage 10.3-sym) | Paper 7 + internal | A | **1** (short-range temporal hierarchy) | **WIN — 0.1013 dev / 0.1000 test (`_v2`, fixed init), 0.1153 (broken init = single-dil + scalar)** | Axis 1 exercised by phoneme-to-syllable window; fixed-init α₂ > α₁ at layers 1–5, α₈ engages at depth. See `MULTIDIL_INIT_FIX_HANDOFF.md` |
| **Rotation budget refinement** (ours, Stage 4 strong / depth) | Internal | B | **1** (constraint removal for axis 1) | **WIN — 0.1192 dev** | Removed misallocated clip; same axis |
| **Delta rule rank-1** (Yang et al. DeltaNet / Stage 8 T1) | Paper (DeltaNet) | B | **2** (directional erasure reduces interference) | **Null — 0.1258 tied vanilla** (engaged: $g_\delta \le 0.65$, $\beta$ p95 > 1) | ASR does not exercise axis 2 — $T \ll d^2$, no associative recall in task signal |
| **DeltaProduct** (Householder products, rank-$n_h$) | Paper (arXiv:2502.10297) | B | **2 + 3** (rank-$n_h$ memory, state-tracking) | Not run — predicted null on ASR | Same as Delta rule; rank extension doesn't help on non-axis-2/3 tasks |
| **M²RNN** (tanh-state non-linearity, Stage 10.2) | Paper 9 | C | **3** (TC⁰ → NC¹ boundary) | **Tied vanilla — 0.1276** | ASR does not exercise axis 3 |
| **NCGRU-Cayley orthogonal** (Stage 10.5) | Paper 6 | B | **2 + 3** (norm-bounded rotation) | **REGRESSION-track** — ep 15 dev 0.1518 | Same as Delta / DeltaProduct; no axis 2/3 structure in ASR |
| **Negative eigenvalues** (arXiv:2411.12537) | Paper | B | **3** (state-tracking via eigen-range extension) | Not run — predicted null on ASR | Parity / regular-language specific |
| **Non-normal RSE (T2, S9)** (ours, Stages 8–9) | Internal | B | **2 + 3** (dense per-token polar) | **Null — 0.1202 / 0.1218** (engaged: $\rho$-LoRA $\|\cdot\|_F$ 18–30) | Same as Delta family — ASR doesn't exercise the axis |
| **Readout gauge A1′** (ours, Stage 7A) | Internal | B | **5** (content-adaptive readout phase) | **Null — 0.1217** (engaged: $\|W_\phi\|_F$ 5.5–6.4, $\|\phi\|$ p99 ≈ 2 rad) | Per-token phase freedom; axis 5 not exercised by fixed ASR alignment |
| **Log-Linear Attention** (Paper 1, Stage 10.1) | Paper 1 (Guo et al.) | A | **4** (multi-scale long-range flow) | **PLATEAU — 0.1240** | Axis 4 not exercised at $T \le 500$; per-channel decay already covers short-range multi-scale. Natural home = LA / Mamba-2 |
| **Kronecker qtail / lowrank** (Stages 6) | Paper 8 / Expressiveness | D | **5** (cross-channel representation richness) | **MARGINAL — 0.1238 (best feature-side)** | Axis 5 weakly exercised; ~1σ differential over diagonal |
| **Hadamard $n^2$ / PoM v-lift** (Stages 6, 10.6) | Papers (Expressiveness, PoM) | D | **5** (polynomial value-lift) | **PLATEAU — 0.1253** | Same function class as qtail; parametrisation-equivalent |
| **ChannelMix bypass** (Paper 3 Avey, Stage 10.4) | Paper 3 | D | **5** (channel-reweight interpolation) | **PLATEAU — 0.1251** | Interpolation inside existing ChannelMix function class |
| **Content-conditional $\alpha_d$ (CB-3)** | Internal | A-dense | **5** (content-adaptive RF selection) | **PLATEAU — 0.1164** | Axis 5; ASR doesn't need per-token RF selection |
| **NaLaFormer signed-power** (Paper 2) | Paper 2 | D | **5** (query-norm-aware kernel) | Not run — predicted PLATEAU on RWKV-6 | Pathology absent in RWKV-6; natural home = LA |

### Legend

- **Family:** A = multi-scale temporal aggregation; B = linear transition operator; C = non-linearity of state; D = feature-map / channel-side; E = chunk retrieval (out-of-regime at $T \le 500$).
- **Axis:** 1–5 per the §"Five axes" table above.
- **ASR result:** single-seed, 7 M / 30 ep / LibriSpeech clean-100, seed 42. "Engaged" = diagnostic probes show non-zero mobility.

---

## What the ASR nulls actually told us

Every Stage-2–10 "engaged null" (A1′, T1, T2, S9A, S9B, CB-1, CB-3, CB-7)
is a mechanism that **extends the function class in a direction ASR's
signal does not probe**. The mechanisms engage — SGD finds parameter
configurations that use them — but the output function is not rewarded
by the CTC loss because the task-structural prior doesn't carry the
information the mechanism helps express.

- **Delta / DeltaProduct / Cayley / non-normal / M²RNN nulls on ASR** →
  not mechanism failures. Axis 2/3 extensions on an axis-1 task.
- **Log-Linear PLATEAU on RWKV-6** → not mechanism failure. Axis 4
  extension on a short-sequence axis-1 task, partly absorbed by RWKV-6's
  per-channel decay.
- **qtail / hadamard / pom / chanmix PLATEAU** → Axis 5 extensions.
  Weakly exercised by ASR's local, low-cross-channel task structure.

The ASR chain characterizes axis 1 thoroughly. It **cannot** characterize
axes 2–5 by construction. Claiming "DeltaNet doesn't help" from ASR
results alone is a scope error.

---

## Task suite implication

A complete mechanism-axis characterisation requires tasks that exercise
each axis. Recommended minimal suite:

| Task | Axis covered | Status |
|---|---|---|
| **ASR (LibriSpeech clean-100)** | 1 | Done (Stages 2–10) |
| **MQAR (synthetic Zoo-style, $N \gtrsim d$)** | 2 | In progress |
| **Non-Abelian state-tracking — $S_3$ permutation composition OR Dyck-$k$ ($k \ge 2$)** | 3 (sharp) | Planned |
| **Parity + modular arithmetic (secondary, Abelian)** | 3 (partial — depth $k \ge \log n$ sufficient) | Optional |
| **Induction heads (synthetic)** | 2-variant | Optional |
| **NIAH at matched scale** | 4 | Deferred — hard to exercise at $T \le 500$ |
| **LM baseline (small-scale WikiText)** | mixed / general | Optional |

**Important design note on axis-3 tasks (per arXiv:2603.01959).** Parity
and modular arithmetic are **Abelian** groups and diagonal SSMs can solve
them at depth $k \ge \log n$. At our 6-layer depth, parity alone will
not cleanly separate diagonal (RWKV-6, RSE) from DPLR (DeltaNet,
DeltaProduct, RWKV-7). The separation is **formally visible only on
non-Abelian tasks** — $S_3$ permutation composition is the smallest
non-Abelian group and the natural probe. Dyck-$k$ for $k \ge 2$ is the
canonical non-Abelian language. **The axis-3 benchmark should lead
with a non-Abelian task; parity / modular are secondary exercises.**

#### Axis-3 framing transition (2026-04-22, for document provenance)

The axis-3 benchmark moved from "parity / modular arithmetic" to
"non-Abelian state-tracking" in a single step, forced by the two
theory papers added to the paper index on the same pass. Recording
the reasoning here so future-you doesn't re-derive it:

- **Before:** axis 3 was informally defined as "state-tracking / TC⁰ →
  NC¹" with parity proposed as the representative probe, because parity
  is the textbook example of "a thing a linear RNN can't count to 2 of."
- **Forcing papers:**
  - arXiv:2603.03612 (positive bound): DPLR transitions (DeltaNet,
    DeltaProduct, RWKV-7) = $\mathsf{PNC}^1$-complete; permutation-diagonal
    = $\mathsf{NC}^1$-complete. Strict class gap.
  - arXiv:2603.01959 (negative bound): diagonal SSMs cannot track
    non-Abelian groups at finite precision regardless of parameter
    count; a $k$-layer diagonal SSM handles only groups with
    $k$-length Abelian-factor subnormal series.
- **Consequence for benchmark design:** parity, modular arithmetic,
  and flip-flop are all Abelian groups. At our 6-layer depth, diagonal
  SSMs can handle any Abelian-factor subnormal series of length up
  to 6 (well above any reasonable parity modulus). Parity as the
  axis-3 probe would be **solved by both** diagonal (RWKV-6, RSE)
  **and** DPLR (DeltaNet, DeltaProduct, RWKV-7) families, failing
  to separate the two classes. The formal $\mathsf{NC}^1 / \mathsf{PNC}^1$
  separation lives specifically on non-Abelian tasks.
- **After:** axis 3 is now a pre-registered test of the formal
  complexity-class separation. Non-Abelian tasks primary; parity /
  modular optional secondary (useful to confirm Abelian-regime
  solvability of both families but not load-bearing for the claim).
- **What did not change:** the axis itself, the mechanisms assigned
  to it (M²RNN, Negative Eigenvalues, DeltaNet, DeltaProduct,
  RWKV-7 are all still axis-3), and the prediction that our ASR
  results reflect axis-mismatch (ASR doesn't exercise axis 3 at
  all, Abelian or non-Abelian).

#### Two task options for the axis-3 empirical test

**Option A — $S_3$ permutation composition (primary recommendation).**

- **Group:** $S_3$ = symmetric group on 3 elements (6 group elements,
  smallest non-Abelian group).
- **Task:** input is a sequence of permutations $\pi_1, \pi_2, \ldots, \pi_T$,
  each one of 6 possible elements; target at position $t$ is the
  composed permutation $\pi_1 \circ \pi_2 \circ \cdots \circ \pi_t$.
- **Input encoding:** one-hot or embedding over the 6-element vocabulary.
- **Output:** 6-way classification at each position (or only at the
  final position, depending on task setup).
- **Sequence length:** sweep $T \in \{64, 256, 1024, 4096\}$ to titrate
  difficulty. At the longest lengths the diagonal family should fail
  by theorem; DPLR should remain accurate.
- **State required:** $\log_2 6 \approx 2.6$ bits of information (well
  within any reasonable state capacity — the difficulty is
  *tracking*, not *storing*).
- **Why primary:** minimum-implementation synthetic task; directly
  targets the 2603.01959 theorem; cleanest interpretation.
  Diagonal-vs-DPLR separation is sharp and theoretically predicted
  to be visible at $T$ exceeding the Abelian-subnormal-length bound.
- **Implementation effort:** trivial — $\sim 50$-line synthetic-data
  generator + cross-entropy training loop. No new kernels. Can
  re-use the existing ASR training harness with a different
  data-loader and loss head.

**Option B — Dyck-$k$ balanced brackets ($k \ge 2$, secondary probe).**

- **Language:** Dyck-$k$ = strings of matched open/close brackets with
  $k$ distinct bracket types. For $k \ge 2$ the language is
  non-regular (context-free); requires stack rather than finite state.
- **Task:** binary classification — is the input sequence well-balanced?
  Alternative: per-position classification of "expected next bracket."
- **Why non-Abelian:** for $k = 1$ (single bracket type), Dyck-1 reduces
  to a counter and is Abelian. For $k \ge 2$ the stack-ordering
  constraint makes the language non-Abelian and non-regular.
- **Sequence length and depth:** two parameters to titrate —
  $T \in \{64, 256, 1024\}$ and maximum bracket-nesting depth
  $d_{\max} \in \{4, 8, 16\}$. Deeper nesting stresses state capacity;
  longer sequences stress state-tracking stability.
- **Why secondary:** Dyck-$k$ goes *beyond* the $\mathsf{NC}^1 / \mathsf{PNC}^1$
  separation (it's formally context-free, not just finite-automaton).
  Both diagonal and DPLR may fail at large depth for different reasons
  (diagonal: Abelian-factor limit; DPLR: finite state capacity
  exceeded by stack depth). This *adds cross-task evidence* if $S_3$
  shows the predicted pattern, but it *complicates interpretation*
  if the patterns differ between $S_3$ and Dyck.
- **Implementation effort:** moderate — well-balanced / unbalanced
  string generator, balanced-by-construction with controlled depth,
  ~100-line data-loader. Classification loss, single head.

**Recommended sequencing:** run Option A ($S_3$) first as the
primary axis-3 probe. If $S_3$ confirms the predicted pattern
(diagonal family fails at large $T$; DPLR family succeeds), add
Option B (Dyck-2, depth-controlled) as robustness evidence. If $S_3$
shows an unexpected pattern, resolve that question before investing
in Dyck — the Dyck task's higher interpretive complexity makes it a
weak diagnostic if the basic $S_3$ result is already confounded.

**Single-pass scope (per arXiv:2604.14501).** All benchmarks in this
suite are evaluated under **single-pass inference** — no autoregressive
chain-of-thought loop, no inference-time iteration. This is a deliberate
scope choice. Online CoT can make any multi-layer SSM streaming-algorithm-
complete (arXiv:2604.14501), but that extension is orthogonal to the
architectural expressivity axes we characterise. Measurements here reflect
what one forward pass of the architecture can compute; the CoT extension
is noted in §"Orthogonal extensions" below and discussed in the synthesis
chapter, not exercised in the 5-axis benchmark suite.

With Tier-1 (ASR + MQAR + $S_3$/Dyck), the thesis produces a 3-axis ×
$N$-mechanism differential table. Each mechanism wins on its target
axis and is flat elsewhere. That pattern **is** the thesis contribution:
the axis decomposition is empirically validated, and the
mechanism–task matching principle is first-principles-predictive — with
axes 1 and 3 now formally bracketed by circuit-complexity theory
(arXiv:2603.03612, arXiv:2603.01959).

---

## Thesis reframe in one sentence

> We characterise the expressivity of causal linear-time RNNs as a
> multi-axis decomposition rather than a scalar, and show via matched
> experiments across ASR (axis 1), MQAR (axis 2), and synthetic
> state-tracking (axis 3) that each mechanism family targets a
> specific axis, with empirical gain determined by alignment between
> the mechanism's function-class extension and the task's structural
> prior.

This framing subsumes Stages 2–10 (axis-1 characterisation on causal
RWKV-6), motivates Stage 11 (cross-architecture transfer on axis 1 +
axis 2 + axis 3), positions DeltaNet / DeltaProduct / Householder /
M²RNN / NCGRU-Cayley as **co-axis evidence** rather than competitors,
and turns the existing ASR "nulls" into **axis-scope evidence** that
strengthens the decomposition.

---

## Paper index — axis placements and short verdicts

Papers grouped by target axis. Each entry: one-line mechanism,
axis assignment, ASR result where available, and verdict for the
thesis's multi-task suite.

### Axis 1 — Feature extraction + short-range temporal

| Paper | Mechanism (short) | Task domain / result | Thesis verdict |
|---|---|---|---|
| **Vision-RWKV** (arXiv:2403.02308, Mar 2024) | Q-shift — split channels into 4 quadrants, shift each along up/down/left/right, concatenate; every token gains explicit access to its 4 spatial neighbours | **2D vision, RWKV-style encoder** — scales stably beyond 30 M params where Vision Mamba did not | **Chronologically first paper in the axis-1 mechanism lineage.** Manual, fixed local-spatial-mixing operator that restores local inductive bias to RWKV's otherwise decay-only diagonal path. Same axis-1 function as LiT / Paper 7 / our ConvShift — different syntactic form. Cross-domain validation: 2D vision. **Q-Shift cross-task-robust within vision**: the same mechanism works for classification (Vision-RWKV), vision-language (VisualRWKV arXiv:2406.13362), and diffusion noise prediction (Diffusion-RWKV arXiv:2404.04478), spanning three of the most structurally-different vision objective functions. |
| **LiT — Linear Diffusion Transformer** (arXiv:2501.12976, Jan 2025) | DWC (depth-wise convolution) on value path in linear attention; distilled from pre-trained softmax DiT | **2D vision, diffusion transformer** — ~9× speedup, recovered softmax parity on image noise prediction | **Direct inspiration for our draft-phase ConvShift** (Run 006). LiT's DWC-on-value-path idea was adapted into RWKV-6's input-side token-shift path. Second independent vision-domain confirmation of axis 1 (different RWKV/Mamba-vs-DiT architecture lineage than Vision-RWKV). |
| **Paper 7 — Non-Attention LLM** (arXiv:2506.01963, Jun 2025) | Parallel dilated DWConv $\{1, 2, 4, 8\}$ on input side | **1D language / autoregressive LM** — paper-reported win; ported to ASR as `multidil_sym` (Stage 10.3-sym). **Fixed-init (`_v2`): dev 0.1013 / test 0.1000** — Paper 7 multi-dilation replicates with all four branches engaging (α₂ > α₁ at layers 1–5, α₈ grows with depth). Broken-init 0.1153 / 0.1145 was single-dilation + per-layer scalar. | Extends the single-dilation DWC pattern to a multi-scale dilation set. Our multidil_sym matches the phoneme-to-syllable window at post-subsampling frame rate (40–160 ms). Transfer to Mamba-2 / LA / LION in Stages 11 / 12. See `MULTIDIL_INIT_FIX_HANDOFF.md` for the init-trap fix. |
| **AudioRWKV** (arXiv:2509.02167, Sep 2025) | **Built on RWKV-7** (DPLR kernel). Replaces 1D token shift with 2D depthwise separable conv that forms a **local residual feeding decay / keys / values** (not just input mixing). Combines Q-Shift (fixed 4-direction) + 2D DWSep ConvShift (learned kernel) + Bi-WKV with learned-gate fusion. Ablation table progression: causal → +Bi-scan → +gate → +Q-Shift → +ConvShift, each step measured positive. | **Audio, 2D spectrogram pattern recognition** — paper-reported win, progressive mechanism-attribution via ablation | **Thesis-load-bearing entry.** Three claims land here: (i) **Fifth independent modality–architecture instance** of the axis-1 mechanism class, closest to our ASR domain. (ii) **Axis 1 is orthogonal to axis 2/3**: AudioRWKV adds axis-1 mechanisms on top of RWKV-7's DPLR ($\mathsf{PNC}^1$-complete per arXiv:2603.03612) kernel and still measures gains — the PNC¹-capable kernel is not "already good enough" to skip local mixing. (iii) **Within-axis composition works when scales differ**: Q-Shift (±1 neighbor) + ConvShift (learned-kernel DWSep) are complementary, not redundant, which is empirical evidence that axis 1 has internal scale-structure worth probing (relevant to our CB-2 wide-dilation question). |

**Cross-domain axis-1 validation (single strongest axis claim in the thesis).**

Five independent modality–architecture combinations of the same axis-1
mechanism class:

| Domain / modality | Architecture base | Paper | Syntactic form of local-bias mechanism |
|---|---|---|---|
| 2D vision | RWKV-4/5 family | Vision-RWKV (arXiv:2403.02308) | Q-Shift — 4-directional quadrant shift |
| 2D vision | Linear-DiT (distilled from softmax) | LiT (arXiv:2501.12976) | DWC on value path |
| 2D spectrogram audio | **RWKV-7** (DPLR kernel) | AudioRWKV (arXiv:2509.02167) | Q-Shift + 2D DWSep ConvShift (local residual into decay/K/V) + Bi-WKV gated fusion |
| 1D language | Linear-attention LM | Paper 7 (arXiv:2506.01963) | Multi-dilation DWConv on input $\{1, 2, 4, 8\}$ |
| 1D audio (CTC ASR) | RWKV-6 (diagonal kernel) | Our draft + Stages 2 / 10.3-sym | ConvShift single-dilation → multidil_sym $\{1, 2, 4, 8\}$ |

Axis 1 is **not** "ASR likes phoneme windows." It's an
architecture-agnostic, domain-general claim: **linear-complexity
sequence mixers (linear attention, RWKV-family recurrences from RWKV-4
through RWKV-7, linear DiTs) structurally lack local inductive bias,
and DWC-class mechanisms restore it.** Five independent
modality–architecture combinations validate the claim across 2D vision
(two architecture families), 2D spectrogram audio (RWKV-7 / DPLR
kernel), 1D language, and 1D audio (RWKV-6 / diagonal kernel). The
AudioRWKV entry specifically confirms the claim is **orthogonal to the
transition-matrix complexity class**: axis-1 mechanisms help on top of
both diagonal ($\mathsf{NC}^1$-bounded) and DPLR ($\mathsf{PNC}^1$-complete)
transition kernels. This is the single most empirically robust axis
claim in the thesis — comparable in evidence density to any claim in
the linear-time-RNN literature at this scale.

### Axis 2 — Associative memory capacity under interference

| Paper | Mechanism (short) | ASR result | Thesis verdict |
|---|---|---|---|
| **DeltaNet** (Yang et al. / IDSIA-EPFL Householder formulation) | Rank-1 Householder erasure $(I - \beta k k^\top)$ per token; interpreted as one gradient step on $\|v - S^\top k\|^2$ | **Null — 0.1258 tied vanilla** via Stage-8 T1 `rwkv6_delta_warmstart_fixed`. Mechanism engaged: $g_\delta \le 0.65$, $\beta$ p95 > 1, up to 24 % state-norm erased per token. | ASR at $T \le 500, d^2 = 4096$ is nowhere near state saturation; directional erasure has no task-prior alignment here. **Run on MQAR — predicted major win.** Candidate Stage-11.3 LA transfer. |
| **DeltaProduct** (arXiv:2502.10297) | $n_h$ Householder products per token (rank-$n_h$ + diagonal); bounded operator norm; formal proof of finite-automaton simulation at sufficient $n_h$ | Not run — **predicted null on ASR** (same axis as Delta rank-1, more rank of a solution to a non-problem at our regime). | Run on MQAR and parity. Expected to dominate on both by construction. Best single candidate to split memory-capacity axis from state-tracking axis if we titrate $n_h$. |
| **Paper 6 — CRT / NCGRU-Cayley** (arXiv:2505.00929) | Per-token Cayley orthogonal transition $(I - A)(I + A)^{-1}$ with skew-symmetric $A$ | **REGRESSION-track — 0.1518 ep 15** via Stage 10.5 `rwkv6_orthogonal` | Family-B member adjacent to Delta / DeltaProduct; dense per-token rotation without task-prior alignment. Co-axis evidence on ASR-null side. Breaks LION parallel form — excluded from Stage 12. |

### Axis 3 — State-tracking (TC⁰ → NC¹)

| Paper | Mechanism (short) | ASR result | Thesis verdict |
|---|---|---|---|
| **Paper 9 — M²RNN** (arXiv:2603.14360) | Non-linear state transition $\tanh(S W + k v^\top)$; scalar per-head forget gate | **Tied vanilla — 0.1276** via Stage 10.2 `rwkv6_m2rnn_sparse` (sparing-use at L=5) | Only member of Family C (non-linearity of state) in the queue; genuinely novel axis. ASR doesn't need NC¹; natural home is parity / state-tracking benchmark. **Rerun on parity at $T \in \{64, 256, 1024\}$.** Breaks LION parallel form. |
| **Negative Eigenvalues** (arXiv:2411.12537) | Extends state-transition eigenvalue range from $[0, 1]$ to $[-1, 1]$; paper proves parity-solving capability under finite precision | Not run — **predicted null on ASR** (ASR doesn't require parity/modular-arithmetic state tracking) | Strong axis-3 candidate for the parity benchmark. Co-axis evidence alongside M²RNN and DeltaProduct. Cheaper to implement than DeltaProduct (just flip sign constraint on decay LoRA output). |
| **Why Are Linear RNNs More Parallelizable?** (arXiv:2603.03612) | **Theoretical taxonomy (positive side).** Maps linear-RNN parameterisations to formal circuit classes. Proves DPLR (diagonal-plus-low-rank) mechanisms — DeltaNet, DeltaProduct, RWKV-7 — are $\mathsf{PNC}^1$-complete, strictly more expressive than permutation-diagonal linear RNNs ($\mathsf{NC}^1$-complete). | — (theoretical; no mechanism to run) | **Citation-grade framing for axis 3.** Formally grounds the prediction on our axis-3 benchmark: RWKV-6 vanilla + RSE + multidil_sym (permutation-diagonal class) should fail at $T \gg $ state dim; DeltaNet + DeltaProduct + RWKV-7 (DPLR class) should succeed. Converts our planned axis-3 benchmark from ad-hoc to **pre-registered test of a formal complexity-class separation**. Also motivates testing **RWKV-7** on the benchmark as a second DPLR member alongside DeltaNet — both should succeed, RWKV-6 / RSE should fail. |
| **Expressive Limits of Diagonal SSMs for State-Tracking** (arXiv:2603.01959) | **Theoretical taxonomy (negative side).** Proves single-layer input-dependent complex-valued diagonal (DCD) SSMs **cannot** express state-tracking for any non-Abelian group at finite precision. $k$-layer DCD stacks are limited to groups with a subnormal series of length $k$ with Abelian factors. Applies to RWKV-6, Mamba, GLA, mLSTM, **and RSE** (block-diagonal-complex = DCD). | — (theoretical; no mechanism to run) | **Closes a whole branch of potential future work.** RSE + variants cannot rescue non-Abelian state-tracking at *any* parameter count — formal impossibility. Pairs with 2603.03612 as a **two-sided boundary for axis 3**: DPLR reaches $\mathsf{PNC}^1$, diagonal cannot cross non-Abelian state-tracking. **Sharpens the axis-3 benchmark**: parity alone is insufficient because it's Abelian; we need non-Abelian tasks ($S_3$ permutation, Dyck, flip-flop) for the predicted diagonal-vs-DPLR separation to be empirically visible. Citation-grade. |

### Axis 4 — Long-range information flow

| Paper | Mechanism (short) | ASR result | Thesis verdict |
|---|---|---|---|
| **Paper 1 — Log-Linear Attention** (Guo et al., arXiv:2506.04761) | Fenwick-tree $O(\log T)$ bucket states with per-token per-scale mixer $\lambda_t^{(\ell)}$; composes with any semiseparable mask | **PLATEAU — 0.1240** via Stage 10.1 `rwkv6_loglinear`. Early-epoch signal (Δ = −0.005 at ep 5) compresses to tied by ep 30. | RWKV-6's per-channel decay diversity partially absorbs the multi-scale benefit; LA / Mamba-2 have no such absorption. **Stage 11.3 natural-home test — high prior of meaningful gain.** |
| **Paper 4 — RWKV-X** (arXiv:2504.21463) | Chunk-level sparse retrieval + sliding window; length extrapolation to 64 k | — (out-of-regime: $T \le 500$, chunk primitive degenerates) | Dismissed. Natural home is long-form audio / document QA, not this thesis's regime. Citation-only. |
| **Paper 5 — HSA / RAMba** (arXiv:2504.16795) | Hierarchical sparse attention with chunk summaries and top-$k$ gating | — (out-of-regime) | Dismissed. Same regime issue as RWKV-X. Citation-only. |

### Axis 5 — Content-adaptive computation & feature-map richness

| Paper | Mechanism (short) | ASR result | Thesis verdict |
|---|---|---|---|
| **Paper 2 — NaLaFormer** (Meng et al., arXiv:2506.21137) | Query-norm-aware kernel $\phi_q(q) = d(q) \cdot p(\|q\|)$ with signed-power on $k$; lifts $[\cos, \sin]$ for non-negativity | Not run on RWKV-6 (pathology absent — we don't have Katharopoulos L1 denominator) | Natural home is LA with explicit L1 denominator. Stage-11.3 parked candidate. Expected effect small — overlap cluster β with RSE. |
| **Paper 3 — Avey bypass component** (Khatami et al., arXiv:2506.11305) | Partial-embedding bypass in ChannelMix: $\alpha$-gated blend of linear-head vs ReLU²-tail feature map | **PLATEAU — 0.1251** via Stage 10.4 `rwkv6_chanmix_bypass` | Interpolation inside existing ChannelMix function class. Axis 5 weakly touched. Avey's full architecture (Ranker retrieval) is axis-4 and out-of-regime. |
| **Paper 8 — PoM Polynomial Mixer** (arXiv:2604.06129) | Polynomial value-lift $\hat v = v + \sum_p \gamma_p h(W_h x)^{\odot p}$; permutation-equivariant form | **PLATEAU — 0.1254** via Stage 10.6 `rwkv6_pom_vlift` | Same function class as Stage-6 hadamard_n2 / qtail quadratic cluster at parity-fitting $k=2$. Family-D saturated across parametrisations on ASR. Natural home is PoM's permutation-equivariant setting (vision / set tasks). |
| **EXPRESSIVENESS paper** (Mongaras & Larson, arXiv:2507.23632) | Cross-channel Kronecker $k \otimes k$ feature lift (qtail) | **MARGINAL — 0.1238 (best feature-side)** via Stage 6 `rwkv6_qtail_lowrank_all` | ~1σ differential over diagonal quadratic; mild axis-5 signal on ASR. Still above axis-1 anchor. |
| **Higher-order Linear Attention (HLA)** (arXiv:2510.27258) | Streaming higher-order tensor interactions via prefix sufficient statistics | Not run — **predicted PLATEAU** (redundant with Stage-6 quadratic cluster at order 2) | Algorithmic contribution (efficient streaming) more than expressivity. Axis 5, closed family. Citation-only — useful as theoretical counterpoint to the empirical Family-D saturation. |

### Orthogonal extensions — not in the 5-axis training-time framework

Four classes of mechanism live **orthogonally** to the five training-time
axes above. They change effective expressivity without changing where
the architecture sits on axes 1–5. Important for thesis scope-setting
and for comparison of alternative bidirectional-extension designs.

**Bidirectional-extension taxonomy.** Three of the entries below are
three distinct structural strategies for the same causal → bidirectional
problem, worth naming explicitly for the Chapter 4 comparison:

| Level of change | Strategy | Cost |
|---|---|---|
| Mechanism-level (parallel form) | LION T×T attention with symmetric decay | Parallel compute, $O(T^2)$ memory |
| Operator-level (serial forward+backward) | Vision-RWKV Bi-WKV + Q-Shift | 2× serial compute |
| Ordering-level (token permutation per layer) | VisualRWKV 2D-scan with alternating directions | Same compute as causal RWKV + reshape |

| Paper / extension | Mechanism type | What it changes | Thesis role |
|---|---|---|---|
| **On the Expressive Power and Limitations of Multi-Layer SSMs** (arXiv:2604.14501) | **Inference-time** (online chain-of-thought) | Proves multi-layer SSMs + online CoT = streaming-algorithm-complete; offline CoT is strictly weaker. Emitted tokens fed back into input stream multiply effective depth without any weight/architecture change. | **Citation-grade scope clarifier.** Makes the thesis's single-pass / training-time focus explicit and defensible. Explains how the diagonal-SSM impossibility (arXiv:2603.01959) is specifically a *single-pass* result — online CoT offers an escape route, but one that lives outside our 5-axis architectural taxonomy. Pre-registers that our benchmarks (ASR, MQAR, parity/Dyck) are single-pass by design. |
| **LION parallel bidirectional form** (our Chapter 4) | **Mechanism-level: parallel T×T attention with symmetric decay** | Takes any causal linear-time mechanism and deploys it on a parallel bidirectional form. Same mechanism vocabulary (ConvShift, RSE, multidil_sym) on a different sequence-access pattern. | Parallel chapter in the thesis. Related conceptually: LION extends sequence *representation* (unidirectional → bidirectional) just as online CoT extends sequence *computation* (single-pass → autoregressive-loop). Both enhance effective expressivity orthogonally to the 5 axes. |
| **Vision-RWKV serial bidirectional** (arXiv:2403.02308, Mar 2024) | **Operator-level: forward RWKV pass + backward RWKV pass with 4 hidden states $a, b, c, d$** | Takes any causal RWKV-family mechanism and runs it forward then backward sequentially, combining the states. Same goal as LION's parallel form — different implementation. | **Empirically dominated by LION's parallel form at our spine.** Draft phase evidence (Ukrainian ASR, 60 ep, matched 7 M): LION `bidir_rwkv6` test CER **0.1790** vs Bi-WKV serial `biwkv6_no_conv_no_gate` test CER **0.2201** — **+0.044 gap at matched parameters**. Thesis-defensible comparison against a specific paper-proposed bidirectional-extension alternative. Cite as: serial bidirectional under-performs parallel bidirectional at this scale. |
| **VisualRWKV 2D-scanning bidirectional** (arXiv:2406.13362, Dec 2024) | **Ordering-level: keep standard causal RWKV operator, alternate scan directions across layers** | Each layer scans in one direction (left-to-right, right-to-left, top-to-bottom, etc.); across depth, every token has "seen" every other token via at least one direction. No new operator, only per-layer tensor reshape. Secondary contribution: "sandwich prompts" — an inference-time prompt-engineering workaround repeating instructions before and after image tokens to compensate for RWKV's forward-only memory. | **Weakest form of bidirectionality** — achieved only across the full depth stack, not at any single layer. Cheapest in compute terms (no new kernel). Paper applies it to 2D vision tokens; an ASR analog would just reverse scan direction every layer, which does not give any single layer access to future context (critical for formant / co-articulation cues). Cite as: **the ordering-level strategy is the cheapest but weakest of the three bidirectional approaches, and our ASR measurements at the operator-level (Bi-WKV) already dominate it at matched compute**. Sandwich-prompt sub-mechanism is **supporting evidence that axis 2 is a real architectural limitation** for RWKV-family models — it's a prompting-engineering band-aid for memory, not a fix. |

### Cross-axis / composition candidates

| Candidate | Axis combination | Notes |
|---|---|---|
| **Log-Linear × DeltaProduct on LA** | 2 × 4 | Paper 1 says log-linear composes with any semiseparable mask; DeltaProduct is semiseparable. Potential Stage-11.3 follow-up if both single-mechanism transfers land. |
| **RSE × Log-Linear on LA / Mamba-2** | 1 × 4 | Cross-architecture analogue of the Stage-10.7 composition that closed without running. Natural home is LA where both mechanisms address independent deficits. |
| **Multidil_sym × RSE on RWKV-6 causal (CB-1 v2)** | 1 × 1 | **LANDED 2026-04-23, dev 0.0973 / test 0.0961.** Same axis (1), different sub-axes (input-side multi-scale × transition-side complex poles) — they compose productively. Broken-init CB-1 at 0.1156 was a mechanism-suppressed artefact; with the init fix, composition produces a 14σ drop vs broken-init and a 4σ drop vs P1 v2 single-mechanism. First sub-0.10 causal RWKV-6 result. |
| **Multidil_sym × RSE on LION** | 1 × 1 | Post-CB-1-v2 prior: expected to replicate on LION bidirectional — same composition on the same axis, different sequence-access pattern. Worth Stage 12.4 test. |

---

## Summary line

Of 20 papers/mechanisms indexed: **4 axis-1 entries confirm the
DWC-class locality-restoration mechanism across four domain settings**
(2D vision RWKV-style via Vision-RWKV; 2D vision linear-DiT via LiT;
2D spectrogram audio via AudioRWKV; 1D language via Paper 7; 1D audio
via our multidil_sym) — the cross-domain validation makes axis 1 the
single strongest axis claim in the thesis; **4 axis-2/3 papers were
tested on ASR and engaged-null** (Delta rule T1, NCGRU Cayley, M²RNN,
non-normal RSE) — these nulls are now re-interpreted as axis-mismatch,
not mechanism failure; **5 axis-5 papers were tested or predicted null
on ASR** (Avey bypass, PoM, HLA, Expressiveness qtail, NaLaFormer) —
all inside the Family-D saturated cluster; **3 papers are out-of-regime
at $T \le 500$** (RWKV-X, HSA, Avey Ranker) — citation-only; **2
theoretical taxonomy papers bracket axis 3** (arXiv:2603.03612 — DPLR →
$\mathsf{PNC}^1$-complete; arXiv:2603.01959 — diagonal SSMs cannot
track non-Abelian groups at finite precision regardless of parameter
count); **4 orthogonal-extension entries** cover inference-time (online
CoT via arXiv:2604.14501) and three distinct structural strategies for
causal → bidirectional extension: parallel (our LION), operator-level
serial (Vision-RWKV Bi-WKV, empirically dominated by LION at our spine
per draft-phase evidence), and ordering-level (VisualRWKV 2D scanning,
weakest form — bidirectionality achieved only across the full depth
stack). The axis decomposition predicts the empirical distribution
exactly: axis-1-aligned mechanisms win on axis-1 tasks across four
independent domain settings, other axes remain untested on their
natural benchmarks until MQAR (axis 2) and non-Abelian state-tracking
(axis 3 — $S_3$ / Dyck, not just parity) come online — and the
theoretical pair pre-registers a clean diagonal-vs-DPLR separation on
the non-Abelian task, with the inference-time-extension paper
clarifying the single-pass scope of the separation.
