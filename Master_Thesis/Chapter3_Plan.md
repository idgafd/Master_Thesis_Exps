# Chapter 3 — Theoretical Background — Working Plan

*Working planning document for the Theoretical Background chapter.
Not part of the LaTeX deliverable. Records section-by-section scope,
formulas to introduce, citations to use, and the validation status of
each formula against the source paper that introduced it.*

*Created 2026-04-29. Iteratively updated during the formula-validation
pass.*

> **Status (2026-04-30):** Chapter 3 has been drafted from this plan;
> the LaTeX file lives at `Master_Thesis/chapters/3_theoretical_background.tex`.
> All formulas were validated against e-print sources before drafting;
> no projection caveats apply to this chapter (it predates the
> empirical matrix).

---

## What this chapter does in the thesis

Chapter 3 is the **mathematical foundation** layer. It establishes the
axis decomposition framework (§3.1), formalises the three causal
linear-time backbones in a unified recurrence template (§3.2),
introduces the bidirectional adaptation via LION (§3.3), presents the
three mathematical primitives that Chapter 4's proposed mechanisms
will build on (§3.4), and brackets the function-class scope of the
diagonal family with formal complexity-class bounds (§3.5).

It does **not** present any of the proposed mechanisms (MSDC, CVD,
DHO) as our contributions — those live in Chapter 4. Chapter 3
keeps each primitive in its existing literature form.

The chapter operates entirely within the diagonal-class function
family. Diagonal-plus-low-rank (DPLR) mechanisms are referenced as
the formal class boundary, not exercised.

---

## 3.1 Axis decomposition of linear-time RNN expressivity

### Purpose

Frame the entire chapter (and the thesis) around an axis decomposition.
Define three primary axes that the thesis will probe (1, 2, 3) plus a
brief mention of two further axes that the thesis cites but does not
lead on (4, 5). Each axis is presented with its diagnostic property and
the task structure that exercises it.

### What this section explains

- Linear-time RNN expressivity does not collapse to a single scalar.
  It decomposes into orthogonal axes, each exercised by a different
  task structure.
- Three primary axes for this thesis:
  1. **Axis 1.** Short-range temporal hierarchy. Structured local
     mixing across small windows. Probed by ASR.
  2. **Axis 2.** Associative-memory capacity under interference.
     Recall when stored pairs approach state dimension. Probed by
     MQAR.
  3. **Axis 3.** State-tracking / non-Abelian permutation. Bracketed
     by formal circuit-complexity theory. Out of scope for the
     diagonal class; pre-registered as future work.
- Two further axes mentioned briefly:
  4. **Axis 4.** Long-range information flow at sequence length far
     beyond the chunk-local window. Out of scope for $T \le 500$.
  5. **Axis 5.** Content-adaptive feature-map richness and
     channel-side computation. Saturates to a narrow band on tasks
     that do not reward dense per-channel freedom.
- One-line position: *the axis a task exercises determines whether
  a mechanism's function-class extension converts into measured
  gain.*

### Math content (light)

No heavy formulas. One axis-vs-diagnostic-property table. Optional
short note that axes 1 and 2 are largely decorrelated empirically
on our matrix (single-mechanism on axis 1 fails MQAR; single-mechanism
on axis 2 fails ASR axis-1).

### Citations

| Concept | Bib key |
|---|---|
| MQAR / Zoology (axis 2 instrument) | `arora2023zoology` |
| Diagonal-SSM in $\mathsf{TC}^0$ (axis 3 negative bound) | `merrill2024illusion` |
| Formal language analysis of diagonal SSMs (axis 3 closure) | `sarrof2024expressive` |
| Log-linear / long-range mechanisms (axis 4 placeholder) | `guo2025loglinear` |
| Polynomial value-lift family (axis 5 placeholder) | `mongaras2025expressiveness` |

### Anti-scope

Do **not** introduce the three proposed mechanisms here. Do not
present empirical results. Do not enumerate all five axes
exhaustively if axis 4 or 5 needs more than two sentences.

### Validation status

Conceptual section. No formulas to validate against source papers.

---

## 3.2 Linear-time recurrence: unified formulation and three backbones

### Purpose

Introduce the three causal linear-time backbones (RWKV-6, Mamba-2,
Linear Attention) in a single unified recurrence template. Identify
the structural axis on which they differ — primarily the transition
operator $A_t$ and the strength of content-adaptive routing — and map
those structural differences to architecture-deficit on the axes from
§3.1.

### What this section explains

- The shared linear-time recurrence template:
  $S_t = A_t \odot S_{t-1} + B_t \otimes V_t$,
  $y_t = C_t^\top S_t$ (placeholder form; final notation locked
  after the validation pass).
- Three instantiations, each differing in $A_t$:
  1. **RWKV-6.** Per-channel data-dependent decay (WKV). Matrix-
     valued state, per-channel exponential decay modulated by token
     content via low-rank projections.
  2. **Mamba-2.** Selective continuous-time SSM with input-modulated
     step size $\Delta_t$. The transition is $\bar A_t = f(\Delta_t, A)$,
     formalised below.
  3. **Linear Attention.** $A_t = I$ (no decay), pure accumulator.
     Recurrence reduces to $S_t = S_{t-1} + \phi(k_t) v_t^\top$.
- Architecture-deficit map:
  - Linear Attention has no decay (axis-1 sub-deficit) and no
    interference cancellation (axis 2 deficit) — leaves both bare.
  - RWKV-6 has decay diversity on axis 1 sub-axis but no interference
    cancellation on axis 2.
  - Mamba-2 has both decay diversity (via $\Delta_t \cdot A$) and
    selectivity on $B_t, C_t$, leaving the smallest residual deficit.

### Math content (heavy)

Four equations to introduce:

#### F-3.2.1 — Generic linear-time recurrence (synthesised template)

```latex
h_t = A_t \odot h_{t-1} + B_t \, x_t,
\qquad y_t = C_t^\top h_t,
```

with $h_t$ the hidden state, $A_t$ the transition operator
(elementwise / diagonal-structured), $B_t x_t$ the input write,
$C_t$ the readout direction. Notation deliberately matches Sarrof,
Veitsman, Hahn 2024 eq:recurrence-ssm and Mamba-2 eq:s6 (post-bar
convention; see §3.2.4 note).

[**verified ✓**]: this template subsumes all three backbones —
RWKV-6 with $A_t = \text{diag}(w_t)$ and rank-1 outer-product write
$k_t^\top v_t$ in the matrix-state setting, Mamba-2 with $A_t$
scalar (SSD restriction) per Dao & Gu 2024, and Linear Attention
with $A_t = I$ per Katharopoulos et al. 2020.

#### F-3.2.2 — Linear Attention recurrence

```latex
S_i = S_{i-1} + \phi(K_i) V_i^\top, \qquad
z_i = z_{i-1} + \phi(K_i), \qquad
V'_i = \frac{\phi(Q_i)^\top S_i}{\phi(Q_i)^\top z_i}
```

with $\phi(\cdot)$ the (positive) feature map.

[**verified ✓** against Katharopoulos et al. 2020 (arXiv:2006.16236)
eq:attn_state line 382 and eq:trnn-3 line 468]. Notation matches
the paper (uppercase $K, V, Q$ and subscript $i$). Final writeup
may switch to lowercase $k, v, q$ + subscript $t$ for cross-backbone
consistency in §3.2.

#### F-3.2.3 — RWKV-6 (Finch) WKV recurrence

[**MAJOR CORRECTION**: original draft used the RWKV-4 softmax-
normalised form. Finch uses a matrix-valued state with diagonal
decay and rank-1 outer-product update; no softmax-style denominator.]

Canonical Finch Time Mixing form (eq:time-mixer6):

```latex
s_t = \mathrm{diag}(w_t) \cdot s_{t-1} + k_t^\top v_t,
\qquad
\mathrm{wkv}_t = s_{t-1} + \mathrm{diag}(u) \cdot k_t^\top v_t
```

with $s_t \in \mathbb{R}^{(D/h) \times (D/h)}$ the per-head matrix-
valued state, $w_t \in \mathbb{R}^{D/h}$ a per-channel
data-dependent decay vector, $u \in \mathbb{R}^{D/h}$ a static bonus,
and $k_t^\top v_t$ the rank-1 outer product. The data-dependent decay
$w_t$ is produced via:

```latex
d_t = \mathrm{lora}_d(\mathrm{ddlerp}_d(x_t, x_{t-1})), \qquad
w_t = \exp(-\exp(d_t))
```

with $\mathrm{lora}_\square(x) = \lambda_\square + \tanh(x A_\square) B_\square$
and $\mathrm{ddlerp}_\square(a, b) = a + (b-a) \odot \mathrm{lora}_\square(a + (b-a) \odot \mu_x)$.

[**verified ✓** against Peng et al. 2024 (Eagle and Finch),
arXiv:2404.05892, eq:time-mixer6 line 587–591 and eq:token-shift6
lines 552–580.] Note the recurrence order: $\mathrm{wkv}_t$ is read
out using the *previous* state $s_{t-1}$ plus the bonus term on the
current $k_t^\top v_t$; the state then updates to $s_t$.

#### F-3.2.4 — Mamba-2 SSD discrete recurrence

[**NOTATION CORRECTION**: the Mamba-2 paper explicitly drops the
bar notation. Quote from background.tex line 67–68: "prior work
referred to ... as $(A, B)$ and $(\bar A, \bar B)$ instead; we have
changed notation to simplify the presentation and focus directly
on the discrete parameters". Use $A_t$ throughout, not $\bar A_t$.]

Selective S6 recurrence (eq:s6):

```latex
h_t = A_t h_{t-1} + B_t x_t, \qquad
y_t = C_t^\top h_t
```

SSD restriction (ssd.tex line 27): the $A$ matrices are
*extremely* structured — $A_t = a_t I$ for scalar $a_t$ and
identity matrix $I$. Substituting:

```latex
h_t = a_t \cdot h_{t-1} + B_t x_t, \qquad
y_t = C_t^\top h_t, \qquad a_t \in \mathbb{R}
```

Continuous-to-discrete bridge (ZOH, inherited from Mamba — see
F-3.2.5): $a_t = \exp(\Delta_t \mathring a)$ where $\mathring a$
is the underlying continuous parameter. In Mamba-2 specifically,
this bridge is mentioned but not the primary form; SSD presents
the recurrence directly in discrete coordinates.

[**verified ✓** against Dao & Gu 2024 (arXiv:2405.21060),
background.tex eq:ssm/eq:s6 line 21–39 (general discrete SSM);
ssd.tex line 26–27 (scalar-identity restriction).]

#### F-3.2.5 — Mamba selective $\Delta_t$ (referenced)

The continuous-to-discrete bridge from selective Mamba.

Selective $\Delta_t$ parameterisation:

```latex
\Delta_t = \tau_\Delta(\mathrm{Parameter} + s_\Delta(x_t)), \qquad
s_\Delta(x) = \mathrm{Broadcast}_D(\mathrm{Linear}_1(x)), \qquad
\tau_\Delta = \mathrm{softplus}
```

ZOH discretisation (eq:zoh):

```latex
\bar A_t = \exp(\Delta_t A), \qquad
\bar B_t = (\Delta_t A)^{-1} (\exp(\Delta_t A) - I) \cdot \Delta_t B_t
```

with $A, B_t, C_t, \Delta_t$ the continuous parameters; the
selective S6 algorithm makes $B_t, C_t$ input-dependent
($s_B(x), s_C(x)$ via $\mathrm{Linear}_N$) and $\Delta_t$
input-dependent as above.

[**verified ✓** against Gu & Dao 2023 (arXiv:2312.00752),
method.tex lines 72–75 (selective parameterisation) and
background.tex line 71–76 (ZOH).] Note that the Mamba paper retains
bar notation $\bar A, \bar B$, while Mamba-2 (F-3.2.4) drops it.
For consistency in our writeup we adopt the Mamba-2 convention
and present the discrete form directly.

### Citations

| Concept | Bib key |
|---|---|
| Linear Attention (Katharopoulos) | `katharopoulos2020transformers` |
| Mamba (selective Δt) | `gu2023mamba` |
| Mamba-2 / SSD | `dao2024transformers` |
| S4 / structured SSM lineage | `gu2022s4` |
| RWKV-4 (background) | `peng2023rwkv` |
| RWKV-5/6 (Eagle and Finch) | `peng2024eaglefinch` |
| RWKV-7 (mentioned, not used) | `peng2025rwkv7` |

### Anti-scope

- Do not introduce LION here (§3.3).
- Do not introduce mechanism extensions here (§3.4).
- Do not discuss Mamba-3 — explicitly out of scope per
  `lahoti2026mamba3` citation in §2.1, no formulas reused.
- Do not derive the SSM continuous-discretisation bridge in full —
  reference Gu & Dao 2023 for the derivation, present only the
  discrete form we use.

### Validation status

Five formulas validated against source papers on 2026-04-29:
- F-3.2.1 (generic template) — verified ✓ (synthesised, subsumes all three).
- F-3.2.2 (LA) — verified ✓ against Katharopoulos 2020.
- F-3.2.3 (RWKV-6 Finch) — **major correction applied** (was RWKV-4-style,
  now matrix-state Finch form).
- F-3.2.4 (Mamba-2 SSD) — notation correction applied (bar dropped).
- F-3.2.5 (Mamba selective Δt) — verified ✓ with refinement
  (Parameter + s_Δ baseline term).

---

## 3.3 Bidirectional adaptation: the parallel form and decay taxonomy

### Purpose

Establish the math of LION's parallel bidirectional form, demonstrate
the mathematical correspondence between causal recurrence and parallel
attention under symmetric decay masks, and introduce the decay-class
taxonomy (LION-LIT vs LION-S) that produces the natural-experiment
falsifier on Linear Attention in Chapter 4.

### What this section explains

- Causal recurrence as a kernel sum: $y_t = \sum_{s \le t} \alpha_{ts} v_s$
  for a decay-modulated kernel $\alpha_{ts}$.
- LION's parallel form replaces causal masking with a symmetric decay
  mask $M_{ts} = M_{st}$ and computes the full $T \times T$ attention
  in a single matrix multiplication.
- For backbones with content-dependent per-channel decay (RWKV-6,
  Mamba-2), the LION form inherits the decay structure naturally —
  this is **LION-S**.
- For Linear Attention, the natural mapping in Afzal et al. 2025
  Table 1 is **LION-LIT** (no decay, $\lambda = 1$). Without decay,
  the parallel $\phi(K)^\top$ row-sum scales as $O(T \cdot d)$ instead
  of being decay-bounded — bidirectional accumulation becomes
  unbounded as $T$ grows.
- The decay taxonomy is therefore not an implementation detail but a
  structural prerequisite: bidirectional value-decorrelation
  mechanisms (§3.4.3) require the underlying form to be decay-bounded.

### Math content

#### F-3.3.1 — LION Full Linear Attention (matrix form)

Canonical form (eq:attvecbid line 14–17):

```latex
\mathbf{Y} = \mathrm{Scale}(\mathbf{Q} \mathbf{K}^\top \odot \mathbf{M}) \mathbf{V}
```

The bidirectional decay mask $\mathbf{M}$ (eq:maskselbir line 28–37)
is symmetric and rank-one semi-separable, with selective form:

```latex
M_{ij} = \begin{cases}
\prod_{k=j}^{i-1} \lambda_k & i > j \\
1 & i = j \\
\prod_{k=i+1}^{j} \lambda_k & i < j
\end{cases}
```

For the causal counterpart (eq:lintransuni elsewhere in the paper),
the lower-triangular slice of $\mathbf{M}$ is used. Three concrete
mask choices appear in the paper:
- **Selective** (per-token data-dependent decay): $\lambda_k$ varies.
- **Learnable fixed decay**: $M_{ij} = \lambda^{|i-j|}$ for a single
  scalar $\lambda$ (KMS structure).
- **All-ones**: $M_{ij} = 1$ — vanilla bidirectional Linear Transformer.

[**verified ✓** against Afzal et al. 2025 (arXiv:2502.16249),
3-LION.tex lines 14–37 and lines 75–84 for the rank-1 semi-separable
construction.]

#### F-3.3.2 — Decay-class taxonomy (LION variants)

Per Afzal et al. 2025 (3-LION.tex lines 286–289), the LION framework
defines specific named variants by the structure of $\lambda_i$:

- **LION-LIT** ($\lambda_i = 1$): bidirectional form of the vanilla
  Linear Transformer (Katharopoulos et al. 2020). No decay.
- **LION-S** ($\lambda_i = \sigma(W x_i + b)$): bidirectional
  extension of Gated Random Feature Attention (Peng 2021), with
  shifted SiLU activation. The paper explicitly states this is
  "inspired by the selectivity of Mamba2".
- **Learnable fixed decay** (e.g. LION-RetNet variant): single
  scalar $\lambda$ shared across positions, KMS-matrix structure.

[**verified ✓** against Afzal et al. 2025 (arXiv:2502.16249),
3-LION.tex lines 287–289 (variant definitions) and the variant
table covering RetNet, GRFA mappings.]

For our matrix:
- RWKV-6 LION → LION-S (inherits per-channel data-dep WKV decay).
- Mamba-2 LION → LION-S (inherits per-head selective Δt·A decay).
- Linear Attention LION → LION-LIT is the table-default
  (Katharopoulos has $\lambda = 1$). We **also** report LION-S
  as a controlled variant on LA, isolating decay as a structural
  prerequisite for the Chunked Value Decorrelation mechanism.

### Citations

| Concept | Bib key |
|---|---|
| LION parallel form | `afzal2025linear` |
| Vision-RWKV bidirectional Bi-WKV (operator-level) | `duan2024visionrwkv` |
| Vision Mamba bidirectional pairing | `zhu2024visionmamba` |
| Comparison of bidirectional strategies | (covered in §2.2 already) |

### Anti-scope

- Do not introduce ordering-level bidirectional strategies (VMamba,
  VisualRWKV, ZigMa) — those were covered in §2.2. Just reference.
- Do not propose any specific mechanism's LION wrapper here — that is
  Chapter 4.

### Validation status

Two formulas verified ✓ on 2026-04-29 against Afzal et al. 2025
(arXiv:2502.16249). F-3.3.1 verified against eq:attvecbid /
eq:maskselbir; F-3.3.2 corrected from "per-head σ-decay" to the
exact LION-paper formulation $\lambda_i = \sigma(W x_i + b)$.

---

## 3.4 Mathematical primitives for mechanism design

### Purpose

**This section is the mathematical workhorse of the chapter.** It
introduces three families of mathematical primitives — input-side
multi-scale convolution, block-complex transitions with Rayleigh
dissipation, and key-similarity preconditioning — in their existing
literature form. Each primitive is the foundation on which Chapter 4
will build a proposed mechanism (MSDC, DHO, CVD respectively).

The section presents each primitive as prior art. Our contributions
in Chapter 4 will be: the specific instantiation (MSDC's dilation
schedule and init-fix), the architecture-specific deployment (CVD's
chunked-recurrent / parallel / SSD-chunk-local forms across three
backbones), and the physical-prior derivation (DHO's Rayleigh
viscosity coupling on top of block-complex transitions).

### 3.4.1 Multi-scale depthwise dilated convolution

#### What it is

A parallel bank of depthwise 1D convolutions with dilations
$\{1, 2, 4, 8\}$ and kernel size $k = 3$, applied to the input
side. Each branch covers a different receptive field; learnable
mixing weights aggregate them.

#### Math — present a *family* of mechanisms, not one formula

[**REFRAMING NEEDED**: original draft listed a single formula
specific to our MSDC. The literature offers a *family* of input-side
local-mixing operators that differ in syntactic form but share a
common functional pattern. Present the family, defer our specific
choice (dilation set $\{1, 2, 4, 8\}$, $k=3$, depthwise, learnable
$\alpha_d$, init-fix) to Chapter 4.]

The family of input-side local-mixing mechanisms in linear-time
RNN literature has four representative instances:

1. **Q-Shift (Vision-RWKV, eq:token_interpolation):**

   ```latex
   X^\dagger[h, w] = \mathrm{Concat}(X[h-1, w, 0:C/4], X[h+1, w, C/4:C/2],
                                     X[h, w-1, C/2:3C/4], X[h, w+1, 3C/4:C])
   \mathrm{Q\text{-}Shift}(X) = X + (1 - \mu) X^\dagger
   ```

   Channels split into 4 groups, each shifted by $\pm 1$ in one of
   {up, down, left, right}. **Fixed** mechanism, no learnable
   per-branch weights.

2. **Depthwise convolution on V (LiT):**

   ```latex
   V \leftarrow \mathrm{DWConv}_k(V)
   ```

   Single depthwise conv kernel applied to the value path of linear
   attention. Learnable conv weights, single dilation.

3. **Multi-resolution dilated convolution (Kiruluta et al. 2025):**

   ```latex
   Z_k = \mathrm{Conv1d}(x, \mathrm{dilation} = d_k), \quad k \in \{1, \ldots, K\}
   Z_\mathrm{Conv} = \mathrm{Combine}(Z_1, Z_2, \ldots, Z_K)
   ```

   Parallel dilated convolutions with "dilation factors typically
   powers of 2, such as 1, 2, 4, and so on" (Kiruluta et al. line
   214). Combination "typically through summation or concatenation"
   (line 144) — no specific learnable mixing prescribed.

4. **Two-dimensional depthwise-separable ConvShift (AudioRWKV):**

   2D DWSep conv "instantiated via a simpler Q-Shift variant"
   (AudioRWKV line 271), inspired by Vision-RWKV. Builds a 2D local
   residual fed into decay/key/value paths of RWKV-7.

The common functional pattern across the four: **input-side local
mixing on top of the linear-time core**, applied either as fixed
neighbour grouping (Q-Shift, ConvShift) or learnable convolution
(LiT, multi-resolution).

[**verified ✓** against Vision-RWKV (arXiv:2403.02308) eq line 321,
LiT (arXiv:2501.12976) cited per Related Work §2.3, Kiruluta
(arXiv:2506.01963) lines 211–219, AudioRWKV (arXiv:2509.02167)
lines 265–271.]

Receptive-field analysis specific to ASR ($T \le 500$ post-2× sub-
sampling, 25 ms frame rate): dilations $\{1, 2, 4, 8\}$ at $k=3$
yield receptive fields of approximately $\{3, 5, 9, 17\}$ frames
each side, matching phoneme-to-syllable scale of 60–425 ms. This
specific dilation choice and its acoustic justification is **our**
contribution in Chapter 4.

#### What's NOT here

- The init-fix for $\alpha_{d>1} = 0.01$ (the v2 contribution) —
  Chapter 4.
- ASR-specific deployment — Chapter 4.

#### Citations

| Concept | Bib key |
|---|---|
| Multi-dilation DWConv on linear attention | `kiruluta2025breaking` |
| Q-Shift on RWKV (cross-modal precedent) | `duan2024visionrwkv` |
| ConvShift on RWKV-7 audio (cross-modal precedent) | `xiong2025audiorwkv` |
| DWC on linear DiT V-path | `wang2025lit` |

---

### 3.4.2 Block-complex transitions and Rayleigh dissipation

#### What it is

A function-class extension that replaces a real-diagonal recurrent
transition $\mathbb{R}_+^K$ with a block-complex transition
$SO(2)^{K/2} \times \mathbb{R}_+^{K/2}$, where each pair of state
channels forms a 2-vector rotated by an angle $\theta$ and scaled by
an exponential damping factor $e^{-\lambda}$ at every step. The
Rayleigh dissipation term $\eta \cdot \theta^2$ couples damping to
rotation magnitude, a standard physical prior from damped-oscillator
mechanics.

#### Math

```latex
G_{t,b} = e^{-\lambda_\text{eff}} \cdot R(\theta_{t,b}), \qquad
R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}, \qquad
\lambda_\text{eff} = \lambda_\text{raw} + \eta \cdot \theta^2
```

with $G_{t,b}$ the 2×2 transition block on channel pair $b$ at time
$t$, $\theta_{t,b}$ the (data-dependent) rotation angle, $\eta \ge 0$
the viscosity coefficient.

#### Provenance

This primitive is internally derived in the thesis. Two upstream
sources support the math:

1. **Lie-group extension argument.** The smallest natural extension
   of real-diagonal transitions $\mathbb{R}_+^K$ that can represent
   damped oscillation is the block-complex form $SO(2)^{K/2} \times \mathbb{R}_+^{K/2}$.
   This is standard Lie-group / matrix-theory material; cite a
   textbook reference.
2. **Rayleigh dissipation function.** The coupling $\eta \cdot \theta^2$
   between damping and squared rotation angle is the discrete-step
   analogue of Rayleigh's dissipation function from analytical
   mechanics. Standard physics textbook reference.

[**pending validation**: locate physics textbook references for both
Lie-group SO(2) and Rayleigh dissipation. Candidates: Goldstein,
*Classical Mechanics* (Rayleigh function in §1.5 or §2.5); Hall,
*Lie Groups, Lie Algebras, and Representations* (SO(2) in early
chapter). Add bib entries for whichever we use.]

#### Diagonal-class membership

Block-complex transitions sit inside the diagonal-class function
family: each $2 \times 2$ block is diagonalisable over $\mathbb{C}$
to $\mathrm{diag}(re^{i\theta}, re^{-i\theta})$ with $r = e^{-\lambda}$.
This places the mechanism inside the formal-language class of diagonal
SSMs.

[**pending validation against**: Sarrof, Veitsman, Hahn 2024,
arXiv:2405.17394, formal characterisation of the diagonal SSM class
and its complex-diagonal extension]

#### Citations

| Concept | Bib key |
|---|---|
| Diagonal SSM formal-language class | `sarrof2024expressive` |
| Lie-group reference (SO(2)) | **needs new bib entry** (Hall textbook?) |
| Rayleigh dissipation reference | **needs new bib entry** (Goldstein?) |
| Diagonal-SSM impossibility for non-Abelian state | `merrill2024illusion` |

---

### 3.4.3 Key-similarity preconditioning and value decorrelation

#### What it is

A class of operators that reduce associative-memory interference in
attention-like or recurrent layers by precomputing a
key-key similarity matrix and using it to decorrelate the values
before they enter the memory operation. The canonical form constructs
a unit-diagonal preconditioner from the exponential of the
key-similarity matrix; the preconditioner is then inverted (or solved
against the value tensor) to produce decorrelated values.

#### Math — LUCID-faithful preconditioner

[**MAJOR CORRECTION**: original draft mixed LUCID-the-paper's
preconditioner with our CVD additions ($\tau$, $\varepsilon I$).
Chapter 3 must present the **LUCID-faithful** form as prior art.
Our additions live in Chapter 4.]

LUCID Attention (final form, lucid_paper.tex line 332–334):

```latex
O = \mathrm{softmax}(QK^\top / \sqrt{d}) \cdot \left( M \circ \exp\!\left(\frac{K_\mathrm{RN} K_\mathrm{RN}^\top}{\sqrt{d}} - \sqrt{d}\right) \right)^{-1} V
```

where $K_\mathrm{RN}$ is the RMS-normalised keys (with row-norm
$\sqrt{d}$), $M$ is the causal mask making the preconditioner
lower-triangular and invertible, and the $-\sqrt{d}$ shift makes
the preconditioner matrix unit-diagonal by construction (line 230
of LUCID paper).

The earlier conceptual form (lucid_paper.tex line 206):

```latex
P = \left( M \circ \exp(KK^\top) \right)^{-1}
```

with the simpler interpretation $\exp(KK^\top)$ being the matrix
of key-key similarities in the exponential-kernel RKHS.

The decorrelated values $P V$ then enter the standard softmax
attention. Matrix multiplication is associative: in implementation,
the $P^{-1} V$ solve uses `torch.linalg.solve_triangular` (LUCID
paper line 119) — never materialising the inverse explicitly.

[**verified ✓** against Duvvuri et al. 2026 (arXiv:2602.10410)
lucid_paper.tex line 206 (initial form), line 235 (with row-norm),
line 332–334 (final practical form).]

#### What's NOT here — Chapter 4 contributions

The following extensions are **our** Chapter 4 contributions on top
of the LUCID-faithful preconditioner:

- **Per-head learnable temperature $\tau$** scaling the exponent
  argument: $\exp(\tau \cdot (K_\mathrm{RN} K_\mathrm{RN}^\top / \sqrt{d} - \sqrt{d}))$.
- **Regulariser $\varepsilon I$** added to $P$ for numerical stability
  in our chunked-recurrent solver ($\varepsilon = 10^{-4}$).
- **Substrate change**: linear-time RNN backbones (RWKV-6, Mamba-2,
  LA) instead of softmax Transformer attention. The $\mathrm{softmax}(QK^\top)$
  factor is absent — values are decorrelated and fed directly into
  the recurrence.
- **Architecture-specific deployment**: chunked recurrent for
  RWKV-6 (within WKV scan chunks), parallel for LION's
  bidirectional T×T, SSD-chunk-local for Mamba-2's dual form using
  the chunk-local $\mathbf{C}_c \mathbf{B}_c^\top$ structure.

#### Provenance — CRITICAL FRAMING

The unit-diagonal preconditioner kernel is introduced by Duvvuri et
al. 2026 in **LUCID**, a softmax-attention mechanism for
long-context retrieval inside a quadratic-complexity Transformer.
Their target task is multi-needle long-context retrieval; their
complexity is $O(N^2 d)$.

Our Chapter 4's CVD mechanism is **related to LUCID by logic, not by
adaptation**:

- LUCID operates on the full $T \times T$ softmax-attention matrix
  inside a Transformer.
- CVD operates on chunk-local $T_c \times T_c$ systems inside the
  recurrence of a linear-time RNN; the full $T \times T$ attention
  matrix is never materialised.
- LUCID retains $O(N^2 d)$. CVD retains the linear-time backbone's
  $O(T \cdot T_c \cdot d)$ envelope at $T_c \ll T$.

Chapter 3 presents the preconditioner kernel **as it is in LUCID**,
attributing the form to Duvvuri et al. 2026. Chapter 4 then introduces
CVD as a new mechanism on a different substrate, sharing the
key-similarity decorrelation principle.

#### Connection to delta rule

The preconditioner has a clean interpretation as a delta-rule update
in the reproducing kernel Hilbert space (RKHS) induced by the
exponential kernel: applying $P^{-1}$ to $V$ removes the components
of each value that are aligned with the projection of earlier keys
under the key-similarity geometry. This matches the structural
intent of DeltaNet's rank-one Householder erasure
$(I - \beta_t k_t k_t^\top)$, but realises it in a closed form
within each chunk rather than online per-token.

[**pending validation against**: Duvvuri et al. 2026 (LUCID),
arXiv:2602.10410, Methods section / preconditioner construction.
Local PDF at `papers/LUCID_2602.10410v1.pdf`; e-print preferred.
Also validate the RKHS connection — confirm that the LUCID paper
itself derives this RKHS framing or whether it is our framing on top
of their kernel]

#### Citations

| Concept | Bib key |
|---|---|
| LUCID preconditioner | `duvvuri2026lucid` |
| Delta rule (rank-1 Householder erasure) | `schlag2021deltanet` |
| Gated DeltaNet (parallelised) | `yang2024deltanet` |
| Compact Recurrent Transformer (related) | `mucllari2025crt` |

---

## 3.5 Formal expressivity bounds and the diagonal-class scope

### Purpose

Recap the formal complexity-class boundaries that bracket
linear-time RNN expressivity, formalise the diagonal-class function
family within which our mechanisms operate, and forward the
state-tracking question to future work.

This section is intentionally short — §2.5 (Related Work) already
established the literature; here we are formalising the *function-class
membership* of our three mechanisms, not re-establishing the bounds.

### What this section explains

- The $\mathsf{TC}^0$ ceiling for log-precision Transformers (Hahn 2020,
  Merrill 2023, Strobl 2024).
- The $\mathsf{TC}^0$ ceiling extended to SSMs / Mamba (Merrill, Petty,
  Sabharwal 2024).
- The formal-language analysis specific to diagonal SSMs (Sarrof,
  Veitsman, Hahn 2024).
- The DPLR boundary: rank-one Householder updates lift the
  architecture from diagonal-class $\mathsf{NC}^1$ to
  $\mathsf{PNC}^1$-complete.
- Our three mechanisms (MSDC, CVD, DHO) all operate within the
  diagonal-class function family. Therefore: by the cited bounds,
  no mechanism in our matrix can exhibit non-Abelian state-tracking
  gain at any scale. This is a formal scoping of the empirical
  matrix, not an empirical limitation.

### Math content (light)

No new equations introduced. One short formal definition recalled
from Sarrof 2024 (the diagonal SSM class definition).

### Citations

| Concept | Bib key |
|---|---|
| Hard-attention PARITY / Dyck-2 limitation | `hahn2020limitations` |
| Log-precision Transformer in $\mathsf{TC}^0$ | `merrill2023parallelism` |
| Survey of formal Transformer expressivity | `strobl2024formal` |
| SSMs in $\mathsf{TC}^0$ (illusion of state) | `merrill2024illusion` |
| Formal-language analysis of diagonal SSMs | `sarrof2024expressive` |
| DPLR / DeltaNet rank-1 lift | `schlag2021deltanet` |
| Gated DeltaNet | `yang2024deltanet` |
| RWKV-7 DPLR | `peng2025rwkv7` |

### Anti-scope

- Do not re-prove anything; this is a citation-grade section.
- Do not propose state-tracking mechanisms; future work paragraph
  closes the chapter.

### Validation status

- Diagonal-SSM formal class definition: **pending validation** against
  `sarrof2024expressive`.
- Other bounds: citation-only, no formula validation needed.

---

## Order of writing

Recommended writing order — frame-first, primitives-second, scope-third:

1. **§3.4** (primitives) — write *first*. This is the math workhorse;
   getting the formulas right unlocks Chapter 4 and prevents
   re-pagination.
2. **§3.2** (backbones) — second. Builds on the validated math from
   §3.4.
3. **§3.3** (LION) — third. Light-weight; depends only on §3.2.
4. **§3.1** (axes) — fourth. Conceptual; easier to write after the
   technical sections are in place because the axis-vs-mechanism
   correspondence will be concrete by then.
5. **§3.5** (formal scope) — last. Closes the chapter.

---

## Papers needing validation pass (e-print download order)

E-print download via `curl -sL https://arxiv.org/e-print/<id>`.
Validation order matches priority for §3.2–§3.4 formulas.

| # | Paper | arXiv ID | Local PDF? | Section using it | Why |
|---|---|---|---|---|---|
| 1 | LA Katharopoulos | 2006.16236 | ❌ | §3.2 F-3.2.2 | Simplest — calibrates prompt |
| 2 | Mamba-2 SSD | 2405.21060 | ❌ (e-print already DL'd) | §3.2 F-3.2.4 | Critical to canonical SSD form |
| 3 | Mamba (selective Δt) | 2312.00752 | ❌ | §3.2 F-3.2.5 | Continuous-discrete bridge |
| 4 | RWKV-5/6 (Eagle and Finch) | 2404.05892 | ✅ | §3.2 F-3.2.3 | RWKV-6 WKV form |
| 5 | LION | 2502.16249 | ✅ | §3.3 F-3.3.1, F-3.3.2 | Parallel form + decay taxonomy |
| 6 | LUCID | 2602.10410 | ✅ | §3.4.3 | Preconditioner form |
| 7 | Non-Attention LLM | 2506.01963 | ✅ | §3.4.1 | Multi-dilation DWConv |
| 8 | Sarrof et al. | 2405.17394 | ❌ | §3.4.2, §3.5 | Diagonal SSM class definition |
| 9 | Vision-RWKV (cross-ref) | 2403.02308 | ❌ | §3.4.1 | Q-Shift cross-validation |
| 10 | AudioRWKV (cross-ref) | 2509.02167 | ✅ | §3.4.1 | ConvShift cross-validation |
| 11 | Hahn 2020 (citation-grade) | 1906.06755 | ❌ | §3.5 | Citation only |
| 12 | Merrill, Sabharwal 2023 | 2207.00729 | ❌ | §3.5 | Citation only |
| 13 | Strobl et al. 2024 | 2311.00208 | ❌ | §3.5 | Citation only |
| 14 | Merrill, Petty, Sabharwal 2024 | 2404.08819 | ❌ | §3.5 | Citation only |
| 15 | Schlag 2021 (DeltaNet) | 2102.11174 | ❌ | §3.4.3 | Light ref for delta-rule connection |
| 16 | Yang 2024 (Gated DeltaNet) | 2406.06484 | ❌ | §3.4.3 | Light ref |

---

## New bib entries that may be needed

For physics-grounded references in §3.4.2 (Block-complex transitions
and Rayleigh dissipation):

1. **Lie-group / SO(2) reference.** Suggested:
   - Hall, B. C. (2015), *Lie Groups, Lie Algebras, and
     Representations: An Elementary Introduction*, 2nd ed. Springer.
   - Or: Gantmacher, F. R. (1959), *Theory of Matrices*, vol. 1,
     Chelsea.

2. **Rayleigh dissipation reference.** Suggested:
   - Goldstein, H., Poole, C., Safko, J. (2002), *Classical
     Mechanics*, 3rd ed. Addison-Wesley. (Rayleigh dissipation
     function in §1.5 / §2.5.)
   - Or: Strogatz, S. (2014), *Nonlinear Dynamics and Chaos*, 2nd ed.
     Westview / CRC. (More accessible, less canonical.)

Decision needed during the §3.4.2 validation pass: pick one canonical
reference per topic, draft the bib entry, propose for inclusion.

---

## Validation pass — summary of findings (2026-04-29)

E-print download workflow used: `curl -sL https://arxiv.org/e-print/<id>`,
`tar -xzf`, `grep + Read .tex` on the source bundle. All 9 papers
downloaded and inspected directly.

### Papers consulted

| # | Paper | arXiv ID | Key extract location |
|---|---|---|---|
| 1 | LA (Katharopoulos) | 2006.16236 | `arxiv.tex` lines 380–391, 467–470 |
| 2 | Mamba-2 SSD | 2405.21060 | `structure/background.tex` 21–39, 67–76; `structure/ssd.tex` 26–27, 41–48 |
| 3 | Mamba (selective) | 2312.00752 | `src/method.tex` 72–75; `src/background.tex` 71–76 |
| 4 | RWKV-5/6 (Eagle Finch) | 2404.05892 | `main.tex` 552–591 (Finch Time Mix) |
| 5 | LION | 2502.16249 | `3-LION.tex` 14–37 (eqs); 286–289 (variants) |
| 6 | LUCID | 2602.10410 | `lucid_paper.tex` 206, 235, 332–334 |
| 7 | Non-Attention LLM | 2506.01963 | `SSM2.tex` 211–219 |
| 8 | Sarrof et al. | 2405.17394 | `camera_ready_draft.tex` 137–186 |
| 9 | Vision-RWKV (cross-ref) | 2403.02308 | `iclr2025_conference.tex` 321 |

### Findings — 4 corrections + 1 reframing

**Major corrections applied:**

1. **F-3.2.3 (RWKV-6 Finch) — major correction.**
   Original draft was the RWKV-4 softmax-normalised single-vector form.
   Corrected to the Finch matrix-state form:
   $s_t = \mathrm{diag}(w_t) \cdot s_{t-1} + k_t^\top v_t$,
   $\mathrm{wkv}_t = s_{t-1} + \mathrm{diag}(u) \cdot k_t^\top v_t$.
   The state is matrix-valued $\mathbb{R}^{(D/h) \times (D/h)}$ per head.

2. **F-3.2.4 (Mamba-2 SSD) — notation correction.**
   Mamba-2 paper explicitly drops bar notation ("we have changed
   notation to simplify the presentation"). Use $A_t$ not $\bar A_t$;
   present scalar-identity restriction $A_t = a_t I$ explicitly.

3. **F-3.4.3 (LUCID) — major framing correction.**
   Original draft conflated LUCID-the-paper with our CVD additions.
   Corrected to LUCID-faithful preconditioner $(M \circ \exp(K_\mathrm{RN} K_\mathrm{RN}^\top / \sqrt d - \sqrt d))^{-1}$
   without learnable $\tau$ or $\varepsilon I$. Those become explicit
   Chapter 4 contributions.

4. **F-3.3.2 (LION-S decay) — refinement.**
   Original draft said "per-head σ-decay $\lambda_h = \sigma(W^{(h)}_\lambda x)$".
   Corrected to LION paper's exact form $\lambda_i = \sigma(W x_i + b)$
   (per-token data-dependent sigmoid, GRFA-derived).

**Reframing applied:**

5. **§3.4.1 multi-dilation primitive — reframed as family.**
   Original draft presented one formula tied to Kiruluta et al. 2025.
   Reframed: §3.4.1 now presents four cross-domain instances of the
   input-side local-mixing family (Q-Shift, DWC-on-V, multi-resolution,
   ConvShift), with our specific dilation set $\{1,2,4,8\}$ +
   learnable $\alpha_d$ + init-fix deferred to Chapter 4.

### Findings — verifications without changes

- **F-3.2.2 (LA)** verified exact match against Katharopoulos 2020
  eq:attn_state and eq:trnn-3.
- **F-3.2.5 (Mamba selective Δt)** verified with one refinement
  (`Parameter +` baseline added).
- **F-3.3.1 (LION matrix form)** verified against eq:attvecbid +
  eq:maskselbir.
- **§3.4.2 (DHO Sarrof bracket)** verified: Sarrof's
  diagonal SSM definition (eq:recurrence-ssm) explicitly accommodates
  complex-valued activations (line 184), so the block-complex
  $SO(2)^{K/2}$ form sits within the diagonal class.

### Findings — physics references for §3.4.2 (still pending)

For DHO's Lie-group + Rayleigh framing, two textbook references
need bib entries (per user direction "варто додавати посилання на
джерела якщо це physics based"):

- **Goldstein, Poole, Safko, *Classical Mechanics* (3rd ed., 2002).**
  Rayleigh dissipation function — §1.5 / §2.5.
  Suggested bib key: `goldstein2002classical`.
- **Hall, *Lie Groups, Lie Algebras, and Representations* (2nd ed.,
  2015).** SO(2) and matrix-Lie-group basics — Chapter 1.
  Suggested bib key: `hall2015lie`.

Both books are canonical; we cite them once in §3.4.2 to anchor the
internally-derived DHO mechanism's mathematical foundation.

### Open items before Chapter 3 writing starts

- Decision on physics references (Goldstein vs Strogatz; Hall vs
  Gantmacher). Default: **Goldstein + Hall**.
- HLA bib entry inserted into `bibliography.bib` (already drafted in
  `Thesis_Positioning.md` §15).
- §3.1 axes — final decision on whether to define 3 axes with 2
  mentioned (preferred per user) or 5 axes formally.
  Per Thesis_Positioning §11 — 3 + 2 confirmed.

---

*End Chapter 3 working plan v2 (2026-04-29). Formula validation pass
complete; ready for chapter writing in the recommended order
(§3.4 → §3.2 → §3.3 → §3.1 → §3.5).*
