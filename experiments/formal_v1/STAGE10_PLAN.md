# Stage 10+ Plan — Final Roadmap Before Big Experiments

**Status:** final planning step before the next wave of empirical work.
Supersedes all earlier ad-hoc next-step lists. Every subsequent run
should be attributable to a stage in this document. When an experiment
completes, log its result to §9 (master comparison matrix) and, if it
needs a full narrative, append a stage-N_RESULTS.md the way Stages 2-9
were documented.

**Authoritative references (read in this order before editing this file):**
- [stages_2_9_summary.md](stages_2_9_summary.md) — chronological record
  of Stages 2-9, per-epoch trajectories, cross-experiment invariant.
- [RESULTS.md](RESULTS.md) — CER/WER table for the existing spine.
- [TODO_FUTURE_IDEAS.md](TODO_FUTURE_IDEAS.md) — paper-by-paper reviews
  (Papers 1-9) and Phase-4 cross-paper synthesis. Each stage below
  links into that document.
- [CLAUDE.md](CLAUDE.md) — architecture rules, naming conventions, and
  common pitfalls that must be respected by every new backbone.

---

## 0. Purpose and scope

This plan defines the experimental queue for the **next wave** of this
thesis — beyond the Stage-2-through-9 chain on RWKV-6 causal — across
three test-hierarchy stages:

1. **Stage 10.X** — new mechanism runs on RWKV-6 causal (same spine as
   Stages 2-9: seed 42, 30 ep, LibriSpeech clean-100, $T \le 500$,
   7 M encoder).
2. **Stage 11.X** — transfer of the surviving mechanisms to Mamba-2
   and Linear Attention baselines.
3. **Stage 12.X** — extension to bidirectional LION; mechanisms that
   break the LION parallel form (P6 NCGRU-Cayley, P9 M²RNN) are
   explicitly excluded from Stage 12 per the Phase-4 structural
   observation (TODO_FUTURE_IDEAS.md §4.10).

Every experiment in this queue has a pre-registered **family**,
**comparison cohort**, **decision rule**, and **metrics schedule**.
The intent is that no run is exploratory — each one answers a
pre-declared question against a pre-declared reference set.

---

## 1. Design principles (binding for all Stage 10+ runs)

These carry forward from the Stages-2-9 discipline and from the
Phase-4 cross-paper synthesis.

1. **Exact reduction at init.** Every new backbone must have a clean
   zero-regression-at-init contract: at step 0, the output is
   bit-identical to an already-characterised baseline. The stages-2-9
   evidence consistently shows this discipline is what separates
   clean attribution from engaged-null ambiguity.
2. **Watch identifiability, not just mobility.** Indep-λ P²-RSE and
   A1′ both moved their parameters and delivered no CER — the failure
   mode is parameterisation degeneracy, not insufficient optimisation.
3. **Keep parameter parity to within 5 % of the 7 M baseline** unless
   the extra capacity itself is the hypothesis. P6 NCGRU-Cayley at
   rank 4 and P8 PoM at $D = 2d$ fail this and must be deployed in
   thin / low-rank configurations.
4. **Do not reject families already explored.** Our existing solutions
   inside Families B (linear transition) and D (feature/channel) may
   not be the global optimum within those families. Stage 10+
   explicitly revisits these families with alternative mechanisms so
   we can *compare against our existing winners*, not bypass them.
5. **Matched-epoch comparison is mandatory.** Stage 9 halted S9A at
   ep 15 purely on matched-epoch tracking vs references. Every
   Stage-10 run must log matched-epoch dev CER at epochs 5, 10, 15,
   20, 25, 30 against its comparison cohort.
6. **Seed noise σ ≈ 0.0014.** Single-seed results are patterns, not
   laws. A single-seed CER delta below ±0.0014 is inside noise.
   Multi-seed (3 seeds) is required for any claim that a mechanism
   breaks the anchor.
7. **Diagnostic value counts even if CER doesn't move.** The invariant
   extensions (A1′, T1, T2, S9A, S9B) are thesis-defensible because
   they *tightened the invariant* even without CER lift. Pre-register
   the diagnostic probes.

---

## 2. Testing hierarchy

```
Stage 10 (RWKV-6 causal, same spine as Stages 2-9)
  └─> Stage 11 (Mamba-2, Linear Attention)
        └─> Stage 12 (LION bidirectional)
```

| Test-hierarchy stage | Scope | Spine | Pass / halt criterion |
|---|---|---|---|
| **Stage 10** | New mechanisms on RWKV-6 causal | Seed 42, 30 ep, `clean-100`, 7 M enc., $d{=}256$, 6 L, 4 H, $K{=}64$ | CER ≤ anchor (0.1185 dev) → promote to Stage 11. CER > vanilla (0.1258) at matched-ep 15 → halt. |
| **Stage 11** | Transfer promoted mechanisms to Mamba-2 and causal Linear Attention baselines | 7 M param-parity, same 30-ep spine | Relative delta to each architecture's own anchor. |
| **Stage 12** | Bidirectional LION port of LION-compatible mechanisms | 7 M param-parity, same 30-ep spine | Relative delta to `lion` baseline. Mechanisms breaking parallel form (P6, P9) are excluded. |

---

## 3. Mechanism families under test

Five families identified in TODO_FUTURE_IDEAS.md §4.6. None is rejected
from Stage 10+ except Family E (chunk-retrieval, out-of-regime at
$T \le 500$).

| Family | Defining characteristic | Stage-10+ members | Existing solution in our spine |
|---|---|---|---|
| **A — Multi-scale temporal aggregation** | Scale-indexed operators + mixer | P1 Log-Linear, P7 multi-dilation ConvShift | `rwkv6_rse_depth` (won), `rwkv6_convshift_trap` (won), Multi-Rate RSE (plateau) |
| **B — Linear transition-operator family** | Change Lie group $\{G_t\}$; linear in state | P6 NCGRU-Cayley (diagnostic only) | `rwkv6_rse_strong_viscosity` (ANCHOR), Stage-8 T2 (AMBIGUOUS) |
| **C — Non-linearity of state ($NC^1$)** | Element-wise non-linearity on matrix state | **P9 M²RNN (sole member)** | None — genuinely new axis |
| **D — Feature-map / channel-side enrichment** | Feature-vector or ChannelMix changes; transition unchanged | P3 Avey bypass, P8 PoM (diagnostic) | `rwkv6_qtail_lowrank_all` (best feature-side), `rwkv6_hadamard_n2` (PLATEAU), `rwkv6_rmsnorm` (PLATEAU) |
| **E — Chunk retrieval / sparse hybrid** | Bolt-on sparse attention over chunked past | **Deferred — out-of-regime** (P4, P5) | — |

**Paper 2 NaLaFormer and Paper 8 PoM** are deferred to Stage 11b
(Linear Attention natural-home tests), not run on the causal RWKV-6
spine, because their pathologies either do not apply there
(NaLaFormer L1-cancellation) or are structurally already probed in the
already-tested feature-side line (PoM at $k{=}2$).

---

## 4. Baseline consolidation (pre-Stage 10)

Read-only inventory of what we already know from Stages 2-9. Pulled
verbatim from [RESULTS.md](RESULTS.md) and the Stage-9 benchmark table
in [stages_2_9_summary.md](stages_2_9_summary.md#stage-9--sparse-edge-layer-specialist-transition).
Column `Dev CER` / `Test CER` / `Test WER` are single-seed, 30-ep
(seed 42). Timing / memory are RTX PRO 6000 Blackwell, $B{=}10$,
$T{=}1200$ mels where available.

All results below are the authoritative prior; **Stage-10 runs update
§9, not this section.**

### 4.1 Winners (by category)

| Category | Winner backbone | Dev CER | Test CER | Notes |
|---|---|---|---:|---|
| Absolute-best causal (cross-axis) | `rwkv6_rse_convshift` | **0.1145** | **0.1126** | mixes input-side ConvShift with transition-side RSE; not a pure-transition row |
| Pure transition-side (anchor) | **`rwkv6_rse_strong_viscosity`** | **0.1185** | **0.1177** | ANCHOR. Strong θ budget + Rayleigh viscosity coupling. |
| Paired-pole control | `rwkv6_p2rse_strong_viscosity` | 0.1190 | 0.1196 | tied anchor within σ |
| Feature-side | `rwkv6_qtail_lowrank_all` | 0.1238 | 0.1240 | MARGINAL; best Stage-6 feature-side |
| Input-side (cross-axis mix) | `rwkv6_convshift_trap` | 0.1150 | 0.1150 | gain is ConvShift, not the solver |
| Vanilla reference | `rwkv6` | 0.1258 | 0.1263 | baseline; seed noise σ ≈ 0.0014 |

### 4.2 Verdicts landscape (summary of Stages 2-9)

| Verdict class | Backbones |
|---|---|
| **WON (ceiling break)** | `rse_strong`, `rse_depth`, `rse_strong_viscosity`, `rse_depth_viscosity`, `rse_convshift`, `convshift_trap` |
| **PLATEAU / tied anchor** | `trap`, `trap_var`, `gen2`, `rse`, `rse_m2`, `rse_m4`, `p2rse`, `p2rse_softmax`, `p2rse_strong_viscosity`, `rmsnorm`, `hadamard_n2`, `qtail`, `qtail_gamma`, `qtail_gamma_dbeta`, `qtail_lowrank`, `qtail_lowrank_all` |
| **AMBIGUOUS-ENGAGED (mechanism moved, no CER)** | `rse_dphi_viscosity` (A1′, +2σ), `delta_warmstart_fixed` (T1, tied-vanilla), `nonnormal_rse_viscosity` (T2, +1σ), `sparse_nonnormal_rse_edge_only_viscosity` (S9B, regression-stability) |
| **REGRESSION** | `ab3`, `p2rse_indeplam_strong_viscosity` (15σ regression), `sparse_nonnormal_rse_viscosity` (S9A, halted ep 15) |
| **Retracted (implementation artefact)** | `delta_warmstart` (Stage 6 delta null; superseded by T1) |

### 4.3 Cross-experiment invariant (to be tested by Stage 10)

> At 7 M / clean-100 / 30 ep on causal RWKV-6, **five independent**
> strict-superset extensions of `rwkv6_rse_strong_viscosity` — A1′
> readout-gauge, T1 recurrent delta, T2 dense non-normal, S9A
> gated-sparse all-layer, S9B hard edge-only — all converge to the
> same signature: mechanism engages SGD aggressively (D5 mobility
> non-zero, D6/D7 realised action substantial), CER does not benefit.

The invariant was scoped to **dense per-(token, head, layer) freedom
on linear-in-state operators.** Stage-10 Phase I is the first probe
that *either extends the invariant across the linearity boundary*
(M²RNN) *or tests a structural-not-dense shape of freedom*
(Log-Linear hierarchical). Both are diagnostically informative
regardless of CER outcome.

### 4.4 Architectures not yet run at the formal_v1 level

These columns are planned for Stage 11 / 12; no results yet in the
formal_v1 codebase:

- **Mamba-2** (`mamba2`, `mamba2_encoder.py`) — Stage 11 baseline.
- **Linear Attention** (`blocks.py` `LinearAttentionLayer`,
  bidirectional ELU+1) — Stage 11 baseline and NaLaFormer natural home.
- **LION** (`lion`, `rwkv6_time_mix.py` mode `lion`) — Stage 12 baseline.
- **LION variants**: `lion_convshift`, `lion_lucid`,
  `lion_lucid_chunked`, `lion_delta`, `lion_headscale`,
  `lion_convshift_headscale` — Stage 12 ablations.

These will be populated as Stages 11–12 run. The master comparison
matrix in §9 has pre-reserved rows.

---

## 5. Experiment roadmap (priority-ordered)

Three phases, ordered by transferability-weighted value.

### Phase I — Transferable high-value (Stage 10.1–10.4)

Target **new axes** (Families A, C) or **validated mechanism
extensions** (Family A input-side), transferable to all three
downstream architectures without major redesign.

- [ ] **Stage 10.1** — `rwkv6_loglinear` (plain Log-Linear RWKV-6) — Family A
- [ ] **Stage 10.2** — `rwkv6_m2rnn_sparse` (M²RNN Phase A, sparing-use) — Family C
- [ ] **Stage 10.3** — `rwkv6_convshift_multidil` (multi-dilation ConvShift) — Family A input-side
- [ ] **Stage 10.4** — `rwkv6_chanmix_bypass` (Avey partial-embedding bypass) — Family D (channel-mix)

### Phase II — Revisit characterised families with alternatives (Stage 10.5–10.7)

These **do not reject** our existing Stage-6 / Stage-8 winners — they
compare alternative parameterisations of the same families so we can
characterise whether the engaged-null band is family-intrinsic or
parameterisation-specific.

- [ ] **Stage 10.5** — `rwkv6_orthogonal` (NCGRU Cayley-orthogonal, diagnostic vs T2) — Family B
- [ ] **Stage 10.6** — `rwkv6_pom_vlift` (PoM polynomial value-lift, diagnostic vs qtail/hadamard_n2) — Family D
- [ ] **Stage 10.7** — `rwkv6_loglinear_rse_strong_viscosity` (Log-Linear × RSE composition) — **Conditional on 10.1 ≥ MARGINAL**

### Phase III — Architecture transfer (Stage 11)

Promoted mechanisms re-tested on Mamba-2 and Linear Attention.
NaLaFormer gets its natural-home test here, not on RWKV-6.

- [ ] **Stage 11.0** — `mamba2` vanilla baseline at formal_v1 parity
- [ ] **Stage 11.1** — Transfer Stage 10 winners to Mamba-2
- [ ] **Stage 11.2** — `linear_attn_causal` baseline + NaLaFormer natural-home test
- [ ] **Stage 11.3** — Transfer Stage 10 winners to Linear Attention

### Phase IV — LION bidirectional (Stage 12)

LION baselines + LION-compatible mechanisms only. P6 NCGRU-Cayley
and P9 M²RNN are excluded per §4.10 of TODO_FUTURE_IDEAS.md (both
break LION's parallel form at the math level).

- [ ] **Stage 12.0** — `lion` baseline
- [ ] **Stage 12.1** — `lion_convshift`, `lion_headscale`, `lion_convshift_headscale`
- [ ] **Stage 12.2** — `lion_lucid`, `lion_lucid_chunked` (corrected LUCID from CLAUDE.md)
- [ ] **Stage 12.3** — `lion_delta` (causal-only delta, corrected)
- [ ] **Stage 12.4** — `lion_loglinear` (mirror-Fenwick, design decision cell from Paper 1)
- [ ] **Stage 12.5** — `lion_pom` (PoM's structural natural home)

---

## 6. Per-experiment specification

Each subsection uses a fixed template:

- **Family** / **Paper review link** / **Hypothesis**
- **Mechanism** (math + backbone name + flag)
- **Zero-regression contract**
- **Parameters / wall-clock estimate**
- **Comparison cohort** (exact backbones for matched-epoch comparison)
- **Decision rule** (pre-registered pass/halt criteria)
- **Diagnostic probes** (what to measure regardless of CER outcome)

### Stage 10.1 — `rwkv6_loglinear` (Log-Linear RWKV-6, plain)

- **Family.** A — Multi-scale temporal aggregation (new log-scale sub-axis).
- **Paper review.** [TODO_FUTURE_IDEAS.md §Paper 1](TODO_FUTURE_IDEAS.md#paper-1--log-linear-attention-guo-et-al-arxiv-250604761).
- **Hypothesis.** Log-scale Fenwick-bucket prefix partition with
  data-dependent per-token, per-scale mixer $\lambda_t^{(\ell)}$
  provides a *structurally-distinct* prefix summary (not per-token
  freedom on the existing transition) that clears the cross-experiment
  invariant's dense-per-token filter.
- **Mechanism.** $L = \lceil \log_2 T_{\max}\rceil + 1 \approx 10$
  Fenwick bucket states $\{S_t^{(\ell)}\}$ evolve under the existing
  per-step diagonal decay; readout
  $y_t = \sum_\ell \lambda_t^{(\ell)} r_t^\top S_t^{(\ell)} + \text{bonus}$
  with $\lambda_t^{(\ell)} = 1 + (W_\lambda^{(2)} \tanh(W_\lambda^{(1)} x^{(w)}_t))^{(\ell)}$,
  $W_\lambda$ zero-init.
- **Backbone / flag.** `rwkv6_loglinear`; flag `use_loglinear: bool = False`,
  `loglinear_levels: int = 10`. Location: new method
  `_forward_recurrent_loglinear` beside `_forward_recurrent` in
  `src/models/rwkv6_time_mix.py`.
- **Zero-regression contract.** $\lambda_t^{(\ell)} \equiv 1$ at init
  ⇒ $\sum_\ell S_t^{(\ell)} = S_t$ (Fenwick buckets partition the
  prefix by construction) ⇒ $y_t$ = vanilla RWKV-6 bit-exact.
- **Parameters.** +57 K encoder-wide (0.8 % of 7 M).
- **Wall-clock estimate.** 2–3× anchor.
- **Comparison cohort (matched-epoch 5/10/15/20/25/30).**
  - Primary: `rwkv6` vanilla (baseline, 0.1258 / 0.1263).
  - Family-A same-axis: `rwkv6_rse_m2` and `rwkv6_rse_m4` (flat
    Multi-Rate RSE, PLATEAU) — direct check that log-scale ≠ flat
    multi-rate.
  - Invariant anchor: `rwkv6_rse_strong_viscosity` (0.1185 / 0.1177).
- **Decision rule.**
  - **BREAK** (dev $< 0.1200$): promote to Stage 10.7 composition.
  - **MARGINAL** (dev $\in [0.1200, 0.1230]$): promote to Stage 10.7.
  - **PLATEAU** (dev $\in [0.1230, 0.1260]$): close log-linear line
    on causal RWKV-6, do not proceed to composition.
  - **REGRESSION** (dev $> 0.1260$): halt at ep 15 if matched-epoch
    tracking is ≥ +0.006 behind `rwkv6`.
- **Diagnostic probes (mandatory regardless of CER).**
  - Per-level $\lambda_{t,h}^{(\ell)}$ mobility by layer: mean, std
    of realised $\lambda$ per layer. Tests the multi-scale depth
    hierarchy thread at the per-position axis (see Stage-2 `gen2`
    α-pattern and Stage-4 `rse_depth` depth pattern).
  - Per-bucket state-norm statistics $\|S_t^{(\ell)}\|_F$ over $t$.
  - Variable-length batch validity: verify padded-position $\lambda$
    contributions are correctly masked.

### Stage 10.2 — `rwkv6_m2rnn_sparse` (M²RNN Phase A, sparing-use)

- **Family.** C — Non-linearity of state (sole member, new axis).
- **Paper review.** [TODO_FUTURE_IDEAS.md §Paper 9](TODO_FUTURE_IDEAS.md#paper-9--mrnn-matrix-to-matrix-recurrent-neural-network-mishra-et-al-arxiv-260314360).
- **Hypothesis.** Non-linear state transition $\tanh(SW + kv^\top)$
  crosses the $TC^0 \to NC^1$ boundary. The cross-experiment
  invariant was scoped to linear-in-state extensions; M²RNN directly
  tests whether the invariant extends across the linearity boundary.
- **Mechanism (parallel-branch, Reviewer-B's form).** At layer
  $L^* = 5$ (deepest), per head:
  - $S^\text{rwkv}_{t,h} = \text{diag}(w_t) S^\text{rwkv}_{t-1,h} + k_{t,h} v_{t,h}^\top$ (unchanged WKV branch)
  - $Z^{m2}_{t,h} = \tanh(S^{m2}_{t-1,h} W_h + k_{t,h} v_{t,h}^\top)$
  - $S^{m2}_{t,h} = f_{t,h} S^{m2}_{t-1,h} + (1 - f_{t,h}) Z^{m2}_{t,h}$
  - $y_{t,h} = r_{t,h}^\top (S^\text{rwkv}_{t,h} + \lambda_h S^{m2}_{t,h}) + \text{bonus}$
- **Backbone / flag.** `rwkv6_m2rnn_sparse`; flag `use_m2rnn: bool = False`,
  `m2rnn_layer: int = 5`. Location: new scan branch in
  `src/models/rwkv6_time_mix.py` dispatched at layer $L^*$ only.
- **Zero-regression contract.** $\lambda_h = 0, W_h = I$ at init
  ⇒ $y_t = r_t^\top S^\text{rwkv}_t + \text{bonus}$ = vanilla RWKV-6
  bit-exact. This is load-bearing: the parallel-branch form (not the
  paper's native substitution) is what preserves the discipline.
- **Parameters.** +17 K encoder-wide (0.25 % of 7 M) for sparing-use.
- **Wall-clock estimate.** Phase A (pure PyTorch + `torch.compile`):
  3–8× anchor for the full 6-layer stack (one layer sequential, five
  parallel). Pushes 30-ep training from ~hours to ~half-day scale.
- **Comparison cohort (matched-epoch).**
  - Primary: `rwkv6` vanilla.
  - Invariant anchor: `rwkv6_rse_strong_viscosity`.
  - Family-adjacent (linear-in-state engaged-nulls):
    - `rwkv6_delta_warmstart_fixed` (T1, tied-vanilla)
    - `rwkv6_nonnormal_rse_viscosity` (T2, +1σ AMBIGUOUS)
    - `rwkv6_rse_dphi_viscosity` (A1′, +2σ)
- **Decision rule (pre-registered, phased).**
  - **Phase A outcome = BREAK** (dev $< 0.1170$): port Triton kernel,
    test all-layer variant (Phase B).
  - **Phase A outcome = MARGINAL** (dev $\in [0.1170, 0.1210]$):
    conditional Phase B; prioritize multi-seed first.
  - **Phase A outcome = PLATEAU** (dev $\in [0.1210, 0.1260]$):
    declare non-linearity axis closed at 7 M / 30 ep; document as
    extension of cross-experiment invariant across linearity boundary.
  - **Phase A outcome = REGRESSION** (dev $> 0.1260$ or matched-ep-15
    tracking ≥ +0.015 behind `rwkv6`): halt at ep 15; investigate
    training instability (tanh saturation, gradient clipping on
    $\partial L/\partial S_t$).
- **Diagnostic probes.**
  - $\lambda_h$ mobility over training: does SGD grow it away from 0?
  - $W_h$ Frobenius distance from identity: does the transition
    matrix learn a substantive rotation?
  - tanh saturation fraction (fraction of $|SW + kv^\top|_{ij} > 2$).
  - Gradient clipping rate on $\partial L / \partial S_t$ per step.

### Stage 10.3 — `rwkv6_convshift_multidil` (multi-dilation ConvShift)

- **Family.** A — Multi-scale temporal aggregation (input-side
  variant); extends validated ConvShift mechanism.
- **Paper review.** [TODO_FUTURE_IDEAS.md §Paper 7](TODO_FUTURE_IDEAS.md#paper-7--non-attention-llm-arxiv-250601963).
- **Hypothesis.** Extending ConvShift's receptive field from $\pm 1$
  frame to $\pm 8$ frames (syllable-envelope scale, 170 ms at 100 fps
  post-subsampling) via parallel dilated branches $\{1, 2, 4, 8\}$
  captures acoustic hierarchy that vanilla ConvShift cannot.
- **Mechanism.**
  $x_\text{mixed} = \sum_{d \in \{1,2,4,8\}} \alpha_d \cdot \text{DWConv1d}_d(x; \text{kernel}=3)$
  with $\alpha_1 = 1, \alpha_{2,4,8} = 0$ at init (per-layer learnable).
  Causal padding in `mode=recurrent`, symmetric in `mode=lion`.
- **Backbone / flag.** `rwkv6_convshift_multidil`; flag
  `use_conv_shift_multidilation: bool = False`. Location:
  `src/models/mechanisms/conv_shift.py`.
- **Zero-regression contract.** $\alpha_1 = 1, \alpha_{2,4,8} = 0$ at
  init recovers single-dilation ConvShift bit-exactly.
- **Parameters.** +18 K encoder-wide (0.25 %).
- **Wall-clock estimate.** $< 5 \%$ overhead (DWConv cost negligible
  vs WKV scan at our $T$).
- **Comparison cohort.**
  - Primary: `rwkv6_convshift_trap` (single-dilation, 0.1150 / 0.1150).
  - Absolute-best: `rwkv6_rse_convshift` (0.1145 / 0.1126) — cross-axis
    reference; composition test in later stage if 10.3 wins.
- **Decision rule.**
  - **BREAK** (dev $< 0.1145$): compose with RSE
    (`rwkv6_rse_convshift_multidil`) as Stage 10.3b.
  - **MARGINAL** (dev $\in [0.1145, 0.1160]$): conditional composition.
  - **PLATEAU** (dev $\in [0.1160, 0.1200]$): close multi-dilation line.
  - **REGRESSION** (dev $> 0.1200$): unexpected; investigate.
- **Diagnostic probes.**
  - $\alpha_d$ mobility per layer: do shallow layers prefer
    $\alpha_1$ and deep layers grow $\alpha_8$? Third data point on
    the depth-graded receptive-field thread (Stage-2 `gen2`,
    Stage-4 `rse_depth`).

### Stage 10.4 — `rwkv6_chanmix_bypass` (Avey partial-embedding bypass)

- **Family.** D — Feature-map / channel-side enrichment (channel-mix
  structural sub-family).
- **Paper review.** [TODO_FUTURE_IDEAS.md §Paper 3](TODO_FUTURE_IDEAS.md#paper-3--avey-dont-pay-attention-khatami-et-al-arxiv-250611305).
- **Hypothesis.** Linear head / non-linear tail split in ChannelMix
  relieves over-smoothing at depth; tests over-smoothing as a
  candidate failure mode complementary to the transition-side line.
- **Mechanism.**
  $[z_h, z_t] = \text{split}(W_k x_k; \rho=0.5)$;
  $\tilde z = [z_h \parallel \text{ReLU}^2(z_t)]$;
  $y = \sigma(W_r x_r) \odot [(1-\alpha) W_v \text{ReLU}^2(z) + \alpha W_v \tilde z]$
  with $\alpha \in \mathbb{R}$ per layer, zero-init.
- **Backbone / flag.** `rwkv6_chanmix_bypass`; flag
  `use_chanmix_bypass: bool = False`. Location:
  `src/models/rwkv6_channel_mix.py`.
- **Zero-regression contract.** $\alpha = 0$ at init recovers vanilla
  ChannelMix bit-exactly.
- **Parameters.** 6 scalars encoder-wide.
- **Wall-clock estimate.** $\approx 0$ overhead.
- **Comparison cohort.**
  - Primary: `rwkv6` vanilla.
  - Invariant anchor: `rwkv6_rse_strong_viscosity`.
- **Decision rule.**
  - **BREAK / MARGINAL**: compose with anchor to test orthogonality.
  - **PLATEAU**: document over-smoothing is not the binding failure
    mode at 6-layer / 7 M.
- **Diagnostic probes.**
  - $\alpha$ mobility per layer: does SGD grow it above 0.3?
  - Activation entropy at each layer (did ChannelMix contribute
    to entropy collapse pre-bypass?).

### Stage 10.5 — `rwkv6_orthogonal` (NCGRU Cayley-orthogonal, diagnostic)

- **Family.** B — Linear transition-operator family (strict superset
  of RSE's $SO(2)^{K/2}$ via full $SO(K)$).
- **Paper review.** [TODO_FUTURE_IDEAS.md §Paper 6](TODO_FUTURE_IDEAS.md#paper-6--crt--ncgru-arxiv-250500929).
- **Hypothesis (diagnostic).** Operator-family extension with
  **matched deployment shape to T2** (dense per-token) reveals
  whether the cross-experiment invariant is deployment-shape-specific
  or operator-family-specific.
- **Mechanism.** $G_{t,h} = e^{-\lambda_{t,h}} \cdot O_{t,h}$,
  $O_{t,h} = (I - A_{t,h})(I + A_{t,h})^{-1}$,
  $A_{t,h} = U_{t,h} V_{t,h}^\top - V_{t,h} U_{t,h}^\top$ skew.
  **Use rank-1 / Householder parametrisation** (not rank-4+) to hold
  parameter parity at 7 M.
- **Backbone / flag.** `rwkv6_orthogonal`; flag
  `use_cayley_orthogonal: bool = False`, `cayley_rank: int = 1`.
  Location: new scan branch in `src/models/rwkv6_time_mix.py`.
- **Zero-regression contract.** $U = V = 0$ at init ⇒ $A = 0$ ⇒
  $O = I$ ⇒ vanilla RWKV-6 bit-exact.
- **Parameters.** Rank-1: $2 \times K \times H \times L = 2 \times 64 \times 4 \times 6 = 3.1$ K
  encoder-wide. Well within parity. Rank-4 was ~3 M (breaks parity)
  — do not use.
- **Wall-clock estimate.** Matrix inverse $O(K^3)$ per token per head.
  3–5× anchor estimated. `torch.compile` helpful.
- **Comparison cohort (the load-bearing comparison).**
  - **Primary: `rwkv6_nonnormal_rse_viscosity` (T2, 0.1202 / 0.1200)**
    — same deployment shape (dense per-token), different operator
    family. The tiebreaker outcome distinguishes deployment-
    vs-operator-family hypotheses.
  - Family-B siblings: `rwkv6_sparse_nonnormal_rse_edge_only_viscosity`
    (S9B, 0.1218 / 0.1216).
  - Anchor: `rwkv6_rse_strong_viscosity`.
- **Decision rule.**
  - **Same band as T2 (dev 0.1195–0.1210)**: corroborates invariant
    at operator-family level — "dense per-token freedom doesn't
    convert regardless of operator family." Thesis-defensible finding.
  - **Significantly better than T2 (dev $< 0.1190$)**: reveals which
    operator subspace the RSE ceiling cares about. Re-open transition-
    family direction with orthogonal-specific structure.
  - **Significantly worse than T2 (dev $> 0.1230$)**: non-commutative
    operator is harder for SGD than polar non-normal; informative
    negative result.
- **Diagnostic probes.**
  - $\|U\|_F, \|V\|_F$ per layer (mobility).
  - Distribution of realised orthogonal rotation angles.
  - Spectral radius of $O_t$: stays at 1 within numerical tolerance
    (orthogonal matrices have spectrum on unit circle).

### Stage 10.6 — `rwkv6_pom_vlift` (PoM polynomial value-lift, diagnostic)

- **Family.** D — Feature-map / channel-side enrichment
  (element-wise polynomial variant).
- **Paper review.** [TODO_FUTURE_IDEAS.md §Paper 8](TODO_FUTURE_IDEAS.md#paper-8--pom-polynomial-mixer-picard-et-al-arxiv-260406129).
- **Hypothesis (diagnostic).** At parity-fitting $k = 2$, PoM's
  polynomial value-lift is a different parametrisation of the same
  quadratic function class as `hadamard_n2` and `qtail_lowrank_all`.
  Does the different parameterisation change the outcome?
- **Mechanism.**
  $\hat v_t = v_t + \sum_{p=2}^{k} \gamma_p \odot h(W_h x_t)^{\odot p}$,
  $\gamma_p = 0$ at init; state update unchanged.
  Thin config: $D = 64, k = 2$.
- **Backbone / flag.** `rwkv6_pom_vlift`; flag
  `use_pom_vlift: bool = False`, `pom_order: int = 2`,
  `pom_expansion: int = 64`. Location:
  `src/models/rwkv6_time_mix.py` pre-scan.
- **Zero-regression contract.** $\gamma_p = 0$ at init ⇒ $\hat v = v$
  bit-exact.
- **Parameters.** $W_h: 256 \times 64 = 16$ K/layer × 6 = 96 K
  encoder-wide (1.4 %).
- **Wall-clock estimate.** Negligible.
- **Comparison cohort.**
  - **Primary: `rwkv6_hadamard_n2` (0.1253 / 0.1251)** — nearest
    existing analog (diagonal element-wise square).
  - **Primary: `rwkv6_qtail_lowrank_all` (0.1238 / 0.1240)** — best
    feature-side result, cross-channel via Kronecker.
  - Vanilla reference: `rwkv6`.
- **Decision rule.**
  - **Matches hadamard_n2 (dev 0.1250–0.1260)**: PoM's $W_h$ mixing
    doesn't add over $k \odot k$. Diagnostic: quadratic family is
    saturated regardless of parametrisation.
  - **Matches qtail_lowrank_all (dev 0.1235–0.1245)**: PoM's $W_h$
    mixing achieves cross-channel expressivity comparable to
    low-rank Kronecker. Interesting parametrisation equivalence.
  - **Beats qtail_lowrank_all (dev $< 0.1230$)**: unexpected; PoM's
    parametrisation adds something the Kronecker family missed.
    Promote to composition with anchor.
- **Diagnostic probes.**
  - $\gamma_2$ mobility per layer.
  - Effective rank of $W_h$ (SVD).

### Stage 10.7 — `rwkv6_loglinear_rse_strong_viscosity` (composition, conditional)

- **Family.** A × B composition (multi-scale × transition-operator).
- **Paper review.** [TODO_FUTURE_IDEAS.md §Paper 1](TODO_FUTURE_IDEAS.md#paper-1--log-linear-attention-guo-et-al-arxiv-250604761).
- **Condition.** Only run if **Stage 10.1 lands ≥ MARGINAL**.
- **Hypothesis.** $M^S \odot M^H$ composition — complex semiseparable
  mask (RSE+viscosity) entry-wise multiplied with real hierarchical
  mask (log-linear) — combines two structurally-distinct mechanisms
  that have both individually shown signal. Load-bearing thesis
  question: does the anchor's CER ceiling yield to structural
  multi-scale readout?
- **Mechanism.** `_forward_recurrent_rse` extended to maintain $L$
  **complex** bucket states $c^{(\ell)}_{t,b}$ evolving with
  $z_{t,b} = e^{-\lambda_\text{eff} + i\theta_t}$ and Rayleigh
  viscosity $\lambda_\text{eff} = \lambda_\text{raw} + \eta \theta_t^2$
  intact inside $M^S$. Readout
  $y_t = \Re[\sum_\ell \lambda_t^{(\ell)} \sum_b \overline{r_{c,t,b}} c^{(\ell)}_{t,b}]$.
- **Backbone / flag.** `rwkv6_loglinear_rse_strong_viscosity`;
  combination of `use_loglinear=True`, `rse=True`,
  `rse_viscosity=True`, plus Stage-4 strong budget settings.
- **Zero-regression contract.** $\lambda_t^{(\ell)} \equiv 1$ at init
  ⇒ $\sum_\ell c^{(\ell)}_{t,b} = c_{t,b}$ ⇒ readout equals
  `rwkv6_rse_strong_viscosity` bit-exact (not vanilla).
- **Parameters.** ~57 K extra on top of anchor.
- **Wall-clock estimate.** ~3× anchor.
- **Comparison cohort.**
  - **Primary: `rwkv6_rse_strong_viscosity` (anchor, 0.1185 / 0.1177)**
    — the headline question.
  - Stage 10.1 result (same log-linear mechanism without RSE).
  - `rwkv6_p2rse_strong_viscosity` (0.1190 / 0.1196, tied anchor).
- **Decision rule.**
  - **BREAK** (dev $< 0.1165$, beyond 1σ below anchor): first real
    ceiling break since Stage 5. Multi-seed required before claiming.
  - **MARGINAL** (dev $\in [0.1165, 0.1195]$): tied anchor; useful
    composition-orthogonality data point.
  - **PLATEAU** (dev $\in [0.1195, 0.1230]$): structural multi-scale
    does not compound with anchor's operator-family advantage.
  - **REGRESSION** (dev $> 0.1230$): $M^S \odot M^H$ composition
    introduces identifiability issues — investigate.

### Stage 11.0 — Mamba-2 vanilla baseline at formal_v1 parity

- **Scope.** Establish a clean `mamba2` baseline at 7 M, 6 layers,
  30 ep on the same LibriSpeech clean-100 spine.
- **Backbone.** `mamba2` (already present: `mamba2_encoder.py`,
  `mamba2_block.py`).
- **Purpose.** Fill the Mamba-2 row in the master comparison matrix.
  No mechanism transfer yet.
- **Comparison cohort.** `rwkv6` (vanilla), `transformer_causal`.

### Stage 11.1 — Transfer surviving Stage-10 mechanisms to Mamba-2

Run promoted mechanisms from Stage 10 on Mamba-2. Candidate list:

- [ ] `mamba2_loglinear` (paper-direct; Triton kernel available)
- [ ] `mamba2_m2rnn_sparse` (paper-tested: Hybrid Mamba-2 + M²RNN)
- [ ] `mamba2_convshift_multidil` (Mamba-2 already has short DWConv;
      replace with multi-dilation)
- [ ] `mamba2_chanmix_bypass` (equivalent MLP/GLU slot)
- [ ] `mamba2_pom_vlift` (value-lift port)

### Stage 11.2 — Linear Attention baseline + NaLaFormer natural-home test

- **Scope.** Establish causal recurrent Linear Attention baseline,
  then test NaLaFormer (Paper 2) in its natural home.
- **Paper review.** [TODO_FUTURE_IDEAS.md §Paper 2](TODO_FUTURE_IDEAS.md#paper-2--nalaformer-meng-et-al-arxiv-250621137).
- **Rationale.** Current `blocks.py` `LinearAttentionLayer` is
  bidirectional parallel ELU+1 without L1 denominator. NaLaFormer's
  diagnosis (query-norm cancellation in Katharopoulos L1 form) does
  not apply there. To test NaLaFormer faithfully, first add a
  causal-recurrent Katharopoulos form with explicit L1 denominator,
  then swap ELU+1 → NaLaFormer $\phi_q, \phi_k$.
- **Sub-stages.**
  - **11.2a** — `linear_attn_causal` — causal recurrent Katharopoulos
    form with explicit $z_t = z_{t-1} + \phi(k_t)$ scalar running-sum
    state and explicit $o_t = \phi(q_t) S_t / \phi(q_t)^\top z_t$
    normalisation. Baseline only.
  - **11.2b** — `linear_attn_causal_nalaformer` — signed-power kernel
    $\tilde r = \text{sign}(r) \odot |r|^{p_t}, \tilde k = \text{sign}(k) \odot |k|^{\lambda_h}$
    from B's synthesis; zero-init $p = \lambda = 1$.

### Stage 11.3 — Transfer surviving Stage-10 mechanisms to Linear Attention

- [ ] `linear_attn_loglinear` (paper-direct; composition with
      causal LA mask)
- [ ] `linear_attn_convshift_multidil` (FFN-agnostic, position-pointwise)
- [ ] `linear_attn_chanmix_bypass` (FFN-after-attention slot)

### Stage 12 — LION bidirectional

LION baselines first, then only LION-compatible mechanisms. **Excluded
from Stage 12 (breaks parallel form):** P6 NCGRU-Cayley, P9 M²RNN.

- **Stage 12.0** — `lion` vanilla baseline.
- **Stage 12.1** — LION ablations per CLAUDE.md backbone naming:
  `lion_convshift`, `lion_headscale`, `lion_convshift_headscale`.
- **Stage 12.2** — `lion_lucid`, `lion_lucid_chunked` (corrected
  LUCID from CLAUDE.md §LUCID Preconditioner).
- **Stage 12.3** — `lion_delta` (causal-only delta from CLAUDE.md
  §Delta Rule).
- **Stage 12.4** — `lion_loglinear` (mirror-Fenwick design cell from
  Paper 1; design decision: causal $M^H_\text{fwd}$ lower triangle
  + suffix-Fenwick $M^H_\text{bwd}$ upper triangle, independent
  $\lambda^{(\ell, \cdot)}$).
- **Stage 12.5** — `lion_convshift_multidil` (direction-agnostic
  DWConv, inherits existing symmetric-padding handling).
- **Stage 12.6** — `lion_chanmix_bypass` (position-pointwise; mode
  independent).
- **Stage 12.7** — `lion_pom` (PoM's structural natural home via
  permutation-equivariance; standalone block as additive branch,
  with positional encoding to break set-equivariance).
- **Stage 12.8** — LION + NaLaFormer cosine-inhibit (if 11.2b
  shows signal; LION is closer to Katharopoulos parallel form
  than causal RWKV).

---

## 7. Metrics to log (mandatory per run)

Every Stage-10+ run must log the following. Missing metrics block the
run from being eligible for a verdict in §9.

### 7.1 Core CER / WER

- **Dev CER** per epoch (epochs 1, 5, 10, 15, 20, 25, 30 at minimum).
- **Test CER** at epoch 30 (single evaluation on `test-clean`).
- **Test WER** at epoch 30.

### 7.2 Timing

- **ms/iter** (forward+backward) — measure at $B = 10, T = 1200$ mels.
- **Time per epoch** (wall-clock, total minutes).
- **Total training time** (30 epochs).

### 7.3 Parameters

- **Exact parameter count** of the encoder (via `sum(p.numel() for p in encoder.parameters())`).
- **Extra parameters vs anchor** — breakdown by mechanism component.
- **Parameter-parity ratio** — extra / 7 M, as a percentage.

### 7.4 Memory

- **Peak GPU memory** during training (GB).

### 7.5 Mechanism-specific diagnostics (pre-registered per stage)

- Stage-specific probes enumerated in §6 must be measured at
  epoch 15 and epoch 30 minimum. Probe output goes into
  `outputs/stage10_XX_<backbone>/diagnostics.json`.

### 7.6 File layout (per run)

```
outputs/stage10_XX_<backbone>/
  history.csv          # per-epoch dev CER, train loss, lr
  config.yaml          # frozen config for reproducibility
  diagnostics.json     # mechanism-specific probes
  test_result.json     # final test CER, WER, ms/iter, memory
  params.txt           # parameter count breakdown
```

---

## 8. Governance and decision protocol

### 8.1 Pre-registered halt criteria (applies to all stages)

- **Matched-epoch 15 halt.** If dev CER at ep 15 is ≥ +0.006 behind
  the stage-specific primary reference in its comparison cohort,
  halt. Record as REGRESSION-track and run the partial diagnostic.
- **Explicit regression.** If dev CER at ep 30 > vanilla (0.1258) by
  more than 2σ (σ ≈ 0.0014), mark REGRESSION. Do not claim the
  mechanism as a ceiling-break candidate on any basis.

### 8.2 Multi-seed requirement for BREAK claims

Any single-seed result claiming to beat the anchor
(`rwkv6_rse_strong_viscosity` at 0.1185 dev) by more than 1σ must be
reproduced with 2 additional seeds before being logged as a ceiling
break in §9 or used to justify composition experiments.

### 8.3 When to compose with anchor

Only if a single mechanism lands **MARGINAL-or-better** on its
primary (vanilla or feature-side) reference. PLATEAU-band mechanisms
should not compose with the anchor until their individual failure
mode is understood.

### 8.4 When to halt a family line

If all mechanisms in a family converge to AMBIGUOUS-ENGAGED or
PLATEAU at our scale, document as "family saturated at 7 M / 30 ep /
clean-100" and stop adding variants within that family. Future runs
within that family require a scale change (more params, longer
training, or different dataset).

### 8.5 Logging discipline

- **Stage number is sticky.** Once a stage number is assigned
  (e.g., 10.1), its backbone name and decision rule are frozen.
  If the mechanism needs to be re-run with a parameter change, it
  gets a new stage number (e.g., 10.1b), not an overwrite.
- **Never overwrite a logged result in §9.** Add a new row; append
  "v2" to the stage label if a re-run is genuinely a correction.

---

## 9. Master result comparison matrix

**Rows ordered by stage.** Pre-populated with Stages 2-9 results from
[RESULTS.md](RESULTS.md) and the Stage-9 timing table in
[stages_2_9_summary.md](stages_2_9_summary.md). Empty cells (`—`) are
either not measured in the original run (timing for most early stages)
or reserved for upcoming runs.

**Key:** `σ` denotes seed noise, ~0.0014 on this spine. `ms/iter` at
$B = 10, T = 1200$ mels on RTX PRO 6000.

### 9.1 Stages 2-9 baselines (frozen, for comparison)

| Backbone | Family | Stage | Dev CER | Test CER | Test WER | Params | ms/iter | Time/ep | Epochs | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `rwkv6` | — | 2 | 0.1258 | 0.1263 | 0.3764 | ≈ 7 M | — | — | 30 | baseline |
| `rwkv6_trap` | B (solver, null) | 2 | 0.1263 | 0.1254 | 0.3746 | ≈ 7 M | — | — | 30 | PLATEAU |
| `rwkv6_trap_var` | B (solver, null) | 2 | 0.1261 | 0.1259 | 0.3749 | ≈ 7 M | — | — | 30 | PLATEAU |
| `rwkv6_gen2` | B (solver, null) | 2 | 0.1264 | 0.1254 | 0.3733 | ≈ 7 M | — | — | 30 | PLATEAU |
| `rwkv6_ab3` | B (solver, unstable) | 2 | 0.1299 | 0.1285 | 0.3789 | ≈ 7 M | — | — | 30 | REGRESSION |
| `rwkv6_convshift_trap` | A (input-side) | 2 | **0.1150** | **0.1150** | 0.3440 | ≈ 7 M | — | — | 30 | WON (cross-axis) |
| `rwkv6_rse` | B (transition) | 3 | 0.1251 | 0.1238 | 0.3705 | ≈ 7 M | — | — | 30 | MARGINAL |
| `rwkv6_rse_convshift` | A × B | 3 | **0.1145** | **0.1126** | 0.3382 | ≈ 7 M | — | — | 30 | **WON (absolute best)** |
| `rwkv6_rse_m2` / `m4` | A (flat multi-rate) | 3 | ~0.1245 | ~0.1240 | — | ≈ 7 M | — | — | 30 | PLATEAU |
| `rwkv6_rse_depth` | A / B (depth-graded) | 4 | 0.1207 | 0.1200 | 0.3593 | ≈ 7 M | — | — | 30 | WON |
| `rwkv6_rse_strong` | B (strong budget) | 4 | 0.1192 | 0.1188 | 0.3579 | ≈ 7 M | — | — | 30 | WON |
| `rwkv6_rse_depth_viscosity` | B | 5 | 0.1198 | 0.1198 | 0.3572 | ≈ 7 M | — | — | 30 | WON |
| **`rwkv6_rse_strong_viscosity`** | **B (ANCHOR)** | **5** | **0.1185** | **0.1177** | **0.3515** | ≈ 7 M + 768 | **94** | — | 30 | **WON / ANCHOR** |
| `rwkv6_p2rse` | B | 5 | 0.1250 | 0.1241 | 0.3740 | ≈ 7 M | — | — | 30 | PLATEAU |
| `rwkv6_p2rse_softmax` | B / δ-mixer | 5 | 0.1220 | 0.1215 | 0.3642 | ≈ 7 M | — | — | 30 | PLATEAU |
| `rwkv6_p2rse_strong_viscosity` | B | 6 | 0.1190 | 0.1196 | — | ≈ 7 M | — | — | 30 | tied anchor |
| `rwkv6_p2rse_indeplam_strong_viscosity` | B | 6 | 0.1394 | 0.1383 | — | ≈ 7 M + 200 K | — | — | 30 | REGRESSION (15σ) |
| `rwkv6_rmsnorm` | D | 6 | 0.1264 | 0.1252 | — | ≈ 7 M | — | — | 30 | PLATEAU |
| `rwkv6_hadamard_n2` | D | 6 | 0.1253 | 0.1251 | — | ≈ 7 M | — | — | 30 | PLATEAU |
| `rwkv6_qtail` | D | 6 | 0.1260 | 0.1240 | — | ≈ 7 M + large | — | — | 30 | near-MARGINAL test |
| `rwkv6_qtail_gamma` | D / δ-mixer | 6 | 0.1257 | 0.1249 | — | ≈ 7 M + large | — | — | 30 | PLATEAU |
| `rwkv6_qtail_gamma_dbeta` | D / δ-mixer | 6 | 0.1247 | 0.1245 | — | ≈ 7 M + large | — | — | 30 | PLATEAU-edge |
| `rwkv6_qtail_lowrank` | D | 6 | 0.1247 | 0.1242 | — | ≈ 7 M + mod | — | — | 30 | PLATEAU-edge |
| **`rwkv6_qtail_lowrank_all`** | **D (best feature-side)** | 6 | **0.1238** | **0.1240** | — | ≈ 7 M + mod | — | — | 30 | **MARGINAL** |
| `rwkv6_delta_warmstart` (retracted) | B (delta) | 6 | 0.1260 | 0.1256 | — | ≈ 7 M | — | — | 30 | implementation artefact |
| `rwkv6_rse_dphi_viscosity` (A1′) | B (readout-gauge) | 7 | 0.1217 | 0.1207 | — | ≈ 7 M + mod | — | — | 30 | AMBIGUOUS-ENGAGED (+2σ) |
| `rwkv6_delta_warmstart_fixed` (T1) | B (delta rank-1) | 8 | 0.1258 | 0.1256 | — | ≈ 7 M + mod | — | — | 30 | AMBIGUOUS-ENGAGED (tied-vanilla) |
| `rwkv6_nonnormal_rse_viscosity` (T2) | B (polar non-normal) | 8 | 0.1202 | 0.1200 | — | ≈ 7 M + mod | **271** | — | 30 | AMBIGUOUS-ENGAGED (+1σ) |
| `rwkv6_sparse_nonnormal_rse_viscosity` (S9A) | B (gated sparse) | 9 | 0.1467 (ep 15) | — | — | ≈ 7 M + mod | **271** | — | halted 15/30 | REGRESSION-track |
| `rwkv6_sparse_nonnormal_rse_edge_only_viscosity` (S9B) | B (edge-only) | 9 | 0.1218 | 0.1216 | — | ≈ 7 M + mod | **152** | — | 30 | REGRESSION STABILITY |

### 9.2 Stage 10 — planned (RWKV-6 causal continuations)

| Backbone | Family | Stage | Dev CER | Test CER | Test WER | Params | ms/iter | Time/ep | Epochs | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `rwkv6_loglinear` | A (log-scale readout) | 10.1 | **0.1240** | **0.1226** | 0.3647 | ≈ 7 M + 57 K | 172 (chunked) | 361 | 30 | **PLATEAU** — log-linear axis closed on causal RWKV-6 (see STAGE10_ANALYSIS.md §10.1) |
| `rwkv6_m2rnn_sparse` | **C (sole member)** | 10.2 | **0.1276** | **0.1264** | 0.3768 | ≈ 7 M + 17 K | 310 | 511 | 30 | **REGRESSION** (tied vanilla within 1.3σ) — non-linearity axis closed at this scale |
| `rwkv6_convshift_multidil` (causal) | A (input-side) | 10.3 | **0.1229** | **0.1224** | 0.3671 | ≈ 7 M + 18 K | 86 | 173 | 30 | **REGRESSION** vs `convshift_trap` primary — causality penalty +0.0076, not mechanism failure |
| `rwkv6_convshift_multidil_symmetric` | A (input-side, control) | 10.3-sym | **0.1153** | **0.1145** | **0.3439** | ≈ 7 M + 18 K | 102 | 177 | 30 | **MARGINAL** — ties `rwkv6_rse_convshift` abs-best without RSE; 10.3b composition gate open |
| `rwkv6_chanmix_bypass` | D (channel-mix) | 10.4 | **0.1251** | **0.1248** | 0.3701 | ≈ 7 M + 6 | 100 | 174 | 30 | **PLATEAU** (tied vanilla 0.5σ) — over-smoothing not the binding failure mode |
| `rwkv6_orthogonal` (rank-1) | B (Cayley-orthogonal) | 10.5 | — | — | — | ≈ 7 M + 3 K | TBD (~3-5× anchor) | — | 30 | TBD — diagnostic vs T2 |
| `rwkv6_pom_vlift` (thin) | D (polynomial) | 10.6 | — | — | — | ≈ 7 M + 96 K | TBD (~0 over) | — | 30 | TBD — diagnostic vs qtail |
| `rwkv6_loglinear_rse_strong_viscosity` | A × B (composition) | 10.7 | — | — | — | ≈ 7 M + 57 K | TBD | — | 30 | **CLOSED** — conditional on 10.1 ≥ MARGINAL, 10.1 landed PLATEAU |

### 9.3 Stage 11 — planned (Mamba-2, Linear Attention transfer)

| Backbone | Family | Stage | Dev CER | Test CER | Test WER | Params | ms/iter | Time/ep | Epochs | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `mamba2` (vanilla) | — | 11.0 | — | — | — | ~7 M | — | — | 30 | TBD (baseline) |
| `mamba2_loglinear` | A | 11.1 | — | — | — | — | — | — | 30 | TBD |
| `mamba2_m2rnn_sparse` | C | 11.1 | — | — | — | — | — | — | 30 | TBD |
| `mamba2_convshift_multidil` | A (input-side) | 11.1 | — | — | — | — | — | — | 30 | TBD |
| `mamba2_chanmix_bypass` | D | 11.1 | — | — | — | — | — | — | 30 | TBD |
| `mamba2_pom_vlift` | D | 11.1 | — | — | — | — | — | — | 30 | TBD |
| `linear_attn_causal` (baseline) | — | 11.2a | — | — | — | ~7 M | — | — | 30 | TBD (baseline) |
| `linear_attn_causal_nalaformer` | D (natural home for Paper 2) | 11.2b | — | — | — | — | — | — | 30 | TBD |
| `linear_attn_loglinear` | A | 11.3 | — | — | — | — | — | — | 30 | TBD |
| `linear_attn_convshift_multidil` | A (input-side) | 11.3 | — | — | — | — | — | — | 30 | TBD |
| `linear_attn_chanmix_bypass` | D | 11.3 | — | — | — | — | — | — | 30 | TBD |

### 9.4 Stage 12 — planned (LION bidirectional)

| Backbone | Family | Stage | Dev CER | Test CER | Test WER | Params | ms/iter | Time/ep | Epochs | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `lion` (vanilla) | — | 12.0 | — | — | — | ~7 M | — | — | 30 | TBD (baseline) |
| `lion_convshift` | A (input-side) | 12.1 | — | — | — | — | — | — | 30 | TBD |
| `lion_headscale` | B (per-head decay bias) | 12.1 | — | — | — | — | — | — | 30 | TBD |
| `lion_convshift_headscale` | A + B | 12.1 | — | — | — | — | — | — | 30 | TBD |
| `lion_lucid` | — (preconditioner) | 12.2 | — | — | — | — | — | — | 30 | TBD (CLAUDE.md corrected) |
| `lion_lucid_chunked` | — | 12.2 | — | — | — | — | — | — | 30 | TBD (CLAUDE.md corrected) |
| `lion_delta` | B (causal-only delta) | 12.3 | — | — | — | — | — | — | 30 | TBD (CLAUDE.md corrected) |
| `lion_loglinear` (mirror-Fenwick) | A | 12.4 | — | — | — | — | — | — | 30 | TBD (design cell) |
| `lion_convshift_multidil` | A (input-side) | 12.5 | — | — | — | — | — | — | 30 | TBD |
| `lion_chanmix_bypass` | D | 12.6 | — | — | — | — | — | — | 30 | TBD |
| `lion_pom` | D (PoM's natural home) | 12.7 | — | — | — | — | — | — | 30 | TBD |
| `lion_nalaformer` | D (Paper 2 on LION) | 12.8 | — | — | — | — | — | — | 30 | TBD |

### 9.5 Explicit exclusions

The following stage-10 mechanisms are **excluded from Stage 12** per
TODO_FUTURE_IDEAS.md §4.10 (non-linearity / non-commutativity
incompatible with parallel-form bidirectional execution):

- `lion_orthogonal` — NCGRU-Cayley breaks $e^{cs}$ factorisation.
- `lion_m2rnn` — M²RNN $\tanh(SW + kv^\top)$ breaks associativity.

Mamba-2 LION-style parallel-scan ports of these are similarly excluded.

---

## 10. Completion criteria (exit of planning phase)

Stage 10+ planning is complete when:

- [x] Families, priorities, and comparison cohorts are pre-registered.
- [x] Baselines consolidated and referenced.
- [x] Decision rules for each experiment are explicit and matched-epoch.
- [x] Metrics schedule is binding for every run.
- [x] Exclusions (LION-breaking mechanisms) are documented.

**Planning is complete. Big experiments begin with Stage 10.1.**

---

## 11. Change log for this document

- **2026-04-21** — Initial creation. Drafted from Phase-4 synthesis in
  TODO_FUTURE_IDEAS.md, Stages-2-9 summary, and the 9 paper reviews.
  All Stage 10+ runs must reference this version or a successor.
