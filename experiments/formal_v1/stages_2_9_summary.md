# Stages 2–9 summary

Sequential log of the causal RWKV-6 experiment chain on LibriSpeech
train-clean-100, seed 42, 30 epochs, RTX PRO 6000 Blackwell.

All runs share the same config spine (`d_model=256`, 6 layers, 4 heads,
head_size=64, AdamW lr=3e-4, cosine+1000-step warmup, SpecAugment LD,
batch ≤ 300 s duration, grad clip 5). Seed noise on this codebase
σ ≈ 0.0014.

Vanilla reference:

| Run | Dev CER | Test CER |
|---|---:|---:|
| `rwkv6` (vanilla, ZOH) | 0.1258 | 0.1263 |

Stage-by-stage per-run intermediate reports (ENV_DISCRETIZATION.md,
RSE_DISTRIBUTION_ANALYSIS.md, STAGE2–9 plan / results / analysis docs)
have been consolidated into this file and deleted from the tree.
Cross-references throughout the text point at the stage sections below
rather than the retired documents. Implementation pointers remain
against live source files.

---

## Stage 2 — higher-order discretization

**Hypothesis.** The RWKV-6 state update $S_t = W_t \odot S_{t-1} + k_t v_t^\top$
is a zero-order hold (ZOH) on an underlying ODE. Higher-order implicit
(trapezoidal) or explicit multistep (Adams–Bashforth 3) integrators
should recover the same function class at lower numerical error and —
if CER is limited by solver precision — deliver an improvement.

**Variants tested** (all in
[src/models/rwkv6_time_mix.py](src/models/rwkv6_time_mix.py) via the
`discretization` flag, driven by backbone name substring):
`trap` (½,½), `trap_var` (geometric-mean decay), `gen2` (learnable
per-head α₀, α₁, init to ZOH), `ab3` (Adams-Bashforth-3 with decay clamp),
`convshift_trap` (input-side depthwise-conv + trap).

**Results** (from STAGE2_RESULTS.md):

| Backbone | Dev CER | Test CER | Test WER | Δ test |
|---|---:|---:|---:|---:|
| `rwkv6` (ZOH, ref) | 0.1258 | 0.1263 | 0.3764 | ref |
| `rwkv6_trap` | 0.1263 | 0.1254 | 0.3746 | −0.7 % |
| `rwkv6_trap_var` | 0.1261 | 0.1259 | 0.3749 | −0.3 % |
| `rwkv6_gen2` | 0.1264 | 0.1254 | 0.3733 | −0.7 % |
| `rwkv6_ab3` | 0.1299 | 0.1285 | 0.3789 | +1.8 % |
| `rwkv6_convshift_trap` | 0.1150 | 0.1150 | 0.3440 | −9.0 % |

**Interpretation.** Pure solver changes (`trap`, `trap_var`, `gen2`)
produced no measurable CER gain — all within ±0.7 %. `ab3` regressed
(+1.8 %) because explicit AB3 is only conditionally stable.
`convshift_trap` improved −9.0 %, but the gain is attributable to the
ConvShift input-side DWConv rather than the discretization — a cross-
axis mechanism.

**Why the null is structural.** Any reparameterisation inside the
diagonal-decay group $(\mathbb{R}_+)^K$ is absorbable by the existing
learnable projections `W_k`, `W_v` and decay LoRA. The feasible
function class is invariant under the multistep family. **To break the
ceiling the transition group itself must change** — this framing
motivates Stage 3.

**gen2 diagnostic** (STAGE2_RESULTS §3): per-head α₁ (the previous-step
weight) grew systematically with depth (0.023 at L0 → 0.107 at L5, one
head per layer reaching 0.3+). The **multi-scale depth hierarchy**
pattern was observed here for the first time — it reappears in
Stages 4, 6, 8.

---

## Stage 3 — Rotational State Evolution (RSE)

**Hypothesis.** Replace the per-channel scalar decay with a
block-diagonal $2\times 2$ decay-rotation transition:
$G_{t,b} = e^{-\lambda_{t,b}} R(\theta_{t,b})$. Eigenvalues become
$e^{-\lambda \pm i\theta}$; the transition Lie group changes from
$(\mathbb{R}_+)^K$ to $\mathrm{SO}(2)^{K/2} \times (\mathbb{R}_+)^{K/2}$
— strictly larger, not continuously reparameterisable from RWKV-6
(Proposal A Thms 1–2).

**Implementation.**
[src/models/rwkv6_time_mix.py](src/models/rwkv6_time_mix.py)
`_forward_recurrent_rse` — chunked scan packing row pairs into complex
scalars $c_t = z_t c_{t-1} + k_c v^\top$ with $z_t = e^{-\lambda + i\theta}$.
Within a chunk of 64 tokens the complex attention factor
$A[t,s,b] = \exp(\mathrm{cumlog}_z[t] - \mathrm{cumlog}_z[s])$ is
parallel; inter-chunk state is carried serially.

**Initialisation (corrected v2).** First attempt with $\gamma_\theta = \pi/2$
and $\theta_\text{base} \sim \mathcal{U}(-\pi/4, \pi/4)$ was unstable
(cumulative phase std ≈ 230 rad ⇒ chaotic interference; CER stuck at
0.85). Corrected to $\gamma_\theta = \pi/4$, $\theta_\text{base} \sim
\mathcal{U}(-\pi/16, \pi/16)$ — caps per-step rotation below per-step
phase aliasing; LoRA outputs zero-init.

**Results**:

| Backbone | Dev CER | Test CER | Δ test vs vanilla | vs convshift_trap |
|---|---:|---:|---:|---:|
| `rwkv6_rse` | 0.1251 | 0.1238 | −2.0 % | +7.7 % |
| `rwkv6_rse_convshift` | 0.1145 | 0.1126 | −10.8 % | −2.1 % |

**Per-epoch trajectory vs vanilla** (dev CER):

| Epoch | `rwkv6` | `rwkv6_rse` | `rwkv6_rse_convshift` |
|---:|---:|---:|---:|
| 5 | ~0.236 | 0.2365 | 0.2261 |
| 10 | ~0.175 | 0.1748 | 0.1639 |
| 15 | ~0.146 | 0.1485 | 0.1437 |
| 20 | ~0.131 | 0.1360 | 0.1236 |
| 25 | ~0.125 | 0.1271 | 0.1167 |
| 30 | 0.1258 | 0.1251 | 0.1147 |

**Paradox.** The mathematically clean function-class extension delivered
only −2.0 % — well under Proposal A §9 P1's ≥3 % prediction. Stage-3.5
multi-rate variants (`rse_m2`, `rse_m4`) tracked single-scale RSE then
plateaued at the same ~0.125 by ep 15: adding parallel rotation
channels did not change the asymptote.

**Post-mortem** (RSE_DISTRIBUTION_ANALYSIS.md):
the static parameter-mobility diagnostic on the trained `rwkv6_rse`
checkpoint showed that per-head/per-block $\theta$ was moving, but the
rotation budget $\gamma_\theta = \pi/4$ was sub-utilised at shallow
layers (mean $|\theta|$ ≈ 0.01 at L0) and pressed against the clip at
deep layers (max $|\theta|$ near π/4 at L4–L5). Conclusion: a *uniform*
rotation budget across depth was the binding constraint, not the
mechanism itself.

---

## Stage 4 — rotation budget refinements (depth-graded, strong)

**Hypothesis.** The Stage-3 ceiling is a parameterisation ceiling of
the rotation budget, not the RSE mechanism. Two variants:
- `rse_depth` — per-layer graded clip (π/8 at L0,L1 → π/4 at L2,L3 →
  π/2 at L4,L5), LoRA dim 16 → 32 → 48, matching the diagnosed depth
  hierarchy.
- `rse_strong` — uniform larger budget (π/2 clip, LoRA 48 everywhere).

**Results** (reported alongside Stage 5):

| Backbone | Dev CER | Test CER | Test WER | Δ test vs vanilla |
|---|---:|---:|---:|---:|
| `rwkv6_rse_depth` | 0.1207 | 0.1200 | 0.3593 | −5.0 % |
| `rwkv6_rse_strong` | 0.1192 | 0.1188 | 0.3579 | −5.9 % |

**First clear ceiling break** — both variants ≥ 3σ under vanilla and
Stage 3 RSE. `rse_strong` edges `rse_depth` by ~σ; zero-extra-parameter
depth allocation tied strong-budget within noise.

**Interpretation.** Proposal A's RSE claim was correct in function-class
terms; the deployment was budget-limited at Stage 3. Both refinements
are inside the same Lie group $\mathrm{SO}(2)^{K/2} \times
(\mathbb{R}_+)^{K/2}$ — the budget axis is orthogonal to the
mechanism axis.

---

## Stage 5 — paired-pole RSE (P²-RSE) and viscosity coupling

**Restriction set identified at end of Stage 4**
(STAGE5_RESULTS.md §1):

- **R1.** Each 2×2 block carries *one* complex pole → single damped
  sinusoid. Formants in speech are 2-pole resonators.
- **R2.** Readout discards quadrature component
  $\mathrm{Im}(\overline{r_b} \cdot S_b)$ — phase-advanced by π/2.
- **R3.** Bonus term `u` is real-valued and axis-aligned while state
  content sits in rotated eigenbasis — a write-side gauge inconsistency.
- **R4.** $\theta$ budget is a hard clip; shallow layers run far below,
  deep layers press against it — piecewise approximation.

### Phase 1 — paired-pole, single budget (R1)

**Hypothesis (H★).** Two complex poles per block $(z^{(1)}_t, z^{(2)}_t)$
sharing $\lambda_t$, independent $\theta^{(j)}_t$, phase-complementary
init ($\theta^{(2)}_\text{base} = -\theta^{(1)}_\text{base}$), mixed
through a real data-dependent $\beta_m(x_t)$. Predicted: ≤ 0.1160.

**Variants.** `rwkv6_p2rse` (linear β), `rwkv6_p2rse_softmax`
(convex β). Both implemented in
[src/models/rwkv6_time_mix.py](src/models/rwkv6_time_mix.py)
via `_forward_recurrent_p2rse`.

**Results**:

| Run | Dev CER | Test CER | Classification |
|---|---:|---:|---|
| `rwkv6_p2rse` (linear β) | 0.1250 | 0.1241 | PLATEAU |
| `rwkv6_p2rse_softmax` | 0.1220 | 0.1215 | PLATEAU |

Softmax variant −2.5 % over Stage-3 `rse` (0.1251). Both under both
Stage-4 variants (`rse_strong` 0.1192, `rse_depth` 0.1207).

**Diagnosis.** 2-pole dynamics alone insufficient at the original
budget. The simplex constraint of softmax-β provided a *stabilising*
prior the linear variant lacked — falsifies the critic hypothesis that
the convex constraint was the bottleneck. Maps to row O5 of the
Phase-1 decision table in STAGE5_PLAN §6: *"2-pole dynamics alone
insufficient"*.

**Bug note (post-training).** `RWKV6Encoder.init_state` hard-coded the
single-pole shape `(B, H, K, K)`. Paired-pole needs `(2, B, H, K, K)`.
Chunked-carry evaluation aborted with IndexError; full-utterance
evaluation unaffected (called with `state=None`). Fix dispatched
on time-mix kind; pre-fix checkpoints recovered via
`scripts/recover_phase1_test.py`.

### Phase 2 — P²-RSE × Stage-4 budget (terminated at ep 7–8)

**Hypothesis (H★₂).** Multiplicative stacking of Phase 1's
−2.5 % gain with Stage-4's budget refinements would land
`p2rse_strong` at ~0.1161, `p2rse_depth` at ~0.1177 (break/marginal).

**Observation at ep 7–8.** Trajectory tracked Stage-4 `rse_depth`
reference (0.2282 ep5, 0.1695 ep10) but **not decisively ahead**.
Following the same tail delta, projected final ≈ 0.1175 — marginal,
not break, within 1σ of Stage-4 ceiling.

**Decision.** Budget redirected to Phase 3 (viscosity) whose predicted
gain is independent of Phase 1 signal magnitude. Partial trajectories
retained in `outputs/stage5_0{3,4}_*/history.csv`.

### Phase 3 — Rayleigh viscosity coupling (R4)

**Hypothesis (H★₃).** Replace the hard clip $\gamma_\theta$ with a
data-coupled soft self-regulating damping. Under a classical damped
harmonic oscillator with Rayleigh dissipation, angular frequency and
damping are tied: larger rotation should cost more damping. Parameterise

$$\lambda_{\text{eff}} = \lambda_{\text{raw}} + \eta_{h,b}\,\theta_t^2$$

with $\eta_{h,b}$ a learnable per-(head, block) scalar, zero-init. When
$\eta = 0$ the mechanism reduces exactly to Stage-4 RSE. When $\eta > 0$,
high-frequency rotations decay faster — self-regulating phase coherence.

**Implementation.** `_forward_recurrent_rse` viscosity branch
(flag `rse_viscosity`), applied before the cumulative log-z.

**Results**:

| Run | Dev CER | Test CER | Test WER | Δ dev vs S4 | Δ test vs S4 |
|---|---:|---:|---:|---:|---:|
| **`rwkv6_rse_strong_viscosity`** | **0.1185** | **0.1177** | 0.3515 | **−0.59 %** | **−0.93 %** |
| `rwkv6_rse_depth_viscosity` | 0.1198 | 0.1198 | 0.3572 | −0.75 % | −0.17 % |

**Per-epoch trajectory** (`rse_strong_viscosity` vs Stage 3 RSE):

| Epoch | `rwkv6_rse` | `rse_strong_viscosity` | Δ |
|---:|---:|---:|---:|
| 5 | 0.2365 | 0.2279 | −0.0086 |
| 10 | 0.1748 | 0.1683 | −0.0065 |
| 15 | 0.1485 | 0.1441 | −0.0044 |
| 20 | 0.1360 | 0.1277 | −0.0083 |
| 25 | 0.1271 | 0.1200 | −0.0071 |
| 30 | 0.1251 | **0.1185** | −0.0066 |

At the close of Stage 5, `rwkv6_rse_strong_viscosity` recorded dev
0.1185 / test 0.1177, with **+768 scalars** (0.013 % encoder params)
vs Stage-4 `rse_strong`. Gap to
pre-registered BREAK threshold (0.1160): +0.0025 dev, +0.0017 test —
~2σ band. Multi-seed not yet run.

---

## Stage 6 — EXPRESSIVENESS paper feature-side adaptations

**Paper** (Mongaras & Larson 2025). Two claims:
- §4.5: the softmax denominator $G_t$ acts primarily as a stabilising
  vector norm; L2/RMS/LayerNorm replacements should work.
- §3.2: linear attention is the $n=1$ truncation of softmax's Taylor
  series; $n \ge 2$ terms add cross-channel $k_i k_j$ interactions
  ($i \ne j$) that element-wise feature maps $\varphi(k)$ cannot
  recover.

**Variants tested** (all implemented in
[src/models/rwkv6_time_mix.py](src/models/rwkv6_time_mix.py) via
substring-triggered flags; `_paper_n2_branch` for the Taylor term).

### Axis A: feature-side results

| Backbone | Dev CER | Test CER | Δ test vs vanilla | Classification |
|---|---:|---:|---:|---|
| `rwkv6_rmsnorm` | 0.1264 | 0.1252 | −0.87 % | PLATEAU |
| `rwkv6_hadamard_n2` (diag $k\odot k$) | 0.1253 | 0.1251 | −0.95 % | PLATEAU |
| `rwkv6_qtail` (full K², top-2 layers) | 0.1260 | 0.1240 | −1.82 % | PLATEAU dev / near-MARGINAL test |
| `rwkv6_qtail_gamma` (+per-head γ decay coupling) | 0.1257 | 0.1249 | −1.11 % | PLATEAU |
| `rwkv6_qtail_gamma_dbeta` (+data-dep β) | 0.1247 | 0.1245 | −1.43 % | PLATEAU-edge |
| `rwkv6_qtail_lowrank` (K'=16, top-2 layers) | 0.1247 | 0.1242 | −1.66 % | PLATEAU-edge |
| **`rwkv6_qtail_lowrank_all`** (K'=16, all 6 layers) | **0.1238** | **0.1240** | **−1.82 %** | **MARGINAL (dev)** |

**Key findings** (STAGE6_ANALYSIS.md,
STAGE6_CONCLUSIONS.md):
- **`rmsnorm` ≈ `hadamard_n2`** on test — diagonal $k^2$ adds essentially
  nothing. Expected: the cross-channel thesis requires full Kronecker,
  not diagonal. This is suggestive, but not a perfectly matched
  isolation, because `hadamard_n2` runs at all layers while full `qtail`
  is top-2 only.
- **`qtail` > both controls** by ~0.0011 on test (+full σ). Asymmetric
  between dev and test — Kronecker features generalise slightly
  differently from linear features.
- **γ refinement (H6.5).** Paper's Taylor derivation imposes no decay
  on order-$n$ state. Natural Kronecker lift has $w_\text{pair}[i,j] =
  w_i + w_j$; γ generalises to $\gamma_h \cdot (w_i + w_j)$, init 1.0.
  SGD drove γ into a **bimodal per-head distribution** (one head per
  layer at γ ≈ 0.57, others near γ ≈ 1) with a depth gradient (deeper
  → smaller mean γ). Matches the Stage-2 `gen2` depth-hierarchy pattern.
  CER did not move beyond σ — γ refinement is a *mechanism-level finding*,
  not a CER delta at this scale.
- **Data-dep β (R2).** Per-head/per-token $\beta_{q,t} = \beta_\text{static} +
  W_\beta x_t$, zero-init. In this single seed, γ and β move jointly:
  smaller learned β coincides with γ < 1, while larger β coincides with
  γ > 1. Treat this as an observed co-adaptation pattern, not a law.
- **Low-rank Kronecker** (K'=16): learned per-head projections
  $U_r^{(h)}, U_k^{(h)} \in \mathbb{R}^{K \times K'}$ collapse $r, k$ to
  $K'=16$ before the outer product → K'² = 256 features instead of
  $K^2 = 4096$. **Matches full-qtail CER at 16 % VRAM and 37 % wall-clock.**
  The safe reading is that a smaller learned quadratic subspace with a
  coarser decay law suffices at this scale; this is weaker than an exact
  Eckart–Young / effective-rank claim for the full K² operator.
- **All-layer deployment.** Adding Kronecker at all 6 layers (vs top-2)
  gives a further consistent ~1σ CER drop. Best Stage-6 result
  0.1238 dev / 0.1240 test.

### Axis B: pole-manifold × viscosity (Phase 2b)

**Hypothesis.** The Phase-1 P²-RSE under-performance diagnosis
(STAGE5_RESULTS §6.4(1)) flagged shared-λ as the suspect constraint.
Phase 2b adds independent decay LoRAs per pole (indep-λ) on top of the
Stage-5 winning composition.

Implementation:
[src/models/p2rse_indep_lambda.py](src/models/p2rse_indep_lambda.py)
`p2rse_indep_lambda_scan` — real-arithmetic fused kernel with both
poles. Zero-regression at init: pole-2 λ = clone of pole-1, LoRA zero.

**Result**:

| Run | Dev CER | Test CER | Δ vs anchor (dev) |
|---|---:|---:|---:|
| `rwkv6_p2rse_strong_viscosity` (shared-λ control) | 0.1190 | 0.1196 | +0.42 % (tied) |
| `rwkv6_p2rse_indeplam_strong_viscosity` | **0.1394** | **0.1383** | **+17.6 %** (REGRESSION) |

**Diagnosis.** 15σ regression. Shared-λ is stable; indep-λ LoRA
degrades badly despite zero-regression-at-init. The later Stage-8
reading is that this is better described as an identifiability /
parameterisation mismatch than simply "too many parameters" against a
shared-viscosity, shared-readout structure.

**Delta-rule diagnostic (Stage 6 supplementary).** `rwkv6_delta_warmstart`
with $a_0 = -5$ init (TODO_DELTA_RULE Tier-1 fix) landed at dev 0.1260 /
test 0.1256. **Documented as a null at the time.** Stage 8 T1 later showed
this was an implementation artefact — the recurrent path did not branch
on `use_delta_rule` at that commit.

### Stage 6 summary

The strongest Stage-6 feature-side row is `rwkv6_qtail_lowrank_all` at
0.1238 dev / 0.1240 test (MARGINAL). It remains above the Stage-5
transition-side line `rse_strong_viscosity` at 0.1185 dev / 0.1177
test. The later Stage-7 reading treats qtail as a productive orthogonal
axis, not a foundation; the original "THE foundation" wording in
Stage-6 conclusions §3.1 was over-claimed.

---

## Stage 7 / 7A — readout-gauge completion

**Observation motivating Stage 7.** Stage 6 left two distinct readings:
the lowest CER line was transition-side (`rse_strong_viscosity`), while
the clearest transferable idea was feature-side (low-rank qtail). These
should not be conflated. Stage 7 commits to *single-concept refinement
only* — no broad stacking.

### 7A central hypothesis (H_A)

> The Stage-5 RSE line is limited not by transition capacity alone, but
> by a mismatch between the rotated latent state and the axis-aligned
> observation / write geometry. A minimal learnable gauge alignment on
> the readout and bonus terms should improve
> `rwkv6_rse_strong_viscosity` beyond 0.1185 dev without meaningful
> extra compute.

**Pre-training diagnostic** (STAGE7_DIAGNOSTICS.md)
run on `p2rse_sv` (closest saved checkpoint in the strong+viscosity
family, within σ of anchor, used explicitly as a proxy rather than the
true anchor). Three probes:
- D1 θ budget saturation: max 2 % fraction above 0.95·clip. **Budget
  not binding.**
- D2 readout quadrature: proxy global $|\mathrm{Im}|/|\mathrm{Re}|$
  grows monotonically with depth (0.21 L0 → 0.70 L5). Per-block ratio ≈ 1 —
  i.e. per-block phase is approximately uniformly distributed.
- D3 viscosity coupling: mean η·θ² is 3–8 % of λ_raw per layer, positive
  correlation between θ² and λ_eff. Engaged but modest.

**Critical insight from D2.** A constant per-(head, block) phase
$\phi_{h,b}$ cannot usefully reshape a uniformly-distributed per-block
phase — it is absorbable by the existing complex `r_proj`. Therefore
**A1 (static φ) would be dead by reparameterisation**. Only a
*data-dependent* phase $\phi_{t,h,b} = W_\phi x_t$ is structurally novel.

### A1′ — data-dependent readout phase

**Implementation.**
[src/models/rwkv6_time_mix.py](src/models/rwkv6_time_mix.py): flag
`use_data_dep_readphase`, parameter `readphase_proj` (zero-init Linear).
Readout becomes
$$y_t = \Re\left(\sum_b e^{-i\phi_{t,h,b}}\,\overline{r_{c,b}}\,S_{t,b}\right).$$
Zero-regression at init via $W_\phi = b_\phi = 0 \Rightarrow \phi \equiv 0$.

Backbone `rwkv6_rse_dphi_viscosity`, κ (soft-clip) = π.

**Result**
(STAGE7A_RESULTS.md):

| Run | Dev CER | Test CER | Δ vs anchor (dev) | Band |
|---|---:|---:|---:|---|
| `rwkv6_rse_strong_viscosity` (re-run) | 0.1189 | 0.1198 | ref | — |
| `rwkv6_rse_dphi_viscosity` (A1′) | **0.1217** | **0.1207** | **+0.0028 (~2σ)** | AMBIGUOUS → regression edge |

**Post-training diagnostic finding.**

| Layer | anchor Im/Re | A1′ **pre**-rotation | A1′ **post**-rotation | Δ (post − pre) |
|---|---:|---:|---:|---:|
| L0 | 0.27 | 0.25 | **0.43** | +0.19 |
| L5 | 0.56 | 0.59 | **0.93** | +0.34 |

**Interpretation.** $W_\phi$ moved substantially (\|F\| 5.5–6.4 per
layer; realised \|φ\| p99 ≈ 2 rad) — SGD engaged the mechanism — but
the resulting action **pushed MORE content into the discarded imaginary
axis**, not less. The "discarded quadrature" the diagnostic measured
was not lost signal; it was a cross-block cancellation structure the
anchor model already exploits as its implicit gauge. Forcing per-token
phase rotation broke that alignment. **Decision-tree outcome 3** per
the pre-registered rule: *mechanism moved but did not do its intended
job*. The true-anchor follow-up in `STAGE7A_RESULTS.md` also replaced
the proxy D2 headline with a milder but still substantial depth pattern
(L5 anchor Im/Re 0.56 rather than proxy 0.70).

### 7A-plus — cross-layer complex residual (spec only)

STAGE7A_PLUS_SPEC.md drafts a complex residual
stream (parallel to the real stream, no LN) that would carry `c_state`
between layers so successive SO(2) rotations compose rather than reset.
Decision rule pre-registered: *implement only if A1′ shows signal*.
A1′ did not. **Spec remains on file; not implemented.**

---

## Stage 8 — transition-geometry pivot

**Framing** (post-Stage 7A) (STAGE8_PLAN.md): the
Stage-8 plan pivots back to transition geometry after the readout-gauge
line is falsified. Two parallel tracks:

- **T1 — recurrent delta wiring.** Code audit revealed the historical
  `rwkv6_delta_warmstart` run (Stage 6) did not branch on `use_delta_rule`
  in `_forward_recurrent` at training commit 3aebd56. The Stage-6 "delta
  null" was an implementation artefact. Retracted. T1 fixes the wiring
  and re-runs.
- **T2 — strict superset of RSE in polar parameterisation.**
  4-DOF 2×2 block $(λ, θ, ρ, ψ)$ per head-block-token, extending the
  normal family $e^{-λ} R(θ)$ to the non-normal class via the symmetric
  anisotropy factor $P(ρ, ψ) = R(ψ)^\top \mathrm{diag}(e^ρ, e^{-ρ}) R(ψ)$.

### Track 1 — recurrent delta (wired correctly)

**Implementation.**
[src/models/rwkv6_time_mix.py](src/models/rwkv6_time_mix.py):
`_recurrent_delta_scan` (chunked affine associative scan, O(log T_c)
depth via Hillis–Steele). Rank-1 erase:
$S_t = w_t \odot (I - \beta_t kk_t kk_t^\top) S_{t-1} + k_t v_t^\top$
with $\beta_t = g_\delta \cdot \sigma(\tilde\beta_t) \cdot 2$, hard gate
`delta_recurrent_gate` zero-init per head (bit-exact zero-regression
to vanilla RWKV-6).

Backbone `rwkv6_delta_warmstart_fixed`.

**Result**
(STAGE8_RESULTS.md §6):

| Run | Dev CER | Test CER | Δ test vs vanilla |
|---|---:|---:|---:|
| `rwkv6` (vanilla anchor) | 0.1258 | 0.1263 | ref |
| `rwkv6_delta_warmstart_fixed` | **0.1258** | **0.1256** | −0.55 % (tied within σ) |

**Diagnostic (D5–D7)** per
[scripts/stage8_diagnostics.py](scripts/stage8_diagnostics.py):

| Layer | per-head g_δ (max) | β_eff p95 | Erase upper bound |
|---:|---:|---:|---:|
| L0 | 0.421 | 0.690 | 0.059 |
| L3 | 0.567 | 1.239 | 0.183 |
| **L5** | **0.650** | **1.482** | **0.242** |

**Interpretation.** All 24 gates (6 × 4) moved from zero. β_eff grew
depth-graded (L0 mean 0.11 → L5 mean 0.67; L5 p95 = 1.48). Up to 24 % of
state norm is erased per token at L5. **Mechanism is engaged, state is
modified, CER does not benefit.** Genuine engaged-null for recurrent
delta at 7 M / clean-100 / 30-ep. Prior "delta is null" claim is now
evidence, not implementation artefact.

### Track 2 — non-normal RSE (polar-form 2×2 extension)

**Block transition**:
$$G_{t,b} = e^{-\lambda_{t,b}}\,R(\psi_{t,b})^\top\,\mathrm{diag}(e^{\rho_{t,b}}, e^{-\rho_{t,b}})\,R(\psi_{t,b})\,R(\theta_{t,b}).$$

Zero-init ρ (via zero-init LoRA + zero base) ⇒ $P = I$ ⇒ exact RSE
reduction. Stability clip $|\rho| \le \kappa \cdot \text{softplus}(\tilde\lambda)$
with κ = 0.6 (conservative; spectrum stable iff $\rho^2 < \lambda^2 + \theta^2$).
Extended viscosity $\lambda_\text{eff} = \lambda + \eta \theta^2 + \mu \rho^2$.

**Implementation.**
[src/models/rwkv6_time_mix.py](src/models/rwkv6_time_mix.py):
`_recurrent_nonnormal_rse_scan` + `_affine_prefix_scan` — exact
associative scan over affine pairs $(G, U) \otimes (G', U') = (GG', GU' + U)$
via Hillis–Steele doubling. Within-chunk parallel (~log T_c passes),
inter-chunk serial state carry. Gave ~6.7× speedup vs initial sequential
implementation (from 1840 ms/iter → 273 ms/iter).

Backbone `rwkv6_nonnormal_rse_viscosity`.

**Result**:

| Run | Dev CER | Test CER | Δ vs anchor (dev) | Band |
|---|---:|---:|---:|---|
| `rwkv6_rse_strong_viscosity` (anchor) | 0.1189 | 0.1198 | ref | — |
| **`rwkv6_nonnormal_rse_viscosity`** | **0.1202** | **0.1200** | +0.0013 (≈1σ) | AMBIGUOUS |

**Diagnostic (D5–D9)**:

- **D5 ρ/ψ mobility.** $\|{\rho_\text{LoRA}}\|_F$ 18–30 per layer; μ
  moved signed at mid-layers. Massive engagement.
- **D6 realised ρ.** Mean $|\rho|$ concentrates at L0 (0.18), L5 (0.11);
  middle layers 0.05–0.07. L0 p99 = 2.17 (saturating). L5 frac-near-clip
  = 0.61.
- **D7 non-normality score** $\|G^\top G - G G^\top\|_F / \|G\|_F^2$.
  L0, L5 mean ≈ 0.10; middle 0.04–0.05; L1 is comparable to L0 on this
  metric. Max > 1.0 at L0/L4/L5. **Genuinely non-normal, with strongest
  realised use at the edges once D6 and D9 are included.**
- **D8 spectral radius.** Max spectral radius stays at ~1 within fp32
  tolerance (one layer reaches 1.000021). Around 6 % of L0 and L2
  tokens have $|\lambda| > 0.99$. Borderline, near-marginal, but not a
  blowup.
- **D9 cross-head specialisation.** L0 per-head: [0.057, 0.077, 0.144,
  **0.456**] — one head at |ρ|=0.46, others 0.06–0.14. Bimodal pattern,
  same shape as Stage-6 qtail-γ.

**Interpretation.** Mechanism engaged, used selectively (concentrated
at edge layers + bimodal head specialisation), not rewarded enough by
CER. **Invariant across A1′, T1, T2**:

> Dense per-token mechanism freedom engages SGD but does not translate
> into CER gain at 7 M / clean-100 / 30-ep.

The three mechanisms engage via different operators (readout phase,
state-erase, non-normal transition) but land in the same band —
mobility non-zero, CER near-anchor.

---

## Stage 9 — sparse edge-layer specialist transition

**Hypothesis (H9)** (STAGE9_PLAN.md):

> Restricting the non-normal extension to **sparse, gated, edge-layer,
> head-specialised** configuration — matching the structure T2's SGD
> discovered but could not reward — will produce a cleaner optimisation
> landscape, retain the informative non-normal directions, and break
> the RSE ceiling.

**Mechanism (Option A, learned sparsity).** Per-(layer, head) gate
$g_{\ell,h} = \sigma(\tilde g_{\ell,h})$, raw zero-init → $g = 0.5$
neutral. Multiplies ρ only (ψ un-gated because ρ=0 already kills
non-normality via sinh(0)=0):
$$\rho_{t,\ell,h,b} = \sigma(\tilde g_{\ell,h}) \cdot \kappa \cdot
\text{softplus}(\tilde\lambda) \cdot \tanh(\tilde\rho_{t,\ell,h,b}).$$
Zero-regression via $\tilde\rho = 0$ at init (tanh(0) = 0), not via
gate = 0 — cleaner identifiability.

**Mechanism (Option B, hard-restricted edge-only).** Structural dispatch
in [src/models/rwkv6_encoder.py](src/models/rwkv6_encoder.py):
only L0 and L_{n-1} instantiate the non-normal path; middle layers run
plain RSE+viscosity. Cuts ~4/6 of the heavy 2×2 scan cost.

**Further changes vs T2.** ψ is static (no token-dependent LoRA) to
reduce identifiability confound. κ tightened 0.6 → 0.4 per STAGE9_PLAN
§2.3 to keep spectra further from the defective boundary seen in
T2's D8.

**Implementation.**
[src/models/rwkv6_time_mix.py](src/models/rwkv6_time_mix.py):
`sparse_nn_gate_raw` parameter + forward-pass ρ-gating.
[src/models/rwkv6_encoder.py](src/models/rwkv6_encoder.py):
per-layer structural dispatch for Option B.
Backbones: `rwkv6_sparse_nonnormal_rse_viscosity` (A),
`rwkv6_sparse_nonnormal_rse_edge_only_viscosity` (B).

**Benchmark** (B=10, T=1200 mels):

| Backbone | ms/iter | Memory |
|---|---:|---:|
| `rwkv6_rse_strong_viscosity` (anchor) | 94 | 7.7 GB |
| `rwkv6_nonnormal_rse_viscosity` (T2 dense) | 271 | 10.7 GB |
| `rwkv6_sparse_nonnormal_rse_viscosity` (9A) | 271 | 10.7 GB |
| `rwkv6_sparse_nonnormal_rse_edge_only_viscosity` (9B) | **152** | 8.9 GB |

**Final status** (after stop + diagnostic, see
STAGE9_RESULTS.md):

| Run | Epochs done | Dev CER | Test CER | Decision-tree leaf |
|---|---|---:|---:|---|
| **S9B** (edge-only) | 30/30 | **0.1218** | **0.1216** | REGRESSION STABILITY |
| **S9A** (learned) | 15/30 (halted) | 0.1467 (ep 15) | — | (halted, partial diag) |

S9A was halted at ep 15 after matched-epoch comparison showed it
running ~0.006 dev CER worse than `rwkv6_p2rse_strong_viscosity`
and ~0.005 worse than T2 consistently through epochs 5–15. Projecting
the typical ep15→30 delta ≈ −0.022 in this family gives an expected
final ~0.1245 — regression band. See STAGE9_RESULTS §4 for the
partial diagnostic.

**S9B decision-tree leaf** (verbatim from `stage8_diagnostics.py`):

> REGRESSION STABILITY — spectral radius max = 1.0002, 11.6 % of G
> have |λ| > 0.99. Fix stability, rerun.

**Both Stage-9 variants show gates stuck near 0.5** (range 0.52–0.71
across all 24 (ℓ, h) slots in S9A; 0.54–0.70 on the two edge layers
in S9B). The sigmoid parameterisation did not push gates to 0 or 1 —
SGD used them as a uniform amplitude knob around 0.6, not as a
selector. Sparse pattern did not emerge.

**S9B matched-epoch trajectory** vs transition-side references:

| Ep | S9B | T2 | p2rse_strong_visc | RSE anchor |
|---:|---:|---:|---:|---:|
| 10 | 0.1698 | 0.1710 | 0.1671 | 0.1692 |
| 20 | 0.1305 | 0.1282 | 0.1287 | 0.1277 |
| 25 | 0.1238 | 0.1216 | 0.1214 | 0.1200 |
| 30 | **0.1218** | 0.1202 | **0.1190** | **0.1185** |

---

## Per-epoch dev CER panel — vanilla vs main winners

Dev CER at selected epochs, all runs seed 42 unless noted:

| Epoch | `rwkv6` | `rwkv6_rse` | `rwkv6_rse_strong_viscosity` | `rwkv6_nonnormal_rse_viscosity` (T2) | S9B |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.52 | 0.5175 | ~0.50 | 0.5253 | 0.5135 |
| 5 | ~0.236 | 0.2365 | 0.2279 | 0.2330 | 0.2273 |
| 10 | ~0.175 | 0.1748 | 0.1683 | 0.1710 | 0.1698 |
| 15 | ~0.146 | 0.1485 | 0.1441 | 0.1418 | 0.1427 |
| 20 | ~0.131 | 0.1360 | 0.1277 | 0.1282 | 0.1305 |
| 25 | ~0.125 | 0.1271 | 0.1200 | 0.1216 | 0.1238 |
| 30 | 0.1258 | 0.1251 | 0.1185 | 0.1202 | 0.1218 |

---

## Final CER summary (all completed runs on this spine)

| Stage | Backbone | Dev CER | Test CER | Test WER |
|---|---|---:|---:|---:|
| 2 | `rwkv6` (vanilla) | 0.1258 | 0.1263 | 0.3764 |
| 2 | `rwkv6_trap` | 0.1263 | 0.1254 | 0.3746 |
| 2 | `rwkv6_trap_var` | 0.1261 | 0.1259 | 0.3749 |
| 2 | `rwkv6_gen2` | 0.1264 | 0.1254 | 0.3733 |
| 2 | `rwkv6_ab3` | 0.1299 | 0.1285 | 0.3789 |
| 2 | `rwkv6_convshift_trap` | 0.1150 | 0.1150 | 0.3440 |
| 3 | `rwkv6_rse` | 0.1251 | 0.1238 | 0.3705 |
| 3 | `rwkv6_rse_convshift` | 0.1145 | 0.1126 | 0.3382 |
| 4 | `rwkv6_rse_depth` | 0.1207 | 0.1200 | 0.3593 |
| 4 | `rwkv6_rse_strong` | 0.1192 | 0.1188 | 0.3579 |
| 5 Ph1 | `rwkv6_p2rse` (linear β) | 0.1250 | 0.1241 | 0.3740 |
| 5 Ph1 | `rwkv6_p2rse_softmax` | 0.1220 | 0.1215 | 0.3642 |
| 5 Ph3 | `rwkv6_rse_depth_viscosity` | 0.1198 | 0.1198 | 0.3572 |
| 5 Ph3 | **`rwkv6_rse_strong_viscosity`** | **0.1185** | **0.1177** | **0.3515** |
| 6 Ph2b | `rwkv6_p2rse_strong_viscosity` (shared-λ) | 0.1190 | 0.1196 | — |
| 6 Ph2b | `rwkv6_p2rse_indeplam_strong_viscosity` (indep-λ) | 0.1394 | 0.1383 | — |
| 6 | `rwkv6_rmsnorm` | 0.1264 | 0.1252 | — |
| 6 | `rwkv6_hadamard_n2` | 0.1253 | 0.1251 | — |
| 6 | `rwkv6_qtail` | 0.1260 | 0.1240 | — |
| 6 | `rwkv6_qtail_gamma` | 0.1257 | 0.1249 | — |
| 6 | `rwkv6_qtail_gamma_dbeta` | 0.1247 | 0.1245 | — |
| 6 | `rwkv6_qtail_lowrank` | 0.1247 | 0.1242 | — |
| 6 | `rwkv6_qtail_lowrank_all` | 0.1238 | 0.1240 | — |
| 6 | `rwkv6_delta_warmstart` (unwired, retracted) | 0.1260 | 0.1256 | — |
| 7A | `rwkv6_rse_dphi_viscosity` (A1′) | 0.1217 | 0.1207 | — |
| 8 T1 | `rwkv6_delta_warmstart_fixed` | 0.1258 | 0.1256 | — |
| 8 T2 | `rwkv6_nonnormal_rse_viscosity` | 0.1202 | 0.1200 | — |
| 9 A | `rwkv6_sparse_nonnormal_rse_viscosity` (halted ep 15) | 0.1467 (ep15) | — | — |
| 9 B | `rwkv6_sparse_nonnormal_rse_edge_only_viscosity` | 0.1218 | 0.1216 | — |

---

## Cross-stage conclusions

### Winners and non-winners

Two winner labels are useful because the experiment chain mixed
transition-side and input-side changes:

- **Best absolute CER on this causal RWKV spine:** `rwkv6_rse_convshift`
  at `0.1145` dev / `0.1126` test. This is the strongest full
  architecture row, but it is not a pure transition comparison because
  it combines RSE with the input-side ConvShift mechanism.
- **Best clean transition-side result:** `rwkv6_rse_strong_viscosity` at
  `0.1185` dev / `0.1177` test. This is the strongest causal RWKV
  result on the main thesis axis after stripping out the ConvShift
  confound.
- **Best stable extension without a real gain:** `rwkv6_p2rse_strong_viscosity`
  at `0.1190` dev / `0.1196` test. This tied the anchor within σ and is
  useful as a control, but it did not break the ceiling.
- **Best feature-side single mechanism:** `rwkv6_qtail_lowrank_all` at
  `0.1238` dev / `0.1240` test. This is the clearest feature-lift
  result, but it remains materially above the Stage-5 transition line.

### What actually made RSE work

The empirical chain supports a specific explanation:

- RSE changed the **dominant transition operator family** on the main
  recurrent path from purely real diagonal decay to structured
  decay-plus-rotation in 2×2 blocks.
- That change was **identifiable**: it could not be absorbed into the
  existing projections the way Stage-2 discretization variants or
  Stage-7 readout-gauge parameters could.
- The mechanism also matched the task prior better than the alternatives:
  damped oscillatory / complex-pole structure is a natural fit for
  speech dynamics.
- Stage 4 and Stage 5 then showed that the gain was not "rotation by
  itself" but **rotation with the right budget allocation and damping
  regularisation**. In other words, the winning ingredient was
  transition geometry plus disciplined parameterisation, not simply more
  degrees of freedom.

The negative results make the reason clearer:

- Stage 2 solver changes were inside the old transition group and were
  absorbed.
- Stage 7A changed readout gauge, not the operator family, and failed.
- Stage 8/9 non-normal extensions changed the operator family but did so
  with dense token-wise local freedom that SGD used without turning into
  better CER.

### Promising existing mechanisms and transferability

- **`rwkv6_rse_strong_viscosity`**
  - Why it matters: strongest clean transition result; minimal extra
    parameters; stable; exact reduction to the simpler RSE line at init.
  - Transfer to Mamba / SSMs: high. The transferable principle is not
    "RWKV-specific complex numbers," but replacing scalar/diagonal decay
    with small block-complex transition structure plus frequency-coupled
    damping.
  - Transfer to recurrent linear attention: moderate to high whenever
    the implementation already has an explicit decayed recurrent state.
  - Drawbacks: requires paired channels / 2×2 block bookkeeping and a
    more expensive scan than vanilla RWKV; budget and damping need care.

- **`rwkv6_p2rse_strong_viscosity`**
  - Why it matters: stable, cheap, tied with the anchor, useful as a
    "safe extension" control.
  - Transfer to Mamba / linear attention: moderate, but only where the
    implementation can naturally support multiple modes per state block.
  - Drawbacks: no measured gain at this scale; independent-λ extension
    failed badly, so pole multiplicity alone is not the missing lever.

- **`rwkv6_qtail_lowrank_all`**
  - Why it matters: best feature-side mechanism and the most obviously
    architecture-portable idea in the chain.
  - Transfer to Mamba / linear attention: high. A low-rank quadratic
    feature lift can be added to any model that maintains a linear
    recurrent summary or state update.
  - Drawbacks: gain is only marginal here; full K² branch is too costly;
    matched controls matter because diagonal-only and full-Kronecker
    versions are easy to conflate.

Taken together, the Stage-6 / Stage-9 reading is:
**Kronecker / qtail is worth keeping as the strongest transferable
feature-side mechanism in the thesis, but it is not the main
ceiling-break route in causal RWKV.**

- **ConvShift / local input filtering**
  - Why it matters: empirically strong and orthogonal to the transition
    line.
  - Transfer to Mamba / linear attention: high; both families already
    tolerate local convolution / local filtering well.
  - Drawbacks: confounds the pure transition story, so it should be
    tracked as a separate axis.

### Drawbacks and what to pay attention to

- Preserve **exact reduction at init**. The clearest negative and
  positive results all depend on knowing whether the new mechanism is a
  real extension or just a different starting point.
- Watch **identifiability**, not just mobility. Several mechanisms
  moved substantially but still failed because the extra freedom was
  absorbed or misused.
- Monitor the **spectral/Jordan boundary** whenever non-normality is
  introduced. Stage 8/9 repeatedly used the extra freedom near
  `|λ| ≈ 1` without producing better CER.
- Keep **matched controls** honest. Layer placement, normaliser changes,
  and branch depth can otherwise make feature-side conclusions look
  stronger than they are.
- Treat single-seed patterns like γ-β co-adaptation as **mechanism-level
  observations**, not laws.

### What to skip

The chain is now strong enough to close several directions at this
budget:

- solver-only discretization variants as a main line
- Stage-7 readout-gauge completion and the cross-layer gauge-residual
  extension at this scale
- independent-λ / extkv P²-RSE variants
- recurrent delta as a mainline improvement route for clean-100 causal
  RWKV-6 at 7 M
- dense token-wise non-normal RSE variants beyond the existing T2 run
- sparse token-wise non-normal RSE variants (`S9A`, `S9B`) beyond
  closure-level stability checks
- stacking multiple weak positives together before causal attribution is
  clear

### Where the next serious work should go

The results do **not** say "transition geometry is over." They say the
wrong transition-side refinement was tried after the RSE win.

The most defensible next direction is:

- **static or slowly varying role-specialised transition structure**
  rather than token-wise local adaptivity

Concretely, that means:

- keep `rse_strong_viscosity` as the causal RWKV transition baseline
- if extending the transition family, prefer head-/layer-specialised
  structure that is fixed or slowly varying over token-wise non-normal
  modulation
- treat `qtail_lowrank_all` as the orthogonal feature-side transfer
  probe for Mamba / linear attention, not as the core explanation for
  why RWKV improved

The sharpest single-sentence conclusion of the whole chain is:

> **RSE worked because it changed the recurrent operator family on the
> dominant path in a way that was both structurally irreducible and easy
> for SGD to use; most later ideas failed because they added flexible
> token-wise freedom without matching the real bottleneck.**

*End of stages 2–9 summary.*
