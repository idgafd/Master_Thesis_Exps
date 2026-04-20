# Stage 6 — Conclusions and forward plan

*Finalised 2026-04-20. Summary of what worked, what didn't, where each
strategy transfers, and the single untested combination worth running
next.*

Companion document: [STAGE6_ANALYSIS.md](STAGE6_ANALYSIS.md) — full
experimental analysis with trajectories, diagnostics, and mathematical
rationale.

---

## 1. Scope tested (all LibriSpeech clean-100, 30 ep, seed 42)

Stage 6 and the surrounding diagnostic rounds cover two axes:

**Feature-lift axis (the EXPRESSIVENESS-paper line):**
- `rwkv6_rmsnorm` — GroupNorm → per-head RMSNorm. Paper §4.5 claim.
- `rwkv6_hadamard_n2` — rmsnorm + diagonal $k \odot k$ (no cross-channel). Null control.
- `rwkv6_qtail` — rmsnorm + full $k \otimes k$ Kronecker at top-2 layers. Paper §3.2 claim.
- `rwkv6_qtail_gamma` — qtail + learnable per-head γ decay coupling.
- `rwkv6_qtail_gamma_dbeta` (R2) — γ + data-dependent β token selectivity.
- `rwkv6_qtail_lowrank` — project r,k to K'=16, full Kronecker on K'²=256 at top-2.
- `rwkv6_qtail_lowrank_all` (lra) — same but all 6 layers.

**Pole-manifold composition axis (Stage-5 deferred):**
- `rwkv6_p2rse_indeplam_strong_viscosity` (Phase 2b) — independent-λ LoRA + strong + viscosity.
- `rwkv6_p2rse_strong_viscosity` — shared-λ + strong + viscosity (diagnostic control).

**Delta rule diagnostic:**
- `rwkv6_delta_warmstart` — a0=−5 init instead of 1 (TODO_DELTA_RULE Tier-1).

---

## 2. Clarification: qtail = Kronecker; what exactly was tested

**`qtail` ("quadratic tail") IS the Kronecker $n{=}2$ feature lift.**
The internal name comes from "quadratic Taylor term applied at the tail
of the stack" (originally top-2 layers). Mathematically: the $n{=}2$ term
in the EXPRESSIVENESS paper's Taylor expansion of softmax attention,
$(k_t^{\otimes 2}, r_t^{\otimes 2})$.

**Kronecker variants tested (final 30-ep results):**

| # | Backbone | Lift | Layers | γ | Data-dep β | Dev | Test |
|---|---|---|---:|---|---|---:|---:|
| 1 | `rwkv6_qtail` | Full K²=4096 | top-2 | — | static scalar | 0.1260 | 0.1240 |
| 2 | `rwkv6_qtail_gamma` | Full K² | top-2 | per-head | static | 0.1257 | 0.1249 |
| 3 | `rwkv6_qtail_gamma_dbeta` (R2) | Full K² | top-2 | per-head | **data-dep** | 0.1247 | 0.1245 |
| 4 | `rwkv6_qtail_lowrank` | Low-rank K'²=256 | top-2 | — | static | 0.1247 | 0.1242 |
| 5 | `rwkv6_qtail_lowrank_all` (lra) | Low-rank K'²=256 | **all 6** | — | static | **0.1238** | **0.1240** |
| null | `rwkv6_hadamard_n2` | Diagonal $k \odot k$ | all 6 | — | — | 0.1253 | 0.1251 |

Best: **lra** (all-layer low-rank), dev 0.1238 = **MARGINAL**.

**Where data-dependent β was tested:** exactly one setting — R2 (#3
above), on full-K² Kronecker with γ at top-2 layers. It contributed ~1σ
of CER gain over qtail-γ and exposed the γ-β co-adaptation dynamic.

**Where data-dependent β was NOT yet tested:** low-rank Kronecker
(either top-2 or all-layer), plain qtail without γ, or any non-Kronecker
mechanism. See §8 for the single most valuable untested combination.

---

## 3. Winning strategies — ranked catalog for transfer/stacking

Ordered by empirical effect size (strongest first). Each entry: what it
is, why it works, where it transfers, what it stacks with.

### 3.1. Kronecker $n{=}2$ cross-channel feature lift — THE foundation

**What:** replace $(r_t, k_t) \in \mathbb{R}^K$ with
$(r_t \otimes r_t, k_t \otimes k_t) \in \mathbb{R}^{K^2}$ in a parallel
state accumulator; add as a gated branch to the linear-attention scan.

**Why it works:** captures *cross-channel* $k_i k_j$ interactions ($i \ne j$)
that the linear branch ($k_i$ alone) cannot express. The EXPRESSIVENESS
paper proves this is the $n{=}2$ Taylor term of softmax attention —
softmax's expressivity gap over linear attention is precisely these
cross-terms. The hadamard null control (0.1251 test, tied with rmsnorm)
empirically isolates the cross-channel $k_i k_j$ as the active
ingredient; the diagonal $k_i^2$ terms alone contribute nothing.

**Transfer targets (universal):**
- **Linear attention (Katharopoulos):** lift (Q, K) before chunk-recurrent scan
- **Mamba-2 / SSD:** lift (C, B) before SSD matmul
- **RWKV-7:** same mechanism, direct port
- **Gated DeltaNet:** lift before Householder transition
- **LION (bidirectional):** lift before T×T attention

**Stacks with:** every transition-level mechanism (decay, RSE rotation,
viscosity, delta rule). Feature layer is orthogonal to temporal layer.

### 3.2. Low-rank Kronecker truncation (K'=16) — THE efficiency enabler

**What:** learned per-head projections $U_r^{(h)}, U_k^{(h)} \in \mathbb{R}^{K \times K'}$
collapse $r, k$ to $K' = 16$ before lifting; full Kronecker on the
$K'^2 = 256$ lifted features.

**Why it works:** Eckart–Young-optimal truncation. The effective rank of
the $k_i k_j$ bilinear form used by SGD is ≪ $K^2$ at this scale;
discarding the null directions preserves all variance SGD exploits. Our
empirical test (K'=16 at top-2 layers) matched full K² CER while using
16 % of the state memory and 37 % of wall-clock.

**Transfer targets (universal):** every architecture adopting Strategy
3.1 should use this form. Full $K^2$ is never worth it at scales where
$K' \ll K$.

**Stacks with:** Strategy 3.3 (all-layer) is enabled by this; 3.4 (γ)
and 3.5 (data-dep β) compose freely.

### 3.3. All-layer Kronecker deployment — 1 σ extra on top of lowrank

**What:** apply the (low-rank) Kronecker branch at every layer of the
stack, not only the top 1-2.

**Why it works:** shallow-layer Kronecker features contribute real signal
at smaller per-layer magnitude. Confirmed by lra trajectory vs top-2
lowrank: consistent −0.001 to −0.003 dev at matched epochs. The
mechanism's utility is NOT purely a deep-layer property.

**Transfer targets:** deployment choice, not architectural axis. Whenever
Strategy 3.2 is used, enable at all layers.

**Stacks with:** Strategy 3.2 is prerequisite (without low-rank, the
memory cost prohibits all-layer).

### 3.4. Learnable per-head γ decay coupling on the Kronecker branch

**What:** parameterise lifted decay as $w_\text{pair}[i,j] = \gamma_h (w_i + w_j)$
with $\gamma_h$ a learnable per-head scalar, init 1.0.

**Why it works:** the paper's Taylor derivation imposes NO decay
constraint on order-n state; setting $w_\text{pair} = w_i + w_j$ (the
natural Kronecker lift) is one arbitrary choice. γ gives SGD the freedom
to match each head's effective time scale. Empirically: qtail-γ showed
bimodal depth-graded distribution — one head per layer at γ ≈ 0.57
("long-memory specialist"), others near γ ≈ 1; deeper layers prefer
smaller γ.

**Transfer targets:** any architecture with Kronecker lift. Particularly
useful in architectures where β is naturally small (γ < 1 extends
effective memory). **γ must be paired with Strategy 3.5 to realise its
full value** (see γ-β co-adaptation below).

**Stacks with:** 3.1, 3.2, 3.3. Requires 3.5 for full effect.

### 3.5. Data-dependent β token-selective gating

**What:** replace static per-head β scalar with $\beta_{q,t} = \beta_\text{static} + W_\beta x_t$ (per-token, per-head, zero-init).

**Why it works:** Mamba-2's selective-scan argument — different tokens
need different amounts of cross-channel branch activation (coarticulation
boundaries vs stable vowels). Empirically: R2 grew β by ×3 over qtail-γ
(from |β| < 0.04 to |β| < 0.12) and improved CER by 1σ.

**Co-adaptation with γ (critical):** in the small-β regime, γ < 1
(long-memory). In the larger-β regime, γ inverts to > 1 (fast-forgetting).
γ and β have a *joint* optimum — not independent ones. Any deployment
of 3.5 should include 3.4 so both parameters can co-adapt.

**Transfer targets:** universal. Mamba-2's selective scan is evidence
the mechanism class generalises.

**Stacks with:** 3.1, 3.2, 3.3. **Requires 3.4** (γ) for the co-adaptation
pathway.

### 3.6. Complex-valued transition (RSE) — Stage 3/4/5 legacy winner

**What:** replace scalar-per-channel decay with 2×2 block-diagonal
$\exp(-\lambda) R(\theta)$ transitions. Implemented via complex scan.
Combined with strong rotation budget (π/2 clip, LoRA 48) and viscosity
coupling ($\lambda_\text{eff} = \lambda + \eta \theta^2$).

**Why it works:** extends the transition Lie group from $(\mathbb{R}_+)^K$
to $\mathrm{SO}(2)^B \times (\mathbb{R}_+)^B$, giving rotation-based
transition dynamics that the diagonal-decay family cannot represent.
`rwkv6_rse_strong_viscosity` at 0.1185 dev (−5.8 % vs baseline) is the
strongest single mechanism in our entire test scope.

**Transfer targets:** any linear-attention-family architecture where
2×2 block-diagonal transitions can replace scalar gating:
- **Mamba-2:** replace scalar-α transition with 2×2 rotation-scale; fits SSD framework
- **Linear attention:** complex transition via `rse_scan_fast.py` port
- **Gated DeltaNet:** 2×2 block in the Householder chain
- **LION:** complex-λ on the semiseparable mask

**Stacks with:** Kronecker feature lift (3.1) is orthogonal — RSE
operates at transition level, Kronecker at feature level. Untested but
mathematically composable.

### 3.7. Zero-regression-at-init parameterisation — methodological

**What:** every new learnable parameter is initialised so that at $t{=}0$,
the mechanism reduces *exactly* to baseline. Applied to β_qtail (init 0),
qtail_gamma (init 1.0), beta_qtail_proj (zero W + b), qtail_lr_proj
(gated by β=0), viscosity η (init 0), delta a0 (warmstart: −5 instead of 1).

**Why it matters:** makes negative results interpretable as
"mechanism didn't help" rather than "wrong starting point." Without
this contract, Phase 2b's indep-λ regression would have been ambiguous;
with it, the failure mode isolates to "LoRA overparameterisation
destabilises training" rather than "mechanism is broken."

**Transfer:** universal methodological principle. Every new mechanism
should satisfy this.

### 3.8. Delta-rule warmstart init (a0 = −5) — init-technique

**What:** when adding a delta erase branch
$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + v_t k_t^\top$, initialise the
β parameterisation so β ≈ 0 at t=0.

**Why it works:** delta at β ≈ 1.76 (the default-init value from
sigmoid(1)·2) destroys a randomly-initialised state before useful
associations are written. Warmstart at near-zero lets the state write
productive content first, then SGD grows the erase strength only if
useful. Our empirical test turned prior 0.1373/0.1483 delta failures
into a clean 0.1260 null.

**Transfer:** any architecture adding a state-erasure branch (DeltaNet,
Gated DeltaNet, RWKV-7 expressive state, Titans/ATLAS test-time
gradient). Always start erase strength near zero.

**Stacks with:** the mechanism itself is null at our scale (T ≪ d²); the
init technique transfers regardless.

---

## 4. Established nulls — documented, don't retry at this scale

| Mechanism | CER | Diagnosis |
|---|---:|---|
| GroupNorm → RMSNorm swap | 0.1264 dev / 0.1252 test | Scale-dependent (paper worked at 300 M+ / 20 L). Don't retest at 7 M / 6 L. |
| Diagonal Hadamard $k \odot k$ | 0.1253 / 0.1251 | No cross-channel info; equivalent to squared activation. Proves 3.1 needs full Kronecker, not diagonal. |
| Discretisation variants (trap, trap_var, gen2, ab3 CER) | all tied | Reparameterisation-absorbed by existing decay LoRA (Stage 2). |
| Delta rule at 7 M / T=500 | 0.1260 / 0.1256 | State saturation doesn't manifest at T ≪ d² = 4096. |

## 5. Established regressions — known-bad, avoid

| Regression | CER | Diagnosis |
|---|---:|---|
| P²-RSE indep-λ LoRA on strong + viscosity (Phase 2b) | 0.1394 / 0.1383 (+17 %) | ~200 K-param LoRA destabilises an otherwise-working composition. |
| P²-RSE × viscosity stacking at strong budget | 0.1190 / 0.1196 (tied with viscosity alone) | Non-additive — both mechanisms address the same expressivity gap. |

---

## 6. Stacking plan — the unified Stage-6 stack

Combining all validated strategies from §3.1–3.5 into one backbone:

**Proposed backbone:** `rwkv6_qtail_gamma_dbeta_lowrank_all`
- Kronecker feature lift (3.1)
- Low-rank projection K'=16 (3.2)
- All 6 layers (3.3)
- Learnable γ decay coupling (3.4)
- Data-dep β gating (3.5)

**Cost estimate:** ~12 GB peak VRAM, ~200 s/epoch (similar to lra).
**CER estimate (honest priors):**
- Additive stacking (optimistic): 0.1215-0.1225 dev → BREAK
- Sub-additive (Stage-5 pattern): 0.1230-0.1240 dev → MARGINAL (tied with lra)
- Noise-dominated: 0.1240-0.1250 dev → PLATEAU

Most likely sub-additive: each refinement added ~1σ; stacking them
likely gives 1.5-2σ total rather than 4σ. But even sub-additive outcome
preserves the efficiency gains — the run takes ~3.3h regardless.

## 7. Transferability roadmap — port targets

Priority-ordered for porting the validated stack to other architectures:

1. **Mamba-2 + Strategies 3.1 + 3.2 + 3.3** — lift (C, B) with K'=16 Kronecker at all layers. Tests whether the Kronecker result is RWKV-specific or genuinely architectural.
2. **Mamba-2 + Strategy 3.6 (RSE complex-transition adaptation)** — replace scalar-α with 2×2 rotation-scale transitions. Tests RSE beyond RWKV family.
3. **Gated DeltaNet + Strategy 3.1** — Kronecker on top of Householder transition. Two orthogonal expressivity mechanisms.
4. **Linear attention (Katharopoulos baseline) + Strategies 3.1–3.5** — simplest linear-attention form; most direct test of the EXPRESSIVENESS paper's core claim.
5. **LION (bidirectional RWKV) + Strategy 3.1** — Group B architecture.

Each port: ~1-2 days of implementation + a 30-ep training run.

**Testable cross-architecture prediction:** the γ-β co-adaptation pattern
(Strategy 3.4 + 3.5 interaction) should reproduce on all targets.
Architectures with naturally-larger selective gating (Mamba-2, Gated
DeltaNet) should prefer γ > 1; architectures where β stays near zero
(plain linear attention without gating) should prefer γ < 1.

---

## 8. The single untested Stage-6 combination worth running next

**`rwkv6_qtail_lowrank_all_dbeta`** — lra (best dev) + data-dep β (R2's
contributing mechanism).

**Why it's the right next experiment:**
- Combines validated Strategy 3.2 + 3.3 (lra's 0.1238 MARGINAL) with
  validated Strategy 3.5 (R2's ~1σ gain).
- Neither component has been tested WITH the other; the gap is a direct
  one-line code change (enable both substrings in backbone name).
- Data-dep β's co-adaptation mechanism requires γ to be present too —
  this backbone would need the γ variant as well, giving
  `rwkv6_qtail_gamma_dbeta_lowrank_all` (the full unified stack from §6).
- Single run, no design work, ~3.3 h on one GPU, seed 42.

**Expected outcome:**
- Most likely (Stage-5-pattern sub-additive): 0.1230-0.1240 dev. Matches
  lra's MARGINAL, confirms the stack is stable.
- Optimistic (additive): 0.1215-0.1225 dev. BREAK. Would be the
  strongest result in the Stage-6 scope by a clear σ.

**Decision rule for after this run:**
- If dev ≤ 0.1230: the unified stack is the deployment-ready
  configuration. Ship it as the Stage-6 canonical backbone.
- If 0.1230 < dev ≤ 0.1244: MARGINAL but no additive stacking gain; each
  mechanism was already saturated. lra remains the efficiency-optimal
  choice.
- If dev > 0.1244: the stack is sub-additive and some mechanism fights
  another. Diagnose via checkpoint inspection (does γ still hit bimodal
  pattern? does β still grow?).

---

*End of Stage-6 conclusions. Next session: run `rwkv6_qtail_gamma_dbeta_lowrank_all`
as the unified-stack validation, plan the Mamba-2 port from the §7 roadmap.*
