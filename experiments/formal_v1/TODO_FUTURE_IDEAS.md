# TODO — Future Ideas for LION / Linear Attention / Beyond Stage 9

Parking lot for mechanisms from the literature that are **not** on the current
causal RWKV-6 critical path but would be natural to test when the thesis
returns to Group B (LION / bidirectional) or to a wider attention-variant
comparison.

**Sibling docs:**
- [stages_2_9_summary.md](stages_2_9_summary.md) — chronological summary
  of Stages 2–9 with per-epoch trajectories, result tables, and links to
  the individual stage reports. Read it first for the evidence base; the
  "resolved status" table below is a compressed index into it.
- [TODO_DELTA_RULE.md](TODO_DELTA_RULE.md) — dedicated catalog for delta-rule
  variants across Mamba-2 / RWKV / Linear Attention, LUCID composition, and a
  diagnostic queue for why our prior delta runs underperformed.

**Inspiration papers:**
| PDF file | HF status | ArXiv |
|---|---|---|
| papers/Log_Linear_Attention_2506.04761v3.pdf | hf papers read 2506.04761 | https://arxiv.org/abs/2506.04761 |
| papers/NaLaFormer_2506.21137v1.pdf | hf papers read 2506.21137 | https://arxiv.org/abs/2506.21137v1 |
| papers/Expressiveness_2507.23632v1.pdf | hf papers read 2507.23632 | https://arxiv.org/abs/2507.23632 |
| papers/Avey_2506.11305v2.pdf | hf papers read 2506.11305 | https://arxiv.org/abs/2506.11305 |
| papers/RWKV_X_2504.21463v2.pdf | hf papers read 2504.21463 | https://arxiv.org/abs/2504.21463 |
| papers/HSA_2504.16795v2.pdf | hf papers read 2504.16795 | https://arxiv.org/abs/2504.16795 |
| papers/CRT_2505.00929v1.pdf | no hf | https://arxiv.org/abs/2505.00929 |
| papers/Non_Attention_2506.01963v1.pdf | hf papers read 2506.01963 | https://arxiv.org/abs/2506.01963 |
| papers/PoM_2604.06129v1.pdf | no hf | https://arxiv.org/abs/2604.06129 |
| papers/M_RNN_2603.14360v1.pdf | no hf | https://arxiv.org/abs/2604.06129 |

---

## Tested hypotheses — resolved status (as of 2026-04-21)

This section compresses the outcomes of Stages 2–9 so the parking lot below
can be read against what has already been decided. See individual stage
result files for the evidence, or [stages_2_9_summary.md](stages_2_9_summary.md)
for the chronological narrative with per-epoch trajectories.

### All-stage CER landscape (causal RWKV-6, seed 42, 30 ep, LibriSpeech clean-100)

| Stage | Backbone | Dev CER | Test CER | Verdict |
|---|---|---:|---:|---|
| 2 | `rwkv6` (baseline) | 0.1258 | 0.1263 | reference |
| 2 | `rwkv6_trap` / `trap_var` / `gen2` | ~0.1261 | ~0.1254 | null — solver ≠ model class |
| 2 | `rwkv6_ab3` | 0.1299 | 0.1285 | regression |
| 3 | `rwkv6_rse` | 0.1251 | 0.1238 | small real signal |
| 3 | `rwkv6_rse_m2` / `m4` (multi-rate) | ~0.1245 | ~0.1240 | PLATEAU — flat multi-rate is not the axis |
| 4 | `rwkv6_rse_depth` | 0.1207 | 0.1200 | first ceiling break (depth-graded θ budget) |
| 4 | `rwkv6_rse_strong` | 0.1192 | 0.1188 | uniform strong budget — second ceiling break |
| 5 | `rwkv6_rse_depth_viscosity` | 0.1198 | 0.1198 | ties depth without viscosity |
| **5** | **`rwkv6_rse_strong_viscosity`** | **0.1185** | **0.1177** | **BEST. Current anchor.** |
| 5 Ph1 | `rwkv6_p2rse` / `p2rse_softmax` | 0.1250 / 0.1220 | 0.1241 / 0.1215 | paired-pole doesn't help at small budget |
| 6 | `rwkv6_p2rse_strong_viscosity` (shared-λ) | 0.1190 | 0.1196 | tied with anchor (stable) |
| 6 | `rwkv6_p2rse_indeplam_strong_viscosity` (indep-λ) | 0.1394 | 0.1383 | **REGRESSION** — parameterisation / identifiability mismatch |
| 6 | `rwkv6_rmsnorm` | 0.1264 | 0.1252 | PLATEAU (paper effect scale-dependent) |
| 6 | `rwkv6_hadamard_n2` | 0.1253 | 0.1251 | PLATEAU (diagonal n=2; suggestive vs qtail, not a perfectly matched control) |
| 6 | `rwkv6_qtail` (full K², top-2 layers) | 0.1260 | 0.1240 | near-MARGINAL on test |
| 6 | `rwkv6_qtail_gamma` / `qtail_gamma_dbeta` | 0.1257 / 0.1247 | 0.1249 / 0.1245 | γ + data-dep β engage; single-seed γ-β co-adaptation pattern |
| 6 | `rwkv6_qtail_lowrank` (top-2, K'=16) | 0.1247 | 0.1242 | matches full K² on CER at 16 % VRAM |
| 6 | **`rwkv6_qtail_lowrank_all`** (all layers) | **0.1238** | **0.1240** | **best Stage-6 feature-side, MARGINAL** |
| 6 | `rwkv6_delta_warmstart` (delta on LION path) | 0.1260 | 0.1256 | **later shown to be an implementation artefact** (Stage 8) |
| 7A | `rwkv6_rse_dphi_viscosity` (data-dep φ readout) | 0.1217 | 0.1207 | AMBIGUOUS — W_φ moved but Im/Re grew, not shrank |
| 8 T1 | `rwkv6_delta_warmstart_fixed` (recurrent delta, wired) | 0.1258 | 0.1256 | AMBIGUOUS engaged null — g_δ up to 0.65 at L5, β p95 > 1, zero CER gain |
| 8 T2 | `rwkv6_nonnormal_rse_viscosity` (dense polar) | 0.1202 | 0.1200 | AMBIGUOUS engaged — ρ-LoRA F=18–30, non-normality score 0.04–0.11, CER near-anchor |
| 9 A | `rwkv6_sparse_nonnormal_rse_viscosity` (learned, **halted ep 15**) | 0.1467 (ep 15) | — | on-track to regression; stopped early |
| **9 B** | **`rwkv6_sparse_nonnormal_rse_edge_only_viscosity`** | **0.1218** | **0.1216** | **REGRESSION STABILITY** (D8: 11.6 % of L5 G with \|λ\|>0.99) |

### The cross-experiment invariant

Three independent extensions of the RSE-strong-viscosity anchor
(A1′ readout-gauge, T1 recurrent delta on vanilla, T2 dense polar
non-normal) all converge to the same signature:

> **Dense per-(token, head, layer) mechanism freedom engages SGD
> aggressively — mobility is always non-zero, the mechanism's realised
> action is substantial — but does not translate into CER gain at
> 7 M / clean-100 / 30-ep.**

| Stage | Mechanism | Engagement | CER vs anchor |
|---|---|---|---:|
| 7A | data-dep readout phase φ | \|W_φ\|_F 5.5–6.4; \|φ\| p99 ≈ 2 rad | +0.0028 |
| 8 T1 | recurrent delta rank-1 erase (on vanilla) | g_δ up to 0.65; β p95 > 1; erase 24 % of \|S\| | 0.0000 (tied vanilla) |
| 8 T2 | dense polar 2×2 non-normal | ρ-LoRA F 18–30; non-normality 0.04–0.11 | +0.0017 |

The productive structure that would break RSE+viscosity is not reachable
by adding more per-token freedom on top of it in these three forms.
Stage 9 is the first experiment that tests *structural* sparsity;
S9B completed at 0.1218 dev (ambiguous), S9A still running. Whether
the "dense-vs-structural" axis matters at this scale is the open cell.
See the matched-epoch trajectories in
[stages_2_9_summary.md](stages_2_9_summary.md#per-epoch-dev-cer-panel--vanilla-vs-main-winners).

### What this resolves below

- **Phase 2b indep-λ P²-RSE** — FALSIFIED at the strong+viscosity
  composition (+17 % rel regression). Queue entry below is kept as
  historical record; do not retry at this scale or composition. The
  failure mode is a **parameterisation / identifiability mismatch**:
  adding per-pole λ LoRAs without correspondingly independent viscosity
  / observation structure creates degrees of freedom the loss cannot
  disambiguate. Not a simple "too many parameters" story.
  See `STAGE6_ANALYSIS.md` §3.5 and
  [stages_2_9_summary.md](stages_2_9_summary.md#stage-6-—-expressiveness-paper-feature-side-adaptations).
- **Phase 2b-ext indep-(k,v)** — dropped in sympathy with Phase 2b; no
  evidence base to justify the stack.
- **Full Kronecker n=2 all-layer** — TESTED as `qtail_lowrank_all` at
  0.1238 dev (MARGINAL). The low-rank form matches full-rank on CER at
  16 % VRAM. Pitched as an Eckart–Young story in early writeups;
  stages_2_9_summary treats this interpretation as *suggestive but not
  proven* — what we actually tested is low-rank projection of (r, k)
  *before* the Kronecker lift, with a simpler decay law — so the result
  is really "a small learned quadratic subspace captures the signal,"
  not "the K² bilinear form has rank 16."
- **Diagonal (hadamard_n2) vs full Kronecker (qtail) isolation** —
  suggestive that cross-channel $k_i k_j$ is the active ingredient,
  but the two controls are not perfectly matched (hadamard_n2 runs on
  all 6 layers; qtail originally runs on top 2). Treat "isolated to
  cross-channel" as the preferred reading, not a proven causal claim.
- **γ-β co-adaptation** (Stage 6.5 / R2) — single-seed pattern: small-β
  regime → γ < 1, larger-β regime → γ > 1. Treat as a hypothesis about
  Kronecker-lift + selective-gating architectures, not a transferable
  theorem.
- **qtail × RSE composition** — explicitly ruled out in Stage 7 and 8 for
  this thesis line: dense freedom stacked on RSE+viscosity does not
  translate into CER at 7 M / clean-100.
- **Readout-gauge (A1′ / A2 / 7A-plus)** — FALSIFIED as within-block
  gauge completion at this scale / composition. The `|Im|/|Re|` at
  readout is not "discarded signal" but a sum-over-blocks cancellation
  structure the anchor already exploits as its implicit gauge.
  Methodological caveat: the pre-training diagnostic was run on
  `p2rse_strong_viscosity` as a proxy (the anchor checkpoint had not
  been saved); the post-training true-anchor read via A1′ adjusted the
  L5 Im/Re figure from 0.70 (proxy) to 0.56 (true anchor), so the
  *magnitude* of available quadrature was overstated but the
  *direction* of the mechanism's failure was confirmed.
- **Recurrent delta-rule (correctly wired)** — engaged-null at this
  scale. Delta stays in the catalog for architectures where state
  saturates (T ~ d²), but at our T ≈ 300 < d² = 4096 it is not the
  lever. Retracts the Stage-6 delta null cited as evidence — that run
  did not branch on `use_delta_rule` in the recurrent path at its
  training commit.
- **Dense non-normal transition (T2)** — engaged-ambiguous-null. The
  mechanism is real (ρ LoRA F 18–30, non-normality score 0.04–0.11 per
  layer; higher at L0 and L5) and structured (bimodal per-head spread
  at L0), but uniform per-token / all-layer deployment does not pay off
  at 7 M / clean-100 / 30-ep. D8 spectral-radius tail (6 % of L0 and L2
  tokens with $|\lambda|>0.99$) flags borderline stability — the
  near-1 tail is worth keeping in mind if anyone revives this line.

### Stage 9 closed (both runs complete / halted — see STAGE9_RESULTS.md)

**S9B** (`rwkv6_sparse_nonnormal_rse_edge_only_viscosity`): dev 0.1218 /
test 0.1216, trained to 30 ep. Decision-tree leaf: **REGRESSION
STABILITY** — max spectral radius 1.0002, 11.6 % of L5 G-matrices at
\|λ\| > 0.99. Hard edge-only structural dispatch (L0+L5 only) pushed the
spectrum against the Jordan boundary without beating the dense T2.

**S9A** (`rwkv6_sparse_nonnormal_rse_viscosity`): halted at ep 15/30
(dev 0.1467) after matched-epoch comparison showed it running ~0.006
worse than p2rse_strong_viscosity and ~0.005 worse than T2 through
eps 5–15. Partial diagnostic (D10 gates) shows **gates stuck in
[0.52, 0.71]** across all 24 (ℓ, h) slots — sigmoid parameterisation
did not induce structural sparsity; SGD used the gate as a uniform
amplitude knob around 0.6. Also reports 8–12 % of L1, L2, L4 G-matrices
with \|λ\| > 0.99 — same stability-edge pattern as S9B.

**The Stage-9 result.** Both the learned and hard-structural sparsity
variants under-performed the anchor, T2 dense, and the
`p2rse_strong_viscosity` paired-pole control. Matched-epoch 15:
S9A 0.1467 vs p2rse 0.1410 (+0.0057) vs T2 0.1418 (+0.0049) vs anchor
0.1441 (+0.0026) vs S9B 0.1427 (+0.0040). The sigmoid gate (no
push-to-0/1 pressure) plus tightened κ=0.4 (which in combination with
the g ≈ 0.5 init gave effective κ ≈ 0.2 at early training) produced a
worse equilibrium than T2's un-gated dense path.

**Combined with A1′ / T1 / T2 / S9 results**, the invariant tightens:

> At 7 M / clean-100 / 30-ep on causal RWKV-6, five independent
> strict-superset extensions of `rwkv6_rse_strong_viscosity` — A1′
> readout-gauge, T1 recurrent delta, T2 dense non-normal, S9A
> gated-sparse all-layer, S9B hard edge-only — all converge to the
> same signature: mechanism engages (D5 mobility non-zero, D6/D7
> realised action substantial), CER does not benefit. Paired-pole
> extension (`p2rse_strong_viscosity`, normal family) ties the anchor
> within σ. Both non-normal gating forms tested in Stage 9
> additionally press the transition spectrum against the Jordan
> boundary without converting that freedom into CER.

Full chronology in [stages_2_9_summary.md](stages_2_9_summary.md).
Frontier-direction discussion will be written separately.

---

**Working principle — refinement before composition.**

The Stage-2 → Stage-5 pipeline in this thesis followed a single discipline:
when a proposed mechanism lands as MARGINAL (not PLATEAU, not BREAK),
**iterate within the single mechanism before stacking another on top.**
This is how Stage-3 `rse` (0.1251) became Stage-4 `rse_strong` (0.1192) via
budget refinement, then Stage-5 `rse_strong_viscosity` (0.1185) via clip
refinement. At each step the math motivation was internal to the mechanism.

**Updated principle after Stage 7–8 (2026-04-21).** The "refine within one
mechanism" discipline held for the winning line (RSE → RSE-strong →
RSE-strong-viscosity). But it was also applied to three strict-superset
extensions of that anchor (A1′ data-dep readout phase; T1 recurrent delta;
T2 dense polar 2×2 non-normal) — all three produced AMBIGUOUS-ENGAGED
nulls. The empirical invariant is now explicit (see top-of-file summary):
*dense per-token mechanism freedom engages SGD but does not translate into
CER gain at this scale.* Stage 9 is the first experiment probing
*structural* sparsity as the unlock.

Composition candidates below remain the **final-ablation slot** once
single-mechanism refinement within each axis has capped out. Read the
queue against the "resolved status" section above before reviving
anything; several entries are now closed by direct evidence.

Each entry names the paper, the mechanism, what it would cost to implement in
this codebase, and the specific experiment it would unlock. Only ideas with a
concrete adaptation path are listed — inspirational notes that don't map to
our code are kept out.

---

## 0. Cross-cutting patterns (the signal across papers)

The same structural ideas keep showing up in independent papers. When a
pattern is the load-bearing trick in multiple works it's almost certainly
worth testing in our codebase; conversely, patterns that only one paper
proposes are usually specific to their setting and deserve more skepticism.

### Pattern A — Multi-scale / hierarchical temporal aggregation

| Source | Mechanism | Our analog |
|---|---|---|
| Log-Linear Attention (Guo et al. 2025, §3.1) | Fenwick-tree O(log T) buckets with data-dependent λ_t^(ℓ) mixer | — (none yet) |
| Avey (§3.2 Enricher) | Partial-embedding bypass: half of expanded hidden skips the contextualizer | — (none yet) |
| Stage-2 `gen2` diagnostic (STAGE2_RESULTS §3) | Per-head α₁ grows monotonically with depth (0.023 → 0.107 across 6 layers) | Measured, not exploited |
| Stage-4 `rse_depth` | Depth-graded rotation budget (π/8 shallow → π/2 deep) | Active, tied with rse_strong |
| Multi-Rate RSE `rse_m2`, `rse_m4` | M parallel (λ, θ) scans, softmax mixer over scales | Ran in Stage-3; plateaued |
| RWKV-6 native multi-head decay | Different heads learn different decay rates | Baked in |

**Reading:** *heterogeneous temporal-scale allocation* is the single most
recurrent idea in the literature relevant to our work. Our Stage-2 and
Stage-4 results are mild empirical positives on this axis. The move
conspicuously missing from our codebase is the **log-scale hierarchy**
(as opposed to the flat multi-rate of `rse_m2/m4` or the per-layer
depth-graded of `rse_depth`). Log-Linear Attention provides the clean
formalization.

### Pattern B — Phase / complex / cosine encoding for non-negativity

| Source | Mechanism | Our analog |
|---|---|---|
| NaLaFormer §3.2 | φ(d(x)) = [cos(d(x)); sin(d(x))], uses Ptolemy for dot-product preservation | — |
| Avey §3.2.2 | Contextualizer mixes via cosine-similarity matrix N(Z)·N(Z)^T | — |
| RSE (ours, Stage 3) | Complex scan z_t = e^{-λ + iθ}; state packed as (S[2b], S[2b+1]) → c_b ∈ ℂ | Active, Stage-3 best 0.1145 |

**Reading:** using phase to encode sign/direction rather than clipping
negatives is a robust pattern. We already do this via RSE. NaLaFormer's
mechanism is functionally kin to ours (both embed real features into 2D
rotation-equivariant space) — so its specific claim about non-negativity
doesn't give us new signal. But the *Avey Contextualizer's cosine-
similarity-weighted mixing* is a different idea: it uses cosine sim as
the attention weight itself rather than as a non-negativity trick.

### Pattern C — Norm-aware / spikiness control

| Source | Mechanism | Our analog |
|---|---|---|
| NaLaFormer §3.2 | Query-norm-dependent power kernel p(‖q‖) · d(q) for spikiness | — |
| EXPRESSIVENESS §4.5 | Replace GroupNorm with vector norm (L2/RMS) | **Stage 6 active** |
| RWKV-6 baseline | GroupNorm post-WKV, per-head temperature as mechanism flag | Baked in / optional |

**Reading:** normalization of the attention output (or analog) is a
bigger lever than we initially gave it credit for. The Stage-6
`rwkv6_rmsnorm` run currently checks one instance of this. NaLaFormer
is a second instance on the query side rather than the output side —
and our model doesn't have that denominator, so NaLaFormer's variant
doesn't port. This pattern is "interesting but mostly already tested."

### Pattern D — Structured data-dependent transition matrices

| Source | Structure of M (in P = A⊙M framework) |
|---|---|
| Mamba-2 (Dao & Gu 2024) | 1-semiseparable, scalar gate α_t per token |
| Gated DeltaNet (Yang et al. 2024) | Semiseparable, with Householder (I - k k^T) transition |
| Log-Linear Mamba-2/DeltaNet (Guo et al. 2025) | Semiseparable ⊙ hierarchical (Fenwick) |
| RWKV-6 (ours) | Semiseparable (product of decays per channel) |
| RSE (ours) | Semiseparable, complex-valued per 2×2 block |
| P²-RSE (ours) | Two independent semiseparable structures, real β mixer |

**Reading:** the general framework is "separate A (feature interactions)
from M (temporal structure), and make M sub-quadratic via some
structure." This is effectively the Stage 3/4/5 design principle.
Log-Linear is the natural next step in this direction: compose the
semiseparable mask with a hierarchical one.

---

## From Stage 5 (deferred internally)

### ~~Phase 2b — independent-λ P²-RSE~~ — **FALSIFIED** (2026-04-20)
Tested as `rwkv6_p2rse_indeplam_strong_viscosity` at Stage-6.  Regressed
15 σ outside seed noise (dev 0.1394 / test 0.1383). The extra ~200 K
params in the decay LoRA cannot find a stable configuration against
shared viscosity coupling within 30 epochs. Do NOT retry in the
strong+viscosity composition. The historical hypothesis that "shared-λ
compresses the paired-pole expressivity" was falsified here.

### ~~Phase 2b-ext — independent (k, v) per pole~~ — **dropped**
Conditional on Phase 2b success. 2b failed ⇒ no evidence base to justify
adding independent drive on top. Not revisiting at this scale.

---

## From EXPRESSIVENESS paper (Mongaras & Larson, arXiv 2507.23632)

Link: https://arxiv.org/abs/2507.23632 · local: `papers/ EXPRESSIVENESS_2507.23632v1.pdf`

### ~~Full Kronecker n=2 at all layers~~ — **TESTED as low-rank** (2026-04-20)
Ran as `rwkv6_qtail_lowrank_all` (all-layer, K'=16 Eckart-Young-optimal
truncation). Best Stage-6 feature-side result at 0.1238 dev / 0.1240
test — MARGINAL. Full-rank K² Kronecker was NOT the source of value; a
small learned quadratic subspace (K'=16) captures the same signal at
16 % VRAM and 37 % wall-clock. Open question: whether qtail stacks with
transition-side wins — but the queue priority is LION not causal RWKV
at this scale.

### Cross-channel Kronecker on LION mode
LION already has explicit `r @ k^T` T×T attention, so the Kronecker lift
maps into LION's matrix attention form more naturally than into the
recurrent scan. Replace `lion_parallel_attention(r, k, v, w)` with a
branched form adding `(r⊗r) @ (k⊗k)^T` at the top layers only, mixed
via β_qtail init 0. Tests the paper's claim on the bidirectional track.

---

## From NaLaFormer (Meng et al., arXiv 2506.21137)

Link: https://arxiv.org/abs/2506.21137 · https://huggingface.co/papers/2506.21137

**Context:** Norm-aware linear attention with query-norm-modulated spikiness
and cosine/sine phase lifting for non-negativity. Target architecture is
L1-normalized linear attention (Katharopoulos form) — not RWKV recurrence.
Vision-first; 340M LM is a side dish.

**Why it doesn't apply to causal RWKV:** the paper's critique is that in the
explicit L1-normalized form `Σφ(k)^T v / Σφ(k)^T`, the query norm `‖φ(q)‖`
**algebraically cancels** between numerator and denominator, making attention
weights blind to query magnitude. Our causal RWKV-6 doesn't use that
denominator — we accumulate into a state S_t and normalize via GroupNorm on
the output. The "norm unawareness" issue they identify simply doesn't exist
in our architecture.

**Where it could apply: LION / bidirectional track.** LION uses explicit
parallel T×T attention that *does* look more like the Katharopoulos form.

### Query-norm-aware kernel in LION
Replace LION's identity feature map on `(r, k)` with:
```
phi_q(q) = d(q) · p(||q||),   p(x) = λ·(0.5 + tanh(x))
phi_k(k) = k^λ                 # element-wise power, fixed
```
- `d(q) = q / ||q||` is the direction component.
- `p(·)` is a norm-aware power, tanh-bounded to avoid overflow.
- λ is a single hyperparameter (paper uses λ ≈ 3 for ViT; tune for audio).
- Cost: <1% runtime, +2 params per head (λ could be per-head learnable).
- Drop-in replacement in `src/models/lion_attention.py`.

### Cosine inhibit for non-negativity in LION
Replace element-wise clipping/ReLU on `(r, k)` with a phase lift:
```
phi(d(x)) = [cos(d(x)); sin(d(x))]    # 2d-dim output
```
- Doubles feature dim per head (`K=64 → 128`). Ptolemy's identity
  ensures same-direction pairs preserve dot-product magnitude;
  opposite-direction pairs cancel via phase.
- Structurally kin to RSE's complex-scan packing — overlap is high,
  novelty gain is therefore low. Low priority unless LION reopens.

---

## From Avey ("Don't Pay Attention", arXiv 2506.11305)

Link: https://arxiv.org/abs/2506.11305 · https://huggingface.co/papers/2506.11305

**Context:** Attention-free + recurrence-free architecture with four pieces —
Ranker (top-k past-split retrieval via MaxSim), Enricher (d→m expansion,
ReLU²), Contextualizer (cosine-similarity-weighted mixing with GLU gate),
Fuser (concatenate and project). Trained LMs up to 1.52B params.

**Why the overall architecture doesn't help us:**
- The headline result is **length extrapolation**: trained on 512 tokens,
  tested on 64k (S-NIAH-1: **97.8%** vs Mamba 0%, RWKV-7 0.8%). Our ASR
  post-subsampling sequences are **≤500 frames** — we don't need
  extrapolation past training length. RWKV-6's constant-memory recurrence
  already covers what we need at this length.
- On the short-range benchmarks that actually match our regime, Avey
  **underperforms RWKV-7 at matched parameter count**:
  - Avey-1.52B: 51.53 avg zero-shot
  - RWKV-7-1.5B: **53.17** avg zero-shot
- The Ranker/MaxSim retrieval component is orthogonal to recurrent
  sequence modeling and would require abandoning RWKV's streaming
  property.
- ReLU² activation in the Enricher is already present in RWKV-6 ChannelMix.
- 4× embedding expansion in the Enricher is already present in RWKV-6
  ChannelMix (`ffn_dim = 3.5·d ≈ 4×`).

**Two component ideas worth parking:**

### Partial-embedding bypass in ChannelMix (Stage-7 candidate)
Avey Table 11 ablation: removing the head/tail bypass costs +8.5% PPL,
−2.2% accuracy — meaningful as regularization against over-smoothing.
Port to RWKV-6 ChannelMix:
```
# current RWKV-6 ChannelMix:  x → Linear(d→ffn) → relu² → Linear(ffn→d) → out
# proposed bypass variant:
#   x → Linear(d→ffn) → split into (z_head, z_tail), z_tail_frac ∈ (0,1)
#   z_head → bypass straight to concat
#   z_tail → relu² → receptance gating
#   out = Linear(concat(z_head, z_tail_mixed) → d)
```
- `z_tail_frac = 0.5` per Avey ablation.
- Zero new params; minor re-shaping only.
- Single backbone: `rwkv6_chanmix_bypass`.
- Diagnostic first: check activation-entropy-per-layer on current RWKV-6 —
  if over-smoothing is present at depth, this should help; if not, skip.

### Cosine-similarity-modulated mixing for LION (speculative, stage-8+)
Avey Contextualizer core: `V ⊙ N(Z)·N(Z)^T · Z`, where V is learnable and
`N(Z)·N(Z)^T` is the pairwise cosine-similarity matrix. Combined with
NaLaFormer's norm-aware kernel, this gives a "cosine-gated LION" that
replaces the current decay-only attention mask with a content-aware mask.
- O(T²) training complexity — no free lunch vs LION baseline.
- Substantial fork of `src/models/lion_attention.py`, not a small diff.
- Only worth it if the LION/BiRWKV track returns to the thesis's main claim.

---

## From Log-Linear Attention (Guo, Yang, Goel, Xing, Dao, Kim, arXiv 2506.04761)

Link: https://arxiv.org/abs/2506.04761 · https://huggingface.co/papers/2506.04761
Code: https://github.com/HanGuo97/log-linear-attention

**Context:** Sits between linear attention (fixed O(1) state) and full
softmax (O(T) KV cache) by maintaining a **logarithmically growing set of
hidden states**, one per Fenwick-tree bucket of the prefix. Each bucket at
level ℓ covers 2^{ℓ-1} consecutive past tokens. Output is a data-dependent
mixture over the per-bucket states:
```
o_t = Σ_{ℓ=0}^{L-1} λ_t^(ℓ) · q_t^⊤ S_t^(ℓ),    L = O(log T)
```
where λ_t^(ℓ) is a learnable per-scale, per-token coefficient. Training has
matmul-rich parallel form via a *hierarchical* mask M^H, composed with any
existing semiseparable mask M^S:
```
O = (Q K^⊤ ⊙ M^S ⊙ M^H) V     # Log-Linear Mamba-2
```
Total compute O(T log T), memory O(log T).

**Results:** Log-Linear Mamba-2 outperforms linear Mamba-2 on 8/9 NIAH
metrics; Log-Linear Gated DeltaNet outperforms linear Gated DeltaNet on all
multi-needle metrics and closes most of the gap to matched-size Transformer.
On short-context WikiText PPL, both log-linear variants beat their linear
counterparts marginally.

**Why this is interesting for us (unlike Avey):**
- Applies **on top of** existing linear-attention variants — it doesn't
  replace the model, it generalizes M. RWKV-6 fits the P = A ⊙ M framework
  (A = r k^⊤ with the appropriate feature map, M = semiseparable via
  cumulative product of decays). So log-linear extends to RWKV-6 the same
  way it extends to Mamba-2.
- Formalizes the *multi-scale depth hierarchy* pattern (§0 Pattern A) that
  keeps surfacing in our experiments — gen2's depth-graded α, rse_depth,
  Multi-Rate RSE — but from a different angle (per-position temporal
  bucketing rather than per-layer parameter allocation).
- Gated DeltaNet's recurrence `S_t = α_t S_{t-1}(I − k_t k_t^⊤) + v_t k_t^⊤`
  is directly supported by our `rwkv6_delta` / `lion_delta` code — the gap
  between "where we are" and "log-linear over our stack" is a hierarchical
  mask composition, not a new scan kernel.

### Log-Linear RWKV-6 (Stage-7 stretch candidate)
Extend the existing `_chunked_wkv` kernel with Fenwick-tree bucket
partitioning:
1. Maintain ⌈log₂ T⌉ per-head state tensors `S_t^(ℓ)` alongside the primary
   WKV state.
2. Add a learned projection `lambda_proj: hidden → H · L` producing
   per-head per-scale mixer weights `λ_t^(ℓ) ∈ ℝ^{H×L}` per token.
3. Output: `o_t = Σ_ℓ softplus(λ_t^(ℓ)) · r_t^⊤ S_t^(ℓ)`.
4. Parallel form: compose the existing decay-product mask with the
   hierarchical mask `M^H`.
5. At our T ≤ 500, L = 9 scales — modest state multiplier, tractable memory.

**Param cost:** single Linear(hidden → H·L) per layer ≈ +9·H·D = ~2.3K
params per layer, below parameter-parity noise floor.

**Where it could shine on ASR:**
- Acoustic signal is hierarchical in time (phoneme ~50ms → syllable ~250ms
  → word ~500ms → utterance-level prosody). Fenwick-tree exponential
  buckets match this 2× scale hierarchy almost exactly.
- Would be a novel combination (log-linear RWKV-6) that the paper doesn't
  claim — Guo et al. only test Mamba-2 and Gated DeltaNet.

### Log-Linear × RSE composition
Log-linear's hierarchical mask composes with *any* semiseparable M. RSE's
complex-valued per-block decay fits the semiseparable framework. So the
composite `M^RSE ⊙ M^H` is mathematically well-defined and would inherit
both RSE's rotation expressiveness and log-linear's multi-scale memory.
Implementation cost is approximately additive: extend `_forward_recurrent_rse`
with per-bucket complex state, mixer weights λ_t^(ℓ) applied to the
readout. If log-linear on plain RWKV-6 shows signal, this is the natural
step-2.

---

## Ideas considered and rejected

### Literature-side (out of thesis scope for this architecture / regime)

- **Avey Ranker / full architecture.** Designed for 64k-token LM
  extrapolation; our ASR sequences are ≤500 frames. Avey also
  underperforms RWKV-7 at matched scale on short-range tasks. The win it
  sells is in a regime we don't operate.
- **NaLaFormer on causal RWKV-6.** Its mechanism targets the
  L1-normalized `φ(q)Σφ(k)^⊤ v / φ(q)Σφ(k)^⊤` form; we don't have that
  denominator.
- **ReLU² activation.** Already present in RWKV-6 ChannelMix.
- **Avey's 4× embedding expansion.** Already present in RWKV-6 ChannelMix
  (ffn_dim = 3.5·d ≈ 4×).
- **Log-Linear Mamba-2 itself** (as a baseline). Our Mamba-2 track is
  separate from the RWKV-6 main thesis line; adding a log-linear Mamba-2
  would be a parallel experimental effort, not a direct improvement of
  the existing code. Revisit only if the Mamba-2 track becomes primary.

### Our-codebase-side (empirically closed after Stage 7A / 8 at this scale)

These were live hypotheses at the start of Stage 7. Each has its own
results + diagnostic writeup. All three reached the same pre-registered
AMBIGUOUS-ENGAGED leaf: SGD used the parameter, the mechanism's action
is measurable, CER did not improve.

- **Within-block readout gauge (`rwkv6_rse_dphi_viscosity`, A1′).**
  STAGE7A_RESULTS.md: W_φ moved substantially but post-rotation
  \|Im\|/\|Re\| went *up* at every layer (0.27 → 0.43 at L0, 0.56 → 0.93
  at L5). The "discarded quadrature" diagnosed in STAGE7_DIAGNOSTICS is
  sum-over-blocks cancellation that the anchor already exploits as its
  implicit gauge — not lost signal waiting for a phase.
- **Cross-layer complex residual (7A-plus, spec only).** Motivation was
  the depth-graded Im/Re growth in the pre-training diagnostic. Once A1′
  falsified the within-block premise, the same premise across blocks is
  unsupported. Spec remains in STAGE7A_PLUS_SPEC.md for future scale-up.
- **Recurrent delta rank-1 erase (`rwkv6_delta_warmstart_fixed`, T1).**
  STAGE8_RESULTS §6: mechanism NOW properly wired (the Stage-6 delta null
  was an implementation artefact, retracted). g_δ engaged to 0.65 at L5,
  β_eff p95 > 1, up to 24 % of state norm erased per token, CER = 0.1258
  (exact tie with vanilla). Delta is a clean null at T ≪ d² = 4096; would
  only become productive at scales where state saturation is the binding
  constraint.
- **Dense polar 2×2 non-normal RSE (`rwkv6_nonnormal_rse_viscosity`, T2).**
  STAGE8_RESULTS §3 and
  [stages_2_9_summary.md — Stage 8](stages_2_9_summary.md#stage-8--transition-geometry-pivot):
  ρ-LoRA F 18–30 per layer; non-normality score 0.04–0.11 per layer,
  higher at L0 and L5; bimodal per-head specialisation at L0 (one head
  |ρ|=0.46, others 0.06–0.14). Mechanism is real and structurally
  selective. CER = 0.1202 dev / 0.1200 test — AMBIGUOUS band.
  Spectral radius D8 shows the tail of G matrices is near 1 in ~6 %
  of tokens at L0 and L2, so the dense deployment operates at
  borderline stability. Reading: dense per-token enrichment is
  *engaged* but not *rewarded* when the useful action is concentrated.

### Stage 9 closed — sparse non-normal transition does not break the ceiling at this scale

See STAGE9_RESULTS.md for the full writeup.

- **S9B** `rwkv6_sparse_nonnormal_rse_edge_only_viscosity` — dev 0.1218 /
  test 0.1216. Decision-tree leaf: **REGRESSION STABILITY**. Hard
  edge-only dispatch (L0 + L5 non-normal, middle layers plain
  RSE+viscosity) pushed the spectrum to the Jordan boundary
  (11.6 % of L5 G-matrices with \|λ\| > 0.99, max 1.0002) without
  matching the dense-T2 or paired-pole `p2rse_strong_viscosity`
  baselines. Gates ended at 0.54–0.70 on the two edge layers —
  soft amplitude scaling, not structural sparsity.
- **S9A** `rwkv6_sparse_nonnormal_rse_viscosity` — **halted at ep 15**
  (dev 0.1467) after tracking ~0.006 behind `p2rse_strong_viscosity`
  and ~0.005 behind T2 through the first half of training. Projected
  final ~0.1245 (regression band). All 24 gates ended in
  [0.52, 0.71] — same amplitude-scaling pattern; no
  push-to-{0, 1} selector emerged.
- The Stage-9 parameterisation choices that plausibly caused the
  regression: (a) sigmoid gate has no penalty driving values to 0/1,
  so it became a damping knob instead of a selector; (b) κ tightened
  0.6 → 0.4, which combined with the g ≈ 0.5 init yields effective
  κ ≈ 0.2 at early training; (c) static ψ removes a freedom T2 was
  using (std ψ ≈ 1.7 rad per layer in T2's D6). These are reading
  notes, not tested causal claims.

---

## Paper reviews

Paper-by-paper reviews assembled iteratively with the PI. Each entry is a
self-contained mechanism record pitched at the **plain RWKV-6 baseline** —
not at compositions with current winners (RSE, RSE+viscosity, qtail, ...).
Compositions are a downstream decision: first each mechanism gets a clean
baseline read, then the best are composed. Stages are also not assigned
here — sequencing is decided once the review queue is stable.

Each entry contains:
- **Paper concept** — one-sentence headline.
- **Can this reformulate to plain RWKV-6?** — honest yes / no / conditional,
  with the reason.
- **Core mathematical idea (baseline)** — minimal equations for the
  mechanism as applied to vanilla RWKV-6.
- **RWKV-6 adaptation plan (baseline only)** — concrete code path,
  parameter count, cost.
- **Transferability expectation** — Mamba-2 / Linear Attention / Lion,
  one line each.
- **Expected gain vs. overhead** — upper tail, lower tail, and what
  distinguishes the mechanism from closed AMBIGUOUS-ENGAGED directions.

### Paper 1 — Log-Linear Attention (Guo et al., arXiv 2506.04761)

Link: https://arxiv.org/abs/2506.04761 · code: https://github.com/HanGuo97/log-linear-attention

**Paper concept.** Replace the single compressed recurrent state with
$L = O(\log T)$ Fenwick-tree bucket states that form a **disjoint partition**
of the prefix at every query time $t$, plus a per-token per-scale learned
mixer $\lambda_t^{(\ell)}$ that combines them at the readout. Training
$O(T \log T)$, inference memory $O(\log T)$.

**Can this reformulate to plain RWKV-6? YES, cleanly.** The paper derives
Log-Linear Mamba-2 and Log-Linear Gated DeltaNet from the unified framework
$P = A \odot M$ with $A$ an outer-product-style term and $M$ any lower-
triangular mask. Vanilla RWKV-6 fits this framework directly:
- $A_{t,s} = \sum_i r_{t,i}\,k_{s,i}$ (channel-summed dot product per $(t,s)$);
- $M^S_{t,s,i} = \prod_{\tau=s+1}^{t} w_{\tau,i}$ — semiseparable, per-channel.

The hierarchical mask $M^H_{t,s} \in \mathbb{R}$ is a per-$(t,s)$ scalar and
broadcasts over the channel dimension, so $M^S \odot M^H$ is well-defined.
In the recurrent form, the matrix state $S_t \in \mathbb{R}^{H \times K \times K}$
generalises to $L$ bucket states $\{S_t^{(\ell)}\}$ evolving under the same
per-step diagonal decay $w_t$; the buckets partition $[0, t)$ by
construction, so $\sum_\ell S_t^{(\ell)} = S_t$. Setting $\lambda^{(\ell)} \equiv 1$
recovers vanilla RWKV-6 bit-exactly — clean zero-regression-at-init contract.

**Core mathematical idea (baseline).**

Readout:
$$y_{t,h,j} = \sum_{\ell=0}^{L-1} \lambda_{t,h}^{(\ell)} \cdot
  \sum_i r_{t,h,i}\, S_{t,h,ij}^{(\ell)} \;+\; u_{h,j}\,(r_{t,h} \odot k_{t,h})_j\,v_{t,h,j}.$$

Bucket evolution at step $t$ (Fenwick rule):
1. Decay all buckets: $S_t^{(\ell)} \leftarrow w_t \odot S_{t-1}^{(\ell)}$.
2. Consolidation: if $\mathrm{lsb}(t+1) = \ell^*+1$ and $\ell^* > 0$, then
   $S_t^{(\ell^*)} \mathrel{+}= S_t^{(\ell^* - 1)}$ and $S_t^{(\ell^*-1)} \leftarrow 0$.
3. Write observation into bucket 0: $S_t^{(0)} \mathrel{+}= k_t v_t^\top$.

Mixer parameterisation (data-dependent, per-head, per-scale):
$$\lambda_{t,h}^{(\ell)} = 1 + \big(W_\lambda^{(2)} \tanh(W_\lambda^{(1)} x^{(w)}_t)\big)_{h,\ell},$$
with $W_\lambda^{(1)} \in \mathbb{R}^{D \times D_\lambda}$, $W_\lambda^{(2)} \in \mathbb{R}^{D_\lambda \times H \cdot L}$,
both zero-init. $D_\lambda = 32$ matches the codebase LoRA convention. $x^{(w)}$
is the token-shifted temporal-modulation stream already computed in
`_compute_rkv_gw`. At init $\lambda \equiv 1$, and
$\sum_\ell S_t^{(\ell)} = S_t$ → readout bit-identical to vanilla.

**RWKV-6 adaptation plan (baseline only).**
- Backbone: `rwkv6_loglinear` — plain RWKV-6 + Log-Linear only, no RSE /
  viscosity / qtail. Answers the isolated question "does hierarchical
  prefix partition convert into CER on the bare baseline?". Compositions
  are deferred.
- Code path: new method `_forward_recurrent_loglinear` beside
  `_forward_recurrent` in `src/models/rwkv6_time_mix.py`. Carry-state
  shape widens from `(B, H, K, K)` to `(L, B, H, K, K)` with $L \approx 10$;
  `RWKV6Encoder.init_state` needs the extra leading dim. Fenwick
  consolidation schedule precomputable from $T$ alone (a $T$-length int
  tensor that returns the level-to-consolidate, or `-1`).
- Variable-length batches: the Fenwick schedule is position-local, so
  identical across samples in a chunk; only validity differs — handled
  by masking $\lambda$-contributions at padded positions.
- Parameters: $D \cdot D_\lambda + D_\lambda \cdot H \cdot L \approx
  256 \cdot 32 + 32 \cdot 40 \approx 9.5\,\text{K}$ per layer,
  $\sim 57\,\text{K}$ encoder-wide (0.8 % of 7 M).
- Cost: wall-clock $\sim 2\text{--}3\times$ vanilla at $T \le 500$, $L = 10$
  (per-step cost is $L\times$ state work plus the $L$-element readout sum).
  Memory: $L \times$ state, $\sim 30\,\text{MB}$ extra for 6 layers at our
  batch size. Pure PyTorch is sufficient for a 30-epoch prototype; the
  paper's Triton kernel is tuned for 16k+ sequences and not needed at
  our lengths.

**Transferability expectation (Mamba-2 / Linear Attention / Lion).**
- **Mamba-2**: HIGH. Paper-direct. $M^S$ 1-semiseparable (scalar $\alpha_t$)
  × $M^H$ hierarchical is exactly the Log-Linear Mamba-2 construction;
  `src/models/mamba2_encoder.py` is the target. Triton kernel available.
- **Linear Attention**: HIGH in principle, CONDITIONAL in our codebase.
  Paper-direct for the causal recurrent Katharopoulos form (Gated
  DeltaNet). Our baseline in `src/models/blocks.py` is bidirectional
  parallel ELU+1 with no recurrent carry — a causal-recurrent LA variant
  must be added first; once added, Log-Linear ports cleanly.
- **Lion (bidirectional)**: MODERATE. Fenwick is causal-prefix. Natural
  bidirectional form is mirror-Fenwick — causal $M^H_{\mathrm{fwd}}$ on
  the lower triangle, suffix-Fenwick $M^H_{\mathrm{bwd}}$ on the upper
  triangle, independent $\lambda^{(\ell,\mathrm{fwd})}, \lambda^{(\ell,\mathrm{bwd})}$.
  Mathematically clean; paper does not test it. Design-cost cell.

**Expected gain vs. overhead.**
- Upper tail: MARGINAL at $T \le 500$, $L = 10$. The asymptotic story
  (wins at 16k+) does not apply at our lengths. Strongest physical
  argument: acoustic hierarchy is natively $\sim 2\times$
  (5-frame phoneme → 10-frame transition → 25-frame syllable →
  50-frame word) and matches Fenwick scales closely.
- Lower tail: NULL. RWKV-6 is not state-saturated at our lengths —
  Stage-8 T1 delta-rule result rules out "more memory" as the binding
  constraint. If hierarchical readout doesn't unlock something
  qualitatively new, the mechanism has no clear conversion path.
- Structural distinction from closed AMBIGUOUS-ENGAGED directions
  (A1′ / T1 / T2 / S9A / S9B): Log-Linear does NOT add per-(token, head,
  block) freedom to the transition operator — the axis the
  cross-experiment invariant has already closed. It changes the
  prefix-summary structure the readout addresses. Distinct from
  already-falsified Multi-Rate RSE (`rse_m2`, `rse_m4`), which held
  $M$ overlapping prefix summaries in the same operator family;
  Fenwick buckets are disjoint partitions. This is the first mechanism
  in the review queue that clears the structural-vs-dense filter.
- Regardless of CER outcome, the per-level $\lambda_{t,h}^{(\ell)}$
  mobility by layer is a clean diagnostic on whether the multi-scale
  depth hierarchy observed in Stage-2 `gen2` / Stage-4 `rse_depth`
  extends to the per-position axis.

**Compositions (defer).** Composition with RSE, RSE+viscosity, qtail,
non-normal, and friends is mathematically well-defined ($M^S$ complex or
polar × $M^H$ real-scalar still gives a well-defined entry-wise product),
but out of scope for this review. Decided once the baseline read is in.

### Paper 2 — NaLaFormer (Meng et al., arXiv 2506.21137)

Link: https://arxiv.org/abs/2506.21137 · https://huggingface.co/papers/2506.21137

**Paper concept.** Two linked fixes for L1-normalized linear attention
(Katharopoulos form $o_t = \phi(q_t)\sum_i \phi(k_i)^\top v_i \,/\, \phi(q_t)\sum_j \phi(k_j)$):
(i) a **norm-aware power kernel** $\phi_q(q) = \|d(q)\|^{p(\|q\|)} \cdot [\cos(d(q)); \sin(d(q))]$
with $p(x) = \lambda(0.5 + \tanh(x))$, restoring query-norm sensitivity
that otherwise algebraically cancels between numerator and denominator
(paper Thm. 1); (ii) a **Ptolemy cosine/sine phase lift** $[\cos(d(x)); \sin(d(x))]$
on queries and keys as a sign-preserving alternative to ReLU/ELU+1 for
non-negativity.

**Can this reformulate to plain RWKV-6? CONDITIONAL / LOW VALUE.**
Mechanism is portable as a feature-map reparameterisation; the pathology
the paper solves is absent in RWKV-6 by construction.

RWKV-6 core:
- Causal: $S_t = \text{diag}(\exp(w_t)) S_{t-1} + k_t v_t^\top$,
  $y_t = r_t^\top S_t + u \odot (r_t \odot k_t) v_t$, then GroupNorm.
- LION: $Y = (A_\text{fwd} + A_\text{bwd}) V$ via `lion_parallel_attention`
  — confirmed by reading `src/models/lion_attention.py`, no L1 denominator.

The denominator $\sum_j \phi(k_j)$ that cancels $\|\phi(q)\|$ in
Katharopoulos does not exist on either path. Output magnitude scales
with $\|r_t\|$ uncancelled, so "query-norm unawareness" is not a failure
mode any of our current baselines exhibits. This corrects the earlier
"Where it could apply: LION / bidirectional track" framing in the
literature section above — LION is closer to Katharopoulos in *parallel
form* but not in *normalization structure*.

Once the diagnosis is removed, each fix becomes a standalone per-token
feature reparameterisation and collides with the cross-experiment
invariant: dense per-(token, head, block) scalar modulators tied to a
data-dependent projection of $x_t$ engage SGD but do not convert into
CER at 7 M / clean-100 / 30-ep. The structural analogy is with A1′
(data-dep readphase, +2σ engaged null), T1 (recurrent delta
$\beta_t = g \cdot \sigma(\tilde\beta_t) \cdot 2$, tied-vanilla engaged
null), T2 (dense non-normal $\rho, \psi$ per token, +1σ ambiguous) —
all closed AMBIGUOUS-ENGAGED.

A secondary concern specific to this paper: the cosine/sine phase lift
overlaps conceptually with RSE's existing complex-scan packing
(both embed real features into 2D rotation-equivariant space — RSE on
the transition operator via $z_t = \exp(-\lambda + i\theta)$, NaLaFormer
on the feature map). The transition-side phase geometry has already won
(+6 % vs vanilla); adding feature-side phase geometry is not in an
independent direction and raises double-counting risk.

**Core mathematical idea (baseline, minimum viable port).**

Signed-power form (cleanest zero-regression-at-init path for RWKV-6):
$$\tilde r_t = \mathrm{sign}(r_t) \odot |r_t|^{p_{t,h}}, \quad \tilde k_t = \mathrm{sign}(k_t) \odot |k_t|^{\lambda_h}$$
$$p_{t,h} = 1 + \beta_h \tanh\big((W_p x^{(w)}_t)_h\big), \quad \lambda_h = 1 + \gamma_h$$

$\beta_h, \gamma_h, W_p$ zero-init → $p \equiv 1, \lambda \equiv 1$ at init
→ $\tilde r = r,\,\tilde k = k$ bit-exactly. State and readout unchanged:
$$S_t = w_t \odot S_{t-1} + \tilde k_t v_t^\top, \quad y_t = \tilde r_t^\top S_t + \text{bonus}_t.$$

Keeps linear-time recurrence and signed interactions; avoids NaLaFormer's
positivity machinery (which RWKV-6 does not need because there is no
ReLU/ELU+1 in the feature map).

The full Ptolemy cosine/sine lift $\psi(x) = a(x) \odot [\cos(d(x)); \sin(d(x))]$
is NOT part of the baseline plan: doubles feature dim per head ($K = 64 \to 128$),
requires a $2K \to K$ post-projection to preserve the scan state shape
$(B, H, K, K)$, and has no clean zero-regression-at-init contract (the
projection would need a specific direction-selector init to recover $r$
from $[\cos(d(r)); \sin(d(r))]$, which is awkward). Deferred.

**RWKV-6 adaptation plan (baseline only, if we decide to run it).**
- Backbone name: `rwkv6_norm_aware`. Flag `use_norm_aware_kernel: bool = False`.
  Implemented between `_compute_rkv_gw` and scan dispatch.
- Parameters: $\beta_h, \gamma_h$ per head (2H = 8 scalars per layer = 48
  encoder-wide) + $W_p \in \mathbb{R}^{D \times H}$ zero-init
  ($256 \times 4 = 1$ K per layer, 6 K encoder-wide). Total ~6 K params,
  <0.1 % of 7 M encoder.
- Cost: per step = one $L_2$ norm + one `torch.pow` + one sign mult on
  each of $r$ and $k$. Pure PyTorch, no CUDA. Wall-clock overhead
  negligible (<5 % of vanilla scan).
- Zero-regression contract: exact via $\beta = \gamma = 0$, $W_p = 0$ init.

**Transferability expectation (Mamba-2 / Linear Attention / Lion).**
- **Mamba-2**: WEAK-TO-MODERATE. Same architecture class as RWKV-6 —
  state-accumulation without L1 denominator, pathology equally absent.
  A norm-aware reparameterisation on the $B, C$ selective-projection
  tensors is heuristic, not a faithful paper transfer.
- **Linear Attention (true Katharopoulos with L1 denominator)**: HIGH.
  This is the paper's stated target. Our `blocks.py` `LinearAttentionLayer`
  uses ELU+1 bidirectional parallel *without* explicit L1 normalization —
  partial fit. For the paper's claim to match the architecture we would
  first add explicit $o_t = \phi(q_t) S_t / (\phi(q_t) z_t)$ with scalar
  running-sum $z_t$, then swap ELU+1 → NaLaFormer $\phi_q, \phi_k$.
  That adds a new architecture class to the thesis scope, not a
  modification of existing baselines.
- **Lion (bidirectional)**: MODERATE-LOW. Closest to Katharopoulos in
  parallel form but `lion_parallel_attention` has no L1 normalization,
  so the paper's diagnosis does not fire. Mechanism portable as
  feature-map reparameterisation; overlaps with RSE complex-scan phase
  geometry (both embed real features into 2D rotation-equivariant space).

**Expected gain vs. overhead.**
- Upper tail: MARGINAL on causal RWKV-6. Without the L1-cancellation
  pathology the "fix" becomes a generic per-token feature-map power
  modulator.
- Lower tail: NULL. Pattern-matches the closed AMBIGUOUS-ENGAGED class —
  per-token scalar multiplier tied to a data-dependent projection of
  $x_t$. Five independent prior probes in that class (A1′, T1, T2,
  S9A, S9B) all engaged without converting.
- Feature-side precedent: Stage-6 `qtail_lowrank_all` at 0.1238 dev —
  the best feature-side result in the chain — remained materially above
  the 0.1185 transition-side anchor. Independent data point that
  feature-side reparameterisations at this scale are MARGINAL at best.
- Structural caveat: phase-geometry double-counting with RSE
  (transition-side win already banked) makes feature-side phase lift
  structurally non-independent.
- Low-priority probe. Not on the current critical path.

**Natural home (if thesis scope expands).** Adding a true Katharopoulos-form
Linear Attention baseline with explicit L1 denominator would give the
paper's mechanism its actual test-bed. That is a new architecture-class
addition rather than a modification of existing baselines — decided
separately from this review queue.

### Paper 3 — Avey ("Don't Pay Attention", Khatami et al., arXiv 2506.11305)

Link: https://arxiv.org/abs/2506.11305 · https://huggingface.co/papers/2506.11305

**Paper concept.** Attention-free + recurrence-free full architecture,
four pieces — **Ranker** (fixed-width sequence splits + ColBERT-style
MaxSim retrieval, selects top-$b$ past splits for the current split),
**Enricher** ($d \to m$ expansion with **partial-embedding bypass** into
head/tail fractions, ReLU² on the tail), **Contextualizer** (embedding-
wise, cross-token, cosine-similarity-weighted mixing
$c(Z_t) = Z_{tl} \odot \sigma\big((V \odot \mathcal{N}(Z_{tr})\mathcal{N}(Z_{tr})^\top) Z_{tr} + b'\big)$,
$O(T^2)$), **Fuser** (concat head-bypass with contextualized tail, project).
Headline result: trained at 512-token window, S-NIAH-1 at 64 k context
97.8 % vs Mamba 0 %, RWKV-7 0.8 %.

**Can this reformulate to plain RWKV-6? COMPONENT-ONLY / LOW-PRIORITY.**
Avey is a full architecture replacement, not a drop-in mechanism.
Decomposing piece-by-piece against our stack:

| Avey piece | Nearest RWKV analog | Verdict |
|---|---|---|
| Ranker (split-based retrieval) | N/A — solves $T > $ training-window | **Drop.** Our $T \le 500$ post-subsampling is single-length; no context-width to extend. |
| Enricher ($d \to m$ expansion) | RWKV-6 ChannelMix ($d \to 3.5d$) | Already present. |
| Enricher (ReLU² activation) | RWKV-6 ChannelMix (ReLU²) | Already present. |
| Enricher (**partial bypass**) | — | **Only genuinely new idea.** Port. |
| Contextualizer (cosine-sim T×T mask) | RWKV-6 TimeMix (WKV scan) | **Drop.** $O(T^2)$, loses decay recency, overlaps closed-MARGINAL qtail. |
| Fuser (concat + project) | ChannelMix output projection | Already present. |

Net: the single portable sub-idea is partial-embedding bypass in
ChannelMix. Everything else is out-of-regime, out-of-axis, or already
present.

**Paper-headline context (relevant to our regime).** Avey's headline win
is length-extrapolation (trained at 512, evaluated at 64 k). Our regime
is $T \le 500$ single-length — extrapolation is not our target. On the
short-range benchmarks that match our regime, Avey under-performs RWKV-7
at matched scale: Avey-1.52 B 51.53 avg zero-shot vs RWKV-7-1.5 B 53.17.
Independent reason not to port the full architecture.

**Core mathematical idea (baseline, partial-embedding bypass only).**

Current RWKV-6 ChannelMix:
$$z = W_k(x_k) \in \mathbb{R}^m, \qquad y_\text{van} = \sigma(W_r(x_r)) \odot W_v\,\text{ReLU}^2(z).$$

Avey-style bypass adaptation (interpolation form — cleanest
zero-regression-at-init contract):
$$[z_h, z_t] = \text{split}(z; \rho = 0.5), \qquad \tilde z = \big[z_h \,\|\, \text{ReLU}^2(z_t)\big]$$
$$y = \sigma(W_r(x_r)) \odot \Big[(1-\alpha)\, W_v\,\text{ReLU}^2(z) \;+\; \alpha\, W_v\,\tilde z\Big].$$

$\alpha \in \mathbb{R}$ per layer, zero-init. At $\alpha = 0$:
$y = y_\text{van}$ bit-exactly. As $\alpha \to 1$: half of the expanded
hidden passes through $W_v$ without ReLU² — the partial bypass. $W_v$ is
reused, no new projection. Tail fraction $\rho = 0.5$ per Avey
Table 11 ablation.

**RWKV-6 adaptation plan (baseline only).**
- Backbone name: `rwkv6_chanmix_bypass`.
- Flag `use_chanmix_bypass: bool = False`. Location:
  `src/models/rwkv6_channel_mix.py`.
- Parameters added: 1 scalar ($\alpha$) per layer = **6 encoder-wide**.
  Negligible.
- Cost: near-zero — one split, one concat, one interpolation. Pure
  PyTorch. No CUDA.
- Zero-regression contract: exact via $\alpha = 0$ init.
- Diagnostic value regardless of CER outcome: mobility of $\alpha$ over
  training is an observable on whether over-smoothing (paper's claimed
  motivation for the bypass) is a binding constraint in our 6-layer
  stack. Paper's ablation: removing the bypass costs +8.5 % PPL and
  −2.2 % accuracy at 1.52 B — but binding strength is depth / scale
  dependent.

**Transferability expectation (Mamba-2 / Linear Attention / Lion).**
- **Mamba-2**: HIGH. Mamba-2 has a gated MLP / GLU block structurally
  analogous to RWKV ChannelMix. Port is equivalent — split, interpolate,
  zero-init $\alpha$.
- **Linear Attention**: HIGH. No `ChannelMix` per se in pure linear
  attention, but the FFN-after-attention slot is the natural home and is
  architecture-agnostic.
- **Lion (bidirectional)**: HIGH. ChannelMix is position-pointwise and
  independent of TimeMix mode. Transfers without change — causal /
  bidirectional distinction is irrelevant here.

**Expected gain vs. overhead.**
- Upper tail: PLATEAU-to-MARGINAL at 7 M / 6-layer / 30-ep. Paper's gain
  is at 1.52 B / deeper stacks where over-smoothing binds harder; our
  6-layer regime is shallower, so over-smoothing is unlikely to be a
  binding constraint here.
- Lower tail: NULL (engaged-or-not-engaged). Zero-regression contract
  protects against regression; SGD may leave $\alpha$ near 0.
- Structural orientation: feature/FFN-side, orthogonal to winning
  transition-side line (RSE + viscosity). Different axis from both the
  Stage-6 qtail feature-side line (which was tested on TimeMix, not
  ChannelMix) and the Stage-8/9 transition-side extensions.
- Tier-3 secondary probe. Cheap, clean zero-regression contract,
  orthogonal axis — keep in the queue, but not a ceiling-break
  candidate.

**Components explicitly not in the baseline plan.**
- **Contextualizer (cosine-similarity cross-token mixing).** Rejected.
  Breaks linear-time ($O(T^2)$), loses the decay-based recency bias that
  acoustic signal depends on, and overlaps with Stage-6 qtail
  feature-side territory already closed as MARGINAL. The paper's
  architecture compensates for the missing recency via the Ranker's
  split windowing — which we drop in our regime — so an isolated
  Contextualizer re-introduces all the long-tail cross-token
  contributions that the paper's own design explicitly suppresses.
  (Note: this corrects the earlier "if the LION/BiRWKV track returns to
  the thesis's main claim" framing in the §"From Avey" literature
  section above — our current cross-stage evidence does not support
  reopening that line.)
- **Ranker (split-based MaxSim retrieval).** Rejected. Solves a
  sequence-length $>$ training-window problem we do not have at
  $T \le 500$.

**Natural home (if thesis scope expands).** The full Avey architecture
as a separate baseline is possible but (a) is out of thesis scope
(length-extrapolation is not our target) and (b) under-performs RWKV-7
at matched scale on short-range benchmarks, so the expected evidentiary
value is low. Not on the critical path.

### Paper 4 — RWKV-X (arXiv:2504.21463)

Link: https://arxiv.org/abs/2504.21463 · https://huggingface.co/papers/2504.21463

**Paper concept.** A hybrid architecture combining a pre-trained RWKV-7
backbone with periodically inserted **Top-k Chunk Sparse Attention**
blocks, continually pretrained on 64 K-token sequences. Four engineering
ingredients:
1. **Top-k Chunk Sparse Attention** (MoBA-style). Partition the prefix
   into fixed-size chunks $C_i$ of $B$ tokens; score each with the chunk
   mean $\bar k_i = \frac{1}{B}\sum_{j \in C_i} k_j$, select
   $I_t = \mathrm{TopK}_k(\{q_t^\top \bar k_i\}_i)$, attend with exact
   softmax over $U_t = \bigcup_{i \in I_t} C_i$.
2. **SnapKV-style KV-cache management** for constant-memory inference
   (compresses past cache to fixed size $m$ via importance-score top-$m$).
3. **LLaMA-Pro block expansion + zero-init.** Start from a pre-trained
   RWKV-7 checkpoint, interleave new sparse-attention blocks with
   zero-initialised output projections, freeze base, unfreeze for long-
   context continual pretraining.
4. **LongCE loss.** Dynamic per-token CE reweighting emphasising tokens
   with long-range contextual dependencies.

Headline result: 1 M-token decoding with constant speed/memory; near-
perfect 64 K passkey retrieval. Short-context benchmarks preserved, not
improved (RWKV-X-3.6 B 71.9 avg ≈ RWKV-7-2.9 B 72.8).

**Can this reformulate to plain RWKV-6? NO / OUT-OF-REGIME.**

This is a systems / engineering paper, not a mechanism paper. It does
not propose, modify, or claim anything about the WKV operator family,
the transition group, the readout geometry, the feature map, or any
in-block expressivity mechanism — **the RWKV blocks in the hybrid are
used unchanged**. At our regime ($T \le 500$ post-subsampling, CTC-ASR
on 30-sec utterances, train-from-scratch in 30 ep) every one of its
four ingredients degenerates or is inapplicable:

| Ingredient | Behaviour at our regime |
|---|---|
| Top-k Chunk Sparse Attention | **Degenerates.** At $T=500$, $B=32$, $k=16$: $kB \ge T$, collapses to full attention (sparse aspect vanishes). At $k=4$: $kB/T \approx 0.25$, hides ¾ of acoustic context → actively hurts ASR (formants, phoneme boundaries, syllable envelopes all need the full utterance). |
| SnapKV KV-cache management | **Irrelevant.** Inference-time trick for constant-memory long-context decoding. 30-sec LibriSpeech utterances fit trivially; we are not memory-bound. |
| LLaMA-Pro block expansion | **Doesn't fit protocol.** Requires a pre-trained checkpoint to expand from. Our protocol is from-scratch 30-epoch training; no base to preserve via freeze-then-unfreeze. |
| LongCE loss | **Doesn't apply.** Designed for next-token CE on long LM sequences with long-range dependencies to up-weight. Our loss is CTC on short ASR utterances — no next-token dynamics, no long-range-dependency tokens to re-weight. |

**No math or tensor-op worth writing out against the RWKV-6 time-mix,**
because the mechanism's central primitive (top-k chunk sparse
attention) collapses in our regime.

**Only portable sub-idea, and it's not a novelty of this paper.** The
pattern "insert full-attention blocks periodically between RWKV blocks"
is the well-known Jamba / Samba / Zamba hybrid stack. It is
effectively already covered by the thesis's existing separate
Transformer and RWKV baseline columns. Running some layers as
Transformer and others as RWKV is a stack-architecture choice, not an
expressivity-mechanism probe, and does not cleanly attribute gains to
any single mechanism. If we ever wanted to run it, the constructive
form is a zero-init hybrid residual:
$$h_t = y_t^\text{rwkv} + \alpha \cdot \sigma(W_g x_t) \odot z_t, \qquad
  z_t = \mathrm{softmax}\!\Big(\tfrac{q_t K_{U_t}^\top}{\sqrt d}\Big) V_{U_t}$$
with $\alpha = 0$ at init, $U_t$ the union of top-$k$ selected chunks.
But the regime degeneracy above means at $T \le 500$ this is either
equivalent to a full-attention block (sparse aspect vanishes, reduces
to our existing Transformer baseline) or an actively-hurting
restricted-context block (sub-covers the utterance). No informative
signal is recoverable from the sparse framing at our $T$.

**Transferability expectation (Mamba-2 / Linear Attention / Lion).**
N/A across the board. The paper's motivation is constant-memory
long-context causal decoding via KV-cache compression + continual
pretraining from a pretrained checkpoint + long-context LM loss — all
four premises fail at our regime, on all our baselines. Bidirectional
Lion is additionally not causal, so the KV-cache compression story
doesn't apply even structurally.

**Expected gain vs. overhead.** N/A. Not a candidate for a baseline
probe at our scale; there is no mechanism in the paper whose
expressivity claim clears our regime's preconditions.

**Natural home (if thesis scope expands).** Long-form audio (hour-scale
utterances, streaming ASR at high sample rate). Adding RWKV-X then
becomes a scope-expansion exercise — a different thesis question
(sequence-length scaling, not operator-family expressivity). Not on
the current critical path.

**Reviewer-side caveat.** One of the three reviewer proposals for this
paper described RWKV-X as introducing "Cross-Mixing," a "Dynamic Decay
Matrix," "Eagle-Scan Injection," and "Fast Weights" — conflating with
RWKV-7 Eagle's actual v-first / data-dependent-decay features and/or
freely inventing mechanisms not in the paper. Flagging here so that if
the paper is revisited later, reviewers should verify against the
arXiv PDF rather than trusting the confabulated description: the
actual paper does not modify the WKV operator.

### Paper 5 — HSA / RAMba (arXiv:2504.16795)

Link: https://arxiv.org/abs/2504.16795 · OpenReview: https://openreview.net/forum?id=dIHSZTx9Lu

**Paper concept.** Hierarchical Sparse Attention (HSA) — a hybrid
sparse-retrieval module bolted onto an RNN backbone (Mamba → RAMba in
the paper). Four technical ingredients:
1. **Chunk selection via mean-pooled keys.** Divide sequence into
   fixed-size chunks $C_i$ of $S$ tokens; compute
   $\bar E_i = \tfrac{1}{S}\sum_{j \in C_i} E_{i,j}$; score with
   $s_{t,i,g} = q^\text{slc}_{t,g} {}^\top \bar E_{i,g} / \sqrt{d_g}$;
   pick top-$K$ chunks (grouped by attention group $g$).
2. **Two-stage hierarchical attention.** Token-level attention within
   each selected chunk; chunk-level weighted aggregation
   $\hat O_t = \sum_{k<K} \hat w_{t,k} \hat O_{t,k}$.
3. **Stick-breaking chunk weights with ordinal recency bias.**
   $\hat w_{t,k} = \hat\beta_{t,k} \prod_{i<k}(1 - \hat\beta_{t,i})$,
   $\hat\beta_{t,k} = \sigma(\hat s_{t, \hat I'_{t,k}})$, chunks
   ordered nearest-to-farthest. Critical detail: gradients flow
   through chunk weights into the selection, enabling end-to-end
   learnable retrieval — **this is the genuine mechanism contribution
   over prior sparse methods (NSA, MoBA).**
4. **Bidirectional chunk encoder** (~5.4 % of RAMba params) producing
   chunked memories, plus a hardware-aligned Triton kernel for the
   two-phase forward / backward pass.

Target: pre-train at 4 K context, extrapolate to 64 M-token passkey
retrieval on Mamba-2.

**Can this reformulate to plain RWKV-6? NO / OUT-OF-REGIME.**

HSA is a hybrid sparse-retrieval module at the block / encoder level,
not a time-mix modification. It does not modify the RWKV WKV operator,
transition family, readout, or feature map — the RNN backbone is used
unchanged. Per-ingredient regime check against $T \le 500$ / CTC-ASR /
train-from-scratch / 30-ep:

| HSA ingredient | Behaviour at our regime |
|---|---|
| Top-K chunk selection via mean-pooled keys | **Degenerates.** $T=500, S=64, K=8 \Rightarrow T/S \approx 8 \le K$, every chunk selected → HSA collapses to full attention (sparse aspect vanishes). $S=16, K=8 \Rightarrow 31$ chunks, $K = 8$ covers 25 % of sequence → hides ¾ of acoustic context → hurts ASR (formants, phoneme boundaries, syllable envelopes). |
| Two-stage hierarchical attention | No compute saving at $T \le 500$; dense attention already cheap. No expressivity add either. |
| Stick-breaking chunk weights with ordinal recency bias | **Same family** as already-tested `rwkv6_p2rse_softmax` (Stage-5 Phase-1 softmax mixer over ordered poles, 0.1220 dev / 0.1215 test — PLATEAU, classified "2-pole dynamics alone insufficient"). Stages-2-9 evidence puts this convex-combination-over-ordered-elements parameterisation in the PLATEAU-to-MARGINAL band, not BREAK. |
| Bidirectional chunk encoder | ~5.4 % of RAMba params. At our 7 M scale, breaks the 5 % parameter-parity rule. |
| Hardware-aligned Triton kernel | Unneeded at $T \le 500$; full attention is cheap. |

**The paper's genuine mechanism contribution (chunk-aware relevance
learning with gradients through chunk selection) is real but specific
to OOD context-length generalisation — a problem we do not have at
fixed $T \le 500$.** The ∂L/∂ selection-score story that makes HSA
more expressive than NSA / MoBA only matters when chunk selection
must generalise to contexts longer than pretraining; at fixed
training length there is no OOD selection error to correct.

**Sub-mechanism isolation doesn't clearly help here either.** Porting
HSA's stick-breaking fusion or two-stage hierarchical attention as
isolated RWKV-6 add-ons would layer per-token chunk-score freedom
($\hat s_{t,i}$) and per-token stick-breaking weights
($\hat w_{t,k}$) on top of the existing WKV state — exactly the
dense-per-(token, head, block) freedom-on-top-of-anchor pattern that
matches the cross-experiment invariant's failure class (A1′, T1, T2,
S9A, S9B — five prior engaged-nulls). Predicted band: MARGINAL at
best, PLATEAU likely.

**Constructive hybrid form if we ever wanted to test it (not
recommended at our scale).** Zero-init residual at a selected layer:
$$h_t^{(l+1)} = h_t^{(l)} + \alpha \cdot W_O \big[z_{t,1}^{(l)}; \dots; z_{t,G}^{(l)}\big], \quad
  z_{t,g}^{(l)} = \sum_{k=1}^K w_{t,k,g} \cdot \mathrm{Attn}\big(q_{t,g}^{(l)}, K_{I'_{t,g,k}}, V_{I'_{t,g,k}}\big)$$
with $\alpha = 0$ at init. But the regime degeneracies above collapse
this to either a full-attention block (reduces to our existing
Transformer baseline) or a sub-coverage block (actively hurts ASR) —
and either way matches the closed dense-freedom engaged-null failure
class.

**Transferability expectation (Mamba-2 / Linear Attention / Lion).**
- **Mamba-2**: paper's native target, but at the long-context regime
  (4 K → 64 M). At our short-context ASR regime, same degeneracy. LOW.
- **Linear Attention**: as a hybrid sparse-attention auxiliary block
  around the baseline, yes — but not a kernel-map modification.
  Same regime mismatch. LOW at our $T$.
- **Lion (bidirectional)**: stick-breaking recency bias is inherently
  causal (ordinal nearest-to-farthest). For bidirectional use there is
  no natural "most recent" direction; the paper does not define the
  extension. The chunk encoder in RAMba is bidirectional, but the
  chunk-aggregation layer HSA is strictly causal by construction.
  Structural asymmetry; math doesn't break but requires non-trivial
  redesign the paper doesn't provide.

**Expected gain vs. overhead.** N/A at our regime. Not a candidate
for baseline probe at $T \le 500$.

**Comparison with RWKV-X (Paper 4).** HSA has a stronger mechanism
contribution (end-to-end learnable chunk selection via chunk-aware
relevance scoring, vs RWKV-X's engineering assembly of existing parts).
But the regime constraints are identical — both fail at $T \le 500$
by the same chunk-degeneracy arithmetic. The two papers occupy the
same "long-context hybrid retrieval" class and share the same natural
home (long-form audio / streaming ASR, should thesis scope expand).

**Natural home (if thesis scope expands).** Long-form audio
(hour-scale utterances, streaming decoding at high sample rate).
Scope expansion, not a mechanism advance on the current thesis line.

**Useful data point extracted.** The stick-breaking convex-combination
mixer family is now evidenced at PLATEAU band twice — once in this
paper's chunk-weight form (not yet tested in our setting but predicted
by structural analogy) and once in `rwkv6_p2rse_softmax` (directly
tested). If any future proposal relies on stick-breaking / softmax
mixers over ordered elements as the load-bearing mechanism, this is
prior-art evidence for deprioritisation.

### Paper 6 — CRT / NCGRU (arXiv:2505.00929)

Link: https://arxiv.org/abs/2505.00929

**Paper concept.** Segment-chunked hybrid Transformer + RNN for
edge-computing / low-SWaP deployments processing sequences in fixed-size
segments. Four ingredients:
1. **Single-vector persistent memory.** $m_{k-1}$ carried across segments
   via a GRU/NCGRU update $m_k = \mathrm{GRU}(Y_k, m_{k-1})$, where $Y_k$
   is the segment-$k$ Transformer output. BPTT through $m_k$.
2. **Memory token.** $m_{k-1}$ concatenated as an extra token at segment
   $k$'s Transformer input — participates in local self-attention.
3. **RNN-based positional encoding.** A second GRU/NCGRU runs over input
   embeddings to produce learned PE, replacing sinusoidal / RoPE.
4. **NCGRU (Norm-Constrained GRU).** GRU variant with Cayley-orthogonal
   recurrent matrix (unit singular values) to stabilize BPTT gradients.

Paper's own ablation attributes the bulk of CRT's gain to RNN-based
positional encoding; memory token alone helps modestly; NCGRU helps
only modestly over plain GRU. **The components that transfer to RWKV
are not the load-bearing pieces in the source paper.**

**Can this reformulate to plain RWKV-6? PARTIAL — one sub-mechanism is
a viable transition-family candidate, low priority.**

Per-ingredient breakdown against $T \le 500$ / CTC-ASR / train-from-
scratch / 30-ep:

| CRT ingredient | Applicable at our regime? |
|---|---|
| Segment-chunked memory token + BPTT | **N/A.** We process full utterances ($T \le 500$) in one pass — no segmentation, no across-segment memory needed. |
| Memory token in local self-attention | **N/A.** RWKV-6 has no local self-attention within a segment; the memory token requires a local-attention host. |
| RNN-based positional encoding | **Orthogonal-axis.** RWKV-6 encodes causal order via token-shift / `time_maa_*` LoRA; positional encoding is not a diagnosed weakness in Stages 2-9. Paper's main ablation gain — but transfers to the Transformer baseline slot only, which is not the thesis line. |
| **NCGRU / Cayley-orthogonal recurrence** | **Yes, as a transition-family extension.** The one porting-worthy sub-mechanism. |

**The viable sub-mechanism: NCGRU Cayley-orthogonal on the RWKV-6
transition.**

Placed on the operator ladder:
$$(\mathbb{R}_+)^K \;\subsetneq\; SO(2)^{K/2} \times (\mathbb{R}_+)^{K/2} \;\subsetneq\; SO(K) \cdot \mathbb{R}_+$$
(vanilla RWKV-6 ⊂ RSE ⊂ Cayley-orthogonal). Strictly richer than RSE
in cross-channel direction — replaces block-diagonal
$2 \times 2$ rotations with full $K \times K$ orthogonal transitions.

**Core mathematical idea (NCGRU-Cayley on RWKV-6 baseline).**

Replace the diagonal decay in the WKV scan with
$$G_{t,h} = e^{-\lambda_{t,h}} \cdot O_{t,h}, \qquad O_{t,h} = (I - A_{t,h})(I + A_{t,h})^{-1},$$
with $A_{t,h}$ per-head skew-symmetric, low-rank parametrised:
$$A_{t,h} = U_{t,h} V_{t,h}^\top - V_{t,h} U_{t,h}^\top, \quad U, V \in \mathbb{R}^{K \times r}.$$
State update:
$$S_t = e^{-\lambda_t}\,O_t\,S_{t-1} + k_t v_t^\top.$$

Zero-regression contract: $U = V = 0$ LoRA zero-init → $A = 0$ →
$O = I$ → vanilla RWKV-6 recovered bit-exactly.

**RWKV-6 adaptation plan (baseline only).**
- Backbone name candidate: `rwkv6_orthogonal` or `rwkv6_ncgru_cayley`.
- Flag `use_cayley_orthogonal: bool = False`. Location: scan path in
  `src/models/rwkv6_time_mix.py`.
- **Parameter-parity concern (major).** Low-rank $r=4$ gives
  $d \cdot K \cdot r = 256 \cdot 64 \cdot 4 \approx 65\,\mathrm{K}$ per
  $U$-projection per head per layer; × 2 ($U$ and $V$) × 4 heads × 6
  layers $\approx$ **3 M extra params**. Exceeds the 5 % parity rule
  (7 M base → ≤ 350 K headroom). Would need rank 1 or a simpler
  Householder-style parametrisation (single per-token reflection
  vector) to fit parity.
- Wall-clock: matrix inverse $(I+A)^{-1}$ is $O(K^3)$ per token per
  head; Taylor-truncated matrix exponential is $O(K^2 \cdot \text{truncation})$.
  Estimated 3-5× anchor, comparable to Stage-8 T2's 2.9× ratio.
- Implementation: pure PyTorch prototype feasible; torch.compile or a
  Householder-based alternative preferred over explicit inverse at
  training scale. No custom CUDA strictly required.
- Zero-regression contract: exact via zero-init $U, V$.

**Transferability expectation (Mamba-2 / Linear Attention / Lion).**
- **Mamba-2**: MODERATE but engineering-heavy. Mamba-2's selective
  $A$ matrix is already diagonal — replacing with Cayley-orthogonal
  changes the scan kernel substantially and loses Mamba-2's efficient
  chunked-scan primitive. Architecturally invasive.
- **Linear Attention (recurrent form)**: same port as RWKV-6 causal.
- **Lion (bidirectional)**: **math clean, parallel form breaks.**
  LION's parallel form $A_\text{fwd} = \text{tril}((r e^{cs})(k e^{-cs})^\top)$
  relies on $e^{cs}$ being a commutative scalar cumulative sum.
  Non-commutative orthogonal products $\prod_\tau O_\tau$ do not admit
  this cs-shift factorisation — you'd need explicit sequential matrix
  products at $O(T^2 K^2)$ per head. Hard performance hit specific to
  LION's fast path.

**Expected gain vs. overhead.**
- Upper tail: AMBIGUOUS-ENGAGED (predicted). Dense per-token
  deployment of Cayley-orthogonal matches the deployment shape of
  the five closed engaged-nulls (A1′, T1, T2, S9A, S9B — all SGD-
  engaged without converting). The operator family is novel
  (full $SO(K)$ vs block-diagonal $SO(2)^{K/2}$) but the deployment
  *shape* is not — which is what the cross-experiment invariant
  cares about.
- Lower tail: REGRESSION risk from parameter-parity break at low
  rank.
- Static (token-independent) version: **absorbable** per Stage-2
  lesson — any fixed global basis rotation is absorbed by existing
  $W_r, W_k$ projections. No new expressivity.
- Middle ground (per-head fixed $O_h$ per layer, token-independent):
  marginally informative but breaks the clean "static vs dense"
  binary that Stage-4 depth-graded RSE exploited.

**Diagnostic value (the one real reason to consider running this).**

NCGRU-Cayley dense per-token offers a **head-to-head tiebreaker with
Stage-8 T2 polar non-normal**. Same deployment shape (dense per-token,
data-dependent); different operator family (full $SO(K)$ orthogonal
vs $2 \times 2$ polar non-normal). Two outcomes, both informative:
- If both land at ~0.1200: corroborates the cross-experiment invariant
  at the **operator-family level** — "dense per-token freedom on top
  of RSE+viscosity doesn't convert, regardless of which operator
  family extends."
- If Cayley-orthogonal significantly beats T2: reveals **which
  operator subspace** (cross-channel orthogonal vs within-block
  non-normal) the RSE ceiling actually cares about.

This is a diagnostic question about the invariant's mechanism, not a
ceiling-break attempt — useful as a tiebreaker only.

**Recommendation.** Not a priority candidate for the baseline review
queue. If pursued at all, frame it as a diagnostic tiebreaker with
Stage-8 T2 rather than a ceiling-break probe; require a rank-1 /
Householder-style parametrisation to hold parameter parity.

**Alternative adaptation (not championed, parked for completeness).**
Reviewer B's segment-memory conditioning: persistent vector $m_s$
updated via GRU at segment boundaries, conditioning RWKV projections
$r, k, v, g, w$ via zero-init affine terms $A_* m_{s-1}$.
Mathematically coherent multi-scale memory adaptation with a clean
zero-regression contract, but (a) doesn't actually need CRT as its
intellectual basis — it is a generic slow-timescale memory idea that
stands on its own, (b) CRT's own ablation attributes the main gain to
RNN-PE rather than the memory token, so the transferred component is
not load-bearing in the source paper, (c) expected gain is
low-to-moderate at best. Parked, not recommended over NCGRU-Cayley as
a diagnostic.

**Out-of-regime components (not in the baseline plan).**
- Segment-chunked memory token + BPTT. N/A, we are single-pass at
  $T \le 500$.
- Memory token in local self-attention. N/A, RWKV-6 has no local
  attention.
- RNN-based positional encoding. Orthogonal axis; possibly applicable
  to the Transformer baseline but not the thesis line.

**Natural home (if thesis scope expands).** Streaming / segmented ASR
at edge-compute targets where local-attention segments replace global
recurrence. Same scope-expansion cell as Papers 4 and 5; different
motivation (edge compute vs long-context retrieval).

### Paper 7 — Non-Attention LLM (arXiv:2506.01963)

Link: https://arxiv.org/abs/2506.01963 · repo: https://github.com/andrew-jeremy/nonAttentionLLM

**Paper concept.** Compositional architecture, not a mechanism paper.
Assembles four existing ingredients:
1. **S4 state-space block** — intra-chunk mixing. (Paper's actual
   implementation relaxes S4 to a simplified depthwise convolution,
   per the linked repo.)
2. **Multi-resolution dilated convolutions** — parallel 1D dilated
   DWConvs with varying dilations (e.g., $\{1,2,4\}$) capture local
   patterns of varying lengths.
3. **FAISS external retrieval** — chunk-level top-$k$ retrieval of
   pooled chunk embeddings.
4. **GRU recurrent supervisor** — cross-chunk hidden state carries
   compressed global context.

Novelty claim: "absence of attention" at 1M+ token LM contexts. **No
new mathematical mechanism, theorem, or parametrisation.** Each
ingredient traces to prior work (S4 → Mamba, dilated convs → WaveNet /
Hyena, FAISS → RETRO, cross-chunk supervisor → Transformer-XL).

**Paper-quality caveat.** The repo landing page exposes only README,
requirements, LICENSE (as of 2026-04-21) — reproducibility looks weak.
The paper's "S4" is explicitly a simplified DWConv, not rigorous S4.
Treat the headline numbers with corresponding caution.

**Can this reformulate to plain RWKV-6? PARTIAL — one sub-mechanism is
a viable, low-risk refinement of our already-validated input-side
mechanism.**

Per-ingredient breakdown:

| Non-Attention ingredient | Fit at our regime |
|---|---|
| S4 block | **Redundant.** Mamba-2 (our separate baseline track) is a strict superset (data-dependent selective $A_t$). Porting S4 into RWKV-6 would regress the capability ladder. |
| **Multi-resolution dilated convolutions** | **Portable as ConvShift extension.** Hooks into our only validated input-side win (`rwkv6_rse_convshift`, 0.1145/0.1126). |
| FAISS external retrieval | **Out-of-regime.** $T \le 500$ single-pass has no "outside context window" content to retrieve. Same regime mismatch as Papers 4/5/6. |
| GRU recurrent supervisor | **Out-of-regime.** Cross-chunk state carry targets segmentation we don't have. |

**The viable sub-mechanism: multi-dilation ConvShift.**

**Empirical hook.** Stage-2 `rwkv6_convshift_trap` at dev 0.1150 /
test 0.1150 and Stage-3 `rwkv6_rse_convshift` at dev 0.1145 / test
**0.1126** — the absolute best row on our causal RWKV spine. The
ConvShift mechanism (DWConv1d kernel=3 replacing fixed token shift)
is our one validated input-side / feature-mixing win and lives on an
orthogonal axis from the RSE→T2→S9 transition-family line. Extending
its receptive field via multiple dilations is the natural refinement
of an already-working mechanism.

**Why this clears the Stage-2-to-Stage-9 filters (where Papers 4-6
don't).**

| Failure-class criterion | Does multi-dilation ConvShift trigger it? |
|---|---|
| Dense per-(token, head, block) freedom on transition operator | **No.** Static per-layer mixer over fixed dilation branches, input-side not transition-side. |
| Absorbable by existing projections (Stage-2 lesson) | **No.** Dilated convs are not absorbable by $W_r, W_k$ — they are explicit convolutional receptive-field expansions. |
| Parameter-parity break | **No.** ~18 K extra params = 0.25 % of 7 M encoder. |
| Overlap with already-falsified mechanism | **No.** Multi-Rate RSE (`rse_m2`, `rse_m4`, PLATEAU) was $M$ parallel *transition-side* scans over the same operator family. Multi-dilation ConvShift is input-side, different axis entirely. |
| Zero-regression-at-init contract | **Yes.** $\alpha_1 = 1, \alpha_{2,4,8} = 0$ at init recovers current ConvShift bit-exactly. |

Static per-layer mechanisms have been systematically successful
(`rse_depth`, `rse_strong` — both ceiling breaks) or stably-tied
(`rmsnorm`, `hadamard_n2`) in our chain. The closed-AMBIGUOUS-ENGAGED
class is uniformly dense per-(token, head, block); multi-dilation
ConvShift's deployment shape is fundamentally different.

**Core mathematical idea.**

Replace the single DWConv1d(kernel=3, dilation=1) in
`src/models/mechanisms/conv_shift.py` with a weighted sum of parallel
dilated branches:
$$x_\text{mixed} = \sum_{d \in \{1, 2, 4, 8\}} \alpha_d \cdot \mathrm{DWConv1d}_{d}\big(x;\,\mathrm{kernel}=3\big).$$

Initialisation: $\alpha_1 = 1$, $\alpha_{2,4,8} = 0$ → matches current
single-dilation ConvShift bit-exactly. $\alpha_d \in \mathbb{R}$
unconstrained, per-layer learnable (24 scalars total encoder-wide, or
per-channel for $6 \times 256 \times 4 \approx 6.1$ K).

Per-layer receptive field at $d = 8$: $\pm 8$ frames = **170 ms** at
100 fps post-subsampling = syllable-envelope scale. Matches the ASR
acoustic hierarchy (phoneme ~50 ms → syllable ~250 ms → word ~500 ms).
Same physical argument as Log-Linear Attention (Paper 1), applied on a
different structural axis (input-side mixing vs semiseparable mask).

Causal padding (left-pad $2d$) in `mode="recurrent"`; symmetric
padding in `mode="lion"` — preserves causality / bidirectionality
without architectural changes, because the existing ConvShift already
handles this distinction.

**RWKV-6 adaptation plan (baseline only).**
- Backbone name: `rwkv6_convshift_multidil`. Flag
  `use_conv_shift_multidilation: bool = False` alongside existing
  `conv_shift`. Location: `src/models/mechanisms/conv_shift.py`.
- Parameters:
  - Scalar per-dilation per-layer: $4 \times 6 = 24$ params.
  - Four parallel DWConv weights: $4 \times 768 = 3.07$ K per layer,
    $\times 6 = 18.4$ K encoder-wide (0.25 % of 7 M).
  - One dilation branch (d=1) reuses the existing ConvShift weights;
    three new branches get fresh DWConv kernels.
- Compute: 4× DWConv cost per layer. At $T=500$, $K=256$: DWConv is
  $O(TKk) = 0.4$ M per layer; 4× = 1.6 M per layer. Negligible vs
  WKV scan ($O(THK^2) \approx 8$ M per layer). Wall-clock overhead
  $\ll 5 \%$.
- Linear-time preserved: DWConv is $O(TK)$ regardless of dilation.
- Pure PyTorch; no CUDA required.
- Zero-regression contract: exact via $\alpha_1 = 1, \alpha_{2,4,8} = 0$.

**Transferability expectation (Mamba-2 / Linear Attention / Lion).**
- **Mamba-2**: HIGH. Mamba-2 block recipe includes a short DWConv
  (kernel=4 typically). Direct port: replace with multi-dilation
  variant.
- **Linear Attention**: HIGH. Pre-layer position-pointwise convs
  slot into any baseline with a local-mixing front-end. Architecture-
  agnostic.
- **Lion (bidirectional)**: HIGH. DWConv is direction-agnostic;
  existing ConvShift already uses symmetric padding for
  `mode="lion"` — multi-dilation extension inherits this without
  changes.

Architecture-agnostic across the board. No parallel-form breakdown
anywhere.

**Expected gain vs. overhead.**
- Upper tail: MARGINAL on the absolute-best `rwkv6_rse_convshift`
  row. Extends a validated mechanism along a physically-motivated
  scale axis.
- Lower tail: NULL. Zero-regression contract; if added context is
  redundant with WKV state decay, SGD leaves $\alpha_{2,4,8}$ near 0
  and the mechanism stays engaged at its single-dilation baseline.
- Diagnostic value regardless: learned $\alpha_d$ per layer gives a
  direct read on *at which scale does ConvShift want to operate?*.
  If shallow layers prefer $\alpha_1$ and deep layers grow $\alpha_8$,
  that matches the depth-graded pattern observed in Stage-2 `gen2`
  and Stage-4 `rse_depth` — a third independent data point for the
  multi-scale depth hierarchy thread, applied to the input-side axis
  this time.
- Tier-3 cheap probe. Sits alongside Paper 3's partial-embedding
  bypass (also Tier-3). Lower-risk than Paper 1's Log-Linear
  (transition-side, larger overhead, different axis).

**Rejected components (not in the baseline plan).**
- **S4 block** — redundant with Mamba-2 track; porting would regress
  the capability ladder.
- **FAISS external retrieval** — out-of-regime at $T \le 500$
  single-pass; same regime mismatch as Papers 4, 5, 6.
- **GRU recurrent supervisor** — out-of-regime; our pipeline is
  single-pass, no cross-chunk state carry needed.

**Reviewer B's retrieval-memory adaptation (not championed).** B
proposed focusing on chunk-summary retrieval memory rather than the
dilated convs. Coherent adaptation but sits in the same out-of-regime
cell as the HSA and RWKV-X sparse-retrieval mechanisms we have
already rejected for $T \le 500$. Parked.

**Natural home for the full paper (if thesis scope expands).** 1M+
token LM contexts. Not the thesis target. But the multi-dilation
ConvShift extension stands on its own merits regardless of the paper's
full architecture — it is effectively a natural continuation of our
own Stage-2 ConvShift line.

### Paper 8 — PoM: Polynomial Mixer (Picard et al., arXiv:2604.06129)

Link: https://arxiv.org/abs/2604.06129 · repo: https://github.com/davidpicard/pom

**Paper concept.** Linear-time, attention-free, permutation-equivariant
token mixer claiming drop-in MHA replacement. Three operations per block:
1. **Per-token polynomial feature lift.**
   $\phi_p(x_t) = \alpha_p \odot h(W_h x_t)^{\odot p}$ for $p = 1, \ldots, k$,
   with $h$ a pointwise activation, $W_h: d \to D$ (typically $D = 2d$),
   $\alpha_p \in \mathbb{R}^D$ learnable per-order coefficients,
   exponentiation element-wise.
2. **Sequence-aggregated vector state.**
   $H(X) = \sum_{t=1}^n \sum_{p=1}^k \alpha_p \odot h(W_h x_t)^{\odot p} \in \mathbb{R}^D$,
   $O(nD)$ compute.
3. **Per-token gated readout.**
   $y_t = W_o[\sigma(W_s x_t) \odot H(X)]$.

Causal variant: $H_t = H_{t-1} + \sum_p \alpha_p \odot h(W_h x_t)^{\odot p}$.

Theoretical claim: Thm 3.2 — universal sequence-to-sequence approximator
at sufficiently large $k$ (non-constructive existence). Permutation-
equivariant by construction.

**Paper's own empirical check (Table 1a, 125 M GPT-2 on 15 B FineWeb).**
- MHA: val loss 3.29, HellaSwag 33.3.
- **Pure PoM**: val loss **3.88**, HellaSwag 27.4 — substantially worse.
- Hybrid (alt PoM + local-128 attention): val loss 3.31, HellaSwag 33.8 —
  matches MHA only with local attention injected to restore recency.

Pure PoM underperforms MHA; the hybrid works because local attention
re-introduces the recency bias PoM lacks. The universal-approximator
theorem is a non-constructive existence result at high $k$, not a
trainability guarantee at $k = 2$ (the paper's default).

**Can this reformulate to plain RWKV-6? PARTIAL — polynomial lift is
portable, but at our parameter budget sits inside the same quadratic-
form family as our already-tested feature-side line.**

Per-component breakdown:

| PoM component | Fit | Comparison to RWKV-6 existing |
|---|---|---|
| Per-token polynomial feature lift | **Portable** as a value-side modification inside WKV | Structurally closest to `rwkv6_hadamard_n2` (PLATEAU 0.1253/0.1251). PoM at $k=2$ is in the same element-wise quadratic-form function class. |
| Sequence-aggregated vector state $H$ (no decay) | **Duplicative of WKV state; missing recency.** | RWKV-6 already has accumulating state with superior structure (matrix-valued per-head, data-dependent decay). PoM's vector-sum state has no decay — misaligned with ASR temporal locality. Port via value-lift into WKV scan *inherits* RWKV's decay, fixing the mismatch. |
| $\sigma(W_s x_t)$ gated readout | Already present | RWKV-6 has sigmoid-gated receptance; this is not a novel ingredient to port. |

**Structural reality check against the already-tested feature-side line.**

At $k = 2$ (what fits parameter budget), PoM's features are
$(W_h x)_j^2 = \sum_{i_1, i_2} W_{h,j i_1} W_{h,j i_2} x_{i_1} x_{i_2}$
— specific rank-1 bilinear projections of the Kronecker space. This is
a **differently-parameterised member of the same quadratic-form function
class** that the following already-tested mechanisms cover:

| Mechanism | Feature structure | Cross-channel $k_i k_j$ ($i \ne j$)? | Result |
|---|---|---|---|
| `rwkv6_hadamard_n2` | Element-wise $k \odot k$ on linear $k$ | No (diagonal only) | PLATEAU 0.1253 / 0.1251 |
| `rwkv6_qtail_lowrank_all` | Low-rank Kronecker $U_r^\top (r) \otimes U_k^\top(k)$ | Yes, via independent $U_r, U_k$ | **MARGINAL 0.1238 / 0.1240** (best feature-side) |
| **PoM** at $k=2$ | Specific rank-1 bilinear via $W_h$ shared projection | Via $W_h$-mixed features, then element-wise squared | Predicted PLATEAU |

At $k \ge 3$ PoM is strictly richer (higher-order polynomials beyond
bilinear), but parameter budget at $k \ge 3$ with useful $D$ breaks
parity (see below). So the parity-fitting regime ($k=2$, thin $D$)
leaves PoM in the quadratic family that's already been probed.

**Core mathematical idea (baseline, cleanest port).**

Additive polynomial lift on $v$ within the existing WKV scan (reuses
decay, clean zero-regression):

$$\hat v_t = v_t + \sum_{p=2}^{k} \gamma_p \odot h(W_h x_t)^{\odot p}, \qquad \gamma_p = 0 \text{ at init}$$
$$S_t = \text{diag}(w_t)\,S_{t-1} + k_t\,\hat v_t^\top, \qquad
  y_t = r_t^\top S_t + u \odot (r_t \odot k_t)\,v_t$$

At init: $\gamma_p = 0 \Rightarrow \hat v = v$ bit-exactly. SGD grows
$\gamma_{2,\ldots,k}$ to activate higher-order features. Inherits
RWKV's data-dependent decay (addresses PoM's missing recency bias for
ASR — paper's own data shows this bias matters).

**RWKV-6 adaptation plan (baseline only).**
- Backbone: `rwkv6_pom_vlift`. Flag
  `use_pom_vlift: bool = False`, `pom_order: int = 2`, `pom_expansion: int = 64`.
- Parameter cost (thin config, $D=64$, $k=2$):
  - $W_h \in \mathbb{R}^{d \times D}$: $256 \times 64 = 16\,\mathrm{K}$ per layer.
  - $\gamma_p$ and $\alpha$-equivalent scalars: negligible.
  - Total: $\sim 96\,\mathrm{K}$ encoder-wide (1.4 % of 7 M). **Parameter-safe.**
- Parameter cost (full config, $D = 2d = 512$, $k = 2$):
  - $W_h$ alone: $256 \times 512 = 131\,\mathrm{K}$ per layer × 6 $\approx$ **786 K (11 %, breaks parity)**.
  - Add $W_s, W_o, \alpha$ if porting the full PoM readout: ~2.4 M, ~34 %.
- Compute (thin config): element-wise powers $+$ multiplications, $O(TDk)$
  per layer. Negligible vs WKV scan's $O(THK^2)$. Wall-clock overhead $\ll 5\%$.
- Linear-time preserved.
- Zero-regression contract: exact via $\gamma_p = 0$ init.
- Pure PyTorch; no CUDA.

**Transferability expectation (Mamba-2 / Linear Attention / Lion).**
- **Mamba-2**: MODERATE. Same structural story — value-side polynomial
  lift alongside selective $A_t$. Element-wise polynomial doesn't add
  cross-channel expressivity beyond what qtail-family already probed.
- **Linear Attention (recurrent)**: MODERATE. Clean auxiliary mixer;
  not a kernel-map replacement.
- **Lion (bidirectional)**: PoM's **natural home.** Sequence-aggregated
  state $H(X) = \sum_t$ is permutation-equivariant by construction,
  matching LION's bidirectional structure. Paper's strongest target
  regimes (3D segmentation, image generation, Earth observation) are
  all bidirectional for exactly this reason. For LION, a **standalone
  `lion_pom`** backbone (pure PoM block as additive branch, with
  positional encoding to break set-equivariance to sequence-equivariance)
  is the structurally-principled port, not a value-side lift inside
  someone else's scan.

**Expected gain vs. overhead.**
- Upper tail: PLATEAU. Same quadratic-form family as already-tested
  hadamard_n2. Pure-PoM underperforming MHA in the paper's own 125 M
  data is independent corroboration that the lift doesn't unlock
  anything the feature-side line hasn't already banked.
- Lower tail: NULL via zero-regression contract.
- At full $D = 2d$, breaks parameter parity. At thin $D$, parameter-safe
  but function class is strictly the already-probed quadratic family.
- Main value: **theoretical foil for the thesis discussion**
  (see below), not ceiling-break candidate.

**Theoretical-discussion value (primary).** PoM is a **linear-time
universal-approximator foil** for the cross-experiment invariant.
Thm 3.2 guarantees sequence-to-sequence approximation at large $k$; at
$k = 2$ pure PoM underperforms MHA by 0.59 val loss at 125 M despite
the formal universal-approximation property. This is the
"expressive-in-principle, sub-budget-in-practice" pattern the
stages-2-9 engaged-null invariant has documented from the
parameter-mobility side; PoM corroborates from the
theoretical-expressivity side. Worth citing in the thesis discussion as
**independent evidence that expressivity theorems do not imply
trainability at fixed budget** — and that the stages-2-9 invariant is a
general phenomenon, not an RWKV-specific artefact.

**Alternative adaptation (B's, parked).** Parallel per-head
polynomial-moment state alongside WKV:
$m_t = \rho_t \odot m_{t-1} + \sum_p \alpha_p \odot h(W_h x_t)^p$,
readout $y_t^{\mathrm{PoM}} = W_o[\sigma(W_s x_t) \odot m_t]$,
fused via zero-init $\beta \cdot y_t^{\mathrm{PoM}}$. Adds decay $\rho_t$,
which is a good instinct, but duplicates state machinery compared to
the value-lift form. Parked as fallback if the value-lift form doesn't
engage SGD cleanly.

**Natural home.** LION bidirectional as a standalone `lion_pom` backbone
— this is the structurally-principled target for PoM's permutation-
equivariance. The causal RWKV-6 port via value-lift is parameter-safe
but diagnostically weak (lands in already-explored quadratic territory).

### Paper 9 — M²RNN: Matrix-to-Matrix Recurrent Neural Network (Mishra et al., arXiv:2603.14360)

Link: https://arxiv.org/abs/2603.14360 · training code:
https://github.com/open-lm-engine/lm-engine · kernels:
https://github.com/open-lm-engine/accelerated-model-architectures

**Paper concept.** Non-linear matrix-state recurrence that steps outside
the $TC^0$ complexity class shared by Transformers and all linear RNNs
(Mamba, RWKV, Gated DeltaNet, linear attention). Core:
$$Z_t = \tanh\!\big(H_{t-1} W + k_t v_t^\top\big),\qquad
  H_t = f_t\,H_{t-1} + (1 - f_t)\,Z_t,\qquad
  y_t = H_t^\top q_t + w_r \odot v_t,$$
with $H_t \in \mathbb{R}^{K \times V}$, input-independent $W$
identity-init, $q_t,k_t \in \mathbb{R}^K$, $v_t \in \mathbb{R}^V$,
scalar per-head forget gate $f_t = 1 / (1 + e^{x_t + \beta_n})^{\alpha_n}$
with $\alpha_n, \beta_n$ learnable per head.

Two defining properties:
1. **Non-linear in the state.** The tanh is applied to the full matrix
   state, not just input projections. Breaks associativity → no parallel
   scan. Admits exact state-tracking on $S_k$ permutation groups —
   provably **$NC^1$**, beyond what Transformers or any linear RNN with
   diagonal / input-independent / semiseparable transition can express
   (Merrill et al. 2024, Grazzi et al. 2024).
2. **Outer-product state expansion.** Same $K \times V$ matrix state as
   linear attention / RWKV / Mamba-2. Paper's core empirical claim:
   the historical underperformance of LSTM/GRU vs linear attention is
   primarily about state size, not about non-linearity — combine
   non-linearity with outer-product state and LM competitiveness is
   recovered *plus* $NC^1$ expressivity is unlocked.

**Empirical anchor (paper).** Perfect generalisation on $S_3$ at
2-4× training length (Fig. 3); **+0.4-0.5 PPL at 7 B MoE** from
replacing one Gated DeltaNet layer with an M²RNN layer; **+8 LongBench
points** from one-layer swap. Paper's recommendation: **sparing use**
(1-2 M²RNN layers in an otherwise-linear hybrid) because all-layer
deployment is too slow without custom Triton kernels.

**Can this reformulate to plain RWKV-6? YES — and this is the most
thesis-relevant paper in the review queue so far.**

**Why this matters structurally for the thesis.** Every
Stage-2-through-9 mechanism we have tested stayed *linear in the state*:

| Stage / source | Transition operator | Linear in state? | Complexity class |
|---|---|---|---|
| 2 | $\text{diag}(\exp(w_t))$ | Yes | $TC^0$ |
| 3/4/5 RSE | $e^{-\lambda} R(\theta_t)$ block-diagonal | Yes | $TC^0$ |
| 8 T2 polar non-normal | $e^{-\lambda} R(\psi)^\top D(\rho) R(\psi) R(\theta)$ | Yes | $TC^0$ |
| 8 T1 delta | $(I - \beta k k^\top) \cdot \text{diag}$ | Yes (rank-1 erase is still linear in state) | $TC^0$ |
| 9 A/B sparse | gated sparse non-normal | Yes | $TC^0$ |
| **M²RNN** | $\tanh(H_{t-1} W + k v^\top)$ | **No** | **$NC^1$** |

All five closed engaged-nulls (A1′/T1/T2/S9A/S9B) were dense per-token
freedom **on linear-in-state operators**. The cross-experiment invariant
was explicitly scoped to linear extensions. **Whether the invariant
extends across the linearity boundary is the open question M²RNN
answers directly.** No other paper in the review queue tests this axis.

**Core mathematical idea (championed form: parallel-branch with clean
zero-regression).**

The paper's native substitution form has no clean zero-regression-at-init
contract (tanh saturation at $W=I, f_t=1$ is a dead layer, not vanilla).
The parallel-branch synthesis solves this by running M²RNN as a
separate per-head matrix state alongside the existing WKV state, fused
via zero-init scalar $\lambda_h$:

At layer $L^*$, per head:
$$S^\text{rwkv}_{t,h} = \text{diag}(w_t)\,S^\text{rwkv}_{t-1,h} + k_{t,h} v_{t,h}^\top\qquad\text{(vanilla WKV branch, unchanged)}$$
$$Z^{m2}_{t,h} = \tanh\!\big(S^{m2}_{t-1,h}\,W_h + k_{t,h} v_{t,h}^\top\big)\qquad\text{(non-linear candidate)}$$
$$S^{m2}_{t,h} = f_{t,h}\,S^{m2}_{t-1,h} + (1 - f_{t,h})\,Z^{m2}_{t,h}\qquad\text{(forget-gated blend)}$$
$$y_{t,h} = r_{t,h}^\top\,\big(S^\text{rwkv}_{t,h} + \lambda_h\cdot S^{m2}_{t,h}\big) + \text{bonus}_t$$

Init: $\lambda_h = 0$, $W_h = I$. At init: $y_t = r_t^\top S^\text{rwkv}_t + \text{bonus}$ = vanilla RWKV-6 **bit-exact**. SGD grows $\lambda_h$ only if the M²RNN branch is productive.

Forget-gate parameterisation (paper's form): $f_{t,h} = \psi(W_f x_t)_h$
with $\psi(z) = 1 / (1 + e^{z + \beta_h})^{\alpha_h}$ and $\alpha_h,
\beta_h$ learnable per head (clamped positive).

**RWKV-6 adaptation plan (baseline only).**
- Backbone: `rwkv6_m2rnn_sparse` — **sparing-use, replace the scan at
  one selected layer $L^*$** (paper suggests deepest; we can use
  $L^* = 5$ for a 6-layer stack and tune).
- Flags: `use_m2rnn: bool = False`, `m2rnn_layer: int = 5`.
- Parameter cost (sparing-use, 1 of 6 layers):
  - $W_h \in \mathbb{R}^{K \times K}$ per head: $K^2 \cdot H = 4096 \cdot 4 = 16$ K per layer.
  - $\alpha_h, \beta_h$ and $W_f: d \to H$: ~1 K.
  - $\lambda_h$: $H = 4$ scalars.
  - **Total ~17 K encoder-wide (0.25 % of 7 M).** Well within parity.
- All-layer version: ~100 K (1.4 %). Inside parity.
- Compute cost — **the binding constraint.**
  - Non-linear scan is strictly sequential per token per head.
  - Per-token matmul $SW$ is $O(K^3) \approx 260$ K ops per head.
  - Sequence cost at $T=500, H=4$ on one M²RNN layer:
    $\sim 0.5$ B ops per forward pass per layer. vs chunked WKV
    $\sim 8$ M ops at the same layer $\Rightarrow$ **$\sim 60\times$**
    the anchor cost at that one layer.
  - Sparing-use (1 of 6 layers) estimated wall-clock in pure PyTorch
    + `torch.compile`: 3-8× anchor for the full 6-layer stack. Pushes
    30-epoch clean-100 from hours to ~half-day scale. Tractable.
  - All-layer pure PyTorch: 30-80× anchor — **not tractable** without
    Triton kernel.
- Zero-regression contract: **exact** via $\lambda_h = 0$ init.
- Reference kernels available from the paper's repo for scale-up; the
  Stage-1 prototype can avoid them via sparing-use.

**Transferability expectation (Mamba-2 / Linear Attention / Lion).**
- **Mamba-2**: HIGH. Paper explicitly tests Hybrid Mamba-2 + M²RNN
  (single-layer swap). Reference code / kernels available. Direct port.
- **Linear Attention (recurrent)**: HIGH. Same structural port —
  replace one linear recurrent layer with M²RNN.
- **Lion (bidirectional)**: **MATH BREAKS at the parallel-form level.**
  LION's speedup comes from $A_\text{fwd} = \text{tril}((r e^{cs})(k e^{-cs})^\top)$
  where $e^{cs}$ is a *commutative* scalar cumulative sum. M²RNN's
  tanh state update admits no such decomposition — state at time $t$
  cannot be computed without sequentially computing every intermediate
  state. Forcing M²RNN into LION collapses LION into a slow
  sequential-causal model with ambiguous bidirectional semantics
  (tanh destroys the future-state synthesis that vanilla BiWKV already
  struggles with — CLAUDE.md Pitfall #7 flags this for the linear
  case; non-linear makes it structurally unfixable). **LOW-to-zero
  transferability.**

  **Thesis-level observation.** Non-linearity in the state transition
  is fundamentally incompatible with parallel-form bidirectional
  execution. One can have either (a) $NC^1$ expressivity via non-linear
  state with sequential execution only, or (b) $TC^0$ expressivity
  with parallel-form bidirectional execution (LION, Mamba-2 parallel),
  **but not both simultaneously at the same layer.** Worth citing in
  the thesis discussion as a theoretical tradeoff structural to the
  sequence-model design space, not a limitation of this specific paper.

**Expected gain vs. overhead.**

Unusually wide spread — this is the only mechanism in the queue whose
result is genuinely unpredictable from our existing evidence base
(which is itself the diagnostic point). Prior-weighted rough split:
- **BREAK** (dev $< 0.1170$): ~20 %. If the $TC^0 \to NC^1$ gap fires
  at our regime, first genuine ceiling break since
  `rwkv6_rse_strong_viscosity`. Paper's 7 B-scale anchor (+0.4-0.5 PPL
  from one-layer swap) suggests benefit extends beyond pure
  state-tracking to general LM competence.
- **MARGINAL** (dev $\sim 0.1170$-$0.1185$): ~40-50 %. Non-linearity
  helps effective state capacity, gain comparable to RSE-strong-
  viscosity but doesn't compound with it.
- **PLATEAU** (dev $\sim 0.1185$-$0.1210$): ~10 %. Extends the invariant
  cleanly across the linearity boundary.
- **REGRESSION** (dev $> 0.1200$): ~25-30 %. Training instability from
  tanh saturation at 7 M / 30-ep; paper's stability recipe (gradient
  clipping on $\partial L / \partial S_t$, identity init, careful
  hybrid placement) tuned for 410 M+ scale.

**Diagnostic value regardless of CER outcome — extremely high.**

Three outcomes, all thesis-relevant:
1. **BREAK**: first real ceiling break by a non-anchor mechanism;
   sharpens the cross-experiment invariant — dense per-token freedom
   on *linear* operators doesn't convert, but *non-linear* state
   transition does. Load-bearing thesis result.
2. **MARGINAL / ENGAGED-NULL**: extends the invariant across the
   linearity boundary. New reading: "dense per-(token, head, layer)
   freedom doesn't convert at 7 M / clean-100 / 30-ep, *regardless of
   whether the operator is linear or non-linear in state*." Tightens
   the invariant significantly and locates the ceiling more precisely.
3. **REGRESSION**: evidence that non-linearity + small-scale budget
   is unstable without the paper's stability-recipe tuning. Narrows
   where non-linear-state mechanisms can be deployed.

All three outcomes advance the thesis.

**Recommended sequencing (phased, pre-registered).**
- **Phase A (low-risk diagnostic).** `rwkv6_m2rnn_sparse` at $L^* = 5$,
  pure PyTorch + `torch.compile`. Accept 3-8× wall-clock. Single-seed
  first read on the linearity-of-state axis.
- **Phase B (conditional on Phase A signal).** If Phase A lands
  MARGINAL-or-better: port paper's Triton kernel (or write one), test
  all-layer variant. If Phase A lands PLATEAU or REGRESSION: declare
  the non-linearity axis closed at 7 M / 30-ep and document as the
  cleanest extension of the cross-experiment invariant.

**This is the highest-priority paper for the thesis main line.** Unlike
the prior 8 papers (long-context hybrids, feature-map
reparameterisations, compositional architectures, universal-approximator
foils), M²RNN directly targets the complexity-theoretic expressivity
gap that the thesis is explicitly framed around ("Toward Expressive
and Efficient Sequence Modeling"). It introduces the *only* unexplored
axis (linearity-of-state) in the entire review queue.

**Alternative adaptation (C's, parked as "strict replacement").**
Reviewer C proposed a direct substitution:
$Z_{t,h} = \tanh(S_{t-1,h} W + k_{t,h} v_{t,h}^\top)$,
$S_{t,h} = \text{diag}(w_t) S_{t-1,h} + (1 - \text{diag}(w_t)) Z_{t,h}$.
Closer to the paper's native form but no clean zero-regression at init
— attribution argument depends on "categorical layer swap" framing
(which is legitimate per the paper's sparing-use design), but
methodologically weaker than B's parallel-branch form. Parked; not
championed.

---

## Phase 4 — Cross-paper overlap & similarity analysis

Meta-analysis across the 9 reviewed papers. Paper-by-paper reviews
treated each mechanism in isolation; this phase looks across them for
(i) conceptual redundancy (synonyms), (ii) mechanism families
(competing solutions to the same bottleneck), (iii) unexplored axes,
and (iv) consolidated summary tables on transferability, implementation
difficulty, and expected gain.

### 4.1 Master summary table

At-a-glance view of all 9 paper reviews:

| # | Paper | Verdict | Championed adaptation | Params | Wall-clock | Zero-regr. | Exp. band |
|---|---|---|---|---|---|---|---|
| 1 | Log-Linear Attn | **YES, cleanly** | Fenwick-bucket recurrence + $\lambda_t^{(\ell)}$ mixer | ~57 K (0.8 %) | 2–3× | Exact | NULL→MARGINAL |
| 2 | NaLaFormer | CONDITIONAL / LOW | Signed-power $\|r\|^{p_t}$ on $(r, k)$ | ~6 K (<0.1 %) | <5 % | Exact | PLATEAU |
| 3 | Avey | COMPONENT-ONLY | Partial-embedding bypass in ChannelMix (interp $\alpha$) | 6 scalars | ~0 | Exact | PLATEAU→MARGINAL |
| 4 | RWKV-X | **NO / OUT-OF-REGIME** | — | — | — | — | N/A |
| 5 | HSA / RAMba | **NO / OUT-OF-REGIME** | — | — | — | — | N/A |
| 6 | CRT / NCGRU | PARTIAL / LOW | Cayley-orthogonal transition $(I-A)(I+A)^{-1}$ | **~3 M (43 %, breaks parity at $r=4$)** | 3–5× | Exact (zero-init $U,V$) | AMBIGUOUS-ENGAGED |
| 7 | Non-Attention | PARTIAL / MOD | Multi-dilation ConvShift $\{1,2,4,8\}$ | ~18 K (0.25 %) | <5 % | Exact ($\alpha_1{=}1$) | MARGINAL→PLATEAU |
| 8 | PoM | PARTIAL / LOW | Polynomial value-lift $\hat v = v + \sum_p \gamma_p h(W_h x)^p$ | ~96 K (1.4 %) thin | Negligible | Exact | PLATEAU |
| 9 | M²RNN | **YES (sparing-use)** | Parallel-branch: $\tanh(SW+kv^\top)$ fused via $\lambda_h$ | ~17 K (0.25 %) | 3–8× sparing | Exact (B's form) | **BREAK 20% / MARG 45% / PLAT 10% / REGR 25%** |

### 4.2 Transferability matrix

For each portable proposal, transferability to the three downstream baselines:

| # | Paper | RWKV-6 | Mamba-2 | Linear Attention | Lion (bidir) |
|---|---|---|---|---|---|
| 1 | Log-Linear | HIGH | **HIGH** (paper-direct) | HIGH | MOD (mirror-Fenwick design cell) |
| 2 | NaLaFormer | LOW | LOW-MOD | **HIGH** (paper's target; needs explicit L1) | MOD-LOW |
| 3 | Avey bypass | MOD | HIGH | HIGH | HIGH (position-pointwise) |
| 6 | NCGRU-Cayley | LOW | MOD (invasive, loses chunked scan) | LOW | **BREAKS** (non-commutative $\prod O_\tau$; no $e^{cs}$) |
| 7 | Multi-dil ConvShift | MOD | HIGH | HIGH | HIGH (direction-agnostic DWConv) |
| 8 | PoM value-lift | MOD | MOD | MOD | **HIGH (PoM's natural home)** — permutation-equiv fits LION |
| 9 | M²RNN sparing-use | HIGH | **HIGH** (paper-tested) | HIGH | **BREAKS** (non-linearity incompatible with parallel form) |

**Structural observation across papers 6 and 9.** Both mechanisms that
cross a fundamental structural boundary (P6 to non-commutative orthogonal
transitions, P9 to non-linear state) **break LION's parallel form at
the math level**. This is a thesis-relevant tradeoff:
$NC^1$-expressive-or-non-commutative mechanisms and parallel-form
bidirectional execution (LION fast path) **cannot coexist at the same
layer**. Either sequential execution with richer expressivity, or
parallel execution with the linear/commutative operator class.

### 4.3 Implementation difficulty matrix

| # | Paper | Primary code location | Kernel complexity | Triton / CUDA required? | Stage-1 prototype scope |
|---|---|---|---|---|---|
| 1 | Log-Linear | `rwkv6_time_mix.py` `_forward_recurrent` | Fenwick schedule + $L$ bucket states | No (pure PyTorch fine at $T \le 500$) | Few days |
| 2 | NaLaFormer | `rwkv6_time_mix.py` pre-scan | $r, k$ signed-power | No | Few hours |
| 3 | Avey bypass | `rwkv6_channel_mix.py` | split + interp | No | Trivial |
| 6 | NCGRU-Cayley | `rwkv6_time_mix.py` scan | $(I+A)^{-1}$ per-token matrix inverse | `torch.compile` or Triton | Parity-breaking at useful $r$; prototype hurts |
| 7 | Multi-dil ConvShift | `mechanisms/conv_shift.py` | 4 parallel DWConv branches | No | Few hours |
| 8 | PoM value-lift | `rwkv6_time_mix.py` pre-scan | element-wise powers | No | Few hours |
| 9 | M²RNN sparing-use | `rwkv6_time_mix.py` one layer | Sequential non-linear scan on one layer | `torch.compile` (Stage A) / Triton (Stage B) | Phased, 3–8× wall-clock Stage A |

### 4.4 Overlap clusters (conceptual synonyms across papers)

Five distinct overlap clusters where multiple papers propose
fundamentally the same mathematical transformation under different
names:

| Cluster | Shared structure | Papers involved | Already tested in our chain? | Elegance winner |
|---|---|---|---|---|
| **α — Multi-scale temporal aggregation** | Parameterised weighting over scale-indexed operators | P1 Log-Linear (log-scale transition mask) / P7 multi-dilation ConvShift (log-scale input side) / Multi-Rate RSE (flat, tested PLATEAU) / Stage-4 `rse_depth` (depth-graded, tested WON) | Partially — flat multi-rate plateaued; depth-graded won; log-scale untested | **P1** (log-scale hierarchy + per-token selectivity) |
| **β — Trig/complex feature lift $[\cos, \sin]$ or $e^{i\theta}$** | Embedding real features into 2D rotation-equivariant space via $SO(2)$ | RSE (anchor, tested WON) / P2 NaLaFormer cosine-inhibit | Yes — RSE won | **RSE** (dim-preserving, clean zero-regression, damped-oscillator interpretation; NaLaFormer doubles dim and has no clean reduction) |
| **γ — Outer-product state $k_t v_t^\top$** | Standard Katharopoulos 2020 linear-attention write | RWKV-6 WKV (anchor) / P8 PoM / P9 M²RNN / Stage-6 qtail line | Yes — universal across the entire linear-RNN family | **Not a differentiating axis.** Consequential for M²RNN attribution (see §4.5). |
| **δ — Learnable mixer over parallel branches $\sum_j \mu_{t,j} B_j(x)$** | Weighted sum of parallel tracks with learnable weights | P1 bucket-λ mixer / P²-RSE softmax-β (tested PLATEAU) / qtail-γ (tested MARGINAL) / P8 $\alpha_p$ / P7 $\alpha_d$ | Yes — softmax-β PLATEAU, qtail-γ MARGINAL | **P1** — branches are structurally disjoint Fenwick partitions of past; P²-RSE/qtail-γ showed that when branches lack distinct physical meaning, the mixer adds dense per-token freedom that engages SGD without CER conversion |
| **ε — Chunk-summary retrieval** | Pool chunk keys → top-$k$ selection → sparse exact attention | P4 RWKV-X / P5 HSA / Avey Ranker | Out-of-regime at $T \le 500$ (chunk-selection primitive degenerates) | Not applicable — all dismissed. HSA cleanest design *if* regime ever opened |

### 4.5 Critical attribution insight: M²RNN's novelty is narrower than its framing

**Overlap γ has consequential implications for how we evaluate and
cite M²RNN.** The $k_t v_t^\top$ outer-product write is the **standard
Katharopoulos 2020 linear-attention write**, used across every member
of the linear-RNN family (RWKV-6 anchor, Mamba-2, Gated DeltaNet,
linear attention). It is not a novel M²RNN contribution despite the
paper's "matrix-valued state" framing.

**M²RNN's actual attributable novelty vs our RWKV-6 anchor:**
- **$\tanh$ on the state transition** — the $TC^0 \to NC^1$ jump.
  This is the load-bearing innovation.
- Scalar per-head forget gate with $(\alpha_n, \beta_n)$-parameterised
  sigmoid (minor).

**Thesis-discussion label.** When discussing M²RNN in the thesis,
label the axis **"non-linearity of state"**, not "matrix-valued
state." The latter conflates M²RNN's actual contribution with the
standard linear-attention state form already used throughout our
baseline and every neighbouring family member.

### 4.6 Mechanism families

Five families organize the 9 papers and the internal Stage-2-through-9
mechanisms. Family membership determines whether a proposal introduces
a **new axis** or sits within an **already-characterised one**.

| Family | Defining characteristic | Members (queue + internal) | Complexity class | Already probed at our scale? |
|---|---|---|---|---|
| **A — Multi-scale temporal aggregation** | Scale-indexed operators + mixer over scales | P1 Log-Linear, P7 multi-dilation ConvShift; (+ internal: Multi-Rate RSE tested PLATEAU; `rse_depth` tested WON; `gen2` tested) | $TC^0$ | Partially — flat plateaued; depth-graded won; log-scale untested |
| **B — Linear transition-operator family** | Change Lie group $\{G_t\}$; linear in state | P6 NCGRU-Cayley; (+ internal: Stage-3/4/5 RSE tested WON; Stage-8 T2 tested AMBIGUOUS; Stage-9 S9A/B tested REGRESSION) | $TC^0$ | Yes, extensively — dense per-token engaged-null band well-established |
| **C — Non-linearity of state (crosses to $NC^1$)** | Element-wise non-linearity on matrix state; breaks associativity | **P9 M²RNN (sole member)** | **$NC^1$** | **No — genuinely new axis** |
| **D — Feature-map / channel-side enrichment** | Feature-vector or ChannelMix changes; transition unchanged | P2 NaLaFormer, P3 Avey bypass, P8 PoM; (P7 borderline A/D); (+ internal: Stage-6 qtail line tested MARGINAL at ceiling 0.1238; `rmsnorm` PLATEAU; `hadamard_n2` PLATEAU) | $TC^0$ | Yes — qtail_lowrank_all at MARGINAL ceiling |
| **E — Chunk retrieval / sparse hybrid** | Bolt-on sparse attention over chunked past | P4 RWKV-X, P5 HSA, Avey Ranker | — (mixed) | Out-of-regime at $T \le 500$ |

### 4.7 Family A has the highest cross-paper convergence

Six+ independent sources converge on **multi-scale temporal
allocation** as the key mechanism:

1. **Log-Linear Attention** (Guo et al., P1) — Fenwick-tree log
   hierarchy on transition mask.
2. **Non-Attention LLM** (P7) — multi-resolution dilated convolutions
   on input side.
3. **Avey Ranker** — split-based temporal segmentation (dismissed
   for regime but the structural idea is the same).
4. **HSA** — chunk-level hierarchical attention (dismissed for regime).
5. **Stage-2 `gen2`** (internal) — per-head $\alpha_1$ grows
   monotonically with depth (0.023 → 0.107 across 6 layers).
6. **Stage-4 `rse_depth`** (internal, WON) — depth-graded rotation
   budget (π/8 shallow → π/2 deep); ceiling break.

This **convergent prior** — combined with P1's clean mathematical
composition on RWKV-6 ($M^S \odot M^H$ preserves exact
zero-regression), cheapest per-mechanism parameter cost, and pure
PyTorch tractability — makes P1 the strongest single recommendation
from the entire review queue. Six independent works propose the same
mechanism class; its correctness is indirectly triangulated.

### 4.8 Final priority ranking (post cross-paper synthesis)

| Rank | Proposal | Family | Novel axis vs Stages 2-9? | Why |
|---|---|---|---|---|
| **1** | **P1 Log-Linear Fenwick** | A (new log-scale sub-axis on transition) | **Yes** — log-scale per-token selectivity not yet tested | 6+ cross-source convergence; cleanest $M^S \odot M^H$ composition; exact zero-regression; ~0.8 % params; pure PyTorch |
| **2** | **P9 M²RNN (sparing-use)** | **C — sole member** | **Yes** — only queue entry crossing $TC^0 \to NC^1$ | Sole non-linearity probe; B's parallel-branch form solves zero-regression; sparing-use tractable in pure PyTorch; diagnostic value extremely high regardless of CER (extends or tightens cross-experiment invariant); paper-validated at 7B MoE |
| **3** | **P7 Multi-dilation ConvShift** | A (input-side variant) | Partial — extends validated mechanism | Cheap (~18 K); exact zero-regression; hooks into absolute-best row `rwkv6_rse_convshift` (0.1145/0.1126); physical ASR motivation (±8 frames = syllable envelope); diagnostic on depth-graded receptive-field allocation |
| 4 | P3 Avey partial-embedding bypass | D (channel-mix) | No — Tier-3 probe on under-tested sub-axis | 6 scalars, near-zero cost; tests over-smoothing as failure mode diagnostically |
| **Drop** | P2 NaLaFormer | D / overlap β with RSE | **No** — pathology absent in RWKV-6 | Diagnosis (L1-cancellation) doesn't apply; "fix" reduces to closed AMBIGUOUS-ENGAGED class; overlap β with RSE (RSE is elegance winner) |
| **Drop** | P6 CRT NCGRU-Cayley | B (linear transition sibling) | **No** — sibling of T2 | Parameter parity breaks at useful rank ($r \ge 4$); dense per-token deployment matches T2 / S9 failure shape; LION parallel form breaks; diagnostic tiebreaker only |
| **Drop** | P8 PoM | D / overlaps γ (Katharopoulos) + δ | **No** at $k=2$ | Element-wise lift at parity-fitting $k=2$ sits in already-probed quadratic function-class family (hadamard_n2 PLATEAU, qtail_lowrank_all MARGINAL); natural home is standalone `lion_pom`, not causal RWKV anchor; theoretical-foil value for thesis discussion |
| **N/A** | P4 RWKV-X, P5 HSA | E (out-of-regime) | No | Chunk-selection degenerates at $T \le 500$; natural home is long-form audio scope expansion, not current thesis line |

### 4.9 Three actionable takeaways

**1. The queue is more redundant than it looks.** Of 9 papers, only
**3 target axes the thesis has not already explored** in Stages 2-9:

- **P1 Log-Linear** — new log-scale sub-axis within Family A (log-scale
  per-token selectivity on transition side).
- **P9 M²RNN** — sole Family C member, genuinely new axis (crosses
  $TC^0 \to NC^1$).
- **P7 multi-dilation ConvShift** — input-side variant of Family A
  (extends a validated mechanism, not a wholly new direction).

The remaining 6 papers are either out-of-regime (P4, P5), duplicates of
existing mechanisms under different names (P2 overlaps RSE via β;
P8 overlaps qtail via γ and δ at parity-fitting $k$), or variants
within already-characterised families (P3 Family D Tier-3; P6 Family B
sibling of T2 engaged-null).

**2. M²RNN's attributable novelty is narrower than its framing.** The
matrix-valued outer-product state ($k v^\top$ write) is standard
Katharopoulos 2020, used across every linear-RNN family member
including our RWKV-6 anchor. M²RNN's novelty vs RWKV-6 reduces to
**$\tanh$ on state + scalar per-head forget gate**. Label the axis
*"non-linearity of state"*, not "matrix-valued state", for attribution
clarity in the thesis.

**3. Family A has the highest cross-paper convergence.** Six
independent sources (P1 Log-Linear, P7 Non-Attention, Avey Ranker,
HSA chunk hierarchy, internal Stage-2 `gen2`, internal Stage-4
`rse_depth`) propose multi-scale temporal allocation as the
mechanism of interest. Combined with P1's clean mathematical
composition on RWKV-6 (preserves exact zero-regression) and cheapest
per-mechanism parameter cost, this convergent prior makes P1
**the strongest single recommendation from the entire review queue**.

### 4.10 A structural observation: non-linearity / non-commutativity vs LION parallel form

Two proposals (P6 NCGRU-Cayley, P9 M²RNN) break LION's parallel form
at the math level. Non-commutative $\prod O_\tau$ (Cayley-orthogonal)
and non-linear $\tanh(SW + kv^\top)$ (M²RNN) both fail to admit the
$e^{cs}$ cumulative-sum factorisation LION's fast path depends on.

This surfaces a structural tradeoff at the thesis level:

- **$NC^1$-expressive or non-commutative** mechanisms (which unlock
  state tracking beyond $TC^0$) require **sequential execution**.
- **Parallel-form bidirectional execution** (LION, Mamba-2 parallel)
  is structurally confined to the **linear, commutative operator
  class** ($TC^0$).

These two properties **cannot coexist at the same layer**. Either
sequential execution with richer expressivity, or parallel execution
with the weaker operator class. This is worth citing in the thesis
discussion as a design-space-level constraint, not a limitation of any
specific paper.

