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

