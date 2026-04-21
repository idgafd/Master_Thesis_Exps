# TODO — Future Ideas for LION / Linear Attention / Beyond Stage 6

Parking lot for mechanisms from the literature that are **not** on the current
causal RWKV-6 critical path but would be natural to test when the thesis
returns to Group B (LION / bidirectional) or to a wider attention-variant
comparison.

**Sibling docs:**
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

**Working principle — refinement before composition.**

The Stage-2 → Stage-5 pipeline in this thesis followed a single discipline:
when a proposed mechanism lands as MARGINAL (not PLATEAU, not BREAK),
**iterate within the single mechanism before stacking another on top.**
This is how Stage-3 `rse` (0.1251) became Stage-4 `rse_strong` (0.1192) via
budget refinement, then Stage-5 `rse_strong_viscosity` (0.1185) via clip
refinement. At each step the math motivation was internal to the mechanism.

When qtail projected ~0.1237 (MARGINAL), the correct next step was **R1
qtail-γ** (learnable per-head decay coupling on the Kronecker branch —
principled by the paper's decay-free Taylor derivation), NOT a composition
like qtail-on-all-layers or qtail-×-RSE. The same principle applies to
Phase-2b: if it shows MARGINAL, refine the indep-λ parameterization (e.g.,
γ-coupled λ₂/λ₁) before stacking indep-(k,v) on top.

Composition candidates are kept in this file for the **final ablation
stage** — after each single mechanism has been iterated until it caps out.
Stacking in the exploratory phase multiplies uncertainty without adding
clear signal; refinement gives a clean math story and a small-params delta
with a targeted hypothesis.

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

### Phase 2b — independent-λ P²-RSE
Stage 5 §6.4(1) diagnosed shared-λ as the most likely reason P²-RSE
under-performed. The clean one-run test is paired poles with **independent**
decay LoRAs `lambda_lora_1`, `lambda_lora_2` in addition to the already-
independent θ LoRAs.
- Stacks on `rwkv6_rse_strong_viscosity` (current causal best, test CER 0.1177).
- Pre-registered thresholds from STAGE5_PLAN §3 still apply (BREAK 0.1160, MARGINAL 0.1180).
- Full memo in `memory/project_phase2b.md`.

### Phase 2b-ext — independent (k, v) per pole
If 2b shows a meaningful gain, the next variant relaxes the shared-drive
constraint — each pole gets its own `key`/`value` linear projections. Costs
2× on those two projections, but matches the acoustic prior that two
formants read from different regions of the cochlear spectrum. Stage 5
§6.4(2).

---

## From EXPRESSIVENESS paper (Mongaras & Larson, arXiv 2507.23632)

Link: https://arxiv.org/abs/2507.23632 · local: `papers/ EXPRESSIVENESS_2507.23632v1.pdf`

### Full Kronecker n=2 at all layers (currently running only at top 2)
If Stage 6 `rwkv6_qtail` (top-2 layers) shows a clear win over `rwkv6_rmsnorm`,
re-run with the Kronecker branch at all 6 layers. ~1.6× runtime vs qtail but
tests whether the depth-gated restriction was the bottleneck. Only worth it
if top-2 qtail delivers a positive signal.

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
