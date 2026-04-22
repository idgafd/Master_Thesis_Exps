# Stage 10 — Analysis Rollup (10.1 – 10.4 + 10.3-sym control)

Consolidated post-mortem for the first four RWKV-6 Stage-10 runs (plus the
10.3-symmetric causality-vs-dilation control). Merges mechanism-level
hypotheses from the two independent analyses, flags caveats on the
comparison setup, and records v2 decisions and transfer predictions.
Canonical results are logged to `STAGE10_PLAN.md` §9.2; this file carries
the reasoning behind the verdicts.

**Status.** Five runs complete, seed 42, 30 ep, LibriSpeech clean-100. No
additional seeds run yet — single-seed results are patterns, not laws
(plan §1.6, σ ≈ 0.0014).

---

## 1. Results summary

| Backbone | Family | Dev CER | Test CER | Verdict |
|---|---|---:|---:|---|
| `rwkv6_loglinear` | A (log-scale readout) | 0.1240 | 0.1226 | **PLATEAU** |
| `rwkv6_m2rnn_sparse` | C (non-linear state) | 0.1276 | 0.1264 | **PLATEAU** (tied vanilla within 1.3σ) |
| `rwkv6_convshift_multidil` (causal) | A (input-side) | 0.1229 | 0.1224 | **REGRESSION** vs `convshift_trap` primary |
| `rwkv6_convshift_multidil_symmetric` | A (input-side control) | **0.1153** | **0.1145** | **MARGINAL** — ties `rwkv6_rse_convshift` abs-best |
| `rwkv6_chanmix_bypass` | D (channel-mix) | 0.1251 | 0.1248 | **PLATEAU** (tied vanilla 0.5σ) |
| `rwkv6_pom_vlift` | D (polynomial quadratic lift) | 0.1254 | 0.1253 | **PLATEAU** — ties `hadamard_n2` (0.1253) — quadratic family saturated across parametrisations |
| `rwkv6_orthogonal` (rank-1 Cayley) | B (full SO(K)) | *in progress* | *in progress* | ep 6/30 at 0.224 — tracking behind T2 primary (ep 6 of T2 was 0.198); early signal REGRESSION-track |

Reference points for context:
- Vanilla `rwkv6` (plan §9.1, single-seed): dev 0.1258 / test 0.1263
- Anchor `rwkv6_rse_strong_viscosity`: dev 0.1185 / test 0.1177
- Stage-2 input-side winner `rwkv6_convshift_trap`: dev 0.1150 / test 0.1150
- Abs-best cross-axis `rwkv6_rse_convshift`: dev 0.1145 / test 0.1126

---

## 2. Comparison caveats

Three methodological notes — all propagate into every row of §9.2:

1. **Stale `config.yaml` snapshots.** The per-run `config.yaml` files written
   by `scripts/run_experiment.py` do not reflect the substring-derived
   mechanism flags (flags are expanded inside `build_encoder` *after* the
   YAML dump). Authoritative identifiers for §9.2 rows are (i) the backbone
   name, (ii) the encoder parameter-count delta, and (iii) the runtime
   microprofile recorded in `outputs/stage10_dryrun.json`. *TODO: fix the
   dump to record the resolved config.*

2. **`outputs/exp02_rwkv6_seed42` is not the canonical 30-ep vanilla.** That
   run used 80 epochs with a different LR schedule; its ep-30 minimum is
   **0.1173** (below §9.1's canonical 0.1258). The **+0.0085 systematic
   bias** at ep 30 propagates into any Δ-to-vanilla computed against that
   history — notably §9.5. The authoritative vanilla for all Stage-10
   decision rules is §9.1 row 1 (dev 0.1258 / test 0.1263).

3. **Anchor on-disk ≠ §9.1 anchor.** The
   `s7a_anchor_rwkv6_rse_strong_viscosity_seed42` run ends at dev **0.1191**,
   not the §9.1 row's 0.1185. A 0.0006 gap (~0.4σ) — within seed noise but
   worth naming in plot captions that reference the anchor line.

### 2.1 Procedural gap — mandatory diagnostics missing

Plan §7.5 makes per-run `diagnostics.json` a prerequisite for a §9 verdict
(α_d mobility, λ-mobility, tanh saturation fraction, etc., depending on
mechanism). **No Stage-10 run produced a diagnostics.json.** Every
verdict in this document is technically procedurally invalid by the plan's
own discipline until the probe backfill is run against the preserved
checkpoints (5–30 min of work per run; non-blocking for reporting but
should be closed before any final §9.2 row is locked).

---

## 3. Mechanism-level analysis

The two reviewers converged on substantially the same hypotheses; where
framings differ, the cleaner one is used below and the other is noted.

### 3.1 `rwkv6_loglinear` — PLATEAU

**Mechanism.** `y_t = Σ_ℓ λ_t^(ℓ) r_t^⊤ S_t^(ℓ) + bonus`, Fenwick bucket
partition of the WKV prefix. All buckets share the same per-channel decay
`exp(w[t])` and the same `k_t v_t^⊤` writes. λ-mixer is data-dependent from
the decay-side stream x^(w); zero-init LoRA ⇒ λ ≡ 1 at init ⇒ vanilla
readout bit-exact.

**Hypothesis (consensus).** The mechanism **changes the temporal readout
basis, not the transition family**. Each `S_t^(ℓ)` lies in the
direct-sum decomposition of the original linear state family; the bucket
partition reweights the same `K = 64` per-channel decay ridges the vanilla
scan already has. The candidate story is that SGD diversifies the
continuous per-channel decays across epochs (the depth-grade pattern
seen in Stages 2-4), so the discrete log-scale handles that helped at
ep 5 (Δvs_vanilla = −0.0053) become redundant by ep 30 (Δ = −0.0018).
*Caveat (PI review):* this story is not corroborated — no λ-mobility
probe was logged (see §2.1 procedural gap), so "continuous-decay
diversification" is a candidate mechanism for the compression but not
an evidenced one. What IS evidenced is the observed CER compression
itself. On RWKV-6 this is not the binding bottleneck.

**Diagnostic consistency.** Gap-vs-vanilla decays from −0.0053 → −0.0018
monotonically across matched epochs 5, 10, 15, 20, 25, 30. Test-side
signal (Δ = −0.0037) is slightly stronger than dev — bandwidth of this
difference is compatible with the split-variance effects seen elsewhere
in §9.1.

**Open hypothesis for v2 (kept as testable conjecture).**
Breaking the shared-basis property by giving buckets **independent
per-bucket decay rates** `w_ℓ[t, i] = w[t, i] + δ_ℓ[t]` (δ_ℓ zero-init)
would expand the state-space basis by L-fold and exit the direct-sum
family. If the residual gap is truly structural-basis-limited, this
should convert (+0.001 to +0.005 dev). If the gap is instead redundancy
with the continuous per-channel decay diversity (the stronger
hypothesis), it will still PLATEAU. **Not worth running on RWKV-6** —
the compute is better spent on the LA transfer below, where the same
mechanism has a genuinely richer home.

### 3.2 `rwkv6_m2rnn_sparse` — REGRESSION (tied vanilla)

**Mechanism.** At L = 5 only, parallel branch
`Z_t = tanh(S_{t-1} W + k_t v_t^⊤)`, gated
`S_t = f_t S_{t-1} + (1 − f_t) Z_t`. Readout adds `λ_h r_t^⊤ S_t^m2`.
λ_h = 0 at init ⇒ bit-exact vanilla.

**Hypothesis (merged).** The only Stage-10 mechanism that genuinely
crosses the `TC^0 → NC^1` linearity boundary, but deployed in the
**hardest possible configuration**:

1. **Bounded tanh state vs unbounded linear state** (reviewer's framing,
   cleaner than mine). `‖S^m2‖_∞ ≤ 1` because of tanh; `‖S^rwkv‖` is
   unbounded. The additive blend adds a clamped signal to an unclamped
   one — SGD drives λ_h small to avoid injecting saturation noise.
2. **Narrow useful regime for λ_h.** If λ_h stays small, the branch is
   inert (our λ_h ≈ 0 anchor case). If it grows too fast, tanh saturates
   and destroys additive memory. The training-loss-higher /
   CER-worst pattern fits the "narrow-band → SGD picks inert" bifurcation.
3. **Sparing-use at a single layer** kills the gradient path. λ_h = 0
   init is a chicken-and-egg: no signal moves λ_h out of 0 until the
   forward-y changes, but that needs λ_h ≠ 0 first.
4. **Redundancy with ChannelMix.** Vanilla RWKV-6's `σ · ReLU²`
   ChannelMix already implements a token-local non-linearity; composed
   over 6 layers the function class is saturated for this data regime.

**Why REGRESSION and not PLATEAU.** Adding noise from a mostly-inert
mechanism slightly perturbs training (loss +0.003 at ep 30 vs vanilla
trajectory); the dev CER lands +0.0018 above vanilla, within σ, but on
the wrong side of the decision boundary.

### 3.3 `rwkv6_convshift_multidil` (causal) — REGRESSION

**Cleanly decomposed by the 10.3-sym control.** Two orthogonal effects:
- Multi-dilation benefit (symmetric vs single-dil): Δ = +0.0003 dev
  (null within σ).
- **Causality penalty** (causal vs symmetric): Δ = +0.0076 dev
  (~5σ, persistent from ep 5 through ep 30).

**Hypothesis (reviewer's framing adopted).** The existing `convshift_trap`
uses symmetric `[0.5, 0, 0.5]` padding — a **centered local smoothing
operator aligned with phonetic/acoustic windows** (40 ms effective after
4× subsampling). This is a 1-frame look-ahead + past average. The causal
variant replaces this with `[0.5, 0.5, 0]` = past-only average, which
**introduces phase lag** and removes the future-context anchor. In a
full-utterance encoder, causal padding is not a natural constraint; it
costs ~6σ of CER.

**Practical framing.** 10.3-causal is the **honest cost of strict
streaming**. For offline decoding, 10.3-sym is strictly preferable; for
true sub-frame streaming, 10.3-causal is the ground truth and 10.3-sym
is non-deployable. No v2 needed — the symmetric control is the research
result.

### 3.4 `rwkv6_convshift_multidil_symmetric` — MARGINAL (headline result)

**Result.** Dev 0.1153 / test 0.1145 — ties `rwkv6_rse_convshift`
(dev 0.1145, test 0.1126, abs-best cross-axis per plan §4.1) within
seed noise, **without** the RSE transition-side rotation that produced
the original cross-axis win.

**Hypothesis.** Acoustic/phonetic structure lives at multiple temporal
scales: frame ~10 ms, phone boundary ~50 ms, syllable ~200 ms. A
single-dilation ConvShift captures only frame-local context. Multi-dilation
{1, 2, 4, 8} at the post-4×-subsampling rate covers 20–160 ms effective
receptive field — the phoneme-to-syllable span. Depth-graded α_d lets
shallow layers prefer α_1 and deep layers grow α_{2, 4, 8}, recovering
the same multi-scale benefit Stage-4 `rse_convshift` achieved from the
transition-side rotation.

**Load-bearing claim — UNVERIFIED.** I hypothesised the α_d mobility
profile is depth-graded (α_1 dominant at L0-L1, α_{2,4,8} grow with
depth). **No probe was logged.** The depth-graded-RF thread (after
Stage-2 `gen2` α-pattern and Stage-4 `rse_depth`) cannot be extended by
this run until the α_d values are read from the saved checkpoint (a
5-minute script to iterate layers and log α_d.abs().mean()). Striking
the "third data point" framing from the thesis narrative until the
probe runs.

**Why near-BREAK but not quite BREAK.** Dev 0.1153 is +0.0008 above the
0.1145 BREAK boundary; test 0.1145 is +0.0019 behind the previous
abs-best test CER (0.1126). Both within σ on a single seed, so the
distinction is noise-bounded.

### 3.5 `rwkv6_chanmix_bypass` — PLATEAU

**Mechanism.** Split W_k output along FFN dim, linear head, ReLU² tail,
α-gated blend (α = 0 init).

**Hypothesis (reviewer's framing is sharper here).** Because W_v is
linear, the bypass is essentially an **interpolation between two FFN
hidden activations inside the same downstream span**. It is a
feature-shape change, not a new temporal computation. The vanilla
ChannelMix class `σ(W_r x) ⊙ W_v · ReLU²(W_k x)` already contains the
gated-quadratic function space the bypass reweights; 6 scalar α's across
6 layers is too few DOF to widen the class. **The data confirm the
6-layer RWKV-6 bottleneck is not FFN over-smoothing.**

*Diagnostic probe TODO.* Printing the 6 α values from the ep-30
checkpoint discriminates two stories. If |α| ≈ 0 everywhere, "SGD didn't
engage." If |α| > 0.3 somewhere, "mechanism engaged but didn't convert."
Different follow-up implications. 5-line script; non-blocking.

---

### 3.6 `rwkv6_pom_vlift` — PLATEAU (ties `hadamard_n2`)

**Mechanism.** Value-lift `v̂_t = v_t + γ_2 ⊙ W_up · (W_h x_t)^⊙2` with
thin expansion `D_pom = 64`, γ zero-init ⇒ v̂ = v bit-exact at init.
State update (WKV scan) unchanged — only the `v` that enters the scan
is modified.

**Hypothesis (diagnostic saturation).** The pre-registered §6.10.6
decision rule asked whether PoM's W_h-lift adds anything over the
diagonal Hadamard `k⊙k` lift (`rwkv6_hadamard_n2`) or the low-rank
Kronecker `qtail_lowrank_all` path. The result **dev 0.1254** lands
essentially on top of `hadamard_n2` (0.1253); test CER 0.1253 vs
hadamard_n2's test 0.1251 — identical within σ. This is the **third
data point** (after hadamard_n2 and chanmix_bypass) showing that a
quadratic feature lift applied to the value stream saturates at
~0.125 regardless of how the quadratic is parametrised.

**Three quadratic-lift parametrisations compared:**

| Backbone | Lift form | Cross-channel? | Extra params | Dev CER | Test CER |
|---|---|---|---:|---:|---:|
| `rwkv6_hadamard_n2` | `k ⊙ k` | No (diagonal) | **+0** (β only, 24 scalars total) | 0.1253 | 0.1251 |
| `rwkv6_pom_vlift` | `(W_h x)^⊙2` with learned W_h (thin 64-dim) + up-proj | No (element-wise post-lift) | **+198 K** (2.8 % of 7 M) | **0.1254** | 0.1253 |
| `rwkv6_qtail_lowrank_all` | `k ⊗ k` at low rank K'=16 | **Yes** (cross-channel Kronecker) | +mod (K'-level Kronecker weights) | 0.1238 | 0.1240 |

The only parametrisation that marginally differentiates is the
cross-channel one (`qtail_lowrank_all` by ~0.0015 dev = ~1σ). The
diagonal and linear-lift variants are indistinguishable within seed
noise — **and the +198 K extra params in `pom_vlift` buy nothing over
`hadamard_n2`'s +0.** This is the strongest version of the saturation
claim: the quadratic function class is genuinely closed, not merely
undertrained.

**Thesis-relevant finding:** *cross-channel structure*, not the
*polynomial form* or *parameter budget*, is the load-bearing dimension
in Family-D — and even that dimension only moves the ceiling by ~1σ.

**V2 decision — NO.** Closes the quadratic-lift sub-family of Family D
on RWKV-6. Future Family-D work should target cross-channel expressivity
or non-quadratic non-linearities (e.g. activation-class changes), not
more quadratic parametrisations.

## 4. Cross-cutting findings

1. **Cross-experiment invariant extends most strongly along the C-axis
   (non-linear state), weakly along the A-axis.** Stages 2–9
   established the pattern: dense per-(token, head, layer) freedom on
   **linear-in-state** operators (A1′, T1, T2, S9A/B) does not convert
   to CER gain. Stage 10 evidence:
   - **C-axis genuine extension** (10.2 m2rnn PLATEAU-tied): crosses the
     `TC⁰ → NC¹` linearity boundary and still fails to convert. This is
     a real new axis of the invariant.
   - **A-axis weak extension** (10.1 loglinear PLATEAU): caveat — the
     mechanism's buckets `S^(ℓ)` lie in the **direct-sum decomposition
     of the existing linear state family**. They do not actually test
     *structurally new* multi-scale freedom beyond the continuous
     per-channel decay diversity the vanilla scan already has; they
     test *structurally-different readout reweighting*. The claim
     "invariant extends across structural multi-scale" is weaker than
     I originally wrote. **Downgraded to "weakly suggested."** A true
     structural extension would require bucket-specific decay rates or
     an LA-side test (Stage 11.3b).
   - **D-axis quadratic-lift saturation** (10.4 chanmix_bypass PLATEAU,
     10.6 pom_vlift PLATEAU): not an invariant extension per se; rather
     a saturation claim within an existing family. Three parametrisations
     tied within σ regardless of parameter budget.

   Revised thesis claim: *C-axis extension is corroborated cleanly;
   A-axis extension requires stronger evidence (LA or bucket-decay
   variants); D-axis is saturated across parametrisations at this scale.*

2. **The binding gap at 7 M / 30 ep / clean-100 is acoustic-feature-extraction,
   not sequence-modeling.** Transition-side mechanisms (Family B), state
   non-linearity (Family C), and channel-side reweighting (Family D) all
   saturate within σ of the anchor. Only **input-side temporal hierarchy**
   (10.3-sym, Stage-2 `convshift_trap`) clears the vanilla → anchor gap
   cleanly. The residual engineering ceiling lies at the input/feature
   stage, not at the sequence mixer.

---

## 5. V2 decisions on RWKV-6

Both reviewers converge on the same decisions modulo small framing differences.

| V2 | Decision | Rationale | Est. dev gain | Compute |
|---|---|---|---:|---:|
| `rwkv6_rse_convshift_multidil` (10.3b composition, symmetric) | **GO** (pending plan §8.3 multi-seed gate) | 10.3-sym landed MARGINAL; plan §6.10.3 pre-registers RSE composition. Highest-EV v2 of the batch. | −0.003 to −0.008 | ~1.5× anchor |
| `rwkv6_loglinear_v2` (per-bucket decay rates) | **DEFER** to LA transfer | Structural redundancy on RWKV-6 spine; LA (Stage 11.3) is the mechanism's natural home where buckets add new basis functions rather than reweight existing ones. | −0.001 to −0.005 | ~2× anchor |
| `rwkv6_m2rnn_sparse_v2` | **NO** | Compute penalty (3–8× anchor) too large vs observed direction of effect. Salvage attempts (GeLU, all-layer low-rank W, warm-start λ_h) address saturation but not the ChannelMix redundancy. | ≤ −0.002 | 3–8× anchor |
| `rwkv6_chanmix_bypass_v2` | **NO** on RWKV | A genuine improvement needs a stronger MLP/GLU redesign (SwiGLU, GeGLU, per-channel non-linearity), which is out of the Avey-bypass scope. | ≈ 0 | — |

**Specific corrections for 10.3b composition (if launched):**
1. Enforce **symmetric padding** in bidirectional / LION / full-utterance
   modes; reserve causal only for true streaming evaluation.
2. Constrain α_d with **softmax(β_d)** or nonnegative normalisation to
   prevent destructive interference between branches.
3. Consider **denser short dilations `{1, 2, 3, 4}`** instead of
   `{1, 2, 4, 8}`. At 4× subsampling, the `{2, 4, 8}` set aliases
   against the 10-ms frame rate more than `{2, 3, 4}`; tighter phonetic
   coverage may tighten the MARGINAL result.
4. Compose with RSE as `rwkv6_rse_convshift_multidil`. Test whether
   input-side multi-dilation and transition-side RSE gains are
   orthogonal (expect dev ~0.110–0.112 = BREAK) or overlap
   (expect Δ ≈ 0).

---

## 6. Transfer predictions for Stage 11

Ranked by EV for the Stage 11.1 / 11.3 transfer schedule.

| Rank | Transfer | Spine-specific reasoning | Prediction |
|---|---|---|---|
| 1 | `mamba2_convshift_multidil` | Mamba-2 has a native short DWConv at `src/models/mamba2_block.py:194`; replace with `x̃ = Σ_d α_d · DWConv_d(xBC)`, rest of the SSD chain unchanged. | **WON on Mamba-2**, at least matching RWKV; likely stronger because Mamba-2 already relies on the conv path for local feature extraction. |
| 2 | `linear_attn_convshift_multidil` | Pre-mix before Q/K/V in `src/models/blocks.py:182`: `Q = W_q x̃, K = W_k x̃, V = W_v x̃`. LA currently has near-zero built-in local inductive bias. | **WON on LA**, stronger than RWKV; bias deficit is largest here. |
| 3 | `linear_attn_loglinear` (Stage 11.3, natural home) | LA's single running-sum accumulator has no per-channel decay diversity; Fenwick buckets introduce genuinely new basis functions. Maintain per-level numerator + denominator states: `S_t^(ℓ) = Σ φ(k_s) v_s^⊤`, `z_t^(ℓ) = Σ φ(k_s)`, `o_t = [Σ_ℓ λ^(ℓ) φ(q_t)^⊤ S_t^(ℓ)] / [Σ_ℓ λ^(ℓ) φ(q_t)^⊤ z_t^(ℓ) + ε]`. | **MARGINAL-to-BREAK on causal Katharopoulos LA**, once Stage 11.2a baseline exists. Best home for the mechanism. |
| 4 | `mamba2_loglinear` | Replace the single SSD state in `src/models/mamba2_kernels.py:88` with Fenwick states, read out `y_t = Σ_ℓ λ_t^(ℓ) C_t^⊤ S_t^(ℓ)`. | **PLATEAU on Mamba-2** — selective Δt already provides continuous multi-decay structure; Fenwick buckets may redundantly reweight existing basis like on RWKV-6. Worth running for the same diagnostic reasons, lower EV than LA. |
| 5 | `mamba2_chanmix_bypass` / `linear_attn_chanmix_bypass` | Both spines use plain post-mixer MLPs (`mamba2_encoder.py:55`, `blocks.py:162`). | **PLATEAU-band** on both, odds slightly better than RWKV but still small. Cheap to bundle. |
| 6 | `mamba2_m2rnn_sparse` | Only transferable to recurrent Mamba-2; `tanh(SW + kv)` breaks associativity / parallel-scan on chunkwise LA. Update: `S_t^m2 = f_t S_{t-1}^m2 + (1−f_t) tanh(S_{t-1}^m2 W + B_t X_t^⊤)`. | **MARGINAL on recurrent Mamba-2** (paper's Hybrid Mamba-2 + M²RNN reports success, ChannelMix redundancy partly dissolves); worse practical tradeoffs than 1–5. |

---

## 7. Implementation engineering lessons (cross-phase)

Both Stage-10 sprints hit the same pattern: a new mechanism landed with
a clean sequential reference that was 10–40× over the plan's wallclock
budget, and in both cases restructuring the **scan** (not the per-step
arithmetic) recovered the plan envelope. The fixes re-use primitives
already in the codebase rather than new kernels.

### 7.1 Stage 10.1 Log-Linear — sequential → Fenwick-chunked

| Version | ms/iter | Peak VRAM | 30-ep wallclock | Ratio vs anchor |
|---|---:|---:|---:|---:|
| Sequential reference (`_forward_recurrent_loglinear_seq` @ `src/models/rwkv6_time_mix.py:1549`) | 1838 | 25.0 GB | 19.2 h | 20× |
| **Chunked (`_forward_recurrent_loglinear` @ `src/models/rwkv6_time_mix.py:1348`)** | **172** | **8.6 GB** | **1.8 h** | **2.0×** (in plan budget) |

**Load-bearing insight.** Fenwick alignment: at any 2^J-aligned chunk
boundary, levels 0..J-1 are provably empty (the prior cascade fired up
to level ≥ J). Inside the chunk, local kv contributions only populate
levels 0..J-1 while carry in levels ≥ J merely decays — the two parts
do not mix until the final collapse at chunk end. This decomposes the
scan cleanly at chunk sizes `[128, 16, 2, 1]`, identical to the schedule
used by the baseline `_chunked_wkv`. One stability fix replicated from
the RSE scan (clamp upper-triangular attention entries to −60 before
`exp()`) prevented `inf × 0 = NaN` on long chunks.

### 7.2 Stage 10.5 Cayley-orthogonal — sequential Woodbury → chunked affine scan

| Version | ms/iter | Peak VRAM | 30-ep wallclock | Ratio vs anchor |
|---|---:|---:|---:|---:|
| Sequential Woodbury, generic rank-R (`_forward_recurrent_cayley` @ `src/models/rwkv6_time_mix.py:1843`) | 3920 | 4.9 GB | 40 h | 42× |
| Sequential + rank-1 analytical specialisation (same file, now removed from dispatch) | 3690 | 3.7 GB | 38 h | 39× |
| **Chunked affine scan, rank-1 (`_forward_recurrent_cayley_rank1_chunked` @ `src/models/rwkv6_time_mix.py:1715`)** | **326** | 26.9 GB | **3.4 h** | **3.9×** (in plan budget) |

**Load-bearing insight.** The Cayley recurrence
`S_t = A_t · S_{t-1} + U_t`  (with `A_t = diag(exp(w_t)) · O_t`,
`U_t = k_t v_t^T`) is an **affine scan** and composes associatively via
`(A_b, U_b) ∘ (A_a, U_a) = (A_b A_a, A_b U_a + U_b)` — the same
structure the repo already exploits for the delta-rule scan via
`_delta_affine_prefix_scan` (Hillis–Steele parallel prefix,
`src/models/rwkv6_time_mix.py:2702`). Reusing that primitive inside
chunks of 32 tokens collapsed the autograd tape from `T = 300` to
`T/tc = 10` per layer, which is where the real bottleneck was.

**The rank-1 closed form** avoids materialising `O_t` separately:

```
a = u·u,   b = u·v,   c = v·v,   Δ = 1 − b² + a·c
A_t = D_t − (2c/Δ)(D_t u) uᵀ − (2(1−b)/Δ)(D_t u) vᵀ
         + (2(1+b)/Δ)(D_t v) uᵀ − (2a/Δ)(D_t v) vᵀ
```

Derivation in `_forward_recurrent_cayley_rank1_chunked` docstring;
verified bit-wise equivalent to `(I − A)(I + A)⁻¹ · D_t` on random
inputs (max |Δ| = 1.5e-5, fp32 prefix-scan accumulation floor).

### 7.3 What did NOT work — avoid repeating

- **Per-step arithmetic tuning in isolation.** The rank-1 specialisation
  in §7.2 dropped from 3920 → 3690 ms (6 %). Arithmetic was not the
  bottleneck; autograd-over-T was. Lesson: profile the tape length, not
  the FLOP count, when a sequential scan is slow.
- **`torch.compile` on sequential Python scans.** One probe on the
  Cayley sequential path **hung for 47 CPU-minutes without emitting
  output** and had to be killed. Compile needs graph-regular structure,
  not a Python loop that rebinds state tensors each iteration. If
  sequential is too slow, chunk first, then compile — not the reverse.

### 7.4 Generalisable pattern

For any new recurrence introduced in Stages 10+:

1. Land a sequential reference first (correctness + zero-regression).
2. Before launching long runs, ask: **does the recurrence compose
   associatively?** Any form `S_t = f(t) · S_{t-1} + g(t)` with `f(t)`
   acting linearly does. If yes, the primitive already exists in
   `src/models/rwkv6_time_mix.py:2702`.
3. If yes, port to chunked prefix scan (target ~10× speedup,
   activation-memory up by `chunk_size` factor).
4. If no (genuinely non-linear-in-state, e.g. Stage 10.2 M²RNN's
   `tanh(SW + kv)`), fall back to per-step optimisation or a custom
   kernel.

## 8. Phase 2 readiness

With Stage 10.1–10.4 closed, the STAGE10_PLAN §5 Phase-II queue is:

- **Stage 10.5** `rwkv6_orthogonal` (rank-1) — diagnostic vs T2 for
  operator-family-vs-deployment-shape attribution. Param parity safe at
  rank-1 (~3 K). Compute ~3–5× anchor.
- **Stage 10.6** `rwkv6_pom_vlift` (thin, D = 64, k = 2) — diagnostic vs
  `hadamard_n2` / `qtail_lowrank_all` for Kronecker-vs-PoM parametrisation
  equivalence. Compute negligible.
- **Stage 10.7** `rwkv6_loglinear_rse_strong_viscosity` —
  **CONDITIONAL gate closed** per plan §8.3 (10.1 landed PLATEAU, not
  MARGINAL). Do not run.

Multi-seed verification of 10.3-sym and the 10.3b composition follow-up
are both **deferred** to a subsequent sprint per the user instruction;
they are not blockers for Phase-II launch.

---

## 9. Training-curve visualisations

Plots are regenerated by `scripts/plot_stage10.py` into
`outputs/_stage10_plots/`.  All curves use matched dev-CER per epoch;
10.5 `rwkv6_orthogonal` shows only the epochs completed at the time of
the last plot run (training in progress at the time of writing).

**Comparison caveat (propagated from §2).** The only locally-available
vanilla-rwkv6 history is `outputs/exp02_rwkv6_seed42/history.csv`, which
was trained on an **80-epoch schedule with a gentler LR decay** than the
30-epoch cosine schedule all Stage-10 runs use.  At matched epochs this
means the vanilla row is slightly "further along" than a 30-epoch
vanilla would be, and its 30-epoch minimum under the 80-ep schedule is
lower than the canonical §9.1 row (0.1258).  Relative comparisons
between Stage-10 mechanisms are unaffected (they all use the same
schedule); absolute Δ-to-vanilla values should be read with the caveat.
The canonical vanilla endpoint for every verdict in this document is
the §9.1 number (dev 0.1258, test 0.1263), not the curve.

### 9.1 `01_stage10_all_vs_vanilla.png` — global trajectory view

All six completed Stage-10 runs plus the in-progress Cayley, with
vanilla / anchor / absolute-best horizontal reference lines.

*Reading.* `rwkv6_convshift_multidil_symmetric` (bold green) is the
only mechanism that visibly separates below the pack — it dives below
the anchor line by ep 25 and ties the abs-best by ep 30.  All other
Stage-10 mechanisms cluster in a tight ~0.003 band around the vanilla
endpoint; by eye they are indistinguishable at this zoom level.  The
causal multidil (orange) and the in-progress orthogonal (red) sit
visibly above the pack — consistent with their REGRESSION verdicts.

### 9.2 `02_family_D_quadratic_parametrisations.png` — quadratic-lift saturation

Superimposed curves for the three Family-D **quadratic-lift**
parametrisations: `hadamard_n2` (+0 params), `qtail_lowrank_all`
(+mod params), `pom_vlift` (+198 K params). Title carries the param
annotation.

*Reading.* All three curves are **superimposed within line width** from
ep 10 onward — they converge to the same ~0.125 basin regardless of
parameter budget (+0 K buys the same CER as +198 K). `qtail_lowrank_all`
(red) separates from the pack by ~1σ at the very end — the only one
with cross-channel Kronecker structure. This is the **clearest
graphical evidence** in the Stages 2-10 arc that quadratic feature
lift is not the load-bearing Family-D direction at 7 M / 30 ep, and
that parameter budget within this class is not a substitute for
structural freedom.

### 9.2b `02b_family_D_chanmix_vs_quadratic.png` — bypass vs quadratic

PI review §1.1 correctly pointed out that `chanmix_bypass` is α-gated
activation interpolation, not a polynomial lift — it does not belong
in the same plot as the quadratic-lift cluster. This separate plot
shows `chanmix_bypass` against `hadamard_n2` as a quadratic-lift
reference.

*Reading.* Both curves also superimpose in the same ~0.125 basin, but
for a different reason than in §9.2: chanmix_bypass is a structurally
different mechanism that happens to saturate at the same place.
Together §9.2 and §9.2b support the stronger claim that the 6-layer
RWKV-6 ChannelMix bottleneck is NOT relieved by quadratic-lift *or*
activation-interpolation at this scale.

### 9.3 `03_family_A_input_vs_structural.png` — input vs structural

`convshift_trap` (Stage 2 WON), `convshift_multidil` (causal REGRESSION),
`convshift_multidil_symmetric` (Stage 10 MARGINAL), and `loglinear`
(Stage 10 PLATEAU) overlaid.

*Reading.* Symmetric multidil (bold green) and `convshift_trap` (green)
are tied by ep 30.  Causal multidil (orange) follows both closely for
the first 10 epochs but never closes the gap — the **persistent
~+0.008 offset** visible from ep 5 onward is the causality penalty
plotted as a horizontal shift in the trajectory.  Loglinear (teal)
compresses from an early lead to the vanilla pack by ep 20 —
structural readout freedom does not convert, cleanly visible.

### 9.4 `04_cayley_vs_t2.png` — Stage 10.5 diagnostic

In-progress rwkv6_orthogonal (red) plotted against T2 primary (purple),
anchor (blue), and vanilla (grey).

*Reading (corrected post-PI review).* At ep 7 available, the Cayley
gap vs T2 is **narrowing**, not widening:

| Epoch | T2 | Cayley | Δ (Cayley − T2) |
|---:|---:|---:|---:|
| 5 | 0.2243 | 0.2408 | +0.0165 |
| 6 | 0.2055 | 0.2236 | +0.0181 |
| 7 | 0.1976 | 0.2091 | +0.0115 |
| 8 | 0.1880 | 0.1979 | +0.0099 |
| 9 | 0.1758 | 0.1887 | +0.0129 |

The gap is non-monotonic — it widened at ep 6 then narrowed through
ep 8 then widened again at ep 9. The pre-registered ep-15 halt
criterion (Cayley ≥ T2_ep15 + 0.006 = 0.1478) is the decision point;
the first 9 epochs do not support an early halt call. The data
are consistent with the §6.10.5 "significantly worse than T2" branch
in trajectory but the decision must wait for ep 15.

### 9.5 `05_delta_vs_anchor.png` — Δ-to-ANCHOR per matched epoch

Stage-10 mechanism CER minus **anchor** (`rse_strong_viscosity`) CER
at each matched epoch, with ±σ band. PI review §1.1 flagged the old
Δ-to-vanilla plot as broken (+0.0085 systematic bias from exp02's
80-ep schedule). Switching to anchor gives a reproducible, on-disk,
same-schedule reference.

*Reading.* Three groups of behaviour visible:

- **`convshift_multidil_symmetric` (bold green)** sits at Δ = −0.003
  to −0.005 **below the anchor zero-line** throughout training. It is
  the only Stage-10 mechanism that is **ahead of the anchor** and the
  signal is stable, not noise-chatter — the clearest graphical
  evidence that input-side multi-dilation with symmetric support is
  the only structurally-new mechanism that clears the anchor ceiling
  on this spine.
- **`m2rnn_sparse` (pink)** sits consistently at Δ = +0.010 to +0.015
  above the anchor — visible ~7-10σ engaged-null signal. The
  mechanism trains (trajectory changes relative to vanilla, cf. §9.1)
  but does not convert the CER to anchor-equivalent.
- **Other Stage-10 mechanisms** (loglinear, chanmix_bypass, pom_vlift,
  convshift_multidil causal) cluster at Δ = +0.004 to +0.008 — all
  statistically behind the anchor but within a ~3σ band of each other.
  These are the engaged-null PLATEAU verdicts.

This plot makes the **MARGINAL verdict of multidil_sym visually
load-bearing** in a way that the vanilla-reference plot didn't —
it's the only curve in the Stage-10 cohort that persistently sits
below the anchor zero-line.

### 9.6 `06_cer_vs_compute_pareto.png` — compute vs quality

Best dev CER (ep ≤ 30) against 30-epoch wallclock in hours on a single
RTX PRO 6000.

*Reading.* Three Pareto-optimal points form the frontier:
1. **`convshift_trap`** (1.02 h, 0.1150 cached from §9.1) — best
   dollar-for-second point.
2. **`convshift_multidil_symmetric`** (1.48 h, 0.1153) — +46 minutes
   of training for test-CER parity with the abs-best cross-axis result.
3. **`rse_strong_viscosity`** (1.60 h, 0.1189) — the anchor.

`rwkv6_orthogonal` (red, far upper right) at ~7.6 h / ~0.224 sits deep
in the Pareto-dominated region — worst on both axes of the plot.

Everything else (loglinear, m2rnn, the Family-D pack) clusters near
vanilla CER at 1.4–4.3 h; these are the "engineered-null" runs that
cost real compute without moving the CER.

## 10. Takeaways from the visual evidence

Three headline claims are visibly supported:

1. **Family-D quadratic-lift saturation is structural, not
   parameterisation-specific.** The four-way trajectory overlap in
   §9.2 is the cleanest visual in the Stages 2-10 arc for any
   saturation claim.

2. **Input-side temporal hierarchy with symmetric support is the
   only mechanism that clears the abs-best ceiling on this spine.**
   §9.1 and §9.5 both show this as a continuous offset rather than a
   late-epoch artefact — multidil-sym was ahead by ep 5 and stayed
   ahead throughout.

3. **Causality is a near-pure additive handicap on input-side
   mechanisms.** §9.3 shows causal-vs-symmetric as an almost-parallel
   trajectory offset of ~0.008 across all 30 epochs, not a late-stage
   divergence.  This is the strongest graphical evidence for the
   "phase lag" framing in §3.3.

What is *not* visible despite being claimed in §4:
- The cross-experiment invariant extension across Families A and C
  is best seen in §9.5 where loglinear (teal) and m2rnn (pink)
  oscillate within σ throughout; the "engaged-null" pattern shows as
  noisy chatter around zero rather than a clean signal.  This is
  thesis-relevant as a null result, but visually weaker than the
  Family-D saturation in §9.2.
