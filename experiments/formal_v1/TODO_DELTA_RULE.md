# TODO — Delta Rule Variants Across Mamba-2, RWKV, and Linear Attention

The delta rule is the single most universal "state update correction"
mechanism in the modern linear-attention / SSM / RWKV literature — it
appears independently in every architecture family, under different
names, and composes with nearly every other mechanism we use (decay,
gating, rotation, LUCID). This file catalogs the mathematical
reformulations, maps each to our codebase, and lists the concrete
experiments worth queueing.

**Related docs:**
- [TODO_FUTURE_IDEAS.md](TODO_FUTURE_IDEAS.md) — broader future-work parking lot
- STAGE5_RESULTS.md — Stage 5 pre-registered thresholds apply
- [RESULTS.md](RESULTS.md) §Chunked — prior delta runs & numbers
- `src/models/mechanisms/delta_rule.py` — current implementation
- `src/models/lion_attention.py` `lion_attention_with_delta` — LION delta kernel
- `src/models/rwkv6_time_mix.py` `_chunked_wkv(self_reg=True)` — RWKV-6 RKHS delta

---

## 1. Why delta rule is a unifying mechanism

In every linear attention / SSM / RWKV variant, the state update looks like:

$$S_t = \mathcal{T}_t(S_{t-1}) + \text{write}(k_t, v_t)$$

where $\mathcal{T}_t$ is some transition operator. The delta rule is the
specific choice

$$\boxed{\;\mathcal{T}_t(S) = S\,(I - \beta_t\,k_t k_t^\top)\;}$$

which is mathematically the **Householder reflection** that projects $S$
away from the subspace spanned by $k_t$ before writing the new
association $(k_t, v_t)$. Equivalently, the composite update

$$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + v_t k_t^\top
      = S_{t-1} + \beta_t(v_t - S_{t-1} k_t)\,k_t^\top$$

is **one step of SGD on the local least-squares loss**
$\tfrac12\|S k_t - v_t\|^2$ with learning rate $\beta_t$. This is the
"erase the current prediction, then write the target" reading — the
mechanism removes stale associations before adding new ones, which
prevents a fixed-size state from saturating with redundant content.

**Why it shows up everywhere:**
- A fixed-size matrix state $S \in \mathbb{R}^{d\times d}$ can store at
  most $d$ linearly independent key→value associations. Without an
  erase step, an RNN-style linear recurrence accumulates $O(T)$ writes
  into $O(d^2)$ slots, which over-saturates the state for $T \gg d$.
- The Householder form keeps the erase step rank-1, preserving
  the semiseparable mask structure needed for subquadratic training.
- The least-squares interpretation gives a principled way to pick
  $\beta_t$ (step size ≈ inverse of $\|k_t\|^2$).

---

## 2. Connection to LUCID

LUCID and delta rule address the **same problem** — redundancy in the
state / attention output — but at **different stages** of the
computation. They are not redundant; they can compose.

| | Operates on | When | Cost |
|---|---|---|---|
| **Delta rule** | State $S$ | *During* each write, streaming | O(1) per token |
| **LUCID** | Attention matrix $A = QK^\top$ or output $AV$ | *After* the scan, one-shot per chunk | O(chunk_size²) |

### LUCID preconditioner (Proposal A §3.4, CORRECTED)

$$P = I + \exp(\tau \cdot K_{\text{norm}} K_{\text{norm}}^\top),
\qquad Y = P^{-1}(A V)$$

decorrelates the attention output by inverting the Gram matrix of
normalized keys. This is a **one-shot** whitening of the output — you
accumulate the full attention first, then decorrelate in the output
space.

### Delta rule as streaming-LUCID

$$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + v_t k_t^\top$$

decorrelates *online* — each write removes the component along $k_t$
first. After $T$ steps, the cumulative effect is approximately
equivalent to applying a diagonal preconditioner on the key-Gram
matrix, truncated to one step of power iteration per token.

### Why they compose

- LUCID operates in *attention-output space* (post-$AV$).
- Delta rule operates in *state space* (pre-readout).
- **No algebraic cancellation** between them — a LUCID preconditioner
  applied to a delta-state readout would further whiten whatever
  residual correlation survives the streaming erase.
- The RKHS self-regulation variant we already have (`_chunked_wkv
  self_reg=True`) is literally LUCID + delta: the key is
  normalized per-RKHS, then used both in the delta erase and in the
  implicit preconditioning. Under-tested as a combined mechanism.

### Prior empirical evidence

- `rwkv6_lucid` (LUCID alone, causal): 0.1216 dev CER
- `rwkv6_lucid_sr` (LUCID + RKHS self-reg ≈ delta on state): 0.1483 dev CER — *worse* than LUCID alone. The self-reg variant regressed, suggesting the combined mechanism needs careful tuning or that our specific RKHS formulation conflicts with LUCID's preconditioner.
- `lucid_exp04_lion_lucid`: 0.1074 dev CER — LUCID on LION works moderately.
- `lion_delta` (causal-only delta on LION): 0.1373 — underperformed baseline LION (0.0711) by a lot.

**Takeaway:** every prior delta run we have is a **negative result**.
Either the mechanism genuinely doesn't help at our scale/budget, or our
specific parameterization is buggy/sub-optimal. This is worth
diagnosing before adding more variants.

---

## 3. Where delta lives in each architecture family

### 3.1 Linear Attention — DeltaNet and Gated DeltaNet

**DeltaNet** (Schlag, Irie, Schmidhuber 2021, *Linear Transformers are
Secretly Fast Weight Programmers*):

$$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + v_t k_t^\top, \qquad o_t = S_t q_t$$

**Gated DeltaNet** (Yang et al. 2024a):

$$S_t = \alpha_t\,S_{t-1}(I - \beta_t k_t k_t^\top) + v_t k_t^\top$$

where $\alpha_t \in (0,1)$ is a scalar data-dependent decay.
Yang et al. (2024b) show a parallel form via compact WY representations
of products of Householder matrices — this is the bridge to Log-Linear
Attention's framework (see TODO_FUTURE_IDEAS.md §Log-Linear).

**Mapping to our code:** not currently implemented as a standalone
backbone. Would require a new `linear_attn_delta` module; not high
priority since RWKV-6 subsumes this family.

### 3.2 Mamba-2 — Gated DeltaNet IS "Mamba-2 with Householder transition"

Mamba-2's SSD (Dao & Gu 2024):

$$S_t = \alpha_t S_{t-1} + B_t x_t$$

Gated DeltaNet (Yang et al. 2024a):

$$S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + v_t k_t^\top$$

The only structural difference is the $(I - \beta_t k_t k_t^\top)$
Householder transition. In the `P = A⊙M, O = PV` framework (Log-Linear
Attention §2), both sit at the same level of the taxonomy, differing
only in whether $A = QK^\top$ (Mamba-2) or $A = \mathcal{T}_K(QK^\top)$
(DeltaNet). So any delta-rule variant immediately yields a Mamba-2
variant.

**Mapping to our code:** our `mamba2` and `mamba2_lion` backbones don't
have a delta transition. Adding it is a well-defined change in
`src/models/mamba2_kernels.py`: replace the scalar $\alpha_t$
transition with $\alpha_t(I - \beta_t k_t k_t^\top)$. Stage-6 results
show `mamba2_lion` already hits 0.1018 test CER — adding delta gives a
"Gated DeltaNet LION" which is exactly the thing Yang et al. test.
Estimated param cost: +$\beta$ LoRA ($\sim$3K params).

### 3.3 RWKV — three different delta-rule paths, all ours

| Backbone | Delta flavor | Where | Tested? |
|---|---|---|---|
| `rwkv6_delta` | Classic DeltaNet update on recurrent scan | `rwkv6_time_mix._forward_recurrent` + `delta_rule.py` | **No** (registered but never run) |
| `lion_delta` | Causal-only delta correction on LION attention matrix | `lion_attention.lion_attention_with_delta` | Yes — 0.1373 (poor) |
| `rwkv6_lucid_sr` | RKHS self-regulation (state-space delta with K-normalization) | `_chunked_wkv(self_reg=True)` | Yes — 0.1483 (poor) |

So the RWKV family has the most diverse delta implementations in our
codebase. Two out of three tested are negative. The untested one
(`rwkv6_delta`) is the one closest to canonical DeltaNet form.

---

## 4. Reformulation catalog — open variants to try

Ordered roughly by math complexity / expected delta from vanilla.

### A. Householder parameterization

**A1. Key normalization mode.**
- `||k|| = 1` (F.normalize): canonical Householder; $\beta=2$ makes it a
  true reflection. **Current choice** (delta_rule.py L56).
- `||k||` unnormalized: $\beta_t$ absorbs the scale; data-dependent step
  size. Tested in Yang et al. 2024b.
- Learnable per-head scale on $k$ before normalization (the `k_k`
  parameter in our `delta_rule.py` L31 — init 0.85).

**A2. Asymmetric delta: $(I - \beta_t k_t q_t^\top)$ vs $(I - \beta_t k_t k_t^\top)$.**
Replacing the second $k_t$ with $q_t$ makes the erase depend on the
*query* rather than the key. Theoretically unusual (Householder-like
but not orthogonal), and the semiseparable structure of the resulting
mask needs re-derivation. Not tried in literature. **Speculative** —
only worth attempting if A1 and B variants don't crack the baseline.

**A3. Multi-step delta / compact WY.**
Apply $n$ Householder reflections per token:
$(I - \beta_t^{(1)} k_t^{(1)} k_t^{(1)\top}) \cdots
(I - \beta_t^{(n)} k_t^{(n)} k_t^{(n)\top})$

with $n$ auxiliary keys per token (e.g. $k_t^{(j)} = W_j x_t$). Yang et
al. (2024b) use compact WY to parallelize the standard $n=1$ case; for
$n=2$ the same machinery extends at 2× the key projection cost. Could
give delta more expressive power (rank-$n$ erase instead of rank-1)
without giving up subquadratic training.

### B. Learning-rate / gate parameterization

**B1. β global scalar vs per-head vs per-channel.**
Our current `k_a` + LoRA (`a0`, `a1`, `a2`) gives per-head-per-channel
data-dependent $\beta$ (delta_rule.py L33–40). Literature-default is
per-head scalar. Ablate to see whether the expressive parameterization
is helping or hurting.

**B2. β_t derivation.**
- Sigmoid of LoRA (our current form): $\beta_t = 2\sigma(a_0 + a_1 x_t)$
  ranges in $(0, 2)$. Bounded.
- Softplus: unbounded positive. Matches optimal SGD step
  $\beta = 1/\|k\|^2$ form.
- Data-independent: $\beta$ learned scalar per-head. Minimal.

**B3. Warm-start β.**
Init $\beta \equiv 0$ gives zero-regression at step 0 (same contract as
viscosity and β_hadamard / β_qtail in Stage 6). Our current
`a0 = ones` initialization does **not** have this property — it starts
with delta active at full strength, which may be why prior runs
underperform. **First thing to try on `rwkv6_delta`: init
$\beta_t \equiv 0$ and let SGD grow it.**

### C. Second-order / momentum / TTT-style variants

**C1. Momentum delta (Polyak-heavyball analog).**
Keep a running "update direction" $m_t = \mu m_{t-1} + (v_t - S_{t-1} k_t) k_t^\top$
and apply $S_t = S_{t-1} - \beta_t m_t$. Costs +1 extra state per head.

**C2. Second-order (Newton-style) delta.**
Replace one SGD step with one Newton step: $\beta_t = 1/(k_t^\top k_t)$
(exact least-squares), giving the *projected* target $v_t$. Equivalent
to $\beta_t = 1/\|k_t\|^2$ which is exactly Yang et al. 2024b's
"unnormalized key with adaptive β" recipe. Free upgrade if it works.

**C3. TTT / ATLAS: multi-step test-time gradient descent.**
Behrouz et al. (2024, 2025) — "Titans" and "ATLAS" — generalize the
delta update to multiple gradient steps per token, computing $\hat v$
iteratively before the write. Expensive but theoretically more
expressive. Only worth attempting if B3 (zero-init $\beta$) resolves
the current negative results.

### D. Composition with decay / gating / rotation

**D1. Delta-then-decay vs decay-then-delta.**
- "Decay-then-delta": $S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + v_t k_t^\top$ — Gated DeltaNet form.
- "Delta-then-decay": $S_t = \alpha_t \big[S_{t-1}(I - \beta_t k_t k_t^\top)\big] + v_t k_t^\top$ — same as above due to commutativity.
- "Write-then-decay-then-erase": different, and rarely considered.

Low value; the literature has settled on Gated DeltaNet form. Skip.

**D2. Delta × RSE composition.**
RSE's transition is $z_t = e^{-\lambda_t + i\theta_t}$ on each complex
block. Delta could be applied **within each 2×2 block**: reflect the
block off the direction $k_t^{(b)} = k_{2b} + i k_{2b+1}$ before
applying the rotation-decay. Formally:
$$c_{t,b} = (1 - \beta_t |k_{c,t,b}|^2)\,z_{t,b}\,c_{t-1,b} + k_{c,t,b}\,v_t$$
(the reflection in $\mathbb{C}$ is the conjugate-magnitude projection).

Stacks on `rwkv6_rse` and, by extension, `rwkv6_rse_strong_viscosity`
(current causal best, 0.1177 test CER). Would require a new scan
helper `_forward_recurrent_rse_delta` mirroring the P²-RSE pattern
(two scans, but with a delta-mode second pass rather than a
complementary-phase second pass).

**D3. Delta × Log-Linear composition.**
Log-Linear Attention maintains O(log T) bucket states. Delta could be
applied to each bucket state independently. Gated DeltaNet → Log-Linear
Gated DeltaNet is exactly the composition Guo et al. 2025 already test,
with positive results (+8/9 NIAH metrics). Would transfer to RWKV-6
only after the Log-Linear RWKV-6 implementation in
TODO_FUTURE_IDEAS.md §Log-Linear.

### E. Normalization / spectral variants

**E1. Spectral-normalized delta.**
Enforce $\|I - \beta_t k_t k_t^\top\|_2 \le 1$ by clipping $\beta_t \le
2/\|k_t\|^2$. Guarantees the state can't grow under delta. Literature
default when $\|k\|=1$ and $\beta \in (0,2)$ — which our current
parameterization satisfies. So we already have this.

**E2. Delta with RMSNorm on state.**
Apply RMSNorm to $S$ after the delta update. Related to the Stage-6
EXPRESSIVENESS paper finding that the post-WKV norm matters; delta +
RMSNorm may stack cleanly.

---

## 5. Diagnostic: why our prior delta runs underperformed

Three testable hypotheses before adding more variants:

**H1. β initialization is wrong.**
Current `a0 = ones` starts $\beta_t \approx \text{sigmoid}(1)\cdot 2
\approx 1.46$ — full-strength delta from step 0. This *destroys* the
randomly-initialized state before any useful associations are written.
Fix: init $a_0 = -3$ so $\beta_t \approx 2\sigma(-3) \approx 0.095$, or
$a_0 = -\infty$ giving $\beta \equiv 0$ and let the LoRA grow it.

**H2. Key normalization mode is wrong for this scale.**
`k_k = 0.85` (L31) applies a learnable scale *before* normalization.
At small init this rescales $k$ toward its direction before
normalizing. Ablate to $k_k = 1.0$ and also try removing the
normalization entirely (let $\beta$ absorb the scale).

**H3. Delta + GroupNorm interaction.**
Delta erases state content aligned with $k_t$; the GroupNorm on the
output then re-normalizes the readout. These two normalizations may
fight each other. The Stage-6 `rwkv6_rmsnorm` swap (per-head L2 norm
instead of GroupNorm) may make delta more stable — because L2 norm is
better matched to the inner-product geometry that delta operates in.

**Minimum-cost diagnostic:** run `rwkv6_delta_warmstart` with H1 fix
(β_init = 0) on pure causal. Expected improvement alone should take us
from 0.14ish back to at least baseline 0.126; if not, delta on RWKV-6
has a deeper issue.

---

## 6. Experiment queue (ordered by cost + signal value)

### Tier 1 — diagnostic, cheap, unblocking

1. **`rwkv6_delta_warmstart`** — zero-init $\beta$, nothing else changed.
   Runtime same as `rwkv6_delta` baseline (~110 s/ep × 30 ≈ 55 min).
   Tests H1 above. If this doesn't at least match baseline RWKV-6
   (0.1258), the delta path has a deeper bug or is genuinely not
   helpful on this task/scale.

2. **`rwkv6_delta_rmsnorm`** — zero-init $\beta$ + RMSNorm readout (from
   Stage 6). Tests H1 + H3 together. Should land between
   `rwkv6_rmsnorm` and `rwkv6_delta_warmstart` if both mechanisms help.

### Tier 2 — β parameterization ablation

3. **`rwkv6_delta_scalar`** — per-head scalar $\beta$, no data-dependence.
   Clean test of whether the LoRA parameterization is helping or
   hurting.

4. **`rwkv6_delta_newton`** — $\beta_t = 1/(\|k_t\|^2 + \epsilon)$ with
   unnormalized $k$. Implements the exact one-step least-squares (Yang
   et al. 2024b variant). No new params beyond a scalar epsilon.

### Tier 3 — compositional

5. **`rwkv6_rse_delta`** — RSE × delta in complex 2×2 block form
   (§D2). Stacks on `rwkv6_rse_strong_viscosity` (current best). If
   Tier 1 clears, this is the natural composition. Estimated param
   cost: +~3K (β LoRA in complex form).

6. **`lion_delta_warmstart`** — re-run `lion_delta` with zero-init β.
   Cheap to test because LION runs at ~70 s/ep.

7. **`mamba2_lion_delta`** — Gated DeltaNet LION, i.e. Mamba-2 + delta.
   Port from `mamba2_lion` by replacing the scalar-λ transition with
   $\alpha_t(I - \beta_t k_t k_t^\top)$. Transfer path for the main
   thesis claim across architecture families.

### Tier 4 — higher-order / more speculative

8. **Momentum delta (§C1)** — on RWKV-6 only, only if Tier 1–2 crack.

9. **Multi-step Householder (§A3)** — implemented via repeat
   application of `delta_rule.DeltaRuleParams`. Only if everything
   above caps out.

---

## 7. Open research questions

- **Does the delta rule genuinely help at our 7M-param, 500-frame,
  LibriSpeech scale?** Every prior run we have is a negative result;
  two diagnostic hypotheses (H1, H3) remain untested. The answer could
  genuinely be "no" — delta's theoretical motivation is state
  saturation under long sequences, and at $T=500 \ll d^2 = 4096$, RWKV's
  baseline state has plenty of headroom.
- **Is RWKV's LUCID self-reg (`rwkv6_lucid_sr`) a mis-specified delta?**
  The 0.1483 result is much worse than either LUCID-alone (0.1216) or
  baseline (0.1258). The RKHS formulation may be interacting badly
  with LUCID's preconditioner — or it may be the unfortunate case that
  the two mechanisms, both of which aim at decorrelation, are
  double-counting.
- **Does delta transfer between causal and bidirectional?** `lion_delta`
  uses the causal-only correction (draft's bidirectional version was
  wrong per CLAUDE.md). The asymmetry between causal and bidirectional
  delta may explain why `lion_delta` underperforms baseline LION so
  severely — the correction only fixes half the attention.
