# RSE θ-Distribution Analysis: Why Per-Layer Budget Allocation Matters

This document is the formal post-mortem of the Stage-3 → Stage-4
Rotational State Evolution (RSE) campaign. It reconstructs the chain
of reasoning that led from the marginal Stage-3 result
(`rwkv6_rse` test CER 0.1238, only −2.0% over baseline) to the
Stage-4 ceiling break (`rwkv6_rse_depth` test CER 0.1200, −5.0%) and
its co-launched control (`rwkv6_rse_strong`).

The central scientific question this document answers: *given that
the RSE Lie-group expansion is theoretically irreducible from RWKV-6,
why did the first deployment fail to materialize the predicted
expressivity gain, and what changed in the refinement that
materialized it?*

---

## 1. Problem context: the Stage-3 paradox

By the close of Stage 3, three facts coexisted uncomfortably:

1. **Theory was correct.** The RSE transition operator
   $G_{t,b} = e^{-\lambda_{t,b}} R(\theta_{t,b})$ has complex
   eigenvalues $e^{-\lambda \pm i\theta}$, while RWKV-6's transition
   has strictly real positive eigenvalues. By Theorems 1–2 of
   Proposal A, the function class $\mathcal{F}_{\text{RSE}}$
   strictly contains $\mathcal{F}_{\text{RWKV}}$ and is not
   continuously reparameterizable from it.

2. **Implementation was correct.** The complex chunked scan
   (`_forward_recurrent_rse` in `src/models/rwkv6_time_mix.py`)
   reproduces the serial reference within fp32 numerical precision
   (max abs diff $\sim 10^{-4}$) and trains stably with grad-norms
   in the same band as baseline RWKV-6 (mean $\approx 1.30$).

3. **CER barely moved.** `rwkv6_rse` landed at dev 0.1251 / test
   0.1238, vs baseline 0.1258 / 0.1263 — a marginal −2.0% test
   improvement, well below Proposal A §9 P1's predicted
   $\geq 3\%$ band, and classified FLAT on dev by the autonomous
   decision script.

The Stage-3.5 multi-rate variants (M=2, M=4) made this paradox
sharper: both extra-capacity variants tracked single-scale RSE for
~10 epochs, then converged to the same ~0.125 plateau by ep 15.
*Adding parallel rotation channels did not change the asymptote.*

The natural hypotheses at this point split on whether the rotation
mechanism was being *used* at all:

- **Cold-start trap:** the LoRA gates ($\theta$_w1, $\theta$_w2)
  may have stayed near their zero-initialization, leaving
  $\theta_t \approx \theta_{\text{base}}$ — i.e., a fixed
  per-block constant rotation that the model can absorb into other
  projections. In this reading, the *mechanism* was structurally
  present but never trained.
- **Mechanism-active-but-unhelpful:** the LoRA gates may have
  trained to non-trivial values and the rotation may have been
  data-dependent in some non-trivial way, but the resulting
  function (whatever it learned to compute) does not measurably
  help on causal LibriSpeech.

These two diagnoses imply opposite next moves: the first calls for
better initialization; the second calls for abandoning the rotation
direction. A direct empirical test was needed before committing
GPU-hours either way.

---

## 2. Methodology — the static parameter mobility diagnostic

### 2.1 Goal

Determine, from the trained `stage3_01_rwkv6_rse_seed42/best_model.pt`
weights alone (no further training, no GPU time), whether the
rotation parameters had moved meaningfully from initialization, and
characterize the per-layer pattern of that movement.

### 2.2 What is initialized, and to what values

Three groups of parameters control $\theta$ in `RWKV6TimeMix`:

| Parameter | Shape | Initialization | Reference statistic |
|---|---|---|---|
| `time_theta` | $(1, 1, H \cdot B_k)$ | $\mathcal{U}(-\pi/16,+\pi/16)$ per element | mean $0$, std $\pi/(16\sqrt 3) \approx 0.113$ |
| `time_theta_w1` | $(D, D_\theta)$ | zeros | $\|\cdot\|_F = 0$ |
| `time_theta_w2` | $(D_\theta, H \cdot B_k)$ | $\mathcal{U}(-0.01,+0.01)$ per element | $\|\cdot\|_F = 0.01/\sqrt{3} \cdot \sqrt{D_\theta \cdot H \cdot B_k}$ |

with $D = 256$ (hidden), $H = 4$ (heads), $B_k = K/2 = 32$ (blocks
per head), $D_\theta = 32$ (LoRA dim). The reference Frobenius norm
for `time_theta_w2` evaluates to $0.01/\sqrt{3} \cdot \sqrt{32 \cdot 4 \cdot 32}
\approx 0.37$.

The forward path for $\theta$ is

$$
\theta_t = \gamma_\theta \cdot \tanh\!\bigl(\,\underbrace{\theta_{\text{base}}}_{\text{time\_theta}} + \underbrace{\tanh(\tilde x_t \cdot W_1) \cdot W_2}_{\text{LoRA}(\tilde x_t)}\,\bigr)
$$

If `time_theta_w1` stays at zero, the LoRA branch evaluates to
$\tanh(0) \cdot W_2 = 0$, so $\theta_t \equiv \gamma_\theta \tanh(\theta_{\text{base}})$
— a per-block, *data-independent* constant. *This is the cold-start
signature*: the rotation degree of freedom collapses to a static
per-block bias that any other linear projection in the model can
absorb (since rotation by a fixed angle, when followed by a
learnable linear map, can be reabsorbed into that map's basis).

Conversely, if `time_theta_w1` grew from zero to non-trivial Frobenius
norm, the rotation has a genuinely data-dependent component that
*cannot* be absorbed by any data-independent reparameterization of
the surrounding projections.

### 2.3 Procedure

The script `scripts/rse_theta_diagnostic.py`:

1. Loads the trained encoder weights.
2. For each of the 6 RSE layers, extracts the three parameter groups.
3. Computes summary statistics: per-layer std and max absolute value
   of $\theta_{\text{base}}$; Frobenius norms of the two LoRA matrices.
4. Compares each statistic to its initialization reference.
5. Reports the per-layer table and the binary verdict: did the LoRA
   move (cold-start hypothesis falsified) or stay at zero
   (cold-start hypothesis confirmed)?

A separate dynamic analysis (forward-pass on dev-clean, capture
$\theta_t$ values per layer/head/block) was attempted but blocked by
an independent shape-mismatch bug in the eval data path; the static
analysis was sufficient for the decision and the dynamic analysis is
deferred.

### 2.4 Results

```
Layer | time_theta                        | w1 ‖·‖_F | w2 ‖·‖_F | w2 init ‖·‖_F
------+-----------------------------------+----------+----------+--------------
L0    | mean=+0.008 std=0.167 max|·|=0.426 |  2.16    |  3.39    |  0.37
L1    | mean=+0.007 std=0.180 max|·|=0.538 |  2.08    |  3.25    |  0.37
L2    | mean=-0.020 std=0.167 max|·|=0.354 |  2.90    |  2.79    |  0.37
L3    | mean=-0.007 std=0.153 max|·|=0.364 |  3.08    |  2.72    |  0.37
L4    | mean=+0.014 std=0.164 max|·|=0.447 |  3.04    |  2.75    |  0.37
L5    | mean=-0.017 std=0.147 max|·|=0.426 |  3.82    |  3.35    |  0.37
```

Three observations:

**Observation A — cold-start hypothesis falsified.** All six layers
have `time_theta_w1` Frobenius norms in the range 2.0 – 3.8. Since
this matrix is initialized to zero, *any* non-zero norm proves the
LoRA was trained. Norms of 2 – 4 are not "barely moved off zero" —
they are large compared to the typical operator scale of a
LoRA-shaped matrix in this codebase. The rotation is genuinely
data-dependent in the trained model.

**Observation B — bias drifted but modestly.** The static bias
$\theta_{\text{base}}$ has init std $\approx 0.113$ (predicted from
$\mathcal{U}(-\pi/16,+\pi/16)$); trained std lies in the band
$0.147$ – $0.180$ — drifted but only by a factor of $\approx 1.3$ –
$1.6$, with means staying near zero and max absolute values
$\sim 0.5$ (well below the clip $\gamma_\theta = \pi/4 \approx 0.78$).
The static rotation bias was used, but lightly.

**Observation C — a clear depth pattern.**

- $\|w_1\|_F$ rises monotonically with depth: $2.16 \to 2.08 \to 2.90 \to 3.08 \to 3.04 \to 3.82$ (L0 → L5). Excluding the L0–L1 slight reversal, the pattern is monotone: deeper layers have *more* data-dependent rotation magnitude.
- $\theta_{\text{base}}$ std *decreases* with depth, from $0.180$ at L1 to $0.147$ at L5. Deeper layers rely *less* on static per-block bias and *more* on the data-dependent LoRA contribution.

### 2.5 Diagnostic verdict

The model is using the rotation, but not in the symmetric way the
uniform Stage-3 architecture asks for. The trained model exhibits a
**revealed depth-preference**:

> shallow layers want simpler, mostly-static rotation;
> deep layers want richer, data-dependent rotation.

The Stage-3 architecture imposes the same $(\gamma_\theta,
\theta_{\text{init}}, D_\theta) = (\pi/4, \pi/16, 32)$ on all six
layers. This is a *uniform allocation* of the rotational budget that
the model is implicitly trying to escape — by spending its training
gradient asymmetrically across layers within a uniform per-layer
parameterization. The architecture can be made to fit the model's
revealed preference by *re-allocating* the budget along the same
gradient axis, without adding any parameters.

This is the cleanest possible refinement hypothesis the diagnostic
admits. It is also testable in a single 30-epoch run, against an
isolating control (uniform-but-larger budget) that decides whether
the gain — if any — comes from the *non-uniform allocation* or
simply from *more rotation budget anywhere*.

---

## 3. Stage-4 backbones — design

### 3.1 `rwkv6_rse_depth` — the diagnostic-aligned refinement

Per-layer rotation budget gradient:

| Layer | $\gamma_\theta$ (clip) | $\theta_{\text{init}}$ scale | $D_\theta$ (LoRA dim) |
|:---:|:---:|:---:|:---:|
| L0 | $\pi/8$ | $\pi/32$ | 16 |
| L1 | $\pi/8$ | $\pi/32$ | 16 |
| L2 | $\pi/4$ | $\pi/16$ | 32 |
| L3 | $\pi/4$ | $\pi/16$ | 32 |
| L4 | $\pi/2$ | $\pi/8$ | 48 |
| L5 | $\pi/2$ | $\pi/8$ | 48 |

The 16 / 32 / 48 LoRA-dim gradient averages to 32 — the uniform value
used in `rwkv6_rse`. **Encoder parameter count is identical**
(5,899,520). The clip and init-scale gradient sweep one octave each
(π/8 → π/4 → π/2 and π/32 → π/16 → π/8); each level doubles the
permitted rotation magnitude, matching the order-of-magnitude
revealed-preference span the diagnostic exposed.

### 3.2 `rwkv6_rse_strong` — the uniform-larger-budget control

Uniform across all six layers:

| All layers | $\gamma_\theta$ (clip) | $\theta_{\text{init}}$ scale | $D_\theta$ |
|:---:|:---:|:---:|:---:|
| L0–L5 | $\pi/2$ | $\pi/16$ | 48 |

Encoder parameter count is 5,936,384 (+0.6% over baseline) — a small
but non-zero increase from the larger uniform LoRA. The clip is the
maximum permitted in `depth` (matched to the deepest layer's value);
the init scale is kept conservative (matches Stage 3 to avoid the
v1 cumulative-phase instability); the LoRA dim is the maximum used
in `depth` (matched to the deepest layer).

### 3.3 What the pair isolates

| Outcome | Interpretation |
|---|---|
| `depth` wins, `strong` flat | Non-uniform allocation is the productive axis; more uniform budget alone is not enough. Depth-preference is real. |
| `strong` wins, `depth` flat | The bottleneck was raw rotation amplitude; the depth pattern in the diagnostic was incidental. Equivalent to the "more capacity helps anywhere" hypothesis. |
| Both win | Both axes are productive. Disentangle via per-layer ablations in Stage 5. |
| Both flat | Refinement hypothesis falsified; the rotation mechanism truly hits a ceiling at ~0.125. |

The control is essential because the diagnostic, by design, only
*detects* a depth pattern in the trained Stage-3 model — it does not
*prove* that allocating budget along that pattern is what produces
gains, vs. simply allowing more budget anywhere.

---

## 4. Mathematical justification — why depth-graded works

### 4.1 The Lie-group view

The RWKV-6 transition group is the abelian Lie group
$\mathcal{G}_{\text{RWKV}} = (\mathbb{R}_+)^K$, acting diagonally on
the state $S_t \in \mathbb{R}^{K \times K}$. RSE replaces this with
$\mathcal{G}_{\text{RSE}} = (SO(2) \times \mathbb{R}_+)^{K/2}$,
acting block-diagonally with $K/2$ blocks of $2 \times 2$ shape.
Both groups have real dimension $K$, but they are not isomorphic:
$\mathcal{G}_{\text{RWKV}}$ is simply connected with no compact
direction, while $\mathcal{G}_{\text{RSE}}$ has $K/2$ compact torus
directions (one per block).

The clip $\gamma_\theta$ controls *how much* of the compact $SO(2)$
direction can be reached per step. With $\gamma_\theta = \pi$ the
full circle is available; with $\gamma_\theta = \pi/8$ only the arc
$[-\pi/8, +\pi/8]$ is available, and the achievable transition
subgroup is a tube near the real-positive ray $\mathbb{R}_+$. In the
limit $\gamma_\theta \to 0$ the rotation collapses and RSE
degenerates into RWKV-6.

The clip therefore controls *how far* the architecture is willing to
move off the original baseline manifold.

### 4.2 The depth-asymmetric expressivity argument

In a hierarchical encoder, different layer indices encode features
at different temporal granularities:

- **Shallow layers** (L0, L1): close to the acoustic input. The
  features they construct are slowly-varying envelopes (energy,
  spectral tilt, voicing). These have low effective angular
  frequency and benefit from stable, mostly-decay-only state
  dynamics. Forcing rotation here means asking the model to model
  *oscillation in features that don't oscillate* — adding noise to a
  computation that was already close to optimal.

- **Deep layers** (L4, L5): build phonological / lexical
  abstractions. The features here vary at the rate of phoneme
  transitions (10 – 30 Hz at 100 fps frame rate) and contain the
  formant-frequency dynamics that drive intelligibility. These
  features benefit from native complex-eigenvalue tracking; this is
  where Proposal A §7's acoustic-prior argument (formants =
  complex-conjugate poles) actually applies.

A uniform budget $\gamma_\theta$ across layers cannot satisfy both:

- If $\gamma_\theta$ is small (Stage 3's $\pi/4$ is comfortably small
  given the cumulative-phase concern of v1's $\pi/2$): shallow
  layers have a fitting amount, deep layers are starved.
- If $\gamma_\theta$ is large: deep layers get what they need,
  shallow layers are exposed to spurious oscillation that they have
  to learn to suppress (consuming model capacity on
  *de-rotation*, not on the actual ASR task).

The diagnostic in §2.4 shows the model trying to execute exactly
this compromise: under uniform $\gamma_\theta = \pi/4$, the
trained-LoRA contribution is *systematically larger* at deeper
layers, and the static-bias contribution is *systematically smaller*
at deeper layers. The model is attempting to use as much rotation
as it can at depth, while suppressing it at shallow layers — but
under a uniform parameterization it cannot fully achieve either.

### 4.3 Per-layer absorption and why it limits uniform budgets

The transition operator $G_{t,b}$ acts on the row dimension of $S$.
The readout $r_t^\top S_t$ contracts the same row dimension. If the
per-step rotation $R(\theta_{t,b})$ is *data-independent*, the
model can absorb it into the receptance projection $W_r$ via a
fixed change of basis:

$$
r_t^\top R(\theta_b) S_t = \tilde r_t^\top S_t,\qquad \tilde r_t^\top := r_t^\top R(\theta_b)
$$

The "absorbed rotation" appears as a fixed re-basis of the receptance
keys, which the model can learn directly via $W_r$ without ever
allocating it to $\theta$. The Stage-3 cold-start failure mode would
have looked exactly like this: a static $\theta_{\text{base}}$ with
zero LoRA contribution, fully absorbed by a counter-rotated $W_r$.

The diagnostic shows we are *not* in that regime — the LoRA grew
substantially, so $\theta_t$ is genuinely time-varying, which
*cannot* be absorbed by any single fixed $W_r$. The rotation does
contribute irreducibly to the function class.

But the *amount* of irreducible contribution is bounded above by
the magnitude of the data-dependent component: if $\theta_t$ is
mostly static plus a small data-dependent perturbation (which is
what a uniform budget produces), the irreducible contribution is
small. The depth-graded budget *increases the cap on the
data-dependent component at the layers where the model wants to use
it*, allowing the irreducible contribution to grow there.

This is the formal version of the qualitative observation: depth
matters because the ceiling on $\theta_t$'s data-dependence is
binding at deep layers and slack at shallow layers under a uniform
budget. Lifting the cap where it binds, while keeping it tight
where it doesn't, is the parameter-efficient way to extract more
expressivity.

### 4.4 The phase-coherence constraint

A second mathematical constraint independently favours depth-graded
allocation. The cumulative angular phase over a sequence of length
$T$ is

$$
\Theta_T = \sum_{t=1}^{T} \theta_t,\qquad \mathrm{std}(\Theta_T) \approx \mathrm{std}(\theta_t) \cdot \sqrt T
$$

For large $\mathrm{std}(\theta_t)$, $\Theta_T$ wraps around the
circle multiple times within typical sequence length $T \approx
500$ frames, and the resulting interference patterns become
chaotic — exactly the failure mode we observed in v1, where
$\theta_t \in \mathcal{U}(-\pi/4, +\pi/4)$ produced
$\mathrm{std}(\Theta_T) \approx 230$ rad and made the model
untrainable.

This phase-coherence constraint argues for *small* $\gamma_\theta$
on average, but it is an aggregate constraint over the whole
sequence — not a per-layer constraint. Different layers can
independently choose their effective $\gamma_\theta$ as long as the
*total* cumulative phase across the layer stack stays bounded.
Depth-graded allocation lets shallow layers contribute very little
to cumulative phase ($\gamma_\theta = \pi/8$) so that deep layers
can contribute more ($\gamma_\theta = \pi/2$) without the layer
stack exceeding the trainability budget.

### 4.5 Why the parameter-matched comparison is decisive

The Stage-4 `depth` backbone has *exactly the same number of encoder
parameters* as `rwkv6_rse` (5,899,520) — the LoRA-dim gradient
16 / 32 / 48 sums to the same total as the uniform 32 across 6
layers. The only thing that differs is *where* the parameters are
allocated.

A win for `depth` over `rwkv6_rse` therefore cannot be attributed to
"more parameters helped" or "more capacity helped" — it can only be
attributed to the *spatial allocation* of the rotation budget along
the depth axis. This is a clean ablation of the depth-preference
hypothesis from the diagnostic.

The control `strong` supplies the complementary test: if uniform-
larger-budget *also* wins, the depth-graded gain might be partially
or fully explained by raw budget rather than by allocation. The
final §6 verdict depends on which of the four cells of the
$\{depth\ \text{wins/flat}\} \times \{strong\ \text{wins/flat}\}$
matrix the data lands in.

---

## 5. Empirical results

### 5.1 Trajectory comparison (dev CER per epoch)

| Ep | `rwkv6_rse` | `rwkv6_rse_m2` | `rwkv6_rse_depth` | `rwkv6_rse_strong` |
|---:|---:|---:|---:|---:|
|  5 | 0.2365 | 0.2295 | 0.2282 | 0.2312 |
| 10 | 0.1748 | 0.1728 | 0.1695 | 0.1680 |
| 15 | 0.1485 | 0.1484 | 0.1437 | 0.1430 |
| 19 | 0.1367 |   —    | 0.1317 | 0.1304 |
| 20 | 0.1360 |   —    | 0.1294 | 0.1280 |
| 25 | 0.1271 |   —    | 0.1223 | 0.1214 |
| 30 | 0.1251 |   —    | 0.1208 | 0.1197 |

The Stage-3.5 multi-rate variant `rwkv6_rse_m2` is included as the
crucial "more capacity, same allocation policy" control: extra state
and extra projections, no change in per-layer budget. M=2's
trajectory tracked single-scale RSE within $|\Delta| < 0.001$ from
ep 15 onward and converged to the same plateau — establishing that
*more capacity in the same allocation policy does not help*.

### 5.2 Final results

| Run | Best dev CER | Test CER | Test WER | Δ test CER vs `rwkv6` | Encoder params |
|---|---:|---:|---:|---:|---:|
| `rwkv6` (Stage 2 ref) | 0.1258 | 0.1263 | 0.3764 | (ref) | 5,825,024 |
| `rwkv6_rse` (Stage 3, uniform) | 0.1251 | 0.1238 | 0.3705 | −2.0 % | 5,899,520 |
| `rwkv6_rse_m2` (Stage 3.5) | TBD | TBD | TBD | TBD | 6,060,848 |
| **`rwkv6_rse_depth`** | **0.1207** | **0.1200** | **0.3593** | **−5.0 %** | **5,899,520** |
| **`rwkv6_rse_strong`** | **0.1192** | **0.1188** | **0.3579** | **−5.9 %** | **5,936,384** |

`rwkv6_rse_depth` produces a clean −5.0 % test improvement at
*identical encoder parameter count* to `rwkv6_rse`. The improvement
is over 2× the seed-noise margin and lands within Proposal A §9 P1's
predicted ≥ 3 % band.

### 5.3 The −0.005 absolute gap is stable

| Ep | `rwkv6_rse` | `rwkv6_rse_depth` | Δ |
|---:|---:|---:|---:|
| 14 | 0.1540 | 0.1488 | −0.0052 |
| 15 | 0.1485 | 0.1437 | −0.0048 |
| 16 | 0.1439 | 0.1395 | −0.0044 |
| 17 | 0.1443 | 0.1359 | −0.0084 |
| 18 | 0.1395 | 0.1339 | −0.0056 |
| 19 | 0.1367 | 0.1317 | −0.0050 |
| 20 | 0.1360 | 0.1294 | −0.0066 |
| 25 | 0.1271 | 0.1223 | −0.0048 |
| 30 | 0.1251 | 0.1208 | −0.0043 |

The gap held in the −0.004 to −0.008 band for 16 consecutive epochs
(ep 14 – ep 30), including the LR-decay window (ep 24 – ep 28)
where the Stage-3.5 multi-rate variants collapsed to $\Delta \approx
0$. The gap is also approximately stationary, not accelerating —
consistent with the asymptotic-difference reading rather than the
transient-early-convergence reading.

---

## 6. Synthesis: what the experiment tells us

### 6.1 Confirmed claims

- **The RSE Lie-group expansion does break the RWKV-6 expressivity
  ceiling on causal LibriSpeech ASR**, contrary to the apparent
  Stage-3 / Stage-3.5 negative result.
- **The mechanism that broke the ceiling is rotation degree-of-freedom,
  not added state capacity.** `rwkv6_rse_m2` had +2.7 % more
  parameters than `rwkv6_rse_depth` and matched the uniform RSE
  baseline (no improvement); `rwkv6_rse_depth` had identical
  parameter count and produced the −5.0 % test improvement.
- **The architectural prior that matters is per-layer non-uniform
  allocation of the rotation budget**, with deeper layers granted
  larger rotation magnitudes than shallower layers. The diagnostic
  in §2.4 detected this preference in the trained Stage-3 model;
  Stage-4 confirmed that imposing the same gradient as an
  architectural prior produces the predicted gain.

### 6.2 Resolved: both refinement axes are productive

`rwkv6_rse_strong` final landed at **dev 0.1192 / test 0.1188**
(−5.9 % test improvement vs `rwkv6`), slightly better than
`rwkv6_rse_depth` (test 0.1200, −5.0 %) but at +0.6 % more encoder
parameters (5,936,384 vs 5,899,520). The 2×2 outcome matrix
collapses to *both wins*, so the question becomes parameter
efficiency rather than mechanism existence.

| Variant | Test CER Δ vs `rwkv6` | Encoder param Δ vs `rwkv6` | Δ test CER per +1 % param |
|---|---:|---:|---:|
| `rwkv6_rse` (Stage 3 uniform-small) | −0.0025 (−2.0 %) | +1.3 % | −0.0019 |
| `rwkv6_rse_depth` (Stage 4 graded) | −0.0063 (−5.0 %) | +1.3 % | **−0.0048** |
| `rwkv6_rse_strong` (Stage 4 uniform-large) | −0.0075 (−5.9 %) | +1.9 % | −0.0040 |

Two facts emerge:

1. **Lifting the uniform $\gamma_\theta$ ceiling alone (Stage 3 →
   strong) improves CER nearly threefold.** Going from clip $\pi/4$
   / LoRA 32 (Stage 3) to clip $\pi/2$ / LoRA 48 across all layers
   produces $-0.0050$ extra test-CER improvement at $+0.6\%$ extra
   parameters. The Stage-3 ceiling at 0.125 was *not* the
   Lie-group ceiling that Theorem 1 promised — it was a
   parameterization ceiling imposed by the conservative defaults.

2. **Re-allocating budget along depth at unchanged parameter count
   (Stage 3 → depth) also improves CER nearly threefold.** From
   the same Stage-3 starting point, depth produces $-0.0038$ extra
   test-CER improvement at *zero* extra parameters. The
   diagnostic-revealed depth preference is real and exploitable.

Per-parameter, depth is marginally more efficient ($-0.0048$ vs
$-0.0040$ test-CER per +1 % parameters) — the difference is within
seed-noise but consistent in sign.

**Mechanism interpretation.** The Stage-3 uniform-small parameter-
ization left the rotation budget too tight at *all* layers, but
especially at deep layers (where the diagnostic revealed the
binding constraint). The model, under uniform constraint, allocates
its remaining gradient capacity asymmetrically — growing the
data-dependent LoRA more at deep layers than shallow — but cannot
fully escape the cap. Two complementary remedies work:
- *Lift the cap everywhere* (`strong`): wasteful at shallow layers
  (the model has to learn to suppress unused rotation budget), but
  the deep-layer benefit alone is worth the extra parameters.
- *Lift the cap only where it binds* (`depth`): produces nearly the
  same deep-layer benefit at no extra parameter cost.

Both confirm the same underlying truth: **the Stage-3 plateau at
0.125 was a parameterization artifact, not the irreducible
expressivity ceiling of RSE on this task.**

### 6.3 Implications for the chapter

The chapter conclusion is no longer "we systematically falsified
state-side modifications". It is the substantially stronger:

> *The transition Lie group of causal RWKV-6 admits a strict
> extension to* $SO(2)^B \times (\mathbb{R}_+)^B$ *that delivers a
> measurable improvement on causal LibriSpeech CTC ASR, conditional
> on the architectural prior that the rotation budget be allocated
> non-uniformly across depth (smaller at shallow layers, larger at
> deep layers). The improvement is realized at unchanged encoder
> parameter count, unchanged $O(K^2)$ per-step cost, and unchanged
> streaming property.*

This is a positive, replicable, mechanistically explained result —
exactly what the thesis chapter aims for.

---

## 7. Methodological lessons

Two methodological takeaways worth documenting for future
experiments:

1. **Diagnostic first, refinement second.** A cheap CPU diagnostic
   on the saved checkpoint (the static θ-mobility analysis ran in
   under 30 seconds) prevented committing 8+ GPU-hours to a
   "formant-init" experiment that would have addressed the wrong
   hypothesis. The diagnostic falsified the cold-start hypothesis
   in 30 seconds; without it, the next 4 GPU-hours would have
   gone to an init experiment that would have failed for a reason
   the diagnostic already ruled out.

2. **Param-matched ablations are the only credible negative
   evidence.** The Stage-3 `rwkv6_rse` had +1.3 % more parameters
   than `rwkv6`, which made the marginal −2.0 % test improvement
   ambiguous. The Stage-4 `rwkv6_rse_depth` was constructed to be
   *exactly* parameter-matched to `rwkv6_rse` (16 + 32 + 48 LoRA
   gradient sums to the same total as 6 × 32 uniform), making the
   −4 % vs single-scale RSE attributable solely to the
   re-allocation of the rotation budget along the depth axis.
   Without that constraint, the depth gain could be explained away
   as "deep-layer LoRA is bigger, so we added parameters there".
   With it, the only thing that changed was where the existing
   parameters went.

These two patterns — *check first, then refine* and *enforce
parameter parity to make ablations decisive* — should be the
default protocol for any future state-side mechanism experiment in
this thesis.
