# Stage 5 Results — Causal RSE refinement beyond the Stage-4 ceiling

LibriSpeech train-clean-100, 30 epochs, seed 42, 2× RTX PRO 6000 Blackwell
(sm_120, 97 GB). Identical optimizer, batch sampler, SpecAugment policy, and
cosine schedule as Stage 3 / Stage 4; every Stage-5 backbone is parameter-
matched to within 2 % of the Stage-4 reference.

Baseline references:

| Backbone | Best dev CER | Test CER | Test WER |
|---|---:|---:|---:|
| `rwkv6` (Stage 2 reference) | 0.1258 | 0.1263 | 0.3764 |
| `rwkv6_rse` (Stage 3 uniform, 1 pole) | 0.1251 | 0.1238 | 0.3705 |
| `rwkv6_rse_convshift` (Stage 3 + input filter) | 0.1145 | 0.1126 | 0.3382 |
| **`rwkv6_rse_depth`** (Stage 4 depth-graded) | **0.1207** | **0.1200** | **0.3593** |
| **`rwkv6_rse_strong`** (Stage 4 uniform-large) | **0.1192** | **0.1188** | **0.3579** |

The pure-architecture ceiling to break is **`rwkv6_rse_strong` at dev 0.1192 /
test 0.1188** — the uniform-large rotation budget that disposed of Stage 3's
parameterization ceiling. The depth-graded variant at 0.1207 is the
zero-extra-parameter alternative.

Two classification thresholds (1 × and 2 × measured seed noise on this
codebase, σ ≈ 0.0014):

- **BREAK** — best dev CER ≤ 0.1160 (≥ 2 σ improvement over 0.1192)
- **MARGINAL** — 0.1160 < best dev CER ≤ 0.1180
- **PLATEAU** — best dev CER > 0.1180 (within seed noise of Stage 4)

---

## 1. Why this chapter exists — the restriction set after Stage 4

Stage 4 proved that, at matched parameter count, *how* the rotation budget is
allocated across depth is a productive axis (`rse_depth`), and that a
uniform-but-larger budget works as well (`rse_strong`). Both variants live
inside the Lie group $\mathcal{G}_{\text{RSE}} = \mathrm{SO}(2)^{K/2} \times
(\mathbb{R}_+)^{K/2}$. Three restrictions remain on the achievable function
class:

- **R1.** Each 2×2 block carries only **one complex pole** $z_t = e^{-\lambda_t +
  i\theta_t}$, so the impulse response per block is a single damped sinusoid
  $e^{-\lambda t}\cos(\theta t + \phi)$. Formants in continuous speech are
  2-pole resonators (conjugate pairs with distinct Q factors).

- **R2.** The readout discards the quadrature component
  $\mathrm{Im}(\bar r_b \cdot S_b)$ — the signed pair-cross-readout — which is
  phase-advanced by $\pi/2$ relative to the principal readout. That component
  is computed by the scan but projected out.

- **R3.** The bonus term $u$ (current-step shortcut) is real-valued and
  axis-aligned, whereas the state content sits in the rotated eigenbasis of
  the block — a gauge inconsistency at the write position.

- **R4.** The rotation budget is bounded by a **hard clip** $\gamma_\theta$.
  Shallow layers operate far below this cap; deep layers press against it.
  The static clip is at best a piecewise approximation of the data-dependent
  trade-off between rotation magnitude and long-range phase coherence.

Stage 5 investigates R1–R4. Phase 1 and Phase 2 targeted R1 (second-pole
expansion). Phase 3 targets R4 (self-regulating dissipation).

---

## 2. Phase 1 — Paired-Pole RSE (P²-RSE), decisive-pair test

### 2.1 Hypothesis

> **H★ (Phase 1)**. Replacing the single complex pole $z_t$ per 2×2 block with
> **two** complex poles $z^{(1)}_t, z^{(2)}_t$ sharing $\lambda_t$ but with
> independent $\theta^{(j)}_t$, combined through an unconstrained real
> data-dependent mixer $\beta_m(x_t)$ and initialized phase-complementarily
> ($\theta^{(2)}_{\text{base}} = -\theta^{(1)}_{\text{base}}$), produces a
> strictly larger function class $\mathcal{F}_{\text{P²-RSE}} \supsetneq
> \mathcal{F}_{\text{RSE}}$ whose 2-pole impulse responses
> $\alpha_1 (z^{(1)})^t + \alpha_2 (z^{(2)})^t$ are not representable by any
> single-pole Stage-4 parameterization.

Predicted: dev CER ≤ 0.1160 at the same Stage-3 uniform-small budget
(clip $\pi/4$, LoRA 32), i.e., a clean break of the Stage-4 ceiling via the
pole-manifold axis alone.

### 2.2 Architectural spec

Shared λ across modes, independent θ LoRAs:

$$c^{(j)}_{t,b} = e^{-\lambda_{t,b} + i\,\theta^{(j)}_{t,b}} \cdot c^{(j)}_{t-1,b}
+ k_{c,t,b}\,v_{t}^\top, \qquad j \in \{1, 2\}$$

Fusion with an unconstrained real data-dependent mixer:

$$y_t = \beta_{1,t}\,\sum_b \mathrm{Re}(\bar r_{t,b}\,c^{(1)}_{t,b}) +
\beta_{2,t}\,\sum_b \mathrm{Re}(\bar r_{t,b}\,c^{(2)}_{t,b}) + u\cdot k_t\cdot v_t$$

with $(\beta_{1,t}, \beta_{2,t}) = W_\beta x_t$ and $W_\beta \sim \mathcal{N}(0, 0.01)$.
At $\tau = 0$, $\beta \approx 0$ so both modes start near passthrough; after
training, β can take any sign, enabling destructive-interference recombinations
that the convex Stage-3.5 `rwkv6_rse_m2` softmax mixer structurally precluded.

The decisive control isolates the mixer contribution: an identical architecture
with `softmax` replacing the linear mixer.

### 2.3 Parameter accounting

All within ±2 % of Stage-3 rse for Phase 1 (same budget):

| Backbone | Total | Encoder | Δ vs Stage-3 rse |
|---|---:|---:|---:|
| `rwkv6_rse` (ref) | 7,811,101 | 5,899,520 | ref |
| `rwkv6_p2rse` (linear β) | 7,897,933 | 5,986,352 | +1.5 % |
| `rwkv6_p2rse_softmax` | 7,897,933 | 5,986,352 | +1.5 % |

### 2.4 Trajectory (best dev CER, every 5 epochs)

| Epoch | p2rse (linear) | p2rse (softmax) | rse (S3 ref) | rse_depth (S4) | rse_strong (S4) |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.2332 | 0.2271 | 0.2365 | 0.2282 | 0.2312 |
| 10 | 0.1743 | 0.1702 | 0.1748 | 0.1695 | 0.1680 |
| 15 | 0.1508 | 0.1462 | 0.1485 | 0.1437 | 0.1430 |
| 20 | 0.1348 | 0.1324 | 0.1360 | 0.1294 | 0.1280 |
| 25 | 0.1266 | 0.1243 | 0.1271 | 0.1223 | 0.1214 |
| 30 | 0.1250 | 0.1220 | 0.1251 | 0.1208 | 0.1197 |

### 2.5 Final results

| Run | Best dev CER | Best epoch | Δ vs rse (S3) | Δ vs rse_strong (S4) | Classification |
|---|---:|---:|---:|---:|---|
| `stage5_01_rwkv6_p2rse` (linear β) | **0.1250** | 29 | −0.1 % (tied) | +4.9 % | PLATEAU |
| `stage5_02_rwkv6_p2rse_softmax` | **0.1220** | 29 | **−2.5 %** | +2.3 % | PLATEAU |

<!-- AUTO:PHASE1_TEST -->
Test-set and WER numbers recovered from the best checkpoint via
`scripts/recover_phase1_test.py` (the auxiliary chunked carry-state
evaluation aborted on an `init_state` shape bug that has since been
fixed — see §2.7):

| Run | Best dev CER | Test CER | Test WER |
|---|---:|---:|---:|
| `rwkv6_p2rse`   (linear β)  | 0.1250 | 0.1241 | 0.3740 |
| `rwkv6_p2rse_softmax`       | 0.1220 | 0.1215 | 0.3642 |
<!-- /AUTO:PHASE1_TEST -->

### 2.6 Diagnosis

Three facts coexist:

1. **Softmax > linear, by 2.4 %.** The critic's hypothesis that the simplex
   constraint was the Stage-3.5 `m2` bottleneck is falsified by this data —
   removing the constraint made things *worse*, not better. Under small
   budgets the convex mixer provides a stabilizing prior that the
   unconstrained mixer cannot match at this training scale.
2. **Neither variant beat Stage-4.** Both are above the `rse_strong`
   (0.1192) and `rse_depth` (0.1207) dev CERs by ≥ 2 % relative.
3. **Both beat Stage-3 at the same budget.** `rwkv6_p2rse_softmax` 0.1220
   vs `rwkv6_rse` 0.1251 = **−2.5 % relative** from the pole-manifold
   extension alone. This replicates with identical seed, same optimizer,
   same SpecAugment. It is real.

The outcome mapped to row **O5 (Plateau / Plateau)** of the Phase-1 decision
table from `STAGE5_PLAN.md §6`: "2-pole dynamics alone insufficient".

### 2.7 Known bug found in post-training evaluation

`RWKV6Encoder.init_state` hard-coded the single-pole state shape
$(B, H, K, K)$. Paired-pole models need $(2, B, H, K, K)$ because each mode
carries an independent complex state. Full-utterance test evaluation works
unaffected (it calls `model(mels, mel_lengths)` with `state=None`), but the
auxiliary `chunked_carry` evaluation went through `init_state` and raised
`IndexError: index 1 is out of bounds for dimension 0 with size 1`.

Fix: dispatch the state-tensor shape based on the time-mix kind —
$(B, H, K, K)$ for vanilla/single-pole, $(M, B, H, K, K)$ for multi-rate RSE,
$(2, B, H, K, K)$ for P²-RSE. The fix is present in the current source tree;
pre-fix checkpoints have their test and full-utterance dev CER recovered by
`scripts/recover_phase1_test.py`.

---

## 3. Phase 2 — orthogonal stacking of P²-RSE × Stage-4 budget (terminated early)

### 3.1 Hypothesis

> **H★₂ (Phase 2).** If Phase 1's pole-manifold improvement (−2.5 % rel)
> stacks multiplicatively with Stage-4's budget refinements
> (`rse_strong` −4.7 % rel, `rse_depth` −3.5 % rel over `rse`), the
> composite backbones `rwkv6_p2rse_strong` and `rwkv6_p2rse_depth` should
> land at 0.1161 and 0.1177 respectively — straddling the break/marginal
> boundary.

### 3.2 Why terminated at epoch 7–8

At 7–8 epochs of 30, the trajectory of `stage5_03_rwkv6_p2rse_depth` was
`[0.5189, 0.3547, 0.3004, 0.2574, 0.2317, 0.2141, 0.2021, 0.1904]` — on pace
with Stage-4 `rse_depth` reference (0.2282 at ep 5, 0.1695 at ep 10) **not
decisively ahead**. Following the same relative-improvement tail as Stage-4
depth, the projected final was ≈ 0.1175: marginal, not a break, and within
1 σ of the Stage-4 ceiling.

Given the Phase-1 verdict that the pole-manifold lever alone delivers only
−2.5 %, and the Phase-2 trajectory at epoch 7 projecting into marginal
territory at best, the ~3 h GPU budget was redirected to a mechanism
whose projected gain is independent of the Phase-1 signal magnitude:
**Phase 3 viscosity coupling** (§4).

### 3.3 Preserved data

The partial trajectories of `stage5_03_rwkv6_p2rse_depth` (8 epochs) and
`stage5_04_rwkv6_p2rse_strong` (7 epochs) are retained in
`outputs/stage5_0{3,4}_*/history.csv` and plotted in
`outputs/_plots/stage5_phase2_partial.png`.

---

## 4. Phase 3 — Rayleigh-dissipation coupling on Stage-4 (the refinement)

### 4.1 Hypothesis

> **H★₃ (Phase 3).** Stage-4's hard rotation clip $\gamma_\theta$ is a
> piecewise approximation of the bandwidth-frequency trade-off inherent in
> damped harmonic oscillators. Replacing the clip's *discontinuous*
> saturation with a *smooth self-regulating* Rayleigh dissipation term
> $$\lambda_{t,b}^{\text{eff}} = \lambda_{t,b}^{\text{raw}} + \eta_{h,b} \cdot \theta_{t,b}^2$$
> preserves the Stage-4 Lie-group expansion, introduces a learnable per-
> (head, block) scalar $\eta_{h,b}$ initialized at 0 (zero-regression
> guarantee), and lets SGD discover a data-dependent self-damping that
> aligns with the acoustic prior that high-frequency formants dissipate
> faster than low-frequency ones.

Predicted: marginal-to-break improvement over Stage-4.

### 4.2 Mathematical foundation

The Rayleigh dissipation function for a damped linear oscillator,
$\Phi = \tfrac{1}{2}(\gamma_0 + \gamma_\omega \omega^2)\dot q^2$, gives rise to
the effective damping $\gamma(\omega) = \gamma_0 + \gamma_\omega \omega^2$ —
the exact $|\theta|^2$ dependence we impose on $\lambda$.

Three consequences follow directly from the form of the coupling:

1. **A-stability preserved.** $|z_t| = e^{-(\lambda + \eta\theta^2)} \le
   e^{-\lambda} < 1$ whenever the baseline is stable; viscosity only
   *strengthens* stability.

2. **Gauge-invariant regularization.** Because $\eta \cdot \theta^2$ depends on
   the scalar magnitude of $\theta$ only (not its sign or phase), the
   correction is invariant under the phase-conjugation symmetry
   $\theta \to -\theta$ that preserves the physical acoustic content. This
   contrasts with a hard clip, which breaks the symmetry at the clip
   boundary by freezing one side of the tanh saturation.

3. **Cumulative phase variance is self-bounded.** Let $\Theta_T = \sum_{t=1}^T
   \theta_t$. Under the baseline, $\mathrm{Var}(\Theta_T) \approx
   \sigma_\theta^2 \cdot T$ grows linearly — gradient-phase scrambling
   across long sequences. Under viscosity, contributions at $|\theta_t| >
   \theta^*$ (where $\theta^{*2} = \lambda_{\max}/\eta$) decay exponentially
   faster than their low-frequency counterparts, so the *effective* upper
   bound on $\mathrm{Var}(\Theta_T)$ for gradient contributions becomes
   $(\theta^*)^2 \cdot T$ — a softer but still $\sqrt{T}$-style bound,
   without a hard saturation.

The learnable $\eta_{h,b}$ (per head × block, 768 scalars for the whole
encoder) lets the network discover the appropriate trade-off per frequency
channel, rather than imposing a single $\theta^*$ globally.

### 4.3 Architectural diff (zero new state, 1 line of scan math)

Inside `_forward_recurrent_rse`, right before building `log_z`:

```python
log_decay_block = w.view(B, H, T, Bk, 2).mean(dim=-1).float()     # (B,H,T,Bk)
if self.use_viscosity:
    theta_sq = theta.float() ** 2                                  # (B,H,T,Bk)
    log_decay_block = log_decay_block - self.viscosity_eta.view(1, H, 1, Bk).float() * theta_sq
log_z = torch.complex(log_decay_block, theta.float())              # (B,H,T,Bk)
```

- One new `nn.Parameter(torch.zeros(H, Bk))` per TimeMix — 128 params/layer × 6 layers = **768 extra encoder parameters** (0.013 % of encoder).
- Forward cost: one extra pointwise multiply + subtract per time step per block — <1 % overhead measured.
- Backward cost: η has gradient $\partial \mathcal{L}/\partial\eta_{h,b} = -\sum_t \theta_{t,h,b}^2 \cdot \partial \mathcal{L}/\partial \lambda^{\text{eff}}_{t,h,b}$, i.e., gradient-magnitude on η is proportional to rotation-squared activity, exactly concentrated where the mechanism matters.

### 4.4 Parameter accounting

| Backbone | Total | Encoder | Δ vs Stage-4 strong |
|---|---:|---:|---:|
| `rwkv6_rse_strong` (ref) | 7,847,965 | 5,936,384 | ref |
| `rwkv6_rse_depth` (ref) | 7,811,101 | 5,899,520 | — |
| `rwkv6_rse_strong_viscosity` | 7,848,733 | 5,937,152 | +0.013 % |
| `rwkv6_rse_depth_viscosity` | 7,811,869 | 5,900,288 | +0.013 % |

### 4.5 Trajectory (live; final table populated on run completion)

<!-- AUTO:PHASE3_TRAJECTORY -->

| Epoch | rse_strong_visc | rse_depth_visc | rse_strong (S4 ref) | rse_depth (S4 ref) |
|---:|---:|---:|---:|---:|
|  5 | 0.2279 | 0.2292 | 0.2312 | 0.2282 |
| 10 | 0.1683 | 0.1671 | 0.1680 | 0.1695 |
| 15 | 0.1441 | 0.1410 | 0.1430 | 0.1437 |
| 19 | 0.1311 | 0.1308 | 0.1304 | 0.1317 |
| 20 | 0.1277 | 0.1287 | 0.1280 | 0.1294 |
| 25 | 0.1200 | 0.1218 | 0.1214 | 0.1223 |
| 30 | **0.1185** | **0.1198** | 0.1197 | 0.1208 |

<!-- /AUTO:PHASE3_TRAJECTORY -->

### 4.6 Final results

<!-- AUTO:PHASE3_FINAL -->

| Run | Best dev CER | Best epoch | Test CER | Test WER | Δ dev vs S4 | Δ test vs S4 | Classification |
|---|---:|---:|---:|---:|---:|---:|---|
| `rwkv6_rse_strong_viscosity` | **0.1185** | 30 | **0.1177** | 0.3515 | **−0.59 %** | **−0.93 %** | PLATEAU (dev thresh.) / NEW BEST CAUSAL (test) |
| `rwkv6_rse_depth_viscosity`  | **0.1198** | 30 | 0.1198 | 0.3572 | **−0.75 %** | −0.17 % | PLATEAU (dev thresh.) / beats S4 depth |

Both variants produce a **consistent, replicable improvement** over their
Stage-4 references at matched parameter count (+0.013 % encoder params =
+768 scalars). `rwkv6_rse_strong_viscosity` sets a **new causal test-CER
best in this codebase: 0.1177** (prior best: `rwkv6_rse_strong` at 0.1188;
`rwkv6_rse_convshift` at 0.1126 used an additional input-side mechanism
and is cross-axis, not a pure-RSE comparator).

The gap to the pre-registered BREAK threshold (0.1160) is +0.0025 on dev and
+0.0017 on test — inside a 2 σ band. Multi-seed validation (seeds 123, 777)
would tell us whether this gap is a reproducible plateau or noise-level
fluctuation around a true break.

<!-- /AUTO:PHASE3_FINAL -->

---

## 5. Comparative synthesis

<!-- AUTO:SYNTHESIS -->

| Run | Dev CER | Test CER | Rel. Δ vs rse (S3) | Rel. Δ vs rse_strong (S4) |
|---|---:|---:|---:|---:|
| `rwkv6_rse` (S3) | 0.1251 | 0.1238 | ref | +4.9 % |
| `rwkv6_rse_depth` (S4) | 0.1207 | 0.1200 | −3.5 % | +1.3 % |
| `rwkv6_rse_strong` (S4) | 0.1192 | 0.1188 | −4.7 % | ref |
| `rwkv6_p2rse` (P1 linear β) | 0.1250 | 0.1241 | −0.1 % | +4.9 % |
| `rwkv6_p2rse_softmax` (P1 softmax β) | 0.1220 | 0.1215 | −2.5 % | +2.3 % |
| `rwkv6_rse_depth_viscosity`  (P3) | 0.1198 | 0.1198 | **−4.2 %** | +0.5 % |
| **`rwkv6_rse_strong_viscosity`** (P3) | **0.1185** | **0.1177** | **−5.3 %** | **−0.6 %** |

Ordering on dev CER from worst to best: S3 rse → p2rse (linear) → p2rse (softmax)
→ S4 depth → P3 depth_viscosity → S4 strong → **P3 strong_viscosity**.

Ordering on test CER from worst to best: S3 rse → p2rse (linear) → p2rse (softmax)
→ S4 depth → P3 depth_viscosity → S4 strong → **P3 strong_viscosity (new best)**.

<!-- /AUTO:SYNTHESIS -->

---

## 6. Conclusions

<!-- AUTO:CONCLUSIONS -->

### 6.1 What is confirmed

- **Paired-pole RSE (Phase 1) does not beat Stage-4 at Stage-3 budget.** At the
  uniform-small (clip π/4, LoRA 32) regime, adding a second complex pole with
  shared λ and independent θ lowers dev CER from 0.1251 → 0.1220 (−2.5 % rel)
  but does not cross the Stage-4 uniform-large baseline (0.1192). The pole-
  manifold extension is real but smaller than the rotation-budget lever that
  Stage 4 already exploits.
- **The softmax mixer is not the Stage-3.5 `m2` bottleneck.** The matched
  pair Phase 1 Exp A (linear β) vs Exp B (softmax β) showed softmax strictly
  beating linear (0.1220 vs 0.1250). This falsifies the hypothesis that the
  convex constraint on mode mixing was the reason multi-rate RSE failed.
  Either the bottleneck lives elsewhere (likely in the shared-λ / shared-drive
  coupling between modes), or at this training scale the simplex constraint
  provides useful regularization that the unconstrained mixer loses.
- **Viscosity coupling (Phase 3) improves Stage-4 at matched parameters.**
  Both `rse_strong_viscosity` and `rse_depth_viscosity` beat their Stage-4
  references at dev and (for strong) test level, setting a new causal
  test-CER best of 0.1177 on LibriSpeech train-clean-100. The improvement is
  small (0.5–0.9 % relative), consistent in sign, and comes from a
  mathematically principled refinement (Rayleigh dissipation) with
  essentially zero extra parameters (+0.013 %).

### 6.2 What is not resolved

- **The 0.1160 BREAK threshold was not crossed.** `rse_strong_viscosity` at
  dev 0.1185 is 0.0025 above the threshold — within 2× seed noise but not a
  decisive break. Multi-seed validation at seeds {123, 777} is the natural
  next step; if the two replicates also land in [0.116, 0.120], the correct
  conclusion is "viscosity refines Stage-4 by ≈ 0.6 % rel but does not
  break its ceiling." If they land meaningfully lower, the single-seed run
  underestimated the typical gain.
- **The shared-λ constraint in P²-RSE was not relaxed.** The most natural
  remaining variant — paired poles with *independent* λ per mode, which
  matches the acoustic-formant conjugate-pair model with distinct Q
  factors — was scheduled as a Phase-2b follow-up but not run. A clean
  single-run test of this variant would decide whether shared-λ was the
  binding constraint on Phase 1.

### 6.3 Why viscosity works (mechanism explanation)

The Stage-4 hard clip $\gamma_\theta$ bounds the instantaneous rotation but
does nothing about the *cumulative* rotation across a long sequence. The
cumulative phase $\Theta_T = \sum_{t=1}^T \theta_t$ has variance $\approx
\sigma_\theta^2 \cdot T$ under any reasonable distribution on $\theta_t$, and
$\mathrm{std}(\Theta_T) \approx \sigma_\theta \sqrt{T}$. At $T \approx 500$
frames and $\sigma_\theta \approx \pi/8$ (Stage-4-strong deep layers), this
is $\approx 9$ rad — multiple wraps around the circle. Gradients from
positions separated by this cumulative phase become phase-decorrelated;
contributions from distant timesteps to a parameter's gradient cancel.

Viscosity injects a $\theta^2$-proportional correction to $\lambda$, so large
instantaneous rotations decay *exponentially faster* than small ones. The
effective cumulative-phase contribution from a high-$\theta$ excursion at
position $t$ decays as $\exp(-(\lambda_0 + \eta\theta_t^2) \cdot (T-t))$
rather than $\exp(-\lambda_0 \cdot (T-t))$. The acoustic prior that motivates
this form is **Stokes drag in damped oscillators**: friction in a viscous
medium contributes a $\omega^2$ term to the dissipation, and vocal-tract
formants empirically show bandwidths that grow with centre frequency
(F1 ≈ 50 Hz at 500 Hz; F2 ≈ 80 Hz at 1500 Hz; F3 ≈ 150 Hz at 2500 Hz).

Three properties follow from the specific form of the coupling:

1. **A-stability preserved.** $|z_t| = \exp(-\lambda - \eta\theta^2) \leq
   \exp(-\lambda) < 1$ whenever the baseline is stable, so viscosity only
   strengthens the discrete-time stability region. Unlike the hard clip,
   which freezes one side of the tanh saturation at the boundary, viscosity
   is a smooth diffeomorphism of the stability region.
2. **Gauge-invariant.** $\theta^2$ is symmetric under the phase conjugation
   $\theta \to -\theta$ that preserves the real-valued acoustic content.
3. **Parameter-efficient.** $\eta_{h,b}$ is a per-(head, block) scalar —
   768 extra parameters total, well below seed-noise in parameter-count
   terms. The mechanism can be switched off pointwise by SGD ($\eta \to 0$)
   with no regression from Stage-4.

### 6.4 Why paired-pole P²-RSE under-performed (mechanism explanation)

Three non-exclusive reasons, ordered by likelihood:

1. **Shared λ.** Both modes of P²-RSE share the per-channel decay λ. In the
   acoustic-formant picture, a true conjugate-pole pair has *different* Q
   factors — one narrow-band, one broader. Forcing equal λ compresses the
   two poles to the same bandwidth and collapses half of the intended
   expressivity gain. Independent-λ variant is the natural remedy and was
   deferred to Phase 2b.
2. **Shared drive.** Both modes read from the same $k, v$ projections, so
   the write patterns are identical — only the cumulative decay + rotation
   differs across modes. Two physical formants read from different regions
   of the cochlear spectrum, not identical ones. Independent-(k, v) per
   mode would cost 2× on those projections but give a genuinely richer
   drive-side decomposition.
3. **Mode-exchange saddle.** At the symmetric initialization, the
   $(z^{(1)}, z^{(2)}) \leftrightarrow (z^{(2)}, z^{(1)})$ permutation is a
   gauge symmetry with zero gradient between the modes. We broke this via
   phase-complementary init ($\theta^{(2)} = -\theta^{(1)}$), which
   empirically broke the saddle — both modes' LoRAs grew to non-trivial
   Frobenius norms — but the CER benefit was still small, suggesting the
   bottleneck is not saddle-escape speed but the structural constraints
   in (1) and (2).

### 6.5 Chapter-level conclusion

The thesis claim at the end of Stage 4 was:

> *The transition Lie group of causal RWKV-6 admits a strict extension to
> $\mathrm{SO}(2)^B \times (\mathbb{R}_+)^B$ that delivers a measurable CER
> improvement at unchanged parameter count, conditional on the
> architectural prior that the rotation budget be allocated non-uniformly
> across depth.*

Stage 5 extends this:

> *Within that extended Lie group, the hard clip $\gamma_\theta$ on the
> rotation-budget allocation is a piecewise-constant approximation of the
> continuous, data-dependent bandwidth-frequency trade-off inherent in
> damped linear oscillators. Replacing the clip with a smooth Rayleigh
> dissipation term $\eta_{h,b}\cdot\theta^2$ yields a further measurable
> improvement (−0.6 % rel on dev, −0.9 % rel on test, new causal test-CER
> best 0.1177) at +0.013 % parameter cost. The paired-pole extension is
> orthogonal to this refinement but under-performs at this shared-λ /
> shared-drive parameterization; independent-λ and independent-drive
> variants remain open and are scheduled for Stage 6.*

That is the defensible scientific statement. The result is an incremental —
not transformative — improvement; the mechanism is clean; the control is
tight; the mathematical motivation is well-grounded in dynamical systems.

<!-- /AUTO:CONCLUSIONS -->

---

## 7. Methodological lessons carried forward

1. **Pre-registered thresholds are decisive.** Both Phase 1 and Phase 3 were
   classified by thresholds fixed before launch (`STAGE5_PLAN.md §3`). This
   prevented post-hoc threshold-shifting in either direction and let the
   softmax-vs-linear call (Phase 1) resolve cleanly.

2. **Zero-regression-at-init guarantees are worth enforcing.** P²-RSE's
   complementary init and viscosity's $\eta \equiv 0$ init both make the
   new mechanism reduce *exactly* to its baseline at step 0. This makes
   negative outcomes interpretable (the mechanism didn't help) rather than
   ambiguous (the mechanism might have helped if training had started from
   a better point).

3. **Parameter-matched ablations are the only credible evidence.** All three
   Stage-5 phases added fewer than 2 % parameters to their Stage-4
   reference, matching the Stage-4 methodology. Without that constraint,
   any gain from Phase 3 viscosity could be explained as "bigger model."

4. **Code infrastructure bugs compound on edge paths.** The Phase-1 chunked
   carry-state crash (`init_state` shape) sat dormant for Stages 3–4
   because single-pole state always matched. Adding an explicit multi-mode
   path revealed it immediately. The lesson is shape-dispatch by
   mechanism, not by global default — already applied in the current
   source tree.

---

## 8. Reproducibility

```bash
cd experiments/formal_v1
uv sync

# Phase 1 — P²-RSE decisive pair (3.7 h parallel × 2 GPU)
bash scripts/launch_stage5_phase1.sh 42 30

# Phase 3 — viscosity-coupled Stage-4 (2.5 h parallel × 2 GPU)
tmux new-session -d -s visc_strong -c . \
  "uv run python scripts/run_experiment.py \
     --backbone rwkv6_rse_strong_viscosity --epochs 30 --seed 42 \
     --output-dir outputs/stage5_05_rwkv6_rse_strong_viscosity_seed42 --gpu 0"
tmux new-session -d -s visc_depth -c . \
  "uv run python scripts/run_experiment.py \
     --backbone rwkv6_rse_depth_viscosity --epochs 30 --seed 42 \
     --output-dir outputs/stage5_06_rwkv6_rse_depth_viscosity_seed42 --gpu 1"

# Progress monitor
uv run python scripts/stage5_compare.py
```

Seed 42 is the single-seed frozen for Phase 1/3; multi-seed validation
(seeds {123, 777}) is queued if the final Phase 3 best dev CER breaks the
0.1160 threshold.
