# Stage 5 Plan — P²-RSE: Paired-Pole Rotational State Evolution

Breaking the Stage-4 ceiling (`rwkv6_rse_strong` at dev CER **0.1192** / test 0.1188) via a strict extension of the transition manifold from one complex pole per block to two.

---

## 1. Central hypothesis (modest form)

> **H★.** The Stage-4 plateau at ~0.119 dev CER is the single-pole ceiling of $\mathcal{F}_{\text{RSE}}$. Replacing the single pole $z_t = e^{-\lambda_t + i\theta_t}$ per 2×2 block with **two complex poles** $z^{(1)}_t, z^{(2)}_t$ driven by shared λ and independent θ, combined through an **unconstrained data-dependent real mixer**, and initialized at **phase-complementary** $\theta^{(2)} = -\theta^{(1)}$, lowers dev CER to **≤ 0.1160** on LibriSpeech train-clean-100 at 30 epochs, seed 42.

The threshold is 1.5× seed-noise below the Stage-4 baseline. This is a strictly falsifiable prediction in a single run, no re-seeding required for an initial go/no-go.

## 2. Stage-4 baseline reference (fixed)

| Run | Best dev CER | Test CER |
|---|---:|---:|
| `rwkv6_rse_strong` (Stage 4) | **0.1192** | **0.1188** |
| `rwkv6_rse_depth` (Stage 4) | 0.1207 | 0.1200 |
| `rwkv6_rse` (Stage 3 baseline) | 0.1251 | 0.1238 |

## 3. Decision thresholds (use consistently across Phase 1, 2, 3)

Seed-noise estimate on this codebase: **σ ≈ 0.0014** (from LION vs lion_lucid paired-seed gap, `PLAN.md` §7.7).

| Classification | Dev CER vs Stage-4 strong (0.1192) | Interpretation |
|---|---|---|
| **Break** | ≤ 0.1160 (−2.7 %) | ≥ 2× seed-noise improvement |
| **Marginal** | 0.1160 < x ≤ 0.1180 | 1× seed-noise; needs replication |
| **Plateau** | > 0.1180 | Within seed-noise of Stage 4 |

---

## 4. Architectural specification (P²-RSE, final form for Phase 1)

### 4.1 State updates (two conjugate modes, shared λ)

$$
\boxed{\;c^{(j)}_{t,b} \;=\; e^{-\lambda_{t,b} + i\,\theta^{(j)}_{t,b}} \cdot c^{(j)}_{t-1,b} \;+\; k_{c,t,b}\,v_t^\top,\qquad j\in\{1,2\}\;}
$$

Drive $k_{c,t,b} = k_{2b,t} + i\,k_{2b+1,t}$ and $v_t$ shared between modes (no drive-side parameter doubling).

### 4.2 θ production (independent LoRAs)

$$
\theta^{(1)}_t = \gamma_\theta\,\tanh\!\bigl(\theta^{(1)}_{\text{base}} + \text{LoRA}^{(1)}(x_{\text{mix},t})\bigr),\qquad
\theta^{(2)}_t = \gamma_\theta\,\tanh\!\bigl(\theta^{(2)}_{\text{base}} + \text{LoRA}^{(2)}(x_{\text{mix},t})\bigr)
$$

LoRA dims: 32 per mode (matching Stage-3 RSE's `time_theta_w1/w2`). **Total new parameters ≈ 12,400 per layer** — within 2 % of Stage-4 RSE-strong encoder budget.

### 4.3 Fusion readout (unconstrained real, data-dependent)

$$
\boxed{\;y_t \;=\; \beta_{1,t}\;\!\sum_b \mathrm{Re}\!\bigl(\bar r_{t,b}\,c^{(1)}_{t,b}\bigr) \;+\; \beta_{2,t}\;\!\sum_b \mathrm{Re}\!\bigl(\bar r_{t,b}\,c^{(2)}_{t,b}\bigr) \;+\; u\cdot k_t\cdot v_t\;}
$$

with

$$\bigl(\beta_{1,t},\beta_{2,t}\bigr) = W_\beta\,x_t,\qquad W_\beta\in\mathbb{R}^{D\times 2H},\quad W_\beta\sim\mathcal{N}(0,\,0.01)$$

Per-head, per-step real scalars. **No softmax, no nonnegativity.** Starts near $(\beta_1,\beta_2) \approx (0,0)$ at init (passthrough-like), grows freely under training. Allows $\beta_1\beta_2 < 0$ ⇒ destructive interference ⇒ 2-pole bandpass isolation.

### 4.4 Phase-complementary initialization

$$
\theta^{(1)}_{\text{base},b} \sim \mathcal{U}\!\bigl(-\pi/16,\,\pi/16\bigr),\qquad \theta^{(2)}_{\text{base},b} \;=\; -\,\theta^{(1)}_{\text{base},b}
$$

LoRA weights $W_1^{(j)} = 0$, $W_2^{(j)} \sim \mathcal{U}(-0.01, 0.01)$ per existing RSE convention.

### 4.5 Exact causal masking (numerical stability)

Replace the `-60` clamp in `_forward_recurrent_rse` with:

```python
A_unmasked = torch.exp(diff_complex)
A = torch.where(mask, A_unmasked, torch.zeros_like(A_unmasked))
```

No arithmetic change where mask is true; exact zero where mask is false.

### 4.6 Scan cost

Two sequential calls to the existing `_forward_recurrent_rse` core per layer per forward pass. **~2× RSE step time, same asymptotic $O(T K^2)$.** Expected epoch wall-clock: ~500 s (vs Stage 3 RSE's 240–252 s).

---

## 5. Phase 1 — The decisive pair (2 GPUs, in parallel)

### Experiment A (GPU 0): `stage5_01_rwkv6_p2rse_seed42`

- **Backbone:** `rwkv6_p2rse` (the full §4 specification).
- **Config:** seed 42, 30 epochs, LibriSpeech train-clean-100, identical optimizer / schedule / SpecAugment / batching as Stage 3–4.
- **Hypothesis H_A:** Dev CER ≤ 0.1160 at the best-epoch checkpoint.
- **Why this experiment:** direct test of H★.

### Experiment B (GPU 1): `stage5_02_rwkv6_p2rse_softmax_seed42`

- **Backbone:** `rwkv6_p2rse_softmax` — identical to Exp A except:

    $$(\beta_{1,t}, \beta_{2,t}) = \mathrm{softmax}\!\bigl(W_\beta x_t\bigr)$$

- **Hypothesis H_B:** Dev CER > 0.1180 (plateau, matching the Stage-3.5 `m2` result at ~0.125 or at best the mid-0.11s).
- **Why this experiment:** isolates whether the unconstrained mixer is *the* critical element vs. whether the 2-pole dynamics alone break the ceiling regardless of mixer. Without this control, a win in Exp A leaves open the attribution between "2-pole structure" and "unconstrained mixing".

---

## 6. Decision tree after Phase 1

Decision is made on **best dev CER**, not test. Test is computed post-hoc on the best-dev checkpoint for final reporting.

| # | Exp A | Exp B | Diagnosis | Action |
|---|:---:|:---:|---|---|
| **O1** | Break | Break | 2-pole dynamics breaks the ceiling regardless of mixer constraint. The Stage-3.5 plateau attributable to something other than softmax alone (possibly the lack of phase-complementary init). | **→ Phase 2a** (stack mechanisms) |
| **O2** | Break | Plateau | Confirmed: unconstrained mixer was the load-bearing element. Matches the critic's mechanistic hypothesis precisely. Softmax ⇒ no destructive interference ⇒ no true 2-pole basis. | **→ Phase 2a** (stack mechanisms) |
| **O3** | Marginal | Plateau | Weak positive signal on H★. 2-pole + unconstrained mixer directionally correct but magnitude insufficient. | **→ Phase 2b** (diagnostic — is it init? is it budget? is it depth?) |
| **O4** | Marginal | Break | Highly unexpected — softmax stabilizing something that the unconstrained version is not. Likely optimization pathology (β blowing up, bimodal collapse). | **→ Phase 2b** (diagnostic) |
| **O5** | Plateau | Plateau | H★ falsified at this configuration. 2-pole dynamics alone insufficient. | **→ Phase 2c** (pivot back to Stage-4-style refinements) |
| **O6** | Plateau | Break | Essentially impossible if implementation is correct. Re-audit code before acting. | **→ Phase 2d** (correctness audit) |

---

## 7. Phase 2a — Stack mechanisms (if Phase 1 breaks ceiling)

Two parallel runs to answer: does P²-RSE compose with the Stage-2 and Stage-4 winners?

### Experiment C (GPU 0): `stage5_03_rwkv6_p2rse_convshift_seed42`

- **Backbone:** `rwkv6_p2rse_convshift` — P²-RSE + ConvShift (Stage-2 winner on rwkv6 causal, Stage-3 winner on rwkv6_rse).
- **Hypothesis:** ConvShift is input-side, P²-RSE is transition-side; they should be orthogonal and stack additively. Predicted dev CER ≤ 0.1100 (Stage-3 ConvShift added ~−9 % on rwkv6, same structure should carry).
- **Reasoning:** `rwkv6_rse_convshift` already tested on Stage 3 (−2 % test vs rwkv6_rse). If P²-RSE is genuinely in a larger function class, the two orthogonal mechanisms should compose at or near their independent gains.

### Experiment D (GPU 1): `stage5_04_rwkv6_p2rse_depth_seed42`

- **Backbone:** `rwkv6_p2rse_depth` — P²-RSE with Stage-4 depth-graded budget applied per mode.
- **Schedule:** $\gamma_\theta$ = π/8 (L0–L1), π/4 (L2–L3), π/2 (L4–L5). LoRA dim 16/32/48. Same on both modes.
- **Hypothesis:** Deep layers still benefit from larger rotation budget under 2-pole structure. Predicted dev CER ≤ 0.1100.
- **Reasoning:** Stage-4 depth adds −3 % test vs Stage-3 uniform. If P²-RSE doesn't already saturate the depth budget axis, this axis remains productive.

### Decision after Phase 2a

| C (ConvShift) | D (Depth) | Action |
|:---:|:---:|---|
| Both break further | Combine: Phase 3 → `p2rse_convshift_depth` + 3-seed validation |
| C breaks, D flat | Depth absorbed by P²-RSE. Publish P²-RSE + ConvShift as the final combo. |
| C flat, D breaks | ConvShift redundant with P²-RSE (input-side filter absorbed). Publish P²-RSE + Depth. |
| Both flat | P²-RSE saturates all known orthogonal axes. Publish alone; potentially run 3-seed on best P²-RSE variant. |

---

## 8. Phase 2b — Diagnostic (if Phase 1 marginal)

Two parallel runs to disambiguate **which element** of P²-RSE is under-performing.

### Experiment C2 (GPU 0): `stage5_03_rwkv6_p2rse_indeplambda_seed42`

- **Backbone:** `rwkv6_p2rse_indeplambda` — identical to Phase 1 Exp A but with independent λ per mode (each mode has its own decay LoRA).
- **Why:** tests whether the "shared λ" constraint in §4.1 was the wrong parameter-efficiency tradeoff. Independent λ allows each pole to have its own bandwidth.

### Experiment D2 (GPU 1): `stage5_04_rwkv6_p2rse_warmstart_seed42`

- **Backbone:** `rwkv6_p2rse` — Exp A architecture with θ-base init scale bumped to $\pi/8$ (from π/16) and $W_\beta$ init bumped to $\mathcal{N}(0, 0.1)$ (from 0.01).
- **Why:** tests whether the near-zero β init prevented the modes from being used at all in Phase 1. Warm-start the β pathway so the modes start meaningfully contributing.

### Decision after Phase 2b

- If C2 breaks → independent λ needed; upgrade spec and re-run as Phase 1 equivalent.
- If D2 breaks → init schedule needed rework; use warm-start init as the base.
- If neither → H★ has a more fundamental hole. Pivot to Phase 2c.

---

## 9. Phase 2c — Pivot to Stage-4 refinements (if Phase 1 plateaus)

H★ is falsified. The 2-pole extension is not the productive axis for causal ASR. Pivot to refining Stage 4.

### Experiment C3 (GPU 0): `stage5_03_rwkv6_rse_strong_viscosity_seed42`

- **Backbone:** `rwkv6_rse_strong_viscosity` — Stage-4 `rse_strong` with viscosity coupling $\lambda^{\text{eff}}_{t,b} = \lambda_{t,b} + \eta_b|\theta_{t,b}|^2$, $\eta_b$ learnable per-block, init 0.
- **Hypothesis:** the Stage-4 uniform-π/2 clip is conservative at deep layers precisely because the static clip cannot adapt to instantaneous phase. A data-dependent self-damping should allow effectively larger rotation while maintaining long-range coherence.
- **Predicted gain:** dev CER ≤ 0.1180 (1 % relative improvement).
- **Why this experiment:** the cheapest, most-defensible Stage-4 refinement if P²-RSE fails.

### Experiment D3 (GPU 1): `stage5_04_rwkv6_rse_depth_viscosity_seed42`

- **Backbone:** `rwkv6_rse_depth` + viscosity coupling (same η formulation, init 0).
- **Why:** tests whether viscosity and depth-grading are orthogonal. If both mechanisms contribute, the combination should land at ~0.117 dev CER.

### Decision after Phase 2c

- If viscosity breaks 0.118 cleanly on either config → Stage 4 has a productive third refinement axis; write chapter around "three-axis Lie-group refinement".
- If neither moves → the ~0.119 plateau is the true causal-RSE ceiling. Write chapter as "RSE ceiling identified; causal→bidirectional gap is information-theoretic". Pivot direction: attack the information gap (e.g., partial-bidirectional via LION-mode stacking).

---

## 10. Phase 3 — Multi-seed validation (if Phase 2a succeeds)

Any CER improvement smaller than 3× seed-noise (~0.004) requires 3-seed replication before publication claims. 2 GPUs, 2 configs × seed {123, 777} ⇒ 4 additional runs (batched in 2 × 2 serial).

- GPU 0: best-P²RSE × seed 123, then × seed 777.
- GPU 1: 2nd-best-P²RSE × seed 123, then × seed 777.

Report mean ± 2·SE across 3 seeds.

---

## 11. Phase 4 — Bidirectional transfer (optional, high-upside)

If P²-RSE decisively wins on causal (≤ 0.110 dev CER), the same 2-pole structure can be applied to LION-mode via `bidir_serial`. LION's current best: 0.0711 dev. P²-LION predicted: ~0.065 dev. Would substantially tighten the offline-ASR chapter.

Single run: `stage5_phase4_lion_p2rse_convshift_seed42`. ~7 GPU-hours. Run only after Phase 3 confirms P²-RSE is replicable.

---

## 12. Implementation notes

### 12.1 Code changes (Phase 1)

Confined to `src/models/rwkv6_time_mix.py`, following the Stage-3 RSE pattern:

1. Add `p2rse: bool` flag in `__init__` (line 70 region).
2. Build a second set of `time_theta_base_2`, `time_theta_w1_2`, `time_theta_w2_2` parameters (mirror lines 220–234).
3. Build `W_beta: nn.Linear(D, 2*H)` (real data-dependent mixer).
4. In `_compute_rkv_gw`, compute both $\theta^{(1)}$ and $\theta^{(2)}$ plus $\beta_1, \beta_2$.
5. Write `_forward_recurrent_p2rse`: two calls to a modified `_forward_recurrent_rse_core` (that returns **complex state and real readout per step**, not tuple `(y_real, final_state)`). Reuse the existing chunked scan verbatim per mode.
6. Combine outputs: `y = β₁·y₁ + β₂·y₂ + bonus(u)`. Mask with `torch.where` instead of `-60` clamp.
7. Register `rwkv6_p2rse` and `rwkv6_p2rse_softmax` in `encoder.py::BACKBONES`.
8. Add entries to `configs/experiments.yaml` with Phase-1 IDs.

Engineering estimate: **3–4 hours** to a first trainable version, using `_forward_recurrent_rse` and `_forward_recurrent_rse_multi` as templates. Budget 2 extra hours for smoke-test debugging of the scan, following the Stage-3 protocol (serial reference vs chunked parallel, max abs diff < 1e-4 in FP32).

### 12.2 Parameter budget check

Encoder param counts must stay within 5 % of Stage-4 strong (5,936,384):

- `rwkv6_p2rse`: +1 θ-LoRA pair (~12.4 k per layer) + W_β (~2 k per layer) ≈ +86 k total ≈ **+1.5 %**. Within budget.
- `rwkv6_p2rse_softmax`: same.

Parameter matched enough that any gain is attributable to the mechanism, not capacity.

### 12.3 Launch protocol

```bash
cd experiments/formal_v1

# Phase 1 parallel launch
uv run python scripts/run_experiment.py \
    --backbone rwkv6_p2rse --epochs 30 --seed 42 \
    --output-dir outputs/stage5_01_rwkv6_p2rse_seed42 --gpu 0 &

uv run python scripts/run_experiment.py \
    --backbone rwkv6_p2rse_softmax --epochs 30 --seed 42 \
    --output-dir outputs/stage5_02_rwkv6_p2rse_softmax_seed42 --gpu 1 &

wait

# Then inspect best_dev_cer from the two history.csv files and apply
# the decision table in §6 to pick Phase 2 branch.
```

### 12.4 Resource estimate

- Phase 1: 2 × ~4.5 GPU-hours (30 epochs × 500 s/epoch / 3600). **~9 GPU-hours wall-clock (parallel).**
- Phase 2: 2 × ~4.5 GPU-hours ≈ 9 additional wall-clock hours.
- Phase 3 (if triggered): 4 × 4.5 = 18 GPU-hours, sequentially on each GPU = 9 wall-clock hours.
- Phase 4 (if triggered): 7 GPU-hours on 1 GPU.

**Minimum viable Stage 5 campaign: ~18 wall-clock hours on 2 GPUs if P²-RSE wins straight through.**

---

## 13. What we are *not* doing in Phase 1 (deferred to Phase 3+)

From the prior design discussion, three refinements are deferred:

| Deferred refinement | Reason for deferral |
|---|---|
| Complex γ mixing (quadrature channel) | Gain speculative; confounds the core H★ test. Revisit if Phase 2a succeeds. |
| Complex u bonus term | Mostly absorbable by existing real projections. Low priority. |
| Per-chunk phase preconditioner | Gauge transformation; zero expressivity impact. Only matters if we move to BF16 training (Phase 4+). |

Viscosity coupling (`λ^eff = λ + η|θ|²`) is deferred **unless Phase 1 plateaus** — in which case it becomes the primary Phase 2c candidate.

---

## 14. Success criteria for the full Stage-5 campaign

1. **Go/no-go on P²-RSE:** Phase 1 Exp A reaches dev CER ≤ 0.1160 OR is within marginal band + Phase 2b recovers it.
2. **Mechanism isolation:** Exp B confirms or refutes the softmax-as-bottleneck story — both outcomes are publishable.
3. **Orthogonal stacking:** at least one of Phase 2a C/D breaks dev CER ≤ 0.110 or matches that level.
4. **Seed replication:** Phase 3 confirms the best Phase 2a config has mean dev CER below 0.115 with 3-seed SE < 0.002.
5. **Chapter completeness:** either a ceiling break is documented (P²-RSE wins) or a ceiling confirmation is documented (Phase 2c viscosity falls short), with full mechanism attribution via controls.

Both outcomes constitute a publishable chapter.

---

*Draft: Stage 5 plan, pre-experimental. Awaiting Phase-1 implementation + launch.*
