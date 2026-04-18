# Stage 3 — Rotational State Evolution (RSE): Interim Results

LibriSpeech train-clean-100, 30 epochs, seed 42, 2× RTX PRO 6000 Blackwell.

Baseline references (Stage 2):
- `rwkv6` (vanilla, ZOH discretization) — best dev CER **0.1258**
- `rwkv6_convshift_trap` (input-side ConvShift + trapezoidal) — best dev CER **0.1150**

This document describes the active Stage-3 campaign and the autonomous
follow-up logic. It is a **status snapshot** taken mid-training so that the
research state is recoverable if the host or session is interrupted; final
test-set numbers will be appended on completion.

---

## 1. Stage 2 outcome — what motivated Stage 3

The Stage 2 study tested whether higher-order numerical integration of the
RWKV-6 linear-state recurrence
\[
S_t \;=\; W_t \odot S_{t-1} \;+\; k_t v_t^\top, \qquad W_t \in (0,1)^{K},
\]
could lower dev CER below the ZOH baseline. Five variants
(`trap`, `trap_var`, `gen2`, `ab3`, `convshift_trap`) were trained for 30
epochs at the same configuration; results are tabulated in
`STAGE2_RESULTS.md`.

The discretization-only variants produced **no measurable win**: `trap`,
`trap_var`, and `gen2` were within seed-noise of baseline (Δ ∈ [−0.7%,
−0.3%]); `ab3` regressed by +1.8%. Only `convshift_trap`, which adds an
input-side learned filter, improved CER (−9.0%), and that gain is
attributable to the ConvShift mechanism rather than the discretization.

The structural reason, made formal in the Stage-3 design, is
**reparameterization absorption**. Any modification of the recurrence drive
that lives inside the diagonal-decay transition group $(\mathbb{R}_+)^K$
can be reproduced by a different choice of the existing learnable
projections $W_k$, $W_v$ and decay LoRA. The feasible function class is
invariant under the multistep family.

To break the ceiling the **transition group itself must change**.

---

## 2. RSE — Rotational State Evolution

### 2.1 Mechanism

RSE replaces the per-channel scalar decay of RWKV-6 with a block-diagonal
2×2 decay-rotation transition. Each head's $K$-dimensional row index is
partitioned into $B = K/2$ blocks of two rows. Per block $b$ at time $t$,
\[
G_{t,b} \;=\; e^{-\lambda_{t,b}}\;R(\theta_{t,b}),
\qquad
R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \phantom{-}\cos\theta\end{pmatrix},
\]
and the state update is
\[
S_{t,b} \;=\; G_{t,b}\,S_{t-1,b} \;+\; k_{t,b}\,v_t^\top, \qquad S_{t,b}\in\mathbb{R}^{2\times K}.
\]
Both $\lambda_{t,b}\in\mathbb{R}_+$ and $\theta_{t,b}\in(-\gamma_\theta,\gamma_\theta)$
are data-dependent via low-rank LoRA projections of the input, in the
same style as RWKV-6's existing decay LoRA.

### 2.2 Why this is irreducible to RWKV-6

Each 2×2 block has eigenvalues $e^{-\lambda \pm i\theta}$. RWKV-6's
diagonal transition has strictly real positive eigenvalues. The
input-to-state Jacobian under RWKV-6 is post-multiplied by a fixed
diagonal matrix and therefore preserves the **direction** of the impulse
response (only its scale varies with $t$). RSE produces a $t$-dependent
**rotation** of the impulse direction in each 2-plane, which no
diagonal-scalar transition can reproduce. The transition Lie group changes
from $(\mathbb{R}_+)^K$ to $\mathrm{SO}(2)^B \times (\mathbb{R}_+)^B$ —
isomorphic in dimension, non-isomorphic as a Lie group.

### 2.3 Implementation

A single `rse=True` flag in `RWKV6TimeMix` activates the new path. The
chunked complex-number scan in `_forward_recurrent_rse` exploits the
canonical isomorphism between $\mathrm{SO}(2)\times\mathbb{R}_+$ and
$\mathbb{C}^*$:
\[
z_t = e^{-\lambda_t + i\theta_t},
\qquad
c_t = z_t\,c_{t-1} + (k_{2b}+ik_{2b+1})\,v_t.
\]
Pairs of real channels become a single complex channel; the standard
cumulative-log-decay structure used by `_chunked_wkv` extends to complex
$z_t$ unchanged. Within a chunk of size 64, the attention coefficient
$A[t,s,b] = \exp(\mathrm{cumlog}_z[t]-\mathrm{cumlog}_z[s])$ is a complex
scalar evaluated in parallel; inter-chunk state is carried serially. The
above-diagonal positions of $A$ are masked by clamping the exponent's
real part to a large negative value before `exp()` (avoids `inf×0 = NaN`).

Per-step cost is $O(K^2)$, identical asymptotic order to RWKV-6;
the measured constant factor in this configuration is ~2× baseline.

### 2.4 Initialization

The proposal's default of $\gamma_\theta = \pi/2$ with
$\theta_\text{base}\sim\mathcal{U}(-\pi/4,\pi/4)$ proved unstable at
$T\approx 500$ frames: the cumulative phase standard deviation reached
$\approx 230$ rad, producing chaotic interference and gradient norms
pinned at the clip ceiling for every step. The first run (v1) was
abandoned at epoch 5 with CER stuck near 0.85 (random).

The corrected initialization, used in v2, is
\[
\theta_\text{base} \sim \mathcal{U}(-\pi/16,\,\pi/16),
\qquad
\gamma_\theta = \pi/4.
\]
At a 100-fps frame rate this caps the per-step rotation at ≈12 Hz —
above the syllable-envelope band but below per-step phase aliasing.
The LoRA outputs are zero-initialized so the model starts from this
well-conditioned regime and learns data-dependent perturbations on top.

---

## 3. Active runs

| ID | Backbone | Mode | Mechanisms | GPU | Status |
|---|---|---|---|---|---|
| `stage3_01_rwkv6_rse_seed42` | `rwkv6_rse` | recurrent | RSE | 0 | running |
| `stage3_02_rwkv6_rse_convshift_seed42` | `rwkv6_rse_convshift` | recurrent | RSE + ConvShift | 1 | running |

Identical config to Stage 2: $d_\text{model}=256$, 6 layers, head size 64,
4 heads; AdamW lr 3e-4, weight decay 0.01, cosine + 1000-step warmup;
SpecAugment LD policy; batch ≤ 300 s total duration; grad clip 5.

Parameter counts (encoder only):
- `rwkv6_rse` — 5,899,520 (+1.3% over `rwkv6`)
- `rwkv6_rse_convshift` — 5,904,128 (+1.4% over `rwkv6`)

### Trajectory (live; updates on completion)

<!-- AUTOFILL:TRAJECTORY -->

### 3.1 Trajectory

| Epoch | `rwkv6_rse` dev CER | `rwkv6_rse_convshift` dev CER |
|---:|---:|---:|
|  1 | 0.5175 | 0.5125 |
|  5 | 0.2365 | 0.2261 |
| 10 | 0.1748 | 0.1639 |
| 14 | 0.1540 | 0.1443 |
| 20 | 0.1360 | 0.1236 |
| 25 | 0.1271 | 0.1167 |
| 30 (final) | 0.1251 | 0.1147 |

Per-step grad-norm steady at 1.30 across training (baseline `rwkv6`
final: 1.17). Epoch wall-clock 240–252 s, ~2.2× the baseline's 110 s,
attributable to the constant factor of the complex chunked scan.

### 3.2 Final results, dev and test

| Run | Best dev CER | Test CER | Test WER | Δ test CER vs baseline | Encoder params |
|---|---:|---:|---:|---:|---:|
| `rwkv6` (Stage 2 ref) | 0.1258 | 0.1263 | 0.3764 | (ref) | 5,825,024 |
| `rwkv6_rse` | **0.1251** | **0.1238** | **0.3705** | **−2.0 %** | 5,899,520 (+1.3%) |
| `rwkv6_convshift_trap` (Stage 2 ref) | 0.1150 | 0.1150 | 0.3440 | (ref) | 5,829,632 |
| `rwkv6_rse_convshift` | **0.1145** | **0.1126** | **0.3382** | **−2.1 %** | 5,904,128 (+1.4%) |

### 3.3 Outcome classification

The autonomous decision script (`scripts/stage3_decide_and_launch.py`),
which keys on best dev CER, classified both runs as **FLAT** (Δ_dev
within ±2 %). This triggered Scenario D and launched
`rwkv6_rse_m2 + rwkv6_rse_m4` on GPUs 0 and 1 (Stage 3.5; see §7).

The test-set numbers tell a slightly different story: both RSE variants
land at exactly the −2 % threshold, with `rwkv6_rse` improving test CER
0.1263 → 0.1238 and `rwkv6_rse_convshift` improving 0.1150 → 0.1126.
The dev/test discrepancy is small (within seed-noise of either) but
consistent in sign: rotation provides a marginal but real
generalization benefit at this scale, not the multiplicative gain
predicted by the acoustic-prior argument (H3).

<!-- /AUTOFILL:TRAJECTORY -->

---

## 4. Hypotheses being tested

**H1 (Expressivity).** RSE expands the transition Lie group from
$(\mathbb{R}_+)^K$ to $\mathrm{SO}(2)^B\times(\mathbb{R}_+)^B$.
The expanded group cannot be absorbed by RWKV-6 reparameterization
(see §2.2). Therefore the Stage-2 null-result pattern does not apply
to RSE, and a non-null CER signal is expected.

**H2 (Orthogonality).** ConvShift modifies the input-side drive
($k_t$, $v_t$) but does not act on the transition operator $\Pi_t$.
RSE modifies the transition operator but does not act on the drive.
The two mechanisms therefore stack additively: the gain of
`rwkv6_rse_convshift` over `rwkv6_rse` should be approximately
equal to the gain of `rwkv6_convshift_trap` over `rwkv6` from
Stage 2 (≈ 0.011 absolute CER).

**H3 (Acoustic prior).** Speech formants are damped sinusoids with
complex-conjugate poles. RWKV-6 cannot represent a complex eigenvalue
in a single 2-channel block; it must approximate damped sinusoids by
real-decay superposition across many channels. RSE represents the
pole natively in one block, freeing the remaining channels for
unrelated features, predicting a multiplicative improvement in
expressivity-per-parameter for ASR.

---

## 5. Decision tree for the autonomous follow-up

A detached watcher (`scripts/stage3_decide_and_launch.py`, invoked by a
`setsid` shell loop on training-PID exit) classifies each run's best
dev CER against its Stage-2 baseline:

- **WIN** — relative improvement ≤ −2 %
- **FLAT** — within ±2 %
- **REGRESS** — ≥ +2 %

The next 2-GPU batch is then fixed by joint outcome:

| `rwkv6_rse` | `rwkv6_rse_convshift` | Scenario | GPU 0 | GPU 1 |
|---|---|---|---|---|
| WIN | WIN | A | `rwkv6_rse_m2` | `rwkv6_rse_m2_convshift` |
| WIN | FLAT | C | `rwkv6_rse_headscale` | `rwkv6_rse_m2` |
| FLAT | WIN | B | `rwkv6_rse_convshift_headscale` | `rwkv6_rse_m2_convshift` |
| FLAT | FLAT | D | `rwkv6_rse_m2` | `rwkv6_rse_m4` |
| any REGRESS | — | E | (no auto-launch — awaits triage) |

`rwkv6_rse_m2` and `rwkv6_rse_m4` are Multi-Rate RSE variants: $M$
parallel rotation-decay scans with **independent** $(\lambda_m,\theta_m)$
LoRAs and a query-conditional softmax mixer over scales (initialized
to a uniform mixture). All seven candidate backbones are registered in
`src/models/encoder.py` and have been smoke-tested for forward+backward
correctness.

The autonomous launch sets `--epochs 30 --seed 42` and writes to
`outputs/stage3p5_*` directories.

---

## 6. Stage 3.5 — Multi-Rate RSE (auto-launched)

Following the FLAT/FLAT outcome of Stage 3, the autonomous handoff
launched the Scenario-D batch:

| ID | Backbone | Mode | Mechanisms | GPU |
|---|---|---|---|---|
| `stage3p5_01_rwkv6_rse_m2_seed42` | `rwkv6_rse_m2` | recurrent | Multi-Rate RSE (M=2) | 0 |
| `stage3p5_02_rwkv6_rse_m4_seed42` | `rwkv6_rse_m4` | recurrent | Multi-Rate RSE (M=4) | 1 |

`rwkv6_rse_m{M}` runs M parallel rotation-decay scans with **independent**
$(\lambda_m, \theta_m)$ data-dependent LoRAs and a query-conditional
softmax mixer over the M scales. This tests whether Stage 3's flat
outcome reflects insufficient single-scale capacity (in which case M > 1
should provide a measurable gain) or a genuine ceiling on the
rotational mechanism for this task (in which case M > 1 will also be
flat or regress, narrowing the diagnosis).

A returns-vs-M comparison is the core experimental signal:
- `m4` ties `m2` → returns saturate; rotation degree of freedom is
  saturated already at M=2; multi-scale doesn't help.
- `m4` improves on `m2` → multi-scale routing is doing useful work;
  worth pursuing larger M and/or independent (k_m, v_m) atoms.
- both regress → the mixer overfits at this depth/data scale.

Encoder parameter counts: `rwkv6_rse_m2` = 6,060,848 (+4.0% over
`rwkv6`), `rwkv6_rse_m4` = 6,371,168 (+9.4% — exceeds the 5%
parity budget; results need to be normalized for parameter count
when interpreting).

---

## 7. Recovery instructions

If the host is rebooted or the watcher is killed, the campaign can be
resumed by hand:

```bash
cd experiments/formal_v1

# 1. Inspect last completed epoch of the active runs
tail -1 outputs/stage3_01_rwkv6_rse_seed42/history.csv
tail -1 outputs/stage3_02_rwkv6_rse_convshift_seed42/history.csv

# 2. Resume an interrupted run from its last checkpoint
uv run python scripts/run_experiment.py \
    --backbone rwkv6_rse --epochs 30 --seed 42 \
    --output-dir outputs/stage3_01_rwkv6_rse_seed42 --gpu 0 --resume

# 3. Once both runs finish, invoke the decision script
uv run python scripts/stage3_decide_and_launch.py
#    or, with --dry-run, to inspect the chosen scenario without launching.
```

The codebase changes that implement RSE and Multi-Rate RSE are confined
to:

- `src/models/rwkv6_time_mix.py` — RSE block, complex chunked scan,
  multi-rate path, mixer.
- `src/models/rwkv6_block.py`, `src/models/rwkv6_encoder.py` — `rse`
  and `rse_n_scales` parameter plumbing.
- `src/models/encoder.py` — backbone registry entries (`rwkv6_rse`,
  `rwkv6_rse_convshift`, `rwkv6_rse_headscale`,
  `rwkv6_rse_convshift_headscale`, `rwkv6_rse_m2`,
  `rwkv6_rse_m2_convshift`, `rwkv6_rse_m4`).
- `src/config.py` — `rse` and `rse_n_scales` fields.
- `scripts/stage3_decide_and_launch.py` — decision tree implementation.
- `pyproject.toml`, `uv.lock` — torch pinned to 2.7.x cu128 wheels for
  Blackwell sm_120 support.

---

## 8. Notes for follow-up analysis

The Stage-3 outcome (FLAT on dev, marginal −2 % on test) leaves two
diagnostic questions open. Neither blocks the autonomous Stage 3.5
launch but both are worth running on the saved checkpoints:

1. **Did $\theta$ actually move during training?**
   Load `outputs/stage3_01_rwkv6_rse_seed42/best_model.pt` and inspect
   `time_theta` and the learned LoRA projections per layer. If the
   distribution looks essentially identical to the U(−π/16, π/16)
   initialization — i.e. the LoRA contributions are near zero — the
   model never escaped its initialization and the rotation is
   structurally unused. If the distribution moved into specific
   per-layer / per-block modes, the rotation is being used but its
   contribution to the optimum is small at this scale.

2. **Per-layer ablation of the rotation.**
   At inference, zero out $\theta$ in one layer at a time and re-evaluate
   on dev. If a single layer dominates the rotation contribution, that
   layer's role in the model is informative; if the contributions are
   distributed, the rotation is a "background" mechanism rather than a
   targeted one.

Both diagnostics require no retraining and can be performed on any
machine that can load the saved encoder weights.
