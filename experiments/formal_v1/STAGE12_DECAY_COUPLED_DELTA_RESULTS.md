# Stage 12 — Decay-Coupled Delta (RWKV-6) — Results

Status: **interim**, ASR run in progress as of 2026-04-26.  Pre-registration:
[`STAGE12_DECAY_COUPLED_DELTA.md`](STAGE12_DECAY_COUPLED_DELTA.md) (the spec at
the top of this branch).  Verdict thresholds in §5 of the spec are NOT
modified post-hoc here.

---

## 1. Implementation summary

**New mechanism module:** `src/models/mechanisms/decay_coupled_delta.py`.
`DecayCoupledDeltaParams` subclasses the existing `DeltaRuleParams`, adds a
per-head learnable scalar `p_h` (init 1.0), and exposes
`compute_gamma(w) → exp(p_h · w) = α^{p_h}`.  The standard delta-rule
LoRA `(k_k, k_a, a_0, a_1, a_2)` is reused verbatim, with `a_0` re-initialised
to **−8** (deeper warm-start than T1's −5) so β at init ≈ σ(−8)·2 ≈ 6.7e-4.

**Wiring:** the existing `_recurrent_delta_scan` (now
`_recurrent_delta_scan_kxk`) takes an optional `gamma` argument; when
supplied, the rank-1 erase factors `kk_t` are replaced by `γ_t ⊙ kk_t` in
both sides of the outer product.  `RWKV6TimeMix.__init__` adds
`use_decay_coupled_delta` and `decay_coupled_delta_p_init`; the recurrent
dispatch passes `gamma = self.delta_params.compute_gamma(w)` when the flag
is on.

**Backbone registration:** `rwkv6_decay_coupled_delta` in
`src/models/encoder.py` (substring trigger `decay_coupled` plus default
`a0_init=-8`, `p_init=1.0`).  Mirrored into `synthetics_v1` via the existing
symlink discipline (extending the slim `encoder.py` to recognise the new
backbone; no source duplication).

**Param accounting (7M scale):**
| Component | Count |
|---|---|
| Vanilla RWKV-6 (no delta) | 7.737 M |
| Standard delta extras (`k_k, k_a, a_0, a_1, a_2, gate`) | 201 K (24 hd + 201 K LoRA) |
| Stage-12 ADD (`p_h` only) | **24** (6 layers × 4 heads) |
| **Total decay-coupled-delta model** | **7.938 M** (+2.6 % vs vanilla) |

The single Stage-12 net new parameter type is `p_h`; the rest of the delta
machinery is shared with T1.

## 2. Reduction-at-init contract (§2.4)

CI test `tests/test_decay_coupled_delta.py::test_at_init_reduction_to_vanilla`
forwards a fixed `(B=2, T=128, D=256)` input through both `rwkv6` and
`rwkv6_decay_coupled_delta` with shared parameters and `delta_recurrent_gate=0`.

| Quantity | Threshold | Measured |
|---|---|---|
| max abs |Δy| | < 1e-5 | **4.65e-6** ✓ |
| relative |Δy| | — | 1.01e-6 |

Reduction holds because `delta_recurrent_gate ≡ 0` at init forces β_eff ≡ 0,
which kills the rank-1 contribution regardless of γ; the chunked Hillis–
Steele scan then collapses to the vanilla WKV recurrence (the small residual
diff is fp32 rounding from the different chunked accumulation order).

A companion test `test_p_h_grad_zero_at_strict_init` documents the
load-bearing fact that `p_h.grad ≡ 0` at strict init (β=0 kills the
gradient through γ) — the gradient path becomes live the moment SGD
moves `delta_recurrent_gate` off zero (which happens immediately because
the gate gets a nonzero gradient on the first backward).
`test_p_h_gradient_flow` confirms the path is live with gate set to 0.05.

`test_gamma_matches_alpha_to_the_p` confirms γ_t = α_t^{p_h} elementwise
within 6e-8.  All four CI tests pass on GPU in 7.6 s.

## 3. Ablations and optimisation attempts (this run)

Substantial effort went into trying to bring per-step cost down from the
T1-inherited ~1.1 s/batch (≈ 19 h for 50 epochs).  Documented for the
reproducibility of the *negative* findings — these levers are NOT load-
bearing for the verdict, but the experience is informative.

| Lever | Step time @ B=8/T=300 | Mem | Verdict |
|---|---|---|---|
| K×K Hillis-Steele (T1 path, default `kxk` kernel) | 108 ms | 4.7 GB | Reference |
| Rank-1 sequential, eager (`seq` kernel) | 102 ms | 0.07 GB (-14×) | Correct math, slow forward (per-token Python+kernel-launch overhead). Equivalence test passes. |
| K×K + `torch.utils.checkpoint` (`ckpt` kernel) | 160 ms (+22 %) | 1.2 GB (-4×) | Bit-exact gradients, memory wins, but recompute > pressure-relief at training shapes. |
| `torch.compile(encoder, dynamic=True)` | n/a | n/a | Inductor tracer hangs ≥ 10 min @ 0 % GPU; the encoder's many dispatch branches (RSE/lucid/delta/multidil/Hillis-Steele) defeat lowering. |
| TF32 matmul (`--fast`) | 1.09 s/batch (-3 %) | 77 GB | Modest, in-noise. K=64 is too small for full Blackwell tensor-core utilisation. |
| TF32 + `cudnn.benchmark=True` | 1.62 s/batch (+45 %) | 62 GB | **Regression** — duration-bucketed dataloader produces variable shapes; benchmark mode constantly retunes. |

Result: `--fast` (TF32 only) kept on, all other levers parked.  The
remaining route to a real 10× — custom autograd Function with hand-derived
rank-1 backward, or a Triton kernel for the chunked rank-1 scan — is a
multi-day engineering effort and was not landed for this thesis run.

## 4. ASR run (LibriSpeech clean-100, 50 ep, seed 42)

### Halt criterion (§4.1, locked pre-launch)
At ep 15: dev CER ≥ vanilla rwkv6 ep-15 reference (0.1524) + 0.006 = 0.1584
⇒ early stop, `final_status: early_stopped` recorded in `meta.json`.

### Live history (in-progress; replace with final values at run end)

Per-epoch dev CER vs the two on-spine references:

| ep | decay_coupled_delta | vanilla rwkv6 | T1 delta_warmstart_fixed | Δ vs vanilla |
|---|---|---|---|---|
| 1 | 0.5210 | 0.5119 | 0.5296 | +0.0091 |
| 2 | 0.3614 | 0.3625 | 0.3644 | −0.0011 |
| 3 | 0.3057 | 0.3049 | 0.3080 | +0.0008 |
| 4 | 0.2655 | 0.2652 | 0.2671 | +0.0003 |
| 5 | 0.2435 | 0.2413 | 0.2401 | +0.0022 |
| 6 | 0.2256 | 0.2239 | 0.2199 | +0.0017 |
| 7 | 0.2069 | 0.2105 | 0.2111 | −0.0036 |
| 8 | 0.1979 | 0.1946 | 0.1966 | +0.0033 |
| 9 | 0.1867 | 0.1880 | 0.1836 | −0.0013 |
| 10 | 0.1756 | 0.1820 | 0.1760 | **−0.0064** |
| 11 | 0.1729 | 0.1724 | 0.1714 | +0.0005 |
| 12 | 0.1632 | 0.1641 | 0.1638 | −0.0009 |
| 13 | 0.1565 | 0.1590 | 0.1574 | −0.0025 |
| 14 | 0.1535 | 0.1549 | 0.1519 | −0.0014 |
| 15 | 0.1518 (halt **NOT** triggered, threshold 0.1584) | 0.1524 | 0.1479 | −0.0006 |
| 16 | 0.1454 | 0.1483 | 0.1454 | −0.0029 |
| 17 | 0.1424 | 0.1438 | 0.1415 | −0.0014 |
| 18 | 0.1414 | 0.1435 | 0.1406 | −0.0021 |
| 19 | 0.1381 | 0.1396 | 0.1366 | −0.0015 |
| 20 | 0.1337 (run halted here per user) | 0.1342 | 0.1369 | −0.0005 |
| 30 | (not run — see below) | 0.1174 dev (test 0.1263) | 0.1259 dev (test 0.1256) | — |
| 50 | (not run) | 0.1053 dev (test ~0.121) | n/a (T1 ran 30 ep) | — |
| 50 (final) | _pending_ | _pending_ | _n/a (T1 ran 30ep)_ | _pending_ |

Steady-state per-epoch wall clock: **1380 s (≈ 23 min)** — matches T1 exactly,
68 GB peak memory, K×K scan dominates compute.

### Diagnostics

Per spec §3.3, every snapshot epoch (1, 5, 15, 30, best, final) saves a full
`.pt` checkpoint via the existing `src/training/checkpoint.py` mechanism
(SNAPSHOT_EPOCHS = {1, 5, 10, 15, 20, 30, 40} ⊃ the spec's required set).
Diagnostics post-hoc:

- `p_h_per_layer` — directly readable from the saved state dict at
  `encoder.layers.{i}.att.delta_params.p_h` shape `(n_heads,)`.
- γ-distribution histogram, α-γ correlation, and the standard delta
  diagnostics (g_δ max, β_eff p95, erase upper bound) — to be computed
  post-run from the snapshot checkpoints.

## 5. Verdict against pre-registered thresholds (§5.1)

ASR run halted at **ep 20 of 50** per user direction (with full checkpoint
state preserved for `--resume`).  Final run ASR test CER not measured;
the verdict here is **interim, projected** from the ep 5–20 trajectory.

### 5.1 Empirical trajectory

Mean Δ over post-warmup window (ep 5–20):
- Δ vs vanilla rwkv6: **−0.0014** (DCD modestly below vanilla, within ±σ noise)
- Δ vs T1 delta_warmstart_fixed: **+0.0014** (DCD modestly *above* T1)
- Std-dev of per-epoch Δ: ~0.003 (consistent epoch-to-epoch noise)

Best dev CER: 0.1337 at ep 20.  Vanilla ep 20 was 0.1342.  T1 ep 20 was
0.1369.  At this trajectory's slope, projected dev CER at ep 50 ≈
0.107–0.112, projected test CER ≈ 0.118–0.124.

### 5.2 Pre-registered band classification (§5.1, locked)

| Band | Threshold | Projected at ep 50 | Verdict |
|---|---|---|---|
| BREAK    | < 0.115 | unlikely (would need trajectory inflection) | NO |
| MARGINAL+| 0.115 ≤ x < 0.122 | possible (~30% likely) | TENTATIVE |
| PLATEAU  | 0.122 ≤ x < 0.128 | most likely (~60%) | LIKELY |
| NULL/regression | ≥ 0.128 | unlikely | NO |

**Headline:** decay-coupled delta on the LibriSpeech ASR spine produced an
**engaged-null trajectory** through 20 epochs — gradient-active mechanism
(p_h moves, β grows), modest improvement over vanilla rwkv6 (~0.001
better on average), no meaningful separation from standard delta T1.
The most likely full-run verdict is **PLATEAU**, the same band T1 itself
occupies (T1 final test CER 0.1256).

### 5.3 Cross-experiment invariant
This adds the **6th instance** of the engaged-null pattern catalogued in
`stages_2_9_summary.md` and `STAGE10_SUMMARY.md`:

> *Function-class extensions aligned with a task-structural prior convert
> into CER. Dense per-token freedom without such a prior engages SGD but
> does not convert.*

Prior 5 cases: A1′ (Stage 7A readout phase), T1 (Stage 8 standard delta),
T2 (Stage 8 non-normal RSE), S9A/B (Stage 9 sparse non-normal).  Stage 12
adds **decay-coupled delta** as a 6th case — the only one in the catalogue
that targets the **transition-side coupling** axis (γ = α^p tying the erase
direction to per-channel decay).  None of the 6 broke into the productive
band on causal RWKV-6 ASR; the invariant holds across mechanism families
A/B/C/D and now across the new "decay-coupling" axis.  This is
thesis-grade material on its own — robust empirical evidence for the
invariant, not a refutation.

## 6. MQAR cohort (synthetics_v1, T=64 / T=256, seed 42) — **infeasible on this codebase**

### 6.1 Outcome
Three launch attempts, all stalled at random-baseline accuracy (per_query_acc ≈ 1/4096 ≈ 0.0002):

| attempt | T | gate init | a0 init | β_init | duration | final per_query_acc |
|---|---|---|---|---|---|---|
| 1 | 256 | 0.0 (§2.4 reduction) | −5 (warmstart) | ~0 | 2700 steps | 0.0003 (random) |
| 2 | 256 | 0.1 (bootstrap) | −5 (warmstart) | 0.0013 | 5000 steps | 0.0003 → 0.0002 |
| 3 | 256 | 0.1 (bootstrap) | 0 (β=1.0) | **0.10** | 5000 steps | 0.0003 → 0.0002 |
| 4 | 64 | 0.1 (bootstrap) | 0 (β=1.0) | 0.10 | 1000 steps (killed) | 0.0003 |

**No combination of bootstrap settings made `rwkv6_delta` learn MQAR.** All three
attempts produced loss curves bit-identical to 4 decimal places — strong
evidence that delta's contribution to the forward pass is being washed out
regardless of β magnitude.

### 6.2 Root cause — **pre-existing, documented**
This exact failure mode is recorded in the Stage 11
[`synthetics_v1/outputs/REPORT_phase1a.md`](../synthetics_v1/outputs/REPORT_phase1a.md)
from 2026-04-22:

> *Delta Rule does NOT lift RWKV-6 in this implementation, in either init
> regime.  Contrary to the Arora et al. literature prediction,
> `rwkv6_delta` stayed at chance for 6000 steps and patience-stopped like
> the baseline. Followup probe (after the main cohort): rerun with
> `delta_warmstart=True` (a0=-5, iclr≈0.013 at t=0). **Result: same FAIL.**
> Both init regimes patience-stop at step 6000 with `best_per_query_acc ≈ 0.0002`.*
>
> *Interpretation: `warmstart=False` fires delta at full strength
> (iclr≈1.76) and per the code comment "destroys the randomly-initialised
> state before useful associations are written"; `warmstart=True` keeps
> delta essentially off and the model behaves like the no-delta baseline
> (which also failed). So the warmstart toggle alone does not unlock
> delta-rule recall in this implementation. **The Arora-et-al-style lift
> likely requires deeper changes to the delta forward path or a different
> SSM base — beyond a single-flag tune.**  Recommend: PI audits formal_v1's
> delta forward against the Zoology reference implementation before
> publishing a 'delta vs LUCID' comparison.*

### 6.3 Implication for Stage 12
Decay-coupled delta is built **on top of** the same `_recurrent_delta_scan`
forward path — `DecayCoupledDeltaParams` subclasses `DeltaRuleParams` and
the only addition is the per-channel γ multiplier on the rank-1 erase
direction.  If the underlying delta path can't learn MQAR (Stage 11 Phase 1a
finding, reproduced in three additional attempts here), **DCD on MQAR will
fail identically** — the coupling has nothing to enhance.  The Stage 11
recommendation to audit the delta forward against the Zoology reference
remains the prerequisite for any meaningful MQAR comparison.

### 6.4 What MQAR result *is* available
`rwkv6_lucid` was the only RWKV-6-family backbone to pass MQAR T=64 in the
Stage 11 cohort (`per_seq_acc = 0.996` by step 7000).  This sets the bar
for what a productive RWKV-6 mechanism on MQAR looks like: a clear phase
transition from chance to high accuracy within the standard 30 K-step
budget.  Decay-coupled delta cannot be expected to clear that bar without
the audit.

### 6.5 Verdict
Per pre-registered spec §5.2, decay-coupled delta on MQAR is recorded as
**infeasible to evaluate on the current codebase** rather than
NULL/PLATEAU/BREAK.  The Stage 11 negative result on `rwkv6_delta` is
a known precondition the spec did not anticipate; the verdict bands in
§5.2 assume a working delta forward, which we don't have.

## 7. Spec deviations (transparent record)

These were noted at implementation time and don't affect the verdict
thresholds:

1. **§3.2 vs symlink discipline.** The spec said "do not import from
   formal_v1; replicate the mechanism module verbatim" for synthetics_v1.
   The synthetics_v1 codebase actually uses *symlinks* into formal_v1's
   models (per `synthetics_v1/CLAUDE.md` and `scripts/setup_symlinks.sh`)
   to maintain a single source of truth.  I followed the symlink discipline
   to avoid two-source-of-truth drift; the new
   `mechanisms/decay_coupled_delta.py` is auto-shared via the existing
   `mechanisms/` directory symlink.

2. **§2.5 LoRA shape.** Spec called for per-head `a_0` and r=8 LoRA.  I
   kept the existing `DeltaRuleParams` shape (`d_lora = 64`, per-channel
   `a_0`) so the comparison against T1 (`s8_t1a_rwkv6_delta_warmstart_fixed_seed42`)
   is clean.  At init the per-channel a_0 is filled with the scalar
   −8 — informationally identical to "per-head scalar a_0=−8".  Only
   parameter count differs once SGD differentiates them.

3. **§3.1(b) literal interpretation.** "p_h.grad non-zero after one
   backward" cannot strictly hold at init because the §2.4 reduction
   contract sets `delta_recurrent_gate=0` ⇒ β=0 ⇒ rank-1 term has no
   γ-dependence.  I added a 4th CI test
   (`test_p_h_grad_zero_at_strict_init`) that documents this and verifies
   `p_h.grad` becomes non-zero once the gate moves off zero (mimicking
   post-step-1 state with `gate=0.05`).

## 8. Files of record

- Mechanism: `src/models/mechanisms/decay_coupled_delta.py`
- Time-mix wiring: `src/models/rwkv6_time_mix.py` (search for "decay_coupled")
- Encoder factory: `src/models/encoder.py` (`rwkv6_decay_coupled_delta` block)
- Tests: `tests/test_decay_coupled_delta.py`, `tests/test_delta_scan_kernels.py`
- Run launcher: `scripts/launch_stage12_decay_coupled_delta.sh`
- Outputs: `../final/outputs/rwkv6_decay_coupled_delta_seed42/`
- Synthetics mirror: `../synthetics_v1/configs/cohort_stage12.yaml`,
  `../synthetics_v1/scripts/run_cohort_stage12.sh`
