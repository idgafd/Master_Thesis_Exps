# Synthetics v1 — Plan (Stage 1: MQAR)

Tier-1 of the broader synthetics tier proposed in `../formal_v1/stage10_feedback.md`-driven
discussion. This document specifies exactly what we run, what configuration we use,
why we picked it, and what would count as success.

---

## 1. Context & motivation

`formal_v1` evaluates every architecture on **one task** (LibriSpeech CTC ASR).
The PI feedback (`../formal_v1/stage10_feedback.md §2.2–2.3`) flagged that the
~0.115 dev-CER floor is **input-representation-bound, not sequence-mixer-bound**.
That undermines the central thesis claim — *closing the expressivity gap
between linear-time models and softmax attention* — because it leaves the
mixer-vs-attention comparison on a benchmark whose bottleneck is the frontend.

MQAR (Multi-Query Associative Recall, Arora et al. 2024) is the canonical
literature benchmark that **does** isolate the sequence mixer's recall
capability. It is the deciding test in Zoology, RWKV-5/6, RWKV-7, Log-Linear
Attention, and the Mechanistic Architecture Design (MAD) suite. Running MQAR
on our 14 backbone variants — with the matched 7 M / 6-layer / d=256 spine
— turns the thesis from *"LION wins on ASR"* into *"LION (and which mechanisms)
close the gap to softmax attention on the task that defines that gap"*.

**This stage's research question:**

> Among our backbone variants, which mechanisms close the MQAR gap to a
> matched-size Transformer, and does that ranking agree with the per-mechanism
> theoretical predictions extracted from the literature?

A specific sub-question of high thesis-value:

> Does our **corrected Delta Rule** (causal-only on LION; standard on causal
> RWKV-6) deliver the MQAR improvement that the literature attributes to
> delta-rule / removal-and-write update structure?

If the answer is yes, the thesis can claim mechanism-level evidence on the
canonical expressivity benchmark — independent of any ASR-specific result.

---

## 2. Literature foundation

### 2.1 What the local literature actually documents

The local PDFs in `../../papers/` were surveyed (full report in
`COST_ESTIMATE.md`-adjacent agent transcript). Findings:

- **No local paper re-documents the original Zoology defaults in full.** Both
  papers that run MQAR (Log-Linear Attention, RWKV-7) reference Arora et al.
  2024 and report only deltas. The Zoology repo is the ground truth for
  vocab size, training-step count, and per-query accuracy definition.
  **Action item:** reproduce against Zoology code before publishing numbers.

- **Per-sequence accuracy with a >99 % threshold** is the Log-Linear and
  RWKV-7 convention. We adopt this.

- **Two MQAR length / KV-pair protocols** appear in the literature:

  | Protocol | Length × KV pairs | Source |
  |----------|-------------------|--------|
  | Single-length sweep over dim | T = 256, KV ∈ {4 … 64}, dim ∈ {16, 32, 64} | Log-Linear §4.1 |
  | Length-and-KV scaling | (T, KV) ∈ {(64,4), (128,8), (256,16), (512,64), (1024,128), (2048,256)} | RWKV-7 Table 6 |

  We adopt the **RWKV-7 length-scaling protocol** for Stage 1 (sequence
  scaling is the relevant axis for our thesis claim about expressivity at
  long context). The Log-Linear dim-sweep is incompatible with our matched
  parameter envelope.

### 2.2 Mechanism-level theoretical predictions (MQAR)

The literature identifies **four ingredients** that materially help MQAR
(Log-Linear §2 p.2–4, RWKV-7 §1 p.3, M2RNN §3.3 p.10):

1. **Large effective state size** — RWKV-7 Table 6 directly correlates MQAR
   ceiling with WKV state dimension (8192 → 65536 walks the table from
   partial to full recovery).
2. **Delta-rule / removal-and-write** — `Sₜ = Sₜ₋₁(I − βₜkₜkₜᵀ) + vₜkₜᵀ`.
   The literature attributes DeltaNet, Gated DeltaNet, RWKV-7 MQAR strength
   to this update structure. **This is the prediction we test.**
3. **Data-dependent gating** — vector-valued in-context learning rate.
4. **Hierarchical / multi-scale memory** — Log-Linear's Fenwick tree.

### 2.3 Predicted hypothesis matrix (synthesized from §2.2)

| Backbone (formal_v1 name) | MQAR (short T) | MQAR (long T) | Reasoning |
|---|:---:|:---:|---|
| `transformer_causal` | pass | pass | softmax recall is the reference |
| `rwkv6` | pass | partial | diagonal decay limits effective state |
| `rwkv6_lucid` | pass | partial | LUCID decorrelates keys → fewer collisions; not in literature, novel test |
| `rwkv6_delta` | pass | **pass (predicted)** | matches the literature's removal-and-write criterion |
| `rwkv6_lucid_delta` | pass | **pass (predicted)** | strongest RWKV-6 variant per theory |
| `mamba` | pass | partial | scalar/vector decay, no explicit removal |
| `mamba2` | pass | partial | larger state but no delta-style removal |
| `lion` | trivial (bidir) | trivial (bidir) | bidirectionality short-circuits causal MQAR; **see §4.3** |
| `lion_delta` | — | — | only meaningful on a causal task variant |

This matrix is **the experiment's pre-registered prediction**. We score the
result by whether the empirical ranking respects the theory; deviations are
interesting findings either way.

---

## 3. Task specification (MQAR canonical)

### 3.1 Task definition

Following Arora et al. 2024:

- A **vocabulary** of `V` tokens, partitioned into a key alphabet `V_K` and a
  value alphabet `V_V` (typically equal halves) plus a small set of structural
  tokens (PAD, BOS).
- An **example** of length `T` consists of `K` key-value pairs interleaved
  with random distractor tokens. After all pairs, `Q` ≤ `K` query positions
  appear, each presenting a previously-seen key and asking for its value.
- The model receives the sequence and must predict the correct value token at
  each query position. **Loss is masked everywhere except at query positions
  (`ignore_index = -100`).**

### 3.2 Configuration adopted for Stage 1

Anchoring on the RWKV-7 length-scaling protocol (Table 6, p.12), with
explicit defaults filled in from Zoology convention:

| Parameter | Value | Source |
|---|---|---|
| Vocabulary size `V` | **8192** | Zoology default; verify against repo |
| Key alphabet `V_K` | first 4096 tokens | half-and-half partition |
| Value alphabet `V_V` | next 4096 tokens | |
| Sequence length sweep `T` | **64, 128, 256, 512** in Stage 1 | RWKV-7 sub-range |
| KV pairs per example `K(T)` | `T/4` rounded up: 16, 32, 64, 128 | RWKV-7 ratio |
| Queries per example `Q` | equal to `K` (one query per pair) | Zoology default |
| Train examples / epoch | 100 000 (regenerated each epoch) | Zoology default |
| Eval examples (fixed seed) | 3 000 per `T` | Zoology default |
| Max training steps | 50 000 | Log-Linear §4.1: "early stop at >99 %" |
| Early-stop criterion | per-sequence accuracy ≥ 99 % on eval, patience 5 × 1 000 steps | Log-Linear / RWKV-7 |
| Reporting metric | **per-sequence accuracy** (all queries correct) and per-query accuracy | RWKV-7 |
| Batch size | adaptive: see `COST_ESTIMATE.md §3.1` | memory-driven |
| Optimizer | AdamW, lr = 3e-4, wd = 0.01 | matches `formal_v1` spine |
| Scheduler | warmup-cosine, 1000-step warmup | matches `formal_v1` spine |
| Seeds | **3 seeds** (matches `formal_v1` multi-seed protocol) | thesis convention |

**Stage-2 length extrapolation** will add `T ∈ {1024, 2048}` and a
*length-generalization* slice (train at `T=512`, evaluate at longer `T`),
which neither Log-Linear nor RWKV-7 reports — that is the gap the thesis
can fill.

### 3.3 Causality discipline

MQAR is meaningless for bidirectional models — they trivially see the
queries while encoding the pairs. Two reporting tracks:

- **Causal MQAR** (primary): backbones with strict left-to-right encoding.
  Cohort: `transformer_causal`, `rwkv6`, `rwkv6_lucid`, `rwkv6_delta`,
  `rwkv6_lucid_delta`, `mamba`, `mamba2`.
- **Bidirectional MQAR** (sanity): `transformer` (full attention), `lion`.
  Reported separately. We expect ≥99 % across the board; if a bidirectional
  variant fails to saturate, that's a bug, not a finding.

For the Delta Rule on LION question, we use a **causal-only LION** variant
(forward-pass only, no anticausal kernel) so we can test `lion_delta` as a
causal model. This is the same causal-subset trick the formal_v1 corrected
Delta Rule already uses. We add this as a backbone variant in
`src/models/synthetic_model.py`.

---

## 4. Experiment cohort & matrix

### 4.0 First-pass reduced cohort (this iteration's runs)

We start small to de-risk the infrastructure and the literature reproduction
before committing GPU-time to the full sweep. The reduced cohort is:

- **8 backbones × 2 lengths × 1 seed = 16 runs**
- **Lengths:** `T = 64` (warm-up / sanity) and `T = 256` (representative
  middle of the literature length range)
- **Seed:** `42` only (formal_v1 convention; multi-seed deferred to expansion)

**Run order (staircase, fail-fast):**

1. **Phase 0 — smoke test (~8 min total).** All 8 backbones at `T=64`, batch
   32, **only 200 steps** each. The point is to verify every backbone
   instantiates, forwards, backwards, and produces non-trivial loss
   reduction. NO accuracy claim is made from Phase 0 — it is an
   infrastructure check. `scripts/debug_run.py`.
2. **Phase 1a — short-context convergence (~30 min total).** All 8 backbones
   at `T=64` with full early-stopping (max 30k steps, ≥99 % per-sequence
   accuracy threshold). `scripts/run_experiment.py` driven by
   `configs/cohort_reduced.yaml`.
3. **Phase 1b — medium-context convergence (~1 GPU-hour total).** All 8
   backbones at `T=256`, same early-stop protocol.

**Total wall-clock estimate:** **~1.5 GPU-hours** end-to-end.

**Decision rule for expanding to the full cohort (§4.1):**

- If Phase 1 returns a **clean ranking** (Transformer at top, RWKV-6 + Delta
  Rule lifted above plain RWKV-6, Mamba/Mamba-2 in the middle), expand to the
  full 96-run Cohort A and 36-run Cohort B.
- If Phase 1 returns an **unexpected ranking** or a backbone fails to
  converge, debug at this scale before expanding — no point burning the full
  budget on a buggy substrate.

### 4.1 Cohort A — Architecture comparison (full expansion target)

The full sweep, to be run only after Phase 1 of §4.0 returns a clean ranking.
8 backbones × 4 lengths × 3 seeds = **96 runs**.

| # | Backbone | Mechanism flag | What it tests |
|---|---|---|---|
| A1 | `transformer_causal` | — | softmax reference |
| A2 | `rwkv6` | — | RWKV-6 baseline (diagonal decay) |
| A3 | `rwkv6_lucid` | LUCID | key decorrelation effect on recall (novel) |
| A4 | `rwkv6_delta` | Delta Rule | the literature-predicted MQAR booster |
| A5 | `rwkv6_lucid_delta` | LUCID + Delta | combined |
| A6 | `mamba` | — | Mamba baseline |
| A7 | `mamba2` | — | Mamba-2 baseline |
| A8 | `transformer` | — (bidirectional) | sanity — should saturate |

### 4.2 Cohort B — LION on a causal MQAR variant (secondary)

Only run if Cohort A confirms the Delta Rule prediction (otherwise the
question is moot). 3 backbones × 4 lengths × 3 seeds = **36 runs**.

| # | Backbone | What it tests |
|---|---|---|
| B1 | `lion_causal` | bidirectional kernel, causal-masked at runtime |
| B2 | `lion_causal_delta` | does causal Delta Rule transfer to LION? |
| B3 | `lion_causal_lucid_delta` | full-stack on causal-LION |

### 4.3 Total Stage-1 budget (by phase)

| Phase | Runs | Wall-clock |
|---|---|---|
| §4.0 reduced (Phase 0 + Phase 1) | 16 + 8 smoke | **~1.5 GPU-hours** |
| §4.1 full Cohort A expansion | 96 | ~3–4 GPU-hours |
| §4.2 Cohort B (conditional on A confirming Delta Rule) | 36 | ~6 GPU-hours |
| **Stage-1 total at full expansion** | **148** | **~10–12 GPU-hours** |

Comfortably under one GPU-day if everything fires; the reduced first-pass
costs almost nothing and gives us the go/no-go signal for the rest.

---

## 5. Success criteria & decision rules

We borrow the formal_v1 verdict vocabulary (BREAK / MARGINAL / PLATEAU /
REGRESSION) but rebind it to MQAR semantics:

| Verdict | Definition (per-sequence accuracy at the target T) |
|---|---|
| **PASS** | ≥ 99 % at T (Log-Linear / RWKV-7 threshold) |
| **PARTIAL** | 70 % ≤ acc < 99 % |
| **FAIL** | < 70 % |
| **BREAK** | A mechanism upgrade lifts a backbone from FAIL/PARTIAL to PASS at a length where the base variant did not |
| **MARGINAL** | Mechanism lifts accuracy by ≥ 5 pp without crossing a verdict boundary |
| **PLATEAU** | Within ±2 pp of the base variant |
| **REGRESSION** | More than 2 pp below the base variant |

**Headline thesis claim is supported if:**

1. `transformer_causal` is PASS at all four lengths (sanity).
2. `rwkv6_delta` is PASS at lengths where `rwkv6` is PARTIAL/FAIL → BREAK
   verdict for Delta Rule, matching the literature prediction.
3. `rwkv6_lucid` does *not* recover the same gap → distinguishes the
   delta-rule mechanism from generic key processing.
4. The ranking `rwkv6 < rwkv6_lucid ≤ rwkv6_delta ≤ rwkv6_lucid_delta ≤ transformer_causal`
   holds at the longest length tested.

If any of (1)–(4) fail, that is a publishable finding: either our Delta
Rule implementation does not match the literature's update structure (bug
hunt), or the literature's mechanism prediction does not survive at our
matched parameter envelope (substantive negative result).

---

## 6. Engineering plan

Mirroring the formal_v1 stage breakdown.

### 6.1 Stage 1.0 — Infrastructure (this PR)

- [x] Directory skeleton, `pyproject.toml`, `.gitignore`, `README.md`
- [x] `CLAUDE.md`, `PLAN.md` (this doc), `COST_ESTIMATE.md`
- [x] `scripts/setup_symlinks.sh` (mirror backbones from formal_v1)

### 6.2 Stage 1.1 — Data pipeline (~3 h)

- [ ] `src/tasks/mqar.py` — pure-tensor MQAR generator. Public API:
  ```python
  def generate_mqar_batch(
      batch_size: int,
      seq_len: int,
      n_kv_pairs: int,
      vocab_size: int = 8192,
      key_vocab_size: int = 4096,
      value_vocab_size: int = 4096,
      n_queries: int | None = None,  # default = n_kv_pairs
      device: torch.device = "cpu",
      generator: torch.Generator | None = None,
  ) -> tuple[Tensor, Tensor]:  # (input_ids: B×T, targets: B×T)
  ```
- [ ] `src/data/dataset.py` — `MQARDataset` (IterableDataset, regenerates each
  epoch from a per-epoch seed) + `MQAREvalDataset` (fixed seed)
- [ ] `src/data/vocab.py` — trivial `TokenVocab(V)` with PAD=0
- [ ] `tests/test_mqar_generator.py` — verify (a) all queries reference
  in-sequence keys, (b) targets at non-query positions are -100, (c)
  reproducibility under a fixed seed

### 6.3 Stage 1.2 — Model wrapper (~1 h)

- [ ] `src/models/synthetic_model.py` — `SyntheticModel(d_model, vocab_size, cfg)`:
  `nn.Embedding → build_encoder(cfg) → LayerNorm + Linear`
- [ ] `src/models/causal_lion.py` — runtime causal mask over LION's parallel
  attention kernel (for Cohort B)
- [ ] `tests/test_synthetic_model.py` — verify (a) shape contract
  `(B,T) → (B,T,V)`, (b) zero-regression at init for each backbone

### 6.4 Stage 1.3 — Training loop adaptation (~2 h)

- [ ] `src/training/train.py` — fork formal_v1, replace `nn.CTCLoss` with
  `nn.CrossEntropyLoss(ignore_index=-100)` and parameterize so we can swap
  losses cleanly later
- [ ] `src/training/evaluate.py` — per-sequence and per-query accuracy
- [ ] `src/config.py` — `SyntheticsConfig` dataclass (audio fields removed,
  MQAR fields added)
- [ ] `configs/default.yaml` — Stage-1 anchor config

### 6.5 Stage 1.4 — Smoke test & benchmark (~2 h)

- [ ] `scripts/debug_run.py` — 200 steps × all 8 backbones × T=128. Verify:
  (a) every backbone trains, (b) per-step latency matches `COST_ESTIMATE.md §3.2`
  predictions to within 2×, (c) Transformer hits ≥80 % at T=128 in 200 steps
- [ ] `scripts/benchmark_generator.py` — replace `COST_ESTIMATE.md §1.2`
  estimates with measurements
- [ ] Update `COST_ESTIMATE.md §5` with measured numbers

### 6.6 Stage 1.5 — Cohort A sweep

- [ ] `scripts/run_experiment.py` (forked from formal_v1) wired up
- [ ] Cohort A: 96 runs (8 × 4 × 3-seed)
- [ ] `scripts/analyze_mqar.py` — per-position recall plots, per-backbone
  length-vs-accuracy curves, theory-vs-empirical ranking table
- [ ] **Stage-1 RESULTS.md** — verdict per backbone × length, with
  reference to `formal_v1/RESULTS.md` formatting

### 6.7 Stage 1.6 — Cohort B (conditional)

- Only fires if Cohort A confirms Delta Rule prediction. 36 runs,
  ~6 GPU-hours.

---

## 7. Stage 2 preview — state tracking (not implemented yet)

Documented here so the data layout and infra in Stage 1 are forward-compatible.

- **Tasks:** parity (S2), modular addition mod-7, S3 / S4 / S5 group word problems
  (Merrill et al. 2024 protocol; matched against M2RNN §3.2 and RWKV-7 §7.6).
- **Length protocol:** train at `T=128`, evaluate up to `T=512`
  (M2RNN's length-generalization protocol — the gap that DeltaProduct fails on).
- **Hypothesis matrix:** TC0 vs NC1 separation. Transformer / RWKV-6 / Mamba
  predicted to FAIL S3+; only mechanisms with negative eigenvalues or
  non-diagonal transitions can pass. **We do not currently have such a
  mechanism in formal_v1** — Stage 2 may motivate adding RWKV-7-style
  generalized delta or Householder-product transitions.
- **Budget estimate:** similar to Stage 1, ~15 GPU-hours.

This is the leg where our thesis would *propose* an architectural addition
based on the gap that Stage 2 reveals.

---

## 8. Open items requiring external verification

These need to be checked against the Zoology repo (`HazyResearch/zoology`)
before we publish numbers, because no local paper documents them in full
(see literature survey, Section 1):

1. **Vocabulary size.** We assume `V = 8192`. Verify.
2. **Key/value alphabet partition.** We assume disjoint halves. Verify.
3. **Per-query vs per-sequence accuracy.** We adopt per-sequence ≥99 %
   following Log-Linear and RWKV-7. Verify the "official" Zoology metric
   is the same.
4. **Distractor distribution.** The Zoology generator interleaves filler
   tokens; the exact distribution (uniform over `V`? uniform over `V_K ∪ V_V`?
   reserved filler set?) needs to be confirmed.
5. **Training step count.** We pick 50 000 max with ≥99 % early stop.
   Verify this matches the standard reproduction.

The first item we verify by writing the generator in `src/tasks/mqar.py`
and **running it head-to-head against Zoology's reference implementation
on a fixed seed** as a one-off correctness check. If our outputs diverge,
fix our generator (not Zoology's). This check is in Stage 1.1 testing.

---

## 9. Cross-reference

- **Compute & disk cost:** see `COST_ESTIMATE.md`
- **Code reuse map:** see the `Map formal_v1 codebase reusability` agent
  result attached to this stage's planning
- **Sibling project conventions:** `../formal_v1/CLAUDE.md`,
  `../formal_v1/PLAN.md`, `../formal_v1/RESULTS.md`,
  `../formal_v1/stage10_feedback.md`
