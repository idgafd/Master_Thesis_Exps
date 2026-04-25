# CLAUDE.md — Master Thesis Repository

Top-level orientation document for any agent working in this repository.
Per-subproject conventions live in `experiments/<name>/CLAUDE.md`; this
file is the index and the operational discipline that applies across
the whole repo.

---

## 1. What this repository is

Master's thesis codebase for **"On Expressiveness and Mechanism Design
of Linear Recurrent Models"** (Anastasiia Mazur, UCU, 2026).

The thesis studies *causal linear-time RNNs as an alternative
architectural family to Transformer causal attention*. Three architectures
are compared head-to-head:

- **RWKV-6** — per-channel data-dependent decay (WKV)
- **Mamba-2** — selective $\Delta t$-modulated continuous decay (SSD)
- **Linear Attention** — pure accumulator, no decay

Each is studied in both **causal** and **bidirectional (LION)** modes,
across two scales (7M / 14M params, conditional 30M), on two benchmarks
(LibriSpeech ASR, MQAR).

The empirical core is a cross-architecture transfer matrix of three
mechanisms that target three distinct expressivity axes. Each mechanism
shows a different transfer signature; the patterns together support the
thesis claim that *mechanism-prior alignment is the predictive criterion
for when causal linear-time RNN extensions convert into measurable gain*.

**Final-text deadline: 1 May 2026, end of day.**

---

## 2. Repository map

```
Master_Thesis_Exps/
├── CLAUDE.md                  ← this file
├── README.md
├── papers/                    ← all referenced papers as PDFs (16 files)
├── notebooks/                 ← exploratory notebooks (draft phase)
└── experiments/
    ├── asr_backbone_comparison/   ← DRAFT phase, frozen
    ├── formal_v1/                 ← MAIN ASR experiments (Stages 2–11)
    ├── synthetics_v1/             ← MQAR / state-tracking benchmarks
    └── final/                     ← thesis-facing master plan + writeup
```

### What each subproject is for

| Subproject | Purpose | When to touch it |
|---|---|---|
| `experiments/asr_backbone_comparison/` | Draft-phase code (Common Voice Ukrainian). Established infrastructure feasibility. **Frozen reference — do not edit.** | Read-only, for cross-checking infrastructure decisions. |
| `experiments/formal_v1/` | The main ASR mechanism-discovery codebase (LibriSpeech clean-100). Stages 2–11. Has its own detailed `CLAUDE.md`. | Any ASR run, any mechanism implementation, any LibriSpeech result. |
| `experiments/synthetics_v1/` | MQAR (Multi-Query Associative Recall, Zoology paper) and planned state-tracking benchmarks. Has its own `CLAUDE.md`. | Anything that needs to exercise associative recall (axis 2) or state tracking (axis 3) — i.e. anything that ASR cannot test. |
| `experiments/final/` | Thesis-facing artifacts: locked master plan, mechanism overview document, eventually the written chapters. | Reading the master plan; updating the human-facing thesis narrative. **No code lives here.** |

### Key documents (read in this order if onboarding)

1. **`experiments/final/Master_Plan.md`** — the locked v2.1 plan. Architecture × mode matrix, three-mechanism inventory, training schedule, run-count budget, output-directory specification. **Single source of truth for what gets run.**
2. **`experiments/final/Mechanisms_Overview.md`** — advisor-facing explainer of the three mechanisms, their math, paper lineage, transfer evidence, and how the thesis frames each. Read this before working on any mechanism.
3. **`experiments/formal_v1/CLAUDE.md`** — codebase-level conventions for the ASR pipeline (RWKV-6 / Mamba / LA / LION implementation rules, mechanism flags, common pitfalls).
4. **`experiments/synthetics_v1/CLAUDE.md`** + **`PLAN.md`** — MQAR experiment conventions and the length-sweep matrix.
5. **`experiments/formal_v1/EXPRESSIVITY_AXES.md`** — the five-axis decomposition framework and the per-paper / per-mechanism axis assignments. The synthesis chapter of the thesis is built on this.

---

## 3. The three reported mechanisms (one-line each)

| Mechanism | Targets | Where it's implemented |
|---|---|---|
| **multidil_v2** | Axis 1 — short-range temporal hierarchy | `formal_v1/src/models/mechanisms/conv_shift.py` |
| **LUCID** | Axis 2 — associative memory under interference | `formal_v1/src/models/rwkv6_time_mix.py::_apply_lucid_recurrent`, `lion_attention.py::lion_attention_with_lucid`, `mamba2_kernels.py::_apply_lucid_mamba2_chunked` |
| **rse_strong_viscosity** | Axis 1 sub-axis — damped-oscillator dynamics | `formal_v1/src/models/rwkv6_time_mix.py::_forward_recurrent_rse`, `linear_attn_rse.py`, `mamba2_rse.py` |

Full math, lineage, and transfer evidence in `experiments/final/Mechanisms_Overview.md`.

---

## 4. Where we are right now (final stage)

**Status:** Stages 2–11 complete on causal RWKV-6 / Mamba-2 / Linear
Attention. Master Plan v2.1 is locked. Three mechanisms confirmed
transferable. The cross-architecture transfer matrix is partially
populated; the remaining cells are the final ablations and the MQAR
sweep.

**Five days to deadline.** Execution priorities (in order):

1. **Finish the LibriSpeech 60-cell base matrix** (7M + 14M × 6
   architecture-mode cells × 5 mechanism cells). See Master_Plan §9.
2. **Run the MQAR length sweep** (10 backbones × 3 lengths = 30 runs,
   ~6 GPU-hours total). See Master_Plan §11. Engineering prerequisite:
   `mamba-ssm` CUDA backend integration; delta-rule forward-path audit
   vs Zoology reference.
3. **Common Voice pilot** (4 runs at 7M) — gate decision per
   Master_Plan §10. Run only after the LibriSpeech matrix completes.
4. **Conditional 30M scale runs** — only if the 7M→14M transition
   shows a ranking shift among single mechanisms.
5. **Begin shaping the 40-page paper from the existing draft.**
   Expected to overlap the last two days of compute.

The total mandatory run budget is **94 runs / ~156 GPU-hours** per
Master_Plan §16–17. Conditional expansions can push it to ~291 GPU-h.

Out of scope for the thesis (cite as future work):
- State-tracking probes ($S_3$ permutation, Dyck-$k$) — none of the
  three mechanisms target the diagonal-vs-DPLR boundary.
- DeltaNet / DeltaProduct / RWKV-7 — different mechanism family.
- Long-range NIAH / LRA — out-of-regime at $T \le 500$.

---

## 5. Operational discipline (applies everywhere)

These are the disciplines that distinguished signal from noise across
Stages 2–10. Any agent running new experiments must follow them.

### Before approving any new run

Cross-reference the candidate against accumulated evidence
(`stages_2_9_summary.md`, `STAGE10_SUMMARY.md`, the Phase-4 synthesis
in `formal_v1/TODO_FUTURE_IDEAS.md`). If the candidate's mechanism
family and deployment shape match an already-characterised closed cell,
**require an explicit argument for why this run will differ**. If no
such argument exists, do not run.

The most common failure mode in the project so far has been re-running
mechanisms whose outcome was already predicted by a closed cell. See
`formal_v1/CLAUDE.md §"Methodological errors recognised"` for the
canonical list.

### Run-level rules

- **Exact reduction at init.** Every new mechanism must reduce to an
  already-characterised baseline at step 0 (bit-exact or fp32 noise
  floor). This keeps attribution clean for 1–3σ effects.
- **Single-seed is the discovery default.** Multi-seed (3 seeds) is
  required only for BREAK-band claims per `STAGE10_PLAN.md §8.2`.
  Single-seed patterns outside noise are evidence but flagged.
- **Matched-epoch tracking.** Every run logs dev CER at epochs
  {1, 5, 15, 30, best, final} against its comparison cohort, per
  Master_Plan §13.
- **Per-run output directory** must follow the structure in
  Master_Plan §13. All files mandatory unless explicitly noted.
- **Diagnostics at the six checkpoints.** `diagnostics_ep*.json`
  captures param mobility, gradient signal, activation magnitudes,
  spectral statistics, and mechanism-specific probes. Mandatory.

### GPU isolation

Multiple long-running experiments may be on the machine simultaneously.

1. Always check `nvidia-smi` before launching anything.
2. **Do not stop a running experiment**, do not reuse its GPU, do
   not touch its output directory.
3. Use an idle GPU or wait until one frees up. Do not force-share.
4. If both GPUs are occupied, wait. Document the wait in your commit
   message.

### Commit and push discipline

- **Author is `Anastasiia Mazur <nastiamazur.v@gmail.com>`** —
  already set globally.
- **Do not add Claude (or any AI assistant) as Co-Authored-By** on
  commits unless the user explicitly asks for it.
- Use short, descriptive commit messages. Past examples that match
  the project style: "added mechanism explanations", "fix multidil
  init trap", "stage 11 lucid c-correlation results".
- **Do not push to `main` without committing all currently-running
  experiment outputs to the appropriate `outputs/` directory first.**
- Never `git reset --hard`, never `git push --force` to `main`,
  never skip pre-commit hooks (`--no-verify`).

---

## 6. The two benchmarks

### LibriSpeech clean-100 (primary, in `formal_v1/`)

Short-sequence English ASR with CTC. The measurement spine.
Characterizes axis 1 thoroughly. Cannot characterize axes 2–5 by
construction — that is the explicit scope cost the writeup names. See
`formal_v1/RESULTS.md` for the up-to-date results table.

### MQAR (secondary, in `synthetics_v1/`)

Multi-Query Associative Recall from the Zoology paper (Arora et al.).
Synthetic key-value-recall task; sweeps $T \in \{64, 256, 1024\}$.
Exercises axis 2 (associative memory under interference) by
construction. Gives LUCID a fair test where ASR cannot. The standard
benchmark in the linear-time-RNN community (Based, GLA, DeltaNet,
RWKV-7, Log-Linear all report it). See `synthetics_v1/PLAN.md` for the
locked matrix.

### Why these two and not more

Documented in `experiments/final/Mechanisms_Overview.md §"Why we
bounded the main study to ASR"` and `§"Why Zoology / MQAR is the
second benchmark"`. The short version: ASR was the discovery probe
because of its iteration cadence; MQAR is the axis-2 benchmark
because LUCID's claim is undertested without it. State-tracking
(axis 3) is bracketed by theory in the writeup but left for future
work because none of the three mechanisms target the diagonal-vs-DPLR
boundary.

---

## 7. Common pitfalls (across the repo)

These are gotchas that have bitten this project at least once and
should not bite it again. Per-codebase pitfalls (RWKV-6 decay sign,
LION prefix-sum convention, GroupNorm reshape, etc.) live in
`formal_v1/CLAUDE.md §"Common Pitfalls"`.

1. **Multiplicative gradient traps at init.** When a parameter
   appears as a product of two zero-init terms (e.g. $\alpha_d \cdot W_d$
   in multidil_v1), neither term receives gradient and the mechanism
   is silently inert. Always perturb one leg by $\sim 10^{-2}$ at init.
2. **Ceiling drift.** It's easy to start framing single-architecture
   ceilings (e.g. "0.115 on causal RWKV-6") as if they were the
   spine-absolute floor. Vanilla LION at 80 ep is at 0.071 in the
   same codebase. Don't confuse architecture-scoped numbers with
   spine-scoped ones.
3. **Schedule-mismatched references.** If you compare a 30-ep run
   to an 80-ep vanilla baseline, you bias Δ-to-vanilla by a known
   amount. Always match the schedule when comparing.
4. **Single-seed verdicts on MARGINAL claims.** PLATEAU-band results
   are defensible at single seed. MARGINAL+ claims (≤ 2σ) need
   multi-seed before being cited as thesis-grade.
5. **Re-running closed cells.** See §5. The canonical example is
   the Family-D quadratic-lift cluster (Stage 6 closed it; Stage 10
   re-ran four equivalent parametrisations and got the same
   saturation each time).

---

## 8. How to run things (top-level)

Everything is `uv`-managed.

### ASR runs

```bash
cd experiments/formal_v1
uv sync                                          # one-time
uv run scripts/debug_run.py                       # smoke test (5 ep, all backbones)
uv run scripts/run_experiment.py \
    --config configs/default.yaml \
    --backbone <name>                             # full run
```

Backbone names follow the convention in `formal_v1/CLAUDE.md
§"Backbone Naming Convention"`. The `_v2` suffix denotes runs with
the multidil init fix in place.

### MQAR runs

```bash
cd experiments/synthetics_v1
uv sync
# Per-length cohort:
uv run scripts/run_mqar.py --length 64 --backbone-set thesis-cohort
uv run scripts/run_mqar.py --length 256 --backbone-set thesis-cohort
uv run scripts/run_mqar.py --length 1024 --backbone-set thesis-cohort
```

See `synthetics_v1/RUNBOOK.md` for the full launch procedure and the
`mamba-ssm` CUDA prerequisite.

### Shared infrastructure

Both subprojects reuse the encoder factories from
`formal_v1/src/models/` (Transformer, RWKV-6, Mamba-2, LION, plus
mechanism flags). The synthetics codebase imports from formal_v1;
do not duplicate model code.

---

## 9. Don't-do list

- Don't edit `experiments/asr_backbone_comparison/` — frozen draft
  reference.
- Don't edit any file under `outputs/` belonging to a run you didn't
  launch (see GPU isolation, §5).
- Don't add Claude or any AI assistant as Co-Authored-By on commits.
- Don't push to `main` with currently-running experiment results
  uncommitted in your working tree.
- Don't run mechanisms whose outcome is already predicted by a
  closed cell without a written justification.
- Don't change the master plan (`experiments/final/Master_Plan.md`)
  without flagging the change as an explicit decision update — the
  plan is locked and shipping deadline is 1 May.
- Don't introduce new dependencies in `pyproject.toml` without
  bumping the lockfile (`uv lock`).
- Don't create `.md` documentation files unless asked — the docs
  that exist are intentionally curated.

---

## 10. Quick reference — paper index

All referenced papers live as PDFs in `papers/`. The master mapping
of paper → mechanism → axis is in
`experiments/formal_v1/EXPRESSIVITY_AXES.md §"Paper index"`. The
three primary papers for the three reported mechanisms:

| Mechanism | Primary paper | File |
|---|---|---|
| multidil_v2 | Paper 7 — Non-Attention LLM (arXiv:2506.01963) | `papers/Non_Attention_2506.01963v1.pdf` |
| LUCID | LUCID (arXiv:2602.10410) | `papers/LUCID_2602.10410v1.pdf` |
| rse_strong_viscosity | Internally derived; theory bracket from arXiv:2603.01959 + arXiv:2603.03612 | (no local PDF — cited theory only) |

Cross-domain axis-1 confirmation papers (Vision-RWKV, LiT, AudioRWKV)
also live in `papers/` and are listed in `EXPRESSIVITY_AXES.md`.

---

## 11. Contacts and conventions

- **Author:** Anastasiia Mazur (nastiamazur.v@gmail.com)
- **Advisor:** Eric (UCU)
- **Final-text deadline:** 1 May 2026, end of day
- **Style reference for the writeup:**
  https://er.ucu.edu.ua/items/c8097ba7-543e-4927-892d-0e8f2cd00dc9

For any decision that materially changes the master plan, the run
budget, or the thesis claim — flag it explicitly to the user before
acting. The plan is locked; deviations need to be explicit.
