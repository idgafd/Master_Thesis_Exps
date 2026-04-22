# synthetics_v1 — RUNBOOK for the executing agent

This document is written for an agent (or human) who has **not** seen the
conversation that produced this experiment. Read it top-to-bottom before
running anything. Each step has an explicit success criterion and an
explicit failure-mode table.

If a step fails: **stop, do not improvise**. Read the failure-mode table
for that step. If it isn't covered, write a short report (what you ran,
the full traceback, the last 50 lines of `train.log`) and hand back to the
PI.

---

## 0. Context (read first, ~2 min)

This is **Stage 1, Phase 0–1** of a synthetics tier for a master thesis on
expressive sequence models. The task is **MQAR** (Multi-Query Associative
Recall, Arora et al. 2024) — the canonical literature benchmark for
sequence-mixer recall capability.

You will run **8 backbones × 2 sequence lengths × 1 seed = 16 runs**
total, plus a smoke test pass. Total wall-clock target: **~1.5 GPU-hours**
on an RTX PRO 6000 Blackwell.

Sibling project `../formal_v1/` is the main thesis codebase (LibriSpeech
ASR). Synthetics_v1 reuses formal_v1's leaf backbone modules via symlinks
— **do not edit files inside `../formal_v1/` from this experiment**. If
you need to fix a backbone, raise it; do not patch it under your own
authority.

For deeper background, read:
- `PLAN.md` — what we are running and why
- `COST_ESTIMATE.md` — compute / disk envelope
- `CLAUDE.md` — code conventions
- `../formal_v1/CLAUDE.md` — sibling project conventions

You do NOT need to read the formal_v1 source.

---

## 1. Environment & one-time setup (~3 min)

**Prerequisites** (verify but do not install):
- NVIDIA GPU with CUDA 12.8 driver (Blackwell sm_120 or compatible)
- `uv` package manager available on PATH
- `bash` shell

```bash
cd /home/researcher/Master_Thesis_Exps/experiments/synthetics_v1

# 1.1 Sync deps (creates .venv in this directory)
uv sync

# 1.2 Mirror leaf backbone files from formal_v1 via symlinks.
# Idempotent — safe to re-run.
bash scripts/setup_symlinks.sh
```

**Verify the symlinks landed:**

```bash
ls -la src/models/ | grep -E "transformer|rwkv6|mamba|lion|mechanisms|components"
# You should see arrows ('->') pointing into ../formal_v1/src/models/
```

**Verify GPU is visible:**

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA RTX PRO 6000 Blackwell  (or similar)
```

**Failure modes — Step 1:**

| Symptom | Cause | Fix |
|---|---|---|
| `uv sync` fails on torch wheel | wrong CUDA version | check `pyproject.toml` `[[tool.uv.index]]` matches your driver. Driver 570 → cu128. Different driver → STOP, report back. |
| `setup_symlinks.sh: formal_v1 not found` | sibling project missing | STOP. The whole thesis depends on formal_v1; do not try to recreate it. |
| `torch.cuda.is_available() == False` | driver/runtime mismatch | run `nvidia-smi`; if it works, the issue is the wheel. STOP and report. |

---

## 2. Run the test suite (~30 sec)

The tests catch obvious bugs before any GPU time is spent.

```bash
uv run python -m pytest tests/ -v
```

**Success criterion:** all tests pass. Expect ~25 tests, all green.

The `test_synthetic_model.py` tests parameterise across all 8 backbones
and run forward/backward smokes — they will load the formal_v1 backbones
through the symlinks. If the symlinks are broken, this is where you'll
see it.

**Failure modes — Step 2:**

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'TransformerEncoder'` | symlink missing | re-run `bash scripts/setup_symlinks.sh` |
| `ModuleNotFoundError: src.models.X` | one of the symlinked files imports a sibling that wasn't symlinked | check `setup_symlinks.sh` and add the missing file to its `link` list |
| `non-finite logits from <backbone>` | NaN/Inf in encoder forward at init | STOP, report. Could be a regression in the symlinked formal_v1 code. |
| `<backbone>: 5 steps did not reduce loss` | the backbone trains but very slowly OR is broken | STOP, report which backbone and the loss values. Could be a real regression. |

Do **NOT** edit the symlinked files to "fix" failures. If the failure is
in formal_v1 backbone code, hand back to the PI.

---

## 3. Phase 0 — smoke test (~5–10 min wall-clock)

Quick infrastructure check: every backbone trains for 200 steps at T=64
without crashing.

```bash
uv run python scripts/debug_run.py
```

This writes per-backbone output under `outputs/_smoke/<backbone>/` and a
combined `outputs/_smoke/_summary.json`.

**Success criterion:**
- All 8 backbones report verdict `OK`.
- For Transformer (causal or bidirectional): `final_per_query_acc > 0.05`
  after 200 steps at T=64 (chance is `1/4096 ≈ 0.00024`, so even rough
  learning beats it by 200×). RWKV-6 / Mamba may still be near chance
  after 200 steps — that's fine, this phase is infra not convergence.
- All `final_loss` values are finite and below `log(8192) ≈ 9.0`.

**Quick check command:**

```bash
uv run python scripts/analyze_cohort.py --root outputs/_smoke
```

**Failure modes — Step 3:**

| Symptom | Cause | Action |
|---|---|---|
| Any backbone reports `FAIL: ...` in summary | the backbone crashed during training | check `outputs/_smoke/<backbone>/train.log` for the traceback. STOP, report. |
| All backbones run but Transformer per-query acc ≈ 0.0002 | training loop bug | check that targets contain non-`-100` values; check that loss goes down. STOP if unclear. |
| `CUDA out of memory` | batch too large | edit `scripts/debug_run.py:_smoke_cfg`, drop `batch_size` to 16. Re-run. |
| Mamba runs much slower than estimated (>2 min for 200 steps) | pure-PyTorch Mamba is slow without CUDA kernel | EXPECTED — note in your report; do not block on it. |

---

## 4. Phase 1a — T=64 cohort (~30 min wall-clock)

Full early-stopping convergence at the short length.

```bash
bash scripts/run_cohort_reduced.sh --T64
```

This runs the 8 backbones at T=64, K=16. Each run early-stops at per-seq
accuracy ≥ 0.99 OR after 30 000 steps (whichever first) OR after 5
consecutive evals without improvement.

Outputs land in `outputs/cohort_reduced/<backbone>_T64_seed42/`. Each
directory has:
- `config.yaml` — resolved config used
- `train.log` — full training log
- `metrics.jsonl` — per-step train metrics + per-eval-step eval metrics
- `results.json` — final headline metrics

**Mid-stream check (after each backbone finishes):**

```bash
# In a second terminal — see what's done so far:
uv run python scripts/analyze_cohort.py --root outputs/cohort_reduced
```

**Success criterion (Phase 1a):**
- All 8 backbones produce a `results.json`.
- `transformer_causal` and `transformer` reach `best_per_seq_acc ≥ 0.99`
  (PASS). If they don't, the infrastructure is broken — STOP.
- At least 4 of the 8 backbones reach `best_per_seq_acc ≥ 0.95` (most
  variants should converge at T=64 — this is the easy length).

**Failure modes — Step 4:**

| Symptom | Action |
|---|---|
| Transformer fails to PASS | STOP. Either the data is wrong or the optimizer is broken. Check loss trajectory in `metrics.jsonl`. |
| 1–2 RWKV/Mamba variants stall at low accuracy (≤ 0.5) | NOT necessarily a bug — note it and proceed. Phase 1b will give a clearer signal. |
| OOM mid-run | drop `batch_size` in `configs/default.yaml` from 64 to 32, re-run only the failed backbone via `scripts/run_experiment.py` directly. |
| Run wall-clock far exceeds estimate (>10 min for one T=64 backbone at 30k steps) | check GPU utilisation with `nvidia-smi`; if pegged, expected; if low, dataloader bottleneck — drop `num_workers` to 0. |

---

## 5. Phase 1b — T=256 cohort (~1 GPU-hour wall-clock)

The representative length. Same protocol, longer convergence.

```bash
bash scripts/run_cohort_reduced.sh --T256
```

(Or, if Phase 1a was clean and you want both phases in one go, run
`bash scripts/run_cohort_reduced.sh` without flags — it will skip
already-completed Phase-1a runs because each `run_one` checks for
existing `results.json`.)

**Success criterion (Phase 1b):**
- All 8 backbones produce a `results.json`.
- The expected ranking from `PLAN.md §2.3`:
  - `transformer_causal` PASS
  - `rwkv6_delta` and `rwkv6_lucid_delta` ≥ `rwkv6` (Delta Rule lift)
  - `rwkv6_lucid` may or may not lift — this is a novel test
- `transformer` (bidirectional) PASS — sanity that the eval loop is right

**The actual research signal lives here.** The verdict matrix from
`PLAN.md §5` will tell you whether the literature's Delta Rule prediction
holds at our parameter envelope.

---

## 6. Final analysis (~5 min)

```bash
uv run python scripts/analyze_cohort.py --root outputs/cohort_reduced
```

This prints the markdown table and writes `outputs/cohort_reduced/_index.csv`.

**Hand back to the PI:**

1. The analyze_cohort table (paste it).
2. A 3-line interpretation against `PLAN.md §5` decision rules:
   - Did Transformer PASS at both lengths? (sanity)
   - Did Delta Rule lift RWKV-6 above the baseline? (literature confirmation)
   - Any unexpected ranking?
3. Total wall-clock and any failures encountered.

**Do NOT** at this stage:
- Expand to the full Cohort A / B (96 + 36 runs) — that requires PI go.
- Tune hyperparameters to "improve" results — these are pre-registered
  comparisons; tuning would invalidate them.
- Edit `formal_v1` or the symlinked backbone code.

---

## 7. Escalation rules (when to stop and report)

Stop and report back IMMEDIATELY if:

- **Any test in Step 2 fails** — the infrastructure isn't ready.
- **Step 3 smoke fails for ≥ 2 backbones** — likely a systemic bug.
- **Transformer fails to PASS at either length in Steps 4/5** — the data
  or training loop is broken; comparison is meaningless.
- **You feel tempted to edit a symlinked file in `../formal_v1/`** —
  that's authority you don't have here.
- **Total wall-clock exceeds 4 GPU-hours** for the full reduced cohort —
  estimates are off, debug before burning more compute.

Otherwise: proceed to the next step.

---

## 8. File map (cheat sheet)

```
synthetics_v1/
├── PLAN.md                       what we are running, why, success criteria
├── COST_ESTIMATE.md              compute & disk math
├── CLAUDE.md                     project conventions
├── RUNBOOK.md                    THIS FILE
├── README.md                     short orientation
├── pyproject.toml                deps (no audio)
├── configs/
│   ├── default.yaml              anchor config (overrides dataclass defaults)
│   └── cohort_reduced.yaml       reduced 16-run cohort spec (informational)
├── scripts/
│   ├── setup_symlinks.sh         one-time: mirror backbones from formal_v1
│   ├── debug_run.py              Phase 0 smoke (200 steps × all backbones)
│   ├── run_experiment.py         single-run entry point
│   ├── run_cohort_reduced.sh     drives the 16-run reduced cohort
│   └── analyze_cohort.py         walks results.json files → summary table
├── src/
│   ├── config.py                 SyntheticsConfig dataclass + load_config
│   ├── tasks/mqar.py             MQAR generator (the core data)
│   ├── data/{vocab,dataset}.py   token vocab + train/eval DataLoaders
│   ├── models/
│   │   ├── encoder.py            slim 8-backbone dispatcher (NOT symlinked)
│   │   ├── synthetic_model.py    TokenEmbed → encoder → LMHead
│   │   └── (symlinks to formal_v1 leaf backbones)
│   └── training/
│       ├── train.py              self-contained training loop
│       ├── evaluate.py           per-sequence + per-query MQAR accuracy
│       └── schedulers.py         WarmupCosineScheduler
├── tests/
│   ├── test_mqar_generator.py    correctness of the MQAR generator
│   └── test_synthetic_model.py   forward/backward across all 8 backbones
└── outputs/
    ├── _smoke/                   Phase 0 outputs
    └── cohort_reduced/           Phase 1 outputs
```
