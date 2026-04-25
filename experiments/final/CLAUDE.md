# CLAUDE.md — Final Stage Entry Point

This is the entry point for any agent working on the master thesis
during the final stage. Always start here.

The two files in this directory are the only thesis-facing artifacts:

- **`Master_Plan.md`** — locked v2.1 plan. Single source of truth for
  what gets run, the architecture × mechanism × scale matrix, training
  schedule, output-directory specification, run budget. Read it first.
- **`Mechanisms_Overview.md`** — advisor-facing explainer of the three
  reported mechanisms with math, lineage, and transfer evidence. The
  thesis narrative lives here in compressed form.
- **`STATUS.md`** — execution tracker. Which cells of the matrix are
  complete, which are pending. Updated as runs land.

No code lives in this directory.

---

## What this thesis is

**"On Expressiveness and Mechanism Design of Linear Recurrent Models."**

Causal linear-time RNNs (RWKV-6, Mamba-2, Linear Attention) compared
head-to-head as an alternative architectural family to Transformer
causal attention. Three mechanisms are designed and evaluated for
their cross-architecture transfer:

1. **multidil_v2** — input-side multi-scale depthwise convolution
   (axis 1, short-range temporal hierarchy).
2. **LUCID** — within-chunk value-decorrelation preconditioner (axis 2,
   associative memory under interference). Architecture-specific
   deployment per `Master_Plan.md §3`.
3. **rse_strong_viscosity** — block-complex SO(2) transition with
   Rayleigh damping (axis 1 sub-axis, damped-oscillator dynamics).

Two benchmarks: **LibriSpeech clean-100 ASR** (the discovery probe,
characterizes axis 1) and **MQAR** from the Zoology paper (the second
benchmark, exercises axis 2 by construction).

**Final-text deadline: 1 May 2026, end of day.**

---

## Repository layout (from this directory's perspective)

```
../../                              ← repo root
├── papers/                         ← all referenced paper PDFs
├── notebooks/                      ← exploratory notebooks (draft phase)
└── experiments/
    ├── asr_backbone_comparison/    ← DRAFT, frozen, do not edit
    ├── formal_v1/                  ← MAIN ASR codebase, has its own CLAUDE.md
    ├── synthetics_v1/              ← MQAR codebase, has its own CLAUDE.md
    └── final/                      ← YOU ARE HERE
        ├── CLAUDE.md
        ├── STATUS.md
        ├── Master_Plan.md
        └── Mechanisms_Overview.md
```

When the user asks for an ASR run, the work happens in `formal_v1/`.
When the user asks for an MQAR / synthetics run, the work happens in
`synthetics_v1/`. Both subprojects have their own `CLAUDE.md` with
codebase-level conventions; read them before editing source.

---

## Operational discipline (applies everywhere)

These rules distinguished signal from noise across Stages 2–10. Any
new run must follow them.

### Before approving any new run

Cross-reference the candidate against accumulated evidence in the
formal_v1 summaries and the per-stage notes. If the candidate's
mechanism family and deployment shape match an already-characterised
closed cell, **require an explicit argument for why this run will
differ**. If no such argument exists, do not run. The most common
project failure mode has been re-running mechanisms whose outcome was
already predicted by a closed cell.

### Run-level rules

- **Exact reduction at init.** Every new mechanism must reduce to an
  already-characterised baseline at step 0 (bit-exact or fp32 noise
  floor). Keeps attribution clean for 1–3σ effects.
- **Single-seed is the default.** Multi-seed (3 seeds) is required
  only for BREAK-band claims. Single-seed patterns outside noise are
  evidence but flagged.
- **Matched-epoch tracking.** Log dev CER at epochs
  {1, 5, 15, 30, best, final} against the comparison cohort.
- **Per-run output directory** must follow the structure in
  `Master_Plan.md §13`. All files mandatory unless explicitly noted.
- **Diagnostics at the six checkpoints.** Mandatory per §13.
- **Don't change `Master_Plan.md`** without flagging the change as
  an explicit decision update. The plan is locked.

### GPU isolation

- Always check `nvidia-smi` before launching anything.
- Do not stop a running experiment, do not reuse its GPU, do not
  touch its output directory.
- Use an idle GPU or wait. Do not force-share.

### Commit and push discipline

- Use short, descriptive commit messages.
- Do not add Claude (or any AI assistant) as Co-Authored-By on
  commits.
- Never `git reset --hard`, never `git push --force` to `main`,
  never skip pre-commit hooks (`--no-verify`).
- Before pushing, make sure no currently-running experiment outputs
  are uncommitted in the working tree.

---

## Common pitfalls

Top-level only. Per-codebase pitfalls live in
`formal_v1/CLAUDE.md` and `synthetics_v1/CLAUDE.md`.

1. **Multiplicative gradient traps at init.** When a parameter is a
   product of two zero-init terms, neither receives gradient. Always
   perturb one leg by $\sim 10^{-2}$ at init.
2. **Ceiling drift.** Don't conflate architecture-scoped numbers with
   spine-scoped ones. Vanilla LION at 80 ep is at 0.071 in the same
   codebase as causal RWKV-6's 0.115 ceiling.
3. **Schedule-mismatched references.** Always match epoch count when
   comparing.
4. **Single-seed verdicts on MARGINAL claims.** PLATEAU-band results
   are defensible at single seed; MARGINAL+ claims need multi-seed.
5. **Re-running closed cells.** See the pre-run rule above.

---

## Don't-do list

- Don't edit `experiments/asr_backbone_comparison/` — frozen draft.
- Don't edit any file under `outputs/` belonging to a run you didn't
  launch.
- Don't add Claude or any AI assistant as Co-Authored-By on commits.
- Don't push to `main` with currently-running experiment results
  uncommitted.
- Don't run mechanisms whose outcome is already predicted by a
  closed cell without a written justification.
- Don't change `Master_Plan.md` without flagging an explicit decision.
- Don't introduce new dependencies in `pyproject.toml` without
  bumping the lockfile (`uv lock`).
- Don't create `.md` files unless asked.

---

## References

Papers, axis-decomposition documents, and parking-lot mechanism notes
live elsewhere in the repo. The user will point to specific files
(in `papers/`, in `formal_v1/EXPRESSIVITY_AXES.md`, in
`formal_v1/TODO_FUTURE_IDEAS.md`, etc.) when relevant. Do not preload
or assume specific references unless directed.
