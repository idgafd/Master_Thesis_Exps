# CLAUDE.md — Final Stage Entry Point

This is the entry point for any agent working on the master thesis
during the final stage. Always start here.

Files in this directory:

**Locked references (do not edit without explicit decision):**

- **`Master_Plan.md`** — locked v2.1 plan. Single source of truth for
  what gets run, the architecture × mechanism × scale matrix, training
  schedule, output-directory specification, run budget. Read it first.
- **`Mechanisms_Overview.md`** — advisor-facing explainer of the three
  reported mechanisms with math, lineage, and transfer evidence. The
  thesis narrative lives here in compressed form.

**Live trackers (updated as runs land or chapters move):**

- **`STATUS.md`** — execution tracker. Which cells of the matrix are
  complete; per-cell test CER; engineering notes; chronological log.
- **`FULL_RESULTS.md`** (on `main` branch) — canonical aggregate of
  every per-cell `results.json` plus MQAR cohort verdicts. Source of
  truth for the result tables in Appendix C and for the matrix figures.

**Thesis-writing artefacts (live on `thesis` branch):**

- **`Thesis_Positioning.md`** — positioning Variant 3 (characterisation
  study with predictive epistemology), three-mechanism narrative, the
  decay-as-prerequisite finding, risks. The single source of the
  thesis's public framing.
- **`Writeup_Notes.md`** — open issues that surfaced during chapter
  drafting and their resolutions.
- **`Chapter3_Plan.md`**, **`Chapter4_Plan.md`** — section-by-section
  plans behind the corresponding chapters; chapters now drafted in
  `Master_Thesis/chapters/`.
- **`Chapter5_Plots_Plan.md`** — figure inventory. Figures F1–F13 are
  built under `Master_Thesis/figures/chapter5/`.
- **`Appendices_Plan.md`** — five-appendix plan; appendices A–E now
  drafted in `Master_Thesis/appendices/`.

No code lives in this directory.

---

## What this thesis is

**"On Expressiveness and Mechanism Design of Linear Recurrent Models."**

Causal linear-time RNNs (RWKV-6, Mamba-2, Linear Attention) compared
head-to-head as an alternative architectural family to Transformer
causal attention. Three mechanisms are designed and evaluated for
their cross-architecture transfer. The public (thesis) names and
their codebase identifiers are:

1. **MSDC** — Multi-Scale Depthwise Convolution. Input-side multi-scale
   depthwise convolution targeting axis 1 (short-range temporal
   hierarchy). Codebase: `multidil_v2` /
   `convshift_multidil_symmetric_v2`.
2. **CVD** — Chunked Value Decorrelation. Within-chunk
   value-decorrelation preconditioner targeting axis 2 (associative
   memory under interference). Architecture-specific deployment per
   `Master_Plan.md §3`. Codebase: `lucid` (and per-arch variants
   `lucid_chunked` on RWKV-6, `lucid_c` on Mamba-2). Logically related
   to LUCID Attention~\cite{duvvuri2026lucid} on a different substrate;
   not an adaptation.
3. **DHO** — Damped Harmonic Oscillator. Block-complex SO(2) transition
   with Rayleigh damping targeting an axis-1 sub-axis (damped-oscillator
   dynamics). Reported in its canonical depth-graded form. Codebase:
   `rse_depth_viscosity` (preferred where present;
   `rse_strong_viscosity` is the strong-fallback variant where the
   depth-graded run was not produced).

Two benchmarks: **LibriSpeech clean-100 ASR** (the discovery probe,
characterises axis 1) and **MQAR** from the Zoology paper (the second
benchmark, exercises axis 2 by construction). Common Voice EN 100h
serves as a cross-distribution probe.

**Positioning** is Variant 3: characterisation study with predictive
epistemology. The thesis's central claim is the **mechanism-prior
alignment law**: a mechanism's CER gain is proportional to the
deficit it covers in the underlying backbone, predictable from a
five-axis decomposition of expressivity. The positioning, the law,
and the five-axis frame live in `Thesis_Positioning.md`.

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

---

## Current state (2026-04-30)

### Experimental matrix

The full LibriSpeech 7M (causal + LION), 14M (causal-only;
LION dropped per scope), Common Voice EN 100h pilot, and MQAR
length-sweep matrices are landed and aggregated in
`FULL_RESULTS.md` (on `main` branch). Per-cell artefacts under
`experiments/final/outputs/<cell>_seed42/` follow the §13 output
spec (config, history, metrics, results, train log, plots,
`best_model.pt`).

Key matrix-level findings already stable:

- **MSDC universal axis-1 with deficit-proportional ordering.** LA
  $\Delta -0.047 >$ RWKV-6 $\Delta -0.026 >$ Mamba-2 $\Delta -0.021$
  at 7M, holding at 14M.
- **DHO BREAK on Linear Attention.** $\Delta -0.068$ causal,
  $\Delta -0.191$ on LION-LIT, $\Delta -0.039$ on LION-S. Lowest LA
  test CER overall is LION-LIT $\times$ DHO $\times$ MSDC $= 0.0961$.
- **DHO null on Mamba-2 LibriSpeech / helpful on Mamba-2 Common
  Voice.** Task-prior modulated.
- **CVD asymmetric transfer + decay-as-prerequisite.** CVD $\times$
  LION-LIT is the only negative cell of the LION matrix
  ($\Delta +0.024$); CVD $\times$ LION-S converts ($\Delta -0.007$).
  Decay is a structural prerequisite for CVD on bidirectional LA.
- **Matrix ceiling.** Mamba-2 14M $\times$ MSDC = 0.0631 test CER;
  Mamba-2 14M $\times$ CVD $\times$ MSDC = 0.0618.
- **MQAR.** Vanilla linear-time backbones FAIL at $T=64$ already;
  MSDC and CVD PASS at every tested $T \in \{64, 256, 1024\}$ across
  all three architectures. Causal Transformer FAILs at $T = 1024$.

### Thesis writing

Located under `Master_Thesis/` on the `thesis` branch:

- **Chapters 1–4 drafted** (`chapters/1_introduction.tex`,
  `chapters/2_related_work.tex`, `chapters/3_theoretical_background.tex`,
  `chapters/4_proposed_solution.tex`). Chapter 5
  (`chapters/5_experiments_and_results.tex`) currently has the
  `\chapter{}` and `\label{ch:experiments}` only; the prose remains
  to be written from `Chapter5_Plots_Plan.md` and the result tables.
- **Appendices A–E drafted** (`appendices/{a,b,c,d,e}_appendix.tex`).
- **Figures F1–F13 built** under `figures/chapter5/`. Each figure
  has its `.pdf`, `.png`, `_data.csv`, and `_script.py`. Locked
  palette and typography in `figures/chapter5/_style.py`.
- **Bibliography** in `bibliography.bib` covers every citation key
  used by chapters 1–4 and appendices A–E.

### Open work for an agent picking this up

1. **Draft Chapter 5 prose**, weaving the figure inventory of
   `Chapter5_Plots_Plan.md` into the section structure of the
   master plan §6. Forward-reference labels already in use:
   `sec:experimental-design`, `sec:closed-cells-engaged-null`,
   `sec:dho-mechanism-engagement`, `sec:lion-decay-prereq`.
2. **Cross-reference resolution.** Once Chapter 5 lands, the four
   forward-referenced labels above will become defined; verify with
   `grep -nE "\\\\(ref|Cref)\\{" Master_Thesis/`.
3. **Style policy.** No em-dashes or en-dashes anywhere in the
   LaTeX deliverable. No "we adapt LUCID" / "our adaptation of
   LUCID" phrasings; CVD is logically related to LUCID on a
   different substrate. No `v1`/`v2` initialisation narrative; no
   `Stage-N` references in prose. The `_strong_` directory names
   appear only inside `\texttt{...}` cell-directory columns.
4. **Title page, abstract, acknowledgements, ToC.** Front-matter
   sweep before final compile.
