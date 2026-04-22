# synthetics_v1 — debug & runbook addendum

This file documents the diagnostic + fix pass I did on synthetics_v1 while executing `RUNBOOK.md` end-to-end. It explains:

- what broke in the existing code,
- what I changed inside synthetics_v1,
- the cohort result, and
- what the PI should do next.

The headline research output (Phase 1a T=64 cohort) is in `outputs/REPORT_phase1a.md`.

**Note on the formal_v1 dependency.** When I started this session the local clone did not yet contain Stage 10 work, and `formal_v1/src/models/rwkv6_encoder.py` did not accept the mechanism kwargs that the synthetics dispatcher passes (`use_loglinear`, `use_m2rnn`, `use_conv_shift_multidilation`, `use_chanmix_bypass`, `use_cayley_orthogonal`, `use_pom_vlift`, etc.). I worked around this with a temporary local stub and documented it here as "required local formal_v1 patch". After rebasing onto current `origin/main`, those kwargs are present and properly wired by Stage 10's commit `ecaac50` — the stub is no longer needed. The synthetics_v1 pipeline now runs end-to-end against upstream formal_v1 with **no formal_v1 changes required** (verified: `pytest tests/ -v` → 30/30 green). The §Required local formal_v1 patch section below is kept for historical context only.

## TL;DR

The `RUNBOOK.md` pipeline could not be executed as-shipped. In order:

1. `uv run python -m pytest tests/` — required `pytest` as an ephemeral dep; 12/33 tests failed because the synthetics dispatcher passed ~13 kwargs that the currently-symlinked `formal_v1/src/models/rwkv6_encoder.py` does not accept.
2. The `rwkv6_lucid_delta` variant was unreachable: in `mode="recurrent"` the TimeMix takes the LUCID branch and returns before the delta-rule block, so the variant ran as LUCID-only (delta_params dead). I dropped it from the reduced cohort (7 backbones).
3. The training loop's early-stopping patience tracked `per_seq_acc`, which stays at exactly 0 until a model is essentially perfect. The counter fired at step 6000 on every run, regardless of learning. I rewired patience to track `per_query_acc` with a `1e-3` min-delta.
4. `src/data/dataset.py` passed `batch_size=None` to `DataLoader` but the `_collate_fixed_length` fn assumed list-wrapping — asserting `len(items) == 1`. Every fetch blew up before the model ever saw data. Fixed by accepting the item directly.
5. **🔑 The MQAR generator was empirically unlearnable.** It used `placeholder=0` at all prediction positions; every backbone (including a fresh vanilla 2-layer GPT trained with an independent loop for 25k steps) pinned at `per_query_acc = 1/K` (uniform over in-context values) indefinitely. Replaced with the Zoology-convention layout (`[k1 v1 ... kK vK q1 v_q1 ... qQ v_qQ]` + next-token LM loss at query-key positions). Same vanilla GPT hits 100% in 300 steps.
6. `analyze_cohort.py` now emits Zoology-style `steps_to_0.5` / `steps_to_0.9` columns derived from `metrics.jsonl`.

After those fixes, the T=64 cohort runs and produces a clean, thesis-relevant result:

```
transformer (bidir)  PASS at step 2000
transformer_causal   PASS at step 5000
rwkv6_lucid          PASS at step 7000     ← LUCID lifts RWKV-6
rwkv6                FAIL (chance)
rwkv6_delta          FAIL (chance, both warmstart regimes tested)
mamba                FAIL (chance)
mamba2               FAIL (chance)
```

See `outputs/REPORT_phase1a.md` for interpretation, trajectory tables, and follow-up recommendations.

---

## Required local formal_v1 patch (HISTORICAL — no longer needed after Stage 10 rebase)

> **Update:** As noted at the top of this file, the kwargs documented in this section landed in upstream `formal_v1` in Stage 10 commit `ecaac50` and are now properly wired (not stubs — they plumb through to RWKV6Block / RWKV6TimeMix). On a fresh clone of current `main`, no formal_v1 patch is required. This section is kept only as the audit trail for what blocked the cohort run before that rebase.

The synthetics_v1 dispatcher at `src/models/encoder.py:105-157` passes ~13 forward-looking mechanism kwargs (`use_loglinear`, `use_m2rnn`, `use_conv_shift_multidilation`, etc.) to `RWKV6Encoder`. Before Stage 10 the symlinked `formal_v1/src/models/rwkv6_encoder.py` did **not** accept them. Running the test suite against the unpatched formal_v1 reproduces the initial blocker:

```
TypeError: RWKV6Encoder.__init__() got an unexpected keyword argument 'use_loglinear'
    at src/models/encoder.py:105
```

Until those mechanism flags actually land in formal_v1 (they appear to be staged in a separate branch per the PI: "there is some other local work on the other backbones"), **you must apply this patch locally to formal_v1 before running the synthetics_v1 tests or cohort.** It's a no-op for the reduced cohort (all flags default to `False`), but it fails loud with `NotImplementedError` if any flag is toggled on — so a future caller won't silently get a broken mechanism.

### Patch

`experiments/formal_v1/src/models/rwkv6_encoder.py` — extend `RWKV6Encoder.__init__`:

```python
        use_sparse_nonnormal_rse: bool = False,
        sparse_nn_edge_only: bool = False,
        nonnormal_psi_static: bool = False,
        # Accepted-but-unused flags: implementations live on other local
        # branches; the synthetics_v1 dispatcher passes them as False/defaults
        # for the reduced-cohort backbones so accepting them here is a no-op.
        # Land the real implementations before enabling any of these.
        use_loglinear: bool = False,
        loglinear_levels: int = 10,
        use_m2rnn: bool = False,
        m2rnn_layer: int = 5,
        use_conv_shift_multidilation: bool = False,
        conv_shift_multidil_padding_mode: str = "auto",
        conv_shift_multidil_content_conditional: bool = False,
        use_chanmix_bypass: bool = False,
        use_cayley_orthogonal: bool = False,
        cayley_rank: int = 1,
        use_pom_vlift: bool = False,
        pom_order: int = 2,
        pom_expansion: int = 64,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        _unimplemented = {
            "use_loglinear": use_loglinear,
            "use_m2rnn": use_m2rnn,
            "use_conv_shift_multidilation": use_conv_shift_multidilation,
            "use_chanmix_bypass": use_chanmix_bypass,
            "use_cayley_orthogonal": use_cayley_orthogonal,
            "use_pom_vlift": use_pom_vlift,
        }
        _enabled = [k for k, v in _unimplemented.items() if v]
        if _enabled:
            raise NotImplementedError(
                f"RWKV6Encoder kwargs not yet wired in this branch: {_enabled}. "
                "Accept-only stubs exist so dispatcher calls with default=False "
                "don't break; enabling them requires landing the mechanism code."
            )
```

(Insert the new kwargs immediately before the existing `dtype:` line; insert the `_unimplemented` guard immediately after `super().__init__()`.)

### Long-term fix

Two clean options for the PI:

**(A)** Land the actual mechanism implementations in formal_v1's `RWKV6Encoder` / `RWKV6Block` / `RWKV6TimeMix` and drop the stub. This is the "real work" — matches what the dispatcher is already asking for.

**(B)** Trim the synthetics_v1 dispatcher (`src/models/encoder.py:105-157`) to only pass the kwargs that formal_v1 currently supports. This is a 20-line delete, zero-risk, and cleanly decouples synthetics_v1 from whatever mechanism work is in flight in formal_v1. Recommended if the reduced cohort is the thesis scope and the forward-looking flags won't be enabled for months.

I picked (B)-equivalent for my cohort runs (by stubbing formal_v1 locally with a no-op guard) because it didn't change behaviour, only the signature. Either path is legit.

---

## What I changed inside synthetics_v1 (committed)

All of these are in synthetics_v1 only; they don't touch formal_v1 or the symlinked backbones.

### 1. `src/tasks/mqar.py` — layout swap (THE fix)

Old layout (unlearnable):
```
[k1 v1 k2 v2 ... kK vK | distractors | q1 0 q2 0 ... qQ 0]
targets: -100 at all positions except placeholder positions (1, 3, 5, ... within query block)
```

New layout (Zoology / Arora et al. 2024):
```
[k1 v1 k2 v2 ... kK vK | distractors | q1 v_q1 q2 v_q2 ... qQ v_qQ]
targets: -100 everywhere except at query-key positions (0, 2, 4, ... within query block)
         target[query_key_pos] = input[query_key_pos + 1] = v_q (standard next-token LM)
```

Verification done before trusting the fix:
- Independent 2-layer `nn.Transformer` GPT on the old layout: plateau at `per_query_acc = 1/K` across T=8 K=2, T=32 K=4, T=64 K=16; no breakthrough in 25k steps on T=32 K=4.
- Same independent GPT on the new layout: `per_query_acc = 1.0` at step 300 on T=16 K=4 vocab=64.

### 2. `src/data/dataset.py` — collate signature fix

`DataLoader(batch_size=None)` passes the dataset's yielded item directly to `collate_fn` (not wrapped in a list). The old code asserted `len(items) == 1` and did `items[0]`; every fetch raised `AssertionError: MQARTrainDataset yields one batch per __next__` before the model saw data.

### 3. `src/training/train.py` — patience metric

`best_per_seq_acc` is 0 until a model is nearly perfect, so `ev.per_seq_acc > best_per_seq_acc` was always `False` after the first eval, and patience fired at step 6000 regardless of whether the model was learning. Changed to track `best_per_query_acc` with a `1e-3` min-delta for patience; threshold still on `per_seq_acc ≥ 0.99` (Zoology convergence convention). `best_per_query_acc` now also in `results.json`.

### 4. `src/models/encoder.py` — drop `rwkv6_lucid_delta`

Removed from `SUPPORTED_BACKBONES` because in `mode="recurrent"` the TimeMix at `formal_v1/rwkv6_time_mix.py:991-996` silently makes the variant a LUCID-only run — `delta_params` get zero gradient. The time-mix comment at `:1001-1003` documents this mutual exclusivity as intentional ("mixing these would confound attribution"). Also set `delta_warmstart=delta_rule` for the remaining `rwkv6_delta` variant (see §3 of `outputs/REPORT_phase1a.md` for the warmstart probe result).

### 5. `scripts/run_cohort_reduced.sh` — drop `rwkv6_lucid_delta`

Removed from the `BACKBONES` array to match `SUPPORTED_BACKBONES`.

### 6. `scripts/analyze_cohort.py` — Zoology-style steps-to-solve

Added `_steps_to(metrics_path, threshold)` that scans `metrics.jsonl` for the first eval step crossing a per-query-acc threshold. Adds `steps_to_0.5` and `steps_to_0.9` columns to the summary table. `"—"` means the threshold was never crossed within the trained budget.

### 7. `tests/test_mqar_generator.py` — update for new layout

`test_query_targets_match_presented_pairs` now looks up the query key at `ids[qp]` (new layout: query key is AT the target position) instead of `ids[qp - 1]` (old layout: query key was before the placeholder). Also added a sanity check that the value appears as the NEXT input token (next-token-LM contract).

### 8. `tests/test_synthetic_model.py` — warmup before grad-flow check

`test_backward_flows_to_all_params` asserted that every parameter has a non-zero gradient after ONE forward-backward, which contradicts RWKV-6's documented zero-init LoRA contract (`time_maa_w1`, `time_decay_w1`, `delta_params.a1` are all zero-initialized, so their LoRA partners receive exactly-zero gradient on step 0 by construction). Changed to warm up with 3 SGD steps before the gradient-flow check.

---

## Phase 1a result (T=64, K=16, seed=42, single seed)

```
backbone           | verdict | best_per_seq_acc | steps_to_0.9 | wall_min
transformer (bidir)|  PASS   |      0.998       |     2000     |   0.9
transformer_causal |  PASS   |      0.999       |     5000     |   2.1
rwkv6_lucid        |  PASS   |      0.996       |     5000     |   9.5
rwkv6              |  FAIL   |      0.000       |      —       |   6.3
rwkv6_delta        |  FAIL   |      0.000       |      —       |  63.0
mamba              |  FAIL   |      0.000       |      —       |  28.3
mamba2             |  FAIL   |      0.000       |      —       |   8.3
```

Followup probe: `rwkv6_delta` rerun with `delta_warmstart=True` (a0=-5, iclr≈0.013 at t=0) — still FAIL, same trajectory as `warmstart=False`. Side-by-side table in `outputs/REPORT_phase1a.md` §3.

Full per-run artefacts under `outputs/cohort_reduced/<backbone>_T64_seed42/`; flat CSV at `outputs/cohort_reduced/_index.csv`; full report at `outputs/REPORT_phase1a.md`.

---

## How to reproduce

```bash
cd experiments/synthetics_v1

# 1. One-time env
uv sync
bash scripts/setup_symlinks.sh

# 2. Tests (30 expected, all green; verified on current main)
uv run --with pytest python -m pytest tests/ -v

# 3. Phase 0 smoke (~3 min). Note: 200 steps is below breakthrough threshold
#    even for Transformer at this vocab — smoke only checks infra, not learning.
uv run python scripts/debug_run.py

# 4. Phase 1a (T=64 cohort). ~2 hours wall-clock — delta is pure-PyTorch
#    and dominates; everything else is under 30 min.
bash scripts/run_cohort_reduced.sh --T64

# 5. Summary table
uv run python scripts/analyze_cohort.py --root outputs/cohort_reduced
```

---

## Recommended next steps for the PI

Copied from `outputs/REPORT_phase1a.md` "Recommended next steps" section — see that file for full context.

1. ~~Rerun `rwkv6_delta` with `delta_warmstart=True`.~~ Done during this session — same FAIL. The warmstart toggle alone does not unlock delta-rule recall; next step is a forward-path audit.
2. **Audit the formal_v1 delta forward against the Zoology / RWKV-7 reference.** Our `_recurrent_delta_scan` + `DeltaRuleParams` may differ from the literature in a way that prevents the recall lift.
3. **Phase 1b (T=256 cohort)** — ~2.5h. Confirms LUCID's lift extrapolates to longer sequences, where linear-mixer advantages should matter. The `delta` row of that table will be uninformative until step 2 above is resolved.
4. **3-seed resampling of the T=64 cohort** — ~5h, would give error bars for the `LUCID > delta` claim.
5. **Pre-empt the same layout bug in `state_tracking.py` / `induction.py`** (both planned per `CLAUDE.md`) before they ship.
