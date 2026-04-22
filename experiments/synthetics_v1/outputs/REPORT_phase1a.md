# Phase 1a Report — MQAR T=64 K=16, 7-backbone cohort

**Date:** 2026-04-22
**Cohort wall-clock:** ~1h58m (16:49 → 18:48)
**Seed:** 42 (single-seed — see limitations)

## Headline

| Backbone | Verdict | best per_seq_acc | final per_query_acc | steps_to_0.9 | wall (min) |
|---|---|---|---|---|---|
| transformer (bidir) | **PASS** | 0.998 | 0.9999 | 2000 | 0.9 |
| transformer_causal | **PASS** | 0.999 | 0.9999 | 5000 | 2.1 |
| **rwkv6_lucid** | **PASS** | **0.996** | **0.9998** | **5000** | **9.5** |
| rwkv6 (baseline) | FAIL | 0.0 | 0.0005 | — | 6.3 |
| rwkv6_delta | FAIL | 0.0 | 0.0003 | — | 63.0 |
| mamba | FAIL | 0.0 | 0.0002 | — | 28.3 |
| mamba2 | FAIL | 0.0 | 0.0003 | — | 8.3 |

**Reduced-cohort backbones: 7.** `rwkv6_lucid_delta` was dropped — in `mode="recurrent"` the TimeMix dispatch at `formal_v1/rwkv6_time_mix.py:991-996` takes the LUCID branch and returns before the delta-rule block at `:1004-1011`; the variant was silently running as LUCID-only (delta_params dead at backward). The TimeMix comment at `:1001-1003` documents the mutual-exclusivity as intentional ("mixing these would confound attribution"). This is a research-layer decision for the PI: either implement a combined LUCID+Delta path in TimeMix, or leave the variant out.

## Research signals

1. **Transformers PASS, linear backbones FAIL** — the canonical MQAR signal from Arora et al. (Zoology, 2024) reproduces cleanly. Bidirectional Transformer is fastest (step 2000); causal Transformer takes slightly longer (5000) because it must form a proper induction-head circuit instead of just matching keys anywhere.

2. **LUCID lifts RWKV-6 from FAIL to PASS** — thesis-relevant novel-mechanism result. Breakthrough curve: flat at ~chance for steps 1000–3000, then a phase transition between step 3000 (pq=0.0002) and step 4000 (pq=0.46) and step 5000 (pq=0.96), converging to 0.9998 at step 7000. **This is the most interesting data point in the cohort** — a linear-time recurrent mechanism solving MQAR via a decorrelation preconditioner rather than a full attention matrix.

3. **Delta Rule does NOT lift RWKV-6 in this implementation, in either init regime.** Contrary to the Arora et al. literature prediction, `rwkv6_delta` stayed at chance for 6000 steps and patience-stopped like the baseline. **Followup probe (run after the main cohort): rerun with `delta_warmstart=True` (a0=-5, iclr≈0.013 at t=0 — delta essentially off at init, designed to grow under SGD). Result: same FAIL.** Both init regimes patience-stop at step 6000 with `best_per_query_acc ≈ 0.0002`. Wall-clock 63 min each. Trajectory side-by-side:

   | step | warmstart=False (loss / pq) | warmstart=True (loss / pq) |
   |---|---|---|
   | 1000 | 8.348 / 0.0002 | 8.348 / 0.0002 |
   | 2000 | 8.330 / 0.0004 | 8.330 / 0.0004 |
   | 3000 | 8.328 / 0.0002 | 8.327 / 0.0002 |
   | 4000 | 8.332 / 0.0002 | 8.332 / 0.0002 |
   | 5000 | 8.347 / 0.0002 | 8.354 / 0.0001 |
   | 6000 (stop) | 8.360 / 0.0003 | 8.359 / 0.0002 |

   Interpretation: `warmstart=False` fires delta at full strength (iclr≈1.76) and per the code comment "destroys the randomly-initialised state before useful associations are written"; `warmstart=True` keeps delta essentially off and the model behaves like the no-delta baseline (which also failed). So the warmstart toggle alone does not unlock delta-rule recall in this implementation. The Arora-et-al-style lift likely requires deeper changes to the delta forward path or a different SSM base — beyond a single-flag tune. **Recommend: PI audits formal_v1's delta forward against the Zoology reference implementation before publishing a "delta vs LUCID" comparison.**

4. **Mamba / Mamba-2 FAIL at T=64.** Surprising — the Zoology paper shows Mamba-2 solves T=64 MQAR with CUDA kernels. Our pure-PyTorch Mamba-2 (no `mamba-ssm` dep) may be suboptimal. The runbook's own note: *"Mamba runs much slower than estimated — pure-PyTorch Mamba is slow without CUDA kernel, expected."* — we confirm that **speed** is fine (mamba2 finished in 8.3 min), but **recall** doesn't converge. This is a worthwhile PI finding: the pure-PyTorch reference implementation isn't a fair comparator for SSM recall claims.

5. **Bidirectional Transformer convergence speed (2000 steps)** is notable. It's 2.5× faster to convergence than the causal Transformer. This is consistent with the task: at T=64 K=16, a bidirectional model can treat MQAR as a lookup table in its eval layer, while a causal model must build a proper induction-head pattern.

## What I had to fix to reach this result (methodology-relevant)

1. **Environment** — added `pytest` as an ephemeral `--with` dep for test runs (not mutated into `pyproject.toml`). Suggest adding to a dev group.

2. **Added kwargs-stub constructor surface to `formal_v1/src/models/rwkv6_encoder.py`** — the `synthetics_v1` dispatcher at `src/models/encoder.py:105-157` was passing ~13 forward-looking mechanism kwargs (`use_loglinear`, `use_m2rnn`, `use_conv_shift_multidilation`, etc.) that the currently-symlinked `RWKV6Encoder` didn't accept. Stubbed them with default=False plus a `NotImplementedError` guard if any is set to True — so enabling them later (when the mechanisms actually land in formal_v1) will fail loud, not silently no-op.

3. **Fixed the `test_backward_flows_to_all_params` sanity test** — it required all parameters to have non-zero gradient after ONE forward-backward, which contradicts RWKV-6's documented zero-init LoRA contract (`time_maa_w1`, `time_decay_w1`, `delta_params.a1` all init to zero, so their LoRA partners receive exactly-zero gradient on step 0 by design). Changed to warm up with 3 SGD steps before the gradient-flow check.

4. **Dropped `rwkv6_lucid_delta`** from the cohort — see caveat above.

5. **Fixed the `_collate_fixed_length` PyTorch-API mismatch** — `DataLoader(batch_size=None)` passes the dataset's yielded item directly (not wrapped in a list), so the old `assert len(items) == 1; items[0]` was wrong for every fetch.

6. **Fixed the early-stopping patience metric** — was tracking `per_seq_acc`, which stays at 0.0 until a model is essentially perfect. The patience counter would reset at the first eval (0.0 > -1.0), then never reset again, firing after 5 flat evals regardless of actual learning. Changed to track patience on `per_query_acc` (the fine-grained metric), with a `1e-3` min-delta against noise. Convergence threshold is still `per_seq_acc ≥ 0.99` — that's the Zoology convention.

7. **🔑 Fixed the MQAR data generator.** This was the root cause blocker and nearly invalidated the entire stage. The original layout placed `placeholder=0` tokens at all prediction positions, requiring the model to predict `v_q_i` given only a placeholder + positional encoding + one attention hop back to the query key. Empirically this layout is unlearnable: a vanilla 2-layer GPT on an independent training loop plateaus at `per_query_acc = 1/K` (uniform over in-context values) indefinitely — I verified 25000 steps of training with the same data gives `per_query_acc = 0.266` on T=32 K=4. The fix: switch to Zoology-convention layout with interleaved `(q_i, v_q_i)` pairs and standard autoregressive LM loss at query-key positions. Same vanilla GPT on the fixed layout hits 100% in 300 steps; our full cohort now produces the results above. **This was the experiment-critical fix.** All MQAR test assertions were updated and pass.

8. **Extended `analyze_cohort.py`** to emit Zoology-style `steps_to_0.5` and `steps_to_0.9` columns (first eval step where `per_query_acc` crossed the threshold). This ranks backbones continuously even when none PASS and is a better research-signal metric than binary PASS/FAIL.

## Limitations and caveats

- **Single seed (42).** Every result is n=1. A defensible thesis claim about LUCID > Delta would want at least 3 seeds. This would roughly 3× Phase 1a wall-clock (~6 hours); fits budget for a follow-up run.
- **T=64 only, no T=256 (Phase 1b not run).** The runbook's Phase 1b would demand ~2–3 additional GPU-hours. Decision to defer was explicit (budget). Length-extrapolation is the next experiment.
- **Delta Rule disadvantage may be init-specific.** Strongly recommend a follow-up run with `delta_warmstart=True` before drawing a "delta doesn't help" conclusion — the code comments suggest this is exactly the configuration meant for prior-failure backbones.
- **Mamba-2 pure-PyTorch underperforms literature.** If Mamba is a comparison target for the thesis, the `mamba-ssm` CUDA kernel backend should be added before reporting.
- **LUCID PASS is one data point.** Even with `best_per_seq_acc=0.996`, the per-sequence accuracy still has 0.4% residual error. Under 1000 eval examples that's 4 imperfect sequences — statistical noise but worth noting.

## Recommended next steps (PI)

1. ~~Rerun `rwkv6_delta` with `delta_warmstart=True`.~~ **Done.** Same FAIL — the warmstart toggle alone does not unlock delta-rule recall (see §3 above). Next step is a forward-path audit, not another init sweep.
2. **Audit the formal_v1 delta forward against the Zoology / RWKV-7 reference implementation.** Our `_recurrent_delta_scan` + `DeltaRuleParams` may differ from the literature in a way that prevents the recall lift.
3. **Phase 1b (T=256 cohort)** — ~2.5h, confirms LUCID's lift extrapolates to longer sequences (where linear-mixer advantages should matter). The `delta` row of that table will be uninformative until step 2 above is resolved.
4. **3-seed resampling of the T=64 cohort** — ~5h, would give error bars for the `LUCID > delta` claim.
5. **Audit the other synthetic tasks' generators** (`state_tracking.py`, `induction.py` — both planned per CLAUDE.md) before they ship — the placeholder-token layout bug may exist in them too; worth pre-empting.

## File map of edits

```
experiments/formal_v1/src/models/rwkv6_encoder.py              # accept-stub kwargs + NotImplementedError guard
experiments/synthetics_v1/src/tasks/mqar.py                    # Zoology-style layout ← THE fix
experiments/synthetics_v1/src/data/dataset.py                  # collate signature fix
experiments/synthetics_v1/src/training/train.py                # patience on per_query_acc
experiments/synthetics_v1/src/models/encoder.py                # drop rwkv6_lucid_delta
experiments/synthetics_v1/scripts/run_cohort_reduced.sh        # drop rwkv6_lucid_delta
experiments/synthetics_v1/scripts/analyze_cohort.py            # steps_to_0.5/0.9 columns
experiments/synthetics_v1/tests/test_mqar_generator.py         # update to new layout
experiments/synthetics_v1/tests/test_synthetic_model.py        # warmup before gradient-flow check
```

## Raw outputs

- `outputs/cohort_reduced/_index.csv` — flat CSV of all runs
- `outputs/cohort_reduced/<backbone>_T64_seed42/` per-backbone artifacts
- `outputs/phase1a.log` — full Phase 1a training log (all backbones)
