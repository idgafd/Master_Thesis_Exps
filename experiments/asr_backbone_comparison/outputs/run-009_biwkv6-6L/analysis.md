# Run 009 Analysis — biwkv6_no_conv_no_gate 6L (capacity ceiling test)

---

## What Changed vs Run 007

Run 007 tested Bi-WKV at n_layers=3 (6 RWKV-6 blocks, 7.74M params, 60 epochs) — matched
to lion6's parameter budget. Dev CER plateaued at ~0.201 from epoch 30, leaving open whether
the plateau was a capacity ceiling or an architectural ceiling. Run 009 doubles depth to
n_layers=6 (12 RWKV-6 blocks, 13.56M params) and extends training to 100 epochs to separate
these two hypotheses. Gate and ConvShift excluded — run-007 confirmed gate gives zero benefit.

**Two training lengths tested:** 60 epochs (initial run-009 result) and 100 epochs (this file
reflects the 100-epoch result, the canonical result for run-009).

---

## Key Observations (Facts)

| Metric | biwkv6 3L 60ep (007) | biwkv6 6L 60ep | biwkv6 6L 100ep (009 final) |
|---|---|---|---|
| Params | 7.74M | 13.56M | 13.56M |
| BestDevCER | 0.2011 | 0.1850 | **0.1713** |
| BestEpoch | 55 | 58 | **80** |
| TestLoss | 0.9123 | 0.8242 | **0.8185** |
| TestCER | 0.2201 | 0.2039 | **0.1894** |
| TestWER | 0.7788 | 0.7264 | **0.6774** |
| R@2s CER | 0.2504 | 0.2346 | **0.2213** |
| R@5s CER | 0.2256 | 0.2095 | **0.1950** |
| R@10s CER | 0.2201 | 0.2040 | **0.1895** |
| C@10s CER | 0.7137 | 0.7585 | **0.7695** |

Each step of scaling meaningfully improves the result: 3L→6L and 60ep→100ep both contribute.
Total improvement from biwkv6 3L 60ep to biwkv6 6L 100ep: CER 0.2201 → 0.1894, a 14.0%
relative gain.

Dev CER trajectory — 100-epoch run vs 60-epoch 3L:

| Epoch | biwkv6 3L 60ep | biwkv6 6L 100ep |
|---|---|---|
| 5 | 0.3134 | 0.3275 |
| 10 | 0.2620 | 0.2674 |
| 20 | 0.2276 | 0.2199 |
| 30 | 0.2160 | 0.2086 |
| 40 | 0.2054 | 0.1933 |
| 50 | 0.2023 | 0.1839 |
| 60 | 0.2015 | 0.1806 |
| 70 | — | 0.1764 |
| 80 | — | **0.1713** (best) |
| 90 | — | 0.1713 (flat) |
| 99 | — | 0.1715 (flat) |

Early stopping triggered at epoch 99 (patience=20, last improvement at epoch 80). The model
hit a genuine plateau: dev CER 0.1713 from epoch 80 through epoch 99 with no improvement.
This is qualitatively different from lion12 (still declining at epoch 97) — BiWKV 6L found
its actual ceiling within 100 epochs.

### Carry State

| Mode | biwkv6 3L 60ep | biwkv6 6L 60ep | biwkv6 6L 100ep |
|---|---|---|---|
| Reset CER | 0.2201 | 0.2039 | **0.1894** |
| C@2s CER | 0.7122 | 0.7593 | 0.7733 |
| C@10s CER | 0.7137 | 0.7585 | 0.7695 |

Carry state remains broken and is marginally worse with 100 epochs training (0.7695 vs 0.7585
at 60ep). More training makes the reset-mode performance better but does not help carry.
The carry state failure is structural (backward branch train/inference mismatch) — additional
training epochs do not change this.

---

## Anomalies

1. **Real plateau at epoch 80 — architecture ceiling confirmed.** Unlike lion12 (still
   declining at epoch 97), biwkv6 6L genuinely stops improving at epoch 80. Epochs 80–99
   show essentially zero dev CER change (0.1713 held exactly for 19 consecutive epochs before
   early stopping). This is not a schedule artifact — the model has exhausted what it can
   learn from this data at this scale.

2. **6L BiWKV converged faster than 6L LION in early epochs.** At epoch 10: biwkv6-6L
   dev CER=0.2674 vs lion12 dev CER=0.3349. At epoch 20: 0.2199 vs 0.2510. The sequential
   RWKV-6 architecture learns useful local representations faster than the parallel attention
   LION in early training, consistent with the inductive bias hypothesis (RWKV-6 decay
   provides a useful local-context prior from the start).

3. **The 3L plateau at epoch 30 (CER ~0.20) was partially a capacity ceiling and partially
   a depth effect.** At 6L, the same trajectory reaches epoch 30 at 0.2086 (vs 0.2160 for
   3L) — the deeper model was already extracting more information by that point. The 3L wall
   was not just capacity; 6L kept descending past it to 0.1713. This confirms that BiWKV
   benefits meaningfully from additional depth up to at least 6 bidir layers.

4. **Carry state degradation with better reset performance.** Across all three BiWKV entries,
   as reset CER improves (0.2201 → 0.2039 → 0.1894), carry CER worsens (0.7137 → 0.7585 →
   0.7695). A model that has learned stronger bidirectional dependencies is more damaged by
   the carry-state architectural mismatch — the backward branch's corrupted state interferes
   more severely with a well-trained forward branch.

---

## Hypotheses

### Why does BiWKV scale better than LION with depth but hit a hard ceiling?

RWKV-6 blocks use gated state updates that provide a natural regulariser on gradient flow
across depth — each block's gate explicitly controls how much information passes through,
preventing gradient magnitudes from growing uncontrollably with more layers. This is why
BiWKV 6L converges stably and quickly while LION 12L struggles in early epochs.

However, RWKV-6's exponential decay on the WKV kernel limits the effective context range per
block regardless of depth. Adding more layers increases representational depth (more
transformations of features) but does not increase the temporal span each representation can
see. Once the model has enough layers to apply sufficient transformations within the reachable
context window, additional layers provide diminishing returns. The plateau at epoch 80 marks
this point: the model has fully utilised the temporal context available to it at 6 bidir
layers, and more training cannot recover what the architecture structurally cannot see.

Contrast with LION: the parallel attention mechanism sees the full sequence at every layer,
so additional LION layers always have access to new relational information (different
compositions of full-sequence attention). LION does not hit the same structural ceiling —
it hits an optimisation ceiling instead, which is solvable with more training time.

### Why is biwkv6-6L 100ep (0.1894) still worse than lion6 60ep (0.1760)?

The gap (0.013 CER) represents the structural advantage of LION's global attention over
BiWKV's range-limited recurrence. At 7.75M params/60ep, this gap was 0.044 (lion6=0.1760
vs biwkv6-3L=0.2201). At 13.56M params/100ep, it narrows to 0.013 — but does not close.
The narrowing comes from BiWKV scaling better with depth, but the residual gap reflects
information that is simply not accessible to the RWKV-6 decay-limited recurrence even with
optimal depth and training.

### Why does the 60-epoch result (CER 0.2039) look so much worse than 100-epoch (0.1894)?

Unlike lion12 where the 60ep run used a compressed cosine schedule, biwkv6-6L used the same
lr=3e-4 and warmup=500 in both runs. At 60 epochs the model simply hadn't reached its
plateau yet (best_epoch=80 in the 100ep run). The 60-epoch run was not schedule-limited —
it was just stopped before convergence.

---

## Consolidated Results — All Runs

| Model | Params | Epochs | TestCER | TestWER |
|---|---|---|---|---|
| bidir_rwkv6 LION baseline (005) | 7.74M | 60 | 0.1790 | 0.6704 |
| **bidir_rwkv6_conv_nogate 6L (006)** | **7.75M** | **60** | **0.1760** | **0.6563** |
| biwkv6_no_conv_no_gate 3L (007) | 7.74M | 60 | 0.2201 | 0.7788 |
| bidir_rwkv6_conv_nogate 12L (008) | 13.58M | 100 | 0.1816 | 0.6623 |
| biwkv6_no_conv_no_gate 6L (009) | 13.56M | 100 | 0.1894 | 0.6774 |

Key structural finding: the 6-layer 7.75M LION remains the best model at this data scale
(35h). Doubling parameters and training longer does not improve LION; it does improve BiWKV
substantially (from 0.2201 to 0.1894), but BiWKV still cannot match LION's accuracy at any
tested parameter scale.

---

## Next Actions

1. **Run hybrid carry eval on the 100ep checkpoint.** The Option B script
   (`eval_hybrid_carry.py`) confirmed that hybrid carry (fwd carry + bwd reset per chunk)
   rescues BiWKV from CER 0.71 to ~0.26 on the 3L model. The same script should be run on
   the 6L 100ep checkpoint to see if the better reset-mode model also gives better hybrid
   carry (expected: ~0.22–0.24 given improved representations).

2. **Run lion6 for 100 epochs.** Lion6 peaked at epoch 55 with a 60-epoch cosine. Running
   for 100 epochs with properly scaled T_max would show lion6's true ceiling and provide a
   fully fair comparison to lion12 at equal training length.

3. **Accept 7.75M as the optimal scale for this dataset.** The data ceiling hypothesis is
   strengthened: 35h of Ukrainian Common Voice appears to support a ~7.75M model effectively.
   Both larger models (13.58M) with extended training reach ~0.18–0.19 but not 0.176. The
   next meaningful improvement likely requires more data rather than more parameters or depth.
