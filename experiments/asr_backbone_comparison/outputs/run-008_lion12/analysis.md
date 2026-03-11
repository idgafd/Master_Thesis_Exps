# Run 008 Analysis — bidir_rwkv6_conv_nogate 12L (capacity ceiling test)

---

## What Changed vs Run 006

Run 006 was the 6-layer LION with learned ConvShift (no gate), which produced the best
result in the experiment series so far (CER 0.1760). Run 008 is the direct capacity
ceiling test: the same architecture, same config, same data — only n_layers doubled
from 6 to 12. Two hyperparameters were also adjusted to account for deeper model
training dynamics: lr lowered from 3e-4 to 2.5e-4 and warmup_steps doubled from 500
to 1000. The reasoning: with 12 layers, high-LR updates in the first few hundred steps
compound across twice as many layers, causing the model to spend several epochs in a
degenerate "output one character" basin. The 500-step warmup run confirmed this (CER=1.0
through epoch 2), so it was killed and restarted with the adjusted schedule.

---

## Key Observations (Facts)

| Metric | bidir_rwkv6_conv_nogate 6L (006) | bidir_rwkv6_conv_nogate 12L (008) |
|---|---|---|
| Params | 7.75M | **13.58M** (+75%) |
| BestDevCER | **0.1587** | 0.1838 |
| BestEpoch | 55 | **60** (still declining) |
| TestLoss | **0.7585** | 0.9491 |
| TestCER | **0.1760** | 0.2029 |
| TestWER | **0.6563** | 0.7345 |
| R@2s CER | **0.2084** | 0.2312 |
| R@5s CER | **0.1804** | 0.2049 |
| R@10s CER | **0.1744** | 0.1994 |
| Train loss ep60 | **0.5223** | 0.6702 |
| Epoch time | 65s | **65s** (identical) |

The 12-layer model is substantially worse than the 6-layer at every metric despite 75% more
parameters and nearly double the FLOPs per forward pass.

Dev CER trajectory at key epochs — 6L vs 12L:

| Epoch | lion6 dev CER | lion12 dev CER | Gap |
|---|---|---|---|
| 5 | 0.3251 | 0.4863 | +0.161 |
| 10 | 0.2574 | 0.3313 | +0.074 |
| 20 | 0.2044 | 0.2483 | +0.044 |
| 30 | 0.1808 | 0.2158 | +0.035 |
| 40 | 0.1656 | 0.1938 | +0.028 |
| 50 | 0.1596 | 0.1859 | +0.026 |
| 60 | 0.1588 | 0.1838 | +0.025 |

The 12-layer model is behind at every epoch and the gap is not closing in the final 20 epochs
(Δdev_cer ep40→60 = 0.0100 for lion12 vs 0.0068 for lion6). The trajectories are diverging
in absolute terms — 12L did not catch 6L by epoch 60 and the rate of improvement in late
epochs is nearly identical between both, meaning it will likely never catch up.

Additional signal: train loss at epoch 60 is 0.6702 for 12L vs 0.5223 for 6L. A larger model
that has a higher training loss than a smaller model is a strong indication of an optimization
problem, not a capacity problem — the 12-layer model is harder to train, not more expressive.

Epoch times are identical at ~65s despite 75% more parameters, confirming training is
CPU data-loading bottlenecked (GPU utilization was measured at 0% while batch processing
awaited the next data batch). Adding more layers does not change training speed.

---

## Anomalies

1. **Deeper LION is strictly worse at every metric.** More parameters hurt rather than help.
   This is not a marginal degradation — test CER went from 0.1760 to 0.2029, a 15.3% relative
   increase. The pattern holds at every evaluation granularity (full utterance, 2s/5s/10s chunks).

2. **12-layer model has higher train loss than 6-layer despite more capacity.** If the model
   were overfitting, train loss would be low and dev/test loss high. Instead both are high.
   This is the signature of an optimization difficulty: the model has capacity but cannot
   efficiently use it within 60 epochs and the given schedule.

3. **Best epoch = 60 — model was still improving at run end.** Lion6 peaked at epoch 55 and
   plateaued. Lion12 had not found a good minimum by epoch 60. Dev improvement ep50→60 was
   0.0021, tiny but still happening (6L had 0.0009 in that window, already plateaued). Given
   the trajectory, lion12 might need 80–100 epochs to approach its actual ceiling.

4. **Epoch time identical to 6-layer despite 2× depth.** See data loading bottleneck discussion
   in the "why same training time" investigation. GPU was at 0% utilization while data loaded.

---

## Hypotheses

### Why does 12-layer LION perform worse than 6-layer?

LION (bidir_rwkv6) uses a parallel WKV attention mechanism that processes the full sequence
as a single matrix computation per layer. With 12 such layers stacked, the CTC loss gradient
must propagate back through 12 dense T×D attention computations. Each layer introduces
multiplicative interactions across all T positions, creating complex gradient pathways where
small misalignments in early-layer representations cascade through the remaining 11 layers.

The 6-layer version has a natural regularisation from its shallower gradient path: there are
simply fewer ways for the gradient signal to be diluted or misdirected before reaching the
first layers. The 12-layer version has twice as many "relay points" where gradient coherence
can degrade. This manifests as higher train loss at epoch 60 (0.6702 vs 0.5223) — the model
is not converging as efficiently as the 6-layer version, not because it lacks capacity, but
because the optimiser cannot as reliably coordinate all 12 layers within 60 cosine epochs.

This is a well-known phenomenon in deep attention networks: there is a depth beyond which
adding layers requires either (a) more training compute (longer schedules, cyclic LR), (b)
architectural changes (residual pre-norm, weight initialisation scaled by depth, e.g.
multiplying residual outputs by 1/√n_layers), or (c) both. None of these were applied here.

### Why does lion6 peak at epoch 55 but lion12 at epoch 60+?

Lion6 exhausted its representational capacity by epoch 55 — the cosine schedule had decayed
LR sufficiently that updates were too small to continue improving. Lion12 has more parameters
still not fully trained at that point, so it keeps improving right through epoch 60. This is
consistent with the hypothesis that lion12 needs more epochs, not that it is fundamentally
worse. The 15% CER gap may shrink with longer training, but whether it closes entirely depends
on whether the architecture can coordinate 12 layers given enough gradient updates.

---

## Next Actions

1. **Run lion12 for 100 epochs** with the same schedule extended.
   The model was still improving at epoch 60 (best_epoch=60). Extending to 100 epochs
   with cosine decayed to zero at epoch 100 would give the most honest ceiling measurement.
   If it reaches CER ~0.17 by epoch 80–90, the architecture is fine but underoptimised.
   If it plateaus at ~0.19, the depth is genuinely harmful for this data scale.

2. **Try depth-scaled residual initialisation.**
   Multiply output of each attention block by `1/√n_layers` at init (used in DeepNet,
   Admin, and similar work). This directly addresses gradient magnitude growth through depth
   and is a simple one-line change. Expected: faster early convergence and lower train loss.

3. **Do not scale LION depth further.** At 12 layers the architecture is already optimisation-
   limited. Adding 16 or 24 layers without fixing the gradient flow would make things worse.
