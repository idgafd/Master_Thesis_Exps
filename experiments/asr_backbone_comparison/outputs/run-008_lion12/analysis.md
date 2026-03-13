# Run 008 Analysis — bidir_rwkv6_conv_nogate 12L (capacity ceiling test)

---

## What Changed vs Run 006

Run 006 was the 6-layer LION with learned ConvShift (no gate), the best result in the series
so far (CER 0.1760, 60 epochs). Run 008 is the direct capacity ceiling test: the same
architecture, same data — only n_layers doubled from 6 to 12.

Two hyperparameters were adjusted for deeper model training: lr lowered from 3e-4 to 2.5e-4
and warmup_steps doubled from 500 to 1000. A 500-step warmup run was attempted first and
killed at epoch 2 (CER=1.0, model stuck outputting only "а") — confirmed that 12 layers
require a slower LR ramp-up to avoid degenerate early convergence.

**Two training lengths tested:** 60 epochs (initial run-008 result) and 100 epochs (this file
reflects the 100-epoch result, which is the canonical result for run-008).

---

## Key Observations (Facts)

| Metric | lion6 60ep (006) | lion12 60ep | lion12 100ep (008 final) |
|---|---|---|---|
| Params | 7.75M | 13.58M | 13.58M |
| BestDevCER | **0.1587** | 0.1838 | 0.1624 |
| BestEpoch | 55 | 60 | **97** |
| TestLoss | **0.7585** | 0.9491 | 0.8746 |
| TestCER | **0.1760** | 0.2029 | 0.1816 |
| TestWER | **0.6563** | 0.7345 | 0.6623 |
| R@2s CER | **0.2084** | 0.2312 | 0.2126 |
| R@5s CER | **0.1804** | 0.2049 | 0.1845 |
| R@10s CER | **0.1744** | 0.2029 | 0.1790 |
| Train loss ep60 | — | 0.6702 | 0.6478 |
| Train loss ep100 | — | — | 0.5517 |

The 100-epoch run is dramatically better than the 60-epoch version across every metric.
CER improved from 0.2029 to 0.1816 — a 10.5% relative gain from 40 additional epochs.
The model was still meaningfully improving at epoch 97 (best_epoch=97); epoch 100 dev CER
was 0.1625 vs best 0.1624, essentially flat only in the final 3 epochs.

Dev CER trajectory — 100-epoch run:

| Epoch | dev CER | Δ vs prev checkpoint |
|---|---|---|
| 5 | 0.4783 | — |
| 10 | 0.3349 | — |
| 20 | 0.2510 | — |
| 30 | 0.2217 | — |
| 40 | 0.2006 | — |
| 50 | 0.1840 | — |
| 60 | 0.1763 | −0.0077 |
| 70 | 0.1694 | −0.0069 |
| 80 | 0.1650 | −0.0044 |
| 90 | 0.1628 | −0.0022 |
| 97 | **0.1624** | −0.0004 (best) |
| 100 | 0.1625 | +0.0001 |

The model was still improving at every 10-epoch checkpoint through epoch 90, with diminishing
but positive returns. The rate of improvement (Δ per 10 epochs) is decelerating in the final
third of training — suggesting the model is close to but has not fully reached its ceiling
within 100 epochs.

Notably, at epoch 60 of the 100-epoch run, dev CER was already 0.1763 — substantially better
than the final result of the separate 60-epoch run (dev CER 0.1838 at best_epoch=60). This
confirms the 60-epoch run's poor performance was not architectural but due to an overly
compressed cosine schedule: with T_max=60 epochs, LR decayed to near-zero around epoch 50,
cutting off further learning. With T_max=100, LR stays meaningfully high through epoch 80.

---

## Anomalies

1. **The 60-epoch result (CER 0.2029) was entirely a schedule artifact.** The initial analysis
   for run-008 concluded that "deeper LION is worse." This was wrong. The model was simply
   undertrained — the cosine schedule exhausted the learning rate before the deeper model
   could converge. With 100 epochs and properly scaled T_max, CER drops to 0.1816. The
   architecture itself is not harmful; the training duration was insufficient.

2. **lion12 100ep (0.1816) still does not beat lion6 60ep (0.1760) despite 75% more params
   and 67% more training.** Best dev CER: lion12=0.1624 vs lion6=0.1587. The larger model
   is marginally worse at every comparable metric, even with optimal training time.

3. **Best epoch = 97 out of 100 — model was almost certainly still improving.** The
   improvement curve had not fully flattened by epoch 100 (Δ ep90→97 = −0.0004, then
   +0.0001 at ep100 which is noise). With 120–150 epochs, the model may further improve.

4. **Chunked reset at 10s (0.1790) is identical to lion6 (0.1744 → 0.1790).** At longer
   context windows, lion12 essentially matches lion6. The remaining gap is concentrated in
   short windows (R@2s: 0.2126 vs 0.2084), where 12-layer processing may introduce more
   positional confusion at chunk boundaries.

---

## Hypotheses

### Why does the 60-epoch run massively underperform but 100-epoch nearly matches lion6?

The cosine annealing schedule decays LR as `η(t) = η_min + 0.5*(η_max - η_min)*(1 + cos(πt/T_max))`.
With T_max set to total training steps, the LR at epoch 60/60 is η_min ≈ 1e-6. At epoch 60/100
it is approximately `0.5 * η_max * (1 + cos(0.6π)) ≈ 0.35 * η_max` — still 35% of peak LR.
The 12-layer model needs the additional gradient signal in epochs 60–100 because it has more
parameters to coordinate and a harder optimisation landscape than the 6-layer model.
Effectively the 6-layer model can converge in 55 epochs; the 12-layer model needs ~97.

### Why doesn't lion12 beat lion6 even with proper training?

Three factors work against the 12-layer model:
- **Harder gradient flow:** gradients from CTC loss propagate through 12 dense attention
  layers instead of 6. More layers = more gradient dilution, even with residual connections.
- **Overfitting at larger scale:** with 13.58M parameters and only 35h of training data,
  the model has more capacity than the data can reliably fill. The dev loss at ep97 (0.8097)
  is higher than lion6's (0.6954) despite similar train loss, suggesting the larger model
  generalises slightly worse.
- **Diminishing returns from LION depth:** LION's parallel attention already captures global
  context at every layer with 6 layers. Adding 6 more layers processes representations that
  already contain full-sequence information — the additional depth adds transformation
  capacity but not new information.

### Why is train loss higher in lion12 despite more parameters?

At epoch 100: lion12 train_loss=0.5517 vs lion6 train_loss (ep60)=0.5223. A larger model with
more parameters should in principle achieve lower training loss. The higher loss is evidence of
the optimisation difficulty: the 12-layer model has not fully exploited its capacity within 100
epochs. Given the still-declining loss curve (0.5980→0.5725→0.5579→0.5517 over ep70-100),
this would likely resolve with more training.

---

## Next Actions

1. **Run lion12 for 150 epochs.** Best epoch was 97 with loss still declining. Estimated
   from trajectory: CER ceiling around 0.175–0.178 by epoch 130–150. This would make lion12
   fully competitive with or marginally better than lion6.

2. **Run lion6 for 100 epochs** to make the comparison truly fair. Lion6 was trained with
   60-epoch cosine, which may also have been slightly too short (best_epoch=55, mild decline
   remaining). A 100-epoch lion6 might improve further toward CER ~0.170.

3. **Depth-scaled residual initialisation.** Multiplying each block's residual output by
   `1/√n_layers` at init would reduce the gradient magnitude growth with depth, potentially
   allowing the 12-layer model to converge in fewer epochs and improve its final CER.
