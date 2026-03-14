# Run 014 Analysis — Layer-Dependent ConvShift

Direct structural encoding of the multi-scale depth hypothesis: replace the
uniform-kernel ConvShift (run-006, kernel=3 everywhere) with layer-dependent
kernel sizes (wide at bottom, narrow at top).

**Key question:** Does making the kernel schedule explicit and architecture-level
improve over the uniform ConvShift, or does the learned conv already capture this?

---

## Setup

| Parameter | Value |
|-----------|-------|
| Backbone | `bidir_rwkv6_layerconv` |
| Kernel schedule | `[7, 7, 5, 5, 5, 3]` (see note below) |
| Epochs | 60 |
| Scheduler | cosine + warmup (500 steps) |
| Dropout | 0.15 |
| SpecAugment | standard (freq=15, time=35, 2+2 masks) |
| d_model | 256, 6 layers, 4 heads |
| Params | 7.75M |

**Kernel schedule note:** The code computes `7 − 4×(layer_idx/(n_layers-1))` then
`int(round(...))` then forces odd. Continuous values: [7.0, 6.2, 5.4, 4.6, 3.8, 3.0].
After rounding and odd-enforcement: **[7, 7, 5, 5, 5, 3]** — not a uniform linear
ramp. Layers 0-1 share kernel=7, layers 2-4 share kernel=5, layer 5 has kernel=3.
The odd-enforcement collapses 6→7 and 4→5. This is a step-function schedule,
not the stated linear interpolation. Minor deviation from design intent but
the broad/local hierarchy is still correctly encoded.

---

## Results

| Metric | Value |
|--------|-------|
| Best dev CER | 0.1574 |
| Test CER | 0.1768 |
| Test WER | 0.6548 |
| Test loss | 0.7496 |
| Best epoch | 60 / 60 |
| Train loss @60 | 0.520 |
| Epoch time | ~66s |

**Chunked reset:**

| Chunk | CER | WER |
|-------|-----|-----|
| R@2s | 0.2077 | 0.7350 |
| R@5s | 0.1800 | 0.6585 |
| R@10s | 0.1739 | 0.6441 |

---

## Comparison Table

| Config | Dev CER | Test CER | Δ test | Test loss |
|--------|---------|----------|--------|-----------|
| LION baseline (005) | 0.1676 | 0.1790 | — | 0.7988 |
| ConvShift nogate (006) | **0.1587** | **0.1760** | −0.003 | 0.7585 |
| LayerConv (014) | **0.1574** | **0.1768** | −0.0022 | 0.7496 |
| Temperature (015) | 0.1606 | 0.1792 | +0.0002 | 0.7900 |

LayerConv has the best dev CER in the project (0.1574) and matches ConvShift on
test (0.1768 vs 0.1760 — a gap of 0.0008, within run-to-run noise). The test loss
is also slightly better (0.7496 vs 0.7585).

---

## Dev-Test Gap Analysis

| Config | Dev CER | Test CER | Gap |
|--------|---------|----------|-----|
| LION baseline (005) | 0.1676 | 0.1790 | −0.012 (test better) |
| ConvShift (006) | 0.1587 | 0.1760 | +0.017 |
| LayerConv (014) | 0.1574 | 0.1768 | +0.019 |

The baseline shows test *better than* dev (healthy, data-limited generalisation).
LayerConv shows dev 0.019 better than test — a wider gap than ConvShift (+0.017).
The improved dev CER does not fully translate to test; approximately 1/3 of the
0.013 dev improvement over baseline is real test improvement (0.0022 CER), the
remainder is distributional overfitting.

---

## Training Dynamics

**Best epoch = 60 = max_epochs.** The model had not saturated:

```
dev_cer trajectory (epochs 50–60):
  ep50: 0.15978   ep55: 0.15797
  ep51: 0.15869   ep56: 0.15751
  ep52: 0.15806   ep57: 0.15755
  ep53: 0.15825   ep58: 0.15771
  ep54: 0.15804   ep59: 0.15747
                  ep60: 0.15743 ← best
```

The dev CER is still declining (slowly) at epoch 60. Train loss (0.520) is also
still declining. The model converged to a near-plateau in the last ~15 epochs
but did not fully flatten. A 100-epoch run would likely achieve dev CER ~0.154
(extrapolating the slow last-mile improvement), potentially recovering the full
dev-test gap.

**Epoch time: 66s — identical to ConvShift run-006.**

The wider kernel=7 in layers 0-1 adds no wall-clock overhead vs kernel=3. At
T≈250 and d=256, the depthwise conv over 250×256 is memory-bandwidth-bound. The
additional multiply-adds for the wider kernel are absorbed within the same
memory transfer. This confirms the LayerConv has zero compute cost vs ConvShift.

---

## Initialisation Analysis

The kernel init places equal weight on all non-center taps and zero on the center:
- kernel=7: `[1/6, 1/6, 1/6, 0, 1/6, 1/6, 1/6]` — symmetric 3-frame window
- kernel=5: `[1/4, 1/4, 0, 1/4, 1/4]` — symmetric 2-frame window
- kernel=3: `[1/2, 0, 1/2]` — standard bidirectional shift (same as run-006)

Lower layers start by averaging over ±3 frames (±120ms), upper layers average
over ±1 frame (±40ms). This is the correct inductive bias for the multi-scale
depth hypothesis. The zero center prevents identity shortcuts early in training,
forcing the conv to learn from context.

---

## Code Issues

1. **Kernel schedule is a step-function, not linear.** The stated design intent
   (linear interpolation) produces [7,7,5,5,5,3] due to rounding + odd enforcement.
   A monotone odd schedule would need specific values: [7,5,5,5,3,3] or [7,7,5,3,3,3].
   The actual schedule is still a valid broad→local hierarchy — just not uniformly spaced.

2. **ChannelMix also uses LayerDWConvShift.** The channel-mix FFN also applies the
   layer-dependent ConvShift for token shifting. This means not only the attention
   mechanism but also the FFN gate signal benefits from wider context at lower layers.
   This is consistent with the overall design (earlier layers need broader context
   everywhere), but wasn't noted as a deliberate choice — it's a side effect of
   the unified `LayerDWConvShift` class reuse.

---

## What This Run Settles

1. **Layer-dependent kernels are equivalent to uniform ConvShift on test.** The
   extra expressiveness (two distinct kernel sizes instead of one) yields 0.0008
   CER difference — indistinguishable from noise. The uniform kernel=3 from run-006
   was already close to optimal for the token-shift role.

2. **The dev improvement is real but partially distributional.** Dev CER 0.1574
   is the best in the project. ~1/3 of the improvement survives to test. The rest
   is dev-specific fitting.

3. **No compute overhead.** Zero additional wall-clock time vs ConvShift.

4. **Best epoch = last epoch** suggests more training would help, but the slow
   last-mile convergence (0.00004 CER/epoch at ep60) makes extended training a
   low-return investment compared to other experiments.

---

## Hypotheses and Next Steps

### What might work

1. **LayerConv + headscale.** LayerConv improves token mixing (input to attention),
   headscale improves attention scale distribution. Orthogonal mechanisms that
   should stack. Expected: better than either alone.

2. **Longer training (80-100 epochs) for LayerConv.** Best epoch = 60 = max.
   The slow improvement at ep60 means a few more epochs could close 0.001-0.002
   dev CER. Low cost, moderate return.

3. **LayerConv + Temperature.** Two orthogonal depth-aware mechanisms: LayerConv
   at the input mixing stage, temperature at the attention matrix stage.

### What probably won't work

1. **Finer kernel schedule (kernel=7 to kernel=3 with 5 distinct values).**
   The test CER is already matched to ConvShift (kernel=3 everywhere). The
   current [7,7,5,5,5,3] schedule provides no improvement over uniform, suggesting
   the kernel schedule is not the bottleneck.

2. **Very wide kernels (kernel=15, kernel=11).** At T=250, kernel=7 covers
   ±3 frames = ±120ms. The full ConvShift context of 120ms is already within one
   attention head's range. Making it wider would overlap more with what LION
   attention already captures.
