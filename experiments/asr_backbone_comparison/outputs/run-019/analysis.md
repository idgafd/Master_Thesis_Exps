# Run 019 Analysis — Plain RWKV-6 / RWKV-7, 100 Epochs, Cosine

This run revisits the **plain unidirectional** RWKV encoders with the default
ASR recipe: default SpecAugment, 6 layers, d=256, and a cosine scheduler. The
important constraints were:

- train each plain encoder separately
- extend training to 100 epochs
- keep SpecAugment at the default setting
- avoid bidirectional RWKV-6 variants

---

## Setup

| Parameter | Value |
|-----------|-------|
| Backbones | `rwkv6`, `rwkv7` |
| Epochs | 100 |
| Scheduler | cosine + warmup (500 steps) |
| Dropout | 0.15 |
| SpecAugment | default (freq=15, time=35, 2+2 masks) |
| d_model | 256, 6 layers, head_size=64 |
| Params | 7.74M (`rwkv6`), 7.14M (`rwkv7`) |

---

## Results

| Backbone | Best Ep | Best Dev CER | Test Loss | Test CER | Test WER |
|----------|---------|--------------|-----------|----------|----------|
| `rwkv6` | 87/100 | 0.2163 | 0.9888 | 0.2371 | 0.8017 |
| `rwkv7` | 83/100 | 0.3822 | 1.4913 | 0.4000 | 0.9808 |

Chunked results:

| Backbone | Full CER | R@2s | R@5s | R@10s | C@2s | C@5s | C@10s |
|----------|----------|------|------|-------|------|------|-------|
| `rwkv6` | 0.2371 | 0.2962 | 0.2476 | 0.2371 | 0.2813 | 0.2796 | 0.2787 |
| `rwkv7` | 0.4000 | 0.4179 | 0.4028 | 0.3998 | 0.4450 | 0.4423 | 0.4409 |

Carry-state delta (reset − carry):

| Backbone | Δ@2s | Δ@5s | Δ@10s |
|----------|------|------|-------|
| `rwkv6` | +0.0149 | -0.0320 | -0.0416 |
| `rwkv7` | -0.0270 | -0.0395 | -0.0411 |

---

## RWKV-6: Better Dev, Same Test

The main finding is straightforward:

1. `rwkv6` is still the better plain recurrent model.
2. Extending training to 100 epochs improves **dev** meaningfully.
3. The same extension does **not** improve **test** meaningfully.

Within this 100-epoch run:

- dev CER at epoch 60: **0.2286**
- best dev CER at epoch 87: **0.2163**
- final test CER: **0.2371**

Compared with the earlier plain-RWKV6 result you already knew about
(~**0.2355** test CER at 60 epochs), the 100-epoch rerun is effectively the
same on test. So the bottleneck is probably **not** “the model simply needed
more optimization.”

The cleaner interpretation is:

- extra epochs let the model fit the train/dev distribution better
- the test distribution does not move with it
- this points to a **generalisation ceiling**, likely driven by limited data
  and/or a dev-test mismatch inside Common Voice Ukrainian 24.0

So yes: **data ceiling is the first explanation to take seriously**. More
precisely, this is a **generalisation ceiling on the current dataset/splits**,
not a pure optimisation ceiling.

---

## RWKV-7: Longer Training Does Not Rescue Stock RWKV-7

Stock `rwkv7` remains poor even with 100 epochs:

- best dev CER: **0.3822**
- test CER: **0.4000**
- carry-state hurts at all tested chunk sizes

For comparison:

| Model | Test CER |
|-------|----------|
| `rwkv7` (this run, stock, 100ep cosine) | 0.4000 |
| `rwkv7_fix_decay` (run 018) | 0.3776 |
| `rwkv7_fix_all` (run 018) | 0.2602 |

This means **training longer is not a substitute for the RWKV-7 init fixes**.
The stock model is not just “undertrained”; it starts in a bad regime and
stays there.

---

## Short Answer: Why RWKV-7 Fixes Matter at This Scale

The short version is that the stock RWKV-7 defaults encode assumptions from a
larger-model / autoregressive regime, and those assumptions are harmful in a
6-layer, 256-dim, 35-hour ASR setting.

1. `v_first` is too strong for a shallow CTC encoder.
It keeps injecting layer-0 value features into deeper layers, which suppresses
the hierarchical acoustic abstraction ASR needs.

2. `k_a` weakens the attention path at init.
The stock key scaling effectively halves the key magnitude early on, which is a
bad trade on a small model that already has limited learning signal.

3. The stock RWKV-7 decay initialisation is too narrow.
That makes the recurrent memory too short at the start, so the model behaves as
if it barely remembers context.

4. Even after fixing the bad init, RWKV-7 is still unidirectional.
So the fixes help RWKV-7 converge, but they do **not** remove the fundamental
offline-ASR disadvantage versus stronger bidirectional RWKV-6 variants.

So the fixes are **applicable as RWKV-7-specific repairs**, but they are **not**
general “small-scale performance boosters.” Without them stock RWKV-7 degrades;
with them it becomes usable, but still not a best-model candidate for this
dataset.

---

## How To Choose the Best Model For Now

The decision rule is now clearer:

1. **Best offline model overall:** keep `bidir_rwkv6_conv_nogate` (run 006,
   test CER **0.1760**) as the practical best model.
2. **Best plain recurrent encoder:** choose `rwkv6`, not `rwkv7`.
3. **Do not use stock `rwkv7` anymore.** If RWKV-7 is revisited, start from
   `rwkv7_fix_all`, not from the vanilla backbone.

For plain `rwkv6`, the 100-epoch run is still useful because it resolves the
question “was 60 epochs simply too short?” The answer is: **no, not in a way
that matters for test CER.** It helps dev, but it does not change the current
test ceiling.
