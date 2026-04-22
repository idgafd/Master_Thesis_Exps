# Synthetics v1 — Dataset & Compute Cost Estimate

This document estimates dataset-generation cost, disk footprint, and GPU
compute for the Tier-1 MQAR experiment matrix. Numbers are derived from
first-principles arithmetic; the empirical micro-benchmark in
`scripts/benchmark_generator.py` (TBD) will replace these estimates with
measurements.

---

## 1. Dataset generation: essentially free

MQAR examples are pure-tensor synthesis: `torch.randint` for tokens, a few
masking writes, no I/O. The whole training corpus can be regenerated from a
seed in seconds.

### 1.1 Per-example generation cost

For sequence length `T`, `K` key-value pairs, `Q` queries:

| Step | Cost |
|------|------|
| Sample `2K` random key/value tokens (`torch.randint`) | `O(K)` |
| Sample `Q` query positions and `Q` query tokens | `O(Q)` |
| Build input sequence (vectorised scatter) | `O(T)` |
| Build target sequence (`-100` mask + query targets) | `O(T)` |
| **Total per example** | **`O(T)`** |

On a single CPU core, vectorised `torch.randint` produces ~5×10⁸ int32/sec.
Per-example cost at `T=512` is dominated by the two `O(T)` writes (~10 µs),
giving ~10⁵ examples/sec/core. With NumPy vectorised batch generation
(generate 1024 examples at once and reshape), throughput is **~10⁶ examples/sec**.

### 1.2 Wall-clock for full dataset materialisation

(Single CPU core, vectorised batched generation.)

| Example count | T=64 | T=256 | T=512 | T=1024 | T=2048 |
|--------------:|-----:|------:|------:|-------:|-------:|
| 10 000        | <0.1 s | <0.1 s | 0.1 s | 0.2 s | 0.4 s |
| 100 000       | 0.2 s | 0.5 s | 1 s | 2 s | 4 s |
| 1 000 000     | 2 s | 5 s | 10 s | 20 s | 40 s |

**Conclusion:** generation is so cheap that pre-materialising entire epochs
makes no sense. Generate on-the-fly per epoch, fix a seed for
reproducibility. Pre-materialise only the eval set for stable comparisons.

---

## 2. Disk footprint (only matters if you pre-materialise)

Per-example storage: `T × 2 bytes` input (uint16, vocab ≤ 65k) + `T × 2 bytes`
target (int16 with `-100` sentinel, or sparse representation) = **4·T bytes**.

| Example count | T=64 | T=256 | T=512 | T=1024 | T=2048 |
|--------------:|-----:|------:|------:|-------:|-------:|
| 10 000  (eval set) | 2.6 MB | 10 MB | 20 MB | 41 MB | 82 MB |
| 100 000 (1 epoch dump) | 26 MB | 102 MB | 205 MB | 410 MB | 820 MB |
| 1 000 000 (full pre-mat) | 256 MB | 1.0 GB | 2.0 GB | 4.1 GB | 8.2 GB |

**Recommendation:** materialise eval sets only (a few hundred MB total across
all length settings). Treat training data as ephemeral.

---

## 3. GPU compute: dominates total cost

Compute is set by training-step count × per-step latency. Both depend on
the backbone, sequence length, and batch size. Numbers below assume the
RTX PRO 6000 Blackwell (~150 TFLOPs FP16, 96 GB) at ~50 % effective
utilisation, and the formal_v1 model size (`d_model=256`, `n_layers=6`,
`ffn=896`, ~7 M params).

### 3.1 Memory-driven max batch size (training, FP16, with gradients)

Activation memory dominates at long T. For Transformer / LION the
`O(T²)` attention map is the binding term.

| Backbone | T=64 | T=256 | T=512 | T=1024 | T=2048 |
|----------|-----:|------:|------:|-------:|-------:|
| Transformer / LION | 512 | 256 | 128 | 32 | 8 |
| RWKV-6 (recurrent) | 512 | 512 | 256 | 128 | 64 |
| Mamba / Mamba-2 | 512 | 512 | 256 | 128 | 64 |

(Estimates; will be measured at smoke-test time. Use these as ceilings
for the YAML config; production runs use a smaller batch with
gradient accumulation if needed for fairness.)

### 3.2 Per-step latency (forward + backward, ms)

Rough bottom-up estimate. Real numbers from `scripts/debug_run.py` will replace.

| Backbone | T=64 | T=256 | T=512 | T=1024 | T=2048 |
|----------|-----:|------:|------:|-------:|-------:|
| Transformer | 1.5 | 4 | 12 | 35 | 120 |
| LION | 1.5 | 4 | 12 | 35 | 120 |
| RWKV-6 (recurrent) | 2 | 5 | 9 | 18 | 35 |
| Mamba (PyTorch) | 4 | 12 | 25 | 50 | 100 |
| Mamba-2 (PyTorch) | 3 | 9 | 18 | 35 | 70 |
| Mamba (CUDA backend) | 1 | 3 | 6 | 12 | 24 |

### 3.3 Run length

The Zoology paper trains MQAR for ~10⁴–10⁵ steps depending on difficulty.
For a reproduction at our model size we plan **≤ 50 000 steps** with
early stopping on per-query recall plateau (patience 5×1000 steps).

### 3.4 Wall-clock per run (single backbone × single seq_len × single seed)

Using mid-range estimates (~10 ms/step Transformer, ~7 ms/step RWKV-6,
~15 ms/step Mamba PyTorch) and 30 000 steps as the typical convergence point:

| Backbone | T=64 | T=256 | T=512 | T=1024 | T=2048 |
|----------|-----:|------:|------:|-------:|-------:|
| Transformer | 1 min | 2 min | 6 min | 18 min | 60 min |
| LION | 1 min | 2 min | 6 min | 18 min | 60 min |
| RWKV-6 | 1 min | 3 min | 5 min | 9 min | 18 min |
| Mamba | 2 min | 6 min | 13 min | 25 min | 50 min |
| Mamba-2 | 2 min | 5 min | 9 min | 18 min | 35 min |

### 3.5 Full Stage-1 matrix budget

**Stage-1 cohort** (8 backbones, focused subset — see `PLAN.md`):
`transformer_causal`, `rwkv6`, `rwkv6_lucid`, `rwkv6_delta`, `rwkv6_lucid_delta`,
`mamba`, `mamba2`, `transformer` (bidirectional reference, optional).

**Length sweep:** T ∈ {64, 128, 256, 512}. (Skip T≥1024 for Stage 1; defer to
Stage 2 length-extrapolation experiments.)

**Seeds:** 1 seed for the initial sweep; 3 seeds for any backbone × length
that lands a BREAK or MARGINAL result (matches formal_v1 multi-seed protocol).

**Estimate** (sum across the 8 × 4 = 32 single-seed runs at average ~6 min):

| Phase | Runs | Avg time | Total |
|-------|----:|--------:|------:|
| Single-seed sweep | 32 | 6 min | **3.2 GPU-hours** |
| Multi-seed expansion (assume 6 of 32 results need 3-seed confirmation) | 12 extra | 6 min | 1.2 GPU-hours |
| **Stage-1 total** | | | **~4.5 GPU-hours** |

Roughly half a day on a single RTX PRO 6000. Comfortably fits within the
formal_v1 compute envelope — no scheduling pressure on the main ASR runs.

### 3.6 If we extend to long T (Stage-2 extrapolation)

Adding T ∈ {1024, 2048} to the cohort multiplies per-run cost ~3-10×.
A separate length-extrapolation slice (best 4 backbones × 2 long lengths ×
3 seeds = 24 runs × 30 min avg) adds **~12 GPU-hours** — still tractable.

---

## 4. Comparison with formal_v1 spine

For reference, the formal_v1 spine is 30 epochs × ~82 sec/epoch =
**~40 min per run** at T≈500 (LibriSpeech). Stage-1 MQAR at the same length
is **~6 min per run**: ~7× cheaper because (a) no audio I/O, (b) no
SpecAugment, (c) shorter convergence. The synthetics tier is intentionally
cheap so it can be re-run frequently as new mechanisms ship in formal_v1.

---

## 5. To-be-measured (replaces estimates)

After the smoke-test in `scripts/debug_run.py` lands, we will replace
sections 3.1–3.4 with measured numbers:
- Actual per-step latency for each backbone × seq_len combination
- Actual peak GPU memory and the resulting safe batch size
- Actual convergence step count (where per-query recall plateaus)

The single-source-of-truth for these measurements will live in
`outputs/_benchmark/` and feed back into this document.
