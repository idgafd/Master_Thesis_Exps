# Formal v1 — Results

## Dataset
- **Train:** LibriSpeech train-clean-100 (100h, ~28,539 utterances)
- **Dev:** LibriSpeech dev-clean
- **Test:** LibriSpeech test-clean

## Training Configuration
| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| n_layers | 6 |
| n_heads / head_size | 4 / 64 |
| FFN dim | 896 |
| Dropout | 0.1 |
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Scheduler | Cosine + 1000-step warmup |
| Epochs | 80 (patience=15) |
| SpecAugment | LD policy (freq=27, time=100) |
| Batch | max 300s total duration |
| Grad clip | 5.0 |

---

## Model Parameter Counts

| Backbone | Total Params | Frontend | Encoder | CTC Head | vs LION |
|----------|-------------|----------|---------|----------|---------|
| transformer | — | — | — | — | — |
| rwkv6 | — | — | — | — | — |
| mamba | — | — | — | — | — |
| lion | — | — | — | — | ref |
| lion_convshift | — | — | — | — | — |
| lion_lucid | — | — | — | — | — |
| lion_delta | — | — | — | — | — |
| lion_headscale | — | — | — | — | — |
| rwkv6_lucid | — | — | — | — | — |
| rwkv6_delta | — | — | — | — | — |
| mamba_bidir | — | — | — | — | — |

---

## Table 1: Core Baselines (Full-Utterance Evaluation)

| # | Backbone | Dev CER | Dev WER | Test CER | Test WER | Best Epoch |
|---|----------|---------|---------|----------|----------|------------|
| 1 | transformer | — | — | — | — | — |
| 2 | rwkv6 | — | — | — | — | — |
| 3 | mamba | — | — | — | — | — |
| 4 | lion | — | — | — | — | — |
| 5 | lion_convshift | — | — | — | — | — |

## Table 2: LION + Mechanism Improvements

| # | Backbone | Dev CER | Dev WER | Test CER | Test WER | Delta vs LION |
|---|----------|---------|---------|----------|----------|---------------|
| 6 | lion_lucid | — | — | — | — | — |
| 7 | lion_lucid_chunked | — | — | — | — | — |
| 8 | lion_delta | — | — | — | — | — |
| 9 | lion_convshift_headscale | — | — | — | — | — |

## Table 3: RWKV-6 (Recurrent) + Mechanisms

| # | Backbone | Dev CER | Dev WER | Test CER | Test WER | Delta vs rwkv6 |
|---|----------|---------|---------|----------|----------|----------------|
| 10 | rwkv6_lucid | — | — | — | — | — |
| 11 | rwkv6_delta | — | — | — | — | — |
| 12 | rwkv6_lucid_delta | — | — | — | — | — |

## Table 4: Bidirectional Mamba

| # | Backbone | Dev CER | Dev WER | Test CER | Test WER | Delta vs mamba |
|---|----------|---------|---------|----------|----------|----------------|
| 13 | mamba_bidir | — | — | — | — | — |

## Table 5: Chunked Evaluation (Dev, Reset Mode)

| Backbone | Full | 2s | 5s | 10s |
|----------|------|-----|-----|------|
| transformer | — | — | — | — |
| rwkv6 | — | — | — | — |
| mamba | — | — | — | — |
| lion | — | — | — | — |
| lion_convshift | — | — | — | — |

## Table 6: Chunked Evaluation (Dev, Carry-State)

| Backbone | Full | 2s | 5s | 10s |
|----------|------|-----|-----|------|
| rwkv6 | — | — | — | — |
| mamba | — | — | — | — |

---

## Table 7: Statistical Validation (Top-5 x 3 seeds)

| Backbone | Seed 42 CER | Seed 123 CER | Seed 777 CER | Mean | Std |
|----------|-------------|--------------|--------------|------|-----|
| — | — | — | — | — | — |

---

## Notes

- All parameter counts should be within 5% of LION (reference).
- CER computed character-by-character. WER computed word-by-word.
- "Delta vs X" = relative change: `(new - base) / base * 100%`
- Tables will be populated as experiments complete.
