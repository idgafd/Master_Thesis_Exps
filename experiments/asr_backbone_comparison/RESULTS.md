# Experiment Results — Full Metrics

All experiments: CTC-based ASR on Common Voice Ukrainian (35h train / 5h val / 5h test, 16 kHz, 80-mel).
Chunked evaluation: audio split into fixed-length chunks, decoded independently (reset-state) or with hidden state carried across chunks (carry-state).

---

## Table 1 — Full-Utterance Metrics

Frontend is identical across all runs: **1.90M params** (2-layer CNN + BatchNorm, 80-mel → 256-dim).
Encoder = backbone only (RWKV / Mamba / attention blocks). Total = Frontend + Encoder + CTC head (~10K).
Best Ep = best checkpoint epoch / total epochs scheduled.

| Run | Backbone | Encoder | Total | Best Ep | Best Dev CER | Test Loss | Test CER | Test WER | Notes |
|-----|----------|---------|-------|---------|-------------|-----------|----------|----------|-------|
| 003 | `mamba` | 5.79M | 7.70M | 60/60 | 0.1971 | 1.2556 | 0.2098 | 0.7599 | WSD decay @ ep 12 |
| 004 | `mamba` | 5.79M | 7.70M | 58/60 | 0.1997 | 1.2300 | 0.2125 | 0.7688 | WSD decay @ ep 46 |
| 005 | `bidir_linear_attention` | 5.52M | 7.44M | 59/60 | 0.1911 | 0.9011 | 0.2044 | 0.7673 | LION-D recurrence |
| 005 | `bidir_rwkv6` | 5.82M | 7.74M | 59/60 | 0.1676 | 0.7988 | 0.1790 | 0.6704 | **LION baseline** |
| 006 | `bidir_rwkv6_conv_nogate` | 5.83M | 7.75M | 55/60 | 0.1587 | 0.7585 | **0.1760** | **0.6563** | LION + ConvShift, no gate |
| 006 | `bidir_rwkv6_conv` | 6.23M | 8.14M | 57/60 | 0.1635 | 0.7785 | 0.1813 | 0.6845 | LION + ConvShift + gate |
| 007 | `biwkv6_no_conv_no_gate` | 5.83M | 7.74M | 55/60 | 0.2011 | 0.9123 | 0.2201 | 0.7788 | BiWKV6 serial bidir |
| 007 | `biwkv6` | 6.02M | 7.94M | 54/60 | 0.2006 | 0.9340 | 0.2211 | 0.7830 | BiWKV6 + ConvShift + gate |
| 008 | `bidir_rwkv6_conv_nogate` | 11.66M | 13.58M | 97/100 | 0.1624 | 0.8746 | 0.1816 | 0.6623 | LION 12-layer, 100 ep |
| 009 | `biwkv6_no_conv_no_gate` | 11.65M | 13.56M | 80/100 | 0.1713 | 0.8185 | 0.1894 | 0.6774 | BiWKV6 6-layer, 100 ep |
| 010 | `bidir_rwkv6_cplx_b` | 5.82M | 7.74M | 60/60 | 0.1932 | 0.9163 | 0.2140 | 0.7312 | Complex decay θ=0.31 |
| 010 | `bidir_rwkv6_cplx_c` | 5.82M | 7.74M | 60/60 | 0.2109 | 0.9744 | 0.2322 | 0.7625 | Complex decay θ=0.90 |
| 011 | `bidir_rwkv6_cplx_d` | 5.82M | 7.74M | 60/60 | 0.1909 | 0.9002 | 0.2107 | 0.7232 | Learnable θ per layer (+6 params) |
| 011 | `bidir_rwkv6_cplx_b_cos2` | 5.82M | 7.74M | 60/60 | 0.1757 | 0.8546 | 0.1955 | 0.7181 | cos² mask, θ=0.31 |
| 012 | `bidir_rwkv6_cplx_d_cos2` | 5.82M | 7.74M | 60/60 | 0.1782 | 0.8558 | 0.1977 | 0.7212 | Learnable θ + cos² mask (+6 params) |
| 012 | `bidir_rwkv6_headscale` | 5.82M | 7.74M | 60/60 | 0.1660 | 0.8082 | 0.1839 | 0.6874 | Per-head decay bias (+24 params) |
| 013 | `bidir_rwkv6_gaussian` | 5.83M | 7.74M | 60/60 | 0.1693 | 0.8027 | 0.1861 | 0.6796 | Gaussian attn mask (+144 params) |
| 013 | `bidir_rwkv6_dual` | 5.83M | 7.74M | 60/60 | 0.1696 | 0.8220 | 0.1875 | 0.6903 | Dual-decay weighted output (+144 params) |
| 014 | `bidir_rwkv6_layerconv` | 5.84M | 7.75M | 60/60 | 0.1574 | 0.7496 | 0.1768 | 0.6548 | Layer-dep ConvShift k=[7,7,5,5,5,3] |
| 015 | `bidir_rwkv6_temperature` | 5.82M | 7.74M | 60/60 | 0.1606 | 0.7900 | 0.1792 | 0.6681 | Per-head learnable τ (+24 params) |
| 016 | `bidir_rwkv6` | 5.82M | 7.74M | 76/80 | 0.2546 | 1.1431 | 0.2779 | 0.8733 | Heavy reg: dropout=0.25, freq=27, time=70×5 masks, 80 ep |
| 016 | `bidir_rwkv6_temperature` | 5.82M | 7.74M | 75/80 | 0.2649 | 1.1899 | 0.2874 | 0.8924 | Heavy reg: dropout=0.25, freq=27, time=70×5 masks, 80 ep |
| 017 | `bidir_rwkv6_conv_nogate` | 5.83M | 7.75M | 74/80 | 0.2977 | 1.2390 | 0.3189 | 0.9132 | Heavy reg: dropout=0.25, freq=27, time=70×5 masks, 80 ep |
| 017 | `bidir_rwkv6_layerconv` | 5.84M | 7.75M | 74/80 | 0.2978 | 1.2341 | 0.3205 | 0.9020 | Heavy reg: dropout=0.25, freq=27, time=70×5 masks, 80 ep |
| 018 | `rwkv7_fix_decay` | 5.22M | 7.14M | 60/60 | 0.3570 | 1.4094 | 0.3776 | 0.9734 | RWKV-7 decay init fix only; standard reg restored (dropout=0.15, freq=15, time=35×2, WSD sched) |
| 018 | `rwkv7_fix_all` | 5.22M | 7.14M | 60/60 | 0.2388 | 1.0828 | 0.2602 | 0.8597 | RWKV-7 decay + v_first + k_a fixes; standard reg restored (dropout=0.15, freq=15, time=35×2, WSD sched) |
| 019 | `rwkv6` | 5.83M | 7.74M | 87/100 | 0.2163 | 0.9888 | 0.2371 | 0.8017 | Plain RWKV-6 encoder, cosine, 100 ep, default SpecAugment |
| 019 | `rwkv7` | 5.22M | 7.14M | 83/100 | 0.3822 | 1.4913 | 0.4000 | 0.9808 | Plain stock RWKV-7 encoder, cosine, 100 ep, default SpecAugment |

---

## Table 2 — Reset-State Chunked CER

Audio split into chunks decoded independently (no cross-chunk memory). Degradation vs full-utterance shows sensitivity to context truncation.

| Run | Backbone | Full CER | R@2s CER | R@5s CER | R@10s CER |
|-----|----------|----------|----------|----------|-----------|
| 003 | `mamba` | 0.2098 | 0.2593 | 0.2200 | 0.2098 |
| 004 | `mamba` | 0.2125 | 0.2629 | 0.2228 | 0.2125 |
| 005 | `bidir_linear_attention` | 0.2044 | 0.2408 | 0.2101 | 0.2043 |
| 005 | `bidir_rwkv6` | 0.1790 | 0.2161 | 0.1857 | 0.1790 |
| 006 | `bidir_rwkv6_conv_nogate` | **0.1760** | **0.2084** | **0.1804** | **0.1744** |
| 006 | `bidir_rwkv6_conv` | 0.1813 | 0.2123 | 0.1848 | 0.1791 |
| 007 | `biwkv6_no_conv_no_gate` | 0.2201 | 0.2504 | 0.2256 | 0.2201 |
| 007 | `biwkv6` | 0.2211 | 0.2504 | 0.2261 | 0.2211 |
| 008 | `bidir_rwkv6_conv_nogate` (12L) | 0.1816 | 0.2126 | 0.1845 | 0.1790 |
| 009 | `biwkv6_no_conv_no_gate` (6L/100ep) | 0.1894 | 0.2213 | 0.1950 | 0.1895 |
| 010 | `bidir_rwkv6_cplx_b` | 0.2140 | 0.2388 | 0.2161 | 0.2114 |
| 010 | `bidir_rwkv6_cplx_c` | 0.2322 | 0.2523 | 0.2322 | 0.2277 |
| 011 | `bidir_rwkv6_cplx_d` | 0.2107 | 0.2349 | 0.2128 | 0.2081 |
| 011 | `bidir_rwkv6_cplx_b_cos2` | 0.1955 | 0.2272 | 0.1991 | 0.1938 |
| 012 | `bidir_rwkv6_cplx_d_cos2` | 0.1977 | 0.2276 | 0.2010 | 0.1961 |
| 012 | `bidir_rwkv6_headscale` | 0.1839 | 0.2160 | 0.1879 | 0.1820 |
| 013 | `bidir_rwkv6_gaussian` | 0.1861 | 0.2132 | 0.1895 | 0.1846 |
| 013 | `bidir_rwkv6_dual` | 0.1875 | 0.2184 | 0.1899 | 0.1841 |
| 014 | `bidir_rwkv6_layerconv` | 0.1768 | 0.2077 | 0.1800 | 0.1739 |
| 015 | `bidir_rwkv6_temperature` | 0.1792 | 0.2127 | 0.1841 | 0.1781 |
| 016 | `bidir_rwkv6` (strong reg) | 0.2779 | 0.3041 | 0.2783 | 0.2735 |
| 016 | `bidir_rwkv6_temperature` (strong reg) | 0.2874 | 0.3172 | 0.2914 | 0.2858 |
| 017 | `bidir_rwkv6_conv_nogate` (strong reg) | 0.3189 | 0.3408 | 0.3192 | 0.3154 |
| 017 | `bidir_rwkv6_layerconv` (strong reg) | 0.3205 | 0.3449 | 0.3221 | 0.3170 |
| 018 | `rwkv7_fix_decay` | 0.3776 | 0.3983 | 0.3808 | 0.3774 |
| 018 | `rwkv7_fix_all` | 0.2602 | 0.2944 | 0.2655 | 0.2594 |
| 019 | `rwkv6` | 0.2371 | 0.2962 | 0.2476 | 0.2371 |
| 019 | `rwkv7` | 0.4000 | 0.4179 | 0.4028 | 0.3998 |

---

## Table 3 — Carry-State Chunked CER

Hidden state carried across chunk boundaries (recurrent/SSM models only). Lower carry CER = better use of cross-chunk context.

| Run | Backbone | Full CER | C@2s CER | C@5s CER | C@10s CER |
|-----|----------|----------|----------|----------|-----------|
| 003 | `mamba` | 0.2098 | 0.2121 | 0.2102 | 0.2098 |
| 004 | `mamba` | 0.2125 | 0.2150 | 0.2128 | 0.2125 |
| 007 | `biwkv6_no_conv_no_gate` | 0.2201 | 0.7122 | 0.7111 | 0.7137 |
| 007 | `biwkv6` | 0.2211 | 0.7399 | 0.7398 | 0.7400 |
| 009 | `biwkv6_no_conv_no_gate` (6L/100ep) | 0.1894 | 0.7733 | 0.7705 | 0.7695 |
| 018 | `rwkv7_fix_decay` | 0.3776 | 0.3833 | 0.3782 | 0.3774 |
| 018 | `rwkv7_fix_all` | 0.2602 | 0.2629 | 0.2600 | 0.2594 |
| 019 | `rwkv6` | 0.2371 | 0.2813 | 0.2796 | 0.2787 |
| 019 | `rwkv7` | 0.4000 | 0.4450 | 0.4423 | 0.4409 |

> **Note**: Bidirectional models (LION and variants) are stateless by design — no carry-state eval applies. BiWKV6 carry-state is degraded (CER >0.7) because the 6-layer serial architecture with untrained carry-state init performs poorly; this was a known issue at time of run-007 evaluation (state init not optimised). Mamba carry-state is near full-utterance CER, confirming its streaming capability.

---

## Table 4 — Carry-State vs Reset-State Δ CER (Reset − Carry)

Positive = carry-state is better than reset; negative = carry-state hurts.

| Run | Backbone | Δ@2s | Δ@5s | Δ@10s |
|-----|----------|------|------|-------|
| 003 | `mamba` | +0.0472 | +0.0098 | 0.0000 |
| 004 | `mamba` | +0.0479 | +0.0100 | 0.0000 |
| 007 | `biwkv6_no_conv_no_gate` | −0.4618 | −0.4855 | −0.4936 |
| 007 | `biwkv6` | −0.4895 | −0.5137 | −0.5189 |
| 009 | `biwkv6_no_conv_no_gate` (6L/100ep) | −0.5520 | −0.5755 | −0.5800 |
| 018 | `rwkv7_fix_decay` | +0.0149 | +0.0026 | 0.0000 |
| 018 | `rwkv7_fix_all` | +0.0315 | +0.0054 | 0.0000 |
| 019 | `rwkv6` | +0.0149 | −0.0320 | −0.0416 |
| 019 | `rwkv7` | −0.0270 | −0.0395 | −0.0411 |

---

## Summary — Best CER per Model Family

| Architecture family | Best backbone | Test CER | Run |
|--------------------|--------------|----------|-----|
| LION + LayerConv | `bidir_rwkv6_layerconv` | 0.1768 | 014 |
| LION + ConvShift | `bidir_rwkv6_conv_nogate` | **0.1760** | 006 |
| LION + Temperature | `bidir_rwkv6_temperature` | 0.1792 | 015 |
| LION (baseline) | `bidir_rwkv6` | 0.1790 | 005 |
| LION + head mod | `bidir_rwkv6_headscale` | 0.1839 | 012 |
| LION + spatial mod | `bidir_rwkv6_gaussian` | 0.1861 | 013 |
| LION 12-layer | `bidir_rwkv6_conv_nogate` | 0.1816 | 008 |
| Cos² mask | `bidir_rwkv6_cplx_b_cos2` | 0.1955 | 011 |
| BiWKV6 6L/100ep | `biwkv6_no_conv_no_gate` | 0.1894 | 009 |
| Bidir linear attn | `bidir_linear_attention` | 0.2044 | 005 |
| Mamba (best) | `mamba` (WSD-12) | 0.2098 | 003 |
| Complex decay | `bidir_rwkv6_cplx_b` | 0.2140 | 010 |
| RWKV-6 (plain recurrent) | `rwkv6` | 0.2371 | 019 |
| RWKV-7 (fixed) | `rwkv7_fix_all` | 0.2602 | 018 |
| RWKV-7 (stock) | `rwkv7` | 0.4000 | 019 |
