# Run 005 Analysis — bidir_linear_attention / bidir_rwkv6

---

## What Changed vs Runs 003–004 (Mamba Baseline)

Note on numbering: runs 003 and 004 (originally numbered 001 and 002 before the rename commit)
were both mamba experiments — mamba-wsd12 and mamba-wsd46 respectively. Run 005 is the first
experiment introducing bidirectional architectures. There is no prior "bidir version" — this is
the origin point for all bidir work.

The most fundamental change in run 005 is the shift from causal (left-to-right) processing to
full bidirectional processing. Mamba, as a state space model, processes each frame knowing only
what came before it — at position t it has seen frames 0 through t-1. Both new architectures
in run 005 process the entire utterance at once, so every output frame can attend to every
input frame in both directions simultaneously. This is the critical qualitative difference:
mamba is online (causal), the new models are offline (acausal, require full utterance).

The second change is the learning rate scheduler. Runs 003 and 004 used WSD (Warmup-Stable-Decay):
a long flat phase followed by a sharp decay. Run 005 switches to cosine annealing for the first
time, which decays smoothly from the peak LR to near-zero over all 60 epochs. This scheduler
choice became the identifier in the run name ("bidir-cosine"). All hyperparameters otherwise
remain identical: d_model=256, n_layers=6, batch_max_seconds=240, 35h training data, 60 epochs.

Because bidirectional models cannot process streaming audio chunk-by-chunk while accumulating
state, carry-state evaluation was disabled for this run (max_carry_eval_utterances=0). Chunked
reset evaluation is still reported — each chunk is processed independently, discarding any state.

---

## Key Observations (Facts)

| Metric | mamba (003) | mamba (004) | bidir_linear_attn | bidir_rwkv6 |
|---|---|---|---|---|
| Params | 7.70M | 7.70M | 7.44M | 7.74M |
| BestDevCER | 0.1971 | 0.1997 | 0.1911 | **0.1676** |
| BestEpoch | — | — | 59 | 59 |
| TestCER | 0.2098 | 0.2125 | 0.2044 | **0.1790** |
| TestWER | 0.7599 | 0.7688 | 0.7673 | **0.6704** |
| R@2s CER | 0.2593 | 0.2629 | 0.2408 | **0.2161** |
| R@5s CER | 0.2200 | 0.2228 | 0.2101 | **0.1857** |
| R@10s CER | 0.2098 | 0.2125 | 0.2043 | **0.1790** |
| C@2s CER | 0.2121 | 0.2150 | — | — |
| C@10s CER | 0.2098 | 0.2125 | — | — |
| Carry works | yes | yes | no | no |

- bidir_rwkv6 is a major jump: CER 0.1790 vs mamba 0.2098 — 14.7% relative improvement.
- bidir_linear_attention gives only a marginal gain over mamba: CER 0.2044 vs 0.2098 (+2.6%).
- Both bidir models peak at epoch 59 — they were still slowly improving at run end, suggesting
  they were not fully converged at 60 epochs under cosine scheduling.
- bidir_rwkv6 train wall time: 5115s (85s/epoch) vs bidir_linear_attn: 3984s (66s/epoch).
  RWKV-6 costs ~30% more compute per epoch but delivers substantially better accuracy.
- Mamba carry state works cleanly: carry CER ≈ reset CER at 10s windows, slightly worse at 2s
  (0.2121 vs 0.2593 — carry is actually 18% better at 2s because accumulated context helps).
- bidir models show larger chunked degradation at short windows: bidir_rwkv6 R@2s 0.2161 vs
  R@10s 0.1790 — a 20.7% increase, compared to mamba 0.2593 vs 0.2098 — a 23.6% increase.
  The bidir models degrade at short windows similarly to mamba (proportionally), but from a
  lower absolute baseline.

---

## Anomalies

1. **bidir_linear_attention barely beats mamba.** Given that LION dual-recurrence has full
   bidirectional access, the gain of only 2.6% relative CER is surprising. Both models hit
   best epoch 59 and dev CER was still declining — the model may have been undertrained.
   Alternatively, the LION linear attention formulation may saturate earlier than RWKV-6.

2. **Both bidir models converge monotonically to epoch 59 with no plateau.** Mamba plateaued
   earlier (WSD decay triggered at epoch ~48). Under cosine, the bidir models never fully
   plateau — dev CER at epoch 55–59 is still declining, especially for bidir_rwkv6 (dev CER
   at epoch 58: 0.1681, epoch 59: 0.1676). This means cosine with 60 epochs may be cutting
   training short.

3. **bidir_rwkv6 dev loss does not plateau the way train loss does.** Train loss at epoch 60:
   0.425 (still dropping). Dev loss epoch 50–59: 0.754→0.751 (slight continued improvement).
   The model is not overfitting — test loss (0.7988) is close to final dev loss. There is still
   signal to extract with more training.

---

## Hypotheses

### Why does bidir_rwkv6 dominate so strongly over bidir_linear_attention?

bidir_linear_attention uses the LION parallel matrix formulation — a linear attention approximation
that replaces the full softmax attention with a kernel-based decomposition. This reduces the full
T×T attention matrix to an approximation, losing some of the precise pairwise token interaction
that softmax attention captures.

bidir_rwkv6 uses the RWKV-6 WKV kernel which, in the parallel (non-recurrent) form used here,
computes a form of exponentially-weighted attention across the full sequence. It is not a linear
approximation — it explicitly computes weighted sums with learned time-decay parameters w,
allowing the model to learn which temporal distances matter most. For ASR, this is critical:
a phoneme at position t needs precise attention to nearby frames for acoustic context but also
to distant frames for linguistic context, and RWKV-6 can learn different decay rates for these.

### Why does mamba carry state work and bidir carry state cannot?

Mamba is fundamentally recurrent: it maintains a hidden state that is updated left-to-right.
At chunk boundaries, carrying this state forward naturally continues the computation as if the
audio stream never stopped. The model was trained on the same left-to-right recurrent dynamics
it uses at inference.

Bidirectional models process whole utterances. They have no meaningful recurrent state to carry,
because the backward direction would require knowing the future — which doesn't exist in streaming.
Chunked reset evaluation approximates this by treating each chunk as an independent utterance;
the degradation at short windows is the cost of discarding cross-chunk context.

### Why switch from WSD to cosine?

WSD forces a sharp final decay, which can cause training to "lock in" a solution prematurely
if the stable phase wasn't long enough to fully explore the loss landscape. With bidirectional
models that have higher capacity, cosine's smooth decay gives the optimizer more gradient signal
throughout training to refine solutions. The monotone dev improvement to epoch 59 validates
this choice — the cosine schedule did not over-decay early.

---

## Next Actions (as of end of run 005)

1. **Train longer or with LR restart:** dev CER still declining at epoch 59. Running 80–100 epochs
   with the same cosine schedule, or a cosine-with-restart at epoch 60, could recover more gain.
   Estimated ceiling based on dev loss trajectory: bidir_rwkv6 CER may reach ~0.165.

2. **Test ConvShift on bidir_rwkv6:** the fixed (x[t-1]+x[t+1])/2 token shift is a rigid prior.
   A learned DWConv1d initialized to reproduce this shift but allowed to adapt could improve
   temporal feature mixing without changing the WKV attention mechanism. → Run 006.

3. **Test full Bi-WKV architecture:** instead of LION (parallel attention matrix) applied
   bidirectionally, test AudioRWKV-style separate forward and backward RWKV-6 blocks per layer,
   combined with a learned gate. This would also enable carry-state inference for the forward path.
   → Run 007.

4. **Evaluate bidir_linear_attention more carefully:** its weak performance (barely beats mamba)
   may be a training artifact, not a fundamental architectural limit. The LION formulation may
   need a different hyperparameter regime (larger d_model, more layers, or different dropout).
