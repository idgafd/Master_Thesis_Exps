# Run 009 Analysis — biwkv6_no_conv_no_gate 6L (capacity ceiling test)

---

## What Changed vs Run 007

Run 007 tested Bi-WKV at n_layers=3 (3 effective bidirectional layers = 6 RWKV-6 blocks,
7.74M params) — parameter-matched to the 6-layer LION. The result was CER 0.2201: worse
than LION but the dev CER plateau at epoch 30 left open the question of whether the model
was architecture-limited or capacity-limited. Run 009 answers that question by doubling
to n_layers=6 (6 bidir layers = 12 RWKV-6 blocks, 13.56M params), matching the parameter
budget of run-008's 12-layer LION. No other hyperparameters changed. Gate and ConvShift
are excluded — run-007 confirmed the gate provides zero benefit, and a clean no-gate
baseline gives the clearest capacity signal.

---

## Key Observations (Facts)

| Metric | biwkv6_no_conv_no_gate 3L (007) | biwkv6_no_conv_no_gate 6L (009) |
|---|---|---|
| Params | 7.74M | **13.56M** (+75%) |
| BestDevCER | 0.2011 | **0.1850** |
| BestEpoch | 55 | **58** (still declining) |
| TestLoss | 0.9123 | **0.8242** |
| TestCER | 0.2201 | **0.2039** |
| TestWER | 0.7788 | **0.7264** |
| R@2s CER | 0.2504 | **0.2346** |
| R@5s CER | 0.2256 | **0.2095** |
| R@10s CER | 0.2201 | **0.2040** |
| C@2s CER | 0.7122 | 0.7593 (still broken) |
| C@10s CER | 0.7137 | 0.7585 (still broken) |
| Train loss ep60 | 0.5612 | **0.5683** |

Doubling depth gives a real improvement: CER 0.2039 vs 0.2201, a 7.4% relative gain.
The plateau that afflicted 3L (stalled at 0.20 from epoch 30 through epoch 60) did not
occur with 6L — the deeper model kept improving throughout training.

Dev CER trajectory at key epochs — 3L vs 6L:

| Epoch | biwkv6-3L dev CER | biwkv6-6L dev CER |
|---|---|---|
| 5 | 0.3134 | 0.3189 |
| 10 | 0.2620 | 0.2633 |
| 20 | 0.2276 | 0.2278 |
| 30 | 0.2160 | **0.2034** |
| 40 | 0.2054 | **0.1935** |
| 50 | 0.2023 | **0.1868** |
| 60 | 0.2015 | **0.1852** |

The two models are nearly identical through epoch 20 — the extra 6M parameters provide
no advantage in early training. At epoch 30 the trajectories diverge: 3L flatlines at
~0.20 while 6L continues declining. This is a clean capacity effect: the 3L model
exhausted its representational ceiling around epoch 25–30; the 6L model had more layers
to refine representations and kept improving.

Epoch time is 67s vs 66s for 3L — essentially identical, again confirming data loading
is the bottleneck (not model compute). VRAM used: ~14.5GB (vs ~8GB for 3L).

### Carry State

The carry state failure from run-007 persists and is slightly worse:

| Mode | biwkv6 3L | biwkv6 6L |
|---|---|---|
| Reset CER | 0.2201 | 0.2039 |
| Carry C@2s CER | 0.7122 | 0.7593 |
| Carry C@10s CER | 0.7137 | 0.7585 |

The carry CER increased from 0.71 to 0.76 even though reset CER improved. More layers =
more backward state corruption propagating through more RWKV-6 blocks. The fundamental
architectural mismatch (backward branch carry state is meaningless at inference) compounds
with depth.

### The Coincidence

| Model | Params | TestCER |
|---|---|---|
| bidir_rwkv6_conv_nogate 12L (run-008) | 13.58M | **0.2029** |
| biwkv6_no_conv_no_gate 6L (run-009) | 13.56M | **0.2039** |

At ~13.57M parameters, LION 12L and BiWKV 6L converge to virtually identical test CER
(Δ = 0.001). This is a striking result: two architecturally very different models
(parallel global attention vs sequential per-layer RNN, trained with different dynamics)
land at the same accuracy with the same parameter budget.

---

## Anomalies

1. **The 3L plateau is broken at 6L, not just delayed.** BiWKV 3L hit a wall at epoch 30
   and barely moved for 30 epochs (ep30: 0.2160 → ep60: 0.2015). BiWKV 6L was still
   declining at epoch 58 (dev improvement ep50→60: 0.0015, marginal but present). The
   deeper model has not hit a ceiling at 60 epochs, whereas the shallower one had essentially
   converged by epoch 40. This suggests the 3L plateau was a capacity ceiling, not a
   training dynamics issue.

2. **BiWKV scales better than LION across runs 007→009.** Doubling BiWKV depth cut CER by
   0.016 (7.4% relative). Doubling LION depth increased CER by 0.027 (15.3% relative worse).
   BiWKV's sequential architecture with shorter gradient paths scales more gracefully with
   depth. LION's parallel dense attention produces more complex gradients that degrade with
   depth.

3. **Carry state gets worse with more depth despite better reset performance.** The carry
   CER increased from 0.7122 to 0.7593 while reset improved from 0.2201 to 0.2039. This
   confirms that carry state failure is not a generalisation issue — it is a structural
   property that gets amplified by more RWKV-6 blocks processing the corrupted backward state.

4. **Train loss for 6L (0.5683) is marginally higher than 3L (0.5612) at epoch 60.** This
   is unexpected — a larger model should fit training data at least as well. The likely
   cause is that the 6L model has not yet fully converged (best_epoch=58, still declining),
   while the 3L model effectively plateaued earlier and the train loss reflects more settled
   parameters.

---

## Hypotheses

### Why does BiWKV scale better than LION with depth?

RWKV-6 blocks in BiWKV process the sequence as a sequential state update: h_t = f(h_{t-1}, x_t).
Gradients flow back through this recurrence in a controlled way — the gating mechanism (similar
to LSTM gates) explicitly regulates how much gradient each step passes backward. This gives
BiWKV better gradient flow across depth than LION.

LION's parallel WKV attention computes O(T²) interactions per layer with no gating of the
gradient path — gradients from every output position flow back to every input position through
the attention matrix, creating dense gradient interactions between layers that compound with
depth. More LION layers = more multiplicative gradient paths = harder optimisation.

RWKV-6 also has a natural inductive bias toward local context (exponential decay w), which
means early layers learn useful local representations quickly without needing long-range
coordination with deeper layers. This makes each RWKV-6 block more independently trainable,
which is why BiWKV 6L converged in early epochs just as fast as BiWKV 3L.

### Why do the two models converge to the same CER at equal parameter count?

At ~13.6M parameters and 35 hours of data, both architectures appear to hit the same
data-complexity ceiling. The Ukrainian Common Voice data has finite phonetic diversity and
a fixed vocabulary of 36 characters. Once a model has enough parameters to represent all
relevant acoustic-phonetic patterns in the training data, adding more parameters or changing
the architecture no longer helps. The coincidence of 0.2029 vs 0.2039 suggests both models
reached the data ceiling, just via different routes: LION got there despite bad gradient
scaling, BiWKV got there despite context-range limitations.

The more meaningful comparison is at the 7.75M budget: LION 6L (0.1760) vs BiWKV 3L (0.2201).
At the smaller scale, LION's architectural advantage is clear. The advantage disappears at
14M — at that scale, the gradient scaling problem for LION and the context-range limitation
for BiWKV both kick in at similar magnitude.

### Why is carry state worse at 6L?

More RWKV-6 backward blocks = more layers processing a state that was never seen during
training. At 3L, the corrupted backward state (from the carry mismatch) propagates through
3 backward blocks. At 6L, it propagates through 6, with each block further transforming
the corrupted state in ways that diverge further from training distribution. The final
fwd+bwd combination at each of the 6 layers receives a progressively more corrupted bwd
input, leading to more insertion-heavy CTC output.

---

## Next Actions

1. **Run eval_hybrid_carry.py on the 6L checkpoint** to confirm whether Option B
   (carry fwd, reset bwd per chunk) similarly rescues carry performance at 6L.
   Expected: hybrid carry CER ~0.26–0.28 (same pattern as 3L where it dropped from 0.71
   to 0.26). If it works, the hybrid carry mode is architecture-agnostic and valid for
   any BiWKV depth.

2. **Run biwkv6-6L for 80–100 epochs.** Best epoch was 58 and still declining at 60.
   Given that 3L plateaued at epoch 40, 6L likely has 10–15 more useful epochs remaining.
   Estimated ceiling based on trajectory: dev CER ~0.175–0.180, test CER ~0.195–0.200.

3. **Consider the data ceiling interpretation seriously.** If both 13.6M models converge
   to CER ~0.204 and the best 7.75M model is at 0.176, the next gain likely requires
   more data (beyond 35h) rather than more parameters or different architecture. Testing
   LION 6L on 70h or 100h of data would separate the data ceiling from the architecture
   ceiling.
