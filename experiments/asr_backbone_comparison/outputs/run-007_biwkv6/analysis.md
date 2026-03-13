# Run 007 Analysis — biwkv6_no_conv_no_gate / biwkv6

---

## What Changed vs Run 005

| | Run 005 | Run 007 |
|---|---|---|
| Backbones | bidir_rwkv6 (LION baseline) | biwkv6_no_conv_no_gate, biwkv6 |
| Architecture | LION: O(T²K) parallel full attention | Bi-WKV: sequential fwd RWKV-6 + bwd RWKV-6, combined per layer |
| Combination | — | nogate: 0.5*(fwd+bwd) / full: G=σ(W_gate(xres))*(fwd)+(1-G)*(bwd) |
| n_layers | 6 | 3 bidir (= 6 RWKV-6 blocks total) |
| Params | 7.74M | 7.74M (nogate) / 7.94M (full) |
| supports_carry_state | no | **yes** (forward-only recurrent path) |
| Scheduler | cosine 60ep | cosine 60ep (same) |

---

## Key Observations (Facts)

| Metric | bidir_rwkv6 (005) | biwkv6_no_conv_no_gate | biwkv6 |
|---|---|---|---|
| Params | 7.74M | 7.74M | 7.94M |
| BestDevCER | 0.1676 | 0.2011 | 0.2006 |
| BestEpoch | 59 | 55 | 54 |
| TestLoss | 0.7988 | 0.9123 | 0.9340 |
| TestCER | 0.1790 | 0.2201 | 0.2211 |
| TestWER | 0.6704 | 0.7788 | 0.7830 |
| R@2s CER | 0.2161 | 0.2504 | 0.2504 |
| R@5s CER | 0.1857 | 0.2256 | 0.2261 |
| R@10s CER | 0.1790 | 0.2201 | 0.2211 |
| **C@2s CER** | — | **0.7122** | **0.7399** |
| **C@5s CER** | — | **0.7111** | **0.7398** |
| **C@10s CER** | — | **0.7137** | **0.7400** |
| **C@2s WER** | — | **1.1613** | **1.2255** |
| **C@10s WER** | — | **1.2053** | **1.2375** |

- Both BiWKV variants are significantly worse than LION: +23% relative CER (0.220 vs 0.179).
- ConvShift+gate in Bi-WKV provides **zero measurable benefit**: 0.2211 vs 0.2201 (noise-level).
- Dev CER plateau: BiWKV stalls at ~0.201 after epoch 30, dev loss range ep30–60: 0.833–0.850.
  LION kept declining monotonically to epoch 59 (0.1676). BiWKV optimization stalls ~2× earlier.
- Best epochs 54/55 are earlier than LION's 59 — capacity saturated, not schedule-limited.

### Carry State — Critical Results

| Model | Reset CER | Carry CER | Δ (R−C, positive = carry worse) |
|---|---|---|---|
| mamba run-003 (reference) | 0.2098 | 0.2098 | ≈ 0.000 (carry works) |
| biwkv6_no_conv_no_gate | 0.2201 | **0.7137** | **−0.4936** |
| biwkv6 | 0.2211 | **0.7400** | **−0.5189** |

- Mamba carry state functions correctly — carry ≈ reset at 10s windows.
- BiWKV carry CER ~0.71 — **3.2× worse than its own reset-mode performance**.
- BiWKV carry WER **exceeds 1.0** (1.16–1.24) — model produces more words than reference,
  indicating extreme hallucination/insertion errors.
- Carry CER is essentially **flat across chunk sizes** (0.7122 @ 2s ≈ 0.7137 @ 10s):
  larger context windows provide zero recovery. The state is uniformly corrupted.
- biwkv6 (with gate) carry is slightly worse than nogate (0.7399 vs 0.7122) — gate makes
  state corruption marginally worse.

---

## Anomalies

1. **Carry CER flat across window sizes.** Normal degradation from carry: larger windows approach
   reset performance as the bad initial state gets diluted by fresh input. Here: 2s and 10s are
   identical (~0.71). This means the corrupted carry state is **actively overriding** fresh input
   at every step — not just a bad initialization that gets forgotten.

2. **WER > 1.0 in carry mode.** WER is bounded at 1.0 for pure deletion errors. Exceeding 1.0
   requires word insertions — the model is hallucinating words not in the reference. This is
   qualitatively different from "bad but bounded" degradation; it indicates the model is
   generating incoherent output streams.

3. **ConvShift+gate provides no benefit in Bi-WKV.** In LION, ConvShift alone gave +1.7% CER
   improvement. In Bi-WKV, the full ConvShift+gate combo provides nothing (0.2201→0.2211,
   slightly worse). The xres-conditioned gate mechanism is inert or harmful in both architectures.

4. **BiWKV dev loss stagnation after epoch 30.** The model enters a flat region early while LION
   continues improving for 60 epochs. This points to a structural optimization ceiling — the
   model exhausts its representational benefit in the first half of training.

5. **Reference carry values context.** A causal unidirectional RWKV-6 in carry mode would be
   expected to produce CER ~0.239/0.236/0.235 (2s/5s/10s) — carry slightly worse at short
   windows, converging at 10s. BiWKV carry at 0.71 is ~3× worse than this expected baseline,
   confirming catastrophic failure rather than expected bidirectional-carry degradation.

---

## Hypotheses

### Why is Bi-WKV worse than LION overall?

LION uses O(T²K) parallel full attention — every output token at position t sees the full
sequence [0..T] through the attention matrix. RWKV-6's WKV mechanism uses an exponentially-
decaying recurrent state that compresses all past context into a fixed D×D matrix. Even with
bidirectional processing, the effective receptive field is limited by the decay parameter w,
which is learned but typically focuses on recent context.

For CTC ASR, long-range dependencies are critical: phoneme disambiguation requires seeing
complete words, speaker-level normalization requires whole-utterance context. LION has a
structural advantage in capturing these. Bi-WKV with 3 bidir layers (6 RWKV-6 blocks) does
not compensate through depth for what it loses in per-layer context range.

### Why is carry state catastrophically broken?

This is an architectural incompatibility between training procedure and inference procedure:

**Training (full utterance, reset state):**
- Forward RWKV-6: processes x[0..T] left-to-right, state summarizes past
- Backward RWKV-6: processes x[T..0] right-to-left (flipped), state summarizes future
- The backward branch's hidden state at position t represents: "all information from [t+1..T]"
- This is valid because the full future exists during training

**Carry inference (chunk-by-chunk, forward-only):**
- Chunk 0 processed → carry fwd state (represents past of chunk 0) ✓
- Chunk 0 also produces a bwd state — but this represents "the reversed content of chunk 0"
- Chunk 1 receives bwd state from chunk 0 and continues backward scan of chunk 1 reversed
- But the bwd branch expects: "a state summarizing the future of chunk 1 [chunk_1_end..T]"
- What it receives: "a state summarizing the past of chunk 0 reversed" → completely wrong

The backward state from carry is **out-of-distribution** for every training example the model
has ever seen. The backward branch was never trained on carry states that represent past context
— only on reset states that represent future context (because training always starts from T going
backward). The result is arbitrary feature activations → hallucinated output.

Why WER > 1.0: the corrupted state injects spurious activation patterns into the WKV output
that decode as phoneme sequences not present in the audio. CTC then segments these into words,
producing insertion-heavy transcripts.

Why flat CER across window sizes: the backward carry state continuously overwrites fresh input
information at every step (the corrupted state has high magnitude), preventing the model from
recovering even with 10 seconds of new audio.

### Why does ConvShift+gate provide nothing in Bi-WKV?

The xres signal (conv_shift(x) - x) is a local difference — how much the 3-frame neighborhood
deviated from the center frame. In LION, this gates a global attention output (wrong pairing
but at least the attention output is rich). In Bi-WKV, the "attention output" being gated is
already a combination of two limited-context recurrent states. Gating two already-noisy
recurrent outputs with a local signal adds noise without adding signal.

The 0.5*(fwd+bwd) combination in nogate is a hard symmetric prior. The gate is meant to learn
a softer content-dependent combination. But if neither fwd nor bwd carries sufficient quality
information, no gate can improve the combination — garbage in, garbage out.

---

## Next Actions

### Carry State Fix (Priority 1)

The backward-carry incompatibility is fundamental and requires a training fix:

**Option A — Chunk-carry training:**
  Train on utterances split into sequential chunks, passing carry state across chunks during
  training. This forces the backward branch to learn what a "real" forward-pass backward state
  looks like — it will learn to treat carry states as past summaries rather than future summaries.
  Cost: doubles training complexity, requires curriculum (start with 2-chunk then longer).

**Option B — Reset backward state at carry inference (zero-cost, current weights):**
  At inference in carry mode: carry only the forward state, reset backward state to zero for
  each new chunk. Backward branch then processes each chunk independently (reset mode),
  forward branch accumulates context. This degrades to asymmetric processing but eliminates
  hallucination. Add `carry_bwd_reset=True` flag to evaluate immediately without retraining.

**Option B is testable right now** with the existing run-007 checkpoints — just change the
inference loop. Recommended first step before any retraining.

### Architecture (Priority 2)

**Scale up BiWKV to confirm capacity ceiling:**
  Run biwkv6 with n_layers=6 (12 RWKV-6 blocks, ~14M params). If dev CER improves past 0.18
  → the n_layers=3 param-matching was too aggressive. If still stalls at ~0.20 → BiWKV has a
  structural ceiling independent of depth.

**Try WSD (warmup-stable-decay) instead of cosine for BiWKV:**
  The dev loss plateau after epoch 30 under cosine (LR already decaying toward 0) may mean
  the model needs a stable training phase to fully exploit its representational capacity.
  WSD with longer stable phase might push past the 0.20 ceiling.

---

## Option B Tested — Hybrid Carry Results

Script: `scripts/eval_hybrid_carry.py`
Method: carry forward state across chunks, reset backward state to zero at each chunk boundary,
combine fwd+bwd as in training. N=500 utterances.

| Mode | biwkv6_no_conv_no_gate CER | biwkv6 CER |
|---|---|---|
| Reset (full utterance) | 0.2201 | 0.2211 |
| Original carry (fwd-only, no bwd) | 0.7137 | 0.7400 |
| **Hybrid carry (fwd carry + bwd reset)** | **0.2585** | **0.2596** |

CER dropped from 0.71 → 0.26 — hallucination is eliminated. The hypothesis was correct:
the original carry failure was because the CTC head received forward-only output, which it
was never trained on. Providing both fwd+bwd (even with bwd reset each chunk) gives the
CTC head a distribution similar to training.

The remaining gap vs reset (0.26 vs 0.22) reflects missing cross-chunk backward context:
the backward branch restarts from zero at every boundary, so it sees at most one chunk of
future context per position. Short windows hurt most: HC@2s = 0.286, HC@10s = 0.259.

Compare to mamba (run-003) carry which actually IMPROVES over reset (carry 0.2098 vs reset 0.2098).
BiWKV hybrid carry is 0.26 — worse than mamba reset (0.21) and worse than its own reset (0.22).
The forward carry brings some context but not enough to compensate for the backward reset cost.

**Conclusion:** Hybrid carry is a valid inference mode (no hallucination), but it does not
make BiWKV competitive with mamba for carry-mode tasks. Training with chunk-carry (Option A)
is the only path to genuine carry-state improvement.

---

### Gate (Priority 3)

The xres-conditioned gate is 0 for 2 runs (LION branch it hurts, BiWKV branch it's inert).
Before any further gate experiments, test **Option B** above and the scalar gate control
(run-006 next actions). If scalar gate also hurts, abandon gate-based approaches entirely.
