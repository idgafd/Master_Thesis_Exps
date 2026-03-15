# Experiment Summary

## Setup

I'm comparing lightweight sequence model architectures (~7M params, 6 layers, d=256) on a CTC-based character recognition task using ~35 hours of Ukrainian speech (Mozilla Common Voice). This is not the final target task — it's a convenient, reproducible benchmark for evaluating architectural ideas. The data is short-utterance, limited-scale, and scripted, which imposes specific constraints on what can be meaningfully tested here. The plan is to move to larger and more diverse datasets once the most promising mechanisms are identified.

The frontend (2-layer CNN, 80-mel to 256-dim) and CTC head are identical — only the encoder backbone changes.

Key data properties that shape the results:
- **Sequences are short** (~250 frames after 4× subsampling, ~5-10s utterances). O(T²) attention is cheap here — there is no efficiency motivation for linear or sparse attention at this length.
- **Data is limited** (35h). Complex architectures tend to overfit. The dev-test CER gap of ~0.02 is consistent across all models — it's a dataset property, not fixable by architecture.
- **CER is in the 0.17–0.25 range.** At this level, a 5% relative improvement is significant.
- **Short utterances only.** Carry-state and streaming behavior cannot be properly evaluated on this data — longer sequences are needed to stress-test recurrent state accumulation.

---

## Part 1: Which Architecture Won (and Why)

### The Architecture Comparison

I started with five backbone families and progressively narrowed down:

**Transformer** — the initial reference (CER 0.199 from the baseline notebook). Solid but nothing special at this scale.

**Mamba (SSM)** — CER 0.210. About 17% worse than the best bidirectional model. Mamba shows signs of **overfitting or too-early convergence**: running 100 epochs gives the same result as 60, and the WSD scheduler ablation (decay at epoch 12 vs 46) had negligible effect. The model fits what it can fit quickly and then stops improving. Mamba is the only architecture where I have a **fully implemented carry-state** inference path, and it works well — at 2-second chunks, carry-state improves CER by ~5% (0.259 → 0.212). Bidirectional Mamba has not been tried yet, so it's premature to write it off. The architecture itself may benefit substantially from bidirectionality and from longer-sequence data where its linear-time recurrence becomes a real advantage.

**Bidirectional Linear Attention** — CER 0.204. Uses ELU+1 feature maps (LION-D style). Decent, but the fixed feature map discards information that learned decay kernels capture.

**Plain RWKV-6 (unidirectional)** — CER 0.237 (100 epochs, cosine scheduler). This is the important unidirectional baseline. Many of the mechanism experiments (Delta Rule, LUCID) are compared against plain RWKV-6 to isolate the effect of the mechanism from the effect of bidirectionality. Note: the training pipeline evolved between early and late runs, so the matched RWKV-6 baseline from the latest experiments gives CER 0.250 at 60 epochs — Delta Rule and LUCID results should be compared against this matched number for fairness (see Part 5).

**LION (Bidirectional RWKV-6, parallel form)** — CER 0.179 baseline (best early result), with the matched re-run baseline giving CER 0.185. LION computes the full T×T attention matrix using learned exponential decay kernels in both causal and anticausal directions, then merges them. At T=250, this is computationally trivial but expressively powerful. Adding **ConvShift** (a learned depthwise convolution for token mixing, see Part 2) improves it to CER 0.176 — the best absolute result in the project. ConvShift is an addition on top of the LION kernel, not part of the core mechanism itself.

**RWKV-7** — CER 0.400 stock, 0.260 after fixes. Did not converge in stock form (see Part 4).

### The Baseline Re-Run Issue

The training pipeline changed between the early experiments and the later ones. To ensure fair comparisons for the Delta Rule and LUCID experiments, I re-ran the baselines (plain RWKV-6 and bidirectional RWKV-6) under identical conditions at 60 epochs with cosine scheduler. The re-run baselines are:
- Plain RWKV-6: CER **0.250** (vs 0.237 at 100 epochs earlier)
- Bidirectional RWKV-6 (LION): CER **0.185** (vs 0.179 earlier)

The best absolute results remain CER 0.176-0.179 (LION with/without ConvShift from the earlier runs), but all mechanism comparisons (Delta Rule, LUCID) use the matched re-run baselines for proper controlled comparison.

### Why Bidirectional RWKV-6 Leads

The bidirectional parallel attention gives LION something the unidirectional models lack: every position sees both past and future context simultaneously. The learned exponential decay means nearby positions get more attention weight, which is a natural inductive bias for sequential data where local context matters most but longer-range dependencies still need to be captured.

### BiWKV6 — VIM-Style Serial Bidirectional (An Engineering Alternative to LION)

There are two fundamentally different ways to make a recurrent architecture bidirectional:

1. **LION (parallel form):** Unfold the recurrence into a full T×T attention matrix, run it in both directions, merge. This gives exact full-sequence bidirectional attention but is O(T²) and not natively streamable.

2. **BiWKV6 (serial form, VIM-style):** Run two independent RWKV-6 passes per layer — one forward, one backward — on the original recurrent kernel, then combine the outputs (simple average or learned gate). This preserves the recurrent structure and is in principle streamable via carry-state.

The serial approach is attractive because it reuses the stock RWKV-6 kernel without modification — it's an engineering-level change (run the same kernel twice, in opposite directions) rather than a kernel-level change. This is the same idea as Vision Mamba (VIM), which runs Mamba bidirectionally for vision tasks.

**The layer-doubling problem.** The catch is that each "bidirectional layer" contains two full RWKV-6 blocks (forward + backward). To keep the parameter count matched with LION (~7.7M), I had to use only 3 bidirectional layers (= 6 RWKV-6 blocks total) instead of 6. This halves the representational depth: the model gets fewer sequential transformations of its features, even though each transformation sees both directions.

**Results at matched parameters (3 bidir layers, 60 epochs):** CER 0.220 — about 23% worse than LION (0.179). The model plateaus early (dev CER stalls around epoch 30) and ConvShift+gate provides zero benefit in this architecture (unlike LION where ConvShift helps).

**Results with doubled parameters (6 bidir layers, 100 epochs):** CER 0.189 — a 14% relative improvement over the 3-layer version. The gap to LION narrows from 0.044 to 0.013. Interestingly, BiWKV6 scales better with depth than LION does on this data (LION 12-layer was worse than LION 6-layer), but it still can't close the gap. The 6-layer BiWKV6 hits a genuine plateau at epoch 80 — dev CER 0.171 held flat for 19 epochs until early stopping. Unlike LION which hits an optimization ceiling (could still improve with more training), BiWKV6 hits a structural ceiling: the exponential decay in each RWKV-6 block limits the effective temporal context per layer, and adding more layers adds representational depth but not temporal span.

**Carry-state is catastrophically broken.** This was the most striking result. BiWKV6 carry-state produces CER ~0.71 — over 3× worse than its own reset-mode performance (0.22). Worse: WER exceeds 1.0, meaning the model hallucinates words not in the reference. The carry CER is flat across all chunk sizes (2s, 5s, 10s), so longer context windows provide zero recovery — the corrupted state actively overrides fresh input.

**Why carry-state fails:** The backward branch is trained to expect an initial state that summarizes the *future* of the sequence (it processes right-to-left). At carry-state inference, the backward branch receives a state from the *previous chunk's backward pass* — which summarizes the reversed past, not the future. This state is completely out-of-distribution for every training example the model has ever seen. The result is arbitrary activations that decode as hallucinated phoneme sequences.

**The hybrid carry fix:** I tested resetting only the backward state at each chunk boundary while carrying the forward state. This drops CER from 0.71 to 0.26 — hallucination is eliminated. But 0.26 is still worse than both the model's own reset mode (0.22) and Mamba's carry mode (0.21). The fix works but doesn't make BiWKV6 competitive for streaming.

**Is BiWKV6 worth pursuing?** On this short-utterance data, no — LION is strictly better at matched parameters, and BiWKV6's carry-state advantage (the whole reason to use the serial form) doesn't materialize without carry-state training. However, the approach could become relevant under specific conditions:
- **If streaming is a hard requirement** and LION's O(T²) is too expensive, BiWKV6 with proper chunk-carry training (training on segmented utterances with carry-state passed between chunks) could potentially learn useful backward-state representations. This hasn't been tested.
- **On much longer sequences** where T² becomes a real bottleneck, the serial O(T) form would have a practical speed advantage. But this needs to be validated on longer data first.
- **The diagonal duplication issue** (each layer uses two identical RWKV-6 blocks with separate weights) means the parameter efficiency is inherently worse than LION. A shared-weight variant or asymmetric forward/backward blocks might help, but I haven't explored this.

For now, BiWKV6 is parked. The serial bidirectional idea is sound in principle, but the engineering cost of fixing carry-state (chunk-carry training) and the parameter inefficiency of layer doubling make it a lower priority than improving the mechanisms that already work (LUCID, Delta Rule) on architectures that already work (RWKV-6, LION).

---

## Part 2: Improving LION's Token Mixing

### ConvShift — Simple, Cheap, Works (CER 0.176)

**The problem:** LION's default token mixing before computing queries/keys/values is a fixed symmetric average: x'(t) = 0.5·x(t-1) + 0.5·x(t+1). This treats past and future identically for every feature dimension.

**The fix:** Replace it with a learned depthwise 1-D convolution (kernel=3, groups=d_model). Initialized to the same [0.5, 0, 0.5] weights, but now trainable. Adds only ~7K parameters.

**Why it works:** Different feature dimensions benefit from different temporal smoothing. The learned per-channel convolution captures asymmetry in temporal context that the fixed average cannot. The model actively moves away from the symmetric initialization during training.

**An important caveat:** ConvShift also widens the dev-test gap. Dev CER drops substantially (0.168 → 0.159), but test CER improves only modestly (0.179 → 0.176). The model fits dev-specific patterns through the extra flexibility. I tried fixing this gap with WSD scheduler and heavier augmentation, but those attempts were either neutral or counterproductive (see Part 6). The gap appears to be a data-size issue.

**The gate variant hurt.** Adding an output gate conditioned on the convolution residual was semantically wrong: a local 3-frame signal was gating a global full-sequence attention output. CER went from 0.176 to 0.181. The 394K gate parameters learned spurious correlations.

### Layer-Dependent ConvShift — Encoding the Multi-Scale Prior (CER 0.177)

Wider kernels in early layers (kernel=7) and narrow kernels in upper layers (kernel=3), encoding the multi-scale depth hierarchy (see Part 3). CER 0.177 — essentially tied with uniform ConvShift. The idea is sound and confirmed by gradient analysis, but at T=250 the benefit is marginal.

---

## Part 3: The Decay Mechanism Exploration (The Biggest Investigation)

This was a connected series of experiments all attacking one question: **can I give LION a richer temporal attention structure beyond its default monotone exponential decay?**

LION's attention weight between positions i and j decays as r^|i-j| where r is in (0,1) — a simple exponential falloff with distance. Every head in every layer gets roughly the same decay profile. I hypothesized this was suboptimal and tried several alternatives.

### What I Tried and What Happened

**Complex/oscillatory decay (cos modulation)** — Multiplied the decay by cos(θ·|i-j|) to create periodic "peaks" at certain distances. Both frequencies tested were significantly worse (CER 0.214 and 0.232 vs baseline 0.179).

*Why it failed:* Cosine produces **negative** attention weights at half-period distances. Negative attention means actively suppressing information from certain distances. There's no reason to always suppress information from a specific distance in sequential data.

**cos² modulation (non-negative fix)** — Squaring the cosine eliminates negative weights. CER improved to 0.196 — much better than raw cosine but still below baseline. The periodic structure itself is unnecessary.

**Learnable θ per layer** — Instead of fixing the oscillation frequency, I let each layer learn its own θ. The model spontaneously learned a hierarchy:
- Layer 0: θ≈0.18 (slow oscillation, ~350ms — broad attention)
- Layers 2-3: θ≈0.30 (~210ms)
- Layers 4-5: θ≈0.40 (fast, ~157ms — local attention)

This was a key finding: **the gradient actively prefers different temporal scales at different depths.** But the cos/cos² vehicle for expressing this was still net-negative (CER 0.211).

**Headscale (per-head decay bias)** — The simplest possible intervention: one learnable scalar per head (24 parameters total) that scales the log-decay rate. Each head can independently become "more global" or "more local."

Result: CER 0.184, best dev CER in the entire project (0.166). The learned biases confirmed the same depth hierarchy:
- Layers 0-1: biases near zero or slightly negative (broader attention)
- Layers 2-5: all positive, monotonically increasing (+0.15 to +0.37 — progressively more local)

**Gaussian attention mask** — The model could learn to peak attention at a specific distance μ. CER 0.186. All heads learned μ→0 with very wide σ (~600ms, flat). The model doesn't want peaked attention at non-zero distances. The "non-monotone attention" hypothesis is **not supported** at this sequence length.

**Dual-decay** — Combined a fast-decaying and slow-decaying exponential. CER 0.188. The mixing weight α stayed near 0.95 (95% fast). The slow component is barely used. 2× compute for essentially no benefit.

### The Cross-Experiment Conclusion

Three completely independent mechanisms — learnable oscillation frequency, headscale bias, and dual-decay mixing — all converge on the same structural finding:

> **Lower layers want to attend broadly and upper layers want to attend locally. This is a robust structural preference, not an artifact of any particular parameterization.**

But all the modulation forms (cosine, Gaussian peaks, dual-decay) failed to beat the baseline despite capturing real gradient signal. The simplest form that helps is headscale. The cost of structured constraints outweighs the benefit because the decay mechanism already works well enough at this sequence length. On longer sequences, these mechanisms might become more relevant.

### Per-Head Temperature — Another Confirmation (CER 0.179)

Per-head learnable temperature τ, initialized as a linear ramp from 1.0 (layer 0) to 2.0 (layer 5). The model preserved the initialized hierarchy (L0 near 1.0, L5 near 1.9). Yet another confirmation that upper layers prefer sharper attention. CER (0.179) is close to baseline — mostly redundant with what the decay already provides.

---

## Part 4: RWKV-7 Built for a Different Scale

Stock RWKV-7 gave CER 0.400 — essentially non-functional. Even 100 epochs didn't help. The problem wasn't training time; it was **starting in a bad operating regime**.

RWKV-7 was designed entirely for large-scale autoregressive language modeling (billions of parameters, very long sequences). Its default initialization, gating mechanisms, and key scaling all assume a regime fundamentally different from a 7M-parameter, 6-layer, 256-dim encoder. Specifically:

1. **Decay init too narrow.** Stock decay range was [0.0009, 0.111] — effectively 1-token memory. RWKV-6's range is [0.007, 0.378]. Fix: shift w0 by +2.0.

2. **v_first too dominant.** The "value residual" gate is designed for very deep LLMs. At 6 layers it suppresses hierarchical feature formation entirely. Fix: initialize to sigmoid(-5) ≈ 0.007, effectively disabling it.

3. **k_a halves keys.** Stock init weakens the attention signal when the model is already small. Fix: set k_a = 0.

Decay fix alone: CER 0.378 (still bad). All three fixes: CER 0.260 (usable but still 45% worse than LION and 10% worse than plain RWKV-6). The gap to RWKV-6 is architectural, not directional — both are unidirectional. RWKV-7's delta-rule recurrence and LoRA-based key scaling add complexity that doesn't pay off at this model size.

**The practical conclusion:** Rather than trying to make the full RWKV-7 architecture work at small scale, the better strategy is to **extract its most valuable ideas — particularly the delta rule — and transfer them to RWKV-6**, which is simpler, works well at this scale, and provides a cleaner substrate for mechanism testing. This is exactly what I did in the Delta Rule experiments (Part 5).

---

## Part 5: Delta Rule and LUCID (The Most Promising Mechanisms)

These are the most important recent experiments. Both mechanisms address fundamental problems in recurrent attention, and both show strong results on unidirectional RWKV-6. The question is how to make them work on the bidirectional architecture too.

**Important note on baselines:** The training pipeline evolved over the course of this project, so I re-ran matched baselines for these experiments. The Delta Rule and LUCID results should be compared against:
- Plain RWKV-6 baseline: CER **0.250**
- Bidirectional RWKV-6 (LION) baseline: CER **0.185**

### LUCID Preconditioner (Key Decorrelation)

**The problem it solves:** When keys are correlated (nearby positions produce similar key vectors), the attention mechanism becomes degenerate. Multiple positions get nearly identical attention weights, and the model can't distinguish between them. Head capacity is wasted because different heads end up attending to the same blurred region.

**How it works:** Compute a Gram matrix of L2-normalized keys, build a preconditioner P = (I + exp(τ · K_gram))⁻¹, and apply it to the values: instead of A·V, compute A·solve(P, V). This pushes the output space apart where the key space is too similar. Only adds 1 learnable temperature parameter per head (24 total).

**The intuition:** Imagine two nearby positions with nearly identical keys. Without LUCID, they get nearly identical attention-weighted outputs, so the model can't tell them apart downstream. LUCID detects this key similarity and adjusts the value output so that even when keys are similar, the model still gets differentiated information. It's like adding a correction term that says "yes, these positions look similar in key space, but here's how their values actually differ."

**On unidirectional RWKV-6: CER 0.250 → 0.212 (−15.1% relative).** This is the largest single-mechanism improvement on the unidirectional encoder in the entire project. The improvement is genuine — train loss drops to 0.605 vs baseline 0.773, so the model is actually learning better, not just getting lucky.

LUCID also produces **cleaner hidden states**. Carry-state degradation at longer windows is milder than baseline (Δ@5s: -0.020 vs baseline -0.030). The decorrelated values accumulate less noise in the recurrent state over time.

**On LION: CER 0.185 → 0.186 (+0.5%, neutral).** This is disappointing but instructive. Two hypotheses for why it doesn't transfer:

*Hypothesis A — Implicit decorrelation:* LION's exponential decay structure already acts as a soft distance-based weighting. Nearby correlated keys get high weight, but the decay ensures that the overall attention pattern is dominated by distance structure rather than key similarity. LUCID's explicit decorrelation is then redundant — the problem it solves doesn't exist in LION.

*Hypothesis B — Granularity mismatch:* LUCID computes a full T×T preconditioner (T≈250). In the unidirectional chunked kernel, the Gram matrix is 64×64 — a local problem where correlations are strong and the preconditioner can effectively act on them. The full-sequence Gram matrix may average out local correlations that actually matter. **Chunked LUCID** (applying the preconditioner within 64-frame windows rather than the full sequence) would test this directly. If chunked LUCID helps LION, the mechanism is sound but the granularity was wrong.

*How to make it work on LION:* The chunked approach is the most promising path. It aligns the preconditioner's scope with where key correlations actually matter (local neighborhoods). Additionally, a **temperature warmup schedule** (start with weak decorrelation, increase over training) could prevent early destabilization before keys are meaningful.

### Delta Rule (Selective State Erasure)

**The problem it solves:** In recurrent processing, the hidden state S accumulates key-value associations over time: S(t) = decay·S(t-1) + v(t)·k(t). Old associations pile up and never get explicitly removed — they just fade through the decay. If the current input is related to an old, now-irrelevant association, the stale information interferes. The delta rule adds a correction: before writing new associations, it erases old ones that are correlated with the current key. Formally, S(t) = decay·S(t-1) + S·(ab) + v(t)·k(t), where the ab term selectively subtracts stale associations.

**The intuition:** Think of the hidden state as a scratchpad where the model writes notes. Without the delta rule, old notes fade but never get erased — they become smudged background noise. The delta rule says "before writing a new note about this topic, erase whatever was previously written about it." This keeps the scratchpad clean and the current information sharp.

**On unidirectional RWKV-6: CER 0.250 → 0.225 (−10.0% relative).** Solid improvement. The state erasure genuinely helps — old associations do become stale in sequential processing. The delta rule adds ~2.6% parameters (LoRA matrices a0/a1/a2, key norm, key scaling).

**On LION: CER 0.185 → 0.202 (+9.0%, harmful).** I applied the delta corrections to both the causal (forward) and anticausal (backward) attention matrices. The causal correction follows RWKV-7's original formulation. The anticausal correction was my extrapolation — there is no theoretical basis for it in the literature. Training dynamics confirm the problem: LION+Delta converges to train loss 0.574 vs baseline 0.545. The anticausal correction constrains the backward attention matrix in a way that hurts optimization.

*How to make it work on LION:* **Causal-only delta** is the clear next experiment. Apply the delta correction to the forward attention path only, leave the backward path untouched. The causal component has solid theoretical grounding (it comes directly from RWKV-7's recurrence), and the −10% improvement on unidirectional suggests it carries real value. If the causal-only version helps or is at least neutral on LION, it would confirm that the anticausal extrapolation was the sole source of degradation. Another path is testing with a **larger LoRA rank** — the current rank-32 bottleneck (at d_model=256) may be too restrictive for the delta rule to fully express the erasure patterns it needs.

### LUCID vs Delta Rule

These two mechanisms address different problems:
- **LUCID** fixes a spatial problem (correlated keys → degenerate attention patterns)
- **Delta Rule** fixes a temporal problem (stale associations → noisy hidden state)

In principle, they should be **complementary** — LUCID improves the quality of each timestep's attention, while the delta rule improves the quality of information carried across timesteps. Whether they actually amplify each other or just stack additively is an open question. Testing them individually first (especially getting each one to work on LION) makes more sense than stacking them prematurely.

### The Dev-Test Gap Is Consistent

All six models from the Delta Rule/LUCID experiment show a dev-test gap of ~0.02. This confirms the gap is a Common Voice Ukrainian split property. Neither mechanism introduces additional overfitting beyond baseline.

---

## Part 6: The Regularization Lesson

I noticed that enhanced models (ConvShift, temperature, layerconv) showed excellent dev CER (0.157–0.161) but test CER barely improved over baseline (0.176–0.179).

**My response:** Aggressive regularization — dropout 0.15→0.25, SpecAugment frequency masks 15→27, time masks 35→70 with 5 masks instead of 2, extended to 80 epochs.

**Result: complete failure.** CER roughly doubled to 0.28–0.32. The models couldn't learn the training data at all (train loss 1.7–1.8 vs normal ~0.5). For 4-5 second utterances, five 70-frame time masks can blank out most of the input. Dropout 0.25 on a 256-dim model zeroes too much capacity.

**The lesson:** The dev-test gap is a data-size problem, not a regularization problem. Heavier regularization doesn't fix this — it just prevents learning entirely. The original settings (dropout=0.15, freq_mask=15, time_mask=35, 2+2 masks) were already near-optimal for this scale.

---

## Part 7: Scaling Depth

I tried doubling LION to 12 layers (100 epochs). CER: 0.182 — *worse* than the 6-layer model (0.176) despite 1.75× more parameters. The model saturates at 6 layers for 35h of data. More capacity doesn't help without more data.

---

## What Worked, What Didn't, What's Next

### Confirmed Improvements

| Mechanism | Effect | Why It Works |
|-----------|--------|-------------|
| **LION (bidir RWKV-6)** | Best architecture overall (CER 0.176-0.185) | Full bidirectional attention with learned distance decay |
| **ConvShift (no gate)** | −1.7% relative on LION | Per-channel learned temporal smoothing, breaks past/future symmetry |
| **LUCID on unidir RWKV-6** | −15% relative (0.250 → 0.212) | Decorrelates keys → better attention diversity + cleaner carry-state |
| **Delta Rule on unidir RWKV-6** | −10% relative (0.250 → 0.225) | Erases stale state associations in recurrent processing |
| **Headscale** | CER 0.184, best dev 0.166 | Per-head decay rate bias, captures depth-dependent attention profile |
| **Multi-scale depth prior** | Confirmed by 5 independent mechanisms | Lower layers broad, upper layers local — robust structural preference |

### What Didn't Work (On This Data)

| Mechanism | Why It Didn't Work | Could It Work Elsewhere? |
|-----------|-------------------|--------------------------|
| **Mamba** | Overfits/saturates early; 100 ep = 60 ep | Bidir Mamba untested; may benefit from longer sequences where linear-time matters |
| **Stock RWKV-7** | Init designed for billion-param LLMs; broken at small scale | Transfer its ideas (delta rule) to RWKV-6 instead |
| **Cosine/cos² decay** | Negative weights (cos) or unnecessary periodic structure (cos²) | Unlikely on any data |
| **Gaussian peak / Dual-decay** | Non-monotone attention not needed at T=250; α→0.95 | Possibly relevant at T>2000 |
| **Aggressive regularization** | Underfitting at 7.7M/35h scale | Different reg settings might work with more data |
| **12-layer depth** | Saturates with 35h data | Would help with more data |
| **Delta Rule on LION (both directions)** | Anticausal extrapolation has no theoretical basis | Causal-only version may work (see below) |

### Next Steps (Prioritized)

**1. Move to bigger datasets with both short and long utterances.** This is the most important step. The current 35h short-utterance dataset limits what can be meaningfully evaluated:
- The dev-test gap is a data-size ceiling, not an architecture problem.
- Carry-state behavior cannot be properly tested on short utterances — longer sequences are needed to stress-test recurrent state accumulation.
- Mechanisms like LUCID and delta rule that improve hidden state quality will show their real value on longer sequences where state management actually matters.
- Two datasets (one short-utterance, one long-form) would let me separate sequence-length effects from data-size effects.

**2. Make LUCID work on bidirectional LION.** The −15% improvement on RWKV-6 is too large to ignore. The most promising approach is **chunked LUCID** — applying the preconditioner within 64-frame windows instead of the full sequence. This tests whether the neutrality on LION is a granularity problem. If chunked LUCID helps, it confirms the mechanism is sound and only the application scope needs adjustment.

**3. Test causal-only Delta Rule on LION.** The anticausal delta correction was the clear source of degradation. Applying delta corrections only to the forward attention path — where the mechanism has theoretical grounding from RWKV-7 — is the obvious next experiment. If it helps or is neutral, the anticausal problem is isolated and the mechanism becomes usable on LION.

**4. Implement carry-state for all main architectures.** Right now only Mamba has fully working carry-state inference. To properly evaluate streaming potential on longer data, I need to implement and test carry-state for plain RWKV-6 (partially working but degrades at >2s), RWKV-6+LUCID (shows cleaner state accumulation), and RWKV-6+Delta (state erasure should directly improve carry behavior).

**5. Test LUCID and Delta Rule independently on larger data before combining.** The question of whether these mechanisms amplify each other or just stack additively matters. If they just add their effects, combining them is no more valuable than picking the better one. I should first confirm each mechanism's individual contribution on the new datasets, understand whether they interact with sequence length differently, and only then consider combinations where there's reason to believe in synergy.

**6. Headscale with longer training.** The headscale biases weren't fully converged at 60 epochs — the depth-varying trend was still developing. Worth trying at 100 epochs. Cheap experiment.

---

## The Big Picture

I started with 5 architectures and identified LION (bidirectional RWKV-6) as the strongest for offline processing and RWKV-6 as the best unidirectional baseline. The main conceptual finding is the **multi-scale depth hierarchy** — lower layers consistently prefer broad attention, upper layers prefer local — confirmed independently by 5 different mechanisms.

The two most promising improvement mechanisms are **LUCID** (−15% on RWKV-6) and **Delta Rule** (−10% on RWKV-6). Both address genuine problems in recurrent attention — key correlation and state staleness respectively. Neither transfers to LION yet, but there are clear hypotheses for why (granularity mismatch for LUCID, anticausal extrapolation for Delta Rule) and clear next experiments to test them.

The immediate priority is switching to larger, longer-sequence datasets. The current 35h short-utterance setup has been invaluable for rapid prototyping and mechanism screening, but it's reached its ceiling. The mechanisms that survive on bigger data with longer sequences are the ones worth writing about in the thesis.
