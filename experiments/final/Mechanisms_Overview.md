# The Three Mechanisms — What We're Reporting and Why

*A short, honest overview of what we built, what claim each mechanism
supports, and how they transfer across architectures. Written for the
advisor, not the thesis itself.*

---

## What this thesis is actually about

The thesis is not "we made ASR a bit better." It is:

> *Causal linear-time RNNs (RWKV-6, Mamba-2, Linear Attention) are an
> alternative architectural family to softmax attention. They each have
> different structural deficits. We can identify those deficits, design
> mechanism-level fixes for them, and show that the fixes transfer
> across the family in a predictable, deficit-proportional way.*

The reason for using ASR (LibriSpeech clean-100, CTC, 7M / 14M params,
50 ep) is that it is a clean, reproducible probe — short sequences,
no autoregressive eval noise, single metric (CER). It's the measurement
spine, not the goal.

Out of about twenty mechanisms we tried during Stages 2–10, three
survived as the **transferable, productive** ones. Those are the three
the thesis reports. Everything else we tried is either supplementary
evidence (the engaged-null catalog — mechanisms that activate but don't
help, which is itself a finding) or out-of-scope.

The three are:

1. **multidil_v2** — multi-scale local temporal mixing on the input side.
2. **LUCID** — within-chunk decorrelation of values via a key-similarity preconditioner.
3. **rse_strong_viscosity** — block-complex transition geometry with Rayleigh damping.

They target three distinct axes of expressivity, and each was tested on
all three architectures plus their bidirectional (LION) wrappers.

---

## Mechanism 1 — `multidil_v2` (input-side multi-scale convolution)

### The intuition

RWKV / Mamba / linear-attention all have a recurrent or near-recurrent
core that mixes information across time, but none of them have a strong
*local* mixing operator the way a Transformer's softmax-with-relative-
position effectively does. Each token essentially sees only its own
embedding plus the running state. For ASR, the relevant local structure
is at the phoneme-to-syllable scale — roughly 40–160 ms after the
2× subsampling, which is 4–16 frames. A single-tap shift can't see that.

The mechanism is a parallel bank of depthwise dilated 1D convolutions
on the input, mixed by learnable per-layer weights:

$$x_\text{mixed}[t] \;=\; \sum_{d \in \{1, 2, 4, 8\}} \alpha_d \cdot \mathrm{DWConv}_{k=3,d}(x)[t]$$

Four dilations × kernel 3 covers receptive fields of 1, 2, 4, 8 frames
on each side. The mixing weights $\alpha_d$ are learned per layer.

### Why "v2"

The first version of this mechanism had a serious init bug. The dilated
branches for $d > 1$ had both $\alpha_d = 0$ *and* their conv weights
$W_d = 0$. This is a multiplicative gradient trap: the gradient w.r.t.
$\alpha_d$ is proportional to $W_d \cdot x$, the gradient w.r.t. $W_d$
is proportional to $\alpha_d$, and when both are zero neither moves.
We verified this on three architectures: at end of training, all
$\alpha_{d>1} = 0$ exactly. The mechanism had been silently collapsing
to single-dilation conv with one learnable scalar.

The fix (`v2`): initialise $\alpha_{d>1} = 0.01$ and
$W_{d>1} \sim \mathcal{N}(0, 0.01^2)$. This breaks one leg of the
zero-zero product, lets gradient flow, and perturbs the init output by
about $3\!\times\!10^{-4}$ — well below activation noise (so the
zero-regression contract still holds at init).

This is a small story but a thesis-grade one — it's a concrete example
of the "exact reduction at init" discipline catching a real bug, and
the v1 vs v2 results are themselves evidence that the multi-scale
hypothesis matters (single-dilation got 0.1153, multi-scale got 0.1000).

### Where the idea comes from

The lineage is the strongest cross-domain story in the thesis. Five
independent groups have built essentially the same mechanism on top of
linear-time mixers across very different domains:

| Domain | Architecture | Paper | What they call it |
|---|---|---|---|
| 2D vision | RWKV-4/5 | Vision-RWKV (arXiv:2403.02308) | Q-Shift |
| 2D vision | Linear-DiT | LiT (arXiv:2501.12976) | Depthwise conv on V |
| 1D language | Linear-attention LM | Paper 7 (arXiv:2506.01963) | Multi-dilation DWConv $\{1,2,4,8\}$ |
| 2D spectrogram | RWKV-7 | AudioRWKV (arXiv:2509.02167) | Q-Shift + 2D DWSep ConvShift |
| 1D audio (CTC) | RWKV-6 / Mamba-2 / LA | This thesis | multidil_v2 |

Each one shipped as a productive component in its respective paper.
Paper 7 is the closest mechanically to our form — they propose the
$\{1, 2, 4, 8\}$ multi-dilation specifically. AudioRWKV is the most
relevant by domain (audio spectrogram on RWKV-7).

### What this mechanism is supposed to claim in the thesis

> *Linear-complexity sequence mixers structurally lack local inductive
> bias, and depthwise-convolution-class mechanisms restore it. This is
> orthogonal to the architecture's transition kernel — it works on
> RWKV-6 (diagonal $\mathsf{NC}^1$), on RWKV-7 (DPLR $\mathsf{PNC}^1$),
> on Mamba-2's selective SSM, and on pure linear attention.*

This is the most empirically robust claim in the thesis because we have
five independent confirmations across two modalities and four different
transition kernels.

### Transfer evidence (LibriSpeech clean-100, 7M, single seed)

| Architecture | Vanilla | + multidil_v2 | Δ | Reading |
|---|---:|---:|---:|---|
| RWKV-6 (causal) | 0.1263 | **0.1000** | −0.0263 | Largest gain among RWKV-style ConvShift variants. |
| Mamba-2 (causal) | 0.1192 | **0.0967** | −0.0225 | Smaller gain — Mamba-2's native k=4 DWConv already covers part of axis 1. |
| Linear Attention (causal) | ~0.220 | ~0.170 | −0.050 | Largest gain — LA had no native local mixing at all. |
| RWKV-6 LION (bidir) | 0.0712 | further drop expected | (running) | Native parallel home for the mechanism. |

The ordering — LA > RWKV-6 > Mamba-2 — is exactly what you'd predict
from how much native axis-1 coverage each architecture already had.
That ordering is the deficit-proportional transfer claim, and it shows
up cleanly here.

---

## Mechanism 2 — `LUCID` (within-chunk value decorrelation)

### The intuition

Linear-time RNNs have an associative-memory problem. They store keys
and values into a state matrix; if many keys are similar, their values
overlap in the state and interfere at readout. This is the standing
gap on Mamba-2 — it has continuous decay (axis 1) and content-dependent
$\Delta t$ (axis 5) but nothing that decorrelates similar keys.

LUCID is a small operator that solves a $T_c \times T_c$ linear system
inside each chunk to "subtract out" what each value adds because of
correlation with earlier keys, before the value enters the recurrence.
Within a chunk:

$$K^{RN} = \sqrt{d}\,\frac{K}{\|K\|_2}, \quad G = K^{RN} (K^{RN})^\top$$

$$P = \exp\!\Big(\tau_h \cdot \big(\tfrac{G}{\sqrt{d}} - \sqrt{d}\big)\Big) + \varepsilon I$$

Then we solve $P \tilde V = V$ once per chunk per head and feed the
preconditioned $\tilde V$ into the recurrence (or the parallel attention,
in LION). $\tau_h$ is a single learnable scalar per head. The
$-\sqrt{d}$ shift makes $P$ unit-diagonal by construction; otherwise a
token would "subtract itself out" and the mechanism would degenerate.

The point: $A \cdot P^{-1} \cdot V = A \cdot \tilde V$. Matrix
multiplication is associative, so we never have to materialise the
full attention matrix $A$ — we precondition the values, and the
recurrence does the rest of the work unchanged.

### How we frame it (and the honest provenance story)

The unit-diagonal preconditioner kernel itself appears in the LUCID
paper (arXiv:2602.10410). What is *ours* in this thesis:

1. **The chunked operational forms.** The LUCID paper proposes the
   preconditioner in the parallel-attention setting. We adapt it to:
   - **Chunked recurrent** form for RWKV-6 (apply within chunks of 64,
     pass preconditioned $\tilde V$ to the chunked WKV scan, no
     materialised attention matrix anywhere).
   - **Parallel** form for LION's bidirectional T×T attention.
   - **SSD-chunk-local** form for Mamba-2, derived from scratch.
2. **The Mamba-2 adaptation.** Earlier we wrote LUCID off as
   "structurally incompatible" with Mamba-2 because Mamba-2 doesn't
   materialise full attention. That was wrong. Mamba-2's SSD dual form
   *does* materialise chunk-local $T_c \times T_c$ attention-like
   structure — that's literally how the chunk-parallel scan works.
   We use Mamba-2's $C$ tensor (the query analog) as the correlation
   source, build $P$ from $C C^\top$ within each chunk, solve, scan
   on preconditioned values. That derivation is original to this
   research.
3. **Per-head learnable temperature** $\tau_h$ instead of the paper's
   fixed $1/\sqrt{d}$, plus an $\varepsilon I$ regulariser bumped to
   $10^{-4}$ after we saw singular solves at trained $\tau \approx 1.5$.

We keep the name "LUCID" so the reader has a clean handle, and we cite
the paper as related work for the preconditioner kernel, but we're
honest in the writeup that the operational adaptations (especially the
Mamba-2 chunked-SSD adaptation) are what the thesis contributes.

### What this mechanism is supposed to claim in the thesis

> *Axis-2 (associative memory under interference) is real, and it is
> the standing gap in Mamba-2's native primitive coverage. A
> within-chunk preconditioner-style decorrelator fills it. The
> mechanism transfers asymmetrically — productive on RWKV-6 and
> Mamba-2, but not on Linear Attention — which is the opposite of
> what a "natural home in explicit attention" prior would predict.*

The asymmetric transfer is itself the interesting thing. It says the
relevant property isn't "does the architecture have an explicit
attention matrix" but "does the architecture have rich state structure
worth decorrelating in the first place."

### Transfer evidence (LibriSpeech clean-100, 7M, single seed)

| Architecture | Vanilla | + LUCID alone | + LUCID × multidil_v2 (P7) | Reading |
|---|---:|---:|---:|---|
| RWKV-6 (causal) | 0.1263 | 0.1216 (−0.0047) | **0.0921** | New causal RWKV-6 ceiling. First productive cross-axis composition. |
| Mamba-2 (causal) | 0.1192 | 0.1109 (−0.0083, ~6σ) | 0.0993 | Productive on its own *and* in composition. |
| Linear Attention (causal) | ~0.220 | 0.2057 (modest) | 0.1718 (tied with multidil_v2 alone) | LUCID-alone weak; composition tied within σ. The anomaly. |

For Mamba-2 we ran both the $B$-correlation variant (key-analog,
0.1113) and the $C$-correlation variant (query-analog, 0.1109). They
agree within noise — the master plan reports `lucid_c` because the
numbers are marginally cleaner and the query-side framing pairs better
with how we describe the mechanism in prose.

---

## Mechanism 3 — `rse_strong_viscosity` (complex-pole transition with Rayleigh damping)

### The intuition

RWKV-6's transition is a real diagonal $(\mathbb{R}_+)^K$ — every state
channel just decays multiplicatively. That is a fine model for
exponential decay, but it cannot represent oscillation. Speech formants
*are* damped oscillations, so a complex-pole transition is the
physically appropriate function class.

Rotational State Embedding (RSE) replaces the real diagonal with a
block-complex structure $SO(2)^{K/2} \times (\mathbb{R}_+)^{K/2}$:
each pair of state channels is a 2-vector that is rotated by an angle
$\theta$ and scaled by a damping factor $e^{-\lambda}$ at every step.
Per 2×2 block:

$$G_{t,b} = e^{-\lambda_\text{eff}} \cdot R(\theta_{t,b}), \qquad R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

The angle $\theta$ is data-dependent (rank-48 LoRA). The
"strong" part of the name means we lifted the rotation budget from
$\pi/4$ (Stage 3 default) to $\pi/2$ uniformly across layers — earlier
stages showed the deeper layers were starved by the tight clip.

The "viscosity" part is the load-bearing physical-prior contribution:

$$\lambda_\text{eff} = \lambda_\text{raw} + \eta_{h,b} \cdot \theta_{t,b}^2$$

This is Rayleigh dissipation. High-frequency rotations damp faster.
Without it, large $\theta$ caused phase drift across many steps and the
mechanism became unstable; with it, the system self-regulates and the
clip becomes a soft constraint. $\eta$ is zero-init, so the mechanism
reduces exactly to plain RSE at step 0 (zero-regression contract).

### Where the idea comes from

This one is internally derived. We can frame it as a Lie-group
extension argument: the function class of real diagonal transitions is
$(\mathbb{R}_+)^K$; the natural smallest extension that can represent
formants is $SO(2)^{K/2} \times (\mathbb{R}_+)^{K/2}$. The Rayleigh
viscosity term is a standard physical prior from damped-oscillator
mechanics.

The thesis-relevant theory papers we *cite* (but don't run mechanisms
from) are:

- arXiv:2603.01959 (negative bound) — proves that diagonal-complex
  SSMs, including ours, cannot track non-Abelian state at finite
  precision. This formally puts state-tracking out of scope for RSE
  and motivates the future-work paragraph.
- arXiv:2603.03612 (positive bound) — DPLR-class transitions
  (DeltaNet, RWKV-7) are $\mathsf{PNC}^1$-complete. This is the
  formal class boundary we *don't* try to cross.

The thesis position is: RSE+viscosity targets the *axis-1 sub-axis*
of damped-oscillator dynamics, not state-tracking. The two theory
papers bracket axis 3 cleanly — they explain why RSE is null on
non-Abelian state-tracking benchmarks and why we don't try to fix that
within the diagonal family.

### What this mechanism is supposed to claim in the thesis

> *Real-diagonal recurrences are the wrong function class for damped
> oscillation. A block-complex transition with Rayleigh damping fixes
> this on architectures that don't already have it (RWKV-6, LA), and
> is null where the architecture already provides equivalent capability
> through a different primitive (Mamba-2's selective $\Delta t$).*

The Mamba-2 null is genuinely informative — Mamba-2's $\Delta t$ is
per-channel content-dependent decay, which absorbs RSE's contribution
through reparameterisation. That null is not a mechanism failure; it
is a clean prediction confirmed.

### Transfer evidence (LibriSpeech clean-100, 7M, single seed)

| Architecture | Vanilla | + rse_strong_viscosity | Δ | Reading |
|---|---:|---:|---:|---|
| RWKV-6 (causal) | 0.1263 | **0.1185 dev / 0.1177 test** | −0.0086 | The original anchor. |
| Mamba-2 (causal) | 0.1192 | 0.1183 | −0.0009 | NULL (within σ). $\Delta t$ already provides equivalent decay diversity. |
| Linear Attention (causal) | ~0.220 | ~0.142 | −0.078 | Largest gain on the entire spine. LA had no decay at all. |

The pattern (LA » RWKV-6 » Mamba-2) is again deficit-proportional and
again shows up cleanly in the numbers.

---

## How the three together support the thesis claim

The master-plan claim, in plain English, is: *if you tell me an
architecture's structural deficit, I can predict whether a mechanism
will help.* The three mechanisms are designed to test this in three
different ways:

| Mechanism | Transfer pattern | What this tells us |
|---|---|---|
| multidil_v2 | Helps everywhere, smallest gain on Mamba-2 | Axis is universal but partially absorbed where native primitives overlap. |
| LUCID | Helps RWKV-6 and Mamba-2; weak on LA | Mechanism identity matters, not just deficit size. State structure has to be there for decorrelation to bite. |
| rse_strong_viscosity | Big gain on LA, baseline on RWKV-6, NULL on Mamba-2 | A clean "absorbed by Δt" prediction confirmed. |

Three mechanisms, three different transfer signatures, all consistent
with a single underlying principle. That's the empirical core of the
thesis: not that any one of these is a giant CER win, but that the
*pattern* across the matrix is what an axis-decomposition framework
predicts in advance.

The supplementary closed-cells evidence (7+ engaged-null mechanisms
from Stages 2–10, Family-D quadratic-lift saturation, Stage-11
Mamba-2 novelty-gate scale-mismatch) backs up the converse: dense
per-token freedom *without* prior alignment doesn't convert. That's
the cross-experiment invariant the thesis discussion section leans on.

## How I'll represent the work in the final stage

A few framing choices the advisor should know about:

1. **multidil_v2 is presented as the well-supported "axis-1 universal"
   mechanism.** Lots of independent confirmation in the literature; we
   are one node in a five-paper cross-domain story. Honest framing,
   strong evidence.

2. **LUCID is presented as a mechanism we adapted from a related
   paper, with the operational form (especially the Mamba-2 SSD
   chunk-local adaptation) as our contribution.** We keep the name;
   we cite the LUCID paper for the unit-diagonal preconditioner kernel;
   the chapter is honest that the chunked deployment forms across
   three architectures are derived in this research. The Mamba-2
   adaptation has its own short derivation section because the earlier
   "structurally incompatible" claim retracted itself, and that story
   is more interesting than glossing over.

3. **rse_strong_viscosity is presented as the internally-derived
   mechanism.** The Lie-group extension argument is ours; the Rayleigh
   coupling is a physical prior we imported deliberately. The Mamba-2
   null is reported as a *prediction confirmed*, not an embarrassment.
   The two theory papers (arXiv:2603.01959, arXiv:2603.03612) bracket
   what RSE can and can't do at the formal level and motivate keeping
   axis-3 / state-tracking out of scope.

4. **The thesis is not claiming SOTA on LibriSpeech.** It is claiming
   that the architectural family has measurable, predictable structure
   under mechanism-level perturbation. CER numbers are a measurement
   instrument. We should be careful not to oversell any one number;
   the matrix as a whole is the artifact.

5. **The LION (bidirectional) chapter is parallel, not the headline.**
   The same three mechanisms get tested in their parallel-attention
   form to quantify the causality penalty and verify the mechanism
   vocabulary doesn't break under bidirectionalisation. It's a clean
   secondary chapter, not the main result.

---

## Why we bounded the main study to ASR

ASR was a deliberate choice as the measurement spine, for four reasons.

**It is a clean, reproducible probe.** Short sequences ($T \le 500$
after subsampling), deterministic CTC evaluation, a single primary
metric (CER), and a HuggingFace dataset that requires no preprocessing
beyond what we already wrote. Single-seed runs at 7M / 30 ep complete
in roughly two GPU-hours, which is what made the Stages 2–10 mechanism-
discovery chain (twenty-plus mechanisms screened) possible at all. A
slower, noisier benchmark would have collapsed the iteration cadence
and we would never have reached the cross-experiment invariant.

**It exercises one axis cleanly.** ASR has well-defined acoustic
structure at the phoneme-to-syllable scale — exactly the regime where
axis 1 (short-range temporal hierarchy) is binding. That made it a
sharp instrument for testing axis-1 mechanisms (multidil_v2,
RSE+viscosity), and a sharp instrument for *failing* to test other
axes — which is also valuable, because it gave us the engaged-null
catalog (mechanisms that activated under SGD but didn't move CER
because the task structure didn't reward them).

**It avoids autoregressive eval noise.** Decoding is greedy CTC, not
sampled generation. That removes a whole layer of variance that would
have made small mechanism-attribution claims (1–3σ effects) impossible
to defend at single-seed.

**It was where we already had infrastructure.** The draft phase in
`asr_backbone_comparison/` established the training loop, the
data pipeline, and the parameter-matched encoder factory. The formal
codebase is a clean rewrite of that, not a new project.

The cost of this scoping is real and we are explicit about it in the
writeup: ASR characterizes axis 1 thoroughly, but it cannot
characterize axes 2–5 by construction. The "engaged null" results from
Stages 2–10 (Delta rule, Cayley orthogonal, M²RNN, dense non-normal
RSE) are not mechanism failures — they are mechanisms that target
axes ASR does not exercise. Claiming "DeltaNet doesn't help" from
ASR alone would be a scope error. So the thesis adds a second
benchmark.

## Why Zoology / MQAR is the second benchmark

The second benchmark is **MQAR (Multi-Query Associative Recall)** from
the Zoology paper (*Arora et al., "Zoology: Measuring and Improving
Recall in Efficient Language Models"*). MQAR is a synthetic
key-value-recall task: the model sees a sequence of $(k, v)$ pairs
followed by query keys, and has to retrieve the matching values.
We sweep sequence length $T \in \{64, 256, 1024\}$.

It earns its place in the thesis for four reasons.

**It exercises axis 2 by construction.** MQAR is built specifically
to stress associative-memory capacity under interference. When the
number of stored pairs $N$ approaches or exceeds the state dimension
$d$, an architecture that cannot decorrelate similar keys collides
in its state matrix and recall accuracy drops. This is precisely the
deficit that LUCID targets. The Zoology paper's central finding —
that efficient sub-quadratic architectures degrade on MQAR while
softmax attention does not — is the empirical motivation for the
entire associative-memory line of work.

**It validates LUCID where ASR cannot.** LUCID's gain on ASR is
small (around −0.005 on RWKV-6) because ASR doesn't push the
state-saturation regime. Without MQAR the LUCID claim is undertested,
and the asymmetric transfer pattern (productive on RWKV-6 + Mamba-2,
weak on LA) is hard to defend purely on ASR-attributed numbers. MQAR
gives LUCID its natural home, and gives the thesis a way to show
that a mechanism we adapted for one task transfers cleanly to a
benchmark designed by an independent group for the exact axis the
mechanism targets.

**It is fast and length-scalable.** MQAR is synthetic, generates
unlimited data, and trains in minutes per length. The full sweep
across three lengths × ten backbones is roughly six GPU-hours, less
than four percent of the total compute budget. Adding it has almost
no cost.

**It is the standard axis-2 benchmark in the field.** Subsequent
papers (Based, GLA, DeltaNet, BlackMamba, RWKV-7) all report MQAR
numbers because Zoology established it as the canonical synthetic
recall probe for efficient architectures. Reporting MQAR alongside
ASR puts our results into the standard comparison frame the
linear-time-RNN community already uses, and lets a reader directly
compare our mechanisms against, e.g., DeltaNet's published numbers
without us having to re-run their work.

In one sentence: ASR is where we discovered the mechanisms; MQAR is
where the axis-2 mechanism in particular gets a fair test. Together
they cover two of the five expressivity axes cleanly. State-tracking
(axis 3) is bracketed by theory in the writeup but left for future
work, since none of the three mechanisms target the diagonal-vs-DPLR
boundary.

That's the plan.
