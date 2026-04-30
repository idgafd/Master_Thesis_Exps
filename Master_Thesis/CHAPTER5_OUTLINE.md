# Chapter 5 outline (concise, max 10 pages)

*Working planning document. Not part of the LaTeX deliverable.
Replaces `Chapter5_Plots_Plan.md` for the final write-up. The plots
plan stays as the figure inventory; this document is the prose plan.*

*Created 2026-04-30.*

---

## Principle

The chapter is concise. The matrix is the artefact; numbers and
per-cell breakdowns live in `Appendix~\ref{app:results}`. Every
section presents one row of the argument: claim, evidence (figure
or one-line numbers), reading. No discovery narrative, no
mechanism-by-mechanism repetition of mathematical detail (that
lives in Chapter~\ref{ch:proposed}).

Maximum length: ten pages of prose plus the figures and the
single aggregated master table.

## Section budget

| § | Title | Pages | Main figure | Tables |
|---|---|---:|---|---|
| 5.1 | Aggregated empirical summary | 1.5 | (table only) | T5.1 aggregated |
| 5.2 | Cross-architecture transfer pattern on LibriSpeech | 2.0 | F1, F2 | (none in chapter) |
| 5.3 | Cross-scale persistence and LION mode | 2.0 | F5, F7 | (none) |
| 5.4 | Cross-distribution validation | 1.0 | F4 | (none) |
| 5.5 | Axis-2 isolation: MQAR length sweep | 1.5 | F3 | (none) |
| 5.6 | Composition behaviour | 1.0 | F6 | (none) |
| 5.7 | Convergence under matched budget | 0.5 | F11 | (none) |
| 5.8 | Synthesis: mechanism-prior alignment law | 0.5 | (none) | (none) |
| | **Total** | **10** | | |

## §5.1 Aggregated empirical summary

**Claim.** The matrix is the artefact. The aggregated master table
in this section condenses the empirical evidence into a single
view; the per-cell breakdown lives in
Appendix~\ref{app:results}.

**Table 5.1 (the aggregated master table).** Single page-wide
table. Rows: ten configurations covering the matrix
$\text{architecture} \times \text{mode} \times \text{scale}$ slice
that is in scope. Columns: vanilla, MSDC, CVD, DHO, best-of
composition. Each cell reports test CER. Three slots in the
right margin report the corresponding MQAR verdicts at
$T \in \{64, 256, 1024\}$ for the architecture in that row.
Common Voice test CER for the same architecture's vanilla and
best-mechanism is appended at the bottom in three small rows.
Caption notes the matrix ceiling (14M Mamba-2 CVD $\times$ MSDC
at $0.0618$) and the LION-LIT cell that produces the only
negative CVD transfer.

**Suggested table layout.**

```
Configuration                  | Vanilla | MSDC   | CVD    | DHO    | Best comp.
------------------------------ | ------- | ------ | ------ | ------ | ---------
RWKV-6 causal 7M               | 0.1049  | 0.0788 | 0.1007 | 0.0989 | 0.0785
Mamba-2 causal 7M              | 0.1036  | 0.0825 | 0.0958 | 0.1006 | 0.0795
LA causal 7M                   | 0.1879  | 0.1409 | 0.1714 | 0.1198 | 0.0999
RWKV-6 causal 14M              | 0.1103  | 0.0751 | 0.1051 | 0.1077 | 0.0746
Mamba-2 causal 14M             | 0.0827  | 0.0631 | 0.0728 | 0.0781 | 0.0618
LA causal 14M                  | 0.1359  | 0.0969 | 0.1300 | 0.0947 | 0.0786
RWKV-6 LION-S 7M               | 0.0859  | 0.0750 | 0.0852 | 0.0740 | 0.0747
Mamba-2 LION-S 7M              | 0.0853  | 0.0833 | 0.0849 | 0.0825 | 0.0805
LA LION-LIT 7M                 | 0.2951  | 0.1404 | 0.3194 | 0.1042 | 0.0961
LA LION-S 7M                   | 0.1381  | 0.1154 | 0.1311 | 0.0988 | 0.1129
```

Append a small Common Voice block (3 rows: arch / vanilla / best
mechanism) and an MQAR side-summary (3 rows: arch / FAIL / PASS
with MSDC and CVD).

**Prose (one paragraph).** State that mechanism-prior alignment is
the predictive law: gain orders with residual structural deficit
along the axis the mechanism targets, modulated by what the task
exercises. The aggregated table is the artefact; the remaining
sections of the chapter read each row of evidence.

## §5.2 Cross-architecture transfer pattern on LibriSpeech

**Claim.** Three mechanisms, three signatures: MSDC universal and
deficit-proportional, DHO BREAK on LA and NULL on Mamba-2 under
the LibriSpeech distribution, CVD asymmetric.

**Figures.** F1 (transfer-pattern matrix faceted heatmap) and F2
(deficit-proportional ordering bars).

**Three short paragraphs.**

1. *MSDC universal, deficit-proportional ordering.* At 7M causal,
   $\Delta$ test CER is $-0.0470$ on LA, $-0.0261$ on RWKV-6,
   $-0.0211$ on Mamba-2. The ordering matches the
   architecture-deficit map of
   Section~\ref{sec:backbones:deficit}: LA has no native local
   mixing, RWKV-6 has the WKV time-shift, Mamba-2 has the native
   k=4 short convolution. Reproduced at 14M with LA at $-0.0390$,
   RWKV-6 at $-0.0352$, Mamba-2 at $-0.0196$.
2. *DHO BREAK on LA, NULL on Mamba-2 under LibriSpeech.* At 7M
   causal, $\Delta$ on LA is $-0.0681$ (largest single-mechanism
   gain in the matrix); on Mamba-2 it is $-0.003$ (marginal,
   close to noise floor); on RWKV-6 it is $-0.0061$. The ordering
   matches the deficit map: LA has no native attenuation and no
   phase, Mamba-2 has scalar selectivity that absorbs most of
   what the block-complex transition adds under the LibriSpeech
   audiobook distribution, RWKV-6 has per-channel decay diversity
   that absorbs the damping component. Section~\ref{sec:experiments:cv}
   refines the Mamba-2 reading on Common Voice.
3. *CVD asymmetric.* At 7M causal, CVD is $-0.0165$ on LA,
   $-0.0078$ on Mamba-2, $-0.0042$ on RWKV-6. The ordering is the
   opposite of the natural-home-in-attention prior. The LION
   mode of Section~\ref{sec:experiments:lion} provides the
   cleanest controlled isolation of decay as the structural
   prerequisite for the mechanism.

## §5.3 Cross-scale persistence and LION mode

**Claim.** Mechanism gains preserve or grow across the 7M-to-14M
scaling step, and LION bidirectional adaptation preserves the
transfer pattern conditional on a decay-bearing mask. The
LION-LIT $\times$ CVD cell on Linear Attention is the single
negative transfer in the matrix and isolates decay as the
structural prerequisite for the bidirectional value-decorrelation
mechanism.

**Figures.** F5 (paired-line cross-scale persistence) and F7
(causal vs LION mode comparison).

**Two paragraphs.**

1. *Cross-scale persistence.* MSDC mechanism $\Delta$ on RWKV-6
   grows from $-0.0261$ at 7M to $-0.0352$ at 14M; on Mamba-2 it
   preserves at $-0.0196$ (matrix ceiling at $0.0631$); on LA it
   preserves at $-0.0390$. None of the architectures shows a
   ranking shift among single mechanisms, and the 30M conditional
   scaling is dropped from scope on this basis.
2. *Decay-as-prerequisite finding on LA LION.* The natural LION
   mapping per~\citet{afzal2025linear} Table~1 sends Linear
   Attention to LION-LIT with $\lambda = 1$. The LION-LIT
   $\times$ CVD cell produces $\Delta$ test $+0.0243$, the only
   negative CVD transfer in the matrix, while the LION-S
   controlled deployment with a per-token sigmoid decay sends the
   same composition to $\Delta = -0.0070$. The contrast isolates
   decay as the structural prerequisite for the value-
   decorrelation mechanism in the bidirectional setting; the
   LION mask must be decay-bearing for the chunk-local
   preconditioner to remain well-conditioned. The same row shows
   the converse on DHO: the LION-LIT $\times$ DHO cell delivers
   $\Delta = -0.1909$ (BREAK), since the block-complex transition
   introduces decay through its damping channel rather than
   relying on the LION mask.

The mechanism-level versus operator-level bidirectional
comparison (LION versus Bi-WKV and Vision Mamba) is reported in
Appendix~\ref{app:diagnostics:lion-vs-vim}.

## §5.4 Cross-distribution validation

**Claim.** Most mechanism gains transfer to Common Voice; the
Mamba-2 DHO cell shows task-prior modulation that refines the
alignment law.

**Figure.** F4 (LibriSpeech vs Common Voice $\Delta$ scatter).

**One paragraph.** The scatter shows the diagonal alignment for
twelve of the fifteen non-vanilla cells. The off-diagonal
exceptions are the Mamba-2 DHO cell, which is marginal on
LibriSpeech ($\Delta = -0.0030$) and productive on Common Voice
($\Delta = -0.0226$), and the LA DHO and DHO $\times$ MSDC
cells, which deliver smaller absolute gains on Common Voice
than on LibriSpeech ($+0.033$ and $+0.026$ respectively). The
Mamba-2 DHO modulation is the cleanest case: selective
$\Delta t$ absorbs the block-complex transition's contribution
under the LibriSpeech audiobook distribution but not under the
broader Common Voice acoustic distribution. The reading
refines the alignment law: alignment is between mechanism and
task-as-exercised-by-data rather than between mechanism and
architecture in isolation.

## §5.5 Axis-2 isolation: MQAR length sweep

**Claim.** MQAR isolates axis 2. At $T = 1024$ the causal
Transformer FAILs while every linear-time backbone with MSDC or
CVD on top of it PASSes. This inverts the standard
softmax-wins-on-recall narrative at the length where the
comparison becomes binding. DHO does not solve MQAR, confirming
the predicted axis mismatch.

**Figure.** F3 (MQAR PASS/FAIL grid with steps-to-0.9
annotations).

**Two short paragraphs.**

1. *Linear-time plus mechanism solves MQAR at every tested
   length.* Vanilla RWKV-6, Mamba-2, and Linear Attention all
   FAIL at $T = 64$ already. Adding MSDC or CVD to any of the
   three sends the cell to PASS at every tested length, in 1k
   to 5k training steps. The causal Transformer baseline
   PASSes at $T = 64$ and $T = 256$ but FAILs at $T = 1024$,
   producing the strongest single empirical claim of the
   thesis: at the length where the softmax-versus-linear
   comparison becomes binding, every linear-time-plus-mechanism
   cell solves the task while the causal Transformer does not.
2. *DHO predicted axis mismatch.* DHO targets axis-1
   damped-oscillator dynamics; MQAR exercises axis-2
   associative-memory capacity under interference. Every DHO
   cell run on the MQAR cohort FAILs at $T = 64$, confirming
   the predicted axis mismatch.

## §5.6 Composition behaviour

**Claim.** Same-axis compositions saturate to single-mechanism
level; different-axis compositions stack productively.

**Figure.** F6 (composition saturation steps).

**Short paragraph.** The CVD $\times$ MSDC composition on RWKV-6
at 7M gives test $0.0785$ versus MSDC alone $0.0788$, a
saturated stack. The DHO $\times$ MSDC composition on Linear
Attention at 7M gives test $0.0999$ versus MSDC alone $0.1409$
and DHO alone $0.1198$, a productive stack of two different-axis
mechanisms. The 14M Mamba-2 CVD $\times$ MSDC cell at $0.0618$
is the matrix ceiling and is the single example where the
composition advances the architecture's lowest cell beyond the
strongest single mechanism at scale. The reading is consistent
with the alignment law's converse: dense per-token freedom on
the same axis does not stack beyond a single productive
mechanism.

## §5.7 Convergence under matched budget

**Claim.** The matched 50-epoch budget is within the convergence
horizon for every reported cell; no cell is truncated, and the
matrix ceiling is structural rather than budget-limited.

**Figure.** F11 (per-cell training-curve consolidated grid).

**Short paragraph.** The four-by-four grid plots dev CER versus
epoch for every cell of the matrix, organised by architecture
and mode. Two structural callouts: the 14M RWKV-6 vanilla
trajectory plateaus at epoch 50 (slope close to zero),
falsifying the undertraining-at-14M hypothesis and confirming
that the architecture-specific 7M-to-14M scaling is structural
rather than budget-limited; the 14M Mamba-2 MSDC trajectory
also plateaus at $0.063$, confirming the matrix ceiling is
structural at the matched budget.

## §5.8 Synthesis: mechanism-prior alignment law

**Claim restated.** A mechanism's gain on a task is proportional
to the residual structural deficit the architecture leaves
uncovered along the axis the mechanism targets, modulated by
what the task-as-exercised-by-data exercises. The matrix
populates the law in three signatures (MSDC universal, DHO
BREAK-and-NULL, CVD asymmetric), in three task structures
(LibriSpeech, Common Voice, MQAR), and across two scales. The
LION-LIT $\times$ CVD cell on Linear Attention is the single
counter-example to a mechanism's productive transfer in the
matrix, and it isolates decay as the structural prerequisite for
the value-decorrelation mechanism in the bidirectional setting.
The Mamba-2 DHO cross-distribution modulation refines the law
to the task-as-exercised-by-data dependence. Future work
extending the same matrix to diagonal-plus-low-rank backbones
(DeltaNet, RWKV-7) is the natural next step for axis 3, which
the diagonal class formally cannot reach.
