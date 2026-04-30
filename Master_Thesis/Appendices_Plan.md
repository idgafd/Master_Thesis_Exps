# Appendices Plan

> **Status (2026-04-30):** appendices A–E have been drafted from this
> plan; the LaTeX files live under `Master_Thesis/appendices/`. This
> document remains the section-by-section specification used as input
> for the drafting and is the place to update if a future revision
> changes scope. The drafted appendices already report every cell at
> measured values; no projection caveats remain.

Five appendices, total 22 to 33 pages. LaTeX files live under
`Master_Thesis/appendices/`.

| ID | Title | Length |
|---|---|---|
| A | Experimental Setup | 5 to 7 pages |
| B | Mechanism Implementation Details | 4 to 6 pages |
| C | Complete Empirical Results | 5 to 8 pages |
| D | Closed-Cell Engaged-Null Catalog | 5 to 7 pages |
| E | Supplementary Figures (optional) | 3 to 5 pages |

---

## Appendix A. Experimental Setup

### A.1 Training configuration

Source: `experiments/final/Master_Plan.md` §7 plus per-cell
`config.yaml`.

Tabular hyperparameters:

- Optimizer: AdamW, learning rate 3e-4, weight decay 0.01
- Scheduler: cosine with 1000-step warmup
- Epochs: 50; patience: 15
- Batch budget: max 300 s total duration
- Gradient clip: 5.0
- SpecAugment LD policy: freq mask 27, time mask 100, 2+2 masks
- Random seed: 42 (single-seed default)
- Mixed precision: fp32 baseline; bf16 only for selected cells

### A.2 ASR baseline architecture

Source: `experiments/formal_v1/src/models/encoder.py`.

- Audio frontend: log-mel spectrogram, 80 mel bins, 25 ms window,
  10 ms hop
- Subsampling: 2x before encoder
- Encoder backbone interface: per-backbone time-mix block plus
  channel-mix block plus residual + LayerNorm
- CTC head: linear projection to vocabulary, softmax, CTC loss
- Decoding: greedy CTC, no language model
- Per-backbone parameter counts at 7M and 14M

### A.3 Datasets

- LibriSpeech clean-100: corpus statistics, train / dev / test
  splits, vocabulary, audio preprocessing
- Common Voice EN 100h: subset construction
  (`100h_train_manifest_seed42.csv`, 64198 clips, 100.0005 hours,
  sha 23f70d1c), dev / test composition
- MQAR: synthetic data generator, sequence lengths
  `T in {64, 256, 1024}`, key-value pair count, vocabulary,
  splits

### A.4 Hardware and software environment

Source: per-cell `meta.json`.

- GPU: RTX PRO 6000 Blackwell, 48 GB VRAM
- GPU count per cell: single GPU baseline; 4-GPU DDP for the
  LA LION-S DHO cell
- Software stack: CUDA, PyTorch, Python versions per
  per-cell `meta.json`
- Repository commit hash for the locked codebase

### A.5 Reproducibility statement

- Single-seed convention; multi-seed reserved for BREAK-band
- Repository structure: `experiments/final/`,
  `experiments/formal_v1/`, `experiments/synthetics_v1/`,
  `papers/`
- Reference implementations adapted: see Section 2.8 of related
  work
- Per-cell artefacts under
  `experiments/final/outputs/<cell>_seed42/`: `config.yaml`,
  `best_model.pt`, `history.csv`, `results.json`,
  `metrics.jsonl`, `train.log`, `plots/`

---

## Appendix B. Mechanism Implementation Details

### B.1 Multi-Scale Depthwise Convolution

- Module location: `mechanisms/conv_shift.py`
- Constructor parameters: `kernel_size=3`, `dilations=(1, 2, 4, 8)`,
  `padding_mode="causal"`
- Per-channel learnable mixing weights
  `alpha_d in R^D` per layer
- Initialization: `alpha_{d=1}=1.0`, `alpha_{d>1}=0.01`;
  branch weights for `d>1` initialized as `N(0, 0.01^2)`;
  branch weight for `d=1` initialized to `[0.5, 0, 0.5]`
- Insertion point per backbone: RWKV-6 token-shift residual,
  Mamba-2 input side, LA pre-projection

### B.2 Damped Harmonic Oscillator

- Module locations: `linear_attn_rse.py`, `mamba2_rse.py`,
  `rwkv6_time_mix.py` time-mix fork
- Block-complex transition equation
- Depth-graded theta-clip schedule:
  `(pi/8, pi/8, pi/4, pi/4, pi/2, pi/2)` for 6 layers; analogous
  extension for 12 layers
- Depth-graded LoRA rank schedule:
  `(16, 16, 32, 32, 48, 48)` for 6 layers
- Viscosity coefficient `eta_{h, b}` with shape
  `(n_heads, n_blocks)`, zero-initialized
- Rayleigh dissipation form:
  `lambda_eff = lambda_raw + eta_{h, b} * theta^2`
- Per-backbone deployment notes

### B.3 Chunked Value Decorrelation

- Module locations: per-backbone dispatch
  (`rwkv6_time_mix._apply_lucid_recurrent`,
  `mamba2_kernels._apply_lucid_mamba2_chunked`,
  `lion_attention.lion_attention_with_lucid` for the LION encoder)
- Preconditioner equation:
  `P_c = exp(tau_h * [K_RN K_RN^T / sqrt(d) - sqrt(d)]) + epsilon I`
- Per-head learnable temperature
  `tau_h = softplus(tau_raw_h)`; init: `tau_raw = log(e - 1)`
  on RWKV-6, Mamba-2, LA-LION (so `tau_h = 1` at init);
  `tau_raw = 0` on LA causal (so `tau_h = ln 2` at init)
- Architecture-split `epsilon`:
  `epsilon = 1e-4` on Mamba-2 deployments,
  `epsilon = 1e-6` on RWKV-6 and LA deployments
- Chunk size `T_c = 64` across all backbones
- Mamba-2 uses C-correlation (query analog in SSD scalar-identity
  dual form)

### B.4 Unified LION wrapper

- Three per-backbone deployments share the LION parallel-form
  algebraic interface but use backbone-specific kernels:
  - RWKV-6: `RWKV6TimeMix(mode="lion")` dispatching into
    `lion_attention.lion_parallel_attention`
  - Mamba-2: `Mamba2Block(mode="lion")` dispatching into
    `mamba2_kernels.ssd_scan_lion`
  - LA: `LIONLinearAttentionEncoder(decay_mode in {"lit", "s"})`
- Decay-class mapping: RWKV-6 inherits LION-S, Mamba-2 inherits
  LION-S, LA defaults to LION-LIT with controlled LION-S
  deployment

---

## Appendix C. Complete Empirical Results

### C.1 Master matrix table

Source: per-cell `experiments/final/outputs/*/results.json` plus
`Master_Thesis/figures/chapter5/F1_transfer_pattern_matrix_data.csv`.

Columns:

- Cell directory name
- Architecture
- Mechanism column
- Mode (causal, LION-LIT, LION-S)
- Scale (7M, 14M)
- Best dev CER
- Test CER
- Delta test CER vs same-(arch, mode, scale) vanilla
- Parameter count
- Wall-clock training time

Sort by mode, then scale, then architecture. Apply DHO selection
rule: depth-graded preferred where present, strong-fallback
labelled DHO.

### C.2 Common Voice cross-distribution table

For 16 CV pilot cells: same column structure as C.1, plus a
column for `Delta_CV - Delta_LS` (delta-of-deltas measuring
task-prior modulation). Sort by `|Delta_CV - Delta_LS|`
descending.

### C.3 MQAR cohort full table

Source:
`experiments/synthetics_v1/outputs/cohort_reduced/_index.csv`.

Verdict (PASS / FAIL / PARTIAL / SKIP / OOM) per
(backbone, sequence length T) plus `steps_to_0.5` and
`steps_to_0.9` where defined. Replaces F3 figure.

### C.4 Chunked-streaming evaluation per cell

Source: per-cell `results.json::chunked`.

Six chunked-evaluation CER values
(2.0s reset / carry, 5.0s reset / carry, 10.0s reset / carry)
alongside full-utterance test CER. Sort by architecture and cell.

---

## Appendix D. Closed-Cell Engaged-Null Catalog

Source: `experiments/formal_v1/EXPRESSIVITY_AXES.md`,
`experiments/formal_v1/stages_2_9_summary.md`,
`experiments/formal_v1/STAGE10_SUMMARY.md`.

### D.1 Seven engaged-null mechanisms

Per entry: name, source paper or internal derivation, target
axis, cell that ran, dev / test CER, Delta vs vanilla, parameter
mobility evidence (the engaged signature), why classified as
engaged-null rather than failed.

1. Delta rule (Stage 8 T1): rank-1 Householder erasure on
   RWKV-6, axis 2
2. NCGRU-Cayley (Stage 10.5): Cayley-orthogonal transition,
   axis 2 + 3
3. M-squared RNN (Stage 10.2): non-linear state transition,
   axis 3
4. Non-normal RSE T2 / S9 (Stages 8 to 9): dense per-token
   polar, axis 2 + 3
5. Readout gauge A1' (Stage 7A): content-adaptive readout
   phase, axis 5
6. Content-conditional alpha_d (CB-3): per-token receptive
   field selection, axis 5
7. Mamba-2 novelty gate (Stage 11): structurally inert, scale
   mismatch

### D.2 Family-D quadratic-lift saturation cluster

Stage 6: hadamard n^2, qtail, qtail_lowrank, PoM v-lift saturate
at dev approximately 0.124. Axis-5 ceiling on the ASR spine.
Brief table: parametrisation, parameter count, dev / test CER.

### D.3 Discussion: axis-mismatch interpretation

Each engaged-null entry targets an axis (2, 3, or 5) that the
LibriSpeech ASR probe does not exercise. Parameter mobility
confirms engagement; absent CER reduction confirms the task did
not reward that function-class extension. The seven entries plus
Family-D cluster form the converse evidence for the
cross-experiment invariant articulated in Chapter 5 synthesis.

---

## Appendix E. Supplementary Figures (Optional)

### E.1 Rotation-angle distributions (deferred F9 sub-figure)

Per-layer histograms of trained `theta_{t, h, b}` for DHO cells.
Forward pass over evaluation data required.

### E.2 Per-cell training curves grid

Multi-page small-multiple covering all 50+ reported cells, if
rendered.

### E.3 Engagement classifier dashboard detail

Enlarged version of F12 showing every reported cell's
classification.

### E.4 Other deferred diagnostics

CVD preconditioner condition-number trajectories, per-block eta
correlation plots, etc., if produced.

---

## Order of writing

1. Appendix A: independent of main text completion; hyperparameter
   table machine-generable from `config.yaml`
2. Appendix B: mirrors Chapter 4 §4.2 to §4.4; write after the
   chapter's prose is final
3. Appendix C: pure aggregation from F1 data CSV plus per-cell
   `results.json`
4. Appendix D: requires careful citation work; higher-effort prose
5. Appendix E: conditional on figure landing

## Cross-references from main text

- Chapter 3 §3.5 references Appendix A for hyperparameters
- Chapter 5 prose references Appendix C for full per-cell tables
- Chapter 5 synthesis section references Appendix D for the
  cross-experiment invariant evidence
- Chapter 5 prose references Appendix E for deferred diagnostic
  figures

When the appendix LaTeX is written, replace placeholder text in
the main chapters with `\ref{app:experimental-setup}`, etc.
