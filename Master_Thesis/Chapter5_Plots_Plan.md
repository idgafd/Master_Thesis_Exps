# Chapter 5 — Plots and Figures Plan

*Working document for the empirical chapter. Inventories the data
available on disk, lists every figure the chapter needs, and
specifies for each figure: what it shows, why the thesis needs it,
the data source, suggested visualisation, and effort estimate.
Used for plot preparation; not part of the LaTeX deliverable.*

*Created 2026-04-29.*

> **Status (2026-04-30):** figures F1–F13 are built under
> `Master_Thesis/figures/chapter5/`. Each figure has its `.pdf`,
> `.png`, `_data.csv`, and `_script.py` companion; the locked palette
> and typography live in `figures/chapter5/_style.py`. The figure
> data files include the LA LION-S × DHO cell at test 0.0988 and
> all close-out cells at their measured values. Chapter 5 prose is
> the remaining open item; this plan stays as the figure inventory
> the prose draws on.

---

## 1. Available data on disk

### Per-cell artefacts (every reported cell has these)

For each cell directory under
`experiments/final/outputs/<exp_name>_seed42/`:

| File | Content | Used for |
|---|---|---|
| `best_model.pt` | Model weights at best dev CER | Post-hoc parameter analyses (Section~4 below) |
| `history.csv` | Per-epoch dev/test metrics (CER, WER, loss, grad norms, time) | Training curves, convergence comparisons |
| `metrics.jsonl` | Fine-grained per-step metrics | Step-level analysis if needed |
| `results.json` | Final eval block: `test.cer/wer/loss`, `chunked` (2s/5s/10s reset/carry CER), `history`, `params`, `config_snapshot`, `git_sha` | Headline numbers, chunked-streaming bars |
| `config.yaml` | Full training config | Replay verification, hyperparameter table |
| `plots/` | Pre-rendered: `cer_curve.png`, `loss_curve.png`, `grad_norm.png`, `lr_schedule.png` | Per-cell training curves (already present) |
| `train.log` | Training stdout/stderr | Debug only |

### Cohort indices

| File | Cells | Modality |
|---|---|---|
| `experiments/final/outputs/` | 50+ LibriSpeech cells (7M and 14M, causal and LION, individual mechanisms and compositions) | ASR |
| `experiments/final/outputs/cv_pilot_*` | 16 cells (Common Voice EN 100h, 7M causal only) | ASR cross-distribution |
| `experiments/synthetics_v1/outputs/cohort_reduced/_index.csv` | $T \in \{64, 256, 1024\}$ × multiple backbones, PASS/FAIL + `steps_to_0.5` / `steps_to_0.9` | MQAR axis-2 probe |

### Mechanism-specific parameters (extractable from `best_model.pt`)

| Mechanism | Parameter to extract | Per-cell shape | Use |
|---|---|---|---|
| MSDC | $\alpha_{d}$ mixing weights, $W_{d}$ conv kernels | `(n_layers, |\mathcal{D}|, D)` for $\alpha$; `(n_layers, |\mathcal{D}|, D, k)` for $W$ | F8 below |
| DHO | $\eta_{h, b}$ viscosity, $\theta$ LoRA matrices, $\lambda_{\mathrm{raw}, h, b}$ | `(n_layers, n_heads, K/2)` for $\eta$; LoRA shapes per layer | F9 below |
| CVD | $\tau_{h}$ per-head temperature | `(n_layers, n_heads)` | F10 below |

No `diagnostics_ep*.json` files exist on disk; the discovery-phase
diagnostic apparatus described in `Master_Plan.md` §13 was not
deployed. All mechanism-specific diagnostic plots in §4 below
require a post-hoc extraction script that loads `best_model.pt`,
locates the relevant parameter tensor by name, and reports its
trained values.

---

## 2. Headline figures (must-have for the thesis)

These are the figures Chapter 5 cannot land without. They support
the thesis's central claims directly: matrix-as-artefact,
deficit-proportional ordering, axis-2 separation by MQAR.

### F1 — Transfer-pattern matrix (the central artefact)

**What it shows.** Test CER (or normalised $\Delta$ vs vanilla) for
every cell of the experimental matrix, organised so that the three
mechanism axes, three architectures, two modes, and two scales are
all visible in one figure.

**Why we need it.** This is the single image readers will remember.
It is the visual form of the thesis's "matrix-as-artefact" framing.
Every claim about mechanism-prior alignment is read off it.

**Data source.** `results.json::test.cer` for every cell under
`experiments/final/outputs/`. Vanilla baselines per (architecture,
mode, scale) for the $\Delta$ computation.

**Suggested visualisation.** Faceted heatmap, one facet per
(mode, scale): rows = architecture, columns = mechanism cell.
Cell colour encodes $\Delta$ vs vanilla (diverging colormap, blue
= helpful, red = harmful, white = neutral). Annotate each cell
with both the absolute test CER and the $\Delta$.

Alternative: grouped horizontal bars, one bar group per
(architecture, mode, scale), bars within group = mechanisms.
Cleaner numerically but less compact.

**Effort.** Low. Pure aggregation from `results.json` files.
Roughly 60 numbers total. Recommend producing both heatmap and
bar versions; pick the one that lays out best on the page.

### F2 — Deficit-proportional ordering, per mechanism

**What it shows.** For each of the three mechanisms (MSDC, DHO,
CVD), the $\Delta$ test CER ordering across the three architectures.
The ordering is the empirical signature the thesis claims is
predictable.

**Why we need it.** The "deficit-proportional ordering" claim is
central to the synthesis chapter. A single grouped-bar figure
makes the ordering obvious at a glance.

**Data source.** Same as F1, but reduced to three small bars per
panel.

**Suggested visualisation.** Three small subplots in a row, one per
mechanism. Each subplot has three bars (LA, RWKV-6, Mamba-2) of
$\Delta$ test CER vs vanilla at 7M causal. Annotated with the
ordering pattern: MSDC universal (deficit-proportional), DHO
BREAK-NULL on LA-Mamba-2, CVD asymmetric.

**Effort.** Low. Same data as F1, simpler aggregation.

### F3 — MQAR axis-2 separation grid

**What it shows.** PASS/FAIL on MQAR for every cell of the synthetic
matrix at three sequence lengths. Specifically the headline finding
that the causal Transformer fails at $T = 1024$ while linear-time
backbones with MSDC or CVD pass.

**Why we need it.** This is the cleanest single empirical
demonstration of the axis-2 mechanism transfer. It is the
strongest single inversion-of-the-standard-narrative result in the
thesis. Without it, the axis-2 claim is undertested.

**Data source.**
`experiments/synthetics_v1/outputs/cohort_reduced/_index.csv`.

**Suggested visualisation.** Heatmap or grid: rows = backbone
(transformer_causal, mamba2, rwkv6, linear_attn, plus their MSDC
and CVD variants), columns = $T \in \{64, 256, 1024\}$. Cell colour
binary (green PASS, red FAIL, grey SKIP/OOM). Annotate cells with
the `steps_to_0.9` value where available.

Alternative compact form: three columns at $T = 64, 256, 1024$,
each column a small bar chart of `steps_to_0.9` per backbone, with
$\infty$ bars for FAIL.

**Effort.** Low. Index CSV is already in the right shape.

---

## 3. Supporting figures (load-bearing for chapter claims)

### F4 — Cross-distribution validation (LibriSpeech vs Common Voice)

**What it shows.** For each (architecture, mechanism) cell that
exists on both LibriSpeech and Common Voice, a scatter plot of
$\Delta$ on LibriSpeech vs $\Delta$ on Common Voice. Diagonal =
consistent transfer; off-diagonal = task-prior modulated.

**Why we need it.** Defends the matrix against the "ASR-specific"
attack. Specifically highlights the Mamba-2 $\times$ DHO-depth
divergence (NULL on LibriSpeech, helpful on CV) as evidence for
task-prior modulation of the deficit.

**Data source.** `cv_pilot_*/results.json` for CV side, the
matching `7m_*_seed42/results.json` for LibriSpeech side.

**Suggested visualisation.** Single scatter, $x = \Delta_{\text{LibriSpeech}}$,
$y = \Delta_{\text{CV}}$, one point per cell. Diagonal reference
line; quadrant labels. Annotate the Mamba-2 DHO-depth point and
the LA RSE-strong point as anchors. Use colour for architecture,
shape for mechanism.

**Effort.** Low. 16 CV cells $\times$ matching LibriSpeech cells.

### F5 — Cross-scale persistence (7M vs 14M)

**What it shows.** For every cell that exists at both scales, a
paired bar showing 7M test CER and 14M test CER side by side.
Identifies POS-scaling cells (Mamba-2 vanilla), NEG-scaling cells
(RWKV-6 vanilla), and mechanism-Δ growth across scale.

**Why we need it.** Supports the cross-scale persistence claim of
the thesis; also supports the discussion paragraph on
architecture-specific compute efficiency at matched 50-ep budget.

**Data source.** `7m_*` and `14m_*` cells under
`experiments/final/outputs/`.

**Suggested visualisation.** Grouped bar chart: x-axis = cell
identity (architecture × mechanism), bars within group = scale.
Or paired-line chart connecting 7M to 14M values per cell, with
slope encoding the scaling direction.

**Effort.** Low. Direct aggregation.

### F6 — Composition saturation (P7, P8)

**What it shows.** For RWKV-6, Mamba-2, and LA at 7M and 14M, the
test CER of vanilla, single best mechanism, pairwise composition,
and (where available) triple composition. The expectation is that
compositions saturate to the single-mechanism level on RWKV-6 (the
P8 saturation result).

**Why we need it.** The same-axis composition saturation is
$\sim$3 of the 8 thesis-stake findings (per `Master_Plan.md` §20).
Needs visual evidence.

**Data source.** Cells named `*_p7_*`, `*_lucid_chunked_*`,
`*_multidil_v2_*`, `*_lucid_chunked_x_multidil_*` etc.

**Suggested visualisation.** Stepped bar chart per architecture:
vanilla → +mech1 → +mech2 → composition. Marks saturation when
the slope between bars 3 and 4 is near zero.

**Effort.** Medium. Requires identifying the composition cells
correctly per architecture.

### F7 — Causal vs LION mode comparison

**What it shows.** For each (architecture, mechanism) pair, the
gap between causal mode and LION mode test CER. Quantifies the
bidirectional benefit per mechanism per architecture.

**Why we need it.** Supports the "bidirectional benefit per
mechanism" claim. Also exposes the LION-LIT vs LION-S falsification
on Linear Attention.

**Data source.** Pairs of `7m_<arch>_causal_<mech>_seed42` and
`7m_<arch>_lion_<mech>_seed42` cells.

**Suggested visualisation.** Two-column grouped bars: causal vs
LION test CER, one group per (architecture, mechanism). LA gets a
third bar for LION-S to show the falsification spread relative to
LION-LIT.

**Effort.** Low.

---

## 4. Diagnostic figures (per-mechanism trained-parameter views)

These figures require a post-hoc extraction script that loads
`best_model.pt` for each cell and extracts the mechanism-specific
parameter tensors. They are load-bearing for the
"engaged $\times$ helpful" classifier and for the per-mechanism
discussion in Chapter 5.

### F8 — MSDC trained mixing weights $\alpha_{d}$

**What it shows.** For each (architecture, scale) where MSDC is
deployed, the trained $\alpha_{d}$ values across layers and
dilations. Distinguishes "MSDC actively used the multi-dilation
structure" (broad $\alpha$ distribution) from "MSDC collapsed to a
single dilation branch" (sparse $\alpha$).

**Why we need it.** Direct evidence for the MSDC "engaged" diagnostic
in the four-cell classifier. Also supports the design-notes claim
that the init-fix matters: trained $\alpha_{d > 1}$ values diverging
from their init demonstrates the gradient trap is broken.

**Data source.** `best_model.pt::*conv_shift_module.alpha`
or analogous, depending on backbone naming.

**Suggested visualisation.** Heatmap per architecture: rows =
layers, columns = dilations $\{1, 2, 4, 8\}$, cell value =
trained $\alpha_{d, \ell}$ (per-channel mean). Three heatmaps in a
row (RWKV-6, Mamba-2, LA), at 7M scale. Optional second row for
14M.

**Effort.** Medium. Requires knowledge of state-dict key names.
Cross-reference `experiments/formal_v1/src/models/mechanisms/conv_shift.py`
for parameter names.

### F9 — DHO trained $\eta_{h, b}$ and $\theta$ distributions

**What it shows.** Two sub-figures per architecture-scale cell:

(a) Trained $\eta_{h, b}$ heatmap (rows = layer, columns = block
or head$\times$block), distinguishing "viscosity engaged" from
"viscosity stayed at zero-init".

(b) Distribution of trained rotation angles $\theta_{t, h, b}$
across the eval set, one histogram per layer. Distinguishes
"layer reaches its $\theta$-clip" (saturated near boundary) from
"layer operates near-identity" (concentrated near zero).

**Why we need it.** Direct evidence for the DHO "engaged" diagnostic
in the four-cell classifier. The depth-graded $\theta$-clip
schedule is the thesis's design choice, and the histogram form
demonstrates whether the choice was structurally meaningful (early
layers near zero, deep layers saturating).

**Data source.** `best_model.pt::*viscosity_eta` for $\eta$, plus
a forward pass over the eval set to extract per-step $\theta$
samples for the histograms.

**Suggested visualisation.** Two-row figure per architecture-scale.
First row: $\eta_{h, b}$ heatmap. Second row: per-layer $\theta$
histograms (small multiples).

**Effort.** Medium-high. The $\eta$ heatmap is straightforward;
the $\theta$ histograms require running a forward pass on a sample
of eval data and collecting the projection outputs through the LoRA.

### F10 — CVD trained per-head temperature $\tau_{h}$

**What it shows.** For each (architecture, scale, mode) cell where
CVD is deployed, the trained $\tau_{h}$ values per layer per head.
Distinguishes "preconditioner engaged" (trained $\tau_{h}$ moves
substantially from init) from "preconditioner near-identity"
($\tau_{h}$ stayed at init).

**Why we need it.** Direct evidence for the CVD "engaged" diagnostic.
Also supports the architecture-split $\varepsilon$ rationale: the
chunk-local Gram conditioning becomes ill-conditioned at
$\tau_{h} \approx 1.5$ on Mamba-2, and showing the trained
$\tau_{h}$ values demonstrates whether the matrix is operating in
that regime.

**Data source.** `best_model.pt::*lucid_temperature` for the raw
parameter, then $\tau_{h} = \mathrm{softplus}(\tau_{\mathrm{raw}, h})$.

**Suggested visualisation.** Heatmap per architecture: rows =
layers, columns = heads, cell value = trained $\tau_{h}$. Three
panels (RWKV-6, Mamba-2, LA), 7M and 14M scales.

**Effort.** Low-medium. Pure parameter extraction plus softplus.

---

## 5. Optional / supplementary figures

### F11 — Per-cell training curves consolidated grid

**What it shows.** A grid of training-CER curves, one curve per
cell, organised so that mechanism-mediated convergence speed-ups
are visible.

**Why we need it.** Probably not needed as a standalone figure;
the per-cell `plots/cer_curve.png` already exists. Could go in
the appendix or be omitted.

**Data source.** Per-cell `history.csv`.

**Effort.** Low (data already plotted).

### F12 — Four-cell engagement classifier dashboard

**What it shows.** Every cell of the matrix classified into one of
four cells: engaged-and-helpful, engaged-without-gain, fail-to-engage,
or the residual cell. Implements the classifier defined in
`Master_Plan.md` §13.

**Why we need it.** Frames the closed-cell engaged-null catalogue
as the converse evidence for mechanism-prior alignment. Visually
separates productive mechanisms from absorbed mechanisms.

**Data source.** Combination of $\Delta$ test CER (from results.json)
and parameter-mobility scalars (from F8, F9, F10).

**Suggested visualisation.** Two-by-two grid populated with cell
chips for every reported run; chips coloured by architecture or
mechanism.

**Effort.** High. Requires combining parameter-mobility extractions
from F8, F9, F10 with metric outcomes. Defer to late writing if
short on time.

### F13 — Chunked-streaming evaluation

**What it shows.** For each cell, the test CER under chunked
streaming evaluation at 2s, 5s, 10s reset and carry, compared with
the full-utterance test CER. Shows whether the trained model
generalises to streaming inference.

**Why we need it.** Optional. The chunked evaluation is reported
in `results.json::chunked`, so the data is present, but chunked
streaming is not a load-bearing claim of the thesis. Nice-to-have
for completeness.

**Data source.** `results.json::chunked.{2.0s,5.0s,10.0s}_{reset,carry}.cer`.

**Suggested visualisation.** Small per-architecture grouped-bar
plot.

**Effort.** Low if included.

---

## 6. Summary and priority recommendation

| Priority | Figure | Effort | Required for landing the chapter |
|---|---|---|---|
| 1 | F1 transfer-pattern matrix | Low | yes |
| 1 | F2 deficit-proportional ordering | Low | yes |
| 1 | F3 MQAR axis-2 grid | Low | yes |
| 2 | F4 LibriSpeech vs CV scatter | Low | yes |
| 2 | F5 cross-scale persistence | Low | yes |
| 2 | F7 causal vs LION mode | Low | yes |
| 2 | F6 composition saturation | Medium | yes |
| 3 | F10 CVD $\tau_{h}$ heatmaps | Low-medium | yes (engagement classifier) |
| 3 | F8 MSDC $\alpha_{d}$ heatmaps | Medium | yes (engagement classifier) |
| 3 | F9 DHO $\eta$ + $\theta$ figures | Medium-high | yes (engagement classifier) |
| 4 | F12 four-cell engagement dashboard | High | nice-to-have |
| 4 | F13 chunked streaming | Low | optional |
| 4 | F11 training-curve grid | Low | optional |

Recommended order of preparation:

1. F1, F2, F3 first — these are pure aggregation from
   `results.json` and the MQAR cohort index.
2. F4, F5, F7 next — same kind of aggregation, slightly more
   alignment work between paired cells.
3. F6 — composition cells need correct identification.
4. F10, F8, F9 — require the post-hoc extraction script for
   `best_model.pt`. Recommend writing one shared utility that
   loads any cell's `best_model.pt`, locates the mechanism-specific
   parameter tensors by name, and returns them as numpy arrays.
   Then F10 is one call, F8 one call, F9 two calls.
5. F12 last — depends on F8-F10 outputs and adds the four-cell
   logic on top.
6. F11, F13 if time permits.

---

## 7. Common conventions across all figures

- Three architectures consistently coloured: pick three accessible
  colours and reuse.
- Mechanism shapes/markers consistent: pick three markers and
  reuse.
- $\Delta$ vs vanilla always reported relative to the corresponding
  (architecture, mode, scale) vanilla baseline.
- All $\Delta$ values reported on test CER (not dev), unless
  otherwise stated in the figure caption.
- Causal mode and LION mode colour-distinguished in figures that
  cross modes (F7 in particular).
- 7M and 14M scale shape-distinguished in figures that cross scales
  (F5 in particular).
- Vanilla baselines highlighted with a contrasting marker or
  greyscale fill so they are immediately identifiable.

---

## 8. Post-hoc extraction script (requirement for F8-F10)

A single shared utility is recommended rather than three separate
scripts. Sketch:

```python
# experiments/final/extract_mechanism_params.py
def extract_for_cell(cell_dir):
    # Load best_model.pt
    # Identify the backbone family from config.yaml::backbone
    # Walk the state-dict keys and locate:
    #   - alpha_d (MSDC)
    #   - viscosity_eta, theta_lora_*, theta_clip (DHO)
    #   - lucid_temperature (CVD)
    # Return dict of named numpy arrays.
```

Each F8-F10 figure consumes the dict and renders the relevant
view. Writing this utility is one engineering task that unlocks
three figures.

---

*End Chapter 5 plots plan v1 (2026-04-29). Adjust as data
preparation progresses.*
