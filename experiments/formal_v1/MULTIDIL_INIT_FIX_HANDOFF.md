# MultiDilationDWConvShift — Init-Gradient Trap: Fix & Rerun Handoff

**Audience:** standalone agent, no conversation context required.
**Scope:** fix one initialisation bug in one file, validate, rerun a
small set of affected experiments, update labels in result docs.
**Isolation rule:** **must not interfere with currently running
experiments** — see §2 before launching anything.

**Prerequisite reading** (in order; all in this same directory):
1. `CLAUDE.md` — project operating manual; methodology, zero-regression
   discipline, matched-epoch tracking, diagnostic-probe requirement.
2. `STAGE10_PLAN.md` — experimental queue and per-experiment specs.
3. `STAGE10_SUMMARY.md` — Stage 10 empirical summary.
4. `EXPRESSIVITY_AXES.md` — axis framework and paper index.

---

## 1. Mission summary

A gradient-trap bug in `src/models/mechanisms/conv_shift.py` class
`MultiDilationDWConvShift` causes all dilation branches with $d > 1$ to
receive **zero gradient** at initialisation, and they consequently
**never move from zero** across 30 epochs of training. This has been
verified across three architectures (RWKV-6, Mamba-2, Linear Attention)
by extracting $\alpha_d$ values directly from ep-30 checkpoints.

Empirical evidence at ep 30, across all affected runs:

| Architecture | layer 0 | layer 5 |
|---|---|---|
| RWKV-6 `multidil_sym` | α=[0.77, **0.0, 0.0, 0.0**] | α=[2.19, **0.0, 0.0, 0.0**] |
| Mamba-2 `multidil_sym` | α=[0.89, **0.0, 0.0, 0.0**] | α=[0.99, **0.0, 0.0, 0.0**] |
| LA `multidil_sym` | α=[1.17, **0.0, 0.0, 0.0**] | α=[1.47, **0.0, 0.0, 0.0**] |

The multi-dilation mechanism empirically reduces to single-dilation
($d=1$) DWConvShift with one learnable scalar per layer. The CER gains
already measured are real, but the attributed mechanism is wrong — they
reflect single-dilation ConvShift + per-layer scalar, not a multi-scale
receptive field.

**Your tasks:**
1. Implement the init fix in `conv_shift.py` (see §5).
2. Add/run unit tests verifying gradient flow (see §6).
3. Run one sanity training (1 full ep) to confirm $\alpha_{d>1}$ moves (§7).
4. If the sanity passes, run the full-30-ep rerun queue (§8).
5. Update result-table labels in docs per reporting discipline (§9).
6. Commit results under **new backbone names** (do not overwrite existing runs).

---

## 2. GPU isolation (read before touching anything)

There are ongoing long-running experiments on this machine. Before
doing any training, you must:

1. **Check GPU occupancy:** `nvidia-smi` and check which GPUs have
   running PyTorch processes.
2. **Identify the already-running experiments.** At the time of writing,
   these may still be in progress:
   - `outputs/linear_attn_rse_strong_viscosity_seed42/` (Stage 11.2b)
   - `outputs/mamba2_rse_strong_viscosity_seed42/` (Stage 11.2a)
   Check their `history.csv` to see whether they've reached ep 30 or
   are still training.
3. **Rules:**
   - If a GPU is running an experiment, **do not stop it**, do not
     reuse its GPU, do not touch its output directory.
   - You may use: a different GPU that is idle, or wait until a
     running experiment completes.
   - **Do not modify any file in `outputs/` except for runs you
     launched.**
   - **Do not modify source code other than
     `src/models/mechanisms/conv_shift.py`** (the single fix file)
     unless you need to add a unit test (see §6).

4. If both GPUs are occupied, **wait** for one to free up and document
   that you waited in your commit message. Do not force-share a GPU.

---

## 3. What the bug is (technical)

`MultiDilationDWConvShift.__init__` currently initialises:
- $\alpha$ parameter as `[1.0, 0.0, 0.0, 0.0]` for dilations `(1, 2, 4, 8)`
- Branch `d=1` weights as `[0.5, 0, 0.5]` per channel (symmetric averaging)
- Branches `d ∈ {2, 4, 8}` weights as **all zero**

At step 0, for any $d > 1$:
- Output contribution: $\alpha_d \cdot \text{DWC}_d(x) = 0 \cdot 0 = 0$
- Gradient w.r.t. $\alpha_d$:
  $\partial L / \partial \alpha_d = (\partial L / \partial y) \cdot \text{DWC}_d(x) = (\partial L / \partial y) \cdot 0 = 0$
- Gradient w.r.t. $\text{branch}_d.\text{weight}$: proportional to $\alpha_d = 0$

**Both the scalar and the branch weights have zero gradient when both
start at zero.** SGD cannot escape this configuration. It is a
multiplicative gradient trap — the mechanism is mathematically
decoupled from the training signal for all $d > 1$.

## 4. What it means to fix it

The fix must **break the zero-zero product** in the gradient. At least
one of $\alpha_d$ or $\text{branch}_d.\text{weight}$ must be non-zero
for $d > 1$ at init.

We want to preserve (approximately) the zero-regression property:
at init, the output should still match the single-dilation
`DWConvShift` within a small perturbation (≤10⁻⁴, well below
activation noise after the first forward pass).

## 5. The fix — exact code change

File: `src/models/mechanisms/conv_shift.py`
Class: `MultiDilationDWConvShift.__init__`

Replace the existing initialisation with the patch below. Changes
marked inline.

```python
class MultiDilationDWConvShift(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
        padding_mode: str = "causal",
    ):
        super().__init__()
        assert padding_mode in ("causal", "symmetric")
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dilations = tuple(int(d) for d in dilations)
        self.padding_mode = padding_mode

        self.branches = nn.ModuleList()
        for _ in self.dilations:
            branch = nn.Conv1d(
                d_model, d_model,
                kernel_size=kernel_size,
                padding=0,
                groups=d_model,
                bias=False,
            )
            self.branches.append(branch)

        # ───────── FIXED INIT (Option B, gradient-trap safe) ─────────
        #
        # Zero-regression intent: output at init should be within ~10⁻⁴
        # of single-dilation DWConvShift.
        #
        # Gradient-flow requirement: both α_d AND branch_d.weight must
        # be non-zero at init for d > 1, otherwise the product ∂L/∂α_d
        # and ∂L/∂branch_d.weight are both exactly zero (the trap).
        #
        # Solution:
        #   α_{d=1} = 1.0       (main branch)
        #   α_{d>1} = 0.01      (small, not zero — unblocks gradient
        #                        w.r.t. branch weights)
        #   branch_{d=1}.weight = [0.5, 0, 0.5] (or causal analog)
        #   branch_{d>1}.weight = N(0, 0.01)  (small, not zero —
        #                                      unblocks gradient w.r.t. α)
        #
        # Perturbation to output at init:
        #   Σ_{d>1} α_d · DWC_d(x) ≈ (3 branches) · 0.01 · O(0.01 · x)
        #                          ≈ 3 × 10⁻⁴ × x
        # Below typical activation noise.
        MAIN_DIL = 1 if 1 in self.dilations else self.dilations[0]
        NON_MAIN_ALPHA = 0.01
        NON_MAIN_WEIGHT_STD = 0.01

        alphas = torch.zeros(len(self.dilations))
        for i, d in enumerate(self.dilations):
            alphas[i] = 1.0 if d == MAIN_DIL else NON_MAIN_ALPHA
        self.alpha = nn.Parameter(alphas)

        with torch.no_grad():
            for i, branch in enumerate(self.branches):
                d = self.dilations[i]
                if d == MAIN_DIL:
                    branch.weight.zero_()
                    if kernel_size == 3:
                        if padding_mode == "symmetric":
                            branch.weight[:, 0, 0] = 0.5
                            branch.weight[:, 0, -1] = 0.5
                        else:  # causal
                            branch.weight[:, 0, 1] = 0.5
                            branch.weight[:, 0, 2] = 0.5
                else:
                    nn.init.normal_(
                        branch.weight, mean=0.0, std=NON_MAIN_WEIGHT_STD
                    )
        # ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward unchanged from existing implementation
        ...
```

**Do not change `forward()`.** The fix is init-only.

**Do not change `DWConvShift` (the single-dilation class).** It's not
affected by this bug and its results are clean.

**Do not change any other mechanism file.** This bug is local to
`MultiDilationDWConvShift`.

## 6. Unit test (add and run before any training)

Add this test to `tests/test_mechanisms.py` (create the file if it
doesn't exist). This must pass before you run any training.

```python
import torch
import torch.nn as nn
from src.models.mechanisms.conv_shift import MultiDilationDWConvShift


def test_multidil_init_gradient_flow_symmetric():
    """Verify that gradient reaches α_d and branch weights for all
    d > 1 at init. Regression test for the pre-fix init-trap bug."""
    d_model = 64
    mod = MultiDilationDWConvShift(
        d_model=d_model,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        padding_mode="symmetric",
    )

    x = torch.randn(2, 32, d_model, requires_grad=False)
    y = mod(x).sum()
    y.backward()

    # α_{d=1} must have non-zero gradient (already worked pre-fix)
    assert mod.alpha.grad is not None
    assert mod.alpha.grad[0].abs().item() > 1e-8, \
        f"alpha[d=1] gradient is zero: {mod.alpha.grad}"

    # α_{d>1} must have non-zero gradient (was zero pre-fix)
    for i, d in enumerate(mod.dilations):
        if d == 1:
            continue
        g = mod.alpha.grad[i].abs().item()
        assert g > 1e-8, \
            f"alpha[d={d}] gradient is zero after fix: {mod.alpha.grad}"

    # Branch weights for d > 1 must have non-zero gradient (was zero pre-fix)
    for i, d in enumerate(mod.dilations):
        if d == 1:
            continue
        g = mod.branches[i].weight.grad.norm().item()
        assert g > 1e-8, \
            f"branch[d={d}] weight gradient is zero after fix"


def test_multidil_init_causal_gradient_flow():
    """Same test for causal padding mode."""
    mod = MultiDilationDWConvShift(
        d_model=64, dilations=(1, 2, 4, 8), padding_mode="causal"
    )
    x = torch.randn(2, 32, 64)
    mod(x).sum().backward()
    for i, d in enumerate(mod.dilations):
        if d == 1:
            continue
        assert mod.alpha.grad[i].abs().item() > 1e-8
        assert mod.branches[i].weight.grad.norm().item() > 1e-8


def test_multidil_init_zero_regression_approximate():
    """At init, output should be within 1e-3 of single-dilation output
    (softer than bit-exact zero-regression; explicit in the fix spec)."""
    from src.models.mechanisms.conv_shift import DWConvShift

    d_model = 64
    torch.manual_seed(0)
    multi = MultiDilationDWConvShift(
        d_model=d_model, dilations=(1, 2, 4, 8), padding_mode="symmetric"
    )
    torch.manual_seed(0)
    single = DWConvShift(d_model=d_model, kernel_size=3)

    x = torch.randn(2, 32, d_model)
    with torch.no_grad():
        diff = (multi(x) - single(x)).abs().max().item()
    # Expect ~3 × NON_MAIN_ALPHA × NON_MAIN_WEIGHT_STD × ||x||
    # ≈ 3 × 0.01 × 0.01 × O(1) ≈ 3e-4, well below 1e-3.
    assert diff < 1e-3, f"init perturbation too large: {diff}"
```

Run:
```bash
cd experiments/formal_v1
uv run pytest tests/test_mechanisms.py -v
```

All three tests must pass. If any fails, do not proceed to training —
debug the init code first.

## 7. Sanity training run (1 epoch, ~1-2 minutes)

Before committing to a full 30-ep run, verify α actually moves:

```bash
# Use an idle GPU; see §2 first
uv run scripts/run_experiment.py \
    --config configs/default.yaml \
    --backbone rwkv6_convshift_multidil_symmetric \
    --epochs 1 \
    --seed 42 \
    --output-dir outputs/_multidil_init_fix_sanity_ep1 \
    --gpu <idle_gpu_number>
```

Then extract α at the end-of-epoch-1 checkpoint:

```bash
uv run python <<'EOF'
import torch
sd = torch.load('outputs/_multidil_init_fix_sanity_ep1/checkpoint_ep1.pt',
                map_location='cpu', weights_only=False)
model_sd = sd.get('model', sd.get('state_dict', sd))
for k, v in model_sd.items():
    if k.endswith('.alpha') and 'conv_shift_module' in k:
        print(f"{k}: {v.tolist()}")
EOF
```

**Pass criterion:** for at least some layer, at least one of
$\alpha_{d=2}, \alpha_{d=4}, \alpha_{d=8}$ has moved from 0.01 by more
than 0.001 in absolute value. If every $\alpha_{d>1}$ is still stuck at
essentially 0.01, the fix is incomplete — debug before running 30 ep.

## 8. Rerun queue

Do not overwrite existing runs. Use new backbone names with a `_v2`
suffix that makes the fix explicit in the output path.

### 8.1 Priority order

Run the single-architecture baseline first to confirm the fix changes
anything at 30 ep. Only continue to compositions / Stage-11 variants
if that single baseline shows meaningful movement.

| Priority | Backbone | Existing (broken-init) run | New run output dir |
|---|---|---|---|
| **1** | `rwkv6_convshift_multidil_symmetric_v2` | `outputs/rwkv6_convshift_multidil_symmetric_seed42/` (keep) | `outputs/rwkv6_convshift_multidil_symmetric_v2_seed42/` |
| 2 | `mamba2_convshift_multidil_symmetric_v2` | `outputs/mamba2_convshift_multidil_symmetric_seed42/` (keep) | `outputs/mamba2_convshift_multidil_symmetric_v2_seed42/` |
| 3 | `linear_attn_convshift_multidil_symmetric_v2` | `outputs/linear_attn_convshift_multidil_symmetric_seed42/` (keep) | `outputs/linear_attn_convshift_multidil_symmetric_v2_seed42/` |
| 4 | `rwkv6_rse_convshift_multidil_symmetric_v2` (CB-1 variant) | existing CB-1 dir (keep) | new `_v2` dir |
| 5 | `rwkv6_convshift_multidil_symmetric_gated_v2` (CB-3 variant) | existing CB-3 dir (keep) | new `_v2` dir |
| 6 | `rwkv6_qtail_lowrank_all_convshift_multidil_symmetric_v2` (CB-7 variant) | existing CB-7 dir (keep) | new `_v2` dir |

### 8.2 Decision branches

After Priority-1 completes (30 ep):

- **If $\alpha_{d>1}$ values at ep 30 are still near zero (< 0.1 at all layers and all d > 1):**
  The multi-dilation hypothesis is *structurally rejected* on RWKV-6 —
  even with the gradient trap removed, SGD does not find use for wider
  dilations. Stop the rerun queue at Priority 1. Document as:
  "multi-dilation mechanism does not engage on ASR even when branches
  are gradient-reachable."

- **If $\alpha_{d>1}$ values at ep 30 are meaningfully non-zero (>0.1 at
  some layer for some $d > 1$) AND CER improves by ≥ 1σ (~0.0014) vs
  broken-init run:**
  Multi-dilation genuinely works when unblocked. Continue to
  Priorities 2–6 to re-test transfers and compositions.

- **If $\alpha_{d>1}$ moves but CER does not improve:**
  Mechanism engages, doesn't convert to CER. New engaged-null data
  point. Continue through Priority 3 (the Stage-11 transfers) to check
  whether the non-engagement is architecture-independent.

### 8.3 Implementation of backbone-name `_v2` dispatch

Add to `src/models/encoder.py`:
1. Register new backbone names in `mode_map` (e.g.,
   `"rwkv6_convshift_multidil_symmetric_v2": "recurrent"`).
2. Make sure the substring dispatch for
   `use_conv_shift_multidilation` still triggers (it does — `multidil`
   substring is still present).
3. The `_v2` suffix is purely for output-directory naming and result
   accounting; the code path in `MultiDilationDWConvShift` is the same
   (it reads the fixed init).

Alternative (cleaner): add a boolean flag `conv_shift_multidil_fixed_init:
bool = True` to the config, and have `encoder.py` pass it through to
`MultiDilationDWConvShift`. Default `True` after merging the fix. Keep
the `_v2` naming in output dirs for clarity.

## 9. Reporting discipline

### 9.1 Do not overwrite existing results

The broken-init runs are an accurate record of "what single-dilation
ConvShift + per-layer scalar achieves." They remain valuable and
should stay on disk and in `RESULTS.md` / `STAGE10_SUMMARY.md` /
`STAGE10_PLAN.md` §9.2 master matrix.

### 9.2 Update labels in existing result tables

In the three thesis-facing docs (`STAGE10_SUMMARY.md`,
`STAGE10_PLAN.md`, `EXPRESSIVITY_AXES.md`), add a footnote / caveat
wherever "multidil_sym" or "multi-dilation ConvShift" appears as the
mechanism description. Suggested footnote text:

> *Empirical note (init-gradient trap): in the `multidil_symmetric`
> runs logged below, the dilation branches $d \in \{2, 4, 8\}$ were
> verified at ep 30 to have $\alpha_d = 0$ exactly, due to a
> zero-at-init gradient trap in `MultiDilationDWConvShift.__init__`.
> The effective mechanism in these runs is therefore single-dilation
> symmetric DWConvShift (kernel=3, init `[0.5, 0, 0.5]`) plus one
> learnable scalar per layer. See `MULTIDIL_INIT_FIX_HANDOFF.md` for
> the fix specification. `_v2` rows below reflect the fixed-init
> reruns.*

Add one row per `_v2` backbone once the reruns complete. Do not
delete or overwrite the original rows.

### 9.3 Axis-1 framing in EXPRESSIVITY_AXES.md

The Paper 7 entry in §Axis 1 currently attributes the win to
"multi-dilation DWConvShift" (the Paper 7 mechanism framing). This
needs a caveat that the multi-dilation aspect was inactive in our
runs; the empirical mechanism is single-dilation DWC + per-layer
scalar, which aligns with LiT's single-kernel DWC-on-V and
Vision-RWKV's fixed Q-Shift rather than Paper 7's multi-dilation
specifically.

If the `_v2` reruns show multi-dilation actually engaging, add a
second row with the corrected attribution. If the `_v2` reruns show
$\alpha_{d>1}$ still inert, close the multi-dilation-on-ASR line and
note that Paper 7's multi-dilation claim doesn't replicate on this
task / regime.

## 10. Acceptance criteria

The fix is accepted when all of these pass:

1. All three unit tests in §6 pass: `uv run pytest tests/test_mechanisms.py -v`
2. Sanity 1-ep run (§7) shows at least one $\alpha_{d>1}$ value has
   moved from 0.01 by >0.001 at some layer.
3. Priority-1 30-ep run completes cleanly (no NaN, no
   training-dynamics regression vs broken-init run within matched-epoch
   15 halt criterion per `STAGE10_PLAN.md` §8.1).
4. Ep-30 $\alpha$ extraction shows a distribution of $\alpha_{d>1}$
   values across layers (the numbers can be zero or non-zero, but they
   must not all be identically 0.01 — i.e., SGD must have moved them).
5. Commit the code fix + unit tests + updated docs (with caveats) +
   any `_v2` result directory in a single clean commit.
6. Push.

Escalation: if Priority-1 run trains but the ep-30 $\alpha_{d>1}$
distribution is suspicious (e.g., all exactly 0.01 throughout, or
diverging to extreme values > 10), stop and leave a detailed `train.log`
analysis in the run directory; do not proceed to Priorities 2–6 until a
human reviews.

## 11. Don't-do list

- Don't touch any file in `outputs/` that belongs to a currently-running
  experiment (see §2).
- Don't rename existing backbones (e.g., retroactively renaming
  `rwkv6_convshift_multidil_symmetric` to something else) in
  the dispatch logic. Only add new `_v2` names.
- Don't modify other mechanism files (`delta_rule.py`, `lucid.py`,
  `headscale.py`, `temperature.py`, RSE code in `rwkv6_time_mix.py`).
  They're not affected by this bug.
- Don't change `DWConvShift` (the single-dilation class). It's
  correct.
- Don't remove the existing broken-init runs from result tables.
  They're the authoritative record of "what single-dilation + per-layer
  scalar achieves."
- Don't run CB-2 (`multidil_wide4` / `multidil_dense` per
  `STAGE10_PLAN.md` §6 CB-2) until this fix is in place. CB-2 was
  pre-registered before the bug was known; running it on broken-init
  code would produce null-by-construction results, not evidence.

## 12. References

- `CLAUDE.md` — operating manual.
- `STAGE10_PLAN.md` §6 CB-2 — the wide4/dense spec that was unexecutable
  pre-fix.
- `STAGE10_SUMMARY.md` §1, §2 — where the mechanism attribution needs
  updating.
- `EXPRESSIVITY_AXES.md` §Axis 1 — Paper 7 entry needs caveat / update.
- `src/models/mechanisms/conv_shift.py` — the file to change.
- Source of empirical evidence for the bug:
  `outputs/rwkv6_convshift_multidil_symmetric_seed42/checkpoint_ep30.pt`,
  `outputs/mamba2_convshift_multidil_symmetric_seed42/checkpoint_ep30.pt`,
  `outputs/linear_attn_convshift_multidil_symmetric_seed42/checkpoint_ep30.pt`
  — α tensors are at `encoder.layers.{L}.[att|mamba.conv1d|premix].alpha` (path
  varies by architecture; load state dict and grep for `.alpha`).

---

*End of handoff. Execute the sequence §5 → §6 → §7 → §8 → §9 → §10 in
order. If any step fails the pass criterion, stop and escalate before
continuing.*
