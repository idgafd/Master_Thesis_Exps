"""Phase 2b — Independent-λ Paired-Pole RSE (P²-RSE-indepλ).

Extension of Stage-5 Phase-1 P²-RSE which used SHARED λ across the two
complex poles (with independent θ). STAGE5_RESULTS §6.4(1) diagnosed
shared-λ as the most likely reason P²-RSE under-performed Stage 4:
forcing both poles to the same bandwidth collapses half of the intended
expressivity, whereas real acoustic formants are conjugate pairs with
*distinct Q factors* (distinct λ). Phase 2b relaxes this constraint.

Mathematical spec
-----------------
Per 2×2 block b, time t, for each mode j ∈ {1, 2}:

    z_{t,b}^{(j)} = exp(-λ_{t,b}^{(j)} + i·θ_{t,b}^{(j)})     # two poles, independent λ and θ
    c_{t,b}^{(j)} = z_{t,b}^{(j)} · c_{t-1,b}^{(j)} + k_c_{t,b} · v_t
    y_t = β_{1,t} · Σ_b Re(r̄_c · c^{(1)}) + β_{2,t} · Σ_b Re(r̄_c · c^{(2)}) + u · k_t · v_t

with viscosity applied symmetrically to both poles:
    λ_{t,b}^{(j),eff} = -w_{t,b}^{(j),raw} + η_{h,b} · (θ_{t,b}^{(j)})²

Zero-regression-at-init contract
--------------------------------
* λ^(2) base initialised as a clone of λ^(1) base (same Stage-3 decay schedule).
* λ^(2) LoRA weights initialised to zero (no contribution at t=0).
* θ^(2) base initialised to -θ^(1) (phase-complementary — inherited from Phase 1).
* β mixer: N(0, 0.01) + zero bias → softmax gives ≈ (½, ½) at init.

At t=0 the model is exactly equivalent to Phase-1 P²-RSE (shared-λ, paired
θ). SGD then breaks the pole symmetry in two ways: phase-complementary θ
immediately differentiates the modes, and the new λ^(2) LoRA grows from
zero to give the two poles distinct bandwidths. If λ^(2) never learns,
this reduces exactly to Phase-1.

Speed
-----
Each pole's scan uses ``rse_scan_fast.rse_viscosity_scan`` — the
real-arithmetic reimplementation of ``_forward_recurrent_rse`` that splits
complex state into two real fp32 tensors so the hot matmuls hit Tensor
Cores. See ``src/models/rse_scan_fast.py`` for the full optimisation
rationale. Two independent scans still cost roughly 2× one pole's time;
the fast kernel recovers the ~2.5–3× penalty of the complex64 fallback
path that would otherwise apply to both scans.

See also
--------
* ``TODO_FUTURE_IDEAS.md §Stage 5 (deferred internally)`` — original spec
* ``memory/project_phase2b.md`` — user-flagged priority context
* ``STAGE5_RESULTS.md §6.4(1)`` — diagnosis that motivates this variant
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from src.models.rse_scan_fast import rse_viscosity_scan


def p2rse_indep_lambda_scan(
    r: torch.Tensor,                 # (B, H, T, K)
    k: torch.Tensor,                 # (B, H, T, K)  — pole 1 key (shared default)
    v: torch.Tensor,                 # (B, H, T, K)  — pole 1 value (shared default)
    w_1: torch.Tensor,               # (B, H, T, K) log-decay for pole 1 (real, ≤ 0)
    w_2: torch.Tensor,               # (B, H, T, K) log-decay for pole 2 (independent)
    theta_1: torch.Tensor,           # (B, H, T, Bk) rotation for pole 1
    theta_2: torch.Tensor,           # (B, H, T, Bk) rotation for pole 2
    beta_logits: torch.Tensor,       # (B, H, T, 2) mixer input (unscaled)
    mixer_type: str = "softmax",     # "linear" or "softmax"
    eta: Optional[torch.Tensor] = None,  # (H, Bk) viscosity coupling, applied to BOTH poles
    u: Optional[torch.Tensor] = None,    # (H, K) bonus term — applied once after mixing
    state: Optional[torch.Tensor] = None,  # (2, B, H, K, K) packed dual state
    # ── Phase 2b-ext: independent drive-side (k, v) per pole ──
    # If provided, pole 2 uses (k_2, v_2) instead of the shared (k, v).
    # Left None for the Phase 2b baseline which keeps k/v shared.
    k_2: Optional[torch.Tensor] = None,
    v_2: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Paired-pole RSE scan with INDEPENDENT λ per pole.

    Runs two independent fast-scan calls, one per pole, then mixes outputs
    via a data-dependent β. Each pole has its own (λ, θ) — independent
    decay *and* rotation. Viscosity is shared (same η for both poles) so
    that the Phase-3 zero-regression-at-init contract is preserved.

    Phase 2b-ext (optional): when ``k_2``/``v_2`` are provided, pole 2 uses
    its own drive-side projections instead of the shared ``k``/``v``.

    The bonus term ``u`` is applied **once** on the mixed output (matching
    Phase-1 P²-RSE convention — individual scans skip it to avoid
    double-counting). The bonus uses the shared k for consistency with the
    Phase-1 bonus semantics (u · k · v shortcut is a pole-0 term, not
    pole-specific).

    Returns:
        y:         (B, H, T, K) mixed output in r's dtype.
        new_state: (2, B, H, K, K) packed dual state for carry-state use.
    """
    B, H, T, K = r.shape

    # Split carry state if provided: (2, B, H, K, K) → two (B, H, K, K)
    if state is not None:
        s_1 = state[0].contiguous()
        s_2 = state[1].contiguous()
    else:
        s_1 = None
        s_2 = None

    # Default: pole 2 reuses shared k/v. When Phase 2b-ext is active, the
    # caller passes independent pole-2 drive tensors.
    k_pole_2 = k if k_2 is None else k_2
    v_pole_2 = v if v_2 is None else v_2

    # Each pole runs the fast real-arithmetic scan independently.
    # u=None for both inner calls — bonus applied once externally.
    y_1, s_1_out = rse_viscosity_scan(
        r, k, v, w_1, theta_1,
        eta=eta, u=None, state=s_1,
    )
    y_2, s_2_out = rse_viscosity_scan(
        r, k_pole_2, v_pole_2, w_2, theta_2,
        eta=eta, u=None, state=s_2,
    )

    # β mixer: (B, H, T, 2)
    if mixer_type == "softmax":
        beta = F.softmax(beta_logits, dim=-1)
    else:
        beta = beta_logits
    beta = beta.to(r.dtype)
    beta_1 = beta[..., 0:1]   # (B, H, T, 1) broadcasts across K
    beta_2 = beta[..., 1:2]
    y = beta_1 * y_1 + beta_2 * y_2     # (B, H, T, K)

    # Bonus term applied once post-mix — matches single-pole RSE semantics.
    if u is not None:
        u_view = u.view(1, H, 1, K).to(r.dtype)
        bonus = (r * u_view * k).sum(dim=-1, keepdim=True) * v
        y = y + bonus

    new_state = torch.stack([s_1_out, s_2_out], dim=0)
    return y, new_state
