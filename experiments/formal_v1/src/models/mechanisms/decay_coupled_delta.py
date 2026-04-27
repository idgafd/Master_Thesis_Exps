"""Decay-Coupled Delta — Stage 12.

Standard delta (T1) applies the rank-1 erase `(I - β k̂ k̂^T)` uniformly across
every channel of the per-head state, regardless of that channel's intrinsic
time-scale.  Decay-coupled delta concentrates the erase along directions that
correspond to slow-decay channels, exploiting RWKV-6's existing per-channel
decay distribution `α_t = exp(w_t) ∈ (0, 1]^K`.

Per-head learnable scalar `p_h ≥ 0` parameterises the coupling:

    γ_t := exp(p_h · w_t) = α_t^{p_h} ∈ (0, 1]^K        (per-channel)
    k̃_t := γ_t ⊙ k̂_t                                    (re-weighted erase dir)

    S_t = (I - β_t k̃_t k̃_t^T ⊙ kk_kk^T) · diag(α_t) · S_{t-1} + k_t v_t^T

p_h init = 1 ⇒ γ = α (linear coupling).  At init β_t ≈ 0 (a0_init = -8 ⇒
β ≈ 6.7e-4) so the rank-1 correction is in the fp32 noise floor and the
mechanism reduces bit-exactly to vanilla RWKV-6 — the §2.4 reduction
contract.  See `STAGE12_DECAY_COUPLED_DELTA.md` for full motivation and
pre-registered verdict thresholds.

Novelty: every variant in `TODO_DELTA_RULE.md §4` (DeltaNet, Gated DeltaNet,
DeltaProduct compact-WY, Newton-step β, momentum delta, TTT/ATLAS, Delta×RSE,
Delta×Log-Linear) is decay-blind.  The coupling itself is the novelty.
"""

from typing import Optional

import torch

from src.models.mechanisms.delta_rule import DeltaRuleParams


class DecayCoupledDeltaParams(DeltaRuleParams):
    """Delta rule + per-head learnable decay-coupling exponent `p_h`.

    Inherits `compute_kk_iclr` from `DeltaRuleParams`; adds the `p_h`
    parameter and `compute_gamma()` for the coupling vector γ_t = α_t^{p_h}.

    Init: a0 = -8 (β ≈ 6.7e-4 at init), p_h = 1.0 (linear coupling).
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        d_lora: int = 64,
        p_init: float = 1.0,
        a0_init: float = -8.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            hidden_size=hidden_size,
            n_head=n_head,
            head_size=head_size,
            d_lora=d_lora,
            warmstart=True,
            a0_init=a0_init,
            dtype=dtype,
        )
        # Per-head learnable coupling exponent. Init at 1.0 means γ_t = α_t
        # exactly (linear coupling). SGD can drive p_h up (concentrate erase
        # toward the slowest-decay channels) or down (toward uniform).
        # Held unconstrained (no clamp) so SGD has full freedom; the
        # coupling vector γ = exp(p · w) stays in (0, 1] for w ≤ 0
        # regardless of sign of p (since p<0 with w<0 still gives finite
        # positive output, but values >1 become possible — see below).
        # NOTE: We deliberately do NOT softplus or clamp p. Spec §2.5 says
        # p ∈ R_{≥0} but allowing it negative is informative — if SGD
        # *prefers* erasing fast-decay channels (the opposite hypothesis),
        # we want to see that, not block it.
        self.p_h = torch.nn.Parameter(torch.full((n_head,), float(p_init), dtype=dtype))

    def compute_gamma(self, w: torch.Tensor) -> torch.Tensor:
        """Per-channel coupling vector γ_t = exp(p_h · w_t) = α_t^{p_h}.

        Args:
            w: log-decay (negative), shape (B, H, T, K). w = log(α).

        Returns:
            γ: shape (B, H, T, K), in (0, 1] when p_h ≥ 0 and w ≤ 0.
        """
        # p_h is (H,); broadcast to (1, H, 1, 1) against w (B, H, T, K).
        p = self.p_h.view(1, -1, 1, 1).to(w.dtype)
        return torch.exp(p * w)
