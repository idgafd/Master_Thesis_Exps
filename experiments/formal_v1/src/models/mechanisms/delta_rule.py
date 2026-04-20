"""Delta Rule parameters and computation.

Selective erasure of stale key-value associations.
On LION: causal-only correction (no anticausal delta).
On recurrent RWKV-6: full state update with erasure.

Reference: RWKV-7 (Peng et al.), adapted for RWKV-6 architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaRuleParams(nn.Module):
    """Delta rule learnable parameters."""

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        d_lora: int = 64,
        warmstart: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_head = n_head
        self.head_size = head_size

        # Key normalization scaling
        self.k_k = nn.Parameter(torch.full((1, 1, hidden_size), 0.85, dtype=dtype))
        # Erasure strength (iclr in RWKV-7 notation)
        self.k_a = nn.Parameter(torch.ones(1, 1, hidden_size, dtype=dtype))

        # Data-dependent erasure via LoRA.
        # Default init (a0=1): iclr at t=0 = sigmoid(1)*1*2 ≈ 1.76.  With
        # β=1.76 the delta branch fires at full strength from step 0,
        # destroying the randomly-initialised state before useful
        # associations are written.  This is the failure mode diagnosed
        # in TODO_DELTA_RULE §5 H1 for prior delta runs.
        # Warmstart init (a0=-5): iclr ≈ sigmoid(-5)*1*2 ≈ 0.0134 at t=0,
        # so the delta branch is ~off initially and SGD grows the LoRA
        # contribution if useful.  Preserves zero-regression-at-init in
        # spirit (β ε-small) while allowing SGD to activate the mechanism.
        a0_init = -5.0 if warmstart else 1.0
        self.a0 = nn.Parameter(torch.full((1, 1, hidden_size), a0_init, dtype=dtype))
        self.a1 = nn.Parameter(torch.zeros(hidden_size, d_lora, dtype=dtype))
        self.a2 = nn.Parameter(
            torch.zeros(d_lora, hidden_size, dtype=dtype).uniform_(-0.01, 0.01)
        )

    def compute_kk_iclr(
        self,
        k: torch.Tensor,  # (B, H, T, K)
        B: int, T: int, H: int, K: int,
    ):
        """Compute normalized key (kk) and erasure strength (iclr).

        Returns kk, iclr both (B, H, T, K).
        """
        # k is already (B, H, T, K), need to go to (B, T, D) for LoRA
        k_flat = k.transpose(1, 2).reshape(B, T, H * K)

        # Key normalization
        kk = k_flat * self.k_k
        kk = F.normalize(kk, dim=-1)

        # Data-dependent erasure strength
        a = self.a0 + torch.tanh(k_flat @ self.a1) @ self.a2
        iclr = torch.sigmoid(a) * self.k_a * 2.0

        # Reshape to head form
        kk = kk.view(B, T, H, K).transpose(1, 2)
        iclr = iclr.view(B, T, H, K).transpose(1, 2)

        return kk, iclr
