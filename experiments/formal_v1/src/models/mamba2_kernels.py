"""Pure-PyTorch kernels for the Mamba-2 SSM, with LION bidirectional modes.

This module mirrors ``src/models/lion_attention.py``: stateless, parameter-free
functions that take (X, dt, A, B, C) and return Y. A block module (see
``mamba2_block.py``) owns the learnable parameters and dispatches on ``mode``.

Three kernels:

1. ``ssd_scan_causal``       — Mamba-2 SSD chunked selective scan (Dao & Gu 2024,
                               Listing 1 of the paper).  Carry-state capable.
2. ``ssd_scan_lion``         — LION full T×T bidirectional attention
                               (Afzal et al. 2025, Theorem 3.1, ``LION`` row
                               for Mamba-2 in Table 1).  O(T²d) memory.
3. ``ssd_scan_lion_chunk``   — LION chunkwise bidirectional (paper §3.3), same
                               asymptotic complexity as causal SSD.  Works by
                               running SSD forward + reversed; the two outputs
                               are summed with a diagonal correction.

Shape contract for all three (following ``ssd_minimal.py`` in the reference
repo):

    X     : (B, L, H, P)        per-head values (head-dim P = headdim)
    dt    : (B, L, H)           per-head softplus time-step
    A     : (H,)                per-head continuous A (negative)
    B     : (B, L, H, N)        per-head state-keys (already broadcast from
                                ngroups to nheads by the caller)
    C     : (B, L, H, N)        per-head state-queries
    state : (B, H, P, N) or None    (``recurrent`` only)

Returns Y of shape (B, L, H, P).  ``recurrent`` also returns a final state.

Numerical convention: matches the ``ssd_minimal_discrete`` reference — inside
each kernel we form the *discretised* A = dt * A_continuous and the
*discretised* X = dt * X, so the caller should pass undiscretised inputs.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import repeat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _segsum(a: torch.Tensor) -> torch.Tensor:
    """Stable lower-triangular segment-sum.

    For input shape ``(..., L)`` returns ``(..., L, L)`` where

        out[..., i, j] = sum_{t = j+1..i} a[..., t]        if  i >= j
        out[..., i, j] = -inf                              if  i <  j

    This matches the ``segsum`` helper in ``mamba_ssm/modules/ssd_minimal.py``.
    """
    L = a.size(-1)
    a = repeat(a, "... l -> ... l m", m=L)
    mask_strict = torch.tril(
        torch.ones(L, L, device=a.device, dtype=torch.bool), diagonal=-1
    )
    a = a.masked_fill(~mask_strict, 0.0)
    seg = torch.cumsum(a, dim=-2)
    mask_full = torch.tril(
        torch.ones(L, L, device=a.device, dtype=torch.bool), diagonal=0
    )
    seg = seg.masked_fill(~mask_full, float("-inf"))
    return seg


def _pad_to_multiple(x: torch.Tensor, block_len: int, dim: int = 1) -> Tuple[torch.Tensor, int]:
    """Pad ``x`` along ``dim`` with zeros so its size is a multiple of ``block_len``."""
    sz = x.size(dim)
    pad = (-sz) % block_len
    if pad == 0:
        return x, 0
    pad_spec = [0, 0] * (x.ndim - 1 - dim) + [0, pad] + [0, 0] * dim
    return F.pad(x, pad_spec), pad


# ---------------------------------------------------------------------------
# 1. Causal SSD scan (Dao & Gu 2024, Listing 1)
# ---------------------------------------------------------------------------

def ssd_scan_causal(
    X: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int = 64,
    state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mamba-2 SSD chunked selective scan.

    Numerically equivalent (up to float32 rounding) to
    ``mamba_ssm.modules.ssd_minimal.ssd_minimal_discrete`` with the
    ``X * dt, A * dt`` pre-multiplication baked in.

    Args:
        X:     (B, L, H, P)
        dt:    (B, L, H)       — post-softplus time-step
        A:     (H,)            — negative continuous A
        B, C:  (B, L, H, N)
        state: (B, H, P, N) or None — initial per-chunk SSM state
        chunk_size: int

    Returns:
        Y:           (B, L, H, P)
        final_state: (B, H, P, N)
    """
    Bsz, L, H, P = X.shape
    N = B.shape[-1]

    # Pad to multiple of chunk_size so we can `reshape` cleanly.
    X_p, pad = _pad_to_multiple(X, chunk_size, dim=1)
    dt_p, _ = _pad_to_multiple(dt, chunk_size, dim=1)
    B_p, _ = _pad_to_multiple(B, chunk_size, dim=1)
    C_p, _ = _pad_to_multiple(C, chunk_size, dim=1)
    L_p = X_p.size(1)
    nC = L_p // chunk_size

    # Discretised X and A.
    X_d = (X_p * dt_p.unsqueeze(-1)).float()               # (B, L, H, P)
    A_d = (dt_p * A.view(1, 1, H)).float()                 # (B, L, H)
    B_f = B_p.float()
    C_f = C_p.float()

    # Chunk the time axis.
    X_c = X_d.reshape(Bsz, nC, chunk_size, H, P)
    A_c = A_d.reshape(Bsz, nC, chunk_size, H)
    B_c = B_f.reshape(Bsz, nC, chunk_size, H, N)
    C_c = C_f.reshape(Bsz, nC, chunk_size, H, N)

    # (B, H, nC, L) — move H out for the segsum.
    A_cl = A_c.permute(0, 3, 1, 2)                         # (B, H, nC, L)
    A_cumsum = torch.cumsum(A_cl, dim=-1)                  # (B, H, nC, L)

    # ── 1. intra-chunk (diagonal blocks) ──────────────────────────────────
    L_mask = torch.exp(_segsum(A_cl))                      # (B, H, nC, L, L)
    Y_diag = torch.einsum(
        "bclhn,bcshn,bhcls,bcshp->bclhp",
        C_c, B_c, L_mask, X_c,
    )

    # ── 2. state produced by each chunk (right factor) ───────────────────
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)   # (B, H, nC, L)
    produced = torch.einsum(
        "bclhn,bhcl,bclhp->bchpn",
        B_c, decay_states, X_c,
    )  # (B, nC, H, P, N)

    # ── 3. inter-chunk propagation (left factor) ─────────────────────────
    if state is None:
        init = torch.zeros(Bsz, 1, H, P, N, device=X.device, dtype=torch.float32)
    else:
        init = state.float().unsqueeze(1)                  # (B, 1, H, P, N)
    produced_full = torch.cat([init, produced], dim=1)     # (B, nC+1, H, P, N)

    chunk_boundary = A_cumsum[..., -1]                     # (B, H, nC)
    decay_chunk = torch.exp(_segsum(F.pad(chunk_boundary, (1, 0))))  # (B, H, nC+1, nC+1)
    propagated = torch.einsum(
        "bhzc,bchpn->bzhpn",
        decay_chunk, produced_full,
    )                                                      # (B, nC+1, H, P, N)
    entering, final_state = propagated[:, :-1], propagated[:, -1]

    # ── 4. state → output per chunk (middle factor) ──────────────────────
    state_decay_out = torch.exp(A_cumsum)                  # (B, H, nC, L)
    Y_off = torch.einsum(
        "bclhn,bchpn,bhcl->bclhp",
        C_c, entering, state_decay_out,
    )

    Y = (Y_diag + Y_off).reshape(Bsz, L_p, H, P)
    if pad:
        Y = Y[:, :L]
    return Y.to(X.dtype), final_state.to(X.dtype)


# ---------------------------------------------------------------------------
# 2. LION full bidirectional attention  (paper §3.2, Theorem 3.1)
# ---------------------------------------------------------------------------

def ssd_scan_lion(
    X: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """LION bidirectional full-attention scan for Mamba-2.

    Translates Mamba-2 into LION-S form (per-head scalar decay) and applies
    the same parallel attention we use for RWKV-6 in ``lion_attention.py``:

        w[b, t, h]    = dt[b, t, h] * A[h]          (log decay, negative)
        cs[b, t, h]   = cumsum(w, dim=t)
        cs_b          = cs - w

        A_fwd = tril(  (C ⊙ exp(cs))  (B ⊙ exp(-cs))ᵀ  )
        A_bwd = triu(  (C ⊙ exp(-cs_b)) (B ⊙ exp(cs_b))ᵀ , diagonal=1 )
        Y     = (A_fwd + A_bwd) @ (dt · X)

    Forward pass includes the diagonal; backward is strictly-upper.

    Returns Y: (B, L, H, P).  No state (bidirectional attention sees the whole
    sequence; carry-state is not meaningful).
    """
    Bsz, L, H, P = X.shape

    # ── log-decay per (b, t, h), in log-space; cumulative forward / backward
    w = (dt.float() * A.view(1, 1, H).float())                 # (B, L, H), ≤ 0
    cs = torch.cumsum(w, dim=1)                                # (B, L, H)
    cs_b = cs - w

    # Midpoint shift for numerical stability — same trick as lion_attention.py.
    mid = L // 2
    shift_f = cs[:, mid:mid + 1, :]
    shift_b = cs_b[:, mid:mid + 1, :]
    cs_s = (cs - shift_f).clamp(-60, 60)
    cs_bs = (cs_b - shift_b).clamp(-60, 60)

    # Move H to dim 1 for per-head matmul.
    C_h = C.transpose(1, 2).float()                            # (B, H, L, N)
    B_h = B.transpose(1, 2).float()
    X_h = X.transpose(1, 2).float()                            # (B, H, L, P)
    dt_h = dt.transpose(1, 2).float()                          # (B, H, L)
    cs_sh = cs_s.transpose(1, 2)                               # (B, H, L)
    cs_bsh = cs_bs.transpose(1, 2)

    exp_cs = torch.exp(cs_sh).unsqueeze(-1)                    # (B, H, L, 1)
    exp_neg_cs = torch.exp(-cs_sh).unsqueeze(-1)
    exp_cs_b = torch.exp(cs_bsh).unsqueeze(-1)
    exp_neg_cs_b = torch.exp(-cs_bsh).unsqueeze(-1)

    # Forward (j ≤ i).
    A_fwd = torch.tril((C_h * exp_cs) @ (B_h * exp_neg_cs).transpose(-2, -1))
    # Backward (j > i).
    A_bwd = torch.triu(
        (C_h * exp_neg_cs_b) @ (B_h * exp_cs_b).transpose(-2, -1),
        diagonal=1,
    )

    # Values: V = dt * X  (discretised).
    V = dt_h.unsqueeze(-1) * X_h                               # (B, H, L, P)
    Y = (A_fwd + A_bwd) @ V                                    # (B, H, L, P)
    return Y.transpose(1, 2).to(X.dtype)                       # (B, L, H, P)


# ---------------------------------------------------------------------------
# 3. LION chunkwise bidirectional (paper §3.3)
# ---------------------------------------------------------------------------

def ssd_scan_lion_chunk(
    X: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    """LION chunkwise bidirectional scan.

    Implemented as::

        Y_F = ssd_scan_causal(X, dt, A, B, C)           # j ≤ i
        Y_B = flip( ssd_scan_causal(flip(X), flip(dt), A, flip(B), flip(C)) )   # j ≥ i
        Y   = Y_F + Y_B - diag   (avoid double-counting j = i)

    Same asymptotic complexity as the causal SSD (roughly 2× work).  Use
    ``ssd_scan_lion`` for short sequences (full-attention is faster when the
    T×T attention fits in memory); use this path when ``L²`` blows up.
    """
    Y_F, _ = ssd_scan_causal(X, dt, A, B, C, chunk_size=chunk_size)
    X_r = X.flip(1)
    dt_r = dt.flip(1)
    B_r = B.flip(1)
    C_r = C.flip(1)
    Y_B, _ = ssd_scan_causal(X_r, dt_r, A, B_r, C_r, chunk_size=chunk_size)
    Y_B = Y_B.flip(1)

    # Diagonal correction: at j = i, both forward and backward include the term
    #   (C[i] · B[i]) · dt[i] · X[i]  (decay at zero distance is 1).  Subtract
    # one copy so it is counted exactly once.
    diag_attn = (C.float() * B.float()).sum(-1)                # (B, L, H)
    diag_val = dt.float().unsqueeze(-1) * X.float()            # (B, L, H, P)
    Y_diag = (diag_attn.unsqueeze(-1) * diag_val).to(X.dtype)
    return Y_F + Y_B - Y_diag
