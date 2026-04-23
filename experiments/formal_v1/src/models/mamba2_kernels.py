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


def _compute_novelty_gates(
    B_c: torch.Tensor,                   # (B, nC, Tc, H, N)
    A_cumsum: torch.Tensor,              # (B, H, nC, Tc)
    gamma: torch.Tensor,                 # (H,) — softplus'd, non-negative
    sigma_init: Optional[torch.Tensor],  # (B, H, N, N) or None
    eps_reg: float = 1e-3,
    eps_num: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-chunk write-novelty gate.

    Gate formula (per token):

        q_t^2 = B_t^T · Σ_c^{-1} · B_t
        ω_t   = 1 / (1 + γ_h / (q_t^2 + eps_num))

    Σ_c is the running key-covariance, updated ONCE PER CHUNK at the
    chunk boundary using the (batched) chunk-cumulative decay
    ᾱ_{c-1} = exp(Σ_{l=0..Tc-1} A_d[c-1, l]):

        Σ_c = ᾱ_{c-1} · Σ_{c-1} + B_{c-1}^T B_{c-1}

    Σ_0 is taken from ``sigma_init`` if provided (carry-state across
    segments), else the zero matrix — regularised at inversion time by
    ``eps_reg · I``.

    Returns:
        omega:        (B, nC, Tc, H) — per-token, per-head gate in (0, 1].
        sigma_final:  (B, H, N, N)   — Σ after the last chunk, for carry.
    """
    Bsz, nC, Tc, H, N = B_c.shape
    device = B_c.device
    dtype = B_c.dtype

    if sigma_init is None:
        sigma = torch.zeros(Bsz, H, N, N, device=device, dtype=dtype)
    else:
        sigma = sigma_init.to(dtype)

    eye = torch.eye(N, device=device, dtype=dtype).view(1, 1, N, N)
    gamma_h = gamma.view(1, 1, H).to(dtype)                       # (1, 1, H)

    omega_chunks = []
    for c in range(nC):
        # Regularised inverse of Σ_c (distinct per (b, h)).
        sigma_reg = sigma + eps_reg * eye
        sigma_inv = torch.linalg.inv(sigma_reg)                   # (B, H, N, N)

        B_here = B_c[:, c]                                        # (B, Tc, H, N)
        # q² = Σ_{n,m} B[n] · Σ_inv[n, m] · B[m]
        q2 = torch.einsum(
            "bthn,bhnm,bthm->bth", B_here, sigma_inv, B_here
        )                                                         # (B, Tc, H)
        omega_c = 1.0 / (1.0 + gamma_h / (q2 + eps_num))          # (B, Tc, H)
        omega_chunks.append(omega_c)

        # Σ update for chunk c+1.
        alpha_bar = torch.exp(A_cumsum[:, :, c, -1])              # (B, H)
        BtB = torch.einsum("bthn,bthm->bhnm", B_here, B_here)     # (B, H, N, N)
        sigma = alpha_bar.view(Bsz, H, 1, 1) * sigma + BtB

    omega = torch.stack(omega_chunks, dim=1)                      # (B, nC, Tc, H)
    return omega, sigma


def _apply_lucid_mamba2_chunked(
    X_c: torch.Tensor,        # (B, nC, T_c, H, P)
    key_c: torch.Tensor,      # (B, nC, T_c, H, N) — either B_c or C_c
    lucid_temp: torch.Tensor, # (H,) — already softplus'd per-head temperature
) -> torch.Tensor:
    """LUCID preconditioner on Mamba-2 SSD dual-form chunks.

    ``key_c`` is the key-source: either B (the key-analog in Mamba-2's
    C·B^T dual-form attention — the default) or C (the query-analog —
    an analytical variant).  Per chunk, per head:

        k_rn = sqrt(N) * normalize(key_c, dim=-1)   # RMS-normalised
        G    = k_rn @ k_rn^T                        # (T_c, T_c) Gram
        P    = exp(tau * (G/sqrt(N) - sqrt(N)))     # unit-diagonal
                                                    # + eps*I (numerical)
        X_c_precond = solve(P, X_c)                 # T_c x T_c solve

    Returns X_c_precond with the same shape as X_c.

    Faithful to the algebraic form of the original LUCID (``Y = A · P^{-1} · V``
    via value-side preconditioning) and to RWKV-6's ``_apply_lucid_recurrent``
    implementation, adapted to Mamba-2's (B, L, H, N) key tensor layout.
    """
    Bsz, nC, Tc, H, P = X_c.shape
    N = key_c.shape[-1]
    sqrt_d = N ** 0.5

    # Reshape for batched matmul / solve: move H inside, put T_c as the
    # "matrix" dim.  (B, nC, H, T_c, N) for key_c; (B, nC, H, T_c, P) for X_c.
    key_perm = key_c.permute(0, 1, 3, 2, 4).contiguous()   # (B, nC, H, T_c, N)
    X_perm = X_c.permute(0, 1, 3, 2, 4).contiguous()       # (B, nC, H, T_c, P)

    # RMS-normalised keys.  F.normalize along the feature dim; sqrt(N) scaling
    # gives diag(G) = N by construction.
    B_rn = sqrt_d * F.normalize(key_perm, dim=-1)          # (B, nC, H, T_c, N)

    # Gram matrix over the chunk.
    gram = torch.matmul(B_rn, B_rn.transpose(-2, -1))      # (B, nC, H, T_c, T_c)

    # Broadcast tau over (B, nC, Tc, Tc).
    tau = lucid_temp.view(1, 1, H, 1, 1).to(gram.dtype)

    # Unit-diagonal preconditioner: exp(tau * (G/sqrt(N) - sqrt(N))).
    # Clamp for numerical safety (matches the RWKV-6 LUCID implementation).
    scaled = (tau * (gram / sqrt_d - sqrt_d)).clamp(-30, 30)
    Pm = torch.exp(scaled)

    # eps*I regulariser — ensures invertibility under heavy tau.  Bumped
    # to 1e-4 from 1e-6 after observing singular-solve during chunked eval
    # on a trained mamba2_lucid: highly similar B vectors within a chunk
    # produce near-identical rows of P, and 1e-6 was too small to keep
    # the system invertible post-training (tau reaches ~1.5).  1e-4 is
    # still orders of magnitude below typical gram magnitude (~N=64), so
    # the preconditioner's character is preserved.
    eye = torch.eye(Tc, device=Pm.device, dtype=Pm.dtype)
    Pm = Pm + 1e-4 * eye

    # Solve P · Y = X for Y.  torch.linalg.solve treats the last two dims
    # as the system.  X_perm is (..., T_c, P_val) — T_c matches P's rows.
    X_precond = torch.linalg.solve(Pm, X_perm)             # (B, nC, H, T_c, P)

    # Permute back to (B, nC, T_c, H, P).
    return X_precond.permute(0, 1, 3, 2, 4).contiguous()


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
    lucid_temp: Optional[torch.Tensor] = None,
    lucid_key_source: str = "B",
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
        lucid_temp: Optional (H,) — per-head LUCID temperature.  When
            provided, applies the LUCID preconditioner on X_d within
            each chunk, using B- (default) or C-correlation as the
            key Gram matrix.  See ``_apply_lucid_mamba2_chunked`` for
            the math.
        lucid_key_source: "B" or "C".  Which of B (key-analog, default)
            or C (query-analog) to use as the correlation source.

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

    # ── Optional: LUCID preconditioner on X_c using B-correlation ───────
    # Chunk-local T_c × T_c preconditioner P_c = exp(τ_h · (G_c/√N − √N))
    # applied to X_c via P_c^{-1} X_c.  Inter-chunk scan runs on
    # preconditioned X unchanged.  Faithful to the RWKV-6 LUCID formulation
    # (see _apply_lucid_recurrent in rwkv6_time_mix.py) adapted to the
    # Mamba-2 dual-form B-as-key convention.
    if lucid_temp is not None:
        if lucid_key_source == "B":
            X_c = _apply_lucid_mamba2_chunked(X_c, B_c, lucid_temp)
        elif lucid_key_source == "C":
            X_c = _apply_lucid_mamba2_chunked(X_c, C_c, lucid_temp)
        else:
            raise ValueError(
                f"lucid_key_source must be 'B' or 'C', got "
                f"{lucid_key_source!r}"
            )

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
# 1b. Causal SSD scan with write-novelty gate (chunked Σ variant)
# ---------------------------------------------------------------------------

def ssd_scan_causal_novelty(
    X: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int = 64,
    state: Optional[torch.Tensor] = None,
    sigma_state: Optional[torch.Tensor] = None,
    novelty_gamma: torch.Tensor,                     # (H,) required
    lucid_temp: Optional[torch.Tensor] = None,
    lucid_key_source: str = "B",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mamba-2 SSD chunked scan with per-chunk write-novelty gate.

    Replaces the vanilla write  s_t = α_t s_{t-1} + B_t ⊗ X_{d,t}  with

        s_t = α_t s_{t-1} + ω_t · (B_t ⊗ X_{d,t})
        ω_t = 1 / (1 + γ_h / (B_t^T Σ_c^{-1} B_t + ε))

    Σ_c is a running key-covariance maintained once per chunk
    (see ``_compute_novelty_gates``).  ω_t scales X_{d,t} before the
    intra- and inter-chunk einsums, so the gate feeds both the diagonal
    (C·B^T·X within chunk) and state-carry (Σ B_s^T X_s) paths
    consistently.

    LUCID composes multiplicatively on X_c after the novelty gate:
    ω · X, then P^{-1}(ω · X).

    Extra return value ``sigma_final`` allows cross-chunk carry during
    evaluation (matches the existing ``ssm`` carry semantics).
    """
    if novelty_gamma is None:
        raise ValueError("ssd_scan_causal_novelty requires novelty_gamma")

    Bsz, L, H, P = X.shape
    N = B.shape[-1]

    X_p, pad = _pad_to_multiple(X, chunk_size, dim=1)
    dt_p, _ = _pad_to_multiple(dt, chunk_size, dim=1)
    B_p, _ = _pad_to_multiple(B, chunk_size, dim=1)
    C_p, _ = _pad_to_multiple(C, chunk_size, dim=1)
    L_p = X_p.size(1)
    nC = L_p // chunk_size

    X_d = (X_p * dt_p.unsqueeze(-1)).float()
    A_d = (dt_p * A.view(1, 1, H)).float()
    B_f = B_p.float()
    C_f = C_p.float()

    X_c = X_d.reshape(Bsz, nC, chunk_size, H, P)
    A_c = A_d.reshape(Bsz, nC, chunk_size, H)
    B_c = B_f.reshape(Bsz, nC, chunk_size, H, N)
    C_c = C_f.reshape(Bsz, nC, chunk_size, H, N)

    A_cl = A_c.permute(0, 3, 1, 2)                          # (B, H, nC, Tc)
    A_cumsum = torch.cumsum(A_cl, dim=-1)                   # (B, H, nC, Tc)

    # ── Write-novelty gate ──────────────────────────────────────────────
    # Sequential across chunks; parallel within a chunk.  ω gates the
    # write path (X_c), affecting both intra- and inter-chunk einsums.
    omega, sigma_final = _compute_novelty_gates(
        B_c, A_cumsum, novelty_gamma, sigma_state,
    )                                                       # (B, nC, Tc, H), (B, H, N, N)
    X_c = omega.unsqueeze(-1) * X_c                         # broadcast over P

    # ── Optional LUCID preconditioner, composed after novelty gating ───
    if lucid_temp is not None:
        if lucid_key_source == "B":
            X_c = _apply_lucid_mamba2_chunked(X_c, B_c, lucid_temp)
        elif lucid_key_source == "C":
            X_c = _apply_lucid_mamba2_chunked(X_c, C_c, lucid_temp)
        else:
            raise ValueError(
                f"lucid_key_source must be 'B' or 'C', got "
                f"{lucid_key_source!r}"
            )

    # ── Rest of the chunked SSD (same as ssd_scan_causal) ──────────────
    L_mask = torch.exp(_segsum(A_cl))
    Y_diag = torch.einsum(
        "bclhn,bcshn,bhcls,bcshp->bclhp",
        C_c, B_c, L_mask, X_c,
    )

    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)
    produced = torch.einsum(
        "bclhn,bhcl,bclhp->bchpn",
        B_c, decay_states, X_c,
    )

    if state is None:
        init = torch.zeros(Bsz, 1, H, P, N, device=X.device, dtype=torch.float32)
    else:
        init = state.float().unsqueeze(1)
    produced_full = torch.cat([init, produced], dim=1)

    chunk_boundary = A_cumsum[..., -1]
    decay_chunk = torch.exp(_segsum(F.pad(chunk_boundary, (1, 0))))
    propagated = torch.einsum(
        "bhzc,bchpn->bzhpn",
        decay_chunk, produced_full,
    )
    entering, final_state = propagated[:, :-1], propagated[:, -1]

    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum(
        "bclhn,bchpn,bhcl->bclhp",
        C_c, entering, state_decay_out,
    )

    Y = (Y_diag + Y_off).reshape(Bsz, L_p, H, P)
    if pad:
        Y = Y[:, :L]
    return Y.to(X.dtype), final_state.to(X.dtype), sigma_final.to(X.dtype)


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
