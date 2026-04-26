"""LION parallel full attention kernels for bidirectional RWKV-6.

Implements Theorem B.1 from Afzal et al. 2025:

  Y = (TRIL[(Q*L^F)(K*Inv(L^F))^T] + TRIU[(Q*L^B)(K*Inv(L^B))^T, 1]) V

Forward coeff(j->i), i>j:  exp(cs[i] - cs[j])
Backward coeff(j->i), i<j: exp(cs_b[j] - cs_b[i])

where cs = cumsum(w), cs_b = cumsum(w) - w
"""

import torch


def lion_parallel_attention(
    r: torch.Tensor,  # (B, H, T, K)
    k: torch.Tensor,  # (B, H, T, K)
    v: torch.Tensor,  # (B, H, T, K)
    w: torch.Tensor,  # (B, H, T, K) — log-space decay (negative values)
) -> torch.Tensor:
    """Standard LION bidirectional attention. Returns (B, H, T, K) in float32."""
    B, H, T, K = r.shape

    r = r.float()
    k = k.float()
    v = v.float()
    w = w.float()

    cs = torch.cumsum(w, dim=2)
    cs_b = cs - w

    # Numerical stabilization via midpoint shift
    shift_f = cs[:, :, T // 2:T // 2 + 1, :]
    shift_b = cs_b[:, :, T // 2:T // 2 + 1, :]
    cs_f_s = (cs - shift_f).clamp(-60, 60)
    cs_b_s = (cs_b - shift_b).clamp(-60, 60)

    # Forward (lower-triangular including diagonal)
    exp_cs_f = torch.exp(cs_f_s)
    A_fwd = torch.tril(
        (r * exp_cs_f) @ (k * torch.exp(-cs_f_s)).transpose(-2, -1)
    )

    # Backward (strictly upper-triangular)
    exp_cs_b = torch.exp(cs_b_s)
    A_bwd = torch.triu(
        (r * torch.exp(-cs_b_s)) @ (k * exp_cs_b).transpose(-2, -1),
        diagonal=1,
    )

    return (A_fwd + A_bwd) @ v


def lion_attention_with_delta(
    r: torch.Tensor,  # (B, H, T, K)
    k: torch.Tensor,  # (B, H, T, K)
    v: torch.Tensor,  # (B, H, T, K)
    w: torch.Tensor,  # (B, H, T, K)
    kk: torch.Tensor,  # (B, H, T, K) — normalized key for delta rule
    iclr: torch.Tensor,  # (B, H, T, K) — erasure strength
) -> torch.Tensor:
    """LION attention with causal-only delta rule correction.

    The delta rule correction selectively erases stale associations from
    the forward (causal) direction only. No anticausal delta — this was
    shown to have no theoretical basis and degraded performance in the draft.

    A_delta_fwd = -tril(A_fwd @ kk_corr_causal)
    """
    B, H, T, K = r.shape

    r = r.float()
    k = k.float()
    v = v.float()
    w = w.float()
    kk = kk.float()
    iclr = iclr.float()

    cs = torch.cumsum(w, dim=2)
    cs_b = cs - w

    shift_f = cs[:, :, T // 2:T // 2 + 1, :]
    shift_b = cs_b[:, :, T // 2:T // 2 + 1, :]
    cs_f_s = (cs - shift_f).clamp(-60, 60)
    cs_b_s = (cs_b - shift_b).clamp(-60, 60)

    exp_cs_f = torch.exp(cs_f_s)
    A_fwd = torch.tril(
        (r * exp_cs_f) @ (k * torch.exp(-cs_f_s)).transpose(-2, -1)
    )

    exp_cs_b = torch.exp(cs_b_s)
    A_bwd = torch.triu(
        (r * torch.exp(-cs_b_s)) @ (k * exp_cs_b).transpose(-2, -1),
        diagonal=1,
    )

    # Causal-only delta correction
    # kk_corr[i,j] = sum_d kk[i,d] * (kk[j,d] * iclr[j,d])
    kk_corr = kk @ (kk * iclr).transpose(-2, -1)  # (B, H, T, T)
    A_delta_fwd = -torch.tril(A_fwd @ kk_corr)

    return (A_fwd + A_bwd + A_delta_fwd) @ v


def lion_attention_with_lucid(
    r: torch.Tensor,  # (B, H, T, K)
    k: torch.Tensor,  # (B, H, T, K)
    v: torch.Tensor,  # (B, H, T, K)
    w: torch.Tensor,  # (B, H, T, K)
    lucid_temp: torch.Tensor,  # (H,) or (1, H, 1, 1) — per-head temperature
    chunk_size: int | None = None,
) -> torch.Tensor:
    """LION attention with paper-faithful LUCID preconditioner.

    Paper formulation (Eq. 5, Section 2.3):
        Step 1: Solve P · Y = V  for Y  (precondition values)
        Step 2: Output = A · Y           (attend over preconditioned values)

    where P = exp(K_RN @ K_RN^T / sqrt(d) - sqrt(d)), unit diagonal.

    For LION (bidirectional), P is full (no causal mask) since attention
    is bidirectional. The lucid_temp parameter allows learned scaling
    instead of the fixed 1/sqrt(d).

    Args:
        lucid_temp: learnable per-head temperature (positive, via softplus).
                    Replaces the paper's fixed 1/sqrt(d) for flexibility.
        chunk_size: if not None, apply LUCID within windows of this size
    """
    B, H, T, K = r.shape

    r = r.float()
    k = k.float()
    v = v.float()
    w = w.float()

    if lucid_temp.dim() == 1:
        lucid_temp = lucid_temp.view(1, H, 1, 1).float()
    else:
        lucid_temp = lucid_temp.float()

    cs = torch.cumsum(w, dim=2)
    cs_b = cs - w

    shift_f = cs[:, :, T // 2:T // 2 + 1, :]
    shift_b = cs_b[:, :, T // 2:T // 2 + 1, :]
    cs_f_s = (cs - shift_f).clamp(-60, 60)
    cs_b_s = (cs_b - shift_b).clamp(-60, 60)

    exp_cs_f = torch.exp(cs_f_s)
    A_fwd = torch.tril(
        (r * exp_cs_f) @ (k * torch.exp(-cs_f_s)).transpose(-2, -1)
    )

    exp_cs_b = torch.exp(cs_b_s)
    A_bwd = torch.triu(
        (r * torch.exp(-cs_b_s)) @ (k * exp_cs_b).transpose(-2, -1),
        diagonal=1,
    )

    A = A_fwd + A_bwd

    # Step 1: Precondition values — solve P · Y = V
    if chunk_size is None:
        Y = _apply_lucid_preconditioner(k, v, lucid_temp)
    else:
        Y = torch.zeros_like(v)
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            Y[:, :, start:end, :] = _apply_lucid_preconditioner(
                k[:, :, start:end, :], v[:, :, start:end, :], lucid_temp
            )

    # Step 2: Attend over preconditioned values
    return A @ Y


def lion_complex_attention(
    r: torch.Tensor,           # (B, H, T, K)
    k: torch.Tensor,           # (B, H, T, K)
    v: torch.Tensor,           # (B, H, T, K)
    log_decay: torch.Tensor,   # (B, H, T, Bk) — per-pair log-magnitude, ≤ 0
    theta: torch.Tensor,       # (B, H, T, Bk) — per-pair rotation angle (rad)
    pair_chunk: int = 8,       # chunk size along Bk to bound peak memory
) -> torch.Tensor:
    """LION bidirectional attention with block-complex SO(2) × R+ transition.

    Each pair of channels (2b, 2b + 1) is read as a complex number
    ``z[b] = u + i v`` and the per-step transition for pair b is the
    complex scalar ``g_t[b] = exp(log_decay_t[b] + i · theta_t[b])``.

    Forward / backward attention coefficients are built from the complex
    cumulative log-decay (Hermitian symmetric across the time-reversal):

        A_fwd[i, j, b] = exp(cs[i, b] − cs[j, b])              for i ≥ j
        A_bwd[i, j, b] = exp(conj(cs_b[j, b] − cs_b[i, b]))    for i < j

    with cs = cumsum(log_z) and cs_b = cs − log_z.  The conjugate on the
    backward side gives a unitary-style time-reversal: the rotation
    accumulated backwards in time is the inverse of the forward rotation.
    This matches the bidirectional Lie-group extension of the causal RSE
    transition described in Mechanisms_Overview §3 and is the natural
    drop-in for the LION pattern with a complex log-decay.

    Output: real part of the contracted complex tensor reshaped back to
    ``(B, H, T, K)``.

    Memory: builds an explicit ``(B, H, T, T, Bk)`` complex attention
    tensor.  At typical ASR shapes (T ≤ 500, Bk = 32) this is ≲ 250 MB
    per layer, well below GPU budget.  Caller is expected to ensure
    ``K % 2 == 0`` and ``Bk = K // 2``.
    """
    B, H, T, K = r.shape
    assert K % 2 == 0, f"K must be even for block-complex transition, got {K}"
    Bk = K // 2

    # Pack r, k, v as complex pairs along the K dim.
    r_pairs = r.float().view(B, H, T, Bk, 2)
    k_pairs = k.float().view(B, H, T, Bk, 2)
    r_c = torch.complex(r_pairs[..., 0], r_pairs[..., 1]).contiguous()      # (B, H, T, Bk)
    k_c = torch.complex(k_pairs[..., 0], k_pairs[..., 1]).contiguous()      # (B, H, T, Bk)
    v_c = v.float().to(torch.complex64).contiguous()                        # (B, H, T, K)

    # Complex log-decay per pair, with midpoint shift on the real part for
    # numerical stability (same trick as the standard LION kernel).
    log_z = torch.complex(log_decay.float(), theta.float())                 # (B, H, T, Bk)
    cs = log_z.cumsum(dim=2)                                                # (B, H, T, Bk)
    cs_b = cs - log_z                                                       # (B, H, T, Bk)
    mid = T // 2
    shift_f_re = cs[:, :, mid:mid + 1, :].real
    shift_b_re = cs_b[:, :, mid:mid + 1, :].real
    cs_s = torch.complex(cs.real - shift_f_re, cs.imag)
    cs_bs = torch.complex(cs_b.real - shift_b_re, cs_b.imag)

    fwd_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=r.device))  # i ≥ j
    bwd_mask = ~fwd_mask                                                         # i < j
    fwd_mask_b = fwd_mask.view(1, 1, T, T, 1)
    bwd_mask_b = bwd_mask.view(1, 1, T, T, 1)

    # Per-pair attention is independent across the pair index Bk.  We chunk
    # over Bk to bound the peak memory of the (B, H, T, T, Bk_chunk)
    # intermediate at training time.  Bk_chunk = 8 keeps the (T × T × Bk)
    # tensor under ~250 MB at T = 500, B = 4 in fp32-complex.
    y = r.new_zeros(B, H, T, K)

    for b0 in range(0, Bk, pair_chunk):
        b1 = min(b0 + pair_chunk, Bk)
        cs_s_c = cs_s[:, :, :, b0:b1]                                                # (B, H, T, p)
        cs_bs_c = cs_bs[:, :, :, b0:b1]
        k_c_c = k_c[:, :, :, b0:b1]                                                  # (B, H, T, p)
        r_c_c = r_c[:, :, :, b0:b1]                                                  # (B, H, T, p)

        diff_f = cs_s_c.unsqueeze(3) - cs_s_c.unsqueeze(2)                           # (B, H, T, T, p)
        diff_b = cs_bs_c.unsqueeze(2) - cs_bs_c.unsqueeze(3)                         # diff_b[i, j] = cs_bs[j] - cs_bs[i]
        diff_b_conj = torch.complex(diff_b.real, -diff_b.imag)

        real_f = diff_f.real.clamp(-60.0, 60.0)
        real_b = diff_b_conj.real.clamp(-60.0, 60.0)
        A_fwd = torch.exp(torch.complex(real_f, diff_f.imag))
        A_bwd = torch.exp(torch.complex(real_b, diff_b_conj.imag))
        A = torch.where(fwd_mask_b, A_fwd, torch.zeros_like(A_fwd)) \
            + torch.where(bwd_mask_b, A_bwd, torch.zeros_like(A_bwd))                # (B, H, T, T, p)

        scaled_k = A * k_c_c.unsqueeze(2)                                            # (B, H, T, T, p)

        # S[i, b, c] = sum_s scaled_k[i, s, b] · v_c[s, c].
        S = torch.einsum('bhisB,bhsc->bhiBc', scaled_k, v_c)                         # (B, H, T, p, K)
        y_chunk = torch.einsum('bhiB,bhiBc->bhic', r_c_c.conj(), S).real             # (B, H, T, K)
        y = y + y_chunk

    return y


def _apply_lucid_preconditioner(
    k: torch.Tensor,      # (B, H, T_chunk, K)
    v: torch.Tensor,      # (B, H, T_chunk, K) — values to precondition
    temp: torch.Tensor,   # (1, H, 1, 1)
) -> torch.Tensor:
    """Paper-faithful LUCID: P = exp(K_RN K_RN^T / sqrt(d) - sqrt(d)).

    K_RN = sqrt(d) * K / ||K||_2  (RMSNorm scaled by sqrt(d))
    Then K_RN K_RN^T has diagonal = d, so K_RN K_RN^T / sqrt(d) - sqrt(d)
    has diagonal = 0, giving P_ii = exp(0) = 1 (unit diagonal).

    We use a learnable temperature `temp` instead of the fixed 1/sqrt(d)
    for flexibility, but preserve the unit-diagonal property via the
    -sqrt(d) offset structure.

    Solve P · Y = V for Y via torch.linalg.solve.
    """
    K = k.size(-1)  # head_size = d
    sqrt_d = K ** 0.5

    # RMSNorm: k_rn = sqrt(d) * k / ||k||_2
    k_rn = sqrt_d * torch.nn.functional.normalize(k, dim=-1)  # (B, H, T, K)

    # Gram: K_RN @ K_RN^T — diagonal entries = d
    gram = k_rn @ k_rn.transpose(-2, -1)  # (B, H, T, T)

    # Paper: exp(gram / sqrt(d) - sqrt(d)) — unit diagonal
    # With learnable temp: exp(temp * (gram / sqrt(d) - sqrt(d)))
    # When temp=1 this matches the paper exactly
    scaled = (temp * (gram / sqrt_d - sqrt_d)).clamp(-30, 30)
    P = torch.exp(scaled)  # (B, H, T, T) — unit diagonal

    # Regularize for numerical stability
    T_chunk = k.size(-2)
    P = P + 1e-6 * torch.eye(T_chunk, device=P.device, dtype=P.dtype)

    return torch.linalg.solve(P, v)
