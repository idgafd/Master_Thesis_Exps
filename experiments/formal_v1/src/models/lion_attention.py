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
    """LION attention with LUCID preconditioner (CORRECTED).

    CORRECTED formulation: Y = P^{-1} @ (A @ V)
    (decorrelate the attention output, not the values)

    Draft (wrong): Y = A @ P^{-1} @ V

    Args:
        lucid_temp: learnable per-head temperature (positive, via softplus)
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
    Y_raw = A @ v  # (B, H, T, K)

    # Apply LUCID preconditioner: P^{-1} @ Y_raw
    if chunk_size is None:
        Y_out = _apply_lucid_preconditioner(k, Y_raw, lucid_temp)
    else:
        # Chunked LUCID: apply within windows
        Y_out = torch.zeros_like(Y_raw)
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            k_chunk = k[:, :, start:end, :]
            y_chunk = Y_raw[:, :, start:end, :]
            Y_out[:, :, start:end, :] = _apply_lucid_preconditioner(
                k_chunk, y_chunk, lucid_temp
            )

    return Y_out


def _apply_lucid_preconditioner(
    k: torch.Tensor,      # (B, H, T_chunk, K)
    y: torch.Tensor,      # (B, H, T_chunk, K)
    temp: torch.Tensor,   # (1, H, 1, 1)
) -> torch.Tensor:
    """Apply LUCID: P = I + exp(tau * K_norm @ K_norm^T), solve P @ Y_out = Y."""
    k_norm = torch.nn.functional.normalize(k, dim=-1)
    gram = k_norm @ k_norm.transpose(-2, -1)  # (B, H, T_chunk, T_chunk)
    P = torch.eye(gram.size(-1), device=gram.device, dtype=gram.dtype) + torch.exp(temp * gram)
    return torch.linalg.solve(P, y)
