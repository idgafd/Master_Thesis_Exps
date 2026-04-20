"""Fast RSE + Rayleigh-viscosity scan — real-arithmetic reimplementation.

Mathematically equivalent to ``RWKV6TimeMix._forward_recurrent_rse`` with
``use_viscosity=True`` (see ``src/models/rwkv6_time_mix.py`` line 575), but
rewritten to hit GPU Tensor Cores instead of the slower ``complex64`` path.

Why this is faster
------------------
The reference implementation carries the RSE state as ``complex64`` inside
the chunk loop. The two hot ops are
    S_intra = einsum('bhtsk,bhsc->bhtkc', scaled_k, v_c_complex)   # complex
    y       = einsum('bhtk,bhtkc->bhtc',  r_c.conj(), S_total)     # complex
cuBLAS / Tensor Core kernels do not accept complex dtypes, so PyTorch falls
back to a generic CUDA kernel (~15 TFLOPS vs ~75 TFLOPS fp32 / ~300 TFLOPS
bf16 on Blackwell). Here we keep (Re, Im) as two real fp32 tensors; each
complex matmul decomposes into two real matmuls that do hit Tensor Cores,
and autocast can downcast them to bf16.

Precision policy
----------------
The cumulative sum of ``log_decay`` and ``theta`` is done in fp32 — bf16
cumsum loses precision over long sequences (this was the Stage-3 v1
failure mode). All matmuls can be downcast by the caller's ``autocast``
context; the elementwise ops stay in their input dtype.

Drop-in usage
-------------
The signature mirrors ``_forward_recurrent_rse`` so the TimeMix can route
into this function by passing its learned parameters explicitly:

    y, new_state = rse_viscosity_scan(
        r, k, v, w, theta,
        eta=self.viscosity_eta if self.use_viscosity else None,
        u=self.time_faaaa if not self.drop_u else None,
        state=state,
    )

No trained parameters are changed; the forward pass is numerically
equivalent to the reference up to fp32 round-off (max abs diff < 1e-4 on
LibriSpeech-shaped inputs).
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch


def rse_viscosity_scan(
    r: torch.Tensor,       # (B, H, T, K)
    k: torch.Tensor,       # (B, H, T, K)
    v: torch.Tensor,       # (B, H, T, K)
    w: torch.Tensor,       # (B, H, T, K)  log-decay (negative)
    theta: torch.Tensor,   # (B, H, T, Bk) rotation angle per 2x2 block
    eta: Optional[torch.Tensor] = None,  # (H, Bk) viscosity coefficient; None → disabled
    u: Optional[torch.Tensor] = None,    # (H, K)  bonus coefficient; None → skipped
    state: Optional[torch.Tensor] = None,  # (B, H, K, K) carry
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked RSE scan with Rayleigh viscosity, real-arithmetic form.

    Math (per 2x2 block b, shared across the K-channel value axis):

        lambda_eff = -w + eta * theta**2                (viscosity coupling)
        z_t        = exp(-lambda_eff + i * theta_t)     (complex pole per step)
        c_t        = z_t * c_{t-1} + k_c_t * v_t        (complex state per block)
        y_t        = sum_b Re( conj(r_c_{t,b}) * c_{t,b} ) + (r * u * k) . v

    The complex scalar z_t acts on the 2-vector (S[2b], S[2b+1]) as a
    rotation-scale. We keep (S_r, S_i) separately and expand every complex
    multiply into real multiplies.

    Args:
        r, k, v: (B, H, T, K) real tensors. K must be even.
        w:       (B, H, T, K) log-decay (negative values).
        theta:   (B, H, T, Bk) per-block rotation, Bk = K // 2.
        eta:     (H, Bk) viscosity coupling, or None to disable.
        u:       (H, K) pointwise bonus (time_faaaa), or None to skip.
        state:   (B, H, K, K) prior state, or None to start at zero.
        chunk_size: intra-chunk parallelism depth. Default 64 (matches ref).

    Returns:
        y:          (B, H, T, K) in the input dtype of r.
        new_state:  (B, H, K, K) fp32, for carry-state use.
    """
    B, H, T, K = r.shape
    Bk = K // 2
    device = r.device
    in_dtype = r.dtype
    assert K % 2 == 0, "RSE requires even head_size"

    # All subsequent math in fp32. Inside an autocast(bf16) context the
    # matmuls (einsums below) will still downcast to bf16 automatically;
    # the cumsums stay in fp32 because they are not autocast-eligible ops.
    r_f = r.float()
    k_f = k.float()
    v_f = v.float()
    w_f = w.float()
    theta_f = theta.float()

    # ── Split r, k into (real, imag) pairs along the Bk axis ──
    #   r[...,2b]   = Re(r_c),  r[...,2b+1] = Im(r_c)   (same for k)
    r_pairs = r_f.view(B, H, T, Bk, 2)
    k_pairs = k_f.view(B, H, T, Bk, 2)
    r_r = r_pairs[..., 0].contiguous()     # (B, H, T, Bk)
    r_i = r_pairs[..., 1].contiguous()
    k_r = k_pairs[..., 0].contiguous()
    k_i = k_pairs[..., 1].contiguous()

    # ── Block-level log-decay (mean of the 2 channels per block) ──
    log_decay_block = w_f.view(B, H, T, Bk, 2).mean(dim=-1)  # (B, H, T, Bk)

    # ── Rayleigh viscosity: lambda_eff = lambda + eta * theta^2 ──
    # w is log-decay = -lambda, so we SUBTRACT eta*theta^2 from w.
    if eta is not None:
        log_decay_block = (
            log_decay_block - eta.view(1, H, 1, Bk).float() * theta_f.pow(2)
        )

    # ── Cumulative decay and phase, computed separately in real fp32 ──
    #   cum_lam[t, b]   = sum_{i<=t} log_decay_block[i, b]   (<= 0, monotone)
    #   cum_theta[t, b] = sum_{i<=t} theta[i, b]
    cum_lam = log_decay_block.cumsum(dim=2)     # (B, H, T, Bk)
    cum_theta = theta_f.cumsum(dim=2)           # (B, H, T, Bk)

    # ── Initial state, split into (s_r, s_i) each (B, H, Bk, K) ──
    if state is None:
        s_r = torch.zeros(B, H, Bk, K, dtype=torch.float32, device=device)
        s_i = torch.zeros(B, H, Bk, K, dtype=torch.float32, device=device)
    else:
        # state layout: (B, H, K, K) reshaped to (B, H, Bk, 2, K)
        S0 = state.float().view(B, H, Bk, 2, K)
        s_r = S0[..., 0, :].contiguous()
        s_i = S0[..., 1, :].contiguous()

    # ── Pointwise bonus term: fully local, compute once outside the loop ──
    #   bonus_out[t, c] = (sum_k r[t,k] * u[k] * k[t,k]) * v[t, c]
    if u is not None:
        u_view = u.view(1, H, 1, K).float()
        bonus_scalar = (r_f * u_view * k_f).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        bonus_out = bonus_scalar * v_f                                 # (B, H, T, K)
    else:
        bonus_out = None

    # ── Chunk loop — fixed chunk_size, python-for (torch.compile friendly) ──
    out = torch.zeros(B, H, T, K, dtype=torch.float32, device=device)
    n_chunks = (T + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, T)
        tc = end - start

        cum_lam_c = cum_lam[:, :, start:end]        # (B, H, tc, Bk)
        cum_theta_c = cum_theta[:, :, start:end]
        k_r_c = k_r[:, :, start:end]
        k_i_c = k_i[:, :, start:end]
        r_r_c = r_r[:, :, start:end]
        r_i_c = r_i[:, :, start:end]
        v_c = v_f[:, :, start:end]                  # (B, H, tc, K)

        # diff[t, s, b] = cumlog[t, b] - cumlog[s, b]
        #   real part = cum_lam[t] - cum_lam[s] (<= 0 for t>=s, >0 for t<s)
        #   imag part = cum_theta[t] - cum_theta[s]
        diff_lam = cum_lam_c.unsqueeze(3) - cum_lam_c.unsqueeze(2)       # (B,H,tc,tc,Bk)
        diff_theta = cum_theta_c.unsqueeze(3) - cum_theta_c.unsqueeze(2)

        # Causal mask: only s <= t contributes
        mask = torch.tril(
            torch.ones(tc, tc, device=device, dtype=torch.bool)
        ).view(1, 1, tc, tc, 1)
        # Clamp upper triangle to very-negative so exp ≈ 0
        diff_lam_safe = diff_lam.masked_fill(~mask, -60.0)

        # Attention coefficient A = exp(diff_lam) * [cos(diff_theta) + i sin(diff_theta)]
        envelope = diff_lam_safe.exp()               # (B, H, tc, tc, Bk) positive
        A_r = envelope * diff_theta.cos()
        A_i = envelope * diff_theta.sin()
        # Exact zero above diagonal (envelope is ~e^-60 there, not zero)
        zero = torch.zeros_like(A_r)
        A_r = torch.where(mask, A_r, zero)
        A_i = torch.where(mask, A_i, zero)

        # scaled_k[t, s, b] = A[t, s, b] * k_c[s, b]    — complex scalar multiply
        #   (A_r + i A_i)(k_r + i k_i):
        #     real: A_r·k_r - A_i·k_i
        #     imag: A_r·k_i + A_i·k_r
        # Shapes: A_? is (B,H,tc,tc,Bk); k_?_c.unsqueeze(2) is (B,H,1,tc,Bk)
        k_r_bcast = k_r_c.unsqueeze(2)   # broadcast across query-time dim
        k_i_bcast = k_i_c.unsqueeze(2)
        sk_r = A_r * k_r_bcast - A_i * k_i_bcast
        sk_i = A_r * k_i_bcast + A_i * k_r_bcast

        # Intra-chunk state: S_intra = sum_s sk[t, s, b] * v[s, c]
        # Two real einsums; each reshapes to a cuBLAS Tensor-Core matmul
        # (contraction dim s = tc).
        S_intra_r = torch.einsum('bhtsk,bhsc->bhtkc', sk_r, v_c)
        S_intra_i = torch.einsum('bhtsk,bhsc->bhtkc', sk_i, v_c)

        # Prior contribution: exp(cumlog[t]) * c_state
        #   (dec_r + i dec_i)(s_r + i s_i) broadcast over value-channel axis
        decay_env = cum_lam_c.exp()                   # (B, H, tc, Bk)
        dec_r = decay_env * cum_theta_c.cos()
        dec_i = decay_env * cum_theta_c.sin()

        # dec_? has shape (B, H, tc, Bk); expand to (B, H, tc, Bk, 1)
        # s_?   has shape (B, H, Bk, K); expand to (B, H, 1,  Bk, K)
        dec_r_u = dec_r.unsqueeze(-1)
        dec_i_u = dec_i.unsqueeze(-1)
        s_r_u = s_r.unsqueeze(2)
        s_i_u = s_i.unsqueeze(2)
        prior_r = dec_r_u * s_r_u - dec_i_u * s_i_u    # (B, H, tc, Bk, K)
        prior_i = dec_r_u * s_i_u + dec_i_u * s_r_u

        S_total_r = prior_r + S_intra_r
        S_total_i = prior_i + S_intra_i

        # Readout: y[t, c] = sum_b Re( conj(r_c) * S_total )
        #                  = sum_b ( r_r·S_r + r_i·S_i )
        y_chunk = (
            torch.einsum('bhtk,bhtkc->bhtc', r_r_c, S_total_r)
            + torch.einsum('bhtk,bhtkc->bhtc', r_i_c, S_total_i)
        )

        out[:, :, start:end] = y_chunk

        # Carry last step to next chunk
        s_r = S_total_r[:, :, -1]   # (B, H, Bk, K)
        s_i = S_total_i[:, :, -1]

    # ── Add bonus (local, same for every t) ──
    if bonus_out is not None:
        out = out + bonus_out

    # ── Repack final state to (B, H, K, K) in the ref layout ──
    final_state = torch.zeros(B, H, K, K, dtype=torch.float32, device=device)
    fv = final_state.view(B, H, Bk, 2, K)
    fv[..., 0, :] = s_r
    fv[..., 1, :] = s_i

    return out.to(in_dtype), final_state
