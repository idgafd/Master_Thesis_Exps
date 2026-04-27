"""Stage 12 — Decay-Coupled Delta tests.

Per STAGE12_DECAY_COUPLED_DELTA.md §3.1, three CI-blocking tests:
  (a) at-init reduction to vanilla RWKV-6 within fp32 noise floor
      (max abs diff < 1e-5 on B=2, T=128 input)
  (b) gradient flows to `p_h` (non-zero, non-NaN) after one backward
      pass on a dummy loss
  (c) γ_t = α_t^{p_h} elementwise within fp32 noise

These mirror the discipline of `tests/test_mamba2_kernels.py::test_lion_chunk_matches_lion_full`
— a CI-blocking equivalence guarantee for the new mechanism path.
"""

import torch


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_encoder(backbone: str, seed: int = 42):
    """Construct an RWKV-6 encoder via the formal_v1 factory."""
    from src.config import ExperimentConfig
    from src.models.encoder import build_encoder
    torch.manual_seed(seed)
    cfg = ExperimentConfig(
        d_model=256, n_layers=6, n_heads=4, head_size=64,
        backbone=backbone,
    )
    return build_encoder(cfg).to(_DEVICE)


def test_at_init_reduction_to_vanilla():
    """§2.4 reduction contract — at init, `rwkv6_decay_coupled_delta`
    output equals `rwkv6` output within fp32 noise (max abs diff < 1e-5)
    on the spec's B=2, T=128 input.

    Why this holds:
      `delta_recurrent_gate` is zero-init ⇒ β_eff ≡ 0 ⇒ the rank-1 erase
      term vanishes ⇒ A_full = diag(α) ⇒ the chunked Hillis–Steele scan
      reduces to vanilla WKV.  γ (= α^{p_h}) is multiplied by β; with β=0
      the γ factor contributes nothing to the forward pass.

    Why fp32 noise (not bit-exact):
      The two paths use DIFFERENT chunked schedules — `_recurrent_delta_scan`
      uses Hillis–Steele with chunk_size=64; `_chunked_wkv` uses the
      [128, 16, 2, 1] decreasing schedule.  Each schedule accumulates
      fp32 round-off in a different order.
    """
    enc_dcd = _build_encoder("rwkv6_decay_coupled_delta", seed=42).eval()
    enc_van = _build_encoder("rwkv6", seed=42).eval()

    # Copy the OVERLAPPING parameter set from decay-coupled into vanilla,
    # so both encoders share IDENTICAL projection weights, w_raw, time_faaaa,
    # etc.  The decay-coupled extras (delta_params.*, delta_recurrent_gate,
    # p_h) live only in `enc_dcd` and are absent from `enc_van`.
    sd_dcd = enc_dcd.state_dict()
    sd_van = enc_van.state_dict()
    overlap = {k: sd_dcd[k] for k in sd_van if k in sd_dcd}
    missing, unexpected = enc_van.load_state_dict(overlap, strict=False)
    assert not unexpected, f"unexpected keys when loading overlap: {unexpected}"

    # Sanity: the gate must be exactly zero at init for the reduction to hold.
    for i, layer in enumerate(enc_dcd.layers):
        gate = layer.att.delta_recurrent_gate
        assert torch.all(gate == 0.0), (
            f"layer {i}: delta_recurrent_gate is not zero at init: {gate}"
        )

    B, T, D = 2, 128, 256
    torch.manual_seed(0)
    x = torch.randn(B, T, D, device=_DEVICE)
    lengths = torch.full((B,), T, dtype=torch.long, device=_DEVICE)

    with torch.no_grad():
        y_dcd, _ = enc_dcd(x, lengths)
        y_van, _ = enc_van(x, lengths)

    diff = (y_dcd - y_van).abs().max().item()
    rel = diff / (y_van.abs().max().item() + 1e-12)
    print(f"[at-init reduction] max abs diff = {diff:.3e}  rel = {rel:.3e}")
    assert diff < 1e-5, (
        f"max abs diff = {diff:.3e}, expected < 1e-5 per spec §2.4"
    )


def test_p_h_gradient_flow():
    """§3.1(b) — gradient flows to `p_h`: non-zero, finite.

    Subtlety: at strict init, `delta_recurrent_gate = 0` ⇒ `β_eff = 0`
    EXACTLY ⇒ the rank-1 term `β · γγ^T ⊙ kk kk^T` is identically zero ⇒
    `∂loss/∂γ ≡ 0` ⇒ `p_h.grad ≡ 0`.  This is the §2.4 reduction
    contract working as designed (mechanism dormant at init).

    What §3.1(b) actually requires is that the gradient *path* through
    γ → p_h is structurally live — i.e. once SGD grows β away from 0
    (which happens immediately because `delta_recurrent_gate.grad` is
    non-zero on the first backward), p_h starts moving too.

    Test: nudge `delta_recurrent_gate` to a small non-zero value (mimicking
    post-step-1 state), then verify p_h.grad is non-zero and finite.
    This is the structural connectivity check the spec asks for.
    """
    enc = _build_encoder("rwkv6_decay_coupled_delta", seed=42).train()

    # Mimic post-step-1 training state — gate has been bumped slightly.
    with torch.no_grad():
        for layer in enc.layers:
            layer.att.delta_recurrent_gate.fill_(0.05)

    B, T, D = 2, 128, 256
    torch.manual_seed(0)
    x = torch.randn(B, T, D, device=_DEVICE)
    lengths = torch.full((B,), T, dtype=torch.long, device=_DEVICE)

    y, _ = enc(x, lengths)
    # Cheap surrogate loss — CTC is tested elsewhere; we only need a scalar
    # that depends on every output element so the gradient must traverse
    # the full delta-rule path on every layer.
    loss = (y ** 2).mean()
    loss.backward()

    for i, layer in enumerate(enc.layers):
        p_h = layer.att.delta_params.p_h
        g = p_h.grad
        assert g is not None, f"layer {i}: p_h.grad is None"
        assert torch.isfinite(g).all(), (
            f"layer {i}: p_h.grad has NaN/Inf: {g}"
        )
        assert g.abs().max().item() > 0.0, (
            f"layer {i}: p_h.grad is exactly zero everywhere — gradient "
            f"path through γ = exp(p·w) may be broken: {g}"
        )


def test_p_h_grad_zero_at_strict_init():
    """Companion to test_p_h_gradient_flow: at strict init (gate ≡ 0),
    p_h.grad MUST be exactly zero.  This documents that the §2.4
    reduction contract and §3.1(b) gradient-flow requirement are both
    upheld — they just take effect at different points in training:

      step 0 (gate=0):       p_h.grad = 0    (§2.4 satisfied)
      step ≥ 1 (gate > 0):   p_h.grad ≠ 0    (§3.1(b) satisfied)

    A non-zero gradient at strict init would mean the at-init reduction
    is leaking — a regression we want to catch.
    """
    enc = _build_encoder("rwkv6_decay_coupled_delta", seed=42).train()

    B, T, D = 2, 128, 256
    torch.manual_seed(0)
    x = torch.randn(B, T, D, device=_DEVICE)
    lengths = torch.full((B,), T, dtype=torch.long, device=_DEVICE)
    y, _ = enc(x, lengths)
    (y ** 2).mean().backward()

    for i, layer in enumerate(enc.layers):
        g = layer.att.delta_params.p_h.grad
        assert g is not None
        # Gate is still zero ⇒ rank-1 erase term is identically zero ⇒
        # gradient through γ is killed by β=0 multiplier.
        assert g.abs().max().item() == 0.0, (
            f"layer {i}: p_h.grad is non-zero at strict init "
            f"({g.abs().max().item():.3e}) — the at-init reduction "
            "contract is leaking."
        )


def test_gamma_matches_alpha_to_the_p():
    """§3.1(c) — γ_t = exp(p · w) = α^p elementwise within fp32 noise.

    Unit-style sanity check on the coupling-vector computation in isolation
    (independent of the rank-1 erase machinery).
    """
    from src.models.mechanisms.decay_coupled_delta import DecayCoupledDeltaParams

    H, K = 4, 64
    params = DecayCoupledDeltaParams(
        hidden_size=H * K, n_head=H, head_size=K,
        p_init=1.7,  # arbitrary non-trivial value to make the test non-vacuous
    )

    B, T = 2, 128
    torch.manual_seed(0)
    # Match the RWKV-6 init scale: typical w_raw is ~U(0, 0.5) so the
    # log-decay w = -exp(w_raw) lives in roughly [-1.6, -1].  Using
    # `-torch.exp(torch.randn(...))` here would push w as low as -55 and
    # underflow exp(p·w) to 0 in fp32 — testing fp32 limits, not the
    # mechanism.  Sample a realistic distribution instead.
    w = -torch.rand(B, H, T, K).mul(0.5).add(0.1)  # w ∈ [-0.6, -0.1]
    alpha = torch.exp(w)

    gamma = params.compute_gamma(w)
    expected = alpha.pow(params.p_h.view(1, H, 1, 1))

    diff = (gamma - expected).abs().max().item()
    print(f"[γ vs α^p] max abs diff = {diff:.3e}")
    assert diff < 1e-6, f"γ != α^p: max abs diff = {diff:.3e}"

    # γ ∈ [0, 1] when w ≤ 0 and p ≥ 0 (mathematically (0, 1] but extreme
    # α near 0 can underflow to 0 in fp32 — this is a feature: γ_i = 0
    # means channel i is fully masked from the erase, which is the design
    # intent at extreme decay).
    assert (gamma >= 0).all() and (gamma <= 1.0 + 1e-6).all(), (
        f"γ out of [0, 1]: min={gamma.min().item()} "
        f"max={gamma.max().item()}"
    )


if __name__ == "__main__":
    # Allow running directly without pytest, for quick local iteration.
    test_at_init_reduction_to_vanilla()
    test_p_h_gradient_flow()
    test_p_h_grad_zero_at_strict_init()
    test_gamma_matches_alpha_to_the_p()
    print("OK — all four Stage 12 tests passed.")
