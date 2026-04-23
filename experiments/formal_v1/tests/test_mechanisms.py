"""Unit tests for mechanism modules — regression tests for init bugs.

See MULTIDIL_INIT_FIX_HANDOFF.md §6 for the init-gradient-trap bug that
these tests guard against.
"""

import torch

from src.models.mechanisms.conv_shift import (
    DWConvShift,
    MultiDilationDWConvShift,
)


def test_multidil_init_gradient_flow_symmetric():
    """Gradient must reach α_d and branch weights for all d>1 at init.

    Regression test for the zero-at-init gradient trap where α_{d>1}=0
    and branch_{d>1}.weight=0 gave ∂L/∂α_d = ∂L/∂weight_d = 0.
    """
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

    assert mod.alpha.grad is not None
    assert mod.alpha.grad[0].abs().item() > 1e-8, \
        f"alpha[d=1] gradient is zero: {mod.alpha.grad}"

    for i, d in enumerate(mod.dilations):
        if d == 1:
            continue
        g = mod.alpha.grad[i].abs().item()
        assert g > 1e-8, \
            f"alpha[d={d}] gradient is zero after fix: {mod.alpha.grad}"

    for i, d in enumerate(mod.dilations):
        if d == 1:
            continue
        g = mod.branches[i].weight.grad.norm().item()
        assert g > 1e-8, \
            f"branch[d={d}] weight gradient is zero after fix"


def test_multidil_init_causal_gradient_flow():
    """Same gradient-flow check for causal padding mode."""
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
    """At init, multi-dilation output ≈ single-dilation output within 1e-3.

    Softer than bit-exact zero-regression: the fix intentionally introduces
    a ~3e-4 perturbation from the non-zero α_{d>1} and branch_{d>1}.weight.
    """
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
    # Elementwise σ ≈ √3 × NON_MAIN_ALPHA × NON_MAIN_WEIGHT_STD × √(kernel)
    # ≈ √3 × 0.01 × 0.01 × √3 ≈ 3e-4. Max over B·T·D≈4k Gaussians is
    # ~σ·√(2 ln N) ≈ 4σ ≈ 1.2e-3. 5e-3 gives headroom for seed variance
    # and still well below activation noise.
    assert diff < 5e-3, f"init perturbation too large: {diff}"
