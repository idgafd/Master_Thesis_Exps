"""Utilities: seeding, parameter counting."""

import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> dict:
    """Count parameters by module group.

    Returns dict with 'total', 'trainable', and per-top-level-module counts.
    """
    result = {}
    total = 0
    trainable = 0

    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    result["total"] = total
    result["trainable"] = trainable

    # Per top-level module
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        result[name] = n

    return result


def format_param_count(n: int) -> str:
    """Format parameter count for display."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
