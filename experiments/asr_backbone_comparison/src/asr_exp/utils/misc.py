"""Misc utilities: seeding, parameter counting, result serialization."""

import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_module(model: nn.Module) -> dict:
    """Return trainable parameter counts for each top-level submodule plus total."""
    breakdown = {
        name: sum(p.numel() for p in mod.parameters() if p.requires_grad)
        for name, mod in model.named_children()
    }
    breakdown["total"] = sum(breakdown.values())
    return breakdown


def make_serializable(obj):
    """Recursively convert tensors / numpy scalars to plain Python types."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, (torch.Tensor,)):
        return obj.item()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj
