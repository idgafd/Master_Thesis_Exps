"""Shared post-hoc extraction for chapter-5 mechanism-parameter figures.

Loads `best_model.pt` for a single cell and walks the state-dict keys
to extract trained mechanism parameters as numpy arrays. Reused by
F8 (MSDC alpha), F9 (DHO eta and theta), and F10 (CVD tau).

Source-file line citations (codebase: experiments/formal_v1/src/models/).
The keys are reproduced as string constants here defensively so this
utility does not import from the project source tree.

  CVD `lucid_temperature` (raw; trained tau = softplus(raw)):
    rwkv6_time_mix.py:681             -> encoder.layers.{L}.att.lucid_temperature
    mamba2_block.py:362               -> encoder.layers.{L}.mamba.lucid_temperature
    linear_attn_causal.py:172         -> encoder.layers.{L}.lucid_temperature
    linear_attn_lion.py:122           -> encoder.layers.{L}.lucid_temperature

  MSDC `alpha` (per-dilation mixing weights):
    mechanisms/conv_shift.py:123      -> alpha (the shared parameter; the
                                         containing path varies per arch):
      RWKV-6:  encoder.layers.{L}.att.conv_shift_module.alpha
      Mamba-2: encoder.layers.{L}.mamba.conv1d.alpha
      LA:      encoder.layers.{L}.premix.alpha

  DHO `viscosity_eta` (per-head per-block viscosity coefficient):
    rwkv6_time_mix.py:835             -> encoder.layers.{L}.att.viscosity_eta
    mamba2_rse.py:154                 -> encoder.layers.{L}.mamba.viscosity_eta
    linear_attn_rse.py:122            -> encoder.layers.{L}.viscosity_eta

  DHO theta-init + LoRA:
    rwkv6_time_mix.py:820             -> time_theta             (1, 1, n_blocks)
    rwkv6_time_mix.py:822             -> time_theta_w1          (d_model, rank)
    rwkv6_time_mix.py:825             -> time_theta_w2          (rank, n_blocks)
    mamba2_rse.py:141                 -> theta_base
    mamba2_rse.py:142                 -> theta_w1
    linear_attn_rse.py:97             -> theta_base
    linear_attn_rse.py:100            -> theta_w1

The trained value reported in F10 is tau = softplus(raw):
    rwkv6_time_mix.py uses `F.softplus(self.lucid_temperature)`
    mamba2_block.py:574               same.
    linear_attn_causal.py:207         same.
    linear_attn_lion.py:202 / 239     same.

Returns dict with stacked numpy arrays per parameter family. Layer
ordering follows the integer order of `encoder.layers.{L}.*` keys.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

# ---- Layer-level parameter suffixes per arch family ----
# Each entry is (param_family, list of (suffix, role)). The first
# suffix that matches a state-dict key for a given layer wins.

CVD_SUFFIXES = [
    ("att.lucid_temperature",   "rwkv6"),
    ("mamba.lucid_temperature", "mamba2"),
    ("lucid_temperature",       "linear_attn"),  # last so that more-specific arches match first
]

MSDC_SUFFIXES = [
    ("att.conv_shift_module.alpha", "rwkv6"),
    ("mamba.conv1d.alpha",          "mamba2"),
    ("premix.alpha",                "linear_attn"),
]

DHO_ETA_SUFFIXES = [
    ("att.viscosity_eta",   "rwkv6"),
    ("mamba.viscosity_eta", "mamba2"),
    ("viscosity_eta",       "linear_attn"),
]

DHO_THETA_BASE_SUFFIXES = [
    ("att.time_theta",  "rwkv6"),
    ("mamba.theta_base", "mamba2"),
    ("theta_base",      "linear_attn"),
]

DHO_THETA_W1_SUFFIXES = [
    ("att.time_theta_w1", "rwkv6"),
    ("mamba.theta_w1",    "mamba2"),
    ("theta_w1",          "linear_attn"),
]

DHO_THETA_W2_SUFFIXES = [
    ("att.time_theta_w2", "rwkv6"),
    ("mamba.theta_w2",    "mamba2"),
    ("theta_w2",          "linear_attn"),
]


LAYER_PREFIX = re.compile(r"^encoder\.layers\.(\d+)\.(.+)$")


def _load_state_dict(cell_dir: Path) -> dict[str, torch.Tensor] | None:
    """Load `best_model.pt` and return the model state dict.

    Returns None if the file is an LFS pointer (the actual binary has
    not been fetched).
    """
    pt_path = cell_dir / "best_model.pt"
    if not pt_path.exists():
        return None
    # LFS pointer files start with "version https://git-lfs.github.com".
    head = pt_path.read_bytes()[:64]
    if head.startswith(b"version https://git-lfs"):
        return None
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in ckpt:
                return ckpt[key]
    return ckpt


def _layer_key_index(state_dict) -> dict[int, dict[str, torch.Tensor]]:
    """Group state-dict keys by `encoder.layers.{L}` integer.

    Returns {layer_index: {sub_key_after_layer_prefix: tensor}}.
    """
    layers: dict[int, dict[str, torch.Tensor]] = {}
    for k, v in state_dict.items():
        m = LAYER_PREFIX.match(k)
        if not m:
            continue
        idx = int(m.group(1))
        sub = m.group(2)
        layers.setdefault(idx, {})[sub] = v
    return layers


def _find_layer_param(layer_keys: dict[str, torch.Tensor],
                      suffixes: list[tuple[str, str]]) -> torch.Tensor | None:
    """Return the first parameter whose key suffix matches one of the
    listed suffixes. Returns None if none matches.
    """
    for suffix, _arch in suffixes:
        if suffix in layer_keys:
            return layer_keys[suffix]
    return None


def extract_for_cell(cell_dir: str | Path) -> dict | None:
    """Extract trained mechanism parameters from a cell's
    `best_model.pt`. Returns a dict with whichever of `tau`, `alpha`,
    `eta`, `theta_base`, `theta_w1`, `theta_w2` are present, plus
    `n_layers` and a `backbone_family` string. Returns None if the
    state dict is unavailable (e.g., LFS pointer not fetched).
    """
    cell_dir = Path(cell_dir)
    sd = _load_state_dict(cell_dir)
    if sd is None:
        return None
    layers = _layer_key_index(sd)
    if not layers:
        return {"n_layers": 0}

    n_layers = max(layers.keys()) + 1

    # Detect backbone family from the first layer's keys.
    backbone_family = "unknown"
    layer0 = layers.get(0, {})
    if any(k.startswith("att.") for k in layer0.keys()):
        backbone_family = "rwkv6"
    elif any(k.startswith("mamba.") for k in layer0.keys()):
        backbone_family = "mamba2"
    elif any("lucid_temperature" == k or "viscosity_eta" == k or "premix.alpha" == k
              for k in layer0.keys()):
        backbone_family = "linear_attn"

    out: dict = {"n_layers": n_layers, "backbone_family": backbone_family}

    # ---- CVD: tau = softplus(lucid_temperature) ----
    tau_raw_per_layer = []
    for L in range(n_layers):
        t = _find_layer_param(layers.get(L, {}), CVD_SUFFIXES)
        tau_raw_per_layer.append(t)
    if all(t is not None for t in tau_raw_per_layer):
        raw = torch.stack([t.float() for t in tau_raw_per_layer], dim=0)
        tau = torch.nn.functional.softplus(raw).cpu().numpy()
        out["tau_raw"] = raw.cpu().numpy()
        out["tau"] = tau                                 # (n_layers, n_heads)

    # ---- MSDC: alpha (per-dilation mixing weights) ----
    alpha_per_layer = []
    for L in range(n_layers):
        a = _find_layer_param(layers.get(L, {}), MSDC_SUFFIXES)
        alpha_per_layer.append(a)
    if all(a is not None for a in alpha_per_layer):
        out["alpha"] = torch.stack(
            [a.float() for a in alpha_per_layer], dim=0
        ).cpu().numpy()                                  # (n_layers, n_dilations)

    # ---- DHO: viscosity_eta ----
    eta_per_layer = []
    for L in range(n_layers):
        e = _find_layer_param(layers.get(L, {}), DHO_ETA_SUFFIXES)
        eta_per_layer.append(e)
    if all(e is not None for e in eta_per_layer):
        out["eta"] = torch.stack(
            [e.float() for e in eta_per_layer], dim=0
        ).cpu().numpy()                                  # (n_layers, n_heads, n_blocks)

    # ---- DHO theta-init + LoRA ----
    for outname, suffixes in [
        ("theta_base", DHO_THETA_BASE_SUFFIXES),
        ("theta_w1",   DHO_THETA_W1_SUFFIXES),
        ("theta_w2",   DHO_THETA_W2_SUFFIXES),
    ]:
        per_layer = []
        for L in range(n_layers):
            p = _find_layer_param(layers.get(L, {}), suffixes)
            per_layer.append(p)
        if all(p is not None for p in per_layer):
            shapes = [tuple(p.shape) for p in per_layer]
            if len({tuple(s) for s in shapes}) == 1:
                out[outname] = torch.stack(
                    [p.float() for p in per_layer], dim=0
                ).cpu().numpy()
            else:
                # Depth-graded LoRA rank: keep per-layer ndarrays.
                out[outname] = [p.float().cpu().numpy() for p in per_layer]

    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python _extract_mechanism_params.py <cell_dir>")
        sys.exit(1)
    res = extract_for_cell(sys.argv[1])
    if res is None:
        print("(state dict unavailable; LFS pointer or missing file)")
        sys.exit(0)
    print(f"backbone_family = {res.get('backbone_family')}")
    print(f"n_layers        = {res.get('n_layers')}")
    for k, v in res.items():
        if k in ("backbone_family", "n_layers"):
            continue
        if isinstance(v, list):
            print(f"  {k}: list[{len(v)}], shapes = {[a.shape for a in v]}")
        else:
            print(f"  {k}: shape {v.shape}")
