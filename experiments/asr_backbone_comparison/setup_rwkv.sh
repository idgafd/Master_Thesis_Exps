#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_rwkv.sh
#
# Clone the RWKV-block repository, apply compatibility patches, and install
# it into the active uv environment as an editable package.
#
# Patches applied:
#   1. Remove the invalid `device=` kwarg from nn.Dropout calls.
#   2. Guard reset_parameters() calls on RWKV7 submodules so they only run
#      when the method actually exists (avoids AttributeError during init).
#
# Usage:
#   bash setup_rwkv.sh              # clone into ./RWKV-block (default)
#   bash setup_rwkv.sh /path/to/dir  # clone into a custom directory
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RWKV_DIR="${1:-$(pwd)/RWKV-block}"
REPO_URL="https://github.com/RWKV/RWKV-block.git"

echo "=== RWKV-block setup ==="
echo "Target directory: $RWKV_DIR"

# ── Clone ────────────────────────────────────────────────────────────────────
if [ -d "$RWKV_DIR/.git" ]; then
    echo "Repository already exists at $RWKV_DIR — skipping clone."
else
    echo "Cloning $REPO_URL ..."
    git clone "$REPO_URL" "$RWKV_DIR"
fi

# ── Patch 1: Remove invalid device= kwarg from nn.Dropout ───────────────────
echo "Applying patch 1: nn.Dropout device= kwarg removal ..."
python3 - "$RWKV_DIR" <<'EOF'
import sys
from pathlib import Path

root = Path(sys.argv[1])
replacements = [
    ("nn.Dropout(p = dropout_rate,device=device)", "nn.Dropout(p = dropout_rate)"),
    ("nn.Dropout(p=dropout_rate,device=device)",   "nn.Dropout(p=dropout_rate)"),
    ("nn.Dropout(p = dropout_rate, device=device)", "nn.Dropout(p = dropout_rate)"),
    ("nn.Dropout(p=dropout_rate, device=device)",   "nn.Dropout(p=dropout_rate)"),
]
patched = 0
for p in root.rglob("*.py"):
    txt = p.read_text(encoding="utf-8")
    new = txt
    for old, rep in replacements:
        new = new.replace(old, rep)
    if new != txt:
        p.write_text(new, encoding="utf-8")
        patched += 1
        print(f"  patched: {p.relative_to(root)}")
print(f"Patch 1 done — {patched} file(s) modified.")
EOF

# ── Patch 2: Guard reset_parameters() calls in RWKV7 ────────────────────────
echo "Applying patch 2: RWKV7 reset_parameters() guard ..."
python3 - "$RWKV_DIR" <<'EOF'
import re, sys
from pathlib import Path

root = Path(sys.argv[1])
path = root / "rwkv_block/v7_goose/block/rwkv7_layer_block.py"

if not path.exists():
    print(f"  File not found: {path} — skipping patch 2.")
    sys.exit(0)

lines = path.read_text(encoding="utf-8").splitlines(True)
out = []
patched = 0
pat = re.compile(r"^(\s*)self\.([A-Za-z_][A-Za-z0-9_]*)\.reset_parameters\(\)\s*$")

for line in lines:
    m = pat.match(line.rstrip("\n"))
    if not m:
        out.append(line)
        continue
    indent, name = m.group(1), m.group(2)
    out.append(f"{indent}{name} = self._modules.get('{name}', None)\n")
    out.append(f"{indent}rp = getattr({name}, 'reset_parameters', None)\n")
    out.append(f"{indent}if callable(rp):\n")
    out.append(f"{indent}    rp()\n")
    patched += 1

path.write_text("".join(out), encoding="utf-8")
print(f"Patch 2 done — {patched} call(s) guarded in {path.relative_to(root)}")
EOF

# ── Bootstrap packaging metadata if upstream repo is source-only ────────────
if [ ! -f "$RWKV_DIR/pyproject.toml" ] && [ ! -f "$RWKV_DIR/setup.py" ]; then
    echo "Bootstrapping minimal pyproject.toml for editable install ..."
    cat > "$RWKV_DIR/pyproject.toml" <<'EOF'
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "rwkv-block"
version = "0.0.0"
dependencies = [
    "torch>=2.5.1",
]

[tool.setuptools]
packages = ["rwkv_block"]

[tool.setuptools.package-dir]
"" = "."
EOF
fi

# ── Install ───────────────────────────────────────────────────────────────────
echo "Installing RWKV-block as editable package ..."
UV_PYTHON="$(pwd)/.venv/bin/python"
if [ ! -x "$UV_PYTHON" ]; then
    UV_PYTHON="$(command -v python3)"
fi
if command -v uv &>/dev/null; then
    uv pip install --python "$UV_PYTHON" -e "$RWKV_DIR"
else
    "$UV_PYTHON" -m pip install -e "$RWKV_DIR"
fi

echo ""
echo "=== RWKV-block setup complete ==="
echo "  Location: $RWKV_DIR"
echo "  Import:   from rwkv_block.v6_finch... / from rwkv_block.v7_goose..."
