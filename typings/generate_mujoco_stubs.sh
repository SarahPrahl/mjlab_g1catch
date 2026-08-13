#!/usr/bin/env bash

# Generate MuJoCo stubs using the same approach as mjlab:
# https://github.com/mujocolab/mjlab/tree/main/typings

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUBGEN_VERSION="2.5.5"
STUBGEN=(
  uv run --no-sync --with "pybind11-stubgen==${STUBGEN_VERSION}"
  pybind11-stubgen
)

echo "Generating MuJoCo type stubs..."
"${STUBGEN[@]}" mujoco -o "$SCRIPT_DIR" --ignore-all-errors

# mujoco.viewer can fail to import on headless systems. Preserve the committed
# viewer stub in that case so Linux CI can still regenerate the core bindings.
if ! "${STUBGEN[@]}" mujoco.viewer -o "$SCRIPT_DIR" --ignore-all-errors; then
  echo "Warning: could not generate mujoco.viewer; keeping the existing stub."
fi

uv run --no-sync python "$SCRIPT_DIR/postprocess_mujoco_stubs.py"
echo "Generated platform-independent stubs in $SCRIPT_DIR/mujoco."
