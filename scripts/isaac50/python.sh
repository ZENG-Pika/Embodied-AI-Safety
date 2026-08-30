#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_ROOT="${ISAAC_SIM_50_ROOT:-/home/zxw/isaacsim-5.0}"

if [[ -n "${ISAACSIM_PYTHON:-}" ]]; then
    exec "$ISAACSIM_PYTHON" "$@"
fi

exec "$SCRIPT_DIR/run_in_ubuntu22.sh" "$ISAAC_ROOT/python.sh" "$@"
