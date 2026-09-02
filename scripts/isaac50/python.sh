#!/usr/bin/env bash
set -euo pipefail

# Isaac Sim 5 launcher used by the safety runner. The installation is kept
# outside the repository and can be overridden for another host with
# ISAAC_SIM_50_ROOT (or ISAACSIM_ROOT for compatibility with the legacy
# launcher). Ubuntu 24 on labnew can run the standalone package directly;
# hosts that need a user-space rootfs may opt into bubblewrap explicitly.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${ISAACSIM_PYTHON:-}" ]]; then
    exec "$ISAACSIM_PYTHON" "$@"
fi

ISAAC_ROOT="${ISAAC_SIM_50_ROOT:-${ISAACSIM_ROOT:-}}"
if [[ -z "$ISAAC_ROOT" ]]; then
    for candidate in \
        "/home/pika/Software/isaacsim-5.0.0" \
        "/home/pika/Software/isaacsim5.0" \
        "/home/pika/Software/isaac-sim-5.0.0" \
        "/home/pika/Software/isaacsim" \
        "/home/pika/Software/isaacsim-5.0" \
        "/home/zxw/isaacsim-5.0"; do
        if [[ -x "$candidate/python.sh" ]]; then
            ISAAC_ROOT="$candidate"
            break
        fi
    done
fi

if [[ -z "$ISAAC_ROOT" || ! -x "$ISAAC_ROOT/python.sh" ]]; then
    echo "ERROR: Isaac Sim 5 python.sh not found." >&2
    echo "Set ISAAC_SIM_50_ROOT=/path/to/isaacsim-5.0" >&2
    exit 2
fi

if [[ "${ISAAC_SIM_USE_BWRAP:-0}" == "1" ]]; then
    exec "$SCRIPT_DIR/run_in_ubuntu22.sh" "$ISAAC_ROOT/python.sh" "$@"
fi

exec "$ISAAC_ROOT/python.sh" "$@"
