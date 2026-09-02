#!/usr/bin/env bash
set -euo pipefail

# Portable hand-avoidance launcher (Isaac Sim 5 by default).
# Override when needed:
#   ISAACSIM_ROOT=/path/to/isaacsim ./scripts/run_hand_avoidance.sh
#   CONFIG=configs/simbox/hand_avoidance.yaml ./scripts/run_hand_avoidance.sh

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: ./scripts/run_hand_avoidance.sh

Environment variables:
  ISAACSIM_ROOT  Isaac Sim install root containing python.sh (optional)
  CUROBO_ROOT    CuRobo source root containing src/curobo (4.5 mode only)
  CONFIG         Config path, default configs/simbox/hand_avoidance.yaml
  LOG_FILE       Optional log path. If set, stdout/stderr are redirected there.

Examples:
  ./scripts/run_hand_avoidance.sh
  ISAACSIM_ROOT=/home/pika/Software/isaacsim ./scripts/run_hand_avoidance.sh
  ISAACSIM_ROOT=/home/pika/Software/isaacsim4.5 ./scripts/run_hand_avoidance.sh
EOF
  exit 0
fi

CUROBO_ROOT="${CUROBO_ROOT:-/home/pika/Workspace/pika/InternDataEngine/InternDataAssets/curobo}"
CONFIG="${CONFIG:-configs/simbox/hand_avoidance.yaml}"
LOG_FILE="${LOG_FILE:-}"

if [[ -z "${ISAACSIM_ROOT:-}" ]]; then
  for candidate in \
    "/home/pika/Software/isaacsim-5.0.0" \
    "/home/pika/Software/isaacsim5.0" \
    "/home/pika/isaacsim-5.0.0" \
    "/home/pika/Software/isaacsim" \
    "/home/pika/Software/isaacsim4.5" \
    "/home/pika/isaacsim-4.5.0" \
    "/home/wp/isaacsim-4.1.0" \
    "/home/wp/isaacsim4.5" \
    "/opt/isaacsim"; do
    if [[ -x "$candidate/python.sh" ]]; then
      ISAACSIM_ROOT="$candidate"
      break
    fi
  done
fi

if [[ -z "${ISAACSIM_ROOT:-}" || ! -x "$ISAACSIM_ROOT/python.sh" ]]; then
  echo "ERROR: Could not find Isaac Sim python.sh." >&2
  echo "Set ISAACSIM_ROOT=/home/pika/Software/isaacsim" >&2
  exit 2
fi

if [[ ! -d "$CUROBO_ROOT/src/curobo" ]]; then
  echo "ERROR: Could not find CuRobo sources at $CUROBO_ROOT/src/curobo." >&2
  exit 2
fi
export PYTHONPATH="$CUROBO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "$ISAACSIM_ROOT/VERSION" ]] && grep -qE '^5\.' "$ISAACSIM_ROOT/VERSION"; then
  export INTERNDATA_ISAAC5_COMPAT=1
else
  unset INTERNDATA_ISAAC5_COMPAT
fi

if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  export TORCH_CUDA_ARCH_LIST
  echo "Using TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
fi

echo "Using ISAACSIM_ROOT=$ISAACSIM_ROOT"
echo "Using CONFIG=$CONFIG"

if [[ -n "$LOG_FILE" ]]; then
  "$ISAACSIM_ROOT/python.sh" launcher.py --config "$CONFIG" > "$LOG_FILE" 2>&1
else
  "$ISAACSIM_ROOT/python.sh" launcher.py --config "$CONFIG"
fi
