#!/usr/bin/env bash
set -euo pipefail

# Portable hand-avoidance launcher.
# Override when needed:
#   ISAACSIM_ROOT=/path/to/isaacsim ./scripts/run_hand_avoidance.sh
#   CONFIG=configs/simbox/de_hand_avoidance.yaml ./scripts/run_hand_avoidance.sh

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: ./scripts/run_hand_avoidance.sh

Environment variables:
  ISAACSIM_ROOT  IsaacSim install root containing python.sh
  CONFIG         Config path, default configs/simbox/de_hand_avoidance.yaml
  LOG_FILE       Optional log path. If set, stdout/stderr are redirected there.

Examples:
  ./scripts/run_hand_avoidance.sh
  ISAACSIM_ROOT=/home/pika/Software/isaacsim4.5 ./scripts/run_hand_avoidance.sh
  ISAACSIM_ROOT=/home/wp/isaacsim-4.1.0 LOG_FILE=/tmp/hand-avoidance-run.log ./scripts/run_hand_avoidance.sh
EOF
  exit 0
fi

CONFIG="${CONFIG:-configs/simbox/de_hand_avoidance.yaml}"
LOG_FILE="${LOG_FILE:-}"

if [[ -z "${ISAACSIM_ROOT:-}" ]]; then
  for candidate in     "/home/pika/Software/isaacsim4.5"     "/home/pika/isaacsim-4.5.0"     "/home/pika/isaacsim4.5"     "/home/wp/isaacsim-4.1.0"     "/home/wp/isaacsim4.5"     "/opt/isaacsim"; do
    if [[ -x "$candidate/python.sh" ]]; then
      ISAACSIM_ROOT="$candidate"
      break
    fi
  done
fi

if [[ -z "${ISAACSIM_ROOT:-}" || ! -x "$ISAACSIM_ROOT/python.sh" ]]; then
  echo "ERROR: Could not find IsaacSim python.sh." >&2
  echo "Set ISAACSIM_ROOT, for example:" >&2
  echo "  ISAACSIM_ROOT=/home/pika/Software/isaacsim4.5 ./scripts/run_hand_avoidance.sh" >&2
  echo "  ISAACSIM_ROOT=/home/wp/isaacsim-4.1.0 ./scripts/run_hand_avoidance.sh" >&2
  exit 2
fi

# Some IsaacSim installs keep project-compatible torch wheels outside the default python path.
if [[ -d "$ISAACSIM_ROOT/torch-cu128" ]]; then
  export PYTHONPATH="$ISAACSIM_ROOT/torch-cu128${PYTHONPATH:+:$PYTHONPATH}"
  for libdir in     "$ISAACSIM_ROOT/torch-cu128/nvidia/cudnn/lib"     "$ISAACSIM_ROOT/torch-cu128/nvidia/nccl/lib"     "$ISAACSIM_ROOT/torch-cu128/nvidia/cusparselt/lib"; do
    if [[ -d "$libdir" ]]; then
      export LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  done
fi

if [[ -d /usr/local/cuda-12.8 ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
  export PATH="/usr/local/cuda-12.8/bin:$PATH"
  export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

echo "Using ISAACSIM_ROOT=$ISAACSIM_ROOT"
echo "Using CONFIG=$CONFIG"

if [[ -n "$LOG_FILE" ]]; then
  "$ISAACSIM_ROOT/python.sh" launcher.py --config "$CONFIG" > "$LOG_FILE" 2>&1
else
  "$ISAACSIM_ROOT/python.sh" launcher.py --config "$CONFIG"
fi
