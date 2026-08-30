#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
exec "$SCRIPT_DIR/python.sh" launcher.py \
    --config configs/simbox/de_hand_avoidance_isaac50.yaml \
    --random_seed "${RANDOM_SEED:-0}" \
    "$@"
