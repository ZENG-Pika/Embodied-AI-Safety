#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
exec python3 scripts/run_safety_scenarios.py \
    --config configs/simbox/safety_scenarios_3dgs_usdz_native_isaac50_visual.yaml \
    --run \
    --scenario pick_and_place/franka/single_pick/omniobject3d-dish \
    --policy trained-diffusion \
    --checkpoint models/dp_franka_100k_delivery/checkpoint \
    --model-root models/dp_franka_100k_delivery \
    --policy-python models/dp_franka_100k_delivery/.venv/bin/python \
    --policy-device cuda \
    --policy-seed "${POLICY_SEED:-7}" \
    --policy-replan-steps "${POLICY_REPLAN_STEPS:-8}" \
    --max-episode-steps "${MAX_EPISODE_STEPS:-500}" \
    --evaluation-output-root "${NATIVE_NUREC_OUTPUT_ROOT:-output/native_nurec_policy_visual}" \
    --evaluation-failure-root "${NATIVE_NUREC_FAILURE_ROOT:-failure_output/native_nurec_policy_visual}" \
    --no-retry \
    "$@"
