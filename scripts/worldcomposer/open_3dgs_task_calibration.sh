#!/usr/bin/env bash
# Create a pure-3DGS bundle, import static task assets, and open Isaac Sim for calibration.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  scripts/worldcomposer/open_3dgs_task_calibration.sh \
    --scene-id SCENE_ID --nurec PATH/TO/3DGS.usdz [options]

Create a 3DGS-only WorldComposer bundle, reference the task's static assets,
and open the resulting calibration USD in Isaac Sim. This never runs a task,
controller, planner, physics simulation, or policy.

Options:
  --scene-id ID            Required, unique bundle identifier.
  --nurec PATH             Required, WorldComposer-exported .usdz file.
  --task PATH              Task YAML; default is Franka single-pick dish.
  --asset-root PATH        InternDataAssets/assets directory.
  --output-root PATH       Bundle root; default assets/worldcomposer.
  --overwrite              Replace an existing bundle. This deletes its prior calibration.
  --no-hand                Do not add the fixed MANO hand reference.
  --width N                Isaac Sim viewport width; default 1280.
  --height N               Isaac Sim viewport height; default 720.
  --eye X Y Z              Editable Perspective camera position.
  --target X Y Z           Editable Perspective camera look-at point.
  --focal-length-mm N      Perspective camera focal length; default 18.
  -h, --help               Show this help.

Example:
  scripts/worldcomposer/open_3dgs_task_calibration.sh \
    --scene-id scene21 --nurec "$PWD/scene/840347.usdz"
EOF
}

scene_id=""
nurec=""
task=""
asset_root=""
output_root="$REPO_ROOT/assets/worldcomposer"
overwrite=false
no_hand=false
width=1280
height=720
eye=(0 0 0.2)
target=(6 0 0.2)
focal_length_mm=18

while (($#)); do
    case "$1" in
        --scene-id) scene_id="${2:?missing value for --scene-id}"; shift 2 ;;
        --nurec) nurec="${2:?missing value for --nurec}"; shift 2 ;;
        --task) task="${2:?missing value for --task}"; shift 2 ;;
        --asset-root) asset_root="${2:?missing value for --asset-root}"; shift 2 ;;
        --output-root) output_root="${2:?missing value for --output-root}"; shift 2 ;;
        --overwrite) overwrite=true; shift ;;
        --no-hand) no_hand=true; shift ;;
        --width) width="${2:?missing value for --width}"; shift 2 ;;
        --height) height="${2:?missing value for --height}"; shift 2 ;;
        --eye) eye=("${2:?missing X for --eye}" "${3:?missing Y for --eye}" "${4:?missing Z for --eye}"); shift 4 ;;
        --target) target=("${2:?missing X for --target}" "${3:?missing Y for --target}" "${4:?missing Z for --target}"); shift 4 ;;
        --focal-length-mm) focal_length_mm="${2:?missing value for --focal-length-mm}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$scene_id" || -z "$nurec" ]]; then
    printf '%s\n\n' '--scene-id and --nurec are required.' >&2
    usage >&2
    exit 2
fi

prepare_args=(
    python3 "$SCRIPT_DIR/prepare_3dgs_only_bundle.py"
    --scene-id "$scene_id"
    --nurec "$nurec"
    --output-root "$output_root"
)
if [[ "$overwrite" == true ]]; then
    prepare_args+=(--overwrite)
fi
"${prepare_args[@]}"

bundle_dir="$output_root/$scene_id"
task_args=(
    python3 "$SCRIPT_DIR/create_task_calibration_stage.py"
    --bundle-dir "$bundle_dir"
)
if [[ -n "$task" ]]; then
    task_args+=(--task "$task")
fi
if [[ -n "$asset_root" ]]; then
    task_args+=(--asset-root "$asset_root")
fi
if [[ "$no_hand" == true ]]; then
    task_args+=(--no-hand)
fi
"${task_args[@]}"

stage="$bundle_dir/${scene_id}_TaskCalibration.usda"
printf '[worldcomposer] STATUS=SUCCESS step=create_and_open_task_calibration stage=%s\n' "$stage"
exec "$REPO_ROOT/scripts/isaac50/python.sh" "$REPO_ROOT/scripts/isaac50/open_stage_gui.py" \
    "$stage" \
    --nurec \
    --perspective \
    --eye "${eye[@]}" \
    --target "${target[@]}" \
    --focal-length-mm "$focal_length_mm" \
    --width "$width" \
    --height "$height"
