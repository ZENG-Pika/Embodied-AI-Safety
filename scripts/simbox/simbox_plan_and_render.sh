#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Error: Missing required parameter"
    echo "Usage: bash $0 <config_path> [random_num] [random_seed]"
    echo ""
    echo "Parameters:"
    echo "  config_path  - Full path to the config file (with .yml extension)"
    echo "  random_num   - (Optional) Number of samples to generate (default: 10)"
    echo "  random_seed  - (Optional) Random seed for reproducibility"
    echo "  scene_info   - (Optional) Scene info key to use"
    echo ""
    echo "Example:"
    echo "  bash $0 workflows/simbox/core/configs/tasks/long_horizon/split_aloha/sort_the_rubbish/sort_the_rubbish_part0.yaml"
    echo "  bash $0 workflows/simbox/core/configs/tasks/long_horizon/split_aloha/sort_the_rubbish/sort_the_rubbish_part0.yaml 10"
    echo "  bash $0 workflows/simbox/core/configs/tasks/long_horizon/split_aloha/sort_the_rubbish/sort_the_rubbish_part0.yaml 10 42"
    echo "  bash $0 workflows/simbox/core/configs/tasks/long_horizon/split_aloha/sort_the_rubbish/sort_the_rubbish_part0.yaml 10 42 living_room_scene_info"
    exit 1
fi

cfg_path="$1"
random_num=10
if [ $# -ge 2 ]; then
    random_num="$2"
fi
random_seed=""
if [ $# -ge 3 ]; then
    random_seed="$3"
fi
scene_info=""
if [ $# -ge 4 ]; then
    scene_info="$4"
fi

if [ -z "$cfg_path" ]; then
    echo "Error: Config path parameter is required and cannot be empty"
    exit 1
fi

# Extract custom_path and config_name from the full path
custom_path=$(dirname "$cfg_path")
config_name=$(basename "$cfg_path" .yaml)

echo "Config path: $cfg_path"
echo "Custom path: $custom_path"
echo "Config name: $config_name"
echo "Random num: $random_num"

if [ ! -f "$cfg_path" ]; then
    echo "Error: Configuration file not found: $cfg_path"
    exit 1
fi

name_with_split="${config_name}_plan_and_render${random_seed:+_seed_${random_seed}}"

echo "Running with config: $cfg_path"
echo "Output name: $name_with_split"

set -x
/isaac-sim/python.sh launcher.py --config configs/simbox/de_plan_and_render_template.yaml \
--name="$name_with_split" \
--load_stage.scene_loader.args.cfg_path="$cfg_path" \
--load_stage.layout_random_generator.args.random_num="$random_num" \
--store_stage.writer.args.output_dir="output/$name_with_split/" \
${scene_info:+--load_stage.env_loader.args.scene_info="$scene_info"} \
${random_seed:+--random_seed="$random_seed"}
set +x
