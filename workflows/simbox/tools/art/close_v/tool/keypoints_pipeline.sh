# pylint: skip-file
# flake8: noqa
# Replace the following keypoints_config path with your absolute path
CONFIG_PATH="YOUR_PATH_TO_7265/usd/keypoints_config.json"

# Replace the following close_v_new path with your absolute path
cd workflows/simbox/tools/art/close_v/tool

# Run the following scripts in sequence

# 1. rehier - This should generate peixun/7265/usd/instance.usd file to indicate success
python rehier.py --config $CONFIG_PATH

# 2. select points
python select_keypoint.py --config $CONFIG_PATH

# 3. Transfer keypoints
python transfer_keypoints.py --config $CONFIG_PATH

# 4. Overwrite keypoints
python overwrite_keypoints.py --config $CONFIG_PATH
