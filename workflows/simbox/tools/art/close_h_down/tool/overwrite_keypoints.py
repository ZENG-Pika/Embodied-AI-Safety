# pylint: skip-file
# flake8: noqa
import os
import json
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config file")
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)

TASK = "close_h_down"
dir_name = config["DIR"]
dir_name_kps = os.path.join(config['DIR'], "Kps", TASK)
os.makedirs(dir_name_kps, exist_ok=True)

usd_file = os.path.join(dir_name, "instance.usd")
keypoint_path = os.path.join(dir_name_kps, "keypoints_final.json")
target_keypoint_path = os.path.join(dir_name_kps, "info.json")

if not os.path.exists(target_keypoint_path):
    with open(target_keypoint_path,'w') as file:
        data={"object_keypoints":{}}
        json.dump(data,file,indent=4)

if not os.path.exists(keypoint_path) or not os.path.exists(target_keypoint_path):
    print(f"keypoint file {keypoint_path} or {target_keypoint_path} not found")

kp = json.load(open(keypoint_path))
tkp = json.load(open(target_keypoint_path))

tkp["object_keypoints"]["articulated_object_head"] = kp["keypoints"]["red"]
tkp["object_keypoints"]["articulated_object_tail"] = kp["keypoints"]["yellow"]

tkp["object_scale"] = kp["scale"]
tkp["object_name"] = config["INSTANCE_NAME"]
tkp["object_usd"] = usd_file
tkp["object_link0_rot_axis"] = config["LINK0_ROT_AXIS"]
tkp["object_link0_contact_axis"] = config["LINK0_CONTACT_AXIS"]
tkp["object_base_front_axis"] = config["BASE_FRONT_AXIS"]
tkp["joint_index"] = config["joint_index"]

tkp["object_prim_path"] = os.path.join("/", config["INSTANCE_NAME"])
link0_initial_prim_path = (config["link0_initial_prim_path"]).replace("/root", "instance")
base_initial_prim_path = (config["base_initial_prim_path"]).replace("/root", "instance")
revolute_joint_initial_prim_path = (config["revolute_joint_initial_prim_path"]).replace("/root", "instance")

tkp["object_link_path"] = os.path.join("/", config["INSTANCE_NAME"], link0_initial_prim_path)
tkp["object_base_path"] = os.path.join("/", config["INSTANCE_NAME"], base_initial_prim_path)
tkp["object_revolute_joint_path"] = os.path.join("/", config["INSTANCE_NAME"], revolute_joint_initial_prim_path)

json.dump(tkp, open(target_keypoint_path, "w"), indent=4)
print("Saved keypoints to ", target_keypoint_path)
