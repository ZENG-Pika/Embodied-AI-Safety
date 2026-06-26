# pylint: skip-file
# flake8: noqa
import os
import sys

current_path = os.getcwd()
sys.path.append(f"{current_path}")

import numpy as np
import argparse
import json

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import get_relative_transform
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core import World
from omni.isaac.core.articulations.articulation import Articulation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Use configuration
    dir_name = config['DIR']
    instance_prim_path = os.path.join("/", config["INSTANCE_NAME"])
    link0_initial_prim_path = (config["link0_initial_prim_path"]).replace("/root", "instance")
    base_initial_prim_path = (config["base_initial_prim_path"]).replace("/root", "instance")

    TASK = "close_h_down"
    dir_name_kps = os.path.join(config['DIR'], "Kps", TASK)
    os.makedirs(dir_name_kps, exist_ok=True)

    usd_file = os.path.join(dir_name, "instance.usd")
    keypoint_path = os.path.join(dir_name_kps, "keypoints.json")
    target_keypoint_path = os.path.join(dir_name_kps, "keypoints_final.json")

    if not os.path.exists(keypoint_path):
        print(f"keypoint file {keypoint_path} not found")

    my_world = World()
    reference = add_reference_to_stage(usd_path=usd_file, prim_path=instance_prim_path)
    prim_path = str(reference.GetPrimPath()) 
    prim = Articulation(
        prim_path,
        name=config['INSTANCE_NAME']
    )
    my_world.scene.add(prim)
    instance2link_pose = get_relative_transform(get_prim_at_path(instance_prim_path),get_prim_at_path(os.path.join(instance_prim_path, link0_initial_prim_path)))
    instance2base_pose = get_relative_transform(get_prim_at_path(instance_prim_path),get_prim_at_path(os.path.join(instance_prim_path, base_initial_prim_path)))
    kploc2base = json.load(open(keypoint_path))["keypoints"]
    kplocs = {}
    for name, kploc in kploc2base.items():
        if name == "red" or name == "yellow":
            kploc = np.append(kploc,1)
            kplocs[name] = (instance2link_pose @ kploc).tolist()[:3]
        elif name == "blue":
            kploc = np.append(kploc,1)
            kplocs[name] = (instance2base_pose @ kploc).tolist()[:3]
        else:
            kplocs[name] = kploc

    # compute scale
    my_world.scene.enable_bounding_boxes_computations()
    bbox = my_world.scene.compute_object_AABB(config['INSTANCE_NAME'])
    volume = (bbox[1][0]-bbox[0][0])*(bbox[1][1]-bbox[0][1])*(bbox[1][2]-bbox[0][2])
    scaled_volume=config['SCALED_VOLUME']
    scale = (scaled_volume / volume) **(1/3)

    data = {
        "keypoints": kplocs,
        "scale" : [scale,scale,scale,1.0]
    }
    json.dump(data, open(target_keypoint_path, "w"), indent=4)
    print("Saved keypoints to ", target_keypoint_path)
