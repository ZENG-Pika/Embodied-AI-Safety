import glob
import json
import os
import random

import numpy as np
from core.objects.base_object import register_object
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage

try:
    from omni.isaac.core.materials.omni_pbr import OmniPBR  # Isaac Sim 4.1.0 / 4.2.0
except ImportError:
    from isaacsim.core.api.materials import OmniPBR  # Isaac Sim 4.5.0


@register_object
class ArticulatedObject(Articulation):
    def __init__(self, asset_root, root_prim_path, cfg, *args, **kwargs):
        self.asset_root = asset_root
        self.object_name = cfg["name"]
        self.usd_path = os.path.join(asset_root, cfg["path"])
        self._root_prim_path = root_prim_path
        info_name = cfg["info_name"]
        object_info_path = self.usd_path.replace("instance.usd", f"Kps/{info_name}/info.json")
        with open(object_info_path, "r", encoding="utf-8") as f:
            object_info = json.load(f)
        self.category = cfg["category"]
        self.cfg = cfg
        self.get_articulated_info(object_info)
        self.cfg["scale"] = self.object_scale[:3]
        super().__init__(prim_path=self.object_prim_path, name=cfg["name"], *args, **kwargs)

    def update_articulated_info(self, obj_info_path):
        object_info_path = f"{self.asset_root}/{obj_info_path}"
        with open(object_info_path, "r", encoding="utf-8") as f:
            object_info = json.load(f)

        prim_path_list = ["object_prim_path", "object_link_path", "object_base_path"]
        axis_list = [
            "object_link0_rot_axis",
            "object_link0_move_axis",
            "object_link0_contact_axis",
            "object_base_front_axis",
        ]
        joint_path_list = ["object_joint_path", "object_revolute_joint_path"]
        joint_idx_list = ["joint_index", "object_revolute_joint_idx"]

        self.object_keypoints = object_info["object_keypoints"]  # Object keypoints in link_0 frame
        for key, item in self.object_keypoints.items():
            self.object_keypoints[key] = np.append(item, [1.0], axis=0)

        self.object_scale = np.array(object_info["object_scale"])

        for key, item in object_info.items():
            if key in prim_path_list:
                setattr(self, key, self._root_prim_path + object_info[key])
            elif key in axis_list:
                setattr(self, key, object_info[key])
            elif key in joint_path_list:
                self.object_joint_path = self._root_prim_path + object_info[key]
            elif key in joint_idx_list:
                self.object_joint_index = object_info[key]

        self.articulation_initial_joint_position = self._articulation_view.get_joint_positions()[
            :, self.object_joint_index
        ]

        self.contact_plane_normal = None

    def get_articulated_info(self, object_info):
        prim_path_list = ["object_prim_path", "object_link_path", "object_base_path"]
        axis_list = [
            "object_link0_rot_axis",
            "object_link0_move_axis",
            "object_link0_contact_axis",
            "object_base_front_axis",
        ]
        joint_path_list = ["object_joint_path", "object_revolute_joint_path"]
        joint_idx_list = ["joint_index", "object_revolute_joint_idx"]
        self.object_keypoints = object_info["object_keypoints"]  # Object keypoints in link_0 frame
        for key, item in self.object_keypoints.items():
            self.object_keypoints[key] = np.append(item, [1.0], axis=0)
        self.object_scale = np.array(object_info["object_scale"])
        for key, item in object_info.items():
            if key in prim_path_list:
                setattr(self, key, self._root_prim_path + object_info[key])
            elif key in axis_list:
                setattr(self, key, object_info[key])
            elif key in joint_path_list:
                self.object_joint_path = self._root_prim_path + object_info[key]
            elif key in joint_idx_list:
                self.object_joint_index = object_info[key]

        # Compute joint number
        self.object_joint_number = 0
        # Contact plane normal
        self.contact_plane_normal = None
        add_reference_to_stage(usd_path=self.usd_path, prim_path=self.object_prim_path)

    def get_joint_position(self, stage):
        joint_parent_prim = stage.GetPrimAtPath(self.object_joint_path.rsplit("/", 1)[0])
        for child in joint_parent_prim.GetAllChildren():
            if child.GetTypeName() == "PhysicsPrismaticJoint" or child.GetTypeName() == "PhysicsRevoluteJoint":
                self.object_joint_number += 1

        # Fix the asset base
        if self.cfg.get("fix_base", False):
            parent_prim_path = os.path.dirname(self.object_base_path)
            child_prim_path = self.object_base_path
            joint_prim_path = os.path.join(os.path.dirname(self.object_base_path), "FixedJoint")
            joint_prim = stage.DefinePrim(joint_prim_path, "PhysicsFixedJoint")
            joint_prim.GetRelationship("physics:body0").AddTarget(child_prim_path)
            joint_prim.GetRelationship("physics:body1").AddTarget(parent_prim_path)

    def get_observations(self):
        translation, orientation = self.get_local_pose()
        obs = {
            "translation": translation,
            "orientation": orientation,
        }
        return obs

    def apply_texture(self, asset_root, cfg):
        texture_name = cfg["texture_lib"]
        texture_path_list = glob.glob(os.path.join(asset_root, texture_name, "*.jpg"))
        texture_path_list.sort()
        if cfg["apply_randomization"]:
            texture_id = random.randint(0, len(texture_path_list) - 1)
        else:
            texture_id = cfg["texture_id"]
        texture_path = texture_path_list[texture_id]
        mat_prim_path = f"{self.base_prim_path}/Looks/Material"
        mat = OmniPBR(
            prim_path=mat_prim_path,
            name="Material",
            texture_path=texture_path,
            texture_scale=cfg.get("texture_scale"),
        )
        self.apply_visual_material(mat)

    def initialize(self):
        super().initialize()
        self._articulation_view.set_joint_velocities([0.0])
        if "joint_position_range" in self.cfg:
            self.articulation_initial_joint_position = np.random.uniform(
                low=self.cfg["joint_position_range"][0], high=self.cfg["joint_position_range"][1]
            )
            self._articulation_view.set_joint_positions(
                self.articulation_initial_joint_position, joint_indices=self.object_joint_index
            )

        if "strict_init" in self.cfg:
            self._articulation_view.set_joint_position_targets(
                self.cfg["strict_init"]["joint_positions"], joint_indices=self.cfg["strict_init"]["joint_indices"]
            )
        # self._articulation_view.set_joint_position_targets(self.target_joint_position)
