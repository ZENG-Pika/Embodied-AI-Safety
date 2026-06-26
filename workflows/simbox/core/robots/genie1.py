"""Genie1 robot implementation - Dual-arm with body and head joints."""
import json

import numpy as np
from core.robots.base_robot import register_robot
from core.robots.template_robot import TemplateRobot
from scipy.interpolate import interp1d


# pylint: disable=line-too-long,unused-argument
@register_robot
class Genie1(TemplateRobot):
    """Genie1 dual-arm robot with body and head joints."""

    def _setup_joint_indices(self):
        self.left_joint_indices = self.cfg["left_joint_indices"]
        self.right_joint_indices = self.cfg["right_joint_indices"]
        self.left_gripper_indices = self.cfg["left_gripper_indices"]
        self.right_gripper_indices = self.cfg["right_gripper_indices"]
        self.body_indices = self.cfg["body_indices"]
        self.head_indices = self.cfg["head_indices"]
        self.lift_indices = []

    def _setup_paths(self):
        fl_ee_path = self.cfg["fl_ee_path"]
        fr_ee_path = self.cfg["fr_ee_path"]
        self.fl_ee_path = f"{self.robot_prim_path}/{fl_ee_path}"
        self.fr_ee_path = f"{self.robot_prim_path}/{fr_ee_path}"
        self.fl_base_path = f"{self.robot_prim_path}/{self.cfg['fl_base_path']}"
        self.fr_base_path = f"{self.robot_prim_path}/{self.cfg['fr_base_path']}"
        self.fl_hand_path = self.fl_ee_path
        self.fr_hand_path = self.fr_ee_path

    def _setup_gripper_keypoints(self):
        self.fl_gripper_keypoints = self.cfg["fl_gripper_keypoints"]
        self.fr_gripper_keypoints = self.cfg["fr_gripper_keypoints"]

    def _setup_collision_paths(self):
        self.fl_filter_paths_expr = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_filter_paths"]]
        self.fr_filter_paths_expr = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fr_filter_paths"]]
        self.fl_forbid_collision_paths = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_forbid_collision_paths"]]
        self.fr_forbid_collision_paths = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fr_forbid_collision_paths"]]

    def _load_extra_depth(self, usd_path):
        json_path = usd_path.replace("robot.usd", "tcp2base_offset.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_dict = json.load(f)
            keys = list(json_dict.keys())
            widths = np.array([json_dict[key]["width"] for key in keys[1:]])
            extra_depths = np.array([json_dict[key]["offset"] for key in keys[1:]])
            self._gripper_ed_func = interp1d(widths, extra_depths, kind="cubic")
            self.gripper_max_width = widths.max()
            self.gripper_min_width = widths.min()
        except Exception:
            self._gripper_ed_func = None

    def _get_gripper_state(self, gripper_home):
        return 1.0 if gripper_home and gripper_home[0] == 1.0 else -1.0

    def _setup_joint_velocities(self):
        # Genie1 has 18 joints for velocity control
        all_joint_indices = self.body_indices + self.head_indices + self.left_joint_indices + self.right_joint_indices
        if all_joint_indices:
            self._articulation_view.set_max_joint_velocities(
                np.array([500.0] * 18),
                joint_indices=np.array(all_joint_indices),
            )

    def apply_action(self, joint_positions, joint_indices, *args, **kwargs):
        self._articulation_view.set_joint_position_targets(joint_positions, joint_indices=joint_indices)

        # Genie1 specific: gripper velocity control
        gripper_velocities, gripper_indices = [], []
        for idx in range(len(joint_positions) // 8):
            target_qpos = joint_positions[idx + 7]
            if target_qpos == 0.0:  # close
                gripper_velocities.extend([-100, -100])
                gripper_indices.extend([joint_indices[idx + 7], joint_indices[idx + 7] - 1])
            if target_qpos == 1.0:  # open
                gripper_velocities.extend([100, 100])
                gripper_indices.extend([joint_indices[idx + 7], joint_indices[idx + 7] - 1])

        if gripper_velocities and gripper_indices:
            self._articulation_view.set_joint_velocity_targets(
                np.array(gripper_velocities), joint_indices=np.array(gripper_indices)
            )
