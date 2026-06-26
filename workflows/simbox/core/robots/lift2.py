"""Lift2 robot implementation - Dual-arm with lift joint."""
import numpy as np
from core.robots.base_robot import register_robot
from core.robots.template_robot import TemplateRobot


# pylint: disable=line-too-long,unused-argument
@register_robot
class Lift2(TemplateRobot):
    """Lift2 dual-arm robot with a lift joint for vertical movement."""

    def _setup_joint_indices(self):
        self.left_joint_indices = self.cfg["left_joint_indices"]
        self.right_joint_indices = self.cfg["right_joint_indices"]
        self.left_gripper_indices = self.cfg["left_gripper_indices"]
        self.right_gripper_indices = self.cfg["right_gripper_indices"]
        self.body_indices = []
        self.head_indices = []
        self.lift_indices = self.cfg["lift_indices"]

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

    def _get_gripper_state(self, gripper_home):
        return 1.0 if gripper_home and gripper_home[0] >= 0.044 else -1.0

    def _setup_joint_velocities(self):
        # Lift2 has 13 joints for velocity control
        all_joint_indices = self.lift_indices + self.left_joint_indices + self.right_joint_indices
        if all_joint_indices:
            self._articulation_view.set_max_joint_velocities(
                np.array([500.0] * 13),
                joint_indices=np.array(all_joint_indices),
            )
