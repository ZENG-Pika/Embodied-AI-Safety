"""FrankaRobotiq85 robot implementation - Single-arm with Robotiq 2F-85 gripper."""
import json

import numpy as np
from core.robots.base_robot import register_robot
from core.robots.template_robot import TemplateRobot
from scipy.interpolate import interp1d


# pylint: disable=line-too-long,unused-argument
@register_robot
class FrankaRobotiq85(TemplateRobot):
    """Franka robot with Robotiq 2F-85 gripper."""

    def _setup_joint_indices(self):
        self.left_joint_indices = self.cfg["left_joint_indices"]
        self.right_joint_indices = []
        self.left_gripper_indices = self.cfg["left_gripper_indices"]
        self.right_gripper_indices = []
        self.body_indices = []
        self.head_indices = []
        self.lift_indices = []

    def _setup_paths(self):
        fl_ee_path = self.cfg["fl_ee_path"]
        self.fl_ee_path = f"{self.robot_prim_path}/{fl_ee_path}"
        self.fl_base_path = f"{self.robot_prim_path}/{self.cfg['fl_base_path']}"
        self.fl_hand_path = self.fl_ee_path
        self.fr_base_path = ""
        self.fr_hand_path = ""
        self.fr_ee_path = ""

    def _setup_gripper_keypoints(self):
        self.fl_gripper_keypoints = self.cfg["fl_gripper_keypoints"]
        self.fr_gripper_keypoints = {}

    def _setup_collision_paths(self):
        self.fl_filter_paths_expr = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_filter_paths"]]
        self.fr_filter_paths_expr = []
        self.fl_forbid_collision_paths = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_forbid_collision_paths"]]
        self.fr_forbid_collision_paths = []

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
        if not gripper_home or len(gripper_home) < 2:
            return 1.0
        return 1.0 if (gripper_home[0] == 0.0 and gripper_home[1] == 0.0) else -1.0

    def _build_observations(self, qpos, qvel, T_base_ee_fl, T_world_base):
        from core.utils.transformation_utils import get_fk_solution, pose_to_6d

        gripper_pose = pose_to_6d(get_fk_solution(qpos[self.left_joint_indices]))
        gripper_position = 1.0 if (qpos[self.left_gripper_indices][0] <= 0.4) else 0.0
        return {
            "states.joint.position": qpos[self.left_joint_indices],
            "states.gripper.position": gripper_position,
            "states.gripper.pose": gripper_pose,
            "qvel": qvel,
            "T_base_ee_fl": T_base_ee_fl,
            "T_world_base": T_world_base,
        }
