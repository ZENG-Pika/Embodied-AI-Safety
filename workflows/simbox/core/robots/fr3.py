"""FR3 robot implementation - Single-arm manipulator."""
from core.robots.base_robot import register_robot
from core.robots.template_robot import TemplateRobot


# pylint: disable=line-too-long,unused-argument
@register_robot
class FR3(TemplateRobot):
    """FR3 single-arm robot with 7-DOF arm and parallel gripper."""

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
        self.fr_forbid_collision_paths = []  #

    def _get_gripper_state(self, gripper_home):
        return 1.0 if gripper_home and gripper_home[0] >= 0.04 else -1.0

    def _build_observations(self, qpos, qvel, T_base_ee_fl, T_world_base):
        from core.utils.transformation_utils import get_fk_solution, pose_to_6d

        gripper_pose = pose_to_6d(get_fk_solution(qpos[self.left_joint_indices]))
        return {
            "states.joint.position": qpos[self.left_joint_indices],
            "states.gripper.position": qpos[self.left_gripper_indices] * 2,
            "states.gripper.pose": gripper_pose,
            "qvel": qvel,
            "T_base_ee_fl": T_base_ee_fl,
            "T_world_base": T_world_base,
        }
