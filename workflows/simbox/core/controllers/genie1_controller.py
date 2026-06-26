"""Genie1 dual-arm controller – template-based."""

import numpy as np
from core.controllers.base_controller import register_controller
from core.controllers.template_controller import TemplateController


# pylint: disable=unused-argument
@register_controller
class Genie1Controller(TemplateController):
    def _get_default_ignore_substring(self):
        return ["material", "Plane", "conveyor", "scene", "table", "fluid"]

    def _configure_joint_indices(self, robot_file: str) -> None:
        if "left" in robot_file:
            self.cmd_js_names = [
                "idx21_arm_l_joint1",
                "idx22_arm_l_joint2",
                "idx23_arm_l_joint3",
                "idx24_arm_l_joint4",
                "idx25_arm_l_joint5",
                "idx26_arm_l_joint6",
                "idx27_arm_l_joint7",
            ]
            self.arm_indices = np.array(self.robot.cfg["left_joint_indices"])
            self.gripper_indices = np.array(self.robot.cfg["left_gripper_indices"])
            self.reference_prim_path = self.task.robots[self.name].fl_base_path
            self.lr_name = "left"
            self._gripper_state = 1.0 if self.robot.left_gripper_state == 1.0 else -1.0
        elif "right" in robot_file:
            self.cmd_js_names = [
                "idx61_arm_r_joint1",
                "idx62_arm_r_joint2",
                "idx63_arm_r_joint3",
                "idx64_arm_r_joint4",
                "idx65_arm_r_joint5",
                "idx66_arm_r_joint6",
                "idx67_arm_r_joint7",
            ]
            self.arm_indices = np.array(self.robot.cfg["right_joint_indices"])
            self.gripper_indices = np.array(self.robot.cfg["right_gripper_indices"])
            self.reference_prim_path = self.task.robots[self.name].fr_base_path
            self.lr_name = "right"
            self._gripper_state = 1.0 if self.robot.right_gripper_state == 1.0 else -1.0
        else:
            raise NotImplementedError("robot_file must contain 'left' or 'right'")
        self.raw_js_names = list(self.cmd_js_names)
        self._gripper_joint_position = np.array([1.0])

    def get_gripper_action(self):
        return np.clip(self._gripper_state * self._gripper_joint_position, 0.0, 1.0)

    def _get_sort_path_weights(self):
        """Genie1: weight joints 4 and 5 (index 4,5) by 3.0 for path selection."""
        return [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0]

    def mobile_move(self, target: np.ndarray, joint_indices: np.ndarray = None, initial_position: np.ndarray = None):
        raise NotImplementedError
