"""Lift2 mobile manipulator controller – template-based."""

import numpy as np
from core.controllers.base_controller import register_controller
from core.controllers.template_controller import TemplateController


# pylint: disable=unused-argument
@register_controller
class Lift2Controller(TemplateController):
    def _get_default_ignore_substring(self):
        return ["material", "Plane", "conveyor", "scene", "table", "fluid"]

    def _configure_joint_indices(self, robot_file: str) -> None:
        self.raw_js_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        if "left" in robot_file:
            self.cmd_js_names = ["fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4", "fl_joint5", "fl_joint6"]
            self.arm_indices = np.array(self.robot.cfg["left_joint_indices"])
            self.gripper_indices = np.array(self.robot.cfg["left_gripper_indices"])
            self.reference_prim_path = self.task.robots[self.name].fl_base_path
            self.lr_name = "left"
            self._gripper_state = 1.0 if self.robot.left_gripper_state == 1.0 else -1.0
        elif "right" in robot_file:
            self.cmd_js_names = ["fr_joint1", "fr_joint2", "fr_joint3", "fr_joint4", "fr_joint5", "fr_joint6"]
            self.arm_indices = np.array(self.robot.cfg["right_joint_indices"])
            self.gripper_indices = np.array(self.robot.cfg["right_gripper_indices"])
            self.reference_prim_path = self.task.robots[self.name].fr_base_path
            self.lr_name = "right"
            self._gripper_state = 1.0 if self.robot.right_gripper_state == 1.0 else -1.0
        else:
            raise NotImplementedError("robot_file must contain 'left' or 'right'")
        self._gripper_joint_position = np.array([1.0])

    def get_gripper_action(self):
        return np.clip(self._gripper_state * self._gripper_joint_position, 0.0, 0.1)

    def forward(self, manip_cmd, eps=5e-3):
        ee_trans, ee_ori = manip_cmd[0:2]
        gripper_fn = manip_cmd[2]
        params = manip_cmd[3]
        assert hasattr(self, gripper_fn)
        method = getattr(self, gripper_fn)
        if gripper_fn in ["in_plane_rotation", "mobile_move", "dummy_forward", "joint_ctrl"]:
            return method(**params)
        elif gripper_fn in ["update_pose_cost_metric", "update_specific"]:
            method(**params)
            return self.ee_forward(ee_trans, ee_ori, eps=eps, skip_plan=True)
        else:
            method(**params)
            return self.ee_forward(ee_trans, ee_ori, eps=eps)
