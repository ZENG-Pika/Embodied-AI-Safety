"""FR3 controller – template-based."""

import numpy as np
from core.controllers.base_controller import register_controller
from core.controllers.template_controller import TemplateController


# pylint: disable=unused-argument
@register_controller
class FR3Controller(TemplateController):
    def _get_default_ignore_substring(self):
        return ["material", "Plane", "conveyor", "scene", "table"]

    def _configure_joint_indices(self, robot_file: str) -> None:
        self.raw_js_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        if "left" in robot_file:
            self.cmd_js_names = list(self.raw_js_names)
            self.arm_indices = np.array(self.robot.cfg["left_joint_indices"])
            self.gripper_indices = np.array(self.robot.cfg["left_gripper_indices"])
            self.reference_prim_path = self.task.robots[self.name].fl_base_path
            self.lr_name = "left"
            self._gripper_state = 1.0 if self.robot.left_gripper_state == 1.0 else -1.0
        else:
            raise NotImplementedError
        self._gripper_joint_position = np.array([1.0])

    def _get_motion_gen_collision_cache(self):
        """FR3 uses larger collision cache (1000) for MotionGenConfig than template default (700)."""
        return {"obb": 1000, "mesh": 1000}
