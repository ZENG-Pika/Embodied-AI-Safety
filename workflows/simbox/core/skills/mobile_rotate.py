import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask


# pylint: disable=unused-argument
@register_skill
class Mobile_Rotate(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg
        # !!! keyposes should be generated after previous skill is done
        self.manip_list = []
        self.target = self.skill_cfg.get("target", 0.785)  # 45 degrees
        self.mobile_rotate_indices = [self.robot._articulation_view.dof_names.index("mobile_rotate")]

    def simple_generate_manip_cmds(self):
        manip_list = []
        self.mobile_rotate_initial = self.robot._articulation_view.get_joint_positions()[0, self.mobile_rotate_indices]
        curr_ee_trans, curr_ee_ori = self.controller.get_ee_pose()
        for k in range(100):
            cmd = (
                curr_ee_trans,
                curr_ee_ori,
                "mobile_move",
                {
                    "target": self.target * k / 50,
                    "joint_indices": self.mobile_rotate_indices,
                    "initial_position": self.mobile_rotate_initial,
                },
            )
            manip_list.append(cmd)
        self.manip_list = manip_list

    def is_subtask_done(self, o_eps=5e-3):
        assert len(self.manip_list) != 0
        manip_cmd = self.manip_list[0]
        target = manip_cmd[3]["target"]
        curr_joint_p = self.robot._articulation_view.get_joint_positions()[0, self.mobile_rotate_indices]
        distance = np.abs(curr_joint_p - self.mobile_rotate_initial)
        print(distance)
        return abs(distance - abs(target)) < o_eps

    def is_success(self):
        return True

    def mobile_rotate_done(self):
        curr_joint_p = self.robot._articulation_view.get_joint_positions()[0, self.mobile_rotate_indices]
        distance = np.abs(curr_joint_p - self.mobile_rotate_initial)
        return distance >= np.abs(self.target)

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done():
            self.manip_list.pop(0)
        if self.mobile_rotate_done():
            self.manip_list.clear()
            print("Mobile Rotate Done")
        return len(self.manip_list) == 0
