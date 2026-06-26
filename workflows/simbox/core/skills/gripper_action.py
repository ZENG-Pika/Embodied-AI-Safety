import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask


# pylint: disable=unused-argument
@register_skill
class Gripper_Action(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg
        self._gripper_state = self.skill_cfg["gripper_state"]

        # !!! keyposes should be generated after previous skill is done
        self.manip_list = []

    def simple_generate_manip_cmds(self):
        manip_list = []

        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        if self._gripper_state == 1:  # Open
            cmd = (p_base_ee_cur, q_base_ee_cur, "open_gripper", {}, self.skill_cfg.get("vel", None))
        elif self._gripper_state == -1:  # Close
            cmd = (p_base_ee_cur, q_base_ee_cur, "close_gripper", {}, self.skill_cfg.get("vel", None))
        else:
            raise NotImplementedError

        manip_list.extend([cmd] * self.skill_cfg.get("wait_steps", 10))
        self.manip_list = manip_list

    def is_feasible(self, th=5):
        return self.controller.num_plan_failed <= th

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        p_base_ee, q_base_ee, *_ = self.manip_list[0]
        diff_trans = np.linalg.norm(p_base_ee_cur - p_base_ee)
        diff_ori = 2 * np.arccos(min(abs(np.dot(q_base_ee_cur, q_base_ee)), 1.0))
        pose_flag = np.logical_and(
            diff_trans < t_eps,
            diff_ori < o_eps,
        )
        self.plan_flag = self.controller.num_last_cmd > 10
        return np.logical_or(pose_flag, self.plan_flag)

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done():
            self.manip_list.pop(0)
        return len(self.manip_list) == 0

    def is_success(self):
        return True
