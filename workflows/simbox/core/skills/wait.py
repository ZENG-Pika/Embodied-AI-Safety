import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask


# pylint: disable=consider-using-generator,too-many-public-methods,unused-argument
@register_skill
class Wait(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg
        self.success_threshold = cfg["success_threshold"]
        self.name = cfg["name"]
        self.move_obj = task.objects[cfg["objects"][0]]
        if "left" in controller.robot_file:
            self.robot_lr = "left"
        elif "right" in controller.robot_file:
            self.robot_lr = "right"

        self.manip_list = []

    def simple_generate_manip_cmds(self):
        manip_list = []
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        cmd = (p_base_ee_cur, q_base_ee_cur, "update_pose_cost_metric", {"hold_vec_weight": [0, 0, 1, 0, 0, 0]})
        manip_list.append(cmd)
        ignore_substring = self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", [])
        cmd = (
            p_base_ee_cur,
            q_base_ee_cur,
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)

        self.p_base_ee_tgt = p_base_ee_cur
        self.q_base_ee_tgt = q_base_ee_cur

        cmd = (
            self.p_base_ee_tgt,
            self.q_base_ee_tgt,
            "close_gripper" if self.skill_cfg.get("gripper_state", -1.0) == -1.0 else "open_gripper",
            {},
        )

        for _ in range(self.skill_cfg.get("wait_steps", 50)):
            manip_list.append(cmd)

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
        if self.plan_flag:
            print(f"move_only plan_flag: {self.plan_flag}, num_last_cmd: {self.controller.num_last_cmd}")

        return np.logical_or(pose_flag, self.plan_flag)

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done():
            self.manip_list.pop(0)
        return len(self.manip_list) == 0

    def is_success(self):
        p_base_ee_cur, _ = self.controller.get_ee_pose()
        distance = np.linalg.norm(p_base_ee_cur - self.p_base_ee_tgt)
        flag = (distance < self.success_threshold) and (len(self.manip_list) == 0)

        return flag
