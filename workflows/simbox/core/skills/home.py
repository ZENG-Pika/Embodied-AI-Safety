import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask


# pylint: disable=unused-argument
@register_skill
class Home(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg

        self.lr_hand = "right" if "right" in self.controller.robot_file else "left"
        if self.lr_hand == "left":
            self._joint_indices = self.robot.left_joint_indices
            self._joint_home = self.robot.left_joint_home
            if self.skill_cfg.get("gripper_state", None):
                self._gripper_state = self.skill_cfg["gripper_state"]
            else:
                self._gripper_state = self.robot.left_gripper_state
        elif self.lr_hand == "right":
            self._joint_indices = self.robot.right_joint_indices
            self._joint_home = self.robot.right_joint_home
            if self.skill_cfg.get("gripper_state", None):
                self._gripper_state = self.skill_cfg["gripper_state"]
            else:
                self._gripper_state = self.robot.right_gripper_state

        # !!! keyposes should be generated after previous skill is done
        self.manip_list = []

    def simple_generate_manip_cmds(self):
        manip_list = []
        curr_ee_trans, curr_ee_ori = self.controller.get_ee_pose()
        curr_joints = self.robot.get_joint_positions()[self._joint_indices]
        home_joints = self._joint_home

        for k in range(0, 50):
            arm_action = np.array(home_joints) * ((k + 1) / 40) + np.array(curr_joints) * (1 - (k + 1) / 40)
            cmd = (
                curr_ee_trans,
                curr_ee_ori,
                "dummy_forward",
                {"arm_action": arm_action, "gripper_state": self._gripper_state},
            )
            manip_list.append(cmd)

        self.manip_list = manip_list

    def is_feasible(self, th=5):
        return self.controller.num_plan_failed <= th

    def is_subtask_done(self, t_eps=0.088):
        assert len(self.manip_list) != 0
        curr_joints = self.robot.get_joint_positions()[self._joint_indices]
        target_joints = self.manip_list[0][3]["arm_action"]
        diff_trans = np.linalg.norm(curr_joints - target_joints)
        pose_flag = (diff_trans < t_eps,)
        self.plan_flag = self.controller.num_last_cmd > 10
        return np.logical_or(pose_flag, self.plan_flag)

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done():
            self.manip_list.pop(0)
        if self.is_success():
            self.manip_list.clear()
            print("Home Done")
        return len(self.manip_list) == 0

    def is_success(self, t_eps=0.088):
        curr_joints = self.robot.get_joint_positions()[self._joint_indices]
        diff_trans = np.linalg.norm(curr_joints - self._joint_home)
        # print(diff_trans)
        return diff_trans < t_eps
