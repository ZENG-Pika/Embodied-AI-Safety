# pylint: skip-file
import numpy as np
import torch
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.interpolate_utils import linear_interpolation
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask


# pylint: disable=unused-argument
@register_skill
class Joint_Ctrl(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.name = cfg["name"]
        self.skill_cfg = cfg
        self.robot_base_path = self.controller.robot_base_path
        if "left" in controller.robot_file:
            self.robot_lr = "left"
        elif "right" in controller.robot_file:
            self.robot_lr = "right"
        self.manip_list = []
        self.success_threshold_js = self.skill_cfg.get("success_threshold_js", 5e-3)

    def simple_generate_manip_cmds(self):
        manip_list = []
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        # cmd = (p_base_ee_cur, q_base_ee_cur, 'update_pose_cost_metric', {'hold_vec_weight': [0,0,0,0,0,0]})
        cmd = (p_base_ee_cur, q_base_ee_cur, "update_pose_cost_metric", {"hold_vec_weight": None})
        manip_list.append(cmd)

        curr_js, target_js = self.get_target_js()
        interp_js_list = linear_interpolation(curr_js, target_js, self.skill_cfg.get("num_steps", 10))
        for js in interp_js_list:
            p_base_ee, q_base_ee = self.controller.forward_kinematic(js)
            cmd = (
                p_base_ee,
                q_base_ee,
                "dummy_forward",
                {
                    "arm_action": js,
                    "gripper_state": self.skill_cfg.get("gripper_state", 1.0),
                },
            )
            manip_list.append(cmd)

        self.target_js = js
        self.manip_list = manip_list

    def get_target_js(self):
        """
        Compute target joint configuration based on current joint state and
        joint control commands defined in skill configuration.

        Returns:
            curr_js (np.ndarray): Current joint positions of the controlled arm.
            target_js (np.ndarray): Target joint positions after applying commands.
        """

        # --- Get current joint positions ---
        joint_positions = self.robot.get_joints_state().positions

        if isinstance(joint_positions, torch.Tensor):
            curr_js = joint_positions.detach().cpu().numpy()[self.controller.arm_indices]
        elif isinstance(joint_positions, np.ndarray):
            curr_js = joint_positions[self.controller.arm_indices]
        else:
            raise TypeError(f"Unsupported joint state type: {type(joint_positions)}")

        target_js = curr_js.copy()

        # --- Apply joint control commands ---
        # ctrl_list: list of (joint_index, angle_in_deg, mode), mode in {"abs", "delta"}
        ctrl_list = self.skill_cfg.get("ctrl_list", [])
        for joint_idx, angle_deg, mode in ctrl_list:
            angle_rad = angle_deg * np.pi / 180.0
            if mode == "abs":
                target_js[joint_idx] = angle_rad
            elif mode == "delta":
                target_js[joint_idx] += angle_rad
            else:
                raise ValueError(f"Unknown control mode: {mode}")

        # --- Apply robot-specific joint limits / safety clamps ---
        robot_file = self.controller.robot_file.lower()

        if "piper" in robot_file:
            # Example: clamp elbow and wrist joints for Piper robot
            target_js[2] = min(target_js[2], 0.0)
            target_js[4] = np.clip(target_js[4], -1.22, 1.22)

        elif "r5a" in robot_file:
            # Reserved for R5A-specific constraints
            pass

        return curr_js, target_js

    def is_feasible(self, th=5):
        return self.controller.num_plan_failed <= th

    def is_subtask_done(self, js_eps=5e-3, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        manip_cmd = self.manip_list[0]
        if manip_cmd[2] == "joint_ctrl":
            joint_positions = self.robot.get_joints_state().positions
            if isinstance(joint_positions, torch.Tensor):
                curr_js = joint_positions.numpy()[self.controller.arm_indices]  # JointState
            elif isinstance(joint_positions, np.ndarray):
                curr_js = joint_positions[self.controller.arm_indices]  # JointState
            target_js = self.manip_list[0][3]["target"]
            diff_js = np.linalg.norm(curr_js - target_js)
            js_flag = diff_js < js_eps
            self.plan_flag = self.controller.num_last_cmd > 10
            return np.logical_or(js_flag, self.plan_flag)
        else:
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
        if self.is_subtask_done(self.success_threshold_js):
            self.manip_list.pop(0)

        return len(self.manip_list) == 0

    def is_success(self):
        joint_positions = self.robot.get_joints_state().positions
        if isinstance(joint_positions, torch.Tensor):
            curr_js = joint_positions.numpy()[self.controller.arm_indices]  # JointState
        elif isinstance(joint_positions, np.ndarray):
            curr_js = joint_positions[self.controller.arm_indices]  # JointState
        distance_js = np.linalg.norm(curr_js - self.target_js)
        flag = (distance_js < self.success_threshold_js) and (len(self.manip_list) == 0)

        return flag
