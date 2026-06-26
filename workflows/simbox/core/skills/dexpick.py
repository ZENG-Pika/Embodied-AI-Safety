import os
from copy import deepcopy

import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig, OmegaConf
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)


# pylint: disable=unused-argument
@register_skill
class Dexpick(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        if kwargs:
            self.world = kwargs["world"]
        self.skill_cfg = cfg
        object_name = self.skill_cfg["objects"][0]
        self.object = task.objects[object_name]

        # Get grasp annotation
        usd_path = [obj["path"] for obj in task.cfg["objects"] if obj["name"] == object_name][0]
        usd_path = os.path.join(self.task.asset_root, usd_path)
        dexpick_pose_path = usd_path.replace("Aligned_obj.usd", "dexpick_pose.yaml")
        self.pick_poses = []
        if os.path.exists(dexpick_pose_path):
            with open(dexpick_pose_path, "r", encoding="utf-8") as f:
                pick_data = OmegaConf.load(f)
                pick_poses = pick_data.pick_poses
                for pick_pose in pick_poses:
                    self.pick_poses.append((np.array(pick_pose[:3]), np.array(pick_pose[3:])))

            self.pick_pose_idx = cfg.get("pick_pose_idx", 0)
            self.pose_ee2o = self.pick_poses[self.pick_pose_idx]
        self.manip_list = []
        if "left" in self.controller.robot_file:
            self.robot_ee_path = self.robot.fl_ee_path
            self.robot_base_path = self.robot.fl_base_path
            lr_arm = "left"
        elif "right" in self.controller.robot_file:
            self.robot_ee_path = self.robot.fr_ee_path
            self.robot_base_path = self.robot.fr_base_path
            lr_arm = "right"
        else:
            raise NotImplementedError
        self.pickcontact_view = task.pickcontact_views[robot.name][lr_arm][object_name]
        self.process_valid = True
        self.obj_init_trans = deepcopy(self.object.get_local_pose()[0])

    def simple_generate_manip_cmds(self):
        manip_list = []

        # Update
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        ignore_substring = deepcopy(self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", []))
        ignore_substring.append(self.object.name)
        ignore_substring += self.task.ignore_objects
        cmd = (
            p_base_ee_cur,
            q_base_ee_cur,
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)

        T_world_base = get_relative_transform(
            get_prim_at_path(self.robot_base_path), get_prim_at_path(self.task.root_prim_path)
        )

        # Reach
        T_world_obj = tf_matrix_from_pose(*self.object.get_local_pose())
        T_obj_ee_grasp = tf_matrix_from_pose(*self.pose_ee2o)
        T_world_ee_grasp = T_world_obj @ T_obj_ee_grasp
        T_base_ee_grasp = np.linalg.inv(T_world_base) @ T_world_ee_grasp
        p_base_ee_grasp, q_base_ee_grasp = pose_from_tf_matrix(T_base_ee_grasp)

        # Pre grasp
        pre_grasp_offset = self.skill_cfg.get("pre_grasp_offset", 0.1)
        if pre_grasp_offset:
            T_base_ee_pregrasp = T_base_ee_grasp.copy()
            if "r5a" in self.controller.robot_file:
                T_base_ee_pregrasp[0:3, 3] -= T_base_ee_pregrasp[0:3, 0] * pre_grasp_offset
            else:
                T_base_ee_pregrasp[0:3, 3] -= T_base_ee_pregrasp[0:3, 2] * pre_grasp_offset

            cmd = (*pose_from_tf_matrix(T_base_ee_pregrasp), "open_gripper", {})
            manip_list.append(cmd)

        # Grasp
        cmd = (p_base_ee_grasp, q_base_ee_grasp, "open_gripper", {})
        manip_list.append(cmd)
        cmd = (p_base_ee_grasp, q_base_ee_grasp, "close_gripper", {})
        manip_list.extend(
            [cmd] * self.skill_cfg.get("gripper_change_steps", 40)
        )  # here we use 40 steps to make sure the gripper is fully closed
        ignore_substring = deepcopy(self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", []))
        cmd = (
            p_base_ee_grasp,
            q_base_ee_grasp,
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)

        # Post grasp
        post_grasp_offset = np.random.uniform(
            self.skill_cfg.get("post_grasp_offset_min", 0.05), self.skill_cfg.get("post_grasp_offset_max", 0.05)
        )
        if post_grasp_offset:
            p_base_ee_postgrasp = deepcopy(p_base_ee_grasp)
            p_base_ee_postgrasp[2] += post_grasp_offset
            cmd = (p_base_ee_postgrasp, q_base_ee_grasp, "close_gripper", {})
            manip_list.append(cmd)

        self.manip_list = manip_list

    def get_contact(self, contact_threshold=0.0):
        contact = np.abs(self.pickcontact_view.get_contact_force_matrix()).squeeze()
        contact = np.sum(contact, axis=-1)
        indices = np.where(contact > contact_threshold)[0]
        return contact, indices

    def is_feasible(self, th=10):
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
        if self.is_subtask_done(t_eps=self.skill_cfg.get("t_eps", 1e-3), o_eps=self.skill_cfg.get("o_eps", 5e-3)):
            self.manip_list.pop(0)
        return len(self.manip_list) == 0

    def is_success(self):
        _, indices = self.get_contact()
        flag = len(indices) >= 1

        if self.skill_cfg.get("process_valid", True):
            self.process_valid = np.max(np.abs(self.robot.get_joints_state().velocities)) < 5 and (
                np.max(np.abs(self.object.get_linear_velocity())) < 5
            )

        flag = flag and self.process_valid

        if self.skill_cfg.get("lift_th", 0.0) > 0.0:
            obj_curr_trans = deepcopy(self.object.get_local_pose()[0])
            flag = flag and ((obj_curr_trans[2] - self.obj_init_trans[2]) > self.skill_cfg.get("lift_th", 0.0))

        return flag
