from copy import deepcopy

import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import get_relative_transform
from solver.planner import KPAMPlanner


# pylint: disable=unused-argument
@register_skill
class Artpreplan(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.stage = task.stage
        self.name = cfg["name"]
        art_obj_name = cfg["objects"][0]
        self.skill_cfg = cfg
        self.art_obj = task.objects[art_obj_name]
        self.planner_setting = cfg["planner_setting"]
        self.contact_pose_index = self.planner_setting["contact_pose_index"]
        self.success_threshold = self.planner_setting["success_threshold"]
        self.update_art_joint = self.planner_setting.get("update_art_joint", False)
        if kwargs:
            self.world = kwargs["world"]
            self.draw = kwargs["draw"]
        self.manip_list = []

        if self.skill_cfg.get("obj_info_path", None):
            self.art_obj.update_articulated_info(self.skill_cfg["obj_info_path"])

        lr_arm = "left" if "left" in self.controller.robot_file else "right"
        self.fingers_link_contact_view = task.artcontact_views[robot.name][lr_arm][art_obj_name + "_fingers_link"]
        self.fingers_base_contact_view = task.artcontact_views[robot.name][lr_arm][art_obj_name + "_fingers_base"]
        self.forbid_collision_contact_view = task.artcontact_views[robot.name][lr_arm][
            art_obj_name + "_forbid_collision"
        ]
        self.collision_valid = True
        self.process_valid = True

    def setup_kpam(self):
        self.planner = KPAMPlanner(
            env=self.world,
            robot=self.robot,
            object=self.art_obj,
            cfg_path=self.planner_setting,
            controller=self.controller,
            draw_points=self.draw,
            stage=self.stage,
        )

    def simple_generate_manip_cmds(self):
        if self.skill_cfg.get("obj_info_path", None):
            self.art_obj.update_articulated_info(self.skill_cfg["obj_info_path"])

        self.setup_kpam()
        traj_keyframes, sample_times = self.planner.get_keypose()
        if len(traj_keyframes) == 0 and len(sample_times) == 0:
            print("No keyframes found, return empty manip_list")
            self.manip_list = []
            return

        T_world_base = get_relative_transform(
            get_prim_at_path(self.robot.base_path), get_prim_at_path(self.task.root_prim_path)
        )
        self.traj_keyframes = traj_keyframes
        self.sample_times = sample_times
        if self.draw:
            for keypose in traj_keyframes:
                self.draw.draw_points([(T_world_base @ np.append(keypose[:3, 3], 1))[:3]], [(0, 0, 0, 1)], [7])
        manip_list = []

        # Update
        p_base_ee, q_base_ee = self.controller.get_ee_pose()
        ignore_substring = deepcopy(self.controller.ignore_substring)
        cmd = (
            p_base_ee,
            q_base_ee,
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)

        self.p_base_ee_tgt = p_base_ee
        self.q_base_ee_tgt = q_base_ee
        self.manip_list = manip_list

    def update(self):
        curr_joint_p = self.art_obj._articulation_view.get_joint_positions()[:, self.art_obj.object_joint_index]
        if self.update_art_joint and self.is_success():
            self.art_obj._articulation_view.set_joint_position_targets(
                positions=curr_joint_p, joint_indices=self.art_obj.object_joint_index
            )

    def get_contact(self, contact_threshold=0.0):
        contact = {}
        fingers_link_contact = np.abs(self.fingers_link_contact_view.get_contact_force_matrix()).squeeze()
        fingers_link_contact = np.sum(fingers_link_contact, axis=-1)
        fingers_link_contact_indices = np.where(fingers_link_contact > contact_threshold)[0]
        contact["fingers_link"] = {
            "fingers_link_contact": fingers_link_contact,
            "fingers_link_contact_indices": fingers_link_contact_indices,
        }

        fingers_base_contact = np.abs(self.fingers_base_contact_view.get_contact_force_matrix()).squeeze()
        fingers_base_contact = np.sum(fingers_base_contact, axis=-1)
        fingers_base_contact_indices = np.where(fingers_base_contact > contact_threshold)[0]
        contact["fingers_base"] = {
            "fingers_base_contact": fingers_base_contact,
            "fingers_base_contact_indices": fingers_base_contact_indices,
        }

        forbid_collision_contact = np.abs(self.forbid_collision_contact_view.get_contact_force_matrix()).squeeze()
        forbid_collision_contact = np.sum(forbid_collision_contact, axis=-1)
        forbid_collision_contact_indices = np.where(forbid_collision_contact > contact_threshold)[0]
        contact["forbid_collision"] = {
            "forbid_collision_contact": forbid_collision_contact,
            "forbid_collision_contact_indices": forbid_collision_contact_indices,
        }

        return contact

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
            print("POP one manip cmd")
        if self.is_success():
            self.manip_list.clear()
            print("Close Pre Plan Done")
        return len(self.manip_list) == 0

    def is_success(self, t_eps=5e-3, o_eps=0.087):
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        diff_pos = np.linalg.norm(p_base_ee_cur - self.p_base_ee_tgt)
        diff_ori = 2 * np.arccos(min(abs(np.dot(q_base_ee_cur, self.q_base_ee_tgt)), 1.0))
        pose_flag = np.logical_or(
            diff_pos < t_eps,
            diff_ori < o_eps,
        )
        return pose_flag
