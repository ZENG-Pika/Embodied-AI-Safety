from copy import deepcopy

import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import get_relative_transform
from scipy.spatial.transform import Rotation as R
from solver.planner import KPAMPlanner


# pylint: disable=unused-argument
@register_skill
class Open(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.stage = task.stage
        self.name = cfg["name"]
        self.skill_cfg = cfg
        art_obj_name = cfg["objects"][0]
        self.art_obj = task.objects[art_obj_name]
        self.planner_setting = cfg["planner_setting"]
        self.contact_pose_index = self.planner_setting["contact_pose_index"]
        self.success_threshold = self.planner_setting["success_threshold"]
        self.success_mode = self.planner_setting.get("success_mode", "abs")
        self.update_art_joint = self.planner_setting.get("update_art_joint", False)
        if kwargs:
            self.world = kwargs["world"]
            self.draw = kwargs["draw"]
        self.manip_list = []

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

        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        ignore_substring = deepcopy(self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", []))
        cmd = (
            p_base_ee_cur,
            q_base_ee_cur,
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)

        for i in range(len(self.traj_keyframes)):
            p_base_ee_tgt = self.traj_keyframes[i][:3, 3]
            q_base_ee_tgt = R.from_matrix(self.traj_keyframes[i][:3, :3]).as_quat(scalar_first=True)
            if i <= self.contact_pose_index:
                cmd = (p_base_ee_tgt, q_base_ee_tgt, "open_gripper", {})
            else:
                cmd = (p_base_ee_tgt, q_base_ee_tgt, "close_gripper", {})
            manip_list.append(cmd)

            if i == self.contact_pose_index:
                cmd = (p_base_ee_tgt, q_base_ee_tgt, "close_gripper", {})
                manip_list.extend([cmd] * 40)

            if i == self.contact_pose_index - 1:
                p_base_ee = self.traj_keyframes[i][:3, 3]
                q_base_ee = R.from_matrix(self.traj_keyframes[i][:3, :3]).as_quat(scalar_first=True)
                ignore_substring = deepcopy(self.controller.ignore_substring)
                parent_name = self.art_obj.prim_path.split("/")[-2]
                ignore_substring.append(parent_name)
                cmd = (
                    p_base_ee,
                    q_base_ee,
                    "update_specific",
                    {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
                )
                manip_list.append(cmd)
        self.manip_list = manip_list

    def update(self):
        curr_joint_p = self.art_obj._articulation_view.get_joint_positions()[:, self.art_obj.object_joint_index]
        if self.update_art_joint:
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
            print("Open Done")
        return len(self.manip_list) == 0

    def is_success(self):
        contact = self.get_contact()

        if self.skill_cfg.get("collision_valid", True):
            self.collision_valid = (
                self.collision_valid
                and len(contact["forbid_collision"]["forbid_collision_contact_indices"]) == 0
                and len(contact["fingers_base"]["fingers_base_contact_indices"]) == 0
            )
        if self.skill_cfg.get("process_valid", True):
            self.process_valid = np.max(np.abs(self.robot.get_joints_state().velocities)) < 5 and (
                np.max(np.abs(self.art_obj.get_joints_state().velocities)) < 5
            )

        curr_joint_p = self.art_obj._articulation_view.get_joint_positions()[:, self.art_obj.object_joint_index]
        init_joint_p = self.art_obj.articulation_initial_joint_position
        print(
            curr_joint_p - init_joint_p,
            "collision_valid :",
            self.collision_valid,
            "process_valid :",
            self.process_valid,
        )
        if self.success_mode == "normal":
            return (
                (curr_joint_p - init_joint_p) >= np.abs(self.success_threshold)
                and self.collision_valid
                and self.process_valid
            )
        elif self.success_mode == "abs":
            return (
                np.abs(curr_joint_p - init_joint_p) >= np.abs(self.success_threshold)
                and self.collision_valid
                and self.process_valid
            )
