from copy import deepcopy

import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig, OmegaConf
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import get_relative_transform
from scipy.spatial.transform import Rotation as R
from solver.planner import KPAMPlanner


# pylint: disable=consider-using-generator,too-many-public-methods,unused-argument
@register_skill
class Rotate(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.stage = task.stage
        self.name = cfg["name"]
        art_obj_name = cfg["objects"][0]
        self.art_obj = task.objects[art_obj_name]
        self.cfg = cfg

        # debug start: KPAMPlanner
        self.planner_setting = OmegaConf.to_container(cfg["planner_setting"])
        self.contact_pose_index = self.planner_setting.get("contact_pose_index")
        self.success_threshold = self.planner_setting.get("success_threshold", 0.785)  # 45 degrees
        if kwargs:
            self.world = kwargs["world"]
            self.draw = kwargs["draw"]
        # debug end: KPAMPlanner
        # !!! keyposes should be generated after previous skill is done
        self.manip_list = []

        if self.cfg.get("obj_info_path", None):
            self.art_obj.update_articulated_info(self.cfg["obj_info_path"])

    # debug start: KPAMPlanner
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
        if "additional_labels" in self.planner_setting:
            new_value = self.planner_setting["additional_labels"].get(
                self.art_obj.asset_relative_path, self.planner.modify_actuation_motion
            )
            self.planner.modify_actuation_motion = new_value

    def simple_generate_manip_cmds(self):
        if self.cfg.get("obj_info_path", None):
            self.art_obj.update_articulated_info(self.cfg["obj_info_path"])

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
                self.draw.draw_points([(T_world_base @ np.append(keypose[:3, 3], 1))[:3]], [(0, 0, 0, 1)], [7])  # black
        manip_list = []

        # Update
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        ignore_substring = deepcopy(self.controller.ignore_substring)
        cmd = (
            p_base_ee_cur,
            q_base_ee_cur,
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)
        # Update

        for i in range(len(self.traj_keyframes)):
            p_base_ee_tgt = self.traj_keyframes[i][:3, 3]
            q_base_ee_tgt = R.from_matrix(self.traj_keyframes[i][:3, :3]).as_quat(scalar_first=True)
            if i <= self.contact_pose_index - 1:
                cmd = (p_base_ee_tgt, q_base_ee_tgt, "open_gripper", {})
                manip_list.append(cmd)
                if i == self.contact_pose_index - 1:
                    # Update
                    ignore_substring = deepcopy(self.controller.ignore_substring)
                    parent_name = self.art_obj.prim_path.split("/")[-2]
                    ignore_substring.append(parent_name)
                    cmd = (
                        p_base_ee_tgt,
                        q_base_ee_tgt,
                        "update_specific",
                        {
                            "ignore_substring": ignore_substring,
                            "reference_prim_path": self.controller.reference_prim_path,
                        },
                    )
                    manip_list.append(cmd)
                    # Update
            elif i == self.contact_pose_index:
                if "hearth" in self.art_obj.name:
                    cmd = (p_base_ee_tgt, q_base_ee_tgt, "open_gripper", {})
                    manip_list.append(cmd)
                else:
                    cmd = (p_base_ee_tgt, q_base_ee_tgt, "close_gripper", {})
                    for k in range(40):
                        manip_list.append(cmd)
            else:
                cmd = (p_base_ee_tgt, q_base_ee_tgt, "close_gripper", {})
                manip_list.append(cmd)

        if "hearth" in self.art_obj.name:
            cmd = (p_base_ee_tgt, q_base_ee_tgt, "close_gripper", {})
            for k in range(40):
                manip_list.append(cmd)
            # Rotate
            for k in range(100):
                cmd = (
                    p_base_ee_tgt,
                    q_base_ee_tgt,
                    "in_plane_rotation",
                    {"target_rotate": self.success_threshold * k / 50},
                )
                manip_list.append(cmd)
            # Rotate

        self.manip_list = manip_list

    def update(self):
        curr_joint_p = self.art_obj._articulation_view.get_joint_positions()[:, self.art_obj.object_joint_index]
        self.art_obj._articulation_view.set_joint_position_targets(
            positions=curr_joint_p, joint_indices=self.art_obj.object_joint_index
        )

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
            print("Rotate Done")
        return len(self.manip_list) == 0

    def is_success(self):
        curr_joint_p = self.art_obj._articulation_view.get_joint_positions()[:, self.art_obj.object_joint_index]
        distance = np.abs(curr_joint_p - self.art_obj.articulation_initial_joint_position)
        return distance >= np.abs(self.success_threshold)
