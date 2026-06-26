# pylint: skip-file
import os
import random
from copy import deepcopy

import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.plan_utils import (
    select_index_by_priority_dual,
    select_index_by_priority_single,
)
from core.utils.transformation_utils import poses_from_tf_matrices
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    tf_matrix_from_pose,
)


@register_skill
class Manualpick(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg
        object_name = self.skill_cfg["objects"][0]
        self.pick_obj = task.objects[object_name]

        # Get grasp annotation
        usd_path = [obj["path"] for obj in task.cfg["objects"] if obj["name"] == object_name][0]
        usd_path = os.path.join(self.task.asset_root, usd_path)
        grasp_pose_path = usd_path.replace(
            "Aligned_obj.usd", self.skill_cfg.get("npy_name", "Aligned_grasp_sparse.npy")
        )
        sparse_grasp_poses = np.load(grasp_pose_path)
        grasp_scale = self.skill_cfg.get("grasp_scale", 1)
        lr_arm = "right" if "right" in self.controller.robot_file else "left"
        self.T_obj_ee, self.scores = self.robot.pose_post_process_fn(
            sparse_grasp_poses, lr_arm=lr_arm, grasp_scale=grasp_scale
        )

        # !!! keyposes should be generated after previous skill is done
        self.manip_list = []
        self.pickcontact_view = task.pickcontact_views[robot.name][lr_arm][object_name]

    def simple_generate_manip_cmds(self):
        manip_list = []

        # Update
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        ignore_substring = deepcopy(self.controller.ignore_substring)
        ignore_substring.append(self.pick_obj.name)

        cmd = (
            p_base_ee_cur,
            q_base_ee_cur,
            "update_pose_cost_metric",
            {"hold_vec_weight": self.skill_cfg.get("hold_vec_weight", None)},
        )
        manip_list.append(cmd)

        cmd = (
            p_base_ee_cur,
            q_base_ee_cur,
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)
        if self.skill_cfg.get("start_lr_skill", False):
            cmd = (p_base_ee_cur, q_base_ee_cur, "update_pose_cost_metric", {"hold_vec_weight": None})
            manip_list.append(cmd)

        # Pre grasp
        T_base_ee_grasps = self.sample_ee_pose()  # (N, 4, 4)
        adjust_ori = self.skill_cfg.get("adjust_ori", None)
        if adjust_ori:
            pose_axis = adjust_ori[0]
            base_axis = adjust_ori[1]
            judge_flag = adjust_ori[2]
            axis_index = {"x": 0, "y": 1, "z": 2}
            rotate_axis = self.skill_cfg.get("adjust_rotate_axis", "x")

            if "piper" in self.controller.robot_file or "r5a" in self.controller.robot_file:
                num_poses = T_base_ee_grasps.shape[0]
                adjust_angle_list_cfg = self.skill_cfg.get("adjust_angle_list_cfg", [-15, 15, 7])
                adjust_angle_list = np.linspace(
                    adjust_angle_list_cfg[0], adjust_angle_list_cfg[1], adjust_angle_list_cfg[2]
                )  # (K,)

                # build batch rotation matrices of shape (K, 4, 4)
                thetas = np.radians(adjust_angle_list)
                rots = []
                for theta in thetas:
                    if rotate_axis == "x":
                        rot = np.array(
                            [
                                [1, 0, 0, 0],
                                [0, np.cos(theta), -np.sin(theta), 0],
                                [0, np.sin(theta), np.cos(theta), 0],
                                [0, 0, 0, 1],
                            ]
                        )
                    elif rotate_axis == "y":
                        rot = np.array(
                            [
                                [np.cos(theta), 0, np.sin(theta), 0],
                                [0, 1, 0, 0],
                                [-np.sin(theta), 0, np.cos(theta), 0],
                                [0, 0, 0, 1],
                            ]
                        )
                    elif rotate_axis == "z":
                        rot = np.array(
                            [
                                [np.cos(theta), -np.sin(theta), 0, 0],
                                [np.sin(theta), np.cos(theta), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1],
                            ]
                        )
                    else:
                        rot = np.eye(4)
                    rots.append(rot)
                rots = np.stack(rots, axis=0)  # (K, 4, 4)

                # original poses: (N, 4, 4), broadcast with rotations: (K, 4, 4)
                original_poses = T_base_ee_grasps.copy()
                rotated_poses = original_poses[:, None, :, :] @ rots[None, :, :, :]  # (N, K, 4, 4)

                # compute metric for each (pose, angle) candidate
                base_idx = axis_index[base_axis]
                pose_idx = axis_index[pose_axis]
                current_values = rotated_poses[:, :, base_idx, pose_idx]  # (N, K)

                if judge_flag == "min":
                    best_indices = np.argmin(current_values, axis=1)
                else:
                    best_indices = np.argmax(current_values, axis=1)

                # gather best poses per grasp
                idx_rows = np.arange(num_poses)
                best_poses = rotated_poses[idx_rows, best_indices]  # (N, 4, 4)
                T_base_ee_grasps = best_poses

        manual_adjust_ori = self.skill_cfg.get("manual_adjust_ori", None)
        if manual_adjust_ori:
            for adjust_ori in manual_adjust_ori:
                rotate_axis = adjust_ori[0]
                angle = adjust_ori[1]
                theta = np.radians(angle)
                if rotate_axis == "x":
                    rot = np.array(
                        [
                            [1, 0, 0, 0],
                            [0, np.cos(theta), -np.sin(theta), 0],
                            [0, np.sin(theta), np.cos(theta), 0],
                            [0, 0, 0, 1],
                        ]
                    )
                elif rotate_axis == "y":
                    rot = np.array(
                        [
                            [np.cos(theta), 0, np.sin(theta), 0],
                            [0, 1, 0, 0],
                            [-np.sin(theta), 0, np.cos(theta), 0],
                            [0, 0, 0, 1],
                        ]
                    )
                elif rotate_axis == "z":
                    rot = np.array(
                        [
                            [np.cos(theta), -np.sin(theta), 0, 0],
                            [np.sin(theta), np.cos(theta), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ]
                    )
                else:
                    rot = np.eye(4)
                # apply the same rotation to all grasps in batch
                T_base_ee_grasps = T_base_ee_grasps @ rot

        adjust_trans_offset = self.skill_cfg.get("adjust_trans_offset", [0, 0, 0])
        T_base_ee_grasps[:, :3, 3] += adjust_trans_offset
        T_base_ee_pregrasps = deepcopy(T_base_ee_grasps)
        self.controller.update_specific(
            ignore_substring=ignore_substring, reference_prim_path=self.controller.reference_prim_path
        )
        if "r5a" in self.controller.robot_file:
            T_base_ee_pregrasps[:, :3, 3] -= T_base_ee_pregrasps[:, :3, 0] * self.skill_cfg.get("pre_grasp_offset", 0.1)
        else:
            T_base_ee_pregrasps[:, :3, 3] -= T_base_ee_pregrasps[:, :3, 2] * self.skill_cfg.get("pre_grasp_offset", 0.1)
        pre_grasp_offset_manual = self.skill_cfg.get("pre_grasp_offset_manual", None)
        if pre_grasp_offset_manual:
            T_base_ee_pregrasps[:, :3, 3] += np.array(pre_grasp_offset_manual)

        p_base_ee_pregrasps, q_base_ee_pregrasps = poses_from_tf_matrices(T_base_ee_pregrasps)
        p_base_ee_grasps, q_base_ee_grasps = poses_from_tf_matrices(T_base_ee_grasps)

        if self.controller.use_batch:
            # Check if the input arrays are exactly the same
            if np.array_equal(p_base_ee_pregrasps, p_base_ee_grasps) and np.array_equal(
                q_base_ee_pregrasps, q_base_ee_grasps
            ):
                # Inputs are identical, compute only once to avoid redundant computation
                result = self.controller.test_batch_forward(p_base_ee_grasps, q_base_ee_grasps)
                index = select_index_by_priority_single(result)
            else:
                # Inputs are different, compute separately
                pre_result = self.controller.test_batch_forward(p_base_ee_pregrasps, q_base_ee_pregrasps)
                result = self.controller.test_batch_forward(p_base_ee_grasps, q_base_ee_grasps)
                index = select_index_by_priority_dual(pre_result, result)
        else:
            for index in range(T_base_ee_grasps.shape[0]):
                p_base_ee_pregrasp, q_base_ee_pregrasp = p_base_ee_pregrasps[index], q_base_ee_pregrasps[index]
                p_base_ee_grasp, q_base_ee_grasp = p_base_ee_grasps[index], q_base_ee_grasps[index]
                test_mode = self.skill_cfg.get("test_mode", "forward")
                if test_mode == "forward":
                    result_pre = self.controller.test_single_forward(p_base_ee_pregrasp, q_base_ee_pregrasp)
                elif test_mode == "ik":
                    result_pre = self.controller.test_single_ik(p_base_ee_pregrasp, q_base_ee_pregrasp)
                else:
                    raise NotImplementedError
                if self.skill_cfg.get("pre_grasp_offset", 0.1) > 0:
                    if test_mode == "forward":
                        result = self.controller.test_single_forward(p_base_ee_grasp, q_base_ee_grasp)
                    elif test_mode == "ik":
                        result = self.controller.test_single_ik(p_base_ee_grasp, q_base_ee_grasp)
                    else:
                        raise NotImplementedError
                    if result == 1 and result_pre == 1:
                        print("pick plan success")
                        break
                else:
                    if result_pre == 1:
                        print("pick plan success")
                        break

        cmd = (p_base_ee_pregrasps[index], q_base_ee_pregrasps[index], "open_gripper", {})
        manip_list.append(cmd)

        # Grasp
        cmd = (p_base_ee_grasps[index], q_base_ee_grasps[index], "open_gripper", {})
        manip_list.append(cmd)

        cmd = (p_base_ee_grasps[index], q_base_ee_grasps[index], "close_gripper", {})
        manip_list.extend(
            [cmd] * self.skill_cfg.get("gripper_change_steps", 40)
        )  # here we use 40 steps to make sure the gripper is fully closed

        ignore_substring = self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", [])
        cmd = (
            p_base_ee_grasps[index],
            q_base_ee_grasps[index],
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)
        cmd = (
            p_base_ee_grasps[index],
            q_base_ee_grasps[index],
            "attach_obj",
            {"obj_prim_path": self.pick_obj.mesh_prim_path},
        )
        manip_list.append(cmd)

        # Post-grasp
        post_grasp_offset = np.random.uniform(
            self.skill_cfg.get("post_grasp_offset_min", 0.05), self.skill_cfg.get("post_grasp_offset_max", 0.05)
        )
        if post_grasp_offset:
            p_base_ee_postgrasps = deepcopy(p_base_ee_grasps)
            p_base_ee_postgrasps[index][2] += post_grasp_offset
            cmd = (p_base_ee_postgrasps[index], q_base_ee_grasps[index], self.gripper_cmd, {})
            manip_list.append(cmd)

        self.manip_list = manip_list

    def sample_ee_pose(self, max_length=CUROBO_BATCH_SIZE):
        T_base_ee = self.get_ee_poses("armbase")

        num_pose = T_base_ee.shape[0]
        flags = {
            "x": np.ones(num_pose, dtype=bool),
            "y": np.ones(num_pose, dtype=bool),
            "z": np.ones(num_pose, dtype=bool),
            "direction_to_obj": np.ones(num_pose, dtype=bool),
        }
        filter_conditions = {
            "x": {
                "forward": (0, 0, 1),  # (row, col, direction)
                "backward": (0, 0, -1),
                "upward": (2, 0, 1),
                "downward": (2, 0, -1),
            },
            "y": {"forward": (0, 1, 1), "backward": (0, 1, -1), "downward": (2, 1, -1), "upward": (2, 1, 1)},
            "z": {"forward": (0, 2, 1), "backward": (0, 2, -1), "downward": (2, 2, -1), "upward": (2, 2, 1)},
        }
        for axis in ["x", "y", "z"]:
            filter_list = self.skill_cfg.get(f"filter_{axis}_dir", None)
            if filter_list is not None:
                # direction, value = filter_list
                direction = filter_list[0]
                row, col, sign = filter_conditions[axis][direction]
                if len(filter_list) == 2:
                    value = filter_list[1]
                    cos_val = np.cos(np.deg2rad(value))
                    flags[axis] = T_base_ee[:, row, col] >= cos_val if sign > 0 else T_base_ee[:, row, col] <= cos_val
                elif len(filter_list) == 3:
                    value1, value2 = filter_list[1:]
                    cos_val1 = np.cos(np.deg2rad(value1))
                    cos_val2 = np.cos(np.deg2rad(value2))
                    if sign > 0:
                        flags[axis] = np.logical_and(
                            T_base_ee[:, row, col] >= cos_val1, T_base_ee[:, row, col] <= cos_val2
                        )
                    else:
                        flags[axis] = np.logical_and(
                            T_base_ee[:, row, col] <= cos_val1, T_base_ee[:, row, col] >= cos_val2
                        )
        if self.skill_cfg.get("direction_to_obj", None) is not None:
            direction_to_obj = self.skill_cfg.direction_to_obj
            T_world_obj = tf_matrix_from_pose(*self.pick_obj.get_local_pose())
            T_base_world = get_relative_transform(
                get_prim_at_path(self.task.root_prim_path), get_prim_at_path(self.controller.reference_prim_path)
            )
            T_base_obj = T_base_world @ T_world_obj
            if direction_to_obj == "right":
                flags["direction_to_obj"] = T_base_ee[:, 1, 3] <= T_base_obj[1, 3]
            elif direction_to_obj == "left":
                flags["direction_to_obj"] = T_base_ee[:, 1, 3] > T_base_obj[1, 3]
            else:
                raise NotImplementedError

        combined_flag = np.logical_and.reduce(list(flags.values()))
        if sum(combined_flag) == 0:
            # idx_list = [i for i in range(max_length)]
            idx_list = list(range(max_length))
        else:
            tmp_scores = self.scores[combined_flag]
            tmp_idxs = np.arange(num_pose)[combined_flag]
            combined = list(zip(tmp_scores, tmp_idxs))
            combined.sort()
            idx_list = [idx for (score, idx) in combined[:max_length]]
            score_list = self.scores[idx_list]
            weights = 1.0 / (score_list + 1e-8)
            weights = weights / weights.sum()

            sampled_idx = random.choices(idx_list, weights=weights, k=max_length)
            sampled_scores = self.scores[sampled_idx]

            # Sort indices by their scores (ascending)
            sorted_pairs = sorted(zip(sampled_scores, sampled_idx))
            idx_list = [idx for _, idx in sorted_pairs]

        print(self.scores[idx_list])
        # print((T_base_ee[idx_list])[:, 0, 1])
        return T_base_ee[idx_list]

    def get_ee_poses(self, frame: str = "world"):
        # get grasp poses at specific frame
        if frame not in ["world", "body", "armbase"]:
            raise Exception(
                "poses in {} frame is not supported: accepted values are [world, body, armbase] only".format(frame)
            )

        if frame == "body":
            return self.T_obj_ee

        T_world_obj = tf_matrix_from_pose(*self.pick_obj.get_local_pose())
        T_world_ee = T_world_obj[None] @ self.T_obj_ee

        if frame == "world":
            return T_world_ee

        if frame == "armbase":  # robot base frame
            T_r2w = get_relative_transform(
                get_prim_at_path(self.controller.reference_prim_path), get_prim_at_path(self.task.root_prim_path)
            )
            T_base_world = np.linalg.inv(T_r2w)
            T_base_ee = T_base_world[None] @ T_world_ee
            return T_base_ee

    def get_contact(self, contact_threshold=0.0):
        contact = np.abs(self.pickcontact_view.get_contact_force_matrix()).squeeze()
        contact = np.sum(contact, axis=-1)
        indices = np.where(contact > contact_threshold)[0]
        return contact, indices

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
        contact, indices = self.get_contact()
        return len(indices) >= 1
