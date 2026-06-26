from copy import deepcopy

import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.box import Box, get_bbox_center_and_corners
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.iou import IoU
from core.utils.plan_utils import (
    select_index_by_priority_dual,
    select_index_by_priority_single,
)
from core.utils.transformation_utils import create_pose_matrices, poses_from_tf_matrices
from core.utils.usd_geom_utils import compute_bbox
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)
from scipy.spatial.transform import Rotation as R


# pylint: disable=consider-using-generator,too-many-public-methods,unused-argument
@register_skill
class Place(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task

        self.name = cfg["name"]
        self.pick_obj = task._task_objects[cfg["objects"][0]]
        self.place_obj = task._task_objects[cfg["objects"][1]]
        self.place_align_axis = cfg.get("place_align_axis", None)
        self.pick_align_axis = cfg.get("pick_align_axis", None)
        self.constraint_gripper_x = cfg.get("constraint_gripper_x", False)
        self.place_part_prim_path = cfg.get("place_part_prim_path", None)
        if self.place_part_prim_path:
            self.place_prim_path = f"{self.place_obj.prim_path}/{self.place_part_prim_path}"
        else:
            self.place_prim_path = self.place_obj.prim_path
        self.manip_list = []
        # if "Franka" in self.controller.robot_file or "Franka" in self.robot.cfg["name"]:
        #     self.robot_ee_path = self.robot.ee_path
        #     self.robot_base_path = self.robot.base_path
        # else:
        #     self.robot_ee_path = self.controller.robot_ee_path
        #     self.robot_base_path = self.controller.robot_base_path
        self.robot_ee_path = self.controller.robot_ee_path
        self.robot_base_path = self.controller.robot_base_path

        self.skill_cfg = cfg
        self.align_pick_obj_axis = self.skill_cfg.get("align_pick_obj_axis", None)
        self.align_place_obj_axis = self.skill_cfg.get("align_place_obj_axis", None)
        self.align_plane_x_axis = self.skill_cfg.get("align_plane_x_axis", None)
        self.align_plane_y_axis = self.skill_cfg.get("align_plane_y_axis", None)
        self.align_obj_tol = self.skill_cfg.get("align_obj_tol", None)

    def simple_generate_manip_cmds(self):
        manip_list = []

        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        cmd = (p_base_ee_cur, q_base_ee_cur, "update_pose_cost_metric", {"hold_vec_weight": None})
        manip_list.append(cmd)

        if self.skill_cfg.get("ignore_substring", []):
            ignore_substring = deepcopy(self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", []))
            cmd = (
                p_base_ee_cur,
                q_base_ee_cur,
                "update_specific",
                {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
            )
            manip_list.append(cmd)

        if self.skill_cfg.get("pre_place_hold_vec_weight", None) is not None:
            cmd = (
                p_base_ee_cur,
                q_base_ee_cur,
                "update_pose_cost_metric",
                {"hold_vec_weight": self.skill_cfg.get("pre_place_hold_vec_weight", None)},
            )
            manip_list.append(cmd)

        result = self.sample_gripper_place_traj()

        cmd = (result[0][0], result[0][1], "close_gripper", {})
        manip_list.append(cmd)

        if self.skill_cfg.get("post_place_hold_vec_weight", None) is not None:
            cmd = (
                result[0][0],
                result[0][1],
                "update_pose_cost_metric",
                {"hold_vec_weight": self.skill_cfg.get("post_place_hold_vec_weight", None)},
            )
            manip_list.append(cmd)

        p_base_ee_place, q_base_ee_place = result[1][0], result[1][1]
        cmd = (p_base_ee_place, q_base_ee_place, "close_gripper", {})
        manip_list.append(cmd)
        manip_list.extend([cmd] * self.skill_cfg.get("hesitate_steps", 0))

        cmd = (p_base_ee_place, q_base_ee_place, "open_gripper", {})
        manip_list.extend([cmd] * self.skill_cfg.get("gripper_change_steps", 10))

        cmd = (p_base_ee_place, q_base_ee_place, "detach_obj", {})
        manip_list.append(cmd)

        ignore_substring = deepcopy(self.controller.ignore_substring)
        cmd = (
            p_base_ee_place,
            q_base_ee_place,
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)

        # Postplace
        if self.skill_cfg.get("post_place_vector", None):
            ignore_substring = deepcopy(self.controller.ignore_substring)
            ignore_substring.append(self.pick_obj.name)
            cmd = (
                p_base_ee_place,
                q_base_ee_place,
                "update_specific",
                {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
            )
            manip_list.append(cmd)
            cmd = (result[2][0], result[2][1], "open_gripper", {})
            manip_list.append(cmd)

        self.manip_list = manip_list
        self.place_ee_trans = p_base_ee_place

    def sample_gripper_place_traj(self):
        self.T_world_obj = tf_matrix_from_pose(*self.pick_obj.get_local_pose())
        self.T_world_ee = get_relative_transform(
            get_prim_at_path(self.robot_ee_path), get_prim_at_path(self.task.root_prim_path)
        )
        self.T_base_world = get_relative_transform(
            get_prim_at_path(self.task.root_prim_path), get_prim_at_path(self.robot_base_path)
        )
        self.T_obj_ee = np.linalg.inv(self.T_world_obj) @ self.T_world_ee
        ee2o_distance = np.linalg.norm(self.T_obj_ee[0:3, 3])
        self.T_world_container = tf_matrix_from_pose(*self.place_obj.get_local_pose())

        bbox_place_obj = compute_bbox(get_prim_at_path(self.place_prim_path))
        b_min, b_max = bbox_place_obj.min, bbox_place_obj.max

        def get_range(key, default):
            r = self.skill_cfg.get(key, default)
            return np.random.uniform(r[0], r[1], size=CUROBO_BATCH_SIZE)

        x_ratios = get_range("x_ratio_range", (0.4, 0.6))
        y_ratios = get_range("y_ratio_range", (0.4, 0.6))
        z_ratios = get_range("z_ratio_range", (0.4, 0.6))
        direction = self.skill_cfg.get("place_direction", "vertical")
        pos_constraint = self.skill_cfg.get("position_constraint", "gripper")
        pre_z_off = self.skill_cfg.get("pre_place_z_offset", 0.2)
        plt_z_off = self.skill_cfg.get("place_z_offset", 0.1)

        if direction == "vertical":
            x = b_min[0] + x_ratios * (b_max[0] - b_min[0])
            y = b_min[1] + y_ratios * (b_max[1] - b_min[1])
            pre_place_pos_w = np.stack([x, y, np.full(CUROBO_BATCH_SIZE, b_max[2] + pre_z_off)], axis=-1)
            place_pos_w = np.stack([x, y, np.full(CUROBO_BATCH_SIZE, b_max[2] + plt_z_off)], axis=-1)

        elif direction == "horizontal":
            align_axis = self.T_world_container[:3, :3] @ np.array(self.skill_cfg["align_place_obj_axis"])
            offset_axis = self.T_world_container[:3, :3] @ np.array(self.skill_cfg["offset_place_obj_axis"])

            tmp_pos_w = b_min + np.stack([x_ratios, y_ratios, z_ratios], axis=-1) * (b_max - b_min)

            if pos_constraint == "object":
                pre_align = self.skill_cfg.get("pre_place_align", 0.2)
                pre_offset = self.skill_cfg.get("pre_place_offset", 0.2)
                plt_align = self.skill_cfg.get("place_align", 0.1)
                plt_offset = self.skill_cfg.get("place_offset", 0.1)

                pre_place_pos_w = tmp_pos_w + align_axis * pre_align + offset_axis * pre_offset
                place_pos_w = tmp_pos_w + align_axis * plt_align + offset_axis * plt_offset
            else:
                pre_place_pos_w = tmp_pos_w - align_axis * (pre_z_off + ee2o_distance)
                place_pos_w = tmp_pos_w - align_axis * (plt_z_off + ee2o_distance)
        else:
            raise NotImplementedError

        pre_place_pos_b = (self.T_base_world[:3, :3] @ pre_place_pos_w.T).T + self.T_base_world[:3, 3]
        place_pos_b = (self.T_base_world[:3, :3] @ place_pos_w.T).T + self.T_base_world[:3, 3]

        R_tgts = self.generate_constrained_rotation_batch()

        if pos_constraint == "object":
            R_ee_obj = np.linalg.inv(self.T_obj_ee)[:3, :3]
            R_base_obj_tgts = R_tgts @ R_ee_obj

            T_base_obj_pre_tgt = create_pose_matrices(pre_place_pos_b, R_base_obj_tgts)
            T_base_obj_plt_tgt = create_pose_matrices(place_pos_b, R_base_obj_tgts)

            T_base_ee_preplaces = T_base_obj_pre_tgt @ self.T_obj_ee
            T_base_ee_places = T_base_obj_plt_tgt @ self.T_obj_ee
        elif pos_constraint == "gripper":
            T_base_ee_preplaces = create_pose_matrices(pre_place_pos_b, R_tgts)
            T_base_ee_places = create_pose_matrices(place_pos_b, R_tgts)
        else:
            raise NotImplementedError

        self.controller.update_specific(
            ignore_substring=self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", []),
            reference_prim_path=self.controller.reference_prim_path,
        )

        p_base_ee_preplaces, q_base_ee_preplaces = poses_from_tf_matrices(T_base_ee_preplaces)
        p_base_ee_places, q_base_ee_places = poses_from_tf_matrices(T_base_ee_places)

        if self.controller.use_batch:
            # Check if the input arrays are exactly the same
            if np.array_equal(p_base_ee_preplaces, p_base_ee_places) and np.array_equal(
                q_base_ee_preplaces, q_base_ee_places
            ):
                # Inputs are identical, compute only once to avoid redundant computation
                result = self.controller.test_batch_forward(p_base_ee_places, q_base_ee_places)
                index = select_index_by_priority_single(result)
            else:
                # Inputs are different, compute separately
                pre_result = self.controller.test_batch_forward(p_base_ee_preplaces, q_base_ee_preplaces)
                result = self.controller.test_batch_forward(p_base_ee_places, q_base_ee_places)
                index = select_index_by_priority_dual(pre_result, result)
        else:
            for index in range(T_base_ee_places.shape[0]):
                p_base_ee_pregrasp, q_base_ee_pregrasp = p_base_ee_preplaces[index], q_base_ee_preplaces[index]
                p_base_ee_grasp, q_base_ee_grasp = p_base_ee_places[index], q_base_ee_places[index]
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
                        print("place plan success")
                        break
                else:
                    if result_pre == 1:
                        print("place plan success")
                        break

        res_pre = list(pose_from_tf_matrix(T_base_ee_preplaces[index]))
        res_plt = list(pose_from_tf_matrix(T_base_ee_places[index]))

        if self.skill_cfg.get("post_place_vector", None):
            post_vec = np.array(self.skill_cfg["post_place_vector"])
            T_base_ee_postplace = deepcopy(T_base_ee_places[index])
            T_base_ee_postplace[:3, 3] = T_base_ee_places[index][:3, :3] @ post_vec + T_base_ee_places[index][:3, 3]
            res_post = list(pose_from_tf_matrix(T_base_ee_postplace))
            return [res_pre, res_plt, res_post]

        return [res_pre, res_plt]

    def generate_constrained_rotation_batch(self, batch_size=3000):
        filter_conditions = {
            "x": {
                "forward": (0, 0, 1),  # (row, col, direction)
                "backward": (0, 0, -1),
                "leftward": (1, 0, 1),
                "rightward": (1, 0, -1),
                "upward": (2, 0, 1),
                "downward": (2, 0, -1),
            },
            "y": {
                "forward": (0, 1, 1),
                "backward": (0, 1, -1),
                "leftward": (1, 1, 1),
                "rightward": (1, 1, -1),
                "upward": (2, 1, 1),
                "downward": (2, 1, -1),
            },
            "z": {
                "forward": (0, 2, 1),
                "backward": (0, 2, -1),
                "leftward": (1, 2, 1),
                "rightward": (1, 2, -1),
                "upward": (2, 2, 1),
                "downward": (2, 2, -1),
            },
        }
        rot_mats = R.random(batch_size).as_matrix()
        valid_mask = np.ones(batch_size, dtype=bool)

        for axis in ["x", "y", "z"]:
            filter_list = self.skill_cfg.get(f"filter_{axis}_dir", None)
            if filter_list is not None:
                direction = filter_list[0]
                row, col, sign = filter_conditions[axis][direction]
                elements = rot_mats[:, row, col]
                if len(filter_list) == 2:
                    value = filter_list[1]
                    cos_val = np.cos(np.deg2rad(value))
                    if sign > 0:
                        valid_mask &= elements >= cos_val
                    else:
                        valid_mask &= elements <= cos_val
                elif len(filter_list) == 3:
                    value1, value2 = filter_list[1:]
                    cos_val1 = np.cos(np.deg2rad(value1))
                    cos_val2 = np.cos(np.deg2rad(value2))
                    if sign > 0:
                        valid_mask &= (elements >= cos_val1) & (elements <= cos_val2)
                    else:
                        valid_mask &= (elements <= cos_val1) & (elements >= cos_val2)

        if (
            self.align_pick_obj_axis is not None
            and self.align_place_obj_axis is not None
            and self.align_obj_tol is not None
        ):

            align_pick_obj_axis = np.array(self.align_pick_obj_axis)
            align_place_obj_axis = np.array(self.align_place_obj_axis)
            R_ee_obj = np.linalg.inv(self.T_obj_ee)[:3, :3]
            R_base_container_tgt = (self.T_base_world @ self.T_world_container)[:3, :3]

            R_base_obj_tgts = rot_mats @ R_ee_obj  # (N, 3, 3)
            pick_vecs_tgt = np.einsum("ijk,k->ij", R_base_obj_tgts, align_pick_obj_axis)  # (N, 3)
            place_vec_tgt = R_base_container_tgt @ align_place_obj_axis

            dot_products = np.dot(pick_vecs_tgt, place_vec_tgt)
            norms = np.linalg.norm(pick_vecs_tgt, axis=1) * np.linalg.norm(place_vec_tgt)
            radians = np.arccos(np.clip(dot_products / norms, -1.0, 1.0))

            valid_mask &= radians < np.deg2rad(self.align_obj_tol)

        valid_rot_mats = rot_mats[valid_mask]

        print("length of valid place rots :", len(valid_rot_mats))

        if len(valid_rot_mats) == 0:
            print("Warning: No matrix satisfies constraints")
            return rot_mats[:CUROBO_BATCH_SIZE]
        else:
            indices = np.random.choice(len(valid_rot_mats), CUROBO_BATCH_SIZE)
            return valid_rot_mats[indices]

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
        if self.is_subtask_done(t_eps=self.skill_cfg.get("t_eps", 1e-3), o_eps=self.skill_cfg.get("o_eps", 5e-3)):
            self.manip_list.pop(0)
        # if self.is_success():
        #     self.manip_list.clear()
        #     print("Rotate Done")
        return len(self.manip_list) == 0

    def is_success(self, th=0.0):
        if self.skill_cfg.get("success_mode", "3diou") == "3diou":
            bbox_pick_obj = compute_bbox(self.pick_obj.prim)
            bbox_place_obj = compute_bbox(get_prim_at_path(self.place_prim_path))
            iou = IoU(
                Box(get_bbox_center_and_corners(bbox_pick_obj)), Box(get_bbox_center_and_corners(bbox_place_obj))
            ).iou()
            print(iou)
            return iou > th
        elif self.skill_cfg.get("success_mode", "3diou") == "height":
            T_o2r = get_relative_transform(
                get_prim_at_path(self.pick_obj.prim_path), get_prim_at_path(self.robot_base_path)
            )
            T_o2r_trans = T_o2r[:3, 3]
            return T_o2r_trans[2] < self.place_ee_trans[2] - 0.4
        elif self.skill_cfg.get("success_mode", "3diou") == "xybbox":
            bbox_place_obj = compute_bbox(get_prim_at_path(self.place_prim_path))
            pick_x, pick_y = self.pick_obj.get_local_pose()[0][:2]
            place_xy_min = bbox_place_obj.min[:2]
            place_xy_max = bbox_place_obj.max[:2]
            return ((place_xy_min[0] + 0.015) < pick_x < (place_xy_max[0] - 0.015)) and (
                (place_xy_min[1] + 0.015) < pick_y < (place_xy_max[1] - 0.015)
            )
        elif self.skill_cfg.get("success_mode", "3diou") == "left":
            bbox_place_obj = compute_bbox(get_prim_at_path(self.place_prim_path))
            pick_x, pick_y = self.pick_obj.get_local_pose()[0][:2]
            place_xy_min = bbox_place_obj.min[:2]
            place_xy_max = bbox_place_obj.max[:2]
            return pick_x < place_xy_min[0] - self.skill_cfg.get("threshold", 0.03)
        elif self.skill_cfg.get("success_mode", "3diou") == "right":
            bbox_place_obj = compute_bbox(get_prim_at_path(self.place_prim_path))
            pick_x, pick_y = self.pick_obj.get_local_pose()[0][:2]
            place_xy_min = bbox_place_obj.min[:2]
            place_xy_max = bbox_place_obj.max[:2]
            return pick_x > place_xy_max[0] + self.skill_cfg.get("threshold", 0.03)
        elif self.skill_cfg.get("success_mode", "3diou") == "flower":
            bbox_pick_obj = compute_bbox(self.pick_obj.prim)
            bbox_place_obj = compute_bbox(get_prim_at_path(self.place_prim_path))
            iou = IoU(
                Box(get_bbox_center_and_corners(bbox_pick_obj)), Box(get_bbox_center_and_corners(bbox_place_obj))
            ).iou()
            print("iou", iou)
            th = self.skill_cfg.get("success_th", 0.0)
            middle = self.pick_obj.get_local_pose()[0]
            x_min, y_min, _ = bbox_place_obj.min
            x_max, y_max, _ = bbox_place_obj.max
            x_middle, y_middle, _ = middle[0], middle[1], middle[2]
            if x_min < x_middle < x_max and y_min < y_middle < y_max:
                return iou > th
            else:
                return False
        elif self.skill_cfg.get("success_mode", "3diou") == "cup":
            bbox_pick_obj = compute_bbox(self.pick_obj.prim)
            bbox_place_obj = compute_bbox(get_prim_at_path(self.place_prim_path))
            iou = IoU(
                Box(get_bbox_center_and_corners(bbox_pick_obj)), Box(get_bbox_center_and_corners(bbox_place_obj))
            ).iou()
            x_cup, y_cup, z_cup = bbox_pick_obj.min
            x_shelf, y_shelf, z_shelf = bbox_place_obj.min

            print("iou", iou)
            print("x_cup, y_cup, z_cup", x_cup, y_cup, z_cup)
            print("x_shelf, y_shelf, z_shelf", x_shelf, y_shelf, z_shelf)

            return (z_cup > z_shelf + 0.05) and (iou > self.skill_cfg.get("success_th", 0.0))
