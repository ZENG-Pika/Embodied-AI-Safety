# pylint: skip-file
import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.interpolate_utils import linear_interpolation
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


# pylint: disable=unused-argument
@register_skill
class Approach_Rotate(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.name = cfg["name"]
        self.move_obj = task.objects[cfg["objects"][0]]
        self.approach_obj = task.objects[cfg["objects"][1]]
        self.skill_cfg = cfg
        self.rotate_cfg = self.skill_cfg.get("rotate", None)

        self.T_world_base = get_relative_transform(
            get_prim_at_path(self.controller.robot_base_path), get_prim_at_path(self.task.root_prim_path)
        )

        self.manip_list = []
        self.success_threshold_move = self.skill_cfg["success_threshold"]
        if self.rotate_cfg:
            self.success_threshold_rotate = self.rotate_cfg["success_threshold"]

    def simple_generate_manip_cmds(self):
        manip_list = []
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        cmd = (
            p_base_ee_cur,
            q_base_ee_cur,
            "update_pose_cost_metric",
            {"hold_vec_weight": self.skill_cfg.get("hold_vec_weight", None)},
        )
        manip_list.append(cmd)
        ignore_substring = self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", [])
        cmd = (
            p_base_ee_cur,
            q_base_ee_cur,
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)

        self.dummy_forward_cfg = self.skill_cfg.get("dummy_forward", None)
        if self.dummy_forward_cfg:
            curr_js, tgt_js = self.get_tgt_js()
            interp_js_list = linear_interpolation(curr_js, tgt_js, self.dummy_forward_cfg.get("num_steps", 10))
            for js in interp_js_list:
                p_base_ee, q_base_ee = self.controller.forward_kinematic(js)
                cmd = (
                    p_base_ee,
                    q_base_ee,
                    "dummy_forward",
                    {
                        "arm_action": js,
                        "gripper_state": self.dummy_forward_cfg.get("gripper_state", 1.0),
                    },
                )
                manip_list.append(cmd)

        self.p_base_ee_tgt, self.q_base_ee_tgt = self.getTgtPose()
        cmd = (
            self.p_base_ee_tgt,
            self.q_base_ee_tgt,
            "close_gripper",
            {},
        )
        manip_list.append(cmd)

        self.manip_list = manip_list

    def get_tgt_js(self):
        raise NotImplementedError

    def getTgtPose(self):

        T_world_ee_cur = self.T_world_base @ tf_matrix_from_pose(*self.controller.get_ee_pose())
        T_world_move_obj_cur = tf_matrix_from_pose(*self.move_obj.get_world_pose())

        if self.skill_cfg.get("obj_axis_offset", None):
            obj_axis_offset = self.skill_cfg.get("obj_axis_offset", None)
            axis_map = {"x": 0, "y": 1, "z": 2}
            dT = np.eye(4)
            for axis, offset in obj_axis_offset:
                if axis in axis_map:
                    dT[axis_map[axis], 3] = offset
            T_world_move_obj_cur = T_world_move_obj_cur @ dT

        T_world_move_obj_tgt = T_world_move_obj_cur.copy()
        if self.rotate_cfg:
            q_world_move_obj_tgt = self.getTgtMoveObjOrientation()
            T_world_move_obj_tgt[:3, :3] = R.from_quat(q_world_move_obj_tgt, scalar_first=True).as_matrix()

        p_base_ee_tgt, q_base_ee_tgt = self.calTgtEEPose(
            T_world_ee_cur,
            T_world_move_obj_cur,
            T_world_move_obj_tgt,
        )
        p_base_ee_tgt[2] += self.skill_cfg.get("z_offset", 0.0)

        return p_base_ee_tgt, q_base_ee_tgt

    def calTgtEEPose(self, T_world_ee_cur=None, T_world_move_obj_cur=None, T_world_move_obj_tgt=None):
        # Choose object local axis as approach direction (default: x-axis)
        approach_axis = self.skill_cfg.get("approach_axis", "+x")  # "x", "y", or "z"
        axis_map = {"+x": [0, 1], "+y": [1, 1], "+z": [2, 1], "-x": [0, -1], "-y": [1, -1], "-z": [2, -1]}
        local_axis = axis_map[approach_axis][0]
        local_flag = axis_map[approach_axis][1]
        approach_dir_world = T_world_move_obj_tgt[:3, local_axis] * (local_flag)  # object axis expressed in world frame

        # Apply additional yaw rotation around world Z axis (obj_yaw_offset)
        yaw_offset = self.skill_cfg.get("obj_yaw_offset", 0.0) / 180.0 * np.pi
        if abs(yaw_offset) > 1e-6:
            R_z = R.from_euler("z", yaw_offset).as_matrix()
            approach_dir_world = R_z @ approach_dir_world

        # Construct approach point in front of tgt object (world frame)
        p_world_apr_obj_cur, _ = self.approach_obj.get_world_pose()
        distance = self.skill_cfg.get("distance", 0.1)
        p_world_apr_obj_tgt = p_world_apr_obj_cur - distance * approach_dir_world
        T_world_move_obj_tgt[:2, 3] = p_world_apr_obj_tgt[:2]

        T_base_ee_tgt = (
            np.linalg.inv(self.T_world_base)
            @ T_world_move_obj_tgt
            @ np.linalg.inv(T_world_move_obj_cur)
            @ T_world_ee_cur
        )

        return pose_from_tf_matrix(T_base_ee_tgt)

    def getTgtMoveObjOrientation(self):
        rotate_type = self.rotate_cfg["type"]
        if rotate_type == "random":
            q_world_move_obj_tgt = np.random.uniform(*self.rotate_cfg.get("rotate_obj_euler", [[0, 0, 0], [0, 0, 0]]))
            q_world_move_obj_tgt = (R.from_euler("xyz", q_world_move_obj_tgt, degrees=True)).as_quat(scalar_first=True)

        elif rotate_type == "towards":
            tgt_obj_2 = self.task.objects[self.rotate_cfg["objects"][1]]
            curr_obj_2_trans, _ = tgt_obj_2.get_world_pose()
            x_tgt, y_tgt = curr_obj_2_trans[1] - self.T_world_base[1, 3], -(
                curr_obj_2_trans[0] - self.T_world_base[0, 3]
            )

            tgt_obj_yaw = np.arctan2(y_tgt, x_tgt)

            tgt_obj_yaw = np.mod(tgt_obj_yaw + np.pi, 2 * np.pi) - np.pi
            q_world_move_obj_tgt = np.array([90, 0, tgt_obj_yaw / np.pi * 180])  # z-up
            R_obj_tgt = R.from_euler("xyz", q_world_move_obj_tgt, degrees=True)
            q_world_move_obj_tgt = R_obj_tgt.as_quat(scalar_first=True)

        return q_world_move_obj_tgt

    def is_feasible(self, th=5):
        return self.controller.num_plan_failed <= th

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        p_base_ee, q_base_ee, *_ = self.manip_list[0]
        diff_pos = np.linalg.norm(p_base_ee_cur - p_base_ee)
        diff_ori = 2 * np.arccos(min(abs(np.dot(q_base_ee_cur, q_base_ee)), 1.0))
        pose_flag = np.logical_and(
            diff_pos < t_eps,
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
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        distance = np.linalg.norm(p_base_ee_cur - self.p_base_ee_tgt)
        flag = (distance < self.success_threshold_move) and (len(self.manip_list) == 0)
        if self.rotate_cfg:
            dot = np.clip(np.dot(q_base_ee_cur, self.q_base_ee_tgt), -1.0, 1.0)
            angle_diff = 2 * np.arccos(np.abs(dot))
            flag = (angle_diff < self.success_threshold_rotate) and flag
        return flag
