import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.plan_utils import select_index_by_priority_single
from core.utils.transformation_utils import (
    create_pose_matrices,
    get_orientation,
    perturb_orientation,
    perturb_position,
    poses_from_tf_matrices,
)
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    tf_matrix_from_pose,
)
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


# pylint: disable=line-too-long,unused-argument
@register_skill
class Goto_Pose(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg
        self.frame = self.skill_cfg.get("frame", "robot")
        gripper_state = self.skill_cfg.get("gripper_state", 1)
        self.gripper_fn = "open_gripper" if gripper_state == 1 else "close_gripper"
        position = self.skill_cfg.get("position", None)
        if position is not None:
            self.p_base_ee_tgt = perturb_position(position, self.skill_cfg.get("max_noise_m", 0.0))
        else:
            raise KeyError(
                f"Required config 'position' not found in skill_cfg. Available keys: {list(self.skill_cfg.keys())}"
            )

        if self.skill_cfg.get("quaternion", None) or self.skill_cfg.get("euler", None):
            self.q_base_ee_tgt = perturb_orientation(
                np.array(get_orientation(self.skill_cfg.get("euler"), self.skill_cfg.get("quaternion"))),
                self.skill_cfg.get("max_noise_deg", 0),
            )
        else:
            self.q_base_ee_tgt = None

        # !!! keyposes should be generated after previous skill is done
        self.manip_list = []

        # self.robot_ee_path = self.controller.robot_ee_path
        # self.robot_base_path = self.controller.robot_base_path
        # T_base_world = get_relative_transform(
        #     get_prim_at_path(self.task.root_prim_path),
        #     get_prim_at_path(self.robot_base_path)
        # )
        # self.target_trans = np.array([-0.05, -0.2, 1.1])
        # place_pos = T_base_world[:3, :3] @ self.target_trans + T_base_world[:3, 3]

        # from pdb import set_trace
        # set_trace()

    def simple_generate_manip_cmds(self):
        manip_list = []
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        cmd = (p_base_ee_cur, q_base_ee_cur, "update_pose_cost_metric", {"hold_vec_weight": None})
        manip_list.append(cmd)

        if self.q_base_ee_tgt is None:
            # Start Filter according to constraints
            obj = self.task.objects[self.skill_cfg["objects"][0]]
            T_world_obj = tf_matrix_from_pose(*obj.get_local_pose())
            T_world_ee = get_relative_transform(
                get_prim_at_path(self.controller.robot_ee_path), get_prim_at_path(self.task.root_prim_path)
            )
            self.T_obj_ee = np.linalg.inv(T_world_obj) @ T_world_ee

            self.align_obj_axis = self.skill_cfg["align_obj_axis"]
            self.align_ref_axis = self.skill_cfg["align_ref_axis"]
            self.align_obj_tol = self.skill_cfg["align_obj_tol"]
            R_tgts = self.generate_constrained_rotation_batch()
            p_tgts = perturb_position(
                self.skill_cfg.get("position", None), self.skill_cfg.get("max_noise_m", 0.0), CUROBO_BATCH_SIZE
            )

            pos_constraint = self.skill_cfg.get("position_constraint", "gripper")
            if pos_constraint == "gripper":
                T_base_ee_tgts = create_pose_matrices(p_tgts, R_tgts)
            elif pos_constraint == "object":
                R_ee_obj = np.linalg.inv(self.T_obj_ee)[:3, :3]
                R_base_obj_tgts = R_tgts @ R_ee_obj
                T_base_obj_tgts = create_pose_matrices(p_tgts, R_base_obj_tgts)
                T_base_ee_tgts = T_base_obj_tgts @ self.T_obj_ee
            else:
                raise NotImplementedError

            self.controller.update_specific(
                ignore_substring=self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", []),
                reference_prim_path=self.controller.reference_prim_path,
            )
            p_base_ee_tgts, q_base_ee_tgts = poses_from_tf_matrices(T_base_ee_tgts)
            if self.controller.use_batch:
                result = self.controller.test_batch_forward(p_base_ee_tgts, q_base_ee_tgts)
                index = select_index_by_priority_single(result)
            else:
                for index in range(T_base_ee_tgts.shape[0]):
                    p_base_ee_tgt, q_base_ee_tgt = p_base_ee_tgts[index], q_base_ee_tgts[index]
                    test_mode = self.skill_cfg.get("test_mode", "forward")
                    if test_mode == "forward":
                        result_pre = self.controller.test_single_forward(p_base_ee_tgt, q_base_ee_tgt)
                    elif test_mode == "ik":
                        result_pre = self.controller.test_single_ik(p_base_ee_tgt, q_base_ee_tgt)
                    else:
                        raise NotImplementedError
                    if result_pre == 1:
                        print("goto pose plan success")
                        break
            self.p_base_ee_tgt, self.q_base_ee_tgt = p_base_ee_tgts[index], q_base_ee_tgts[index]

        ignore_substring = self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", [])
        interp_nums = self.skill_cfg.get("interp_nums", 1)
        if interp_nums >= 2:
            interp_trans_list, interp_ori_list = self.interp(
                p_base_ee_cur, q_base_ee_cur, self.p_base_ee_tgt, self.q_base_ee_tgt, interp_nums
            )
            for interp_trans, interp_ori in zip(interp_trans_list, interp_ori_list):
                # if self.controller.test_single_forward(interp_trans, interp_ori):
                cmd = (
                    interp_trans,
                    interp_ori,
                    "update_specific",
                    {
                        "ignore_substring": ignore_substring,
                        "reference_prim_path": self.controller.reference_prim_path,
                    },
                )
                manip_list.append(cmd)
        else:
            cmd = (self.p_base_ee_tgt, self.q_base_ee_tgt, self.gripper_fn, {})
            manip_list.append(cmd)

        self.manip_list = manip_list

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

        if self.align_obj_axis is not None and self.align_ref_axis is not None and self.align_obj_tol is not None:

            align_obj_axis = np.array(self.align_obj_axis)
            align_ref_axis = np.array(self.align_ref_axis)
            R_ee_obj = np.linalg.inv(self.T_obj_ee)[:3, :3]
            Rs_base_obj_tgt = rot_mats @ R_ee_obj  # (N, 3, 3)
            align_obj_vecs = np.einsum("ijk,k->ij", Rs_base_obj_tgt, align_obj_axis)  # (N, 3)
            align_ref_vec = np.eye(3) @ align_ref_axis
            dot_products = np.dot(align_obj_vecs, align_ref_vec)
            norms = np.linalg.norm(align_obj_vecs, axis=1) * np.linalg.norm(align_ref_vec)
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

    def interp(self, curr_trans, curr_ori, target_trans, target_ori, interp_num, normalize_quaternions=True):
        # position interpolation (linear)
        interp_trans = np.linspace(curr_trans, target_trans, interp_num, axis=0)

        # Rotation interpolation (spherical linear interpolation - SLERP)
        if normalize_quaternions:
            curr_ori = curr_ori / np.linalg.norm(curr_ori)
            target_ori = target_ori / np.linalg.norm(target_ori)

        # SciPy uses [x, y, z, w] format
        rotations = R.from_quat([curr_ori, target_ori], scalar_first=True)  # Current  # Target

        slerp = Slerp([0, 1], rotations)
        times = np.linspace(0, 1, interp_num)
        interp_ori = slerp(times).as_quat(scalar_first=True)

        return interp_trans, interp_ori

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

    def is_success(self, t_eps=5e-3, o_eps=0.087):
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        diff_pos = np.linalg.norm(p_base_ee_cur - self.p_base_ee_tgt)
        diff_ori = 2 * np.arccos(min(abs(np.dot(q_base_ee_cur, self.q_base_ee_tgt)), 1.0))
        pose_flag = np.logical_or(
            diff_pos < t_eps,
            diff_ori < o_eps,
        )
        return pose_flag
