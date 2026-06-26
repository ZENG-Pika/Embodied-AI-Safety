# pylint: skip-file
# flake8: noqa
import json
import time

import carb
import IPython
import numpy as np
import solver.kpam.mp_terms as mp_terms
import solver.kpam.SE3_utils as SE3_utils

# The specification of optimization problem
import solver.kpam.term_spec as term_spec
import yaml
from colored import fg
from omni.isaac.core.utils.prims import (
    create_prim,
    get_prim_at_path,
    is_prim_path_valid,
)
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
    tf_matrices_from_poses,
)
from omni.isaac.core.utils.xforms import get_world_pose
from pydrake.all import *
from scipy.spatial.transform import Rotation as R
from solver.kpam.mp_builder import OptimizationBuilderkPAM
from solver.kpam.optimization_problem import OptimizationProblemkPAM, solve_kpam
from solver.kpam.optimization_spec import OptimizationProblemSpecification

# import gensim2.env.solver.planner_utils as plannerutils
from solver.planner_utils import *

MOTION_DICT = {
    "move-up": ["translate_z", 0.06],
    "move-down": ["translate_z", -0.06],
    "move-left": ["translate_y", 0.08],
    "move-right": ["translate_y", -0.08],
    "move-forward": ["translate_x", 0.1],
    "move-backward": ["translate_x", -0.1],
}


class KPAMPlanner:
    """
    A general class of keypoint-based trajectory optimization methods to solve
    robotic tasks. Includes task specific motion.
    https://github.com/liruiw/Fleet-Tools/blob/master/core/expert/base_expert.py
    Mostly works with simple and kinematic tasks.
    """

    def __init__(self, env, robot, object, cfg_path, obj_rot=0, controller=None, draw_points=False, stage=None):
        self.env = env
        self.robot = robot
        self.object = object
        self.cfg_path = cfg_path
        self.controller = controller
        self.obj_rot = obj_rot
        self.draw_points = draw_points
        self.stage = stage
        self.plan_time = 0
        self.goal_joint = None
        self.kpam_success = False
        self.joint_plan_success = False

        if "franka" in self.robot.name:
            self.ee_name = "panda_hand"
            self.robot_dof = 9
        elif "split_aloha" in self.robot.name or "lift2" in self.robot.name:
            self.ee_name = "link6"
            self.robot_dof = 8
            # self.ee_name = "panda_hand"
            # self.robot_dof = 9
        else:
            raise NotImplementedError

        # objects initial
        self.axis_dict = {
            "x": [1.0, 0.0, 0.0],
            "-x": [-1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "-y": [0.0, -1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
            "-z": [0.0, 0.0, -1.0],
        }

        if self.draw_points:
            from omni.isaac.debug_draw import _debug_draw

            self.draw = _debug_draw.acquire_debug_draw_interface()

        if isinstance(cfg_path, dict):
            self.cfg = cfg_path
        else:
            self.cfg = yaml.load(open(cfg_path, "r", encoding="utf-8"), Loader=yaml.SafeLoader)

        if "actuation_time" in self.cfg:  # time cost to reach task goal pose
            self.actuation_time = self.cfg["actuation_time"] + self.env.current_time

            self.pre_actuation_times = [t + self.env.current_time for t in self.cfg["pre_actuation_times"]]
            self.post_actuation_times = [t + self.env.current_time for t in self.cfg["post_actuation_times"]]
            # self.pre_actuation_poses_intool = self.cfg["pre_actuation_poses_intool"]
            # self.post_actuation_poses_intool = self.cfg["post_actuation_poses_intool"]

            self.pre_actuation_motions = self.cfg["pre_actuation_motions"]
            self.post_actuation_motions = self.cfg["post_actuation_motions"]

        elif "post_actuation_motions" in self.cfg:
            self.actuation_time = 24
            self.pre_actuation_motions = self.cfg["pre_actuation_motions"]
            self.post_actuation_motions = self.cfg["post_actuation_motions"]
            self.pre_actuation_times = [24 - 4 * i for i in range(len(self.pre_actuation_motions), 0, -1)]
            self.post_actuation_times = [24 + 4 * (i + 1) for i in range(len(self.post_actuation_motions))]

        elif "pre_actuation_motions" in self.cfg:
            self.actuation_time = 24
            self.pre_actuation_motions = self.cfg["pre_actuation_motions"]
            self.post_actuation_motions = []
            self.pre_actuation_times = [24 - 4 * i for i in range(len(self.pre_actuation_motions), 0, -1)]
            self.post_actuation_times = []

        # elif "post_actuation_motions" in self.cfg:
        #     set_trace()
        #     self.actuation_time = 24 + self.env.current_time
        #     self.pre_actuation_motions = self.cfg["pre_actuation_motions"]
        #     self.post_actuation_motions = self.cfg["post_actuation_motions"]
        #     self.pre_actuation_times = [
        #         24 + self.env.current_time - 4 * i
        #         for i in range(len(self.pre_actuation_motions), 0, -1)
        #     ]
        #     self.post_actuation_times = [
        #         24 + self.env.current_time + 4 * (i + 1)
        #         for i in range(len(self.post_actuation_motions))
        #     ]

        elif "pre_actuation_motions" in self.cfg:
            self.actuation_time = 24 + self.env.current_time
            self.pre_actuation_motions = self.cfg["pre_actuation_motions"]
            self.post_actuation_motions = []
            self.pre_actuation_times = [
                24 - 4 * i + self.env.current_time for i in range(len(self.pre_actuation_motions), 0, -1)
            ]
            self.post_actuation_times = []

        if "modify_actuation_motion" in self.cfg:
            self.modify_actuation_motion = self.cfg["modify_actuation_motion"]

        self.plant, self.fk_context = build_plant(
            robot_name=self.robot.name, init_qpos=self.robot.get_joint_positions()
        )

        self.reset_expert()

    def setup(self):
        # load keypoints
        self.solved_ik_times = []
        self.joint_traj_waypoints = []

    def reset_expert(self):
        """reinitialize expert state"""
        self.joint_space_traj = None
        self.task_space_traj = None
        self.plan_succeeded = False
        self.plan_time = 0.0
        self.setup()

    def check_plan_empty(self):
        """check if already have a plan"""
        return self.joint_space_traj is None

    def get_pose_from_translation(self, translation, pre_pose):
        """get the pose from translation"""
        pose = np.eye(4)
        translation = np.array(translation)
        pose[:3, 3] = translation
        # pre_pose: robot base frame to ee frame
        # pose: ee frame to target pose frame
        # actuation_pose: robot base frame to target pose frame
        actuation_pose = pre_pose @ pose
        return actuation_pose

    def get_pose_from_translation_inworld(self, translation, pre_pose):
        """get the pose from translation"""
        translation = np.array(translation)
        actuation_pose = pre_pose.copy()
        actuation_pose[:3, 3] += translation
        return actuation_pose

    def get_pose_from_rotation(self, rotation, pre_pose):
        """get the pose from rotation"""
        vec = np.array(self.get_object_link0_rot_axis())
        vec *= rotation
        Rot = np.eye(4)
        Rot[:3, :3] = R.from_euler("xyz", vec, degrees=False).as_matrix()
        # Rot = rotAxis(angle=rotation, axis=axis)
        # actuation_pose = (
        #     self.object_pose @ Rot @ se3_inverse(self.object_pose) @ pre_pose
        # )

        world2link_pose = self.get_world2link_pose()  # object to link
        inv_base_pose = se3_inverse(self.base_pose)  # self.base_pose : world to robot base
        base2link = inv_base_pose.dot(world2link_pose)
        base2link = base2link.reshape(4, 4)
        actuation_pose = base2link @ Rot @ se3_inverse(base2link) @ pre_pose
        return actuation_pose

    def generate_actuation_poses(self):
        # get key pose constraints during trajectory
        """build the post-activation trajectory specified in the config
        (1) reach above the screw
        (2) hammer it by moving downward
        """
        self.pre_actuation_poses = []
        self.post_actuation_poses = []

        curr_pose = self.task_goal_hand_pose
        for motion in self.pre_actuation_motions:
            mode = motion[0]
            value = motion[1]

            assert mode in ["translate_x", "translate_y", "translate_z", "rotate"]
            assert type(value) == float

            if mode == "rotate":
                curr_pose = self.get_pose_from_rotation(value, curr_pose)
                self.pre_actuation_poses.append(curr_pose)
            else:
                value_vec = [0, 0, 0]
                if mode == "translate_x":
                    value_vec[0] = value
                elif mode == "translate_y":
                    value_vec[1] = value
                elif mode == "translate_z":
                    value_vec[2] = value
                curr_pose = self.get_pose_from_translation(value_vec, curr_pose)
                self.pre_actuation_poses.append(curr_pose)

        self.pre_actuation_poses.reverse()

        curr_pose = self.task_goal_hand_pose  # pose in robot base frame
        for motion in self.post_actuation_motions:
            if type(motion) == list:
                mode = motion[0]
                value = motion[1]

                if mode == "rotate":
                    curr_pose = self.get_pose_from_rotation(value, curr_pose)
                    self.post_actuation_poses.append(curr_pose)
                else:
                    value_vec = [0, 0, 0]
                    if mode == "translate_x":
                        value_vec[0] = value * np.cos(self.obj_rot)
                        value_vec[1] = value * np.sin(self.obj_rot)
                    elif mode == "translate_y":
                        value_vec[0] = -value * np.sin(self.obj_rot)
                        value_vec[1] = value * np.cos(self.obj_rot)
                    elif mode == "translate_z":
                        value_vec[2] = value
                    curr_pose = self.get_pose_from_translation(value_vec, curr_pose)
                    self.post_actuation_poses.append(curr_pose)

            elif type(motion) == str:
                mode = MOTION_DICT[motion][0]
                value = MOTION_DICT[motion][1]

                value_vec = [0, 0, 0]
                if mode == "translate_x":
                    value_vec[0] = value
                elif mode == "translate_y":
                    value_vec[1] = value
                elif mode == "translate_z":
                    value_vec[2] = value
                curr_pose = self.get_pose_from_translation_inworld(value_vec, curr_pose)
                self.post_actuation_poses.append(curr_pose)

        self.sample_times = (
            [self.env.current_time] + self.pre_actuation_times + [self.actuation_time] + self.post_actuation_times
        )

        self.traj_keyframes = (
            [self.ee_pose.reshape(4, 4)]
            + self.pre_actuation_poses
            + [self.task_goal_hand_pose]
            + self.post_actuation_poses
        )

        # ========================= visualize keypoints =========================
        # base2world=get_relative_transform(get_prim_at_path(self.robot.base_path), get_prim_at_path(self.robot.root_prim_path))
        # for traj_keyframe in self.traj_keyframes:
        #     draw.draw_points(
        #         [(base2world @ np.append(traj_keyframe[:3,3],1))[:3]],
        #         [(0,0,0,1)],    # black
        #         [7]
        #     )
        # ========================= visualize keypoints =========================

    # debug start: KPAMPlanner, refactor all self.task.xxx() calls into planner methods

    def get_world2link_pose(self):
        revolute_prim = self.stage.GetPrimAtPath(self.object.object_joint_path)

        body0_path = str(revolute_prim.GetRelationship("physics:body0").GetTargets()[0])  # base
        body0_prim = self.stage.GetPrimAtPath(body0_path)

        local_pos = revolute_prim.GetAttribute("physics:localPos0").Get()
        local_rot = revolute_prim.GetAttribute("physics:localRot0").Get()  # wxyz
        local_wxyz = np.array(
            [local_rot.GetReal(), local_rot.GetImaginary()[0], local_rot.GetImaginary()[1], local_rot.GetImaginary()[2]]
        )
        base2joint_pose = np.eye(4)
        base2joint_pose[:3, 3] = np.array([local_pos[0], local_pos[1], local_pos[2]])
        base2joint_pose[:3, :3] = R.from_quat(local_wxyz, scalar_first=True).as_matrix()

        base2joint_pose[:3, 3] = base2joint_pose[:3, 3] * self.object.object_scale[:3]  # PL: we need to scale it first
        world2joint_pose = self.get_transform_matrix_by_prim_path(body0_path) @ base2joint_pose

        return world2joint_pose

    def get_tool_keypoints(self):
        if "left_arm" in self.controller.robot_file:
            self.robot.hand_path = self.robot.fl_hand_path
            self.robot.ee_path = self.robot.fl_ee_path
            self.robot.gripper_keypoints = self.robot.fl_gripper_keypoints
            self.robot.base_path = self.robot.fl_base_path
        elif "right_arm" in self.controller.robot_file:
            self.robot.hand_path = self.robot.fr_hand_path
            self.robot.ee_path = self.robot.fr_ee_path
            self.robot.gripper_keypoints = self.robot.fr_gripper_keypoints
            self.robot.base_path = self.robot.fr_base_path

        transform_matrix = self.get_transform_matrix_by_prim_path(self.robot.hand_path)

        tool_head = transform_matrix @ self.robot.gripper_keypoints["tool_head"]
        tool_tail = transform_matrix @ self.robot.gripper_keypoints["tool_tail"]
        tool_side = transform_matrix @ self.robot.gripper_keypoints["tool_side"]

        tool_keypoints = {}
        tool_keypoints["tool_head"] = tool_head[:3]
        tool_keypoints["tool_tail"] = tool_tail[:3]
        tool_keypoints["tool_side"] = tool_side[:3]

        return tool_keypoints

    def get_transform_matrix_by_prim_path(self, path):
        # return world to prim_path transform matrix
        position, orientation = get_world_pose(path)
        rotation_matrix = R.from_quat(orientation, scalar_first=True).as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = position

        return transform_matrix

    def get_object_keypoints(self):
        link2world_transform_matrix = self.get_transform_matrix_by_prim_path(self.object.object_link_path)
        base2world_transform_matrix = self.get_transform_matrix_by_prim_path(self.object.object_base_path)
        # convert rotation matrix to Euler angles
        link2world_euler_angles = R.from_matrix(link2world_transform_matrix[:3, :3]).as_euler("xyz", degrees=False)
        base2world_euler_angles = R.from_matrix(base2world_transform_matrix[:3, :3]).as_euler("xyz", degrees=False)
        print(f"link2world_euler_angles: {link2world_euler_angles}")
        print(f"base2world_euler_angles: {base2world_euler_angles}")
        self.object_keypoints = self.object.object_keypoints
        obejct_keypoints = {}
        for key, value in self.object_keypoints.items():
            if key == "base_contact_point":
                tranformed_keypoints = base2world_transform_matrix @ (value * self.object.object_scale)
                obejct_keypoints[key] = tranformed_keypoints[:3]
                print(f"base_contact_point: {obejct_keypoints[key]}")
            elif key == "base_object_contact_point":
                tranformed_keypoints = base2world_transform_matrix @ (value * self.object.object_scale)
                obejct_keypoints[key] = tranformed_keypoints[:3]
                print(f"base_object_contact_point: {obejct_keypoints[key]}")
            elif key == "link_contact_point":
                tranformed_keypoints = link2world_transform_matrix @ (value * self.object.object_scale)
                obejct_keypoints[key] = tranformed_keypoints[:3]
                print(f"link_contact_point: {obejct_keypoints[key]}")
            elif key == "articulated_object_head":
                tranformed_keypoints = link2world_transform_matrix @ (value * self.object.object_scale)
                obejct_keypoints[key] = tranformed_keypoints[:3]
                print(f"articulated_object_head: {obejct_keypoints[key]}")
            elif key == "articulated_object_tail":
                tranformed_keypoints = link2world_transform_matrix @ (value * self.object.object_scale)
                obejct_keypoints[key] = tranformed_keypoints[:3]
                print(f"articulated_object_tail: {obejct_keypoints[key]}")
        return obejct_keypoints

    def get_robot_default_state(self):
        transform_matrix = self.get_transform_matrix_by_prim_path(self.robot.base_path)
        return transform_matrix

    def get_robot_ee_pose(self):
        return self.get_transform_matrix_by_prim_path(self.robot.ee_path)

    def get_robot_tool_pose(self):
        return self.get_transform_matrix_by_prim_path(self.robot.hand_path)

    def get_object_link_pose(self):
        # return world to object link transform matrix
        transform_matrix = self.get_transform_matrix_by_prim_path(self.object.object_link_path)
        return transform_matrix

    def get_object_link0_contact_axis(self):
        return self.axis_dict[self.object.object_link0_contact_axis]

    def get_object_link0_move_axis(self):
        return self.axis_dict[self.object.object_link0_move_axis]

    def get_object_link0_vertical_axis(self):
        return np.cross(self.get_object_link0_move_axis(), self.get_object_link0_contact_axis())

    def get_object_link0_rot_axis(self):
        return self.axis_dict[self.object.object_link0_rot_axis]

    # debug end: KPAMPlanner, refactor all self.task.xxx() calls into planner methods

    def get_env_info(self):
        # get current end effector pose, joint angles, object poses, and keypoints from the environment
        self.tool_keypoint_in_world = self.get_tool_keypoints()  # robot keypoints in world frame
        # {
        #     'tool_head': np.array([-2.2506687e-01, -2.2929280e-08,  1.8791561e-01], dtype=np.float32),
        #     'tool_tail': np.array([-2.1895210e-01, -1.7026323e-08,  2.7770764e-01], dtype=np.float32),
        #     'tool_side': np.array([-0.22506687, -0.04500002,  0.18791561], dtype=np.float32)
        # }

        self.object_keypoint_in_world = self.get_object_keypoints()  # object keypoints in world frame
        # "articulated_object_head": np.array([-0.01,0.45,0.07], dtype=np.float32)

        if self.draw_points:
            # ========================= visualize keypoints =========================
            list1 = []
            list2 = []
            for key, item in self.tool_keypoint_in_world.items():
                list1.append(item)
            for key, item in self.object_keypoint_in_world.items():
                list2.append(item)

            self.draw.draw_points(
                list1,
                [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)],  # red head, green tail, blue side -> robot keypoints
                [5] * len(list1),
            )
            self.draw.draw_points(
                list2, [(1.0, 0.0, 1.0, 1)] * len(list2), [5] * len(list2)  # purple -> object keypoints
            )
            # ======================================================================

        self.dt = self.env._rendering_dt  # simulator dt
        self.time = self.env.current_time  # current time
        self.base_pose = self.get_robot_default_state()  # robot base pose in world frame
        # self.base_pose = np.array([[ 1.0000000e+00,  0.0000000e+00,  0.0000000e+00, -6.1500001e-01],
        #                         [ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00, -1.1641532e-10],
        #                         [ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00,  1.4901161e-08],
        #                         [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
        #                         dtype=np.float32)

        if "split_aloha" in self.robot.name:
            self.joint_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, -0.05])
        elif "lift2" in self.robot.name:
            self.joint_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04])
        elif "franka" in self.robot.name:
            self.joint_positions = self.robot.get_joint_positions()
        else:
            raise NotImplementedError
            # self.joint_positions = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, 0.0, 0.0, 0.0])

        self.curr_tool_keypoints = self.compute_tool_keypoints_inbase()  # robot keypoints in robot base frame
        self.curr_object_keypoints = self.compute_object_keypoints_inbase()  # object keypoints in robot base frame
        self.ee_pose = self.compute_hand_pose_inbase()  # robot base to end effector transform matrix
        self.tool_pose = self.compute_tool_pose_inbase()  # robot base to gripper transform matrix
        self.object_pose = self.compute_object_pose_inbase()  # robot base to object link_0 transform matrix
        self.tool_keypoints_in_hand = self.compute_tool_keypoints_inhand()  # robot keypoints in ee frame
        self.tool_rel_pose = self.compute_tool_inhand()  # robot gripper pose in ee frame

    def compute_hand_pose_inbase(self):
        ee_pose = self.get_robot_ee_pose()

        inv_base_pose = se3_inverse(self.base_pose)
        hand_pose_inbase = inv_base_pose.dot(ee_pose)
        hand_pose_inbase = hand_pose_inbase.reshape(4, 4)

        return hand_pose_inbase

    def compute_tool_pose_inbase(self):
        tool_pose = self.get_robot_tool_pose()

        inv_base_pose = se3_inverse(self.base_pose)
        tool_pose_inbase = inv_base_pose.dot(tool_pose)
        tool_pose_inbase = tool_pose_inbase.reshape(4, 4)

        return tool_pose_inbase

    def compute_object_pose_inbase(self):
        object_pose = self.get_object_link_pose()

        inv_base_pose = se3_inverse(self.base_pose)
        object_pose_inbase = inv_base_pose.dot(object_pose)
        object_pose_inbase = object_pose_inbase.reshape(4, 4)

        return object_pose_inbase

    def compute_tool_keypoints_inbase(self):
        inv_base_pose = se3_inverse(self.base_pose)
        tool_keypoints_inbase = {}
        for name, loc in self.tool_keypoint_in_world.items():
            tool_keypoints_inbase[name] = inv_base_pose.dot(np.array([loc[0], loc[1], loc[2], 1]))[:3]

        return tool_keypoints_inbase

    def compute_object_keypoints_inbase(self):
        inv_base_pose = se3_inverse(self.base_pose)
        object_keypoints_inbase = {}
        for name, loc in self.object_keypoint_in_world.items():
            object_keypoints_inbase[name] = inv_base_pose.dot(np.array([loc[0], loc[1], loc[2], 1]))[:3]

        return object_keypoints_inbase

    def compute_tool_keypoints_inhand(self):
        inv_ee_pose = se3_inverse(self.ee_pose)
        tool_keypoints_inhand = {}
        for name, loc in self.curr_tool_keypoints.items():
            tool_keypoints_inhand[name] = inv_ee_pose.dot(np.array([loc[0], loc[1], loc[2], 1]))[:3]

        return tool_keypoints_inhand

    def compute_tool_inhand(self):
        inv_ee_pose = se3_inverse(self.ee_pose)
        tool_rel_pose = inv_ee_pose.dot(self.tool_pose)

        return tool_rel_pose

    def add_random_to_keypose(self, transform_matrix, orientation_range, position_range):
        """
        Add random perturbations to a given transform matrix in the local coordinate frame.

        Args:
            transform_matrix (np.ndarray): 4x4 transform matrix expressed in the robot base frame.
            orientation_range (dict): Rotation noise range in degrees, e.g.
                {"x_min": min_angle, "x_max": max_angle, "y_min": ..., "y_max": ..., "z_min": ..., "z_max": ...}.
            position_range (dict): Translation noise range, e.g.
                {"x_min": min_offset, "x_max": max_offset, "y_min": ..., "y_max": ..., "z_min": ..., "z_max": ...}.

        Returns:
            np.ndarray: 4x4 transform matrix with added randomness.
        """
        # extract rotation and translation
        rotation_matrix = transform_matrix[:3, :3]
        translation_vector = transform_matrix[:3, 3]

        # add randomness to rotation (local frame)
        # sample random Euler angles (X, Y, Z axes)
        random_angles = [
            np.random.uniform(orientation_range["x_min"], orientation_range["x_max"]),
            np.random.uniform(orientation_range["y_min"], orientation_range["y_max"]),
            np.random.uniform(orientation_range["z_min"], orientation_range["z_max"]),
        ]

        # convert random Euler angles to rotation matrix (local frame)
        random_rotation_local = R.from_euler("xyz", random_angles, degrees=True).as_matrix()

        # apply random rotation to original rotation matrix (local frame; right-multiply)
        new_rotation_matrix = rotation_matrix @ random_rotation_local  # right-multiply in local frame

        # add randomness to translation (local frame)
        # sample random translation offsets (X, Y, Z axes)
        random_offset = [
            np.random.uniform(position_range["x_min"], position_range["x_max"]),
            np.random.uniform(position_range["y_min"], position_range["y_max"]),
            np.random.uniform(position_range["z_min"], position_range["z_max"]),
        ]

        # transform random offset into global frame
        random_offset_global = rotation_matrix @ np.array(random_offset)

        # add random offset to original translation
        new_translation_vector = translation_vector + random_offset_global

        # build new transform matrix
        new_transform_matrix = np.eye(4)
        new_transform_matrix[:3, :3] = new_rotation_matrix
        new_transform_matrix[:3, 3] = new_translation_vector

        return new_transform_matrix

    def parse_constraints(self):
        """
        Parse constraints from YAML config file into list of constraint dictionaries

        Returns:
            list: List of constraint dictionaries
        """
        constraint_dicts = []

        # read constraint_list from configuration
        for constraint in self.cfg["constraint_list"]:
            # directly copy all fields without predefining a schema
            constraint_dict = {k: v for k, v in constraint.items()}
            if "target_axis" in constraint_dict and constraint_dict["target_axis"] == "link0_contact_axis":
                vec = self.get_object_link0_contact_axis()
                constraint_dict["target_axis"] = vec
            elif "target_axis" in constraint_dict and constraint_dict["target_axis"] == "object_link0_move_axis":
                vec = self.get_object_link0_move_axis()
                constraint_dict["target_axis"] = vec
            elif "target_axis" in constraint_dict and constraint_dict["target_axis"] == "object_link0_contact_axis":
                vec = self.get_object_link0_contact_axis()
                constraint_dict["target_axis"] = vec
            elif "target_axis" in constraint_dict and constraint_dict["target_axis"] == "object_link0_vertical_axis":
                vec = self.get_object_link0_vertical_axis()
                constraint_dict["target_axis"] = vec
            constraint_dicts.append(constraint_dict)

        return constraint_dicts

    # tool use related
    def solve_actuation_joint(self, generate_traj=True):
        # get target pose
        """solve the formulated kpam problem and get goal joint"""

        # optimization_spec = OptimizationProblemSpecification()
        # optimization_spec = self.create_opt_problem(optimization_spec)

        # constraint_dicts = [c.to_dict() for c in optimization_spec._constraint_list]

        # constraint_dicts example:
        # [{'keypoint_name': 'tool_tail',
        # 'target_keypoint_name': 'articulated_object_head',
        # 'tolerance': 0.0001},
        # {'axis_from_keypoint_name': 'tool_head',
        # 'axis_to_keypoint_name': 'tool_side',
        # 'target_axis': [0.0, -1.0, 2.9802322387695312e-08],
        # 'target_axis_frame': 'object',
        # 'target_inner_product': 1,
        # 'tolerance': 0.01},
        # {'axis_from_keypoint_name': 'tool_head',
        # 'axis_to_keypoint_name': 'tool_tail',
        # 'target_axis': [0.0, -1.0, 2.9802322387695312e-08],
        # 'target_axis_frame': 'object',
        # 'target_inner_product': 0,
        # 'tolerance': 0.01}]

        constraint_dicts = self.parse_constraints()
        # need to parse the kpam config file and create a kpam problem
        indexes = np.random.randint(len(anchor_seeds), size=(8,))
        # array([12, 10,  9, 11,  4,  6,  6,  6])
        random_seeds = [self.joint_positions.copy()[: self.robot_dof]] + [
            anchor_seeds[idx][: self.robot_dof] for idx in indexes
        ]
        solutions = []
        for seed in random_seeds:
            res = solve_ik_kpam(
                get_relative_transform(
                    get_prim_at_path(self.object.object_link_path), get_prim_at_path(self.robot.base_path)
                ),
                constraint_dicts,
                self.plant.GetFrameByName(self.ee_name),
                self.tool_keypoints_in_hand,
                self.curr_object_keypoints,
                RigidTransform(self.ee_pose.reshape(4, 4)),
                seed.reshape(-1, 1),
                self.joint_positions.copy()[:9],
                rot_tol=0.01,
                timeout=True,
                consider_collision=False,
                contact_plane_normal=self.object.contact_plane_normal,
            )

            if res is not None:
                solutions.append(res.get_x_val()[:9])

        # solutions example:
        #     [array([ 1.14329376e-06,  3.53816124e-01,  4.80939611e-06, -2.05026707e+00,
        #             -1.92385797e-05,  2.65837018e+00,  7.85377454e-01,  4.00000000e-02,
        #             4.00000000e-02]),
        #     array([-1.774606  , -1.61710153,  1.6115427 , -2.16312007,  2.27760554,
        #             2.11851842, -0.19566194,  0.04      ,  0.04      ]),
        #     array([-8.44218317e-08,  3.53816313e-01, -1.85909422e-07, -2.05026625e+00,
        #             7.54750962e-07,  2.65836829e+00,  7.85399185e-01,  4.00000000e-02,
        #             4.00000000e-02]),
        #     array([-1.774606  , -1.61710153,  1.6115427 , -2.16312007,  2.27760554,
        #             2.11851842, -0.19566194,  0.04      ,  0.04      ]),
        #     array([-1.774606  , -1.61710153,  1.6115427 , -2.16312007,  2.27760554,
        #             2.11851842, -0.19566194,  0.04      ,  0.04      ])]

        if len(solutions) == 0:
            # raise ValueError("empty solution in kpam, target pose is unavailable")
            self.goal_joint = self.joint_positions[:9].copy()
            self.kpam_success = False
        else:
            self.kpam_success = True
            solutions = np.array(solutions)
            joint_positions = self.joint_positions[:9]
            dist_to_init_joints = np.linalg.norm(solutions - joint_positions.copy(), axis=-1)
            res = solutions[np.argmin(dist_to_init_joints)]
            self.goal_joint = res

            # select the best solution from solutions as res
            # res example:
            #     array([-8.44218317e-08,  3.53816313e-01, -1.85909422e-07, -2.05026625e+00,
            #         7.54750962e-07,  2.65836829e+00,  7.85399185e-01,  4.00000000e-02,
            #         4.00000000e-02])

            self.plant.SetPositions(self.fk_context, res)

        # self.task_goal_hand_pose = self.differential_ik.ForwardKinematics(diff_ik_context)
        self.task_goal_hand_pose = self.plant.EvalBodyPoseInWorld(
            self.fk_context, self.plant.GetBodyByName(self.ee_name)
        )

        # self.task_goal_hand_pose example:
        #     RigidTransform(
        #         R=RotationMatrix([
        #             [0.9678432216687978, -1.8063898958705824e-06, 0.25155416567113187],
        #             [-1.9244272431764666e-06, -0.9999999999981233, 2.2322813388666276e-07],
        #             [0.2515541656702566, -7.001475258030282e-07, -0.9678432216704577],
        #         ]),
        #         p=[0.6182456460224572, -1.8402926150504245e-07, 0.29068015466097613],
        #     )

        self.task_goal_hand_pose = np.array(self.task_goal_hand_pose.GetAsMatrix4())

        ### add random to keypose ###
        orientation_range = self.cfg["keypose_random_range"]["orientation"]
        position_range = self.cfg["keypose_random_range"]["position"]
        self.task_goal_hand_pose = self.add_random_to_keypose(
            self.task_goal_hand_pose, orientation_range, position_range
        )
        ### add random to keypose ###

        # self.task_goal_hand_pose example;
        #     array([[ 9.67843222e-01, -1.80638990e-06,  2.51554166e-01,
        #         6.18245646e-01],
        #         [-1.92442724e-06, -1.00000000e+00,  2.23228134e-07,
        #             -1.84029262e-07],
        #         [ 2.51554166e-01, -7.00147526e-07, -9.67843222e-01,
        #             2.90680155e-01],
        #         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #             1.00000000e+00]])

        self.task_goal_tool_pose = self.task_goal_hand_pose @ self.tool_rel_pose

        # # Transform the keypoint
        # self.curr_solution_tool_keypoint_head = SE3_utils.transform_point(
        #     self.task_goal_hand_pose, tool_keypoint_loc_inhand[0, :]
        # )
        # self.curr_solution_tool_keypoint_tail = SE3_utils.transform_point(
        #     self.task_goal_hand_pose, tool_keypoint_loc_inhand[1, :]
        # )
        # self.curr_solution_tool_keypoint_side = SE3_utils.transform_point(
        #     self.task_goal_hand_pose, tool_keypoint_loc_inhand[2, :]
        # )

        self.plan_time = self.env.current_time

        # self.goal_keypoint = np.stack(
        #     (
        #         self.curr_solution_tool_keypoint_head,
        #         self.curr_solution_tool_keypoint_tail,
        #         self.curr_solution_tool_keypoint_side,
        #     ),
        #     axis=0,
        # )

    def get_task_traj_from_joint_traj(self):
        """forward kinematics the joint trajectory to get the task trajectory"""
        self.pose_traj = []
        ik_times = dense_sample_traj_times(self.sample_times, self.actuation_time)
        # print(ik_times)
        self.dense_ik_times = ik_times
        for traj_time in ik_times:
            # diff_ik_context = self.differential_ik.GetMyMutableContextFromRoot(self.context)
            set_joints = self.joint_space_traj.value(traj_time)
            # print(set_joints)
            self.plant.SetPositions(self.fk_context, set_joints)
            pose = self.plant.EvalBodyPoseInWorld(self.fk_context, self.plant.GetBodyByName(self.ee_name))
            self.pose_traj.append(pose.GetAsMatrix4())

        self.task_space_traj = PiecewisePose.MakeLinear(ik_times, [RigidTransform(p) for p in self.pose_traj])

    def modify_actuation_joint(self):

        curr_pose = self.task_goal_hand_pose
        motion = self.modify_actuation_motion

        mode = motion[0]
        value = motion[1]

        assert mode in ["translate_x", "translate_y", "translate_z", "rotate"]
        assert type(value) == float

        if mode == "rotate":
            curr_pose = self.get_pose_from_rotation(value, curr_pose)
            self.pre_actuation_poses.append(curr_pose)
        else:
            value_vec = [0, 0, 0]
            if mode == "translate_x":
                value_vec[0] = value
            elif mode == "translate_y":
                value_vec[1] = value
            elif mode == "translate_z":
                value_vec[2] = value
            curr_pose = self.get_pose_from_translation(value_vec, curr_pose)
            self.task_goal_hand_pose = curr_pose  # update

        return

    def solve_postactuation_traj(self):
        """
        generate the full task trajectory with a FirstOrderHold
        """
        self.generate_actuation_poses()

    def save_traj(self, traj):
        """save the trajectory to a txt file"""
        traj = np.array(traj)
        np.savetxt("traj.txt", traj, delimiter=",")

    def solve_joint_traj(self, densify=True):
        # get traj from key pose constraints
        """
        solve for the IKs for each individual waypoint as an initial guess, and then
        solve for the whole trajectory with smoothness cost
        """
        keyposes = self.traj_keyframes  # 5
        keytimes = self.sample_times  # 5

        self.joint_traj_waypoints = [self.joint_positions.copy()]

        # set_trace()
        # [self.joint_positions.copy(), np.array(self.goal_joint,dtype=np.float32)] is reasonable,
        # but the solved self.joint_space_traj can be poor and self.goal_joint may collide with the object.

        # self.joint_space_traj = PiecewisePolynomial.FirstOrderHold(
        #     [self.env.current_time, self.actuation_time, self.post_actuation_times[-1]],
        #     np.array([self.joint_positions.copy(), np.array(self.goal_joint,dtype=np.float32), self.joint_positions.copy()]).T,
        # )

        # directly interpolate between current pose and target pose as an initial guess
        self.joint_space_traj = PiecewisePolynomial.FirstOrderHold(
            [self.env.current_time, self.actuation_time],
            np.array([self.joint_positions.copy(), np.array(self.goal_joint, dtype=np.float32)]).T,
        )

        print("self.env.current_time: ", self.env.current_time, "  self.actuation_time: ", self.actuation_time)
        print("self.joint_space_traj: ", self.joint_space_traj)

        if densify:
            self.dense_traj_times = dense_sample_traj_times(self.sample_times, self.actuation_time)
        else:
            self.dense_traj_times = self.sample_times

        print("solve traj endpoint")

        # interpolated joint
        res = solve_ik_traj_with_standoff(
            [self.ee_pose.reshape(4, 4), self.task_goal_hand_pose],
            np.array([self.joint_positions.copy(), self.goal_joint]).T,
            q_traj=self.joint_space_traj,
            waypoint_times=self.dense_traj_times,
            keyposes=keyposes,
            keytimes=keytimes,
        )

        # solve the standoff and the remaining pose use the goal as seed.
        # stitch the trajectory
        if res is not None:
            print("endpoint trajectory solved!")
            # use the joint trajectory to build task trajectory for panda
            self.joint_plan_success = True
            self.joint_traj_waypoints = res.get_x_val().reshape(-1, 9)
            self.joint_traj_waypoints = list(self.joint_traj_waypoints)
            self.joint_space_traj = PiecewisePolynomial.CubicShapePreserving(
                self.dense_traj_times, np.array(self.joint_traj_waypoints).T
            )
            if densify:
                self.get_task_traj_from_joint_traj()

        else:
            raise ValueError("endpoint trajectory not solved!")
            self.joint_plan_success = False
            self.env.need_termination = True
            if densify:
                self.get_task_traj_from_joint_traj()

    def get_joint_action(self):
        """get the joint space action"""
        if self.check_plan_empty():
            print("no joint trajectory")
            return self.env.reset()

        # lookahead
        return self.joint_space_traj.value(self.env.current_time + self.env.env_dt).reshape(-1)

    def get_pose_action(self, traj_eff_pose):
        # transform matrix to action(3+3+1)
        """get the task space action"""
        # traj_eff_pose_inworld = self.base_pose @ traj_eff_pose
        action = pack_pose(traj_eff_pose, rot_type="euler")
        # action = np.concatenate([action, [self.env.gripper_state]])
        action = np.concatenate([action, [1]])
        return action

    def get_actuation_joint(self):
        if self.goal_joint is not None:
            return self.goal_joint
        if self.check_plan_empty():
            self.solve_actuation_joint()
            return self.goal_joint
        raise ValueError("no actuation joint")

    def get_keypose(self):
        self.get_env_info()

        if self.check_plan_empty():
            s = time.time()
            self.solve_actuation_joint()
            if self.kpam_success:
                if "modify_actuation_motion" in self.cfg:
                    self.modify_actuation_joint()  # PL: for the rotate knob task, we add this feature to enable goal hand pose modifaction
                self.solve_postactuation_traj()
            else:
                self.traj_keyframes = []
                self.sample_times = []

        return self.traj_keyframes, self.sample_times

    def get_action(self, mode="pose"):
        self.get_env_info()

        if self.check_plan_empty():
            s = time.time()
            self.solve_actuation_joint()
            self.solve_postactuation_traj()
            self.solve_joint_traj()
            print("env time: {:.3f} plan generation time: {:.3f}".format(self.env.current_time, time.time() - s))

        """get the task-space action from the kpam expert joint trajectory"""
        traj_eff_pose = self.task_space_traj.value(self.time + self.dt)  # get transform matrix by current time
        pose_action = self.get_pose_action(traj_eff_pose)  # transform matrix to action(3+3+1)

        return pose_action

    def get_actuation_qpos(self):
        """get the actuation qpos"""
        self.get_env_info()
        self.solve_actuation_joint()

        return self.goal_joint, self.kpam_success

    def get_sparse_traj_qpos(self):
        """get the sparse trajectory qpos"""
        self.get_env_info()
        self.solve_actuation_joint()
        self.solve_postactuation_traj()
        self.solve_joint_traj(densify=False)

        return self.joint_traj_waypoints, self.joint_plan_success


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})

    import numpy as np

    # from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
    from gensim_testing_v2.tasks.close_microwave import CloseMicrowave
    from omni.isaac.core import World
    from omni.isaac.core.utils.types import ArticulationAction

    # from solver.planner import KPAMPlanner

    my_world = World(stage_units_in_meters=1.0)
    my_task = CloseMicrowave()
    my_world.add_task(my_task)
    my_world.reset()
    task_params = my_task.get_params()
    my_franka = my_world.scene.get_object(task_params["robot_name"]["value"])
    expert_planner = KPAMPlanner(
        my_world,
        cfg_path="gensim_testing_v2/solver/kpam/config/CloseMicrowave.yaml",
    )
    expert_planner.get_keypose()
    # print("action:",action)
    # 3d pose + 3d euler + 1d gipper openness
    # action: [ 0.38964379  0.00477079  0.45819783  3.09401237 -0.79489916  0.07860673 1.        ]
