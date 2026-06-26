from copy import deepcopy

import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
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
class Flip(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.name = cfg["name"]
        self.pick_obj = task.objects[cfg["objects"][0]]
        self.skill_cfg = cfg
        self.gripper_axis = cfg.get("gripper_axis", False)
        self.manip_list = []
        if kwargs:
            self.draw = kwargs["draw"]
        if "left" in self.controller.robot_file:
            self.robot_ee_path = self.robot.fl_ee_path
            self.robot_base_path = self.robot.fl_base_path
        elif "right" in self.controller.robot_file:
            self.robot_ee_path = self.robot.fr_ee_path
            self.robot_base_path = self.robot.fr_base_path

    def simple_generate_manip_cmds(self):
        manip_list = []
        place_traj, post_place_level = self.sample_place_traj()
        if len(place_traj) > 1:
            # Having waypoints
            for waypoint in place_traj[:-1]:
                p_base_ee, q_base_ee = waypoint[:3], waypoint[3:]
                cmd = (p_base_ee, q_base_ee, "close_gripper", {})
                manip_list.append(cmd)
        # The last waypoint
        p_base_ee_place, q_base_ee_place = place_traj[-1][:3], place_traj[-1][3:]
        cmd = (p_base_ee_place, q_base_ee_place, "close_gripper", {})
        manip_list.append(cmd)
        cmd = (p_base_ee_place, q_base_ee_place, "open_gripper", {})
        manip_list.extend(
            [cmd] * self.skill_cfg.get("open_wait_steps", 20)
        )  # here we use 20 steps to make sure the gripper is fully open

        # Adding a pose place pose to avoid collision when combining place skill and close skill
        T_base_ee_place = tf_matrix_from_pose(p_base_ee_place, q_base_ee_place)
        # Post place
        T_base_ee_postplace = deepcopy(T_base_ee_place)
        # Retreat for a bit along gripper axis
        T_base_ee_postplace[0:3, 3] = T_base_ee_postplace[0:3, 3] - T_base_ee_postplace[0:3, 0] * post_place_level
        cmd = (*pose_from_tf_matrix(T_base_ee_postplace), "open_gripper", {})
        manip_list.append(cmd)
        self.manip_list = manip_list

    def sample_place_traj(self):
        place_traj = []
        T_base_ee = get_relative_transform(get_prim_at_path(self.robot_ee_path), get_prim_at_path(self.robot_base_path))
        T_world_base = get_relative_transform(get_prim_at_path(self.robot_base_path), get_prim_at_path("/World"))
        T_world_ee = T_world_base @ T_base_ee

        # 1. Obtaining ee_ori
        gripper_axis = np.array(self.gripper_axis)
        gripper_axis = gripper_axis / np.linalg.norm(gripper_axis)  # Normalize the vector
        camera_axis = np.array([0, 1, 0])
        q_world_ee = self.get_ee_ori(gripper_axis, T_world_ee, camera_axis)
        # 2. Obtaining ee_trans
        p_world_ee_init = self.controller.T_world_ee_init[0:3, 3]  # getting initial ee position
        p_world_ee = p_world_ee_init.copy()
        p_world_ee[0] += np.random.uniform(0.19, 0.21)  # 0.2
        p_world_ee[1] += np.random.uniform(0.23, 0.27)  # 0.25
        p_world_ee[2] += 0
        if self.draw:
            self.draw.draw_points([p_world_ee.tolist()], [(1, 0, 0, 1)], [7])  # red
        # 3. Adding waypoint
        # Pre place
        p_world_ee_mid = p_world_ee_init.copy()
        p_world_ee_mid[1] += np.random.uniform(0.23, 0.27)  # 0.25
        p_world_ee_mid[2] += np.random.uniform(0.14, 0.16)  # 0.15
        gripper_axis_mid = np.array([0, 1, -1])
        camera_axis_mid = np.array([0, 1, 1])
        gripper_axis_mid = gripper_axis_mid / np.linalg.norm(gripper_axis_mid)
        camera_axis_mid = camera_axis_mid / np.linalg.norm(camera_axis_mid)
        q_world_ee_mid = self.get_ee_ori(gripper_axis_mid, T_world_ee, camera_axis_mid)

        if self.draw:
            self.draw.draw_points([p_world_ee_mid.tolist()], [(1, 0, 0, 1)], [7])  # red
        place_traj.append(self.adding_waypoint(p_world_ee_mid, q_world_ee_mid, T_world_base))
        # Place
        place_traj.append(self.adding_waypoint(p_world_ee, q_world_ee, T_world_base))
        post_place_level = 0.1

        return place_traj, post_place_level

    def get_ee_ori(self, gripper_axis, T_world_ee, camera_axis=None):
        gripper_x = gripper_axis
        if camera_axis is not None:
            gripper_z = camera_axis
        else:
            current_z = T_world_ee[0:3, 1]
            gripper_z = current_z - np.dot(current_z, gripper_x) * gripper_x

        gripper_z = gripper_z / np.linalg.norm(gripper_z)
        gripper_y = np.cross(gripper_z, gripper_x)
        gripper_y = gripper_y / np.linalg.norm(gripper_y)
        gripper_z = np.cross(gripper_x, gripper_y)
        R_world_ee = np.column_stack((gripper_x, gripper_y, gripper_z))
        q_world_ee = R.from_matrix(R_world_ee).as_quat(scalar_first=True)
        return q_world_ee

    def adding_waypoint(self, p_world_ee, q_world_ee, T_world_base):
        """
        Adding a waypoint, also transform from wolrd frame to robot frame
        """
        T_world_ee = tf_matrix_from_pose(p_world_ee, q_world_ee)
        T_base_ee = np.linalg.inv(T_world_base) @ T_world_ee
        p_base_ee, q_base_ee = pose_from_tf_matrix(T_base_ee)
        waypoint = np.concatenate((p_base_ee, q_base_ee))
        return waypoint

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
        # Calculate the angle between the object's local y-axis and the world's z-axis
        T_world_obj = tf_matrix_from_pose(*self.pick_obj.get_local_pose())
        obj_y_axis = T_world_obj[0:3, 1]  # Extract the object's y-axis in world coordinates
        world_z_axis = np.array([0, 0, 1])  # World z-axis
        # Compute the angle between the two vectors
        dot_product = np.dot(obj_y_axis, world_z_axis)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip to handle numerical errors
        angle_degrees = np.degrees(angle)
        # Position
        ee_init_position = self.controller.T_world_ee_init[0:3, 3]  # getting initial ee position
        obj_position = T_world_obj[0:3, 3]
        delta_y = obj_position[1] - ee_init_position[1]
        return angle_degrees < 90 and 0 < delta_y < 0.7
