import os
from copy import deepcopy

import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.usd_geom_utils import compute_bbox
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
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


# pylint: disable=unused-argument
@register_skill
class Dexplace(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg
        self.name = cfg["name"]
        self.pick_obj = task._task_objects[cfg["objects"][0]]
        self.place_obj = task._task_objects[cfg["objects"][1]]
        self.gripper_axis = cfg.get("gripper_axis", None)
        self.camera_axis_filter = cfg.get("camera_axis_filter", None)
        self.place_part_prim_path = cfg.get("place_part_prim_path", None)
        # Get place annotation
        usd_path = [obj["path"] for obj in task.cfg["objects"] if obj["name"] == self.skill_cfg["objects"][0]][0]
        usd_path = os.path.join(self.task.asset_root, usd_path)
        place_range_path = usd_path.replace("Aligned_obj.usd", "place_range.yaml")
        if os.path.exists(place_range_path):
            with open(place_range_path, "r", encoding="utf-8") as f:
                place_data = OmegaConf.load(f)
                self.x_range = place_data.x_range
                self.y_range = place_data.y_range
        else:
            self.x_range = [0.4, 0.6]
            self.y_range = [0.4, 0.6]
        # Get place_prim
        if self.place_part_prim_path:
            self.place_prim_path = f"{self.place_obj.prim_path}/{self.place_part_prim_path}"
        else:
            self.place_prim_path = self.place_obj.prim_path
        # Get left or right
        if "left" in self.controller.robot_file:
            self.robot_ee_path = self.robot.fl_ee_path
            self.robot_base_path = self.robot.fl_base_path
        elif "right" in self.controller.robot_file:
            self.robot_ee_path = self.robot.fr_ee_path
            self.robot_base_path = self.robot.fr_base_path
        if kwargs:
            self.draw = kwargs["draw"]
        self.manip_list = []

    def simple_generate_manip_cmds(self):
        manip_list = []
        place_traj, post_place_level = self.sample_gripper_place_traj()
        if len(place_traj) > 1:
            # Having waypoints
            for waypoint in place_traj[:-1]:
                p_base_ee_mid, q_base_ee_mid = waypoint[:3], waypoint[3:]
                cmd = (p_base_ee_mid, q_base_ee_mid, "close_gripper", {})
                manip_list.append(cmd)

        # The last waypoint
        p_base_ee_place, q_base_ee_place = place_traj[-1][:3], place_traj[-1][3:]
        cmd = (p_base_ee_place, q_base_ee_place, "close_gripper", {})
        manip_list.append(cmd)
        cmd = (p_base_ee_place, q_base_ee_place, "open_gripper", {})
        manip_list.extend([cmd] * self.skill_cfg.get("gripper_change_steps", 10))

        # Adding a pose place pose to avoid collision when combining place skill and close skill
        T_base_ee_place = tf_matrix_from_pose(p_base_ee_place, q_base_ee_place)
        # Post place
        T_base_ee_postplace = deepcopy(T_base_ee_place)
        # Retreat for a bit along gripper axis
        if "r5a" in self.controller.robot_file:
            T_base_ee_postplace[0:3, 3] = T_base_ee_postplace[0:3, 3] - T_base_ee_postplace[0:3, 0] * post_place_level
        else:
            T_base_ee_postplace[0:3, 3] = T_base_ee_postplace[0:3, 3] - T_base_ee_postplace[0:3, 2] * post_place_level
        cmd = (*pose_from_tf_matrix(T_base_ee_postplace), "open_gripper", {})
        manip_list.append(cmd)
        self.manip_list = manip_list

    def sample_gripper_place_traj(self):
        place_traj = []
        T_base_ee = get_relative_transform(get_prim_at_path(self.robot_ee_path), get_prim_at_path(self.robot_base_path))
        T_world_base = get_relative_transform(
            get_prim_at_path(self.robot_base_path), get_prim_at_path(self.task.root_prim_path)
        )
        T_world_ee = T_world_base @ T_base_ee
        p_world_ee_start, q_world_ee_start = pose_from_tf_matrix(T_world_ee)
        # Getting the object pose
        T_world_obj = tf_matrix_from_pose(*self.pick_obj.get_local_pose())
        # Calculate the pose of the end-effector in the object's coordinate frame
        T_obj_world = np.linalg.inv(T_world_obj)
        # Getting the relation pose and distance of ee to object (after picking, before placing)
        T_obj_ee = T_obj_world @ T_world_ee
        ee2o_distance = np.linalg.norm(T_obj_ee[0:3, 3])
        place_part_prim = get_prim_at_path(self.place_prim_path)
        bbox_place_obj = compute_bbox(place_part_prim)
        x_min, y_min, z_min = bbox_place_obj.min
        x_max, y_max, z_max = bbox_place_obj.max
        self.place_boundary = [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        # Calculate the bounding box vertices
        vertices = [
            [x_min, y_min, z_min],
            [x_min, y_max, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
        ]
        # Draw the bounding box vertices
        if self.draw:
            for vertex in vertices:
                self.draw.draw_points([vertex], [(0, 0, 0, 1)], [7])  # black

        # 1. Obtaining ee_ori
        p_world_ee_init = self.controller.T_world_ee_init[0:3, 3]  # getting initial ee position
        container_position = self.place_obj.get_local_pose()[0]  # getting container position
        container_position[1] += 0.0
        gripper_axis = container_position - p_world_ee_init  # gripper_axis is aligned with the container direction
        gripper_axis = gripper_axis / np.linalg.norm(gripper_axis)  # Normalize the target vector
        q_world_ee = self.get_ee_ori(gripper_axis, T_world_ee, self.camera_axis_filter)
        # 2. Obtaining p_world_ee
        x = x_min + np.random.uniform(self.x_range[0], self.x_range[1]) * (x_max - x_min)
        y = y_min + np.random.uniform(self.y_range[0], self.y_range[1]) * (y_max - y_min)
        z = z_min + 0.15
        obj_place_position = np.array([x, y, z])
        if self.draw:
            self.draw.draw_points([obj_place_position.tolist()], [(1, 0, 0, 1)], [7])  # red
        p_world_ee = obj_place_position - gripper_axis * ee2o_distance
        # 3. Adding Waypoint
        # Pre place
        p_world_ee_mid = (p_world_ee_start + p_world_ee) / 2.0
        p_world_ee_mid[2] += 0.05
        slerp = Slerp([0, 1], R.from_quat([q_world_ee_start, q_world_ee]))
        q_world_ee_mid = slerp([0.5]).as_quat()[0]
        if self.draw:
            self.draw.draw_points([p_world_ee_mid.tolist()], [(0, 1, 0, 1)], [7])  # green
        place_traj.append(self.adding_waypoint(p_world_ee_mid, q_world_ee_mid, T_world_base))
        # Place
        if self.draw:
            self.draw.draw_points([p_world_ee.tolist()], [(0, 1, 0, 1)], [7])  # green
        place_traj.append(self.adding_waypoint(p_world_ee, q_world_ee, T_world_base))
        post_place_level = 0.1

        return place_traj, post_place_level

    def get_ee_ori(self, gripper_axis, T_world_ee, camera_axis_filter=None):
        gripper_x = gripper_axis
        if camera_axis_filter is not None:
            direction = camera_axis_filter[0]["direction"]
            degree = camera_axis_filter[1]["degree"]
            direction = np.array(direction) / np.linalg.norm(direction)  # Normalize the direction vector
            angle = np.radians(np.random.uniform(degree[0], degree[1]))
            gripper_z = direction - np.dot(direction, gripper_x) * gripper_x
            gripper_z = gripper_z / np.linalg.norm(gripper_z)
            rotation_axis = np.cross(gripper_z, gripper_x)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            gripper_z = R.from_rotvec(angle * rotation_axis).apply(gripper_z)

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
        x, y, z = self.pick_obj.get_local_pose()[0]  # pick_obj position
        within_boundary = (
            self.place_boundary[0][0] <= x <= self.place_boundary[1][0]
            and self.place_boundary[0][1] <= y <= self.place_boundary[1][1]
            and self.place_boundary[0][2] <= z  # <= self.place_boundary[1][2]
        )

        print("pos :", self.pick_obj.get_local_pose()[0])
        print("boundary :", self.place_boundary)
        print("within_boundary :", within_boundary)

        return within_boundary
