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


# pylint: disable=consider-using-generator,too-many-public-methods,unused-argument
@register_skill
class Scan(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg
        self.name = cfg["name"]
        object_name = cfg["objects"][0]
        self.pick_obj = task.objects[cfg["objects"][0]]
        self.manip_list = []
        lr_arm = "right" if "right" in self.controller.robot_file else "left"
        self.pickcontact_view = task.pickcontact_views[robot.name][lr_arm][object_name]
        self.process_valid = True
        if "right" in self.controller.robot_file:
            self.robot_ee_path = self.robot.fr_ee_path
            self.robot_base_path = self.robot.fr_base_path
        elif "left" in self.controller.robot_file:
            self.robot_ee_path = self.robot.fl_ee_path
            self.robot_base_path = self.robot.fl_base_path

    def simple_generate_manip_cmds(self):
        manip_list = []
        place_traj = self.sample_place_traj()
        if len(place_traj) > 1:
            # Having waypoints
            for waypoint in place_traj[:-1]:
                p_base_ee, q_base_ee = waypoint[:3], waypoint[3:]
                cmd = (p_base_ee, q_base_ee, "close_gripper", {})
                manip_list.append(cmd)
        # The last waypoint
        p_base_ee, q_base_ee = place_traj[-1][:3], place_traj[-1][3:]
        cmd = (p_base_ee, q_base_ee, "close_gripper", {})
        manip_list.append(cmd)
        self.manip_list = manip_list

    def sample_place_traj(self):
        place_traj = []
        T_base_ee = get_relative_transform(get_prim_at_path(self.robot_ee_path), get_prim_at_path(self.robot_base_path))
        T_world_base = get_relative_transform(get_prim_at_path(self.robot_base_path), get_prim_at_path("/World"))
        T_world_ee = T_world_base @ T_base_ee

        # 1. Objtaining q_world_ee
        gripper_axis = np.array([1, 0, 0])
        gripper_axis = gripper_axis / np.linalg.norm(gripper_axis)  # Normalize the vector
        camera_axis = np.array([0, 1, 1])
        q_world_ee = self.get_ee_ori(gripper_axis, T_world_ee, camera_axis)
        # 2. Obtaining p_world_ee
        p_world_ee_init = self.controller.T_world_ee_init[0:3, 3]  # Getting initial ee position
        p_world_ee = p_world_ee_init.copy()
        p_world_ee[0] += np.random.uniform(-0.02, 0.02)
        p_world_ee[1] += np.random.uniform(0.15, 0.2)
        p_world_ee[2] += np.random.uniform(-0.14, -0.16)
        # Place
        place_traj.append(self.adding_waypoint(p_world_ee, q_world_ee, T_world_base))

        return place_traj

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

    def get_contact(self, contact_threshold=0.0):
        contact = np.abs(self.pickcontact_view.get_contact_force_matrix()).squeeze()
        contact = np.sum(contact, axis=-1)
        indices = np.where(contact > contact_threshold)[0]
        return contact, indices

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
        _, indices = self.get_contact()
        flag = len(indices) >= 1

        if self.skill_cfg.get("process_valid", True):
            self.process_valid = np.max(np.abs(self.robot.get_joints_state().velocities)) < 5 and (
                np.max(np.abs(self.pick_obj.get_linear_velocity())) < 5
            )

        flag = flag and self.process_valid

        return flag
