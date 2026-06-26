import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.transformation_utils import get_orientation, perturb_orientation
from core.utils.usd_geom_utils import compute_bbox
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.objects.cylinder import VisualCylinder
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    tf_matrix_from_pose,
)
from scipy.spatial.transform import Rotation as R


# pylint: disable=consider-using-generator,too-many-public-methods,unused-argument
@register_skill
class Track(BaseSkill):
    def __init__(
        self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, world, *args, **kwargs
    ):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg
        self.world = world
        self.frame = self.skill_cfg.get("frame", "robot")
        self.robot_base_path = self.controller.robot_base_path
        self.T_base_2_world = get_relative_transform(
            get_prim_at_path(self.robot_base_path), get_prim_at_path(self.task.root_prim_path)
        )
        self.table_2_base = self.cal_table_2_base()
        self.way_points = self.sample_waypoints()
        self.last_target_trans = self.way_points[-1][0]
        self.last_target_ori = get_orientation(None, self.way_points[-1][1])
        if "left" in controller.robot_file:
            self.robot_lr = "left"
            self.visual_color = {
                "x": np.array([1.0, 0.0, 0.0]),
                "y": np.array([0.0, 1.0, 0.0]),
                "z": np.array([0.0, 0.0, 1.0]),
            }
        elif "right" in controller.robot_file:
            self.robot_lr = "right"
            self.visual_color = {
                "x": np.array([1.0, 1.0, 0.0]),
                "y": np.array([0.0, 1.0, 1.0]),
                "z": np.array([1.0, 0.0, 1.0]),
            }
        self.vs_cylinder_radius = 0.005
        self.vs_cylinder_height = 0.13
        self.T_tcp_2_ee = np.array(self.skill_cfg.get("T_tcp_2_ee", np.eye(4)))

        # !!! keyposes should be generated after previous skill is done
        self.manip_list = []

        T_tcp_2_world = self.get_tcp_pose()
        trans, ori = self.cal_axis(T_tcp_2_world)
        for axis in ["x", "y", "z"]:
            self.task.visuals[f"{axis}_{self.robot_lr}"] = VisualCylinder(
                prim_path=f"/World/visual/{axis}_{self.robot_lr}",
                radius=self.vs_cylinder_radius,
                height=self.vs_cylinder_height,
                translation=trans[axis],
                orientation=ori[axis],
                color=self.visual_color[f"{axis}"],
            )

    def simple_generate_manip_cmds(self):
        manip_list = []
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        cmd = (p_base_ee_cur, q_base_ee_cur, "update_pose_cost_metric", {"hold_vec_weight": [0, 0, 0, 0, 0, 0]})
        manip_list.append(cmd)

        if self.frame == "robot":
            for target in self.way_points:
                cmd = (np.array(target[0]), get_orientation(None, target[1]), "open_gripper", {})
                manip_list.append(cmd)
        else:
            raise NotImplementedError

        self.manip_list = manip_list

    def get_tcp_pose(self, frame: str = "world"):
        if frame == "world":
            p_base_ee, q_base_ee = self.controller.get_ee_pose()
            T_ee_2_base = tf_matrix_from_pose(p_base_ee, q_base_ee)
            T_tcp_2_world = self.T_base_2_world @ T_ee_2_base @ self.T_tcp_2_ee
            return T_tcp_2_world
        else:
            raise NotImplementedError

    def visualize_target(self, world):
        if len(self.manip_list) > 0:
            p_base_ee, q_base_ee, *_ = self.manip_list[0]
            T_ee_2_base = tf_matrix_from_pose(p_base_ee, q_base_ee)
            T_tcp_2_world = self.T_base_2_world @ T_ee_2_base @ self.T_tcp_2_ee
            trans, ori = self.cal_axis(T_tcp_2_world)

            for axis in ["x", "y", "z"]:
                self.task.visuals[f"{axis}_{self.robot_lr}"].set_world_pose(trans[axis], ori[axis])

            if self.curr_way_points_num > len(self.manip_list):
                self.curr_way_points_num -= 1
                for _ in range(40):
                    world.step(render=True)

        # elif self.task.visuals["x"] and self.task.visuals["y"] and self.task.visuals["z"]:
        #     for axis in ['x', 'y', 'z']:
        #         delete_prim(f"/World/visual/{axis}_{self.robot_lr}")
        #         for i in range(40):
        #             world.step(render=True)

    def cal_axis(self, T):
        origin = T[:3, 3]
        rotation_matrix = T[:3, :3]
        trans = {}
        ori = {}
        axis_table = ["x", "y", "z"]

        for i in range(3):
            axis = rotation_matrix[:, i]

            axis_start = origin
            axis_end = origin + axis * self.vs_cylinder_height
            trans[axis_table[i]] = self.calculate_cylinder_center(axis_start, axis_end)
            ori[axis_table[i]] = self.calculate_orientation(axis)

        return trans, ori

    def calculate_cylinder_center(self, start_point, end_point):
        return (start_point + end_point) / 2

    def calculate_orientation(self, axis_vector):
        default_z_axis = np.array([0, 0, 1])
        target_axis = axis_vector / np.linalg.norm(axis_vector)
        rotation = R.align_vectors([target_axis], [default_z_axis])[0]
        return rotation.as_quat(scalar_first=True)

    def cal_table_2_base(self):
        tgt = self.task.fixtures["table"]
        bbox_tgt = compute_bbox(tgt.prim)
        table_center = (np.asarray(bbox_tgt.min) + np.asarray(bbox_tgt.max)) / 2
        tgt_z_max = bbox_tgt.max[2]
        table_center[2] = tgt_z_max
        base_trans = self.T_base_2_world[:3, 3]
        table_2_base = table_center - base_trans
        table_2_base[0], table_2_base[1] = table_2_base[1], -table_2_base[0]

        return table_2_base

    def from_table_2_base(self, trans, ori):
        return (np.array(trans) + self.table_2_base).tolist(), ori

    def sample_waypoints(self):
        way_points_num = self.skill_cfg.get("way_points_num", 1)
        self.curr_way_points_num = way_points_num
        way_points_trans = self.skill_cfg.get("way_points_trans", None)
        way_points_ori = np.array(self.skill_cfg.get("way_points_ori", None))
        trans_min = np.array(way_points_trans["min"])
        trans_max = np.array(way_points_trans["max"])

        way_points = []
        for i in range(way_points_num):
            while True:
                print(f"... sampling waypoint {i} ...")
                x = np.random.uniform(trans_min[0], trans_max[0])
                y = np.random.uniform(trans_min[1], trans_max[1])
                z = np.random.uniform(trans_min[2], trans_max[2])
                trans = [x, y, z]
                ori = perturb_orientation(way_points_ori, self.skill_cfg.get("max_noise_deg", 5))
                trans, ori = self.from_table_2_base(trans, ori)

                if self.controller.test_single_ik(trans, ori):
                    way_points.append([trans, ori.tolist()])
                    break

        for p in way_points:
            print(p)

        return way_points

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

    def is_success(self, t_eps=5e-3, o_eps=0.087):
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        p_base_ee, q_base_ee = self.last_target_trans, self.last_target_ori
        diff_trans = np.linalg.norm(p_base_ee_cur - p_base_ee)
        diff_ori = 2 * np.arccos(min(abs(np.dot(q_base_ee_cur, q_base_ee)), 1.0))
        pose_flag = np.logical_or(
            diff_trans < t_eps,
            diff_ori < o_eps,
        )
        return pose_flag
