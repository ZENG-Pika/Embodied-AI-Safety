import math
from copy import deepcopy

import numpy as np
from core.utils.usd_geom_utils import compute_bbox
from scipy.spatial.transform import Rotation as R


class RandomRegionSampler:
    @staticmethod
    def A_in_B_region_sampler(obj, tgt, x_bias=0, y_bias=0, z_bias=0):
        bbox_tgt = compute_bbox(tgt.prim)
        tgt_z_max = bbox_tgt.max[2]
        bbox_obj = compute_bbox(obj.prim)
        obj_z_min = bbox_obj.min[2]
        tgt_trans = tgt.get_local_pose()[0]
        obj_trans = deepcopy(tgt_trans)
        obj_trans[0] += x_bias
        obj_trans[1] += y_bias
        obj_trans[2] = tgt_z_max + (obj.get_local_pose()[0][2] - obj_z_min) - 0.005 + z_bias
        obj_ori = obj.get_local_pose()[1]
        return obj_trans, obj_ori

    @staticmethod
    def A_on_B_region_sampler(obj, tgt, pos_range, yaw_rotation):
        # Translation
        shift = np.random.uniform(*pos_range)
        bbox_obj = compute_bbox(obj.prim)
        obj_z_min = bbox_obj.min[2]
        bbox_tgt = compute_bbox(tgt.prim)
        tgt_center = (np.asarray(bbox_tgt.min) + np.asarray(bbox_tgt.max)) / 2
        tgt_z_max = bbox_tgt.max[2]
        place_pos = np.zeros(3)
        place_pos[0] = tgt_center[0]
        place_pos[1] = tgt_center[1]
        place_pos[2] = (
            tgt_z_max + (obj.get_local_pose()[0][2] - obj_z_min) + 0.001
        )  # add a small value to avoid penetration
        place_pos += shift
        # Orientation
        yaw = np.random.uniform(*yaw_rotation)
        dr = R.from_euler("xyz", [0.0, 0.0, yaw], degrees=True)
        r = R.from_quat(obj.get_local_pose()[1], scalar_first=True)
        orientation = (dr * r).as_quat(scalar_first=True)
        return place_pos, orientation

    @staticmethod
    def A_by_B_circle_sampler(obj, tgt, r_range, theta_range, yaw_rotation):
        # Translation
        bbox_tgt = compute_bbox(tgt.prim)
        bbox_obj = compute_bbox(obj.prim)
        tgt_height = (np.asarray(bbox_tgt.max) - np.asarray(bbox_tgt.min)) / 2
        tgt_center = (np.asarray(bbox_tgt.min) + np.asarray(bbox_tgt.max)) / 2
        obj_height = (np.asarray(bbox_obj.max) - np.asarray(bbox_obj.min)) / 2
        r = np.random.uniform(*r_range)
        theta = np.random.uniform(*theta_range)
        delta_x = r * math.cos(theta / 180 * math.pi)
        delta_y = r * math.sin(theta / 180 * math.pi)
        delta_z = obj_height[2] - tgt_height[2]
        place_pos = np.zeros(3)
        place_pos[0] = tgt_center[0] + delta_x
        place_pos[1] = tgt_center[1] + delta_y
        place_pos[2] = tgt_center[2] + delta_z
        # Orientation
        yaw = np.random.uniform(*yaw_rotation)
        dr = R.from_euler("xyz", [0.0, 0.0, yaw], degrees=True)
        r = R.from_quat(obj.get_local_pose()[1], scalar_first=True)
        orientation = (dr * r).as_quat(scalar_first=True)
        return place_pos, orientation

    @staticmethod
    def A_by_B_region_sampler(obj, tgt, pos_range, yaw_rotation):
        # Translation
        shift = np.random.uniform(*pos_range)
        bbox_tgt = compute_bbox(tgt.prim)
        bbox_obj = compute_bbox(obj.prim)
        tgt_height = (np.asarray(bbox_tgt.max) - np.asarray(bbox_tgt.min)) / 2
        tgt_center = (np.asarray(bbox_tgt.min) + np.asarray(bbox_tgt.max)) / 2
        obj_height = (np.asarray(bbox_obj.max) - np.asarray(bbox_obj.min)) / 2
        delta_x = shift[0]
        delta_y = shift[1]
        delta_z = obj_height[2] - tgt_height[2]
        place_pos = np.zeros(3)
        place_pos[0] = tgt_center[0] + delta_x
        place_pos[1] = tgt_center[1] + delta_y
        place_pos[2] = tgt_center[2] + delta_z
        # Orientation
        yaw = np.random.uniform(*yaw_rotation)
        dr = R.from_euler("xyz", [0.0, 0.0, yaw], degrees=True)
        r = R.from_quat(obj.get_local_pose()[1], scalar_first=True)
        orientation = (dr * r).as_quat(scalar_first=True)
        return place_pos, orientation

    @staticmethod
    def A_face_B_circle_sampler(obj, tgt, r_range, yaw_rotation):
        # Translation
        bbox_tgt = compute_bbox(tgt.prim)
        bbox_obj = compute_bbox(obj.prim)
        tgt_height = (np.asarray(bbox_tgt.max) - np.asarray(bbox_tgt.min)) / 2
        tgt_center = (np.asarray(bbox_tgt.min) + np.asarray(bbox_tgt.max)) / 2
        obj_height = (np.asarray(bbox_obj.max) - np.asarray(bbox_obj.min)) / 2
        r = np.random.uniform(*r_range)
        orientation = tgt.get_world_pose()[1]
        rot = R.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])
        euler_angles = rot.as_euler("xyz", degrees=True)
        theta = euler_angles[2]
        delta_x = r * math.cos(theta / 180 * math.pi)
        delta_y = r * math.sin(theta / 180 * math.pi)
        delta_z = obj_height[2] - tgt_height[2]
        place_pos = np.zeros(3)
        place_pos[0] = tgt_center[0] + delta_x
        place_pos[1] = tgt_center[1] + delta_y
        place_pos[2] = tgt_center[2] + delta_z
        # Orientation
        yaw = np.random.uniform(*yaw_rotation)
        dr = R.from_euler("xyz", [0.0, 0.0, yaw], degrees=True)
        r = R.from_quat(obj.get_local_pose()[1], scalar_first=True)
        orientation = (dr * r).as_quat(scalar_first=True)
        return place_pos, orientation

    @staticmethod
    def A_along_B_C_circle_sampler(obj, tgt, tgt2, r_range, yaw_rotation):
        # Translation
        r_tgt_tgt2 = np.linalg.norm(tgt.get_world_pose()[0] - tgt2.get_world_pose()[0])
        bbox_tgt = compute_bbox(tgt.prim)
        bbox_obj = compute_bbox(obj.prim)
        tgt_height = (np.asarray(bbox_tgt.max) - np.asarray(bbox_tgt.min)) / 2
        tgt_center = (np.asarray(bbox_tgt.min) + np.asarray(bbox_tgt.max)) / 2
        obj_height = (np.asarray(bbox_obj.max) - np.asarray(bbox_obj.min)) / 2
        r = np.random.uniform(*r_range) + r_tgt_tgt2
        orientation = tgt.get_world_pose()[1]
        rot = R.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])
        euler_angles = rot.as_euler("xyz", degrees=True)
        theta = euler_angles[2]
        delta_x = r * math.cos(theta / 180 * math.pi)
        delta_y = r * math.sin(theta / 180 * math.pi)
        delta_z = obj_height[2] - tgt_height[2]
        place_pos = np.zeros(3)
        place_pos[0] = tgt_center[0] + delta_x
        place_pos[1] = tgt_center[1] + delta_y
        place_pos[2] = tgt_center[2] + delta_z
        # Orientation
        yaw = np.random.uniform(*yaw_rotation)
        dr = R.from_euler("xyz", [0.0, 0.0, yaw], degrees=True)
        r = R.from_quat(obj.get_local_pose()[1], scalar_first=True)
        orientation = (dr * r).as_quat(scalar_first=True)
        return place_pos, orientation
