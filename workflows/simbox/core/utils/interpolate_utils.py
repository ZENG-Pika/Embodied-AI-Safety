"""
Interpolation utilities for simbox motion skills.

This module provides small helpers for:

- linear_interpolation: joint-space linear interpolation between a current
  joint state and a target joint state over a fixed number of steps. Used in
  skills such as joint_ctrl / approach_rotate / rotate_obj to generate smooth
  joint trajectories.
- pose_interpolation: Cartesian pose interpolation where translation is
  linearly interpolated and orientation uses spherical linear interpolation
  (SLERP) between two quaternions. Used in rotate_random and similar pose-based
  skills.
- cal_midpoint: simple midpoint computation between two 3D points, useful for
  computing intermediate waypoints or grasp targets.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def linear_interpolation(curr_js, target_js, num_steps=10):
    interpolated = []
    for i in range(num_steps + 1):
        alpha = i / num_steps
        interpolated_js = curr_js + alpha * (target_js - curr_js)
        interpolated.append(interpolated_js)

    return np.array(interpolated)


def pose_interpolation(curr_trans, curr_ori, target_trans, target_ori, interp_num, normalize_quaternions=True):
    # Translation interpolation (linear)
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


def cal_midpoint(start_point, end_point):
    return (start_point + end_point) / 2
