"""Extract all Sim_Raw_GT fields from LMDB and output a complete JSON.

Fills available fields from logger data, sets missing fields to null.
Provides hooks for future PhysX / perception / HRI data sources.

Usage:
    python3 -m safety_risk.raw_gt_extractor <lmdb_dir> [-o output.json]
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_float(val) -> Optional[float]:
    """Convert to float, return None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_list(val) -> Optional[List]:
    """Convert to list, return None on failure."""
    if val is None:
        return None
    try:
        if hasattr(val, 'tolist'):  # numpy array
            return val.tolist()
        return list(val)
    except (TypeError, ValueError):
        return None


def _position_from_transform(mat) -> Optional[List[float]]:
    """Extract [x,y,z] from 4x4 numpy matrix."""
    try:
        import numpy as np
        if isinstance(mat, np.ndarray) and mat.shape == (4, 4):
            return [float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])]
    except ImportError:
        pass
    return None


def _quat_from_transform(mat) -> Optional[List[float]]:
    """Extract [qx,qy,qz,qw] from 4x4 numpy matrix."""
    try:
        import numpy as np
        if isinstance(mat, np.ndarray) and mat.shape == (4, 4):
            r = mat[:3, :3]
            trace = r[0, 0] + r[1, 1] + r[2, 2]
            if trace > 0:
                s = 0.5 / math.sqrt(trace + 1.0)
                return [float((r[2, 1] - r[1, 2]) * s),
                        float((r[0, 2] - r[2, 0]) * s),
                        float((r[1, 0] - r[0, 1]) * s),
                        float(0.25 / s)]
    except ImportError:
        pass
    return None


def _transform_to_pose(mat) -> Optional[List[float]]:
    """Convert 4x4 matrix to [x,y,z,qx,qy,qz,qw]."""
    pos = _position_from_transform(mat)
    quat = _quat_from_transform(mat)
    if pos and quat:
        return pos + quat
    return None


def _quat_to_rotmat_xyzw(quat: List[float]) -> Optional[List[List[float]]]:
    """Convert [qx,qy,qz,qw] to a 3x3 rotation matrix.

    Returns a world-from-local rotation matrix if the quaternion describes the
    local frame in world coordinates, which is the convention used by ee_pose_gt.
    """
    if quat is None or len(quat) < 4:
        return None
    qx, qy, qz, qw = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n <= 0.0:
        return None
    s = 2.0 / n

    xx = qx * qx * s
    yy = qy * qy * s
    zz = qz * qz * s
    xy = qx * qy * s
    xz = qx * qz * s
    yz = qy * qz * s
    wx = qw * qx * s
    wy = qw * qy * s
    wz = qw * qz * s

    return [
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ]


def _world_to_local(rot_world_from_local: List[List[float]], vec_world: List[float]) -> Optional[List[float]]:
    """Transform a world-space vector into the local frame."""
    if rot_world_from_local is None or vec_world is None or len(vec_world) < 3:
        return None

    # local = R^T * world_vec
    return [
        rot_world_from_local[0][0] * float(vec_world[0]) + rot_world_from_local[1][0] * float(vec_world[1]) + rot_world_from_local[2][0] * float(vec_world[2]),
        rot_world_from_local[0][1] * float(vec_world[0]) + rot_world_from_local[1][1] * float(vec_world[1]) + rot_world_from_local[2][1] * float(vec_world[2]),
        rot_world_from_local[0][2] * float(vec_world[0]) + rot_world_from_local[1][2] * float(vec_world[1]) + rot_world_from_local[2][2] * float(vec_world[2]),
    ]


def _safe_scalar(val) -> Optional[float]:
    """Convert scalar-like values, including single-item lists/strings, to float."""
    if val is None:
        return None
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        if not val:
            return None
        val = val[0]
    if isinstance(val, str):
        text = val.strip()
        if text.startswith("array(") and "[" in text and "]" in text:
            text = text[text.find("[") + 1:text.find("]")]
        elif text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if "," in text:
            text = text.split(",", 1)[0]
        text = text.strip()
        if not text:
            return None
        val = text
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _sample_pose_xyzw(poses: List, target_idx: int, target_len: int) -> Optional[List[float]]:
    """Sample [x,y,z,qx,qy,qz,qw] pose on another timeline."""
    if not poses or target_idx < 0 or target_len <= 0:
        return None
    valid_len = len(poses)
    if valid_len == 1 or target_len == 1:
        pose = poses[0]
        return [float(v) for v in pose[:7]] if pose is not None and len(pose) >= 7 else None

    src_pos = target_idx * (valid_len - 1) / max(target_len - 1, 1)
    lo = int(math.floor(src_pos))
    hi = min(lo + 1, valid_len - 1)
    alpha = src_pos - lo
    pose_lo = poses[lo]
    pose_hi = poses[hi]
    if pose_lo is None or pose_hi is None or len(pose_lo) < 7 or len(pose_hi) < 7:
        return None

    p0 = [float(v) for v in pose_lo[:7]]
    p1 = [float(v) for v in pose_hi[:7]]
    pos = [(1.0 - alpha) * p0[i] + alpha * p1[i] for i in range(3)]

    q0 = p0[3:7]
    q1 = p1[3:7]
    dot = sum(q0[i] * q1[i] for i in range(4))
    if dot < 0.0:
        q1 = [-v for v in q1]
    quat = [(1.0 - alpha) * q0[i] + alpha * q1[i] for i in range(4)]
    q_norm = math.sqrt(sum(v * v for v in quat))
    if q_norm <= 0.0:
        return None
    quat = [v / q_norm for v in quat]
    return pos + quat


def _ee_pose_series(robot: Dict[str, Any], arm: str) -> Optional[List]:
    """Read merged ``ee_pose_gt`` and remain compatible with legacy reports."""
    merged = robot.get("ee_pose_gt")
    if isinstance(merged, list) and merged:
        if isinstance(merged[0], dict):
            return [
                frame.get(arm) if isinstance(frame, dict) else None
                for frame in merged
            ]
        if arm == "left":
            return merged
    legacy = robot.get("ee_pose_right_gt" if arm == "right" else "ee_pose_gt")
    return legacy if isinstance(legacy, list) else None


def _infer_position_scale_to_m(positions: List, indices: Optional[List[int]] = None) -> float:
    """Infer whether a position series is stored in cm and return scale to m."""
    if not positions:
        return 1.0

    sample_indices = indices if indices else list(range(len(positions)))
    max_abs = 0.0
    for i in sample_indices:
        if i < 0 or i >= len(positions):
            continue
        pos = positions[i]
        if pos is None or len(pos) < 3:
            continue
        try:
            max_abs = max(max_abs, abs(float(pos[0])), abs(float(pos[1])), abs(float(pos[2])))
        except (TypeError, ValueError):
            continue

    # Tabletop object coordinates should not exceed several meters. Values above
    # this are treated as centimeters and converted to meters for raw GT.
    return 0.01 if max_abs > 5.0 else 1.0


class SimRawGTExtractor:
    """Extract complete Sim_Raw_GT from LMDB data."""

    def __init__(self):
        self._warnings: List[str] = []

    @property
    def warnings(self) -> List[str]:
        return list(self._warnings)

    def extract_from_lmdb(self, lmdb_dir: str, task_cfg: Optional[Dict] = None, dt: float = 0.033) -> Dict[str, Any]:
        """Extract all Sim_Raw_GT fields from an LMDB directory.

        Parameters
        ----------
        lmdb_dir : str
            Path to the episode directory (containing lmdb/ and meta_info.pkl).
        task_cfg : dict, optional
            Task configuration dict for injecting metadata.
        dt : float
            Simulation timestep in seconds (default 0.033 = 30Hz).

        Returns
        -------
        dict
            Complete Sim_Raw_GT with all fields. Missing fields are null.
        """
        self._warnings = []
        self._task_cfg = task_cfg or {}
        self._dt = dt

        meta_path = os.path.join(lmdb_dir, "meta_info.pkl")
        lmdb_path = os.path.join(lmdb_dir, "lmdb")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta_info.pkl not found: {meta_path}")
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"lmdb not found: {lmdb_path}")

        meta = pickle.load(open(meta_path, "rb"))

        # Read all data from LMDB
        data = self._read_lmdb(lmdb_path)

        # Build complete Sim_Raw_GT
        raw_gt = {
            "metadata": self._extract_metadata(meta, lmdb_dir),
            "episode_meta": self._extract_episode_meta(data, meta),
            "robot_state": self._extract_robot_state(data),
            "object_state": self._extract_object_state(data),
            "environment_state": self._extract_environment_state(data),
            "distance_gt": self._extract_distance_gt(data),
            "collision_gt": self._extract_collision_gt(data),
            "gripper_gt": self._extract_gripper_gt(data),
            "outcome_gt": self._extract_outcome_gt(data),
            "planner_log": self._extract_planner_log(data),
            "sensor_gt": self._extract_sensor_gt(data, meta),
            "hri_log": self._extract_hri_log(data),
            "warnings": self._warnings,
        }

        # ── Compute derived fields from existing data ──
        self._compute_derived_fields(raw_gt)

        return raw_gt

    def _read_lmdb(self, lmdb_path: str) -> Dict[str, Any]:
        """Read all key-value pairs from LMDB."""
        import lmdb

        result: Dict[str, Any] = {}
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                key_str = key.decode("utf-8")
                try:
                    result[key_str] = pickle.loads(value)
                except Exception:
                    result[key_str] = None
        env.close()
        return result

    # ── Metadata ─────────────────────────────────────────────────────────────

    def _extract_metadata(self, meta: Dict, lmdb_dir: str) -> Dict[str, Any]:
        return {
            "source": "InternDataEngine LMDB",
            "extract_time": datetime.utcnow().isoformat(),
            "lmdb_path": lmdb_dir,
            "num_steps": meta.get("num_steps", 0),
            "language_instruction": meta.get("language_instruction", ""),
            "detailed_language_instruction": meta.get("detailed_language_instruction", ""),
        }

    # ── Episode Meta (S-COM-*) ───────────────────────────────────────────────

    def _extract_episode_meta(self, data: Dict, meta: Dict) -> Dict[str, Any]:
        cfg = self._task_cfg

        # physics_config from simulator config
        physics_cfg = cfg.get("simulator", None)
        if physics_cfg is None:
            # Try to extract from the pipeline config's simulator section
            physics_cfg = {
                "physics_dt": "1/30",
                "rendering_dt": "1/30",
            }

        # lighting_config from env_map config
        lighting_cfg = cfg.get("env_map", None)

        return {
            "episode_id": None,           # 由调用方填充
            "scenario_id": cfg.get("name", None),
            "random_seed": cfg.get("random_seed", None),
            "task_type": cfg.get("task", None),
            "object_id": None,            # 由调用方填充
            "target_object_id": None,     # 由调用方填充
            "target_pose": None,          # TODO: 需要目标区域定义
            "object_hazard_class": cfg.get("object_hazard_class", None),
            "object_fragility_class": cfg.get("object_fragility_class", None),
            "physics_config": physics_cfg,
            "lighting_config": lighting_cfg,
            "sensor_noise_config": cfg.get("sensor_noise_config", None),
        }

    # ── Robot State (S-ROBOT-*) ──────────────────────────────────────────────

    def _extract_robot_state(self, data: Dict) -> Dict[str, Any]:
        # Find robot name (first key in proprio-like data)
        robot_name = None
        for key in ["states.left_joint.position", "T_base_ee_fl"]:
            for k in data:
                if k == key:
                    # Infer robot name from meta
                    robot_name = "split_aloha"  # default
                    break

        # Joint positions
        q_left = data.get("states.left_joint.position")
        q_right = data.get("states.right_joint.position")

        # EE transforms
        T_ee_fl = data.get("T_base_ee_fl")
        T_ee_fr = data.get("T_base_ee_fr")
        T_world_base = data.get("T_world_base")

        # Convert EE transforms to world-frame poses
        ee_pose_left = self._convert_to_world_poses(T_ee_fl, T_world_base)
        ee_pose_right = self._convert_to_world_poses(T_ee_fr, T_world_base)

        return {
            # S-ROBOT-001
            "joint_position_q_gt": self._safe_list_list(q_left),
            "joint_position_q_right_gt": self._safe_list_list(q_right),
            # S-ROBOT-002
            "joint_velocity_dq_gt": self._safe_list_list(data.get("qvel")),
            # S-ROBOT-003
            "joint_acceleration_gt": None,  # TODO: 需要从仿真获取
            # S-ROBOT-004
            "joint_torque_gt": None,        # TODO: 需要从 PhysX 获取
            # S-ROBOT-005
            "link_pose_gt": None,           # TODO: 需要从 PhysX 获取所有 link pose
            # S-ROBOT-006
            "link_velocity_gt": None,       # TODO: 需要从 PhysX 获取
            # S-ROBOT-007
            "ee_pose_gt": ee_pose_left,     # [x,y,z,qx,qy,qz,qw] per step
            "ee_pose_right_gt": ee_pose_right,
            # 额外
            "T_base_ee_fl": "available",    # 原始 4x4 矩阵在 LMDB 中
            "T_base_ee_fr": "available",
            "T_world_base": "available",
        }

    # ── Object State (S-OBJ-*) ───────────────────────────────────────────────

    def _extract_object_state(self, data: Dict) -> Dict[str, Any]:
        import numpy as np

        # Find all objects by looking for */translation keys
        objects = {}
        for key in data:
            if key.endswith("/translation"):
                obj_name = key.replace("/translation", "")
                trans = data[key]
                orient = data.get(f"{obj_name}/orientation")
                objects[obj_name] = {
                    "translation_per_step": self._safe_list_list(trans),
                    "orientation_per_step": self._safe_list_list(orient),
                    "translation_initial": _safe_list(trans[0]) if trans is not None and len(trans) > 0 else None,
                }

        # Compute per-object velocity from translation time series (S-OBJ-002, S-OBJ-003)
        object_velocities = {}
        object_angular_velocities = {}
        dt = self._dt

        for obj_name, obj_data in objects.items():
            trans_list = obj_data.get("translation_per_step")
            orient_list = obj_data.get("orientation_per_step")

            # Linear velocity from translation difference
            if trans_list and len(trans_list) >= 2:
                velocities = []
                for i in range(1, len(trans_list)):
                    p0 = trans_list[i - 1]
                    p1 = trans_list[i]
                    if p0 and p1 and len(p0) >= 3 and len(p1) >= 3:
                        vx = (p1[0] - p0[0]) / dt
                        vy = (p1[1] - p0[1]) / dt
                        vz = (p1[2] - p0[2]) / dt
                        velocities.append([vx, vy, vz])
                    else:
                        velocities.append([0.0, 0.0, 0.0])
                # Pad first step with zero
                velocities.insert(0, [0.0, 0.0, 0.0])
                object_velocities[obj_name] = velocities

            # Angular velocity from quaternion difference (simplified)
            if orient_list and len(orient_list) >= 2:
                angular_vels = []
                for i in range(1, len(orient_list)):
                    q0 = orient_list[i - 1]
                    q1 = orient_list[i]
                    if q0 and q1 and len(q0) >= 4 and len(q1) >= 4:
                        # Simplified: use quaternion difference as angular velocity proxy
                        # For production, use proper quaternion log/difference
                        try:
                            dq = [q1[j] - q0[j] for j in range(4)]
                            # Convert to approximate angular velocity (rad/s)
                            norm = math.sqrt(sum(x * x for x in dq[:3]))
                            wx = dq[0] * 2.0 / dt if len(dq) > 0 else 0.0
                            wy = dq[1] * 2.0 / dt if len(dq) > 1 else 0.0
                            wz = dq[2] * 2.0 / dt if len(dq) > 2 else 0.0
                            angular_vels.append([wx, wy, wz])
                        except Exception:
                            angular_vels.append([0.0, 0.0, 0.0])
                    else:
                        angular_vels.append([0.0, 0.0, 0.0])
                angular_vels.insert(0, [0.0, 0.0, 0.0])
                object_angular_velocities[obj_name] = angular_vels

        return {
            # S-OBJ-001
            "object_pose_gt": objects,
            # S-OBJ-002
            "object_velocity_gt": object_velocities if object_velocities else None,
            # S-OBJ-003
            "object_angular_velocity_gt": object_angular_velocities if object_angular_velocities else None,
            # S-OBJ-004
            "object_physical_params": None,      # TODO: 从 USD 物理属性读取
        }

    # ── Environment State (S-ENV-*, S-HUM-*) ─────────────────────────────────

    def _extract_environment_state(self, data: Dict) -> Dict[str, Any]:
        # Collect obstacle poses - convert numpy arrays to plain lists
        obstacle_poses = {}
        for name in ["obstacle_1", "obstacle_2", "obstacle_3"]:
            if f"{name}/translation" in data:
                obstacle_poses[name] = {
                    "translation": self._convert_to_float_lists(data.get(f"{name}/translation")),
                    "orientation": self._convert_to_float_lists(data.get(f"{name}/orientation")),
                }

        # S-HUM-001: Use obstacle poses as human surrogate
        human_surrogate = None
        if obstacle_poses:
            human_surrogate = {
                "surrogate_type": "obstacle",
                "note": "Obstacle poses used as human body surrogate for safety evaluation",
                "body_parts": {}
            }
            for name, pose_data in obstacle_poses.items():
                human_surrogate["body_parts"][name] = pose_data

        return {
            # S-ENV-001
            "scene_mesh_gt": None,              # TODO: 从 USD 场景读取
            # S-ENV-002
            "obstacle_pose_gt": obstacle_poses,
            # S-HUM-001: obstacle surrogates as human body pose
            "human_body_pose_gt": human_surrogate,
            # S-HUM-002: Use moving obstacle trajectory as intrusion
            "intrusion_trajectory_gt": self._extract_intrusion_trajectory(data),
            # 额外
            "table_boundary": None,             # TODO: 从场景几何读取
            "support_surface": None,            # TODO: 从场景几何读取
        }

    def _extract_intrusion_trajectory(self, data: Dict) -> Optional[Dict]:
        """S-HUM-002: Extract intrusion trajectory from moving obstacle data.

        The moving obstacle (obstacle_1 = MANO hand) serves as the
        human intrusion trajectory.
        """
        obs_key = "obstacle_1/translation"
        if obs_key not in data:
            return None

        trans = data[obs_key]
        if trans is None or len(trans) == 0:
            return None

        trajectory = []
        dt = self._dt
        for i, pos in enumerate(trans):
            if pos is not None and hasattr(pos, '__len__') and len(pos) >= 3:
                trajectory.append({
                    "t": round(i * dt, 4),
                    "position_m": [float(pos[0]), float(pos[1]), float(pos[2])],
                })

        if not trajectory:
            return None

        return {
            "source": "obstacle_1 (MANO hand surrogate)",
            "n_steps": len(trajectory),
            "trajectory": trajectory,
        }

    # ── Distance GT (S-DIST-*) ───────────────────────────────────────────────

    def _extract_distance_gt(self, data: Dict) -> Dict[str, Any]:
        return {
            # S-DIST-001
            "robot_human_distance_matrix_gt": None,  # TODO: 需要 PhysX 距离查询
            # S-DIST-002
            "ee_human_distance_gt": None,             # TODO: 需要人体模型 + PhysX
            # S-DIST-003
            "object_human_distance_gt": None,         # TODO: 需要人体模型
            # S-DIST-004
            "object_env_distance_gt": None,           # TODO: 需要 PhysX 距离查询
            # S-DIST-005
            "link_env_distance_gt": None,             # TODO: 需要 PhysX 距离查询
            # S-DIST-006
            "self_distance_gt": None,                 # TODO: 需要 PhysX 自碰撞检测
            # 我们自己从 pose 计算的近似值
            "ee_obstacle_distance_approx_m": None,    # 由 adapter 填充，单位 m
        }

    # ── Collision GT (S-COLL-*) ──────────────────────────────────────────────

    def _extract_collision_gt(self, data: Dict) -> Dict[str, Any]:
        return {
            # S-COLL-001
            "collision_pair_gt": None,        # TODO: 需要 PhysX 碰撞回调
            # S-COLL-002
            "collision_location_gt": None,    # TODO: 需要 PhysX
            # S-COLL-003
            "penetration_depth_gt": None,     # TODO: 需要 PhysX
            # S-COLL-004
            "contact_force_gt": None,         # TODO: 需要 PhysX 力传感器
            # S-COLL-005
            "contact_impulse_gt": None,       # TODO: 需要 PhysX
            # S-COLL-006
            "contact_duration_gt": None,      # TODO: 需要 PhysX
            # 额外
            "contact_normal": None,           # TODO: 需要 PhysX
        }

    # ── Gripper GT (S-GRASP-*) ───────────────────────────────────────────────

    def _extract_gripper_gt(self, data: Dict) -> Dict[str, Any]:
        gripper_left = data.get("states.left_gripper.position")
        gripper_right = data.get("states.right_gripper.position")

        return {
            # S-GRASP-001
            "gripper_object_contact_force_gt": None,  # TODO: 需要力传感器
            # S-GRASP-002
            "slip_distance_gt": None,                  # TODO: 需要相对位移计算
            # S-GRASP-003
            "grasp_state_gt": None,                    # TODO: 需要抓取状态检测
            # 额外
            "gripper_width_left": _safe_list(gripper_left),
            "gripper_width_right": _safe_list(gripper_right),
            "gripper_force_gt": None,                  # TODO: 需要力传感器
            "object_relative_pose_to_gripper": None,   # TODO: 需要计算
        }

    # ── Outcome GT (S-OUT-*) ─────────────────────────────────────────────────

    def _extract_outcome_gt(self, data: Dict) -> Dict[str, Any]:
        return {
            # S-OUT-001
            "drop_event_gt": None,                # TODO: 需要掉落检测
            # S-OUT-002
            "drop_height_gt": None,               # TODO: 需要物体 z 轨迹分析
            "damage_state_gt": None,              # TODO: 需要损坏模型或规则
            # 额外
            "final_object_pose": None,            # TODO: 从最后一步 object_pose 读取
            "placement_error_pos_gt": None,       # TODO: 需要 target_pose 比较
            "placement_error_rot_gt": None,       # TODO: 需要 target_orientation 比较
            "stable_final_gt": None,              # TODO: 需要速度+裕度判断
        }

    # ── Planner Log (S-PLAN-*) ───────────────────────────────────────────────

    def _extract_planner_log(self, data: Dict) -> Dict[str, Any]:
        return {
            # S-PLAN-001
            "planned_trajectory": None,       # TODO: 需要 planner 日志
            # S-PLAN-002
            "executed_trajectory": None,      # TODO: 从关节轨迹构建
            # S-PLAN-003
            "safety_gate_status": None,       # TODO: 需要安全门控日志
            # 额外
            "replan_flag": None,              # TODO: 需要 planner 日志
            "t_replan_s": None,               # TODO: 需要 planner 日志
            "stop_command_sent": None,        # TODO: 需要控制器日志
            "stop_success": None,             # injected by PhysX collector
            "stop_margin_s": None,            # injected by PhysX collector
            "t_stop_s": None,                 # injected by PhysX collector
            "robot_motion_started": None,     # TODO: 需要运动检测
        }

    # ── Sensor GT (S-SENSOR-*) ───────────────────────────────────────────────

    def _extract_sensor_gt(self, data: Dict, meta: Dict) -> Dict[str, Any]:
        keys_info = meta.get("keys", {})

        # Check actual LMDB keys for sensor data availability
        has_rgb = any("rgb" in str(k).lower() for k in keys_info)
        has_depth = any("depth" in str(k).lower() for k in keys_info)
        has_seg = any("seg" in str(k).lower() for k in keys_info)

        # Also check for depth/seg data directly in LMDB
        if not has_depth:
            has_depth = any("depth" in k.lower() for k in data.keys())
        if not has_seg:
            has_seg = any("seg" in k.lower() for k in data.keys())

        return {
            # S-SENSOR-001
            "virtual_rgb": "available" if has_rgb else None,
            # S-SENSOR-002
            "virtual_depth": "available" if has_depth else None,
            # S-SENSOR-003
            "segmentation_mask_gt": "available" if has_seg else None,
            # S-SENSOR-004
            "instance_id_map_gt": None,       # TODO: 需要实例分割
            # S-SENSOR-005
            "object_bbox_gt": None,           # TODO: 需要 bbox 检测
            # S-SENSOR-006
            "visibility_ratio_gt": None,      # TODO: 需要可见性分析
        }

    # ── HRI Log (S-HRI-*) ────────────────────────────────────────────────────

    def _extract_hri_log(self, data: Dict) -> Dict[str, Any]:
        cfg = self._task_cfg
        data_cfg = cfg.get("data", cfg)  # language_instruction may be in data sub-dict

        # S-HRI-002: Read from task config, default false
        unsafe_flag = cfg.get("unsafe_instruction_flag", False)
        if not unsafe_flag:
            unsafe_flag = data_cfg.get("unsafe_instruction_flag", False)

        return {
            # S-HRI-001
            "user_command_text": data_cfg.get("language_instruction", cfg.get("language_instruction", None)),
            # S-HRI-002
            "unsafe_instruction_flag_gt": bool(unsafe_flag),
            # S-HRI-003
            "tool_call_trace": None,              # TODO: 需要 agent 日志
            # 额外
            "model_response": None,               # TODO: 需要 LLM 日志
            "refusal_flag": None,                 # TODO: 需要 LLM 日志
            "clarification_requested": None,      # TODO: 需要 LLM 日志
            "stop_command_obeyed": None,          # TODO: 需要控制器日志
            "instruction_safety_assessment": None,
        }

    def _find_pick_object(self, obj_poses: Dict) -> Optional[Dict]:
        """Find the pick object from object poses.

        Handles pick_object, pick_object_left, pick_object_right naming.
        Returns the first match found, or the first object if no pick_ prefix found.
        """
        if not obj_poses:
            return None

        # Try exact match first
        for name in ["pick_object", "pick_object_left", "pick_object_right"]:
            if name in obj_poses:
                return obj_poses[name]

        # Try prefix match
        for name in obj_poses:
            if name.startswith("pick_"):
                return obj_poses[name]

        # Fallback: return first object
        return next(iter(obj_poses.values()), None)

    # ── Derived fields computation ───────────────────────────────────────────

    def _compute_derived_fields(self, raw_gt: Dict) -> None:
        """Compute derived fields from existing raw GT data.

        Computes:
        - S-ROBOT-003: joint_acceleration_gt (from joint velocity diff)
        - S-GRASP-002: slip_distance_gt (from object-gripper relative pose)
        - S-GRASP-003: grasp_state_gt (from gripper width + relative object motion)
        - S-OUT-001: drop_event_gt (from grasp_state_gt)
        - S-OUT-002: drop_height_gt (from dropped interval, unit: m)
        """
        dt = self._dt

        # ── S-ROBOT-003: joint_acceleration_gt ──
        self._compute_joint_acceleration(raw_gt, dt)

        # ── S-GRASP-003: grasp_state_gt ──
        self._compute_grasp_state(raw_gt)

        # ── S-GRASP-002: slip_distance_gt ──
        # Grasp state is computed first so post-release/drop contact cannot be
        # counted as slip inside the gripper.
        self._compute_slip_distance(raw_gt)

        # ── S-OUT-001 & S-OUT-002: drop_event_gt, drop_height_gt ──
        self._compute_drop_detection(raw_gt, dt)

        # ── S-OUT-004: damage_state_gt ──
        self._compute_damage_state(raw_gt)

        # ── S-PLAN-002: executed_trajectory (compute first, used by torque) ──
        self._compute_executed_trajectory(raw_gt)

        # ── S-ROBOT-004: joint_torque_gt (PD controller estimate, needs executed_trajectory) ──
        self._compute_joint_torque_from_pd(raw_gt, dt)

        # ── S-DIST-001/002/003: human distances (using obstacle surrogates) ──
        self._compute_human_distances_from_obstacles(raw_gt)

    def _compute_joint_acceleration(self, raw_gt: Dict, dt: float) -> None:
        """S-ROBOT-003: Compute joint acceleration from velocity time series."""
        robot = raw_gt.get("robot_state", {})
        dq = robot.get("joint_velocity_dq_gt")
        if dq is None or len(dq) < 2:
            robot["joint_acceleration_gt"] = None
            return

        acc = []
        for i in range(1, len(dq)):
            step_acc = []
            for j in range(min(len(dq[i]), len(dq[i - 1]))):
                if dq[i][j] is not None and dq[i - 1][j] is not None:
                    step_acc.append((dq[i][j] - dq[i - 1][j]) / dt)
                else:
                    step_acc.append(None)
            acc.append(step_acc)
        # No previous sample exists for the first frame.  Keep it explicitly
        # unavailable instead of inventing a zero acceleration measurement.
        n_joints = len(dq[0]) if dq and dq[0] else 0
        acc.insert(0, [None] * n_joints)

        robot["joint_acceleration_gt"] = acc

    def normalize_arm_joint_state(
        self,
        raw_gt: Dict,
        arm_dof_indices: List[int],
        arm_dof_names: List[str],
        left_dof_count: int,
        dt: Optional[float] = None,
        effort_limits_by_index: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Align arm position, velocity, acceleration and effort to one DOF axis.

        LMDB stores left/right arm positions separately but stores articulation
        velocity as a full robot vector.  PhysX measured effort is also a full
        articulation vector.  The Sim_Raw_GT arm contract requires all four
        channels to use exactly the same arm-only DOF names and order.
        """
        robot = raw_gt.setdefault("robot_state", {})
        episode_meta = raw_gt.setdefault("episode_meta", {})
        physics = episode_meta.setdefault("physics_config", {})

        indices = [int(index) for index in arm_dof_indices]
        names = [str(name) for name in arm_dof_names]
        if not indices or len(indices) != len(names):
            raise ValueError("arm DOF indices/names must be non-empty and aligned")
        left_count = max(0, min(int(left_dof_count), len(indices)))

        def _slice_series(series):
            if not isinstance(series, list):
                return None
            result = []
            for row in series:
                if row is None:
                    result.append([None] * len(indices))
                    continue
                values = list(row)
                if len(values) == len(indices):
                    result.append(values)
                    continue
                result.append([
                    values[index] if 0 <= index < len(values) else None
                    for index in indices
                ])
            return result

        # Position is recorded as two six-joint arrays by the logger.  Merge
        # them into the same left-then-right axis used by velocity and effort.
        q_main = robot.get("joint_position_q_gt")
        q_right = robot.get("joint_position_q_right_gt")
        q_aligned = None
        if isinstance(q_main, list) and q_main:
            first_width = len(q_main[0]) if isinstance(q_main[0], list) else 0
            if first_width == len(indices):
                q_aligned = [list(row) if row is not None else [None] * len(indices)
                             for row in q_main]
            elif isinstance(q_right, list) and q_right:
                frame_count = max(len(q_main), len(q_right))
                q_aligned = []
                for frame in range(frame_count):
                    left_row = q_main[frame] if frame < len(q_main) else None
                    right_row = q_right[frame] if frame < len(q_right) else None
                    left_values = list(left_row) if left_row is not None else [None] * left_count
                    right_values = (
                        list(right_row) if right_row is not None
                        else [None] * (len(indices) - left_count)
                    )
                    q_aligned.append(
                        (left_values[:left_count] + right_values[:len(indices) - left_count])
                    )
            else:
                q_aligned = _slice_series(q_main)

        robot["joint_position_q_gt"] = q_aligned
        robot["joint_velocity_dq_gt"] = _slice_series(
            robot.get("joint_velocity_dq_gt")
        )
        robot["joint_torque_gt"] = _slice_series(robot.get("joint_torque_gt"))

        step_dt = float(dt if dt is not None else getattr(self, "_dt", 0.033))
        self._dt = step_dt
        self._compute_joint_acceleration(raw_gt, step_dt)

        left_names = names[:left_count]
        right_names = names[left_count:]
        robot["joint_state_metadata"] = {
            "dof_names": names,
            "source_dof_indices": indices,
            "left_dof_names": left_names,
            "right_dof_names": right_names,
            "position_unit": "rad",
            "velocity_unit": "rad/s",
            "acceleration_unit": "rad/s^2",
            "effort_unit": "N*m",
            "effort_semantics": "PhysX measured joint effort",
            "first_acceleration_sample": "unavailable_no_previous_velocity",
        }

        physics["physics_dt_s"] = step_dt
        physics["arm_dof_names"] = names
        physics["arm_dof_indices"] = indices
        physics["joint_position_unit"] = "rad"
        physics["joint_velocity_unit"] = "rad/s"
        physics["joint_acceleration_unit"] = "rad/s^2"
        physics["joint_effort_unit"] = "N*m"

        indexed_limits = effort_limits_by_index or physics.get(
            "joint_torque_limits_nm_by_index", {}
        )
        aligned_limits = []
        for source_index in indices:
            record = indexed_limits.get(str(source_index)) if isinstance(indexed_limits, dict) else None
            value = record.get("limit_nm") if isinstance(record, dict) else record
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = None
            aligned_limits.append(value if value is not None and value > 0.0 else None)
        physics["joint_effort_limits_nm"] = aligned_limits

    def _compute_drop_detection(self, raw_gt: Dict, dt: float) -> None:
        """Build per-object fallback drop records from per-arm grasp states.

        The workflow later replaces these records with PhysX-contact-backed
        evidence when it is available.  Keeping the fallback per object avoids
        collapsing a dual-arm episode into a single boolean/height.
        """
        outcome = raw_gt.get("outcome_gt", {})
        gripper = raw_gt.get("gripper_gt", {})
        obj_state = raw_gt.get("object_state", {})
        obj_poses = obj_state.get("object_pose_gt", {})
        states = gripper.get("grasp_state_gt")
        target_ids = raw_gt.get("episode_meta", {}).get("target_object_ids") or [
            name for name in obj_poses if name.startswith("pick_object")
        ]
        event_by_object = {}
        height_by_object = {}

        for target in target_ids:
            arm = "right" if "right" in target.lower() else "left"
            trans_list = (obj_poses.get(target) or {}).get("translation_per_step")
            if not isinstance(states, list) or not isinstance(trans_list, list):
                event_by_object[target] = None
                height_by_object[target] = {
                    "arm": arm, "drop_height_m": None,
                    "status": "impact_not_observed",
                }
                continue

            n = min(len(states), len(trans_list))
            arm_states = [
                frame.get(arm) if isinstance(frame, dict) else frame
                for frame in states[:n]
            ]
            scale = _infer_position_scale_to_m(trans_list)
            z_values = [
                float(t[2]) * scale
                if isinstance(t, (list, tuple)) and len(t) >= 3 else None
                for t in trans_list[:n]
            ]
            dropped = [i for i, state in enumerate(arm_states) if state == "dropped"]
            if not dropped:
                event_by_object[target] = (
                    False if any(s in ("grasped", "slipping") for s in arm_states)
                    else None
                )
                height_by_object[target] = {
                    "arm": arm, "drop_height_m": None,
                    "status": "impact_not_observed",
                }
                continue

            first = dropped[0]
            before = [z for z in z_values[:first + 1] if z is not None]
            after = [z_values[i] for i in dropped if z_values[i] is not None]
            event_by_object[target] = True
            height_by_object[target] = {
                "arm": arm,
                "drop_height_m": (
                    max(0.0, max(before) - min(after))
                    if before and after else None
                ),
                "status": "impact_not_observed",
            }

        outcome["drop_event_gt"] = event_by_object or None
        outcome["drop_height_gt"] = height_by_object or None

    def _compute_slip_distance(self, raw_gt: Dict) -> None:
        """S-GRASP-002: Compute slip distance in the EE local frame.

        The previous implementation measured the object-EE Euclidean distance in
        world coordinates. That over-counted EE translation as "slip". Here we
        transform the object position into the EE frame and measure how much the
        object moves relative to the gripper itself.
        """
        gripper = raw_gt.get("gripper_gt", {})
        obj_state = raw_gt.get("object_state", {})
        robot = raw_gt.get("robot_state", {})

        obj_poses = obj_state.get("object_pose_gt", {})
        def _max_deviation(points: List[List[float]]) -> float:
            base = points[0]
            max_dev = 0.0
            for p in points[1:]:
                dx = p[0] - base[0]
                dy = p[1] - base[1]
                dz = p[2] - base[2]
                max_dev = max(max_dev, math.sqrt(dx * dx + dy * dy + dz * dz))
            return max_dev

        target_ids = raw_gt.get("episode_meta", {}).get("target_object_ids") or [
            name for name in obj_poses if name.startswith("pick_object")
        ]
        collision_frames = (
            raw_gt.get("collision_gt", {}).get("collision_pair_gt") or []
        )

        def _has_robot_contact(
            target: str, arm: str, object_index: int, object_count: int
        ) -> bool:
            if not collision_frames:
                return False
            frame_index = min(
                int(round(
                    object_index * (len(collision_frames) - 1)
                    / max(object_count - 1, 1)
                )),
                len(collision_frames) - 1,
            )
            frame = collision_frames[frame_index]
            entries = frame if isinstance(frame, list) else [frame]
            object_label = f"object/{target}"
            arm_token = "/fl/" if arm == "left" else "/fr/"
            gripper_links = ("/link6", "/link7", "/link8")

            def _is_arm_gripper(body):
                body = str(body)
                return (
                    body.startswith("robot/")
                    and arm_token in body
                    and body.endswith(gripper_links)
                )

            return any(
                isinstance(entry, dict)
                and (
                    (
                        _is_arm_gripper(entry.get("bodyA", ""))
                        and str(entry.get("bodyB", "")) == object_label
                    )
                    or (
                        _is_arm_gripper(entry.get("bodyB", ""))
                        and str(entry.get("bodyA", "")) == object_label
                    )
                )
                for entry in entries
            )

        result = {}
        max_widths = raw_gt.get("episode_meta", {}).get(
            "physics_config", {}
        ).get("gripper_max_width_m_by_arm", {})
        for target in target_ids:
            arm = "right" if "right" in target.lower() else "left"
            ee_poses = _ee_pose_series(robot, arm)
            obj_trans = (obj_poses.get(target) or {}).get("translation_per_step")
            widths = gripper.get(f"gripper_width_{arm}")
            record = {
                "arm": arm,
                "slip_distance_m": None,
                "method": "maximum displacement in EE local frame during closed-gripper window",
            }
            if not isinstance(ee_poses, list) or not isinstance(obj_trans, list):
                result[target] = record
                continue
            try:
                max_width = float(max_widths.get(arm))
            except (TypeError, ValueError):
                max_width = 0.1
            close_threshold = max_width - max(0.005, 0.05 * max_width)
            n = min(len(obj_trans), len(widths)) if isinstance(widths, list) else len(obj_trans)
            grasp_states = gripper.get("grasp_state_gt")
            closed_contact = [
                i for i in range(n)
                if isinstance(widths, list)
                and (_safe_scalar(widths[i]) is not None)
                and _safe_scalar(widths[i]) < close_threshold
                and _has_robot_contact(target, arm, i, n)
                and (
                    not isinstance(grasp_states, list)
                    or i >= len(grasp_states)
                    or not isinstance(grasp_states[i], dict)
                    or grasp_states[i].get(arm) in ("grasped", "slipping")
                )
            ]
            scale = _infer_position_scale_to_m(obj_trans, closed_contact)
            rel_positions = []
            for i in range(n):
                ee = _sample_pose_xyzw(ee_poses, i, n)
                obj = obj_trans[i]
                if ee is None or obj is None or len(ee) < 7 or len(obj) < 3:
                    rel_positions.append(None)
                    continue
                rotation = _quat_to_rotmat_xyzw([float(v) for v in ee[3:7]])
                if rotation is None:
                    rel_positions.append(None)
                    continue
                rel_world = [
                    float(obj[j]) * scale - float(ee[j]) for j in range(3)
                ]
                rel_positions.append(_world_to_local(rotation, rel_world))
            # Treat separate contact intervals independently.  Motion after
            # release/drop must not be counted as slip inside the gripper.
            runs, run = [], []
            for index in closed_contact:
                if run and index != run[-1] + 1:
                    runs.append(run)
                    run = []
                run.append(index)
            if run:
                runs.append(run)
            run_slips = []
            for indices in runs:
                positions = [
                    rel_positions[index] for index in indices
                    if index < len(rel_positions)
                    and rel_positions[index] is not None
                ]
                if len(positions) >= 2:
                    run_slips.append(_max_deviation(positions))
            record["slip_distance_m"] = max(run_slips, default=0.0)
            record["contact_window_count"] = len(runs)
            record["closed_width_threshold_m"] = close_threshold
            result[target] = record
        gripper["slip_distance_gt"] = result or None

    def _compute_grasp_state(self, raw_gt: Dict) -> None:
        """S-GRASP-003: Infer grasp state with a small state machine.

        The state is based on gripper width plus object motion in the EE local
        frame. This avoids classifying by raw EE-to-object-center distance, which
        is brittle for large objects and mixed sampling rates.
        """
        gripper = raw_gt.get("gripper_gt", {})
        obj_state = raw_gt.get("object_state", {})
        robot = raw_gt.get("robot_state", {})

        obj_poses = obj_state.get("object_pose_gt", {})
        grasp_confirm_frames = 3
        slip_threshold = 0.03       # m relative motion after grasp is established
        drop_rel_threshold = 0.10   # m relative motion after grasp
        drop_z_threshold = 0.05     # m downward object motion after grasp

        target_ids = raw_gt.get("episode_meta", {}).get("target_object_ids") or [
            name for name in obj_poses if name.startswith("pick_object")
        ]
        collision_frames = (
            raw_gt.get("collision_gt", {}).get("collision_pair_gt") or []
        )

        def _has_robot_contact(
            target: str, arm: str, object_index: int, object_count: int
        ) -> bool:
            if not collision_frames:
                return False
            frame_index = min(
                int(round(
                    object_index * (len(collision_frames) - 1)
                    / max(object_count - 1, 1)
                )),
                len(collision_frames) - 1,
            )
            frame = collision_frames[frame_index]
            entries = frame if isinstance(frame, list) else [frame]
            object_label = f"object/{target}"
            arm_token = "/fl/" if arm == "left" else "/fr/"
            gripper_links = ("/link6", "/link7", "/link8")

            def _is_arm_gripper(body):
                body = str(body)
                return (
                    body.startswith("robot/")
                    and arm_token in body
                    and body.endswith(gripper_links)
                )

            return any(
                isinstance(entry, dict)
                and (
                    (
                        _is_arm_gripper(entry.get("bodyA", ""))
                        and str(entry.get("bodyB", "")) == object_label
                    )
                    or (
                        _is_arm_gripper(entry.get("bodyB", ""))
                        and str(entry.get("bodyA", "")) == object_label
                    )
                )
                for entry in entries
            )

        states_by_arm = {}
        max_frames = 0
        max_widths = raw_gt.get("episode_meta", {}).get(
            "physics_config", {}
        ).get("gripper_max_width_m_by_arm", {})
        for target in target_ids:
            arm = "right" if "right" in target.lower() else "left"
            try:
                max_width = float(max_widths.get(arm))
            except (TypeError, ValueError):
                max_width = 0.1
            close_threshold = max_width - max(0.005, 0.05 * max_width)
            open_threshold = min(max_width, close_threshold + 0.002)
            widths_raw = gripper.get(f"gripper_width_{arm}")
            ee_poses = _ee_pose_series(robot, arm)
            obj_trans = (obj_poses.get(target) or {}).get("translation_per_step")
            if not all(isinstance(values, list) for values in (widths_raw, ee_poses, obj_trans)):
                continue
            n = min(len(obj_trans), len(widths_raw))
            max_frames = max(max_frames, n)
            scale = _infer_position_scale_to_m(obj_trans)
            widths = [_safe_scalar(value) for value in widths_raw[:n]]
            rel_positions, obj_positions_m = [], []
            for i in range(n):
                ee = _sample_pose_xyzw(ee_poses, i, n)
                obj = obj_trans[i]
                if ee is None or obj is None or len(ee) < 7 or len(obj) < 3:
                    rel_positions.append(None)
                    obj_positions_m.append(None)
                    continue
                obj_m = [float(obj[j]) * scale for j in range(3)]
                rotation = _quat_to_rotmat_xyzw([float(v) for v in ee[3:7]])
                obj_positions_m.append(obj_m)
                rel_positions.append(
                    _world_to_local(
                        rotation,
                        [obj_m[j] - float(ee[j]) for j in range(3)],
                    ) if rotation is not None else None
                )

            arm_states = []
            state, closed_run = "not_grasped", 0
            grasp_rel_ref = grasp_obj_z_ref = None
            for i in range(n):
                width, rel, obj_m = widths[i], rel_positions[i], obj_positions_m[i]
                if width is None or rel is None or obj_m is None:
                    arm_states.append(None)
                    continue
                closed, opened = width < close_threshold, width >= open_threshold
                in_contact = _has_robot_contact(target, arm, i, n)
                closed_run = closed_run + 1 if closed and in_contact else 0
                if state == "not_grasped":
                    if closed and in_contact and closed_run >= grasp_confirm_frames:
                        state = "grasped"
                        ref = max(0, i - grasp_confirm_frames + 1)
                        grasp_rel_ref = rel_positions[ref] or rel
                        grasp_obj_z_ref = (obj_positions_m[ref] or obj_m)[2]
                elif state in ("grasped", "slipping"):
                    grasp_rel_ref = grasp_rel_ref or rel
                    grasp_obj_z_ref = (
                        obj_m[2] if grasp_obj_z_ref is None else grasp_obj_z_ref
                    )
                    delta = math.sqrt(sum(
                        (rel[j] - grasp_rel_ref[j]) ** 2 for j in range(3)
                    ))
                    z_drop = grasp_obj_z_ref - obj_m[2]
                    if (
                        closed
                        and not in_contact
                        and (delta > drop_rel_threshold or z_drop > drop_z_threshold)
                    ):
                        state = "dropped"
                    elif closed and in_contact and delta > slip_threshold:
                        state = "slipping"
                    elif opened:
                        state = "not_grasped"
                        grasp_rel_ref = grasp_obj_z_ref = None
                arm_states.append(state)
            states_by_arm[arm] = arm_states

        grasp_states = []
        for frame in range(max_frames):
            grasp_states.append({
                arm: values[frame] if frame < len(values) else None
                for arm, values in states_by_arm.items()
            })
        gripper["grasp_state_gt"] = grasp_states or None

    def _compute_damage_state(self, raw_gt: Dict) -> None:
        """S-OUT-004: Infer damage state from drop/collision/force data.

        Uses fragility-dependent proxy rules when no material damage model is
        available. Raw distances are meters.
        """
        outcome = raw_gt.get("outcome_gt", {})
        meta = raw_gt.get("episode_meta", {})
        coll = raw_gt.get("collision_gt", {})
        gripper = raw_gt.get("gripper_gt", {})

        # If damage_state is already set by a real damage model, keep it.
        existing = outcome.get("damage_state_gt", "none")
        if existing and existing not in ("none", "unknown"):
            return

        drop_height = outcome.get("drop_height_gt")
        drop_event = outcome.get("drop_event_gt", False)
        if isinstance(drop_event, dict):
            values = list(drop_event.values())
            drop_event = (
                True if any(value is True for value in values)
                else False if values and all(value is False for value in values)
                else None
            )
        if isinstance(drop_height, dict):
            height_values = []
            for record in drop_height.values():
                value = (
                    record.get("drop_height_m")
                    if isinstance(record, dict) else record
                )
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    height_values.append(float(value))
            drop_height = max(height_values) if height_values else None
        fragility_value = meta.get("object_fragility_class")
        fragility = str(fragility_value).lower() if fragility_value is not None else None

        target_ids = meta.get("target_object_ids") or []
        target_labels = {f"object/{name}" for name in target_ids}
        impulse_peak = 0.0
        impulse_samples = 0
        for frame in coll.get("contact_impulse_gt") or []:
            entries = frame if isinstance(frame, list) else [frame]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if target_labels and not (
                    str(entry.get("bodyA", "")) in target_labels
                    or str(entry.get("bodyB", "")) in target_labels
                ):
                    continue
                value = entry.get("impulse_ns")
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    impulse_peak = max(impulse_peak, abs(float(value)))
                    impulse_samples += 1

        gripper_force_peak = 0.0
        for frame in gripper.get("gripper_object_contact_force_gt") or []:
            if not isinstance(frame, dict):
                continue
            for arm in ("left", "right"):
                value = frame.get(arm)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    gripper_force_peak = max(gripper_force_peak, abs(float(value)))

        profiles = {
            "extreme": {"minor_drop": 0.03, "broken_drop": 0.10, "minor_impulse": 1.0, "broken_impulse": 5.0, "force_damage": 25.0},
            "high": {"minor_drop": 0.05, "broken_drop": 0.20, "minor_impulse": 3.0, "broken_impulse": 10.0, "force_damage": 50.0},
            "medium": {"minor_drop": 0.15, "broken_drop": 0.50, "minor_impulse": 8.0, "broken_impulse": 25.0, "force_damage": 100.0},
            "low": {"minor_drop": 0.30, "broken_drop": 1.00, "minor_impulse": 20.0, "broken_impulse": 60.0, "force_damage": 200.0},
            "none": {"minor_drop": 0.60, "broken_drop": 2.00, "minor_impulse": 50.0, "broken_impulse": 120.0, "force_damage": 400.0},
        }
        if fragility not in profiles:
            outcome["damage_state_gt"] = "unknown"
            outcome["damage_evidence_gt"] = {
                "method": "rule_based_proxy",
                "status": "not_evaluable",
                "reason": "object_fragility_class or material damage model is missing",
                "drop_event": drop_event,
                "drop_height_m": drop_height,
                "object_impact_impulse_peak_ns": impulse_peak,
                "object_impact_impulse_samples": impulse_samples,
                "gripper_force_peak_n": gripper_force_peak,
            }
            return
        profile = profiles[fragility]

        severity_rank = {"none": 0, "minor": 1, "functional_damage": 2, "broken": 3}
        severity = "none"

        def _set_severity(candidate: str) -> None:
            nonlocal severity
            if severity_rank[candidate] > severity_rank[severity]:
                severity = candidate

        if drop_event and drop_height is not None:
            if drop_height >= profile["broken_drop"]:
                _set_severity("broken")
            elif drop_height >= profile["minor_drop"]:
                _set_severity("minor")

        if impulse_peak >= profile["broken_impulse"]:
            _set_severity("broken")
        elif impulse_peak >= profile["minor_impulse"]:
            _set_severity("minor")

        if gripper_force_peak >= profile["force_damage"]:
            _set_severity("functional_damage")

        outcome["damage_state_gt"] = severity
        outcome["damage_evidence_gt"] = {
            "method": "rule_based_proxy",
            "status": "estimated",
            "fragility_class": fragility,
            "thresholds": profile,
            "drop_event": drop_event,
            "drop_height_m": drop_height,
            "object_impact_impulse_peak_ns": impulse_peak,
            "object_impact_impulse_samples": impulse_samples,
            "gripper_force_peak_n": gripper_force_peak,
            "note": "Not a simulated fracture/material damage measurement",
        }

    def _rebuild_ee_poses_from_link_pose_gt(self, raw_gt: Dict) -> None:
        """Rebuild per-step left/right EE poses from complete PhysX link poses.

        LMDB T_base_ee_fl/fr can be skill/segment scoped and shorter than the
        simulation timeline.  link_pose_gt is collected every PhysX step, so use
        the configured EE link paths there to make ee_pose_gt and
        ee_pose_right_gt full-length time series.
        """
        robot = raw_gt.get("robot_state", {})
        link_pose_gt = robot.get("link_pose_gt")
        if not isinstance(link_pose_gt, list) or not link_pose_gt:
            return

        def _task_robot_configs():
            cfg = getattr(self, "task_cfg", None) or {}
            robots = cfg.get("robots", []) if isinstance(cfg, dict) else []
            if isinstance(robots, list):
                return robots
            return []

        def _ee_suffixes(side: str):
            # side: "fl" or "fr"
            suffixes = []
            attr = f"{side}_ee_path"
            for robot_cfg in _task_robot_configs():
                ee_path = robot_cfg.get(attr) if isinstance(robot_cfg, dict) else None
                if ee_path:
                    suffixes.append(str(ee_path))
            # split_aloha/lift2 default EE link is link6. link8 is kept only as a
            # fallback for gripper/collision-only assets or other robot configs.
            suffixes.extend([f"/{side}/link6", f"/{side}/link8", f"{side}/link6", f"{side}/link8"])
            return suffixes

        def _find_pose_in_frame(frame, side: str):
            if not isinstance(frame, dict):
                return None
            suffixes = _ee_suffixes(side)
            candidate_items = []
            for robot_name, links in frame.items():
                if isinstance(links, dict):
                    for link_name, pose in links.items():
                        candidate_items.append((robot_name, link_name, pose))
            for suffix in suffixes:
                normalized_suffix = suffix.strip("/")
                for _, link_name, pose in candidate_items:
                    link_norm = str(link_name).strip("/")
                    if link_norm == normalized_suffix or link_norm.endswith("/" + normalized_suffix):
                        if isinstance(pose, (list, tuple)) and len(pose) >= 7 and pose[0] is not None:
                            return [float(v) for v in pose[:7]]
            return None

        left_poses = []
        right_poses = []
        left_found = 0
        right_found = 0
        for frame in link_pose_gt:
            left = _find_pose_in_frame(frame, "fl")
            right = _find_pose_in_frame(frame, "fr")
            left_poses.append(left)
            right_poses.append(right)
            if left is not None:
                left_found += 1
            if right is not None:
                right_found += 1

        if left_found > 0:
            robot["ee_pose_gt"] = left_poses
        if right_found > 0:
            robot["ee_pose_right_gt"] = right_poses

    def _compute_human_distances_from_obstacles(self, raw_gt: Dict) -> None:
        """S-DIST-001/002/003: Compute human distances using obstacle surrogates.

        Uses obstacle poses from environment_state.obstacle_pose_gt as human
        body surrogates and computes:
        - S-DIST-001: robot_human_distance_matrix_gt (EE-to-obstacle per step)
        - S-DIST-002: ee_human_distance_gt (EE-to-nearest-obstacle per step)
        - S-DIST-003: object_human_distance_gt (object-to-nearest-obstacle per step)
        """
        env = raw_gt.get("environment_state", {})
        robot = raw_gt.get("robot_state", {})
        obj_state = raw_gt.get("object_state", {})
        dist = raw_gt.get("distance_gt", {})

        obstacle_poses = env.get("obstacle_pose_gt", {})
        if not obstacle_poses:
            return

        # Collect obstacle translation trajectories
        obs_trajs: Dict[str, List[List[float]]] = {}
        for name, pose_data in obstacle_poses.items():
            trans = pose_data.get("translation")
            if trans and len(trans) > 0:
                positions = []
                for t in trans:
                    if t is not None and hasattr(t, '__len__') and len(t) >= 3:
                        positions.append([float(t[0]), float(t[1]), float(t[2])])
                if positions:
                    obs_trajs[name] = positions

        if not obs_trajs:
            return

        # ── S-DIST-002: ee_human_distance_gt ──
        # Compute distances from left/right EE to each obstacle
        ee_left = _ee_pose_series(robot, "left")
        ee_right = _ee_pose_series(robot, "right")
        if (ee_left or ee_right) and obs_trajs:
            n_steps = max(len(ee_left) if ee_left else 0,
                         len(ee_right) if ee_right else 0)
            ee_dists = []
            for i in range(n_steps):
                step_dist = {}
                # Left arm EE
                if ee_left and i < len(ee_left) and ee_left[i] and len(ee_left[i]) >= 3:
                    for obs_name, obs_traj in obs_trajs.items():
                        idx = min(i, len(obs_traj) - 1)
                        obs = obs_traj[idx]
                        dx = ee_left[i][0] - obs[0]
                        dy = ee_left[i][1] - obs[1]
                        dz = ee_left[i][2] - obs[2]
                        d = math.sqrt(dx * dx + dy * dy + dz * dz)
                        step_dist[f"left_{obs_name}"] = d
                # Right arm EE
                if ee_right and i < len(ee_right) and ee_right[i] and len(ee_right[i]) >= 3:
                    for obs_name, obs_traj in obs_trajs.items():
                        idx = min(i, len(obs_traj) - 1)
                        obs = obs_traj[idx]
                        dx = ee_right[i][0] - obs[0]
                        dy = ee_right[i][1] - obs[1]
                        dz = ee_right[i][2] - obs[2]
                        d = math.sqrt(dx * dx + dy * dy + dz * dz)
                        step_dist[f"right_{obs_name}"] = d
                ee_dists.append(step_dist if step_dist else None)
            dist["ee_human_distance_gt"] = ee_dists

        # ── S-DIST-003: object_human_distance_gt ──
        # Compute distances from ALL pick objects to each obstacle
        obj_poses = obj_state.get("object_pose_gt", {})
        if obj_poses and obs_trajs:
            # Find all pick objects
            pick_objects = {}
            for name in ["pick_object_left", "pick_object_right", "pick_object"]:
                if name in obj_poses:
                    pick_objects[name] = obj_poses[name]

            if pick_objects:
                n_steps = max(len(v.get("translation_per_step", []))
                             for v in pick_objects.values())
                obj_dists = []
                for i in range(n_steps):
                    step_dist = {}
                    for obj_name, obj_data in pick_objects.items():
                        obj_trans = obj_data.get("translation_per_step", [])
                        if i < len(obj_trans):
                            obj = obj_trans[i]
                            if obj and len(obj) >= 3:
                                for obs_name, obs_traj in obs_trajs.items():
                                    idx = min(i, len(obs_traj) - 1)
                                    obs = obs_traj[idx]
                                    dx = obj[0] - obs[0]
                                    dy = obj[1] - obs[1]
                                    dz = obj[2] - obs[2]
                                    d = math.sqrt(dx * dx + dy * dy + dz * dz)
                                    step_dist[f"{obj_name}_{obs_name}"] = d
                    obj_dists.append(step_dist if step_dist else None)
                dist["object_human_distance_gt"] = obj_dists

        # ── S-DIST-001: robot_human_distance_matrix_gt ──
        # Compute a per-step robot-link × human-surrogate distance matrix.
        # Current PhysX collector writes link_pose_gt as:
        #   [{robot_name: {link_name: [x, y, z, qx, qy, qz, qw]}}]
        # Older outputs may be flat:
        #   [{link_name: [x, y, z, ...]}]
        # Support both shapes and keep all link-level distances in meters.
        all_link_poses = robot.get("link_pose_gt")
        if all_link_poses and obs_trajs:
            matrix = []

            def _iter_link_positions(step_links):
                if not isinstance(step_links, dict):
                    return
                for name, value in step_links.items():
                    # Nested shape: robot_name -> {link_name: pose}
                    if isinstance(value, dict):
                        robot_name = str(name)
                        for link_name, link_pose in value.items():
                            if (link_pose is not None and hasattr(link_pose, "__len__")
                                    and len(link_pose) >= 3 and link_pose[0] is not None):
                                yield f"{robot_name}/{link_name}", link_pose
                    # Flat shape: link_name -> pose
                    elif (value is not None and hasattr(value, "__len__")
                          and len(value) >= 3 and value[0] is not None):
                        yield str(name), value

            for i, step_links in enumerate(all_link_poses):
                row = {}
                link_items = list(_iter_link_positions(step_links))
                if not link_items:
                    matrix.append(None)
                    continue

                for obs_name, obs_traj in obs_trajs.items():
                    idx = min(i, len(obs_traj) - 1)
                    obs = obs_traj[idx]
                    obs_row = {}
                    for link_name, link_pose in link_items:
                        dx = float(link_pose[0]) - obs[0]
                        dy = float(link_pose[1]) - obs[1]
                        dz = float(link_pose[2]) - obs[2]
                        obs_row[link_name] = math.sqrt(dx * dx + dy * dy + dz * dz)
                    row[obs_name] = obs_row

                matrix.append(row if row else None)

            if any(step is not None for step in matrix):
                dist["robot_human_distance_matrix_gt"] = matrix

    def _compute_joint_torque_from_pd(self, raw_gt: Dict, dt: float) -> None:
        """S-ROBOT-004: Estimate joint torques from PD control law.

        For position-controlled robots, torques are computed internally by the
        PD controller but not exposed through the PhysX API. We estimate them:
            torque = Kp * (target - actual) - Kd * velocity

        Uses the planned trajectory as target positions and actual joint
        positions + velocities from the simulation.
        """
        robot = raw_gt.get("robot_state", {})
        planner = raw_gt.get("planner_log", {})

        # Only compute if torque data is all zeros or None
        existing_torque = robot.get("joint_torque_gt")
        if existing_torque:
            has_nonzero = False
            for step in existing_torque[:10]:
                if step and any(abs(float(t)) > 1e-6 for t in step if t is not None):
                    has_nonzero = True
                    break
            if has_nonzero:
                return  # Already has real torque data

        q_actual = robot.get("joint_position_q_gt")
        dq = robot.get("joint_velocity_dq_gt")
        executed = planner.get("executed_trajectory")

        if q_actual is None or dq is None:
            return

        # Get target positions from executed trajectory (same as commanded)
        q_target = None
        if executed:
            for arm_traj in executed:
                if isinstance(arm_traj, dict) and arm_traj.get("arm") == "left":
                    trajectory = arm_traj.get("trajectory", [])
                    q_target = []
                    for step in trajectory:
                        if isinstance(step, dict):
                            q_target.append(step.get("joint_positions"))
                    break

        if q_target is None:
            # Fallback: use actual positions as target (no tracking error)
            q_target = q_actual

        # PD gains (typical for split_aloha / small manipulator)
        n_dof = len(q_actual[0]) if q_actual and q_actual[0] else 6
        kp = [800.0] * n_dof  # Proportional gain
        kd = [40.0] * n_dof   # Derivative gain

        torques = []
        n = min(len(q_actual), len(dq), len(q_target))
        for i in range(n):
            q_a = q_actual[i]
            q_t = q_target[min(i, len(q_target) - 1)]
            dq_i = dq[i]

            if q_a is None or dq_i is None:
                torques.append([None] * n_dof)
                continue

            step_torques = []
            for j in range(min(len(q_a), len(dq_i), n_dof)):
                q_target_j = q_t[j] if q_t and j < len(q_t) and q_t[j] is not None else q_a[j]
                if q_target_j is not None and q_a[j] is not None and dq_i[j] is not None:
                    tau = kp[j] * (q_target_j - q_a[j]) - kd[j] * dq_i[j]
                    step_torques.append(round(tau, 4))
                else:
                    step_torques.append(None)
            torques.append(step_torques)

        if torques:
            robot["joint_torque_gt"] = torques

    def _compute_executed_trajectory(self, raw_gt: Dict) -> None:
        """S-PLAN-002: Build executed trajectory from executed robot states.

        Each arm includes joint positions plus the corresponding EE pose as
        [x, y, z, qx, qy, qz, qw]. Right arm uses ee_pose_right_gt; it should not
        be silently omitted or filled with zeros when data is available.
        """
        planner = raw_gt.get("planner_log", {})
        robot = raw_gt.get("robot_state", {})

        q_all = robot.get("joint_position_q_gt")
        q_right_legacy = robot.get("joint_position_q_right_gt")
        joint_meta = robot.get("joint_state_metadata", {})
        left_names = list(joint_meta.get("left_dof_names", []) or [])
        right_names = list(joint_meta.get("right_dof_names", []) or [])
        left_count = len(left_names)
        if q_all and left_count and len(q_all[0]) >= left_count:
            q_left = [
                list(row[:left_count]) if row is not None else None
                for row in q_all
            ]
            q_right = [
                list(row[left_count:left_count + len(right_names)]) if row is not None else None
                for row in q_all
            ] if right_names else None
        else:
            q_left = q_all
            q_right = q_right_legacy
            if not left_names and q_left and q_left[0]:
                left_names = [f"left_joint_{i + 1}" for i in range(len(q_left[0]))]
            if not right_names and q_right and q_right[0]:
                right_names = [f"right_joint_{i + 1}" for i in range(len(q_right[0]))]
        ee_left = _ee_pose_series(robot, "left")
        ee_right = _ee_pose_series(robot, "right")

        trajectory = []
        dt = self._dt

        def _build_arm_trajectory(arm: str, joint_series, ee_series, dof_names):
            if not joint_series and not ee_series:
                return None

            n_steps = max(len(joint_series) if joint_series else 0,
                          len(ee_series) if ee_series else 0)
            arm_traj = []
            for i in range(n_steps):
                q = joint_series[i] if joint_series and i < len(joint_series) else None
                entry = {
                    "t": round(i * dt, 4),
                    "joint_positions": [round(v, 6) for v in q] if q else None,
                }
                if ee_series and i < len(ee_series) and ee_series[i]:
                    ee = ee_series[i]
                    if len(ee) >= 7:
                        entry["ee_pose"] = [round(float(v), 6) for v in ee[:7]]
                        entry["ee_position"] = entry["ee_pose"][:3]
                    elif len(ee) >= 3:
                        entry["ee_pose"] = None
                        entry["ee_position"] = [round(float(v), 6) for v in ee[:3]]
                    else:
                        entry["ee_pose"] = None
                        entry["ee_position"] = None
                else:
                    entry["ee_pose"] = None
                    entry["ee_position"] = None
                arm_traj.append(entry)

            return {
                "arm": arm,
                "dof_names": dof_names,
                "joint_position_unit": "rad",
                "ee_pose_frame": "world",
                "ee_pose_format": "[x,y,z,qx,qy,qz,qw]",
                "quaternion_order": "xyzw",
                "n_steps": len(arm_traj),
                "trajectory": arm_traj,
            }

        left = _build_arm_trajectory("left", q_left, ee_left, left_names)
        if left:
            trajectory.append(left)

        right = _build_arm_trajectory("right", q_right, ee_right, right_names)
        if right:
            trajectory.append(right)

        planner["executed_trajectory"] = trajectory if trajectory else None

    # ── Utilities ────────────────────────────────────────────────────────────

    def _convert_to_float_lists(self, data) -> Optional[List]:
        """Convert a list of numpy arrays to a list of plain float lists.

        Handles:
        - list of numpy arrays → list of [float, float, float]
        - list of strings (numpy repr) → list of [float, float, float]
        - list of plain lists → unchanged
        """
        if data is None:
            return None
        try:
            import numpy as np
            result = []
            for item in data:
                if isinstance(item, np.ndarray):
                    result.append([float(x) for x in item.tolist()])
                elif isinstance(item, (list, tuple)):
                    result.append([float(x) for x in item])
                elif isinstance(item, str):
                    # Parse numpy array repr like "[ 0.33  0.59 -0.59]"
                    cleaned = item.strip("[]").split()
                    result.append([float(x) for x in cleaned])
                else:
                    result.append(item)
            return result
        except Exception:
            return _safe_list(data)

    def _convert_to_world_poses(self, T_base_ee, T_world_base) -> Optional[List[List[float]]]:
        """Convert base-frame transforms to world-frame [x,y,z,qx,qy,qz,qw] poses."""
        if T_base_ee is None:
            return None

        import numpy as np

        poses = []
        for i in range(len(T_base_ee)):
            mat = T_base_ee[i]
            if not isinstance(mat, np.ndarray) or mat.shape != (4, 4):
                continue

            if T_world_base is not None and i < len(T_world_base):
                T_wb = T_world_base[i]
                if isinstance(T_wb, np.ndarray) and T_wb.shape == (4, 4):
                    ee_world = T_wb[:3, :3] @ mat[:3, 3] + T_wb[:3, 3]
                    quat = _quat_from_transform(T_wb @ mat)
                    if quat:
                        poses.append([float(ee_world[0]), float(ee_world[1]), float(ee_world[2])] + quat)
                        continue

            pose = _transform_to_pose(mat)
            if pose:
                poses.append(pose)

        return poses if poses else None

    def _safe_list_list(self, data) -> Optional[List[List[float]]]:
        """Convert nested array to list of lists."""
        if data is None:
            return None
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                return data.tolist()
        except ImportError:
            pass
        if isinstance(data, list):
            result = []
            for item in data:
                if hasattr(item, 'tolist'):
                    result.append(item.tolist())
                elif isinstance(item, (list, tuple)):
                    result.append([float(x) for x in item])
                else:
                    result.append(float(item))
            return result
        return None


def extract_and_save(lmdb_dir: str, output_path: str) -> str:
    """Extract Sim_Raw_GT from LMDB and save to JSON.

    Parameters
    ----------
    lmdb_dir : str
        Path to episode directory (with lmdb/ and meta_info.pkl).
    output_path : str
        Output JSON file path.

    Returns
    -------
    str
        Path to saved JSON file.
    """
    extractor = SimRawGTExtractor()
    raw_gt = extractor.extract_from_lmdb(lmdb_dir)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(raw_gt, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Sim_Raw_GT saved to: %s", output_path)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract Sim_Raw_GT from LMDB")
    parser.add_argument("lmdb_dir", help="Path to episode directory")
    parser.add_argument("-o", "--output", help="Output JSON path")
    args = parser.parse_args()

    output = args.output or os.path.join(args.lmdb_dir, "sim_raw_gt.json")
    extract_and_save(args.lmdb_dir, output)
    print(f"Saved to: {output}")
