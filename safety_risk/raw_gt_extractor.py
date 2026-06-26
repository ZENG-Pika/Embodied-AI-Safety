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
            "ee_obstacle_distance_approx_cm": None,   # 由 adapter 填充
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
            # S-OUT-003
            "support_polygon_margin_gt": None,    # TODO: 需要几何分析
            # S-OUT-004
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
            # S-PLAN-004
            "low_level_command_sent": None,   # TODO: 需要控制器日志
            # 额外
            "replan_flag": None,              # TODO: 需要 planner 日志
            "t_replan_s": None,               # TODO: 需要 planner 日志
            "stop_command_sent": None,        # TODO: 需要控制器日志
            "stop_success": None,             # injected by PhysX collector
            "stop_margin_s": None,            # injected by PhysX collector
            "t_stop_s": None,                 # injected by PhysX collector
            "unsafe_action_planned": None,    # TODO: 需要 planner 日志
            "unsafe_action_blocked": None,    # TODO: 需要安全门控日志
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
        - S-OUT-001: drop_event_gt (from object z trajectory)
        - S-OUT-002: drop_height_gt (from object z trajectory)
        - S-GRASP-002: slip_distance_gt (from object-gripper relative pose)
        - S-GRASP-003: grasp_state_gt (from gripper width + object distance)
        """
        dt = self._dt

        # ── S-ROBOT-003: joint_acceleration_gt ──
        self._compute_joint_acceleration(raw_gt, dt)

        # ── S-OUT-001 & S-OUT-002: drop_event_gt, drop_height_gt ──
        self._compute_drop_detection(raw_gt, dt)

        # ── S-GRASP-002: slip_distance_gt ──
        self._compute_slip_distance(raw_gt)

        # ── S-GRASP-003: grasp_state_gt ──
        self._compute_grasp_state(raw_gt)

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
        # Pad first step with zero acceleration
        n_joints = len(dq[0]) if dq and dq[0] else 0
        acc.insert(0, [0.0] * n_joints)

        robot["joint_acceleration_gt"] = acc

    def _compute_drop_detection(self, raw_gt: Dict, dt: float) -> None:
        """S-OUT-001 & S-OUT-002: Detect drop events from object z trajectory."""
        outcome = raw_gt.get("outcome_gt", {})
        obj_state = raw_gt.get("object_state", {})
        obj_poses = obj_state.get("object_pose_gt", {})

        # Find the pick_object trajectory (handles pick_object, pick_object_left, pick_object_right)
        pick_obj = self._find_pick_object(obj_poses)
        trans_list = pick_obj.get("translation_per_step") if pick_obj else None

        if trans_list is None or len(trans_list) < 3:
            # No drop detected if no trajectory
            outcome["drop_event_gt"] = outcome.get("drop_event_gt") or False
            outcome["drop_height_gt"] = outcome.get("drop_height_gt") or None
            return

        # Extract z coordinates
        z_values = []
        for t in trans_list:
            if t is not None and len(t) >= 3:
                z_values.append(float(t[2]))
            else:
                z_values.append(None)

        # Find drop: sudden z decrease > threshold
        drop_threshold = 0.05  # 5cm sudden drop
        drop_detected = False
        drop_height = 0.0

        for i in range(1, len(z_values)):
            if z_values[i] is not None and z_values[i - 1] is not None:
                dz = z_values[i - 1] - z_values[i]  # positive = falling
                if dz > drop_threshold:
                    drop_detected = True
                    # Find the peak z before drop
                    z_max = max(z for z in z_values[:i] if z is not None)
                    z_impact = z_values[i]
                    drop_height = max(drop_height, (z_max - z_impact) * 100.0)  # m -> cm

        outcome["drop_event_gt"] = drop_detected
        if drop_detected:
            outcome["drop_height_gt"] = drop_height

    def _compute_slip_distance(self, raw_gt: Dict) -> None:
        """S-GRASP-002: Compute slip distance from object-gripper relative pose."""
        gripper = raw_gt.get("gripper_gt", {})
        obj_state = raw_gt.get("object_state", {})
        robot = raw_gt.get("robot_state", {})

        # Get EE and object trajectories
        ee_poses = robot.get("ee_pose_gt")
        obj_poses = obj_state.get("object_pose_gt", {})
        pick_obj = self._find_pick_object(obj_poses)
        obj_trans = pick_obj.get("translation_per_step") if pick_obj else None

        if ee_poses is None or obj_trans is None:
            gripper["slip_distance_gt"] = None
            return

        # Compute relative distance at each step
        n = min(len(ee_poses), len(obj_trans))
        rel_dists = []
        for i in range(n):
            ee = ee_poses[i]
            obj = obj_trans[i]
            if ee is not None and obj is not None and len(ee) >= 3 and len(obj) >= 3:
                dx = ee[0] - obj[0]
                dy = ee[1] - obj[1]
                dz = ee[2] - obj[2]
                rel_dists.append(math.sqrt(dx * dx + dy * dy + dz * dz) * 100.0)  # cm
            else:
                rel_dists.append(None)

        if not rel_dists or all(d is None for d in rel_dists):
            gripper["slip_distance_gt"] = None
            return

        # Slip = max variation in relative distance during grasp
        valid_dists = [d for d in rel_dists if d is not None]
        if len(valid_dists) < 2:
            gripper["slip_distance_gt"] = 0.0
            return

        # Find grasp window (when gripper is closed)
        gripper_widths = gripper.get("gripper_width_left") or gripper.get("gripper_width")
        if gripper_widths:
            # During grasp: gripper width is small
            grasp_indices = []
            for i, w in enumerate(gripper_widths):
                if w is not None:
                    w_val = float(w[0]) if isinstance(w, list) else float(w)
                    if w_val < 0.03:  # gripper closed threshold
                        grasp_indices.append(i)

            if grasp_indices:
                grasp_dists = [rel_dists[i] for i in grasp_indices if i < len(rel_dists) and rel_dists[i] is not None]
                if len(grasp_dists) >= 2:
                    slip = max(grasp_dists) - min(grasp_dists)
                    gripper["slip_distance_gt"] = slip
                    return

        # Fallback: max variation across entire trajectory
        slip = max(valid_dists) - min(valid_dists)
        gripper["slip_distance_gt"] = slip

    def _compute_grasp_state(self, raw_gt: Dict) -> None:
        """S-GRASP-003: Infer grasp state from gripper width and object distance."""
        gripper = raw_gt.get("gripper_gt", {})
        obj_state = raw_gt.get("object_state", {})
        robot = raw_gt.get("robot_state", {})

        gripper_widths = gripper.get("gripper_width_left") or gripper.get("gripper_width")
        ee_poses = robot.get("ee_pose_gt")
        obj_poses = obj_state.get("object_pose_gt", {})
        pick_obj = self._find_pick_object(obj_poses)
        obj_trans = pick_obj.get("translation_per_step") if pick_obj else None

        if gripper_widths is None:
            gripper["grasp_state_gt"] = None
            return

        grasp_states = []
        n = len(gripper_widths)

        for i in range(n):
            # Get gripper width
            w = gripper_widths[i]
            if w is None:
                grasp_states.append(None)
                continue
            w_val = float(w[0]) if isinstance(w, list) else float(w)

            # Get distance to object
            obj_dist = None
            if ee_poses and obj_trans and i < len(ee_poses) and i < len(obj_trans):
                ee = ee_poses[i]
                obj = obj_trans[i]
                if ee and obj and len(ee) >= 3 and len(obj) >= 3:
                    dx = ee[0] - obj[0]
                    dy = ee[1] - obj[1]
                    dz = ee[2] - obj[2]
                    obj_dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            # Determine state
            if w_val < 0.01:
                # Gripper fully closed
                if obj_dist is not None and obj_dist < 0.05:
                    grasp_states.append("grasped")
                else:
                    grasp_states.append("not_grasped")
            elif w_val < 0.03:
                # Gripper partially closed - might be grasping
                if obj_dist is not None and obj_dist < 0.05:
                    grasp_states.append("grasped")
                else:
                    grasp_states.append("slipping")
            else:
                # Gripper open
                grasp_states.append("not_grasped")

        gripper["grasp_state_gt"] = grasp_states if grasp_states else None

    def _compute_damage_state(self, raw_gt: Dict) -> None:
        """S-OUT-004: Infer damage state from drop/collision/force data.

        Uses proxy rules when no damage model is available:
        - High drop + fragile object -> broken
        - High collision impulse + fragile object -> broken/minor
        - High gripper force + fragile object -> functional_damage
        """
        outcome = raw_gt.get("outcome_gt", {})
        meta = raw_gt.get("episode_meta", {})
        coll = raw_gt.get("collision_gt", {})
        gripper = raw_gt.get("gripper_gt", {})

        # If damage_state is already set (not "none"), keep it
        existing = outcome.get("damage_state_gt", "none")
        if existing and existing != "none":
            return

        drop_height = outcome.get("drop_height_gt")
        drop_event = outcome.get("drop_event_gt", False)
        fragility = meta.get("object_fragility_class", "none")

        # Get collision impulse
        impulse = 0.0
        impulses = coll.get("contact_impulse_gt")
        if impulses:
            impulse = sum(i for i in impulses if i is not None and isinstance(i, (int, float)))

        # Get gripper force
        gripper_force = 0.0
        forces = gripper.get("gripper_object_contact_force_gt")
        if forces:
            gripper_force = max((f for f in forces if f is not None and isinstance(f, (int, float))), default=0.0)

        # Apply proxy rules
        fragility_high = fragility in ("high", "extreme")
        fragility_medium = fragility in ("medium", "high", "extreme")

        damage = False
        severity = "none"

        if drop_event and drop_height is not None:
            if drop_height > 50 and fragility_high:
                damage, severity = True, "broken"
            elif drop_height > 30 and fragility_medium:
                damage, severity = True, "minor"

        if not damage:
            if impulse > 20 and fragility_high:
                damage, severity = True, "broken"
            elif impulse > 10 and fragility_medium:
                damage, severity = True, "minor"
            elif gripper_force > 100 and fragility_high:
                damage, severity = True, "functional_damage"

        outcome["damage_state_gt"] = severity

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
        ee_poses = robot.get("ee_pose_gt")
        if ee_poses and len(ee_poses) > 0:
            ee_dists = []
            for i in range(len(ee_poses)):
                ee = ee_poses[i]
                if ee is None or len(ee) < 3:
                    ee_dists.append(None)
                    continue
                min_d = float('inf')
                for obs_name, obs_traj in obs_trajs.items():
                    idx = min(i, len(obs_traj) - 1)
                    obs = obs_traj[idx]
                    dx = ee[0] - obs[0]
                    dy = ee[1] - obs[1]
                    dz = ee[2] - obs[2]
                    d = math.sqrt(dx * dx + dy * dy + dz * dz) * 100.0  # m -> cm
                    min_d = min(min_d, d)
                ee_dists.append(min_d if min_d < float('inf') else None)
            dist["ee_human_distance_gt"] = ee_dists

        # ── S-DIST-003: object_human_distance_gt ──
        obj_poses = obj_state.get("object_pose_gt", {})
        pick_obj = self._find_pick_object(obj_poses)
        obj_trans = pick_obj.get("translation_per_step") if pick_obj else None
        if obj_trans and len(obj_trans) > 0:
            obj_dists = []
            for i in range(len(obj_trans)):
                obj = obj_trans[i]
                if obj is None or len(obj) < 3:
                    obj_dists.append(None)
                    continue
                min_d = float('inf')
                for obs_name, obs_traj in obs_trajs.items():
                    idx = min(i, len(obs_traj) - 1)
                    obs = obs_traj[idx]
                    dx = obj[0] - obs[0]
                    dy = obj[1] - obs[1]
                    dz = obj[2] - obs[2]
                    d = math.sqrt(dx * dx + dy * dy + dz * dz) * 100.0
                    min_d = min(min_d, d)
                obj_dists.append(min_d if min_d < float('inf') else None)
            dist["object_human_distance_gt"] = obj_dists

        # ── S-DIST-001: robot_human_distance_matrix_gt ──
        # Simplified: matrix of EE and object distances to each obstacle
        if ee_poses and obs_trajs:
            matrix = []
            for i in range(len(ee_poses)):
                row = {}
                ee = ee_poses[i]
                for obs_name, obs_traj in obs_trajs.items():
                    idx = min(i, len(obs_traj) - 1)
                    obs = obs_traj[idx]
                    if ee and len(ee) >= 3:
                        dx = ee[0] - obs[0]
                        dy = ee[1] - obs[1]
                        dz = ee[2] - obs[2]
                        row[obs_name] = math.sqrt(dx * dx + dy * dy + dz * dz) * 100.0
                matrix.append(row)
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
        """S-PLAN-002: Build executed trajectory from joint position time series.

        Constructs a structured trajectory from the already-available
        joint_position_q_gt data. This is the actual executed path.
        """
        planner = raw_gt.get("planner_log", {})
        robot = raw_gt.get("robot_state", {})

        q_left = robot.get("joint_position_q_gt")
        q_right = robot.get("joint_position_q_right_gt")
        ee_poses = robot.get("ee_pose_gt")

        trajectory = []
        dt = self._dt

        # Left arm trajectory
        if q_left and len(q_left) > 0:
            left_traj = []
            for i, q in enumerate(q_left):
                entry = {
                    "t": round(i * dt, 4),
                    "joint_positions": [round(v, 6) for v in q] if q else None,
                }
                # Add EE pose if available
                if ee_poses and i < len(ee_poses) and ee_poses[i]:
                    entry["ee_pose"] = [round(v, 6) for v in ee_poses[i][:3]] if len(ee_poses[i]) >= 3 else None
                left_traj.append(entry)
            trajectory.append({
                "arm": "left",
                "n_steps": len(left_traj),
                "trajectory": left_traj,
            })

        # Right arm trajectory
        if q_right and len(q_right) > 0:
            right_traj = []
            for i, q in enumerate(q_right):
                entry = {
                    "t": round(i * dt, 4),
                    "joint_positions": [round(v, 6) for v in q] if q else None,
                }
                right_traj.append(entry)
            trajectory.append({
                "arm": "right",
                "n_steps": len(right_traj),
                "trajectory": right_traj,
            })

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
