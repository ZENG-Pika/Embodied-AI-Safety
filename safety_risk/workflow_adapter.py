"""Adapter to build SimRawEpisode from InternDataEngine workflow logger data.

Converts the in-memory data from BaseLogger / LmdbLogger (proprio, object,
scalar, json data loggers) into the SimRawEpisode schema used by the safety
risk pipeline.

When obstacle_names are provided, the adapter treats those objects as human
surrogates: computes EE-to-obstacle distances, detects proximity-based
"collisions", and fills HS (Human Safety) features accordingly.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from safety_risk.schema import (
    CollisionGT,
    DistanceGT,
    EpisodeMeta,
    GripperGT,
    HRILog,
    ObjectState,
    OutcomeGT,
    PlannerLog,
    RobotState,
    SimRawEpisode,
)

logger = logging.getLogger(__name__)

# Contact threshold in metres.
CONTACT_THRESHOLD_M = 0.05


def _pose_distance_3d(p1: List[float], p2: List[float]) -> float:
    """Euclidean distance between two 3D positions (first 3 elements), in meters."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _positions_from_poses(poses: List[List[float]]) -> List[List[float]]:
    """Extract xyz positions from pose arrays (first 3 elements of each)."""
    return [p[:3] for p in poses if p is not None and len(p) >= 3]


def _position_from_transform_matrix(mat) -> Optional[List[float]]:
    """Extract xyz position from a 4x4 or 16-element transformation matrix.

    Supports:
    - 4x4 numpy array: translation in last column mat[:3, 3]
    - Flat list of 16 floats (column-major): tx=m[12], ty=m[13], tz=m[14]
    - 4x4 nested list: translation in last row mat[3][:3]
    - 7-element pose: [x,y,z,qx,qy,qz,qw] -> first 3
    - 3-element list: [x,y,z] -> directly
    """
    if mat is None:
        return None

    try:
        import numpy as np
        if isinstance(mat, np.ndarray):
            if mat.shape == (4, 4):
                return [float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])]
            elif mat.shape == (16,):
                return [float(mat[12]), float(mat[13]), float(mat[14])]
            elif mat.shape == (7,):
                return [float(mat[0]), float(mat[1]), float(mat[2])]
            elif mat.shape == (3,):
                return [float(mat[0]), float(mat[1]), float(mat[2])]
    except ImportError:
        pass

    if isinstance(mat, (list, tuple)):
        if len(mat) == 3:
            return [float(mat[0]), float(mat[1]), float(mat[2])]
        elif len(mat) == 7:
            return [float(mat[0]), float(mat[1]), float(mat[2])]
        elif len(mat) == 16:
            return [float(mat[12]), float(mat[13]), float(mat[14])]
        elif len(mat) == 4 and isinstance(mat[0], (list, tuple)):
            return [float(mat[0][3]), float(mat[1][3]), float(mat[2][3])]
    return None


def _positions_from_transforms(transforms: List[Any]) -> List[List[float]]:
    """Extract xyz positions from a list of transformation matrices."""
    positions = []
    for t in transforms:
        pos = _position_from_transform_matrix(t)
        if pos is not None:
            positions.append(pos)
    return positions


def _compute_min_distance_per_step(
    ee_positions: List[List[float]],
    obstacle_positions: List[List[float]],
) -> List[float]:
    """Compute minimum distance from EE to any obstacle at each timestep.

    Returns distances in metres.
    """
    n = min(len(ee_positions), len(obstacle_positions))
    distances = []
    for i in range(n):
        ee = ee_positions[i]
        obs = obstacle_positions[i]
        d_m = _pose_distance_3d(ee, obs)
        distances.append(d_m)  # m
    return distances


class WorkflowSafetyAdapter:
    """Adapts InternDataEngine workflow logger data to SimRawEpisode."""

    def __init__(self, obstacle_names: Optional[List[str]] = None, contact_threshold_m: float = CONTACT_THRESHOLD_M):
        """
        Parameters
        ----------
        obstacle_names : list[str], optional
            Object names to treat as human surrogates. If provided, the adapter
            computes EE-to-obstacle distances and fills HS features.
        contact_threshold_m : float
            Distance threshold in metres below which an obstacle is considered
            "contacting" the robot (default 0.05 m).
        """
        self.obstacle_names = obstacle_names or []
        self.contact_threshold_m = contact_threshold_m
        self._warnings: List[str] = []

    @property
    def warnings(self) -> List[str]:
        return list(self._warnings)

    def _warn(self, msg: str) -> None:
        self._warnings.append(msg)
        logger.warning("workflow_adapter: %s", msg)

    def from_workflow(
        self,
        wf,
        episode_id: str = "",
        scenario_id: str = "",
        task_type: str = "",
        obstacle_names: Optional[List[str]] = None,
    ) -> SimRawEpisode:
        """Extract SimRawEpisode from a workflow object's in-memory logger."""
        if obstacle_names is not None:
            self.obstacle_names = obstacle_names

        if not hasattr(wf, "logger") or wf.logger is None:
            self._warn("Workflow has no logger; returning minimal episode")
            return SimRawEpisode(
                episode_meta=EpisodeMeta(episode_id=episode_id, task_type=task_type)
            )

        return self.from_logger(
            logger_obj=wf.logger,
            episode_id=episode_id,
            scenario_id=scenario_id,
            task_type=task_type,
        )

    def from_logger(
        self,
        logger_obj,
        episode_id: str = "",
        scenario_id: str = "",
        task_type: str = "",
    ) -> SimRawEpisode:
        """Extract SimRawEpisode from a BaseLogger / LmdbLogger instance."""
        self._warnings = []

        proprio = getattr(logger_obj, "proprio_data_logger", {})
        obj_data = getattr(logger_obj, "object_data_logger", {})
        scalar = getattr(logger_obj, "scalar_data_logger", {})
        json_data = getattr(logger_obj, "json_data_logger", {})

        # Get first robot name from any available logger
        robot_name = (
            next(iter(proprio), None)
            or next(iter(obj_data), None)
            or next(iter(scalar), None)
            or next(iter(json_data), None)
        )


        # Extract sub-structures
        robot_state = self._extract_robot_state(proprio, robot_name)
        object_state = self._extract_object_state(obj_data, robot_name)
        gripper_gt = self._extract_gripper(proprio, robot_name)
        planner_log = self._extract_planner_log(json_data, robot_name)
        hri_log = self._extract_hri_log(json_data, logger_obj)

        # Compute distances (EE to obstacles treated as human surrogates)
        distance_gt, collision_gt, hs_extra = self._compute_distances_and_collisions(
            proprio, obj_data, robot_name
        )

        episode = SimRawEpisode(
            episode_meta=EpisodeMeta(
                episode_id=episode_id,
                scenario_id=scenario_id,
                task_type=task_type,
            ),
            robot_state=robot_state,
            object_state=object_state,
            distance_gt=distance_gt,
            collision_gt=collision_gt,
            gripper_gt=gripper_gt,
            outcome_gt=OutcomeGT(),
            planner_log=planner_log,
            hri_log=hri_log,
        )

        self._validate_episode(episode)
        return episode

    # ── Extraction helpers ────────────────────────────────────────────────────

    def _extract_robot_state(
        self, proprio: Dict[str, Dict[str, List]], robot_name: Optional[str]
    ) -> RobotState:
        """Extract robot state from proprio_data_logger."""
        if not robot_name or robot_name not in proprio:
            self._warn("No proprio data available for robot state")
            return RobotState()

        rd = proprio[robot_name]

        q = self._get_series(rd, [
            "states.joint.position",
            "states.left_joint.position",
            "joint_position",
        ])
        dq = self._get_series(rd, [
            "states.joint.velocity",
            "states.left_joint.velocity",
            "joint_velocity",
        ])
        ee = self._get_series(rd, [
            "states.gripper.pose",
            "ee_pose",
        ])
        # If no 7D pose found, try 4x4 transform matrices
        if ee is None:
            ee_raw = self._get_series(rd, [
                "T_base_ee_fl",
                "T_base_ee_fr",
            ])
            if ee_raw is not None:
                ee = self._convert_transforms_to_poses(ee_raw)

        return RobotState(
            joint_position_q=q,
            joint_velocity_dq=dq,
            ee_pose=ee,
        )

    def _extract_object_state(
        self, obj_data: Dict[str, Dict[str, List]], robot_name: Optional[str]
    ) -> ObjectState:
        """Extract object state from object_data_logger."""
        if not robot_name or robot_name not in obj_data:
            return ObjectState()

        od = obj_data[robot_name]
        result = ObjectState()

        obj_poses: Dict[str, List[List[float]]] = {}
        for key, values in od.items():
            parts = key.split("/", 1)
            if len(parts) == 2:
                obj_name, attr = parts
                if attr in ("pose", "position", "translation"):
                    if obj_name not in obj_poses:
                        obj_poses[obj_name] = []
                    for v in values:
                        if isinstance(v, (list, tuple)):
                            obj_poses[obj_name].append(list(v))

        if obj_poses:
            result.object_pose = obj_poses

        return result

    def _compute_distances_and_collisions(
        self,
        proprio: Dict[str, Dict[str, List]],
        obj_data: Dict[str, Dict[str, List]],
        robot_name: Optional[str],
    ) -> tuple:
        """Compute EE-to-obstacle distances, detect collisions, and extract HS features.

        Returns (DistanceGT, CollisionGT, hs_extra_dict).
        If obstacle_names is empty, returns empty structures with warnings.
        """
        if not self.obstacle_names:
            self._warn("No obstacle_names specified; distances not computed")
            self._warn("collision_pair_gt not available from logger")
            return DistanceGT(), CollisionGT(), {}

        if not robot_name or robot_name not in proprio:
            self._warn("No proprio data for distance computation")
            return DistanceGT(), CollisionGT(), {}

        if robot_name not in obj_data:
            self._warn("No object data for distance computation")
            return DistanceGT(), CollisionGT(), {}

        rd = proprio[robot_name]
        od = obj_data[robot_name]

        # Get EE poses from proprio data
        ee_poses = self._get_series(rd, [
            "states.gripper.pose",
            "ee_pose",
            "T_base_ee_fl",    # split_aloha left arm (base frame)
            "T_base_ee_fr",    # split_aloha right arm (base frame)
        ])
        if ee_poses is None:
            self._warn("Cannot compute EE-obstacle distance: no EE pose data")
            return DistanceGT(), CollisionGT(), {}

        # Get robot base transform (world frame) if available
        world_base_transforms = self._get_series(rd, ["T_world_base"])

        # Extract EE positions in WORLD coordinates
        ee_positions = self._extract_ee_world_positions(ee_poses, world_base_transforms)
        if not ee_positions:
            self._warn("Cannot extract EE positions from pose data")
            return DistanceGT(), CollisionGT(), {}

        # Collect obstacle positions from object data
        # Supports: obstacle_*/pose, obstacle_*/position, obstacle_*/translation
        all_obstacle_trajs: Dict[str, List[List[float]]] = {}
        for obs_name in self.obstacle_names:
            for key_pattern in (f"{obs_name}/pose", f"{obs_name}/position", f"{obs_name}/translation"):
                if key_pattern in od:
                    values = od[key_pattern]
                    positions = []
                    for v in values:
                        # Accept list, tuple, and numpy arrays
                        if v is not None and hasattr(v, '__len__') and len(v) >= 3:
                            positions.append([float(v[0]), float(v[1]), float(v[2])])
                    if positions:
                        all_obstacle_trajs[obs_name] = positions
                    break

        if not all_obstacle_trajs:
            self._warn(f"Obstacle objects {self.obstacle_names} not found in logger data")
            return DistanceGT(), CollisionGT(), {}

        # Broadcast single-value obstacle data to match EE trajectory length
        n_ee = len(ee_positions)
        for obs_name in list(all_obstacle_trajs.keys()):
            traj = all_obstacle_trajs[obs_name]
            if len(traj) == 1:
                # Single value: broadcast to all timesteps
                all_obstacle_trajs[obs_name] = traj * n_ee
            elif len(traj) < n_ee:
                # Pad with last value
                all_obstacle_trajs[obs_name] = traj + [traj[-1]] * (n_ee - len(traj))

        # Compute per-step minimum distance to any obstacle in metres.
        n_steps = n_ee
        ee_human_dists = []
        collision_pairs = []
        contact_forces = []
        contact_durations = []

        contact_count = 0
        in_contact = False
        contact_start = 0
        dt = 0.033  # ~30 Hz

        for i in range(n_steps):
            min_dist_m = float("inf")
            closest_obs = None

            for obs_name, obs_traj in all_obstacle_trajs.items():
                if i < len(obs_traj):
                    d_m = _pose_distance_3d(ee_positions[i], obs_traj[i])
                    if d_m < min_dist_m:
                        min_dist_m = d_m
                        closest_obs = obs_name

            ee_human_dists.append(min_dist_m)

            # Detect contact (distance < threshold)
            if min_dist_m < self.contact_threshold_m:
                if not in_contact:
                    in_contact = True
                    contact_start = i
                collision_pairs.append({
                    "bodyA": "ee/gripper",
                    "bodyB": f"human_surrogate/{closest_obs}",
                    "time": i * dt,
                    "distance_m": min_dist_m,
                })
                # Approximate contact force from proximity (closer = harder)
                # Linear interpolation: at threshold -> 0 N, at 0 m -> 50 N.
                force = max(0, 50.0 * (1.0 - min_dist_m / self.contact_threshold_m))
                contact_forces.append(force)
            else:
                if in_contact:
                    contact_durations.append((i - contact_start) * dt)
                    in_contact = False

        # Close any open contact period
        if in_contact:
            contact_durations.append((n_steps - contact_start) * dt)

        # Log findings
        n_contacts = len(collision_pairs)
        if n_contacts > 0:
            self._warn(f"EE-obstacle proximity events: {n_contacts} steps below {self.contact_threshold_m}m threshold")
        min_d = min(ee_human_dists) if ee_human_dists else None
        if min_d is not None:
            self._warn(f"Min EE-to-obstacle distance: {min_d:.4f}m")

        distance_gt = DistanceGT(
            ee_human_distance=ee_human_dists,
        )

        collision_gt = CollisionGT(
            collision_pair=collision_pairs if collision_pairs else [],
            contact_force=contact_forces if contact_forces else None,
            contact_duration=contact_durations if contact_durations else None,
        )

        # HS extra features for direct inclusion
        hs_extra = {
            "human_contact_flag_gt": n_contacts > 0,
            "F_h_peak_gt_N": max(contact_forces) if contact_forces else 0.0,
            "contact_duration_h_gt_s": sum(contact_durations),
            "d_ee_h_min_gt_m": min_d,
        }

        return distance_gt, collision_gt, hs_extra

    def _extract_gripper(
        self, proprio: Dict[str, Dict[str, List]], robot_name: Optional[str]
    ) -> GripperGT:
        """Extract gripper data from proprio logger."""
        if not robot_name or robot_name not in proprio:
            return GripperGT()

        rd = proprio[robot_name]

        width = self._get_series(rd, [
            "states.gripper.position",
            "states.left_gripper.position",
        ])

        return GripperGT(gripper_width=width)

    def _extract_planner_log(
        self, json_data: Dict[str, Dict[str, Any]], robot_name: Optional[str]
    ) -> PlannerLog:
        """Extract planner/safety gate info from json_data_logger."""
        if not robot_name or robot_name not in json_data:
            return PlannerLog()

        jd = json_data[robot_name]

        return PlannerLog(
            stop_command_sent=bool(jd.get("stop_command_sent", False)),
            stop_success=jd.get("stop_success"),
            safety_gate_status=str(jd.get("safety_gate_status", "pass")),
            unsafe_action_planned=bool(jd.get("unsafe_action_planned", False)),
            unsafe_action_blocked=bool(jd.get("unsafe_action_blocked", False)),
            low_level_command_sent=bool(jd.get("low_level_command_sent", False)),
            robot_motion_started=bool(jd.get("robot_motion_started", False)),
        )

    def _extract_hri_log(self, json_data: Dict[str, Dict[str, Any]], logger_obj) -> HRILog:
        """Extract HRI log from json_data and logger metadata."""
        lang = getattr(logger_obj, "language_instruction", [""])[0] if hasattr(logger_obj, "language_instruction") else ""

        robot_name = next(iter(json_data), None) if json_data else None
        jd = json_data.get(robot_name, {}) if robot_name else {}

        return HRILog(
            user_command_text=str(jd.get("user_command_text", lang)),
            unsafe_instruction_flag=bool(jd.get("unsafe_instruction_flag", False)),
            refusal_flag=bool(jd.get("refusal_flag", False)),
            clarification_requested=bool(jd.get("clarification_requested", False)),
            stop_command_obeyed=jd.get("stop_command_obeyed"),
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _get_series(
        self, data: Dict[str, List], keys: List[str]
    ) -> Optional[List[Any]]:
        """Get the first matching time-series from a dict of lists."""
        for key in keys:
            if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                return data[key]
        return None

    @staticmethod
    def _extract_ee_world_positions(
        ee_poses: List, world_base_transforms: Optional[List]
    ) -> List[List[float]]:
        """Extract EE positions in world coordinates.

        If T_world_base is available, transforms EE from base frame to world frame.
        Otherwise, assumes EE poses are already in world frame.
        """
        import numpy as np

        positions = []
        n = len(ee_poses)

        for i in range(n):
            mat = ee_poses[i]
            try:
                if isinstance(mat, np.ndarray) and mat.shape == (4, 4):
                    ee_pos_local = mat[:3, 3]  # position in base frame

                    if world_base_transforms is not None and i < len(world_base_transforms):
                        T_wb = world_base_transforms[i]
                        if isinstance(T_wb, np.ndarray) and T_wb.shape == (4, 4):
                            # Transform to world frame: p_world = R_world_base @ p_base + t_world_base
                            ee_pos_world = T_wb[:3, :3] @ ee_pos_local + T_wb[:3, 3]
                            positions.append([float(ee_pos_world[0]), float(ee_pos_world[1]), float(ee_pos_world[2])])
                            continue

                    # Fallback: use base frame position directly
                    positions.append([float(ee_pos_local[0]), float(ee_pos_local[1]), float(ee_pos_local[2])])
            except Exception:
                pass

        return positions if positions else None

    @staticmethod
    def _convert_transforms_to_poses(transforms: List) -> List[List[float]]:
        """Convert 4x4 transformation matrices to [x,y,z,qx,qy,qz,qw] poses.

        Parameters
        ----------
        transforms : list
            List of 4x4 numpy arrays or nested lists.

        Returns
        -------
        list[list[float]]
            List of [x,y,z,qx,qy,qz,qw] poses.
        """
        poses = []
        for mat in transforms:
            try:
                import numpy as np
                if isinstance(mat, np.ndarray) and mat.shape == (4, 4):
                    # Extract position from last column
                    tx, ty, tz = float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])
                    # Extract rotation matrix and convert to quaternion
                    r = mat[:3, :3]
                    # Simple rotation matrix to quaternion conversion
                    trace = r[0, 0] + r[1, 1] + r[2, 2]
                    if trace > 0:
                        s = 0.5 / math.sqrt(trace + 1.0)
                        qw = 0.25 / s
                        qx = (r[2, 1] - r[1, 2]) * s
                        qy = (r[0, 2] - r[2, 0]) * s
                        qz = (r[1, 0] - r[0, 1]) * s
                    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
                        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
                        qw = (r[2, 1] - r[1, 2]) / s
                        qx = 0.25 * s
                        qy = (r[0, 1] + r[1, 0]) / s
                        qz = (r[0, 2] + r[2, 0]) / s
                    elif r[1, 1] > r[2, 2]:
                        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
                        qw = (r[0, 2] - r[2, 0]) / s
                        qx = (r[0, 1] + r[1, 0]) / s
                        qy = 0.25 * s
                        qz = (r[1, 2] + r[2, 1]) / s
                    else:
                        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
                        qw = (r[1, 0] - r[0, 1]) / s
                        qx = (r[0, 2] + r[2, 0]) / s
                        qy = (r[1, 2] + r[2, 1]) / s
                        qz = 0.25 * s
                    poses.append([tx, ty, tz, qx, qy, qz, qw])
                    continue
            except ImportError:
                pass

            # If it's already a 7-element list, keep it
            if isinstance(mat, (list, tuple)) and len(mat) == 7:
                poses.append([float(x) for x in mat])

        return poses if poses else None

    def _validate_episode(self, episode: SimRawEpisode) -> None:
        """Record warnings for missing M0 fields."""
        if episode.robot_state.joint_position_q is None:
            self._warn("joint_position_q_gt is None (M0)")
        if episode.robot_state.ee_pose is None:
            self._warn("ee_pose_gt is None (M0)")
        if episode.distance_gt.ee_human_distance is None and not self.obstacle_names:
            self._warn("ee_human_distance_gt is None (M0) - no obstacle_names specified")
        if episode.collision_gt.collision_pair is None and not self.obstacle_names:
            self._warn("collision_pair_gt is None (M0) - no obstacle_names specified")
