"""Simulation raw GT extractor.

Converts simulation episode / rollout data from various sources into the
unified SimRawEpisode schema. Provides adapters for common data formats
used in InternDataEngine.

If the source data is already a dict matching SimRawEpisode schema, use
SimRawExtractor.from_dict(). For LMDB-based logger output, use
SimRawExtractor.from_lmdb(). For custom formats, subclass and override.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from safety_risk.schema import (
    CollisionGT,
    DistanceGT,
    EnvironmentState,
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


class SimRawExtractor:
    """Extracts SimRawEpisode from various simulation data sources."""

    def __init__(self, warnings_as_errors: bool = False):
        self.warnings_as_errors = warnings_as_errors
        self._warnings: List[str] = []

    @property
    def warnings(self) -> List[str]:
        return list(self._warnings)

    def _warn(self, msg: str) -> None:
        self._warnings.append(msg)
        if self.warnings_as_errors:
            raise ValueError(f"Warning treated as error: {msg}")
        logger.warning("sim_raw_extractor: %s", msg)

    # ── Main entry points ────────────────────────────────────────────────────

    def from_dict(self, data: Dict[str, Any]) -> SimRawEpisode:
        """Create SimRawEpisode from a dictionary.

        The dict should have keys matching SimRawEpisode fields:
        episode_meta, robot_state, object_state, etc.
        Each sub-dict is passed to the corresponding Pydantic model.
        """
        meta_data = data.get("episode_meta", {})
        if "episode_id" not in meta_data:
            meta_data["episode_id"] = data.get("episode_id", "unknown")

        episode = SimRawEpisode(
            episode_meta=EpisodeMeta(**meta_data),
            robot_state=RobotState(**data.get("robot_state", {})),
            object_state=ObjectState(**data.get("object_state", {})),
            environment_state=EnvironmentState(**data.get("environment_state", {})),
            distance_gt=DistanceGT(**data.get("distance_gt", {})),
            collision_gt=CollisionGT(**data.get("collision_gt", {})),
            gripper_gt=GripperGT(**data.get("gripper_gt", {})),
            outcome_gt=OutcomeGT(**data.get("outcome_gt", {})),
            planner_log=PlannerLog(**data.get("planner_log", {})),
            hri_log=HRILog(**data.get("hri_log", {})),
        )

        self._validate_episode(episode)
        return episode

    def from_json_file(self, path: str) -> SimRawEpisode:
        """Load SimRawEpisode from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.from_dict(data)

    def from_lmdb(self, lmdb_path: str, task_dir: str = "") -> SimRawEpisode:
        """Extract SimRawEpisode from an LMDB logger output.

        Reads the meta_info.pkl and logged data from the LMDB directory
        and maps it to the SimRawEpisode schema.
        """
        try:
            import lmdb
            import pickle
        except ImportError as e:
            raise ImportError("lmdb and pickle are required for LMDB extraction") from e

        meta_path = os.path.join(lmdb_path, "meta_info.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta_info.pkl not found in {lmdb_path}")

        with open(meta_path, "rb") as f:
            meta_info = pickle.load(f)

        # Extract episode metadata from meta_info
        episode_id = meta_info.get("episode_id", os.path.basename(lmdb_path))
        scenario_id = meta_info.get("scenario_id", task_dir)
        task_type = meta_info.get("task_type", "")

        # Read logged data
        proprio_data = self._read_lmdb_dict(lmdb_path, "proprio_data")
        object_data = self._read_lmdb_dict(lmdb_path, "object_data")
        scalar_data = self._read_lmdb_dict(lmdb_path, "scalar_data")
        json_data = self._read_lmdb_json(lmdb_path, "json_data")

        # Map to SimRawEpisode fields
        robot_state = self._extract_robot_state(proprio_data)
        object_state = self._extract_object_state(object_data)
        outcome_gt = self._extract_outcome(scalar_data, json_data)
        planner_log = self._extract_planner_log(json_data)
        hri_log = self._extract_hri_log(json_data)

        episode = SimRawEpisode(
            episode_meta=EpisodeMeta(
                episode_id=str(episode_id),
                scenario_id=str(scenario_id),
                task_type=str(task_type),
            ),
            robot_state=robot_state,
            object_state=object_state,
            outcome_gt=outcome_gt,
            planner_log=planner_log,
            hri_log=hri_log,
        )

        self._validate_episode(episode)
        return episode

    def from_logger_data(
        self,
        logger_data: Dict[str, Any],
        episode_id: str = "unknown",
        scenario_id: str = "",
        task_type: str = "",
    ) -> SimRawEpisode:
        """Extract from the InternDataEngine BaseLogger data format.

        Parameters
        ----------
        logger_data : dict
            Dictionary with keys like proprio_data_logger, object_data_logger,
            scalar_data_logger, json_data_logger, etc.
        episode_id : str
            Unique episode identifier.
        scenario_id : str
            Scenario identifier.
        task_type : str
            Task type string.
        """
        proprio = logger_data.get("proprio_data_logger", {})
        obj = logger_data.get("object_data_logger", {})
        scalar = logger_data.get("scalar_data_logger", {})
        json_d = logger_data.get("json_data_logger", {})

        robot_state = self._extract_robot_state_from_logger(proprio)
        object_state = self._extract_object_state_from_logger(obj)
        outcome_gt = self._extract_outcome_from_logger(scalar, json_d)

        episode = SimRawEpisode(
            episode_meta=EpisodeMeta(
                episode_id=episode_id,
                scenario_id=scenario_id,
                task_type=task_type,
            ),
            robot_state=robot_state,
            object_state=object_state,
            outcome_gt=outcome_gt,
        )

        self._validate_episode(episode)
        return episode

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate_episode(self, episode: SimRawEpisode) -> None:
        """Check for missing M0 fields and emit warnings."""
        meta = episode.episode_meta
        if not meta.episode_id or meta.episode_id == "unknown":
            self._warn("episode_id is missing or 'unknown'")

        rs = episode.robot_state
        if rs.joint_position_q is None:
            self._warn("joint_position_q_gt is None (M0)")
        if rs.ee_pose is None:
            self._warn("ee_pose_gt is None (M0)")
        if rs.link_pose is None:
            self._warn("link_pose_gt is None (M0)")

        dist = episode.distance_gt
        if dist.ee_human_distance is None:
            self._warn("ee_human_distance_gt is None (M0)")
        if dist.object_env_distance is None:
            self._warn("object_env_distance_gt is None (M0)")
        if dist.link_env_distance is None:
            self._warn("link_env_distance_gt is None (M0)")

        if episode.collision_gt.collision_pair is None:
            self._warn("collision_pair_gt is None (M0)")

    # ── LMDB helpers ─────────────────────────────────────────────────────────

    def _read_lmdb_dict(self, lmdb_path: str, subdir: str) -> Dict[str, List[Any]]:
        """Read a dict of time-series data from LMDB."""
        import lmdb
        import pickle

        result: Dict[str, List[Any]] = {}
        db_path = os.path.join(lmdb_path, subdir)
        if not os.path.exists(db_path):
            return result

        env = lmdb.open(db_path, readonly=True, lock=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                try:
                    result[key_str] = pickle.loads(value)
                except Exception:  # pylint: disable=broad-except
                    pass
        env.close()
        return result

    def _read_lmdb_json(self, lmdb_path: str, subdir: str) -> Dict[str, Any]:
        """Read JSON data from LMDB."""
        result: Dict[str, Any] = {}
        db_path = os.path.join(lmdb_path, subdir)
        if not os.path.exists(db_path):
            return result

        try:
            import lmdb

            env = lmdb.open(db_path, readonly=True, lock=False)
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                    try:
                        result[key_str] = json.loads(value)
                    except (json.JSONDecodeError, Exception):  # pylint: disable=broad-except
                        pass
            env.close()
        except ImportError:
            self._warn("lmdb not available for reading json data")
        return result

    def _extract_robot_state(self, proprio: Dict[str, List[Any]]) -> RobotState:
        """Extract robot state from proprio data dict."""
        return RobotState(
            joint_position_q=self._safe_get_series(proprio, "joint_position"),
            joint_velocity_dq=self._safe_get_series(proprio, "joint_velocity"),
            joint_torque=self._safe_get_series(proprio, "joint_torque"),
            ee_pose=self._safe_get_series(proprio, "ee_pose"),
        )

    def _extract_robot_state_from_logger(self, proprio: Dict[str, Dict[str, List]]) -> RobotState:
        """Extract robot state from BaseLogger proprio_data_logger format.

        Logger format: proprio_data_logger[robot_name][key] = [values per step]
        """
        # Use the first robot's data
        if not proprio:
            return RobotState()

        robot_name = next(iter(proprio))
        robot_data = proprio[robot_name]

        return RobotState(
            joint_position_q=self._safe_get_logger_series(robot_data, "joint_position"),
            joint_velocity_dq=self._safe_get_logger_series(robot_data, "joint_velocity"),
            joint_torque=self._safe_get_logger_series(robot_data, "joint_torque"),
            ee_pose=self._safe_get_logger_series(robot_data, "ee_pose"),
        )

    def _extract_object_state(self, obj_data: Dict[str, List[Any]]) -> ObjectState:
        """Extract object state from object data."""
        result = ObjectState()
        if "object_pose" in obj_data:
            result.object_pose = {"default": obj_data["object_pose"]}
        if "object_velocity" in obj_data:
            result.object_velocity = {"default": obj_data["object_velocity"]}
        return result

    def _extract_object_state_from_logger(self, obj_data: Dict[str, Dict[str, List]]) -> ObjectState:
        """Extract from logger format."""
        result = ObjectState()
        if not obj_data:
            return result

        robot_name = next(iter(obj_data))
        robot_obj = obj_data[robot_name]

        if "object_pose" in robot_obj:
            result.object_pose = {"default": robot_obj["object_pose"]}
        if "object_velocity" in robot_obj:
            result.object_velocity = {"default": robot_obj["object_velocity"]}
        return result

    def _extract_outcome(
        self, scalar: Dict[str, List[Any]], json_d: Dict[str, Any]
    ) -> OutcomeGT:
        """Extract outcome GT from scalar/json data."""
        return OutcomeGT(
            drop_event=bool(json_d.get("drop_event", False)),
            drop_height=scalar.get("drop_height", [None])[0] if scalar.get("drop_height") else None,
        )

    def _extract_outcome_from_logger(
        self, scalar: Dict[str, Dict[str, List]], json_d: Dict[str, Dict[str, Any]]
    ) -> OutcomeGT:
        """Extract from logger format."""
        result = OutcomeGT()
        if not scalar:
            return result

        robot_name = next(iter(scalar))
        robot_scalar = scalar[robot_name]

        if "drop_height" in robot_scalar and robot_scalar["drop_height"]:
            result.drop_height = robot_scalar["drop_height"][-1]
        return result

    def _extract_planner_log(self, json_d: Dict[str, Any]) -> PlannerLog:
        """Extract planner log from json data."""
        return PlannerLog(
            replan_flag=bool(json_d.get("replan_flag", False)),
            stop_command_sent=bool(json_d.get("stop_command_sent", False)),
            stop_success=json_d.get("stop_success"),
            unsafe_action_planned=bool(json_d.get("unsafe_action_planned", False)),
            unsafe_action_blocked=bool(json_d.get("unsafe_action_blocked", False)),
            low_level_command_sent=bool(json_d.get("low_level_command_sent", False)),
            robot_motion_started=bool(json_d.get("robot_motion_started", False)),
        )

    def _extract_hri_log(self, json_d: Dict[str, Any]) -> HRILog:
        """Extract HRI log from json data."""
        return HRILog(
            user_command_text=str(json_d.get("user_command_text", "")),
            unsafe_instruction_flag=bool(json_d.get("unsafe_instruction_flag", False)),
            refusal_flag=bool(json_d.get("refusal_flag", False)),
            clarification_requested=bool(json_d.get("clarification_requested", False)),
            stop_command_obeyed=json_d.get("stop_command_obeyed"),
        )

    def _safe_get_series(self, data: Dict[str, List[Any]], key: str) -> Optional[List[Any]]:
        """Safely get a time-series from a dict."""
        val = data.get(key)
        if val is not None and isinstance(val, list) and len(val) > 0:
            return val
        return None

    def _safe_get_logger_series(self, data: Dict[str, List], key: str) -> Optional[List[Any]]:
        """Safely get a time-series from logger data."""
        val = data.get(key)
        if val is not None and isinstance(val, list) and len(val) > 0:
            return val
        return None


def create_example_episode() -> SimRawEpisode:
    """Create an example SimRawEpisode for testing and documentation.

    Returns a minimal but complete episode with synthetic data that covers
    all required fields.
    """
    n_steps = 100
    n_joints = 7

    # Generate simple synthetic trajectories
    q_trajectory = [[0.1 * i + 0.01 * j for j in range(n_joints)] for i in range(n_steps)]
    dq_trajectory = [[0.01] * n_joints for _ in range(n_steps)]
    torque_trajectory = [[5.0 + 0.1 * i] * n_joints for i in range(n_steps)]

    ee_trajectory = [
        [0.5 + 0.001 * i, 0.0, 0.3 + 0.0005 * i, 0.0, 0.707, 0.0, 0.707]
        for i in range(n_steps)
    ]

    # Object moves toward edge
    obj_trajectory = [
        [0.5, -0.3 + 0.002 * i, 0.8, 0.0, 0.0, 0.0, 1.0]
        for i in range(n_steps)
    ]

    # Distance decreases over time (cm)
    ee_human_dist = [50.0 - 0.3 * i for i in range(n_steps)]
    obj_env_dist = [20.0 - 0.15 * i for i in range(n_steps)]
    link_env_dist = [30.0 - 0.1 * i for i in range(n_steps)]

    return SimRawEpisode(
        episode_meta=EpisodeMeta(
            episode_id="example_001",
            scenario_id="pick_place_table",
            random_seed=42,
            task_type="pick",
            object_id="cup_01",
            target_object_id="cup_01",
            object_hazard_class="none",
            object_fragility_class="medium",
        ),
        robot_state=RobotState(
            joint_position_q=q_trajectory,
            joint_velocity_dq=dq_trajectory,
            joint_torque=torque_trajectory,
            ee_pose=ee_trajectory,
        ),
        object_state=ObjectState(
            object_pose={"cup_01": obj_trajectory},
        ),
        distance_gt=DistanceGT(
            ee_human_distance=ee_human_dist,
            object_env_distance=obj_env_dist,
            link_env_distance=link_env_dist,
        ),
        collision_gt=CollisionGT(
            collision_pair=[],
        ),
        gripper_gt=GripperGT(
            gripper_object_contact_force=[10.0 + 0.5 * i for i in range(n_steps)],
        ),
        outcome_gt=OutcomeGT(
            drop_event=False,
            stable_final=True,
            damage_state="none",
        ),
        planner_log=PlannerLog(
            stop_success=True,
            safety_gate_status="pass",
        ),
        hri_log=HRILog(
            user_command_text="Pick up the cup",
            stop_command_obeyed=True,
        ),
    )
