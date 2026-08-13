"""PhysX runtime data collector for Sim_Raw_GT fields.

Collects per-step physics data from Isaac Sim's PhysX engine:
- Joint torques, link poses, link velocities
- Contact forces, collision pairs, penetration depth
- Object-link distances

Usage:
    In plan_with_render() loop, after world.step():
        collector.collect_step(task, step_id)
    After episode ends:
        collector.save(output_dir)
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PhysXDataCollector:
    """Collects per-step PhysX runtime data for safety risk evaluation.

    Call collect_step() after each world.step() in the simulation loop.
    Call save() after the episode to persist data alongside LMDB.
    """

    def __init__(self):
        self._data: Dict[str, List] = {
            # S-ROBOT-004: joint torques
            "joint_torque_gt": [],
            # S-ROBOT-005: all link poses (all links, per step)
            "link_pose_gt": [],
            # S-ROBOT-006: link velocities
            "link_velocity_gt": [],
            # S-HUM-001: per-step world poses for every human-surrogate body
            "human_body_pose_gt": [],
            # S-COLL-001: collision pairs
            "collision_pair_gt": [],
            # S-COLL-002: collision locations
            "collision_location_gt": [],
            # S-COLL-003: penetration depths
            "penetration_depth_gt": [],
            # S-COLL-004: contact forces (per contact view, per step)
            "contact_force_gt": [],
            # S-COLL-005: contact impulses (derived from forces)
            "contact_impulse_gt": [],
            # S-COLL-006: contact duration tracking
            "contact_events": [],  # raw events for duration computation
            # S-GRASP-001: gripper-object contact force
            "gripper_object_contact_force_gt": [],
            # S-DIST-001..003: collider-surface human distances
            "robot_human_distance_matrix_gt": [],
            "ee_human_distance_gt": [],
            "object_human_distance_gt": [],
            # S-DIST-004: object-env distances
            "object_env_distance_gt": [],
            # S-DIST-005: link-env distances
            "link_env_distance_gt": [],
            # S-DIST-006: self distances
            "self_distance_gt": [],
            # S-PLAN-001: planned trajectory (captured per plan event)
            "planned_trajectory": [],
            # S-PLAN-003: safety gate status per step
            "safety_gate_status": [],
            # Safety gate / stop tracking (episode-level results)
            "stop_success": None,
            "stop_margin_s": None,
            "t_stop_s": None,
            # Metadata
            "step_ids": [],
        }
        self._link_paths: Dict[str, List[str]] = {}  # robot_name -> [link_paths]
        self._initialized = False
        self._contact_state: Dict[tuple, dict] = {}  # (bodyA, bodyB) -> active contact state
        self._physics_dt_s = 0.033
        self._pending_contact_reports = []
        self._contact_report_sub = None
        self._data["contact_coverage_status"] = {
            "configured": False,
            "successful_steps": 0,
            "failed_steps": 0,
            "last_error": None,
        }
        self._data["contact_report_status"] = {
            "subscribed": False,
            "steps_with_reports": 0,
            "total_report_headers": 0,
        }
        self._data["runtime_physics"] = {
            "physics_dt_s": None,
            "gravity_direction": None,
            "gravity_magnitude_mps2": None,
        }
        self._data["distance_coverage_status"] = {
            "method": "collider_world_aabb_surface_clearance",
            "robot_body_count": 0,
            "human_body_count": 0,
            "object_count": 0,
            "environment_body_count": 0,
            "mano_proxy_body_count": 0,
        }
        try:
            from omni.physx import get_physx_simulation_interface
            from omni.physx.scripts import physicsUtils

            def _decode_path(encoded):
                try:
                    return str(physicsUtils.PhysicsSchemaTools.intToSdfPath(encoded))
                except Exception:
                    return ""

            def _on_contact_report(contact_headers, contact_data):
                pending = []
                for header in contact_headers:
                    start = int(header.contact_data_offset)
                    count = int(header.num_contact_data)
                    if count <= 0:
                        continue
                    impulses = []
                    positions = []
                    separations = []
                    normals = []
                    for index in range(start, start + count):
                        datum = contact_data[index]
                        impulses.append([float(value) for value in datum.impulse])
                        positions.append([float(value) for value in datum.position])
                        separations.append(float(datum.separation))
                        normals.append([float(value) for value in datum.normal])
                    pending.append({
                        "actor0": _decode_path(header.actor0),
                        "actor1": _decode_path(header.actor1),
                        "collider0": _decode_path(header.collider0),
                        "collider1": _decode_path(header.collider1),
                        "impulses_ns": impulses,
                        "positions_m": positions,
                        "separations_m": separations,
                        "normals": normals,
                        "source": "PhysX contact-report callback",
                    })
                self._pending_contact_reports.extend(pending)

            self._contact_report_sub = (
                get_physx_simulation_interface().subscribe_contact_report_events(
                    _on_contact_report
                )
            )
            self._data["contact_report_status"]["subscribed"] = True
        except Exception as exc:
            logger.warning("PhysX contact-report subscription unavailable: %s", exc)

        # Safety gate runtime state
        self._stop_triggered = False
        self._stop_step = None
        self._detect_step = None
        self._detect_distance_m = None
        self._prev_ee_obstacle_dist_m = None
        self._safety_stop_active = False
        self._safety_gate_enabled = True

    def collect_step(self, task, step_id: int) -> None:
        """Collect PhysX data for one simulation step.

        Call this AFTER world.step() returns.

        Parameters
        ----------
        task : BananaBaseTask
            The task object with robots, objects, and contact views.
        step_id : int
            Current step index.
        """
        self._data["step_ids"].append(step_id)

        try:
            self._collect_joint_torques(task)
        except Exception as e:
            self._data["joint_torque_gt"].append(None)

        try:
            self._collect_all_link_poses(task)
        except Exception as e:
            self._data["link_pose_gt"].append(None)

        try:
            self._collect_human_body_poses(task)
        except Exception as e:
            logger.warning("Failed to collect human body poses: %s", e)
            self._data["human_body_pose_gt"].append(None)

        try:
            self._collect_contact_data(task, step_id)
        except Exception as e:
            self._data["collision_pair_gt"].append(None)
            self._data["contact_force_gt"].append(None)
            self._data["contact_impulse_gt"].append(None)

        try:
            self._collect_distances(task)
        except Exception as e:
            logger.warning("Failed to collect collider surface distances: %s", e)
            self._data["robot_human_distance_matrix_gt"].append(None)
            self._data["ee_human_distance_gt"].append(None)
            self._data["object_human_distance_gt"].append(None)
            self._data["object_env_distance_gt"].append(None)
            self._data["link_env_distance_gt"].append(None)
            self._data["self_distance_gt"].append(None)

        # Planner data (from controllers)
        try:
            self._collect_planner_step(task, step_id)
        except Exception:
            self._data["safety_gate_status"].append(None)

    # ── Safety Gate (stop_success / stop_margin) ───────────────────────────

    # Default values (overridable via configure_safety_gate)
    SAFETY_DISTANCE_M = 0.30
    STOP_VELOCITY_THRESHOLD = 0.01
    STOP_VERIFY_STEPS = 5

    def configure_safety_gate(self, config: dict) -> None:
        """Configure safety gate from YAML config.

        Parameters
        ----------
        config : dict
            safety_gate section from task YAML, e.g.:
                enabled: true
                distance_threshold_m: 0.30
                stop_verify_steps: 5
        """
        if not config:
            return
        self._safety_gate_enabled = config.get("enabled", True)
        if "distance_threshold_m" in config:
            self.SAFETY_DISTANCE_M = config["distance_threshold_m"]
        if "stop_verify_steps" in config:
            self.STOP_VERIFY_STEPS = config["stop_verify_steps"]
        logger.info(
            f"[safety_gate] configured: enabled={self._safety_gate_enabled}, "
            f"threshold={self.SAFETY_DISTANCE_M}m, verify_steps={self.STOP_VERIFY_STEPS}"
        )

    def check_safety_gate(self, task, step_id: int, dt: float = 0.033) -> bool:
        """Check EE-to-obstacle distance and manage protective stop.

        Call AFTER collect_step(). Returns True if the safety gate is actively
        blocking actions (robot should hold position).

        Parameters
        ----------
        task : BananaBaseTask
            The task with robots and objects.
        step_id : int
            Current simulation step.
        dt : float
            Simulation timestep in seconds.

        Returns
        -------
        bool
            True if the safety stop is active and actions should be suppressed.
        """
        if not self._safety_gate_enabled:
            return False

        # If stop is already active, continue blocking
        if self._safety_stop_active:
            # Verify stop: check if robot has actually halted
            if self._stop_step is not None:
                steps_since_stop = step_id - self._stop_step
                if steps_since_stop >= self.STOP_VERIFY_STEPS:
                    self._verify_stop(task, step_id, dt)
            return True

        # Compute min link-to-obstacle distance (all links, not just EE)
        d_min_m = self._compute_link_obstacle_distance(task)
        if d_min_m is None:
            if not hasattr(self, '_debug_none_printed'):
                print(f"[safety_gate] step {step_id}: d_min_m is None")
                self._debug_none_printed = True
            return False

        # Debug: print distance periodically
        if step_id % 50 == 0 or d_min_m < self.SAFETY_DISTANCE_M:
            closest_link = getattr(self, '_debug_closest_link', '?')
            closest_obj = getattr(self, '_debug_closest_obj', '?')
            print(
                f"[safety_gate] step {step_id}: d_min={d_min_m:.4f}m "
                f"({closest_link}→{closest_obj}) threshold={self.SAFETY_DISTANCE_M:.4f}m"
            )

        # Compute approach velocity from distance history
        approach_vel_mps = None
        if self._prev_ee_obstacle_dist_m is not None:
            approach_vel_mps = (self._prev_ee_obstacle_dist_m - d_min_m) / dt
        self._prev_ee_obstacle_dist_m = d_min_m

        # Detection: distance below threshold AND approaching
        if d_min_m < self.SAFETY_DISTANCE_M:
            self._stop_triggered = True
            self._detect_step = step_id
            self._detect_distance_m = d_min_m
            self._safety_stop_active = True

            # Compute TTC at detection
            if approach_vel_mps is not None and approach_vel_mps > 0.01:
                ttc_s = d_min_m / approach_vel_mps
            else:
                # Fallback: assume obstacle speed (0.06 m/s)
                ttc_s = d_min_m / 0.06
            self._data["t_stop_s"] = step_id * dt
            self._data["_detect_ttc_s"] = ttc_s
            self._stop_step = step_id + 1  # stop takes effect next step

            logger.info(
                f"[safety_gate] STOP triggered at step {step_id}: "
                f"d={d_min_m:.4f}m, TTC={ttc_s:.2f}s"
            )
            return True

        return False

    def _compute_link_obstacle_distance(self, task) -> Optional[float]:
        """Compute minimum distance from any robot link to any object (meters).

        Checks base, EE links, and any intermediate links registered in _link_paths.
        """
        if not hasattr(task, 'objects') or not hasattr(task, 'robots'):
            if not hasattr(self, '_debug_printed'):
                print(f"[safety_gate] task missing objects/robots: objects={hasattr(task, 'objects')}, robots={hasattr(task, 'robots')}")
                self._debug_printed = True
            return None

        # Check if obstacle_1 exists
        if 'obstacle_1' not in task.objects:
            if not hasattr(self, '_debug_printed'):
                print(f"[safety_gate] obstacle_1 not in task.objects, available: {list(task.objects.keys())}")
                self._debug_printed = True
            return None

        min_dist = float('inf')
        for robot_name, robot in task.robots.items():
            # Collect all known link positions
            link_positions = []

            # Base position
            try:
                base_pos, _ = robot.get_world_pose()
                link_positions.append(("base", list(base_pos)))
            except Exception:
                pass

            # EE positions
            for ee_attr in ['fl_ee_path', 'fr_ee_path']:
                if not hasattr(robot, ee_attr):
                    continue
                ee_path = getattr(robot, ee_attr)
                if not ee_path:
                    continue
                try:
                    from omni.isaac.core.utils.xforms import get_world_pose
                    pos, _ = get_world_pose(ee_path)
                    link_positions.append((ee_attr, list(pos)))
                except Exception:
                    pass

            # Intermediate links from _link_paths (if initialized by collect_step)
            if robot_name in self._link_paths:
                for link_path in self._link_paths[robot_name]:
                    try:
                        from omni.isaac.core.utils.xforms import get_world_pose
                        pos, _ = get_world_pose(link_path)
                        link_positions.append((link_path, list(pos)))
                    except Exception:
                        pass

            # Compute distance from each link to obstacle
            for link_name, link_pos in link_positions:
                for obj_name, obj in task.objects.items():
                    if obj_name != 'obstacle_1':
                        continue  # only check obstacle (human hand)
                    try:
                        obj_pos, _ = obj.get_world_pose()
                        d = float(np.linalg.norm(np.array(link_pos) - np.array(obj_pos)))
                        if d < min_dist:
                            min_dist = d
                            if not hasattr(self, '_debug_dist_printed') or step_id % 50 == 0:
                                self._debug_closest_link = link_name
                                self._debug_closest_obj = obj_name
                    except Exception:
                        pass

        return min_dist if min_dist < float('inf') else None

    def _verify_stop(self, task, step_id: int, dt: float) -> None:
        """After stop, check if robot halted before contact with obstacle."""
        if self._data["stop_success"] is not None:
            return  # already verified

        # Check if contact with OBSTACLE occurred (not pick objects)
        contact_occurred = False
        if self._data["collision_pair_gt"]:
            recent_pairs = self._data["collision_pair_gt"][-self.STOP_VERIFY_STEPS:]
            for pairs in recent_pairs:
                if pairs is None:
                    continue
                if isinstance(pairs, list):
                    for pair in pairs:
                        if isinstance(pair, dict):
                            body_b = pair.get("bodyB", "")
                            if "obstacle" in body_b.lower() or "mano" in body_b.lower():
                                contact_occurred = True
                                break
                if contact_occurred:
                    break

        if contact_occurred:
            self._data["stop_success"] = False
            logger.info(f"[safety_gate] stop_success=False (contact occurred before halt)")
        else:
            self._data["stop_success"] = True
            logger.info(f"[safety_gate] stop_success=True (robot halted safely)")

        # Compute stop margin
        detect_ttc = self._data.get("_detect_ttc_s")
        t_stop = self._data.get("t_stop_s")
        if detect_ttc is not None and t_stop is not None:
            # stop_latency = time from detection to robot actually halting
            stop_latency = (step_id - self._detect_step) * dt if self._detect_step else 0.0
            self._data["stop_margin_s"] = detect_ttc - stop_latency
            logger.info(f"[safety_gate] stop_margin={self._data['stop_margin_s']:.3f}s")

        # Clean up internal key
        self._data.pop("_detect_ttc_s", None)

    def is_stop_active(self) -> bool:
        """Check if safety stop is currently active."""
        return self._safety_stop_active

    def _collect_joint_torques(self, task) -> None:
        """Collect joint torques from all robots.

        For position-controlled robots, the physics solver computes torques
        internally. We try multiple APIs to read them:
        1. get_applied_joint_torques() - explicitly applied torques
        2. get_measured_joint_efforts() - measured efforts (works for pos control)
        3. get_joint_torques() - direct PhysX API
        """
        all_torques = []
        for robot_name, robot in task.robots.items():
            torques_found = False

            # Try articulation_view APIs
            if hasattr(robot, '_articulation_view') and robot._articulation_view is not None:
                av = robot._articulation_view
                for method_name in ('get_measured_joint_efforts', 'get_applied_joint_torques',
                                    'get_applied_joint_efforts', 'get_joint_torques'):
                    if hasattr(av, method_name):
                        try:
                            torques = getattr(av, method_name)()
                            if torques is not None:
                                if hasattr(torques, 'tolist'):
                                    torques = torques.tolist()
                                if isinstance(torques, (list, tuple)) and len(torques) > 0:
                                    first = torques[0]
                                    if hasattr(first, 'tolist'):
                                        first = first.tolist()
                                    if isinstance(first, (list, tuple)):
                                        vals = [float(x) for x in first]
                                        # Check if non-zero
                                        if any(abs(v) > 1e-6 for v in vals):
                                            all_torques.extend(vals)
                                            torques_found = True
                                            break
                        except Exception:
                            pass

            if not torques_found:
                all_torques.extend([None] * 6)

        self._data["joint_torque_gt"].append(all_torques if all_torques else None)

    # ── Link Poses & Velocities (S-ROBOT-005, S-ROBOT-006) ─────────────────

    def _collect_all_link_poses(self, task) -> None:
        """Collect world-frame poses and velocities for ALL robot links.

        Discovers links by traversing the robot prim tree on the first call,
        then reads world pose for each cached link path every step.
        Computes linear velocity from pose differences.

        Output format per step:
            link_pose_gt: {robot_name: {link_name: [x,y,z,qx,qy,qz,qw]}}
            link_velocity_gt: {robot_name: {link_name: [vx,vy,vz,0,0,0]}}
        """
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.xforms import get_world_pose as get_xform_world_pose

        if not hasattr(self, '_all_link_cache'):
            self._all_link_cache = {}
        if not hasattr(self, '_prev_all_link_poses'):
            self._prev_all_link_poses = {}

        result_poses = {}
        result_velocities = {}
        dt = self._physics_dt_s

        for robot_name, robot in task.robots.items():
            # Discover link paths on first step
            if robot_name not in self._all_link_cache:
                try:
                    link_map = self._discover_robot_links(robot)
                    self._all_link_cache[robot_name] = link_map
                except Exception as e:
                    logger.warning("Failed to discover links for %s: %s", robot_name, e)
                    self._all_link_cache[robot_name] = {}

            link_map = self._all_link_cache[robot_name]
            robot_poses = {}
            robot_velocities = {}

            for link_name, link_path in link_map.items():
                try:
                    prim = get_prim_at_path(link_path)
                    if prim and prim.IsValid():
                        pos, ori = get_xform_world_pose(link_path)
                        pose = [
                            float(pos[0]), float(pos[1]), float(pos[2]),
                            float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])
                        ]
                    else:
                        pose = [None] * 7
                except Exception:
                    pose = [None] * 7

                robot_poses[link_name] = pose

                # Compute velocity from pose difference
                prev_poses = self._prev_all_link_poses.get(robot_name, {})
                prev_pose = prev_poses.get(link_name)
                if (pose[0] is not None and prev_pose is not None
                        and prev_pose[0] is not None):
                    # Linear velocity
                    vx = (pose[0] - prev_pose[0]) / dt
                    vy = (pose[1] - prev_pose[1]) / dt
                    vz = (pose[2] - prev_pose[2]) / dt

                    # Angular velocity from quaternion difference
                    wx, wy, wz = self._compute_angular_velocity(
                        prev_pose[3:7], pose[3:7], dt
                    )

                    robot_velocities[link_name] = [vx, vy, vz, wx, wy, wz]
                else:
                    robot_velocities[link_name] = [0.0] * 6

            self._prev_all_link_poses[robot_name] = robot_poses
            result_poses[robot_name] = robot_poses
            result_velocities[robot_name] = robot_velocities

        self._data["link_pose_gt"].append(result_poses)
        self._data["link_velocity_gt"].append(result_velocities)

    def _discover_robot_links(self, robot) -> Dict[str, str]:
        """Discover all robot link prim paths by traversing the USD tree.

        Returns dict: {relative_path: prim_path}
        Only includes actual robot links, filtering out joints, materials,
        shaders, visuals, collisions, and other non-link prims.
        """
        from omni.isaac.core.utils.prims import get_prim_at_path

        link_map = {}

        if not hasattr(robot, 'prim_path'):
            return link_map

        robot_prim = get_prim_at_path(robot.prim_path)
        if not robot_prim or not robot_prim.IsValid():
            return link_map

        def _is_link(prim):
            """Check if a prim is an actual robot link (not joint/material/etc)."""
            name = prim.GetName()
            name_lower = name.lower()

            # Skip joints
            if 'joint' in name_lower:
                return False
            # Skip materials and shaders
            if 'material' in name_lower or name_lower == 'shader':
                return False
            # Skip visual-only containers
            if name_lower in ('visuals', 'looks', 'collisions'):
                return False
            # Skip non-robot prims
            if name_lower in ('physscene', 'world', 'scene'):
                return False

            # Keep prims with link-like names
            # Match: xxx_link, link1, link2, ..., arm_base, base_link
            if name_lower.endswith('_link') or name_lower == 'arm_base':
                return True
            import re
            if re.match(r'^link\d+$', name_lower):
                return True

            return False

        def _traverse(prim, depth=0, max_depth=8):
            if depth > max_depth:
                return
            for child in prim.GetChildren():
                child_path = str(child.GetPrimPath())
                # Use full relative path as key to avoid name collisions
                rel_name = child_path.replace(robot.prim_path + "/", "")
                if _is_link(child) and rel_name not in link_map:
                    link_map[rel_name] = child_path
                _traverse(child, depth + 1, max_depth)

        _traverse(robot_prim)
        return link_map

    def _collect_human_body_poses(self, task) -> None:
        """Collect world poses for every rigid link in configured human obstacles.

        MANO is an articulated obstacle.  Recording only its root transform
        loses palm/finger motion, so each rigid body is discovered from the USD
        hierarchy and sampled every physics step.

        Per-step output:
            {
              "obstacle_1": {
                "palm": {
                  "prim_path": ".../mano/palm",
                  "pose": [x,y,z,qx,qy,qz,qw]
                },
                ...
              }
            }
        """
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.xforms import get_world_pose
        from pxr import Usd, UsdPhysics

        if not hasattr(self, "_human_body_link_cache"):
            self._human_body_link_cache = {}

        objects = getattr(task, "objects", {}) or {}
        frame = {}
        for obstacle_name, obstacle in objects.items():
            if "obstacle" not in str(obstacle_name).lower():
                continue

            if obstacle_name not in self._human_body_link_cache:
                root_path = (
                    getattr(obstacle, "prim_path", None)
                    or getattr(obstacle, "object_prim_path", None)
                )
                link_map = {}
                if root_path:
                    root_prim = get_prim_at_path(root_path)
                    if root_prim and root_prim.IsValid():
                        root_prefix = str(root_prim.GetPath()).rstrip("/") + "/"
                        for prim in Usd.PrimRange(root_prim):
                            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                                continue
                            prim_path = str(prim.GetPath())
                            relative_path = prim_path.replace(root_prefix, "", 1)
                            link_name = prim.GetName()
                            if link_name in link_map:
                                link_name = relative_path
                            link_map[link_name] = prim_path
                self._human_body_link_cache[obstacle_name] = {
                    "root_prim_path": root_path,
                    "links": link_map,
                }

            cached = self._human_body_link_cache[obstacle_name]
            body_parts = {}
            for link_name, prim_path in cached["links"].items():
                pose = None
                try:
                    prim = get_prim_at_path(prim_path)
                    if prim and prim.IsValid():
                        position, orientation_wxyz = get_world_pose(prim_path)
                        # Isaac Sim get_world_pose returns scalar-first WXYZ.
                        # Sim_Raw_GT consistently declares quaternion order XYZW.
                        pose = [
                            float(position[0]),
                            float(position[1]),
                            float(position[2]),
                            float(orientation_wxyz[1]),
                            float(orientation_wxyz[2]),
                            float(orientation_wxyz[3]),
                            float(orientation_wxyz[0]),
                        ]
                except Exception:
                    pose = None
                body_parts[link_name] = {
                    "prim_path": prim_path,
                    "pose": pose,
                }

            frame[obstacle_name] = {
                "root_prim_path": cached["root_prim_path"],
                "body_parts": body_parts,
            }

        self._data["human_body_pose_gt"].append(frame)

    def build_human_body_pose_gt(self) -> Optional[Dict[str, Any]]:
        """Convert per-step MANO samples into body-part pose time series."""
        frames = self._data.get("human_body_pose_gt", [])
        if not frames or not any(isinstance(frame, dict) and frame for frame in frames):
            return None

        obstacles: Dict[str, Any] = {}
        obstacle_names = sorted({
            obstacle_name
            for frame in frames if isinstance(frame, dict)
            for obstacle_name in frame
        })
        for obstacle_name in obstacle_names:
            root_path = None
            link_records: Dict[str, Dict[str, Any]] = {}
            link_names = sorted({
                link_name
                for frame in frames if isinstance(frame, dict)
                for link_name in (
                    frame.get(obstacle_name, {}).get("body_parts", {})
                    if isinstance(frame.get(obstacle_name), dict) else {}
                )
            })
            for link_name in link_names:
                prim_path = None
                poses = []
                for frame in frames:
                    obstacle_frame = (
                        frame.get(obstacle_name, {})
                        if isinstance(frame, dict) else {}
                    )
                    if root_path is None and isinstance(obstacle_frame, dict):
                        root_path = obstacle_frame.get("root_prim_path")
                    record = (
                        obstacle_frame.get("body_parts", {}).get(link_name)
                        if isinstance(obstacle_frame, dict) else None
                    )
                    if isinstance(record, dict):
                        prim_path = prim_path or record.get("prim_path")
                        poses.append(record.get("pose"))
                    else:
                        poses.append(None)
                link_records[link_name] = {
                    "prim_path": prim_path,
                    "pose_per_step": poses,
                }
            obstacles[obstacle_name] = {
                "root_prim_path": root_path,
                "body_parts": link_records,
            }

        return {
            "surrogate_type": "articulated_mano_hand",
            "coordinate_frame": "world",
            "pose_format": "[x,y,z,qx,qy,qz,qw]",
            "quaternion_order": "xyzw",
            "position_unit": "m",
            "num_steps": len(frames),
            "obstacles": obstacles,
            "source": "runtime USD rigid-body world poses sampled after each physics step",
        }

    def _compute_angular_velocity(self, q_prev, q_curr, dt):
        """Compute angular velocity from two quaternions.

        Parameters
        ----------
        q_prev : list
            Previous quaternion [qx, qy, qz, qw]
        q_curr : list
            Current quaternion [qx, qy, qz, qw]
        dt : float
            Time step

        Returns
        -------
        tuple
            Angular velocity (wx, wy, wz) in rad/s
        """
        # Ensure quaternions are normalized
        import math

        def normalize(q):
            n = math.sqrt(sum(x*x for x in q))
            return [x/n for x in q] if n > 0 else q

        q0 = normalize(q_prev)
        q1 = normalize(q_curr)

        # Quaternion difference: dq = q1 * conj(q0)
        # conj(q0) = [-q0x, -q0y, -q0z, q0w]
        dq = [
            q1[3]*(-q0[0]) + q1[0]*q0[3] + q1[1]*(-q0[2]) - q1[2]*(-q0[1]),
            q1[3]*(-q0[1]) - q1[0]*(-q0[2]) + q1[1]*q0[3] + q1[2]*(-q0[0]),
            q1[3]*(-q0[2]) + q1[0]*(-q0[1]) - q1[1]*(-q0[0]) + q1[2]*q0[3],
            q1[3]*q0[3] - q1[0]*(-q0[0]) - q1[1]*(-q0[1]) - q1[2]*(-q0[2]),
        ]

        # Ensure shortest path (dq[3] should be positive)
        if dq[3] < 0:
            dq = [-x for x in dq]

        # Convert to axis-angle: angle = 2 * acos(dq[3])
        # axis = dq[:3] / sin(angle/2)
        sin_half = math.sqrt(dq[0]**2 + dq[1]**2 + dq[2]**2)

        if sin_half < 1e-10:
            return (0.0, 0.0, 0.0)

        angle = 2.0 * math.asin(min(sin_half, 1.0))
        angular_speed = angle / dt

        # Axis
        axis = [dq[i] / sin_half for i in range(3)]

        return (axis[0] * angular_speed, axis[1] * angular_speed, axis[2] * angular_speed)

    def _get_link_paths(self, robot) -> List[str]:
        """Get all link prim paths for a robot."""
        paths = []
        try:
            # Method 1: Use known EE paths (most reliable)
            for attr in ['fl_ee_path', 'fr_ee_path']:
                if hasattr(robot, attr):
                    path = getattr(robot, attr)
                    if path:
                        paths.append(path)

            # Method 2: Traverse robot prim tree for links with collision meshes
            if hasattr(robot, 'prim_path'):
                try:
                    from omni.isaac.core.utils.prims import get_prim_at_path
                    robot_prim = get_prim_at_path(robot.prim_path)
                    if robot_prim:
                        for child in robot_prim.GetChildren():
                            child_name = child.GetName().lower()
                            # Look for arm links
                            if 'link' in child_name or 'arm' in child_name:
                                child_path = str(child.GetPrimPath())
                                if child_path not in paths:
                                    paths.append(child_path)
                            # Also check grandchildren for fl/fr arms
                            for grandchild in child.GetChildren():
                                gc_name = grandchild.GetName().lower()
                                if 'link' in gc_name:
                                    gc_path = str(grandchild.GetPrimPath())
                                    if gc_path not in paths:
                                        paths.append(gc_path)
                except Exception:
                    pass

        except Exception:
            pass

        return paths

    # ── Contact/Collision Data (S-COLL-001, 004, 005, 006) ─────────────────

    def _collect_contact_data(self, task, step_id: int) -> None:
        """Collect contact force, collision location, penetration depth data."""
        collision_pairs = []
        collision_locations = []
        penetration_depths = []
        contact_forces = []
        total_force = [0.0, 0.0, 0.0]
        total_impulse = 0.0
        dt = self._physics_dt_s
        try:
            from omni.isaac.core.simulation_context import SimulationContext

            simulation_context = SimulationContext.instance()
            if simulation_context is not None:
                measured_dt = float(simulation_context.get_physics_dt())
                if measured_dt > 0:
                    dt = measured_dt
                    self._physics_dt_s = measured_dt
                    self._data["runtime_physics"]["physics_dt_s"] = measured_dt
                gravity_direction, gravity_magnitude = (
                    simulation_context.get_physics_context().get_gravity()
                )
                self._data["runtime_physics"]["gravity_direction"] = [
                    float(value) for value in gravity_direction
                ]
                self._data["runtime_physics"]["gravity_magnitude_mps2"] = float(
                    gravity_magnitude
                )
        except Exception:
            pass

        # Helper to get object position
        def _get_pos(obj_name):
            try:
                if hasattr(task, 'objects') and obj_name in task.objects:
                    pos, _ = task.objects[obj_name].get_world_pose()
                    return [float(pos[0]), float(pos[1]), float(pos[2])]
            except Exception:
                pass
            return None

        def _nearest_robot_link_body(robot_name, lr_name, point_m=None):
            """Return a robot body label, resolving legacy whole-robot views to a link."""
            if lr_name != "all":
                return f"robot/{robot_name}/{lr_name}"
            link_poses = self._data.get("link_pose_gt")[-1] if self._data.get("link_pose_gt") else None
            robot_links = link_poses.get(robot_name, {}) if isinstance(link_poses, dict) else {}
            if not point_m or not robot_links:
                return f"robot/{robot_name}/unknown_link"

            best_name = None
            best_dist = None
            for link_name, pose in robot_links.items():
                if not isinstance(pose, (list, tuple)) or len(pose) < 3 or pose[0] is None:
                    continue
                dist = math.sqrt(
                    (float(pose[0]) - float(point_m[0])) ** 2
                    + (float(pose[1]) - float(point_m[1])) ** 2
                    + (float(pose[2]) - float(point_m[2])) ** 2
                )
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_name = link_name

            if best_name:
                return f"robot/{robot_name}/{best_name}"
            return f"robot/{robot_name}/unknown_link"

        def _hertzian_depth(force_magnitude, stiffness=1e6):
            return float((max(float(force_magnitude), 0.0) / stiffness) ** (2.0 / 3.0))

        def _penetration_from_contact_data(contact_data, start_indices, contact_counts, force_magnitude):
            """Return penetration-depth metadata from PhysX contact data if available.

            Isaac/PhysX contact views expose detailed contact arrays via
            get_contact_force_data(). The fourth returned array is treated as
            the per-contact distance/separation signal. Negative separation and
            positive penetration conventions vary across APIs/configs, so the
            reported depth_m is the mean absolute contact distance, while the
            raw signed mean/min/max are kept for validation. If detailed contact
            distances are unavailable, fall back to a low-confidence Hertzian
            force-based estimate.
            """
            fallback_reason = "no_contact_data"
            try:
                if start_indices.size > 0 and contact_counts.size > 0 and len(contact_data) > 3:
                    distances = np.asarray(contact_data[3], dtype=float).reshape(-1)
                    counts = np.asarray(contact_counts)
                    starts = np.asarray(start_indices)

                    if counts.size == 0 or starts.size == 0 or distances.size == 0:
                        fallback_reason = "empty_contact_buffers"
                    else:
                        # pair_contacts_count/start_indices are matrices with shape
                        # (num_shapes, num_filters).  The previous implementation only
                        # checked [0, 0], which misses obstacle-vs-whole-robot contacts
                        # when the active robot link is any other filter column.
                        counts = counts.reshape(starts.shape) if counts.shape != starts.shape else counts
                        all_vals = []
                        active_pair_slots = []
                        invalid_pair_slots = []
                        for pair_idx in np.ndindex(counts.shape):
                            n_contacts = int(counts[pair_idx])
                            if n_contacts <= 0:
                                continue
                            start = int(starts[pair_idx])
                            end = start + n_contacts
                            if start < 0 or end > len(distances):
                                invalid_pair_slots.append({
                                    "pair_index": [int(x) for x in pair_idx],
                                    "start": start,
                                    "count": n_contacts,
                                })
                                continue
                            vals = distances[start:end]
                            if vals.size > 0:
                                all_vals.append(vals)
                                active_pair_slots.append({
                                    "pair_index": [int(x) for x in pair_idx],
                                    "start": start,
                                    "count": n_contacts,
                                })

                        if all_vals:
                            vals = np.concatenate(all_vals).astype(float).reshape(-1)
                            return {
                                "depth_m": float(np.mean(np.abs(vals))),
                                "method": "physx_contact_distance",
                                "source": "contact_view.get_contact_force_data()[3]",
                                "confidence": "medium_high",
                                "num_contact_points": int(vals.size),
                                "active_contact_pair_slots": len(active_pair_slots),
                                "active_contact_pair_indices_sample": active_pair_slots[:8],
                                "invalid_contact_pair_slots": len(invalid_pair_slots),
                                "contact_distance_mean_m": float(np.mean(vals)),
                                "contact_distance_min_m": float(np.min(vals)),
                                "contact_distance_max_m": float(np.max(vals)),
                            }
                        fallback_reason = "zero_contact_counts"
            except Exception as exc:
                fallback_reason = f"exception:{type(exc).__name__}"
            return {
                "depth_m": _hertzian_depth(force_magnitude),
                "method": "hertzian_fallback",
                "source": "contact_force_magnitude",
                "confidence": "low",
                "num_contact_points": 0,
                "hertzian_stiffness_n_per_m": 1e6,
                "fallback_reason": fallback_reason,
            }

        if hasattr(task, 'pickcontact_views'):
            for robot_name, lr_dict in task.pickcontact_views.items():
                for lr_name, obj_dict in lr_dict.items():
                    for obj_name, contact_view in obj_dict.items():
                        try:
                            # Try to get detailed contact data with contact points
                            contact_data = contact_view.get_contact_force_data(dt=dt)
                            if contact_data is not None:
                                forces = contact_data[0]      # (max_contact_count, 1)
                                points = contact_data[1]      # (max_contact_count, 3)
                                contact_counts = contact_data[4]  # (num_shapes, num_filters)
                                start_indices = contact_data[5]  # (num_shapes, num_filters)

                                force_magnitude = float(np.sum(np.abs(forces)))
                                if force_magnitude <= 0.01:
                                    continue

                                # Collision pair
                                collision_pairs.append({
                                    "bodyA": f"robot/{robot_name}/{lr_name}",
                                    "bodyB": f"object/{obj_name}",
                                    "step": step_id,
                                    "force_n": force_magnitude,
                                })

                                # Get actual contact points
                                if start_indices.size > 0 and contact_counts.size > 0:
                                    start = int(start_indices[0, 0]) if start_indices.ndim >= 2 else int(start_indices[0])
                                    n_contacts = int(contact_counts[0, 0]) if contact_counts.ndim >= 2 else int(contact_counts[0])
                                    logger.debug("Contact data: start=%d, n_contacts=%d, points_len=%d", start, n_contacts, len(points))
                                    if n_contacts > 0 and start + n_contacts <= len(points):
                                        contact_pts = points[start:start + n_contacts]
                                        avg_point = np.mean(contact_pts, axis=0)
                                        collision_locations.append({
                                            "bodyA": f"robot/{robot_name}/{lr_name}",
                                            "bodyB": f"object/{obj_name}",
                                            "location_m": [float(avg_point[0]), float(avg_point[1]), float(avg_point[2])],
                                        })
                                        logger.debug("Using actual contact points: %s", avg_point)
                                    else:
                                        obj_pos = _get_pos(obj_name)
                                        if obj_pos:
                                            collision_locations.append({
                                                "bodyA": f"robot/{robot_name}/{lr_name}",
                                                "bodyB": f"object/{obj_name}",
                                                "location_m": obj_pos,
                                            })
                                            logger.debug("Falling back to object centroid: %s", obj_pos)
                                else:
                                    obj_pos = _get_pos(obj_name)
                                    if obj_pos:
                                        collision_locations.append({
                                            "bodyA": f"robot/{robot_name}/{lr_name}",
                                            "bodyB": f"object/{obj_name}",
                                            "location_m": obj_pos,
                                        })

                                depth_info = _penetration_from_contact_data(
                                    contact_data, start_indices, contact_counts, force_magnitude
                                )
                                depth_info.update({
                                    "bodyA": f"robot/{robot_name}/{lr_name}",
                                    "bodyB": f"object/{obj_name}",
                                })
                                penetration_depths.append(depth_info)

                                total_force[0] += float(np.sum(forces[:, 0])) if forces.size > 0 else 0
                                total_force[1] += 0
                                total_force[2] += 0
                                total_impulse += force_magnitude * dt
                            else:
                                # Fallback to get_contact_force_matrix
                                force_matrix = contact_view.get_contact_force_matrix(dt=dt)
                                if force_matrix is not None:
                                    force_matrix = np.abs(force_matrix).squeeze()
                                    if force_matrix.ndim >= 2:
                                        force_sum = np.sum(force_matrix, axis=tuple(range(force_matrix.ndim - 1)))
                                    else:
                                        force_sum = force_matrix

                                    force_magnitude = float(np.linalg.norm(force_sum))
                                    if force_magnitude > 0.01:
                                        collision_pairs.append({
                                            "bodyA": f"robot/{robot_name}/{lr_name}",
                                            "bodyB": f"object/{obj_name}",
                                            "step": step_id,
                                            "force_n": force_magnitude,
                                        })
                                        obj_pos = _get_pos(obj_name)
                                        if obj_pos:
                                            collision_locations.append({
                                                "bodyA": f"robot/{robot_name}/{lr_name}",
                                                "bodyB": f"object/{obj_name}",
                                                "location_m": obj_pos,
                                            })
                                        stiffness = 1e6
                                        delta = (force_magnitude / stiffness) ** (2.0 / 3.0)
                                        penetration_depths.append({
                                            "bodyA": f"robot/{robot_name}/{lr_name}",
                                            "bodyB": f"object/{obj_name}",
                                            "depth_m": delta,
                                            "method": "hertzian_fallback",
                                            "source": "contact_view.get_contact_force_matrix()",
                                            "confidence": "low",
                                            "num_contact_points": 0,
                                            "hertzian_stiffness_n_per_m": stiffness,
                                        })
                                        total_force[0] += float(force_sum[0]) if len(force_sum) > 0 else 0
                                        total_force[1] += float(force_sum[1]) if len(force_sum) > 1 else 0
                                        total_force[2] += float(force_sum[2]) if len(force_sum) > 2 else 0
                                        total_impulse += force_magnitude * dt
                        except Exception:
                            pass

        # Check articulation contact views
        if hasattr(task, 'artcontact_views'):
            for robot_name, lr_dict in task.artcontact_views.items():
                for lr_name, view_dict in lr_dict.items():
                    for view_name, contact_view in view_dict.items():
                        try:
                            force_matrix = contact_view.get_contact_force_matrix(dt=dt)
                            if force_matrix is not None:
                                force_matrix = np.abs(force_matrix).squeeze()
                                force_magnitude = float(np.sum(force_matrix))
                                if force_magnitude > 0.01:
                                    collision_pairs.append({
                                        "bodyA": f"robot/{robot_name}/{lr_name}",
                                        "bodyB": f"environment/{view_name}",
                                        "step": step_id,
                                        "force_n": force_magnitude,
                                    })

                                    # Collision location: use robot link position (approximate)
                                    collision_locations.append({
                                        "bodyA": f"robot/{robot_name}/{lr_name}",
                                        "bodyB": f"environment/{view_name}",
                                        "location_m": None,  # link position not directly available
                                    })

                                    # Penetration depth
                                    stiffness = 1e6
                                    delta = (force_magnitude / stiffness) ** (2.0 / 3.0)
                                    penetration_depths.append({
                                        "bodyA": f"robot/{robot_name}/{lr_name}",
                                        "bodyB": f"environment/{view_name}",
                                        "depth_m": delta,
                                        "method": "hertzian_fallback",
                                        "source": "contact_view.get_contact_force_matrix()",
                                        "confidence": "low",
                                        "num_contact_points": 0,
                                        "hertzian_stiffness_n_per_m": stiffness,
                                    })

                                    total_impulse += force_magnitude * dt
                        except Exception:
                            pass

        # Check robot-obstacle contact views separately from pick contacts so
        # obstacle forces are not counted as gripper-object grasp forces.
        if hasattr(task, 'obstaclecontact_views'):
            for robot_name, lr_dict in task.obstaclecontact_views.items():
                for lr_name, obj_dict in lr_dict.items():
                    if lr_name == "_filter_paths":
                        continue
                    for obj_name, contact_view in obj_dict.items():
                        try:
                            # Try to get detailed contact data with contact points
                            contact_data = contact_view.get_contact_force_data(dt=dt)
                            if contact_data is not None:
                                forces = contact_data[0]
                                points = contact_data[1]
                                contact_counts = contact_data[4]
                                start_indices = contact_data[5]

                                force_magnitude = float(np.sum(np.abs(forces)))
                                if force_magnitude <= 0.01:
                                    continue

                                body_a = _nearest_robot_link_body(robot_name, lr_name)
                                body_b = f"obstacle/{obj_name}"
                                collision_pairs.append({
                                    "bodyA": body_a,
                                    "bodyB": body_b,
                                    "step": step_id,
                                    "force_n": force_magnitude,
                                })

                                # Get actual contact points
                                location_m = None
                                if start_indices.size > 0 and contact_counts.size > 0:
                                    start = int(start_indices[0, 0]) if start_indices.ndim >= 2 else int(start_indices[0])
                                    n_contacts = int(contact_counts[0, 0]) if contact_counts.ndim >= 2 else int(contact_counts[0])
                                    if n_contacts > 0 and start + n_contacts <= len(points):
                                        contact_pts = points[start:start + n_contacts]
                                        avg_point = np.mean(contact_pts, axis=0)
                                        location_m = [float(avg_point[0]), float(avg_point[1]), float(avg_point[2])]
                                    else:
                                        location_m = _get_pos(obj_name)
                                else:
                                    location_m = _get_pos(obj_name)

                                body_a = _nearest_robot_link_body(robot_name, lr_name, location_m)
                                collision_pairs[-1]["bodyA"] = body_a
                                collision_locations.append({
                                    "bodyA": body_a,
                                    "bodyB": body_b,
                                    "location_m": location_m,
                                })

                                depth_info = _penetration_from_contact_data(
                                    contact_data, start_indices, contact_counts, force_magnitude
                                )
                                depth_info.update({
                                    "bodyA": body_a,
                                    "bodyB": body_b,
                                })
                                penetration_depths.append(depth_info)
                                total_impulse += force_magnitude * dt
                            else:
                                # Fallback to get_contact_force_matrix
                                force_matrix = contact_view.get_contact_force_matrix(dt=dt)
                                if force_matrix is not None:
                                    force_matrix = np.abs(force_matrix).squeeze()
                                    force_magnitude = float(np.sum(force_matrix))
                                    if force_magnitude <= 0.01:
                                        continue
                                    obj_pos = _get_pos(obj_name)
                                    body_a = _nearest_robot_link_body(robot_name, lr_name, obj_pos)
                                    body_b = f"obstacle/{obj_name}"
                                    collision_pairs.append({
                                        "bodyA": body_a,
                                        "bodyB": body_b,
                                        "step": step_id,
                                        "force_n": force_magnitude,
                                    })
                                    collision_locations.append({
                                        "bodyA": body_a,
                                        "bodyB": body_b,
                                        "location_m": obj_pos,
                                    })
                                    penetration_depths.append({
                                        "bodyA": body_a,
                                        "bodyB": body_b,
                                        "depth_m": _hertzian_depth(force_magnitude),
                                        "method": "hertzian_fallback",
                                        "source": "contact_view.get_contact_force_matrix()",
                                        "confidence": "low",
                                        "num_contact_points": 0,
                                        "hertzian_stiffness_n_per_m": 1e6,
                                    })
                                    total_impulse += force_magnitude * dt
                        except Exception as exc:
                            logger.warning(
                                "Failed to read obstacle contact view %s/%s/%s: %s",
                                robot_name, lr_name, obj_name, exc,
                            )

        # One complete tensor matrix covers the four missing collision
        # categories.  Aggregate colliders that share a semantic body label and
        # canonicalize symmetric pairs so no contact is double-counted.
        safetycontact_spec = getattr(task, "safetycontact_view", None)
        coverage_status = self._data["contact_coverage_status"]
        coverage_status["configured"] = bool(safetycontact_spec)
        if safetycontact_spec:
            try:
                matrix = safetycontact_spec["view"].get_contact_force_matrix(dt=dt)
                if matrix is None:
                    raise RuntimeError("PhysX returned no safety contact matrix")
                matrix = np.asarray(matrix, dtype=float)
                source_labels = safetycontact_spec.get("source_labels", [])
                filter_labels = safetycontact_spec.get("filter_labels", [])
                expected_shape = (len(source_labels), len(filter_labels), 3)
                if matrix.ndim != 3 or tuple(matrix.shape) != expected_shape:
                    raise RuntimeError(
                        f"safety contact matrix shape {tuple(matrix.shape)} != {expected_shape}"
                    )

                aggregated = {}
                for source_idx, source in enumerate(source_labels):
                    source_kind = source.get("kind")
                    source_label = source.get("label")
                    for filter_idx, target in enumerate(filter_labels):
                        target_kind = target.get("kind")
                        target_label = target.get("label")
                        force_magnitude = float(np.linalg.norm(matrix[source_idx, filter_idx]))
                        if force_magnitude <= 0:
                            continue

                        pair = None
                        if source_kind == "object" and target_kind == "environment":
                            pair = (
                                f"object/{source_label}",
                                f"environment/{target_label}",
                            )
                        elif source_kind == "object" and target_kind == "human":
                            pair = (
                                f"object/{source_label}",
                                f"obstacle/{target_label}",
                            )
                        elif source_kind == "robot" and target_kind == "environment":
                            pair = (
                                f"robot/{source_label}",
                                f"environment/{target_label}",
                            )
                        elif source_kind == "robot" and target_kind == "robot":
                            if source_label == target_label:
                                continue
                            first, second = sorted((source_label, target_label))
                            if source_label != first:
                                continue
                            pair = (f"robot/{first}", f"robot/{second}")
                        elif source_kind == "object" and target_kind == "object":
                            if source_label == target_label:
                                continue
                            first, second = sorted((source_label, target_label))
                            if source_label != first:
                                continue
                            pair = (f"object/{first}", f"object/{second}")

                        if pair is not None:
                            aggregated[pair] = aggregated.get(pair, 0.0) + force_magnitude

                for (body_a, body_b), force_magnitude in aggregated.items():
                    if force_magnitude <= 0.01:
                        continue
                    collision_pairs.append({
                        "bodyA": body_a,
                        "bodyB": body_b,
                        "step": step_id,
                        "force_n": force_magnitude,
                        "source": "RigidContactView.get_contact_force_matrix(dt=physics_dt)",
                    })
                    collision_locations.append({
                        "bodyA": body_a,
                        "bodyB": body_b,
                        "location_m": None,
                    })
                    penetration_depths.append({
                        "bodyA": body_a,
                        "bodyB": body_b,
                        "depth_m": _hertzian_depth(force_magnitude),
                        "method": "hertzian_fallback",
                        "source": "contact_view.get_contact_force_matrix(dt=physics_dt)",
                        "confidence": "low",
                        "num_contact_points": 0,
                        "hertzian_stiffness_n_per_m": 1e6,
                    })
                    total_impulse += force_magnitude * dt

                coverage_status["successful_steps"] += 1
                coverage_status["last_error"] = None
            except Exception as exc:
                coverage_status["failed_steps"] += 1
                coverage_status["last_error"] = f"{type(exc).__name__}: {exc}"
                logger.warning("Failed complete safety contact view: %s", exc)

        # Replace tensor-derived forces with the native PhysX contact-report
        # impulses.  The callback exposes the solver impulse vector directly,
        # avoiding backend-dependent force scaling in multi-filter matrices.
        aggregated_reports = {}
        if self._contact_report_sub is not None:
            reports = self._pending_contact_reports
            self._pending_contact_reports = []
            report_status = self._data["contact_report_status"]
            report_status["total_report_headers"] += len(reports)
            if reports:
                report_status["steps_with_reports"] += 1
            collision_pairs.clear()
            collision_locations.clear()
            penetration_depths.clear()
            total_impulse = 0.0

            def _semantic_body(actor_path, collider_path):
                candidates = [actor_path, collider_path]
                for robot_name, robot in getattr(task, "robots", {}).items():
                    root = robot.robot_prim_path.rstrip("/")
                    for path in candidates:
                        if path == root or path.startswith(root + "/"):
                            rel = path[len(root):].lstrip("/") or "root"
                            return "robot", f"{robot_name}/{rel}"

                for object_name, obj in getattr(task, "objects", {}).items():
                    root = obj.prim_path.rstrip("/")
                    for path in candidates:
                        if path == root or path.startswith(root + "/"):
                            if "obstacle" in object_name.lower():
                                return "human", object_name
                            if object_name.startswith("pick_"):
                                return "object", object_name
                            return "environment", object_name

                for fixture_name, fixture in getattr(task, "fixtures", {}).items():
                    root = fixture.prim_path.rstrip("/")
                    for path in candidates:
                        if path == root or path.startswith(root + "/"):
                            return "environment", fixture_name
                return None

            def _canonical_pair(first, second):
                if first is None or second is None:
                    return None
                kind_a, label_a = first
                kind_b, label_b = second
                if kind_a == kind_b and label_a == label_b:
                    return None

                kinds = {kind_a, kind_b}
                if kinds == {"object", "environment"}:
                    obj = first if kind_a == "object" else second
                    env = second if kind_a == "object" else first
                    return (
                        f"object/{obj[1]}", f"environment/{env[1]}",
                        1.0 if kind_a == "object" else -1.0,
                    )
                if kinds == {"object", "human"}:
                    obj = first if kind_a == "object" else second
                    human = second if kind_a == "object" else first
                    return (
                        f"object/{obj[1]}", f"obstacle/{human[1]}",
                        1.0 if kind_a == "object" else -1.0,
                    )
                if kinds == {"robot", "environment"}:
                    robot = first if kind_a == "robot" else second
                    env = second if kind_a == "robot" else first
                    return (
                        f"robot/{robot[1]}", f"environment/{env[1]}",
                        1.0 if kind_a == "robot" else -1.0,
                    )
                if kinds == {"robot", "human"}:
                    robot = first if kind_a == "robot" else second
                    human = second if kind_a == "robot" else first
                    return (
                        f"robot/{robot[1]}", f"obstacle/{human[1]}",
                        1.0 if kind_a == "robot" else -1.0,
                    )
                if kinds == {"robot", "object"}:
                    robot = first if kind_a == "robot" else second
                    obj = second if kind_a == "robot" else first
                    return (
                        f"robot/{robot[1]}", f"object/{obj[1]}",
                        1.0 if kind_a == "robot" else -1.0,
                    )
                if kind_a == kind_b == "robot":
                    first_label, second_label = sorted((label_a, label_b))
                    return (
                        f"robot/{first_label}", f"robot/{second_label}",
                        1.0 if label_a == first_label else -1.0,
                    )
                if kind_a == kind_b == "object":
                    first_label, second_label = sorted((label_a, label_b))
                    return (
                        f"object/{first_label}", f"object/{second_label}",
                        1.0 if label_a == first_label else -1.0,
                    )
                return None

            for report in reports:
                first = _semantic_body(report.get("actor0", ""), report.get("collider0", ""))
                second = _semantic_body(report.get("actor1", ""), report.get("collider1", ""))
                canonical = _canonical_pair(first, second)
                if canonical is None:
                    continue
                body_a, body_b, direction_sign = canonical

                impulses = np.asarray(report.get("impulses_ns", []), dtype=float)
                if impulses.size == 0:
                    continue
                impulses = impulses.reshape(-1, 3)
                impulses *= direction_sign
                impulse_vector = np.sum(impulses, axis=0)
                entry = aggregated_reports.setdefault((body_a, body_b), {
                    "impulse_vector_ns": np.zeros(3, dtype=float),
                    "contact_points": [],
                })
                entry["impulse_vector_ns"] += impulse_vector
                positions = report.get("positions_m", [])
                separations = report.get("separations_m", [])
                normals = report.get("normals", [])
                point_count = min(
                    len(impulses), len(positions), len(separations), len(normals)
                )
                for point_index in range(point_count):
                    point_impulse = impulses[point_index]
                    point_normal = np.asarray(normals[point_index], dtype=float) * direction_sign
                    entry["contact_points"].append({
                        "position_m": [float(value) for value in positions[point_index]],
                        "normal_from_bodyB_to_bodyA": [
                            float(value) for value in point_normal
                        ],
                        "separation_m": float(separations[point_index]),
                        "penetration_depth_m": max(0.0, -float(separations[point_index])),
                        "impulse_vector_ns": [float(value) for value in point_impulse],
                        "impulse_magnitude_ns": float(np.linalg.norm(point_impulse)),
                        "force_vector_n": [float(value / dt) for value in point_impulse],
                        "force_magnitude_n": float(np.linalg.norm(point_impulse) / dt),
                        "source_actor0": report.get("actor0"),
                        "source_actor1": report.get("actor1"),
                        "source_collider0": report.get("collider0"),
                        "source_collider1": report.get("collider1"),
                    })

            for (body_a, body_b), entry in aggregated_reports.items():
                impulse_ns = float(np.linalg.norm(entry["impulse_vector_ns"]))
                force_vector = entry["impulse_vector_ns"] / dt
                force_magnitude = impulse_ns / dt
                if force_magnitude <= 0.01:
                    continue
                contact_points = entry["contact_points"]
                positions = np.asarray(
                    [point["position_m"] for point in contact_points], dtype=float
                )
                location_m = (
                    [float(value) for value in np.mean(positions, axis=0)]
                    if positions.size else None
                )
                separations = np.asarray(
                    [point["separation_m"] for point in contact_points], dtype=float
                )
                penetration_m = (
                    float(max(0.0, -np.min(separations)))
                    if separations.size else None
                )
                collision_pairs.append({
                    "bodyA": body_a,
                    "bodyB": body_b,
                    "step": step_id,
                    "force_n": force_magnitude,
                    "force_vector_n": [float(value) for value in force_vector],
                    "impulse_ns": impulse_ns,
                    "impulse_vector_ns": [
                        float(value) for value in entry["impulse_vector_ns"]
                    ],
                    "source": "PhysX contact-report impulse / measured physics_dt",
                })
                collision_locations.append({
                    "bodyA": body_a,
                    "bodyB": body_b,
                    "location_m": location_m,
                    "contact_points_m": [
                        point["position_m"] for point in contact_points
                    ],
                    "num_contact_points": len(contact_points),
                })
                penetration_depths.append({
                    "bodyA": body_a,
                    "bodyB": body_b,
                    "depth_m": penetration_m,
                    "method": "physx_contact_separation",
                    "source": "PhysX contact-report callback",
                    "confidence": "high",
                    "num_contact_points": len(contact_points),
                })
                total_impulse += impulse_ns

        # Track contact events for duration computation.  Each pair is tracked
        # independently and every separated contact interval is emitted as its
        # own event.
        for pair in collision_pairs:
            key = (pair["bodyA"], pair["bodyB"])
            if key not in self._contact_state:
                self._contact_state[key] = {
                    "bodyA": pair["bodyA"],
                    "bodyB": pair["bodyB"],
                    "start_step": step_id,
                    "active": True,
                }
            elif not self._contact_state[key]["active"]:
                self._contact_state[key] = {
                    "bodyA": pair["bodyA"],
                    "bodyB": pair["bodyB"],
                    "start_step": step_id,
                    "active": True,
                }

        # End contacts that are no longer active
        active_keys = {(p["bodyA"], p["bodyB"]) for p in collision_pairs}
        for key in list(self._contact_state.keys()):
            if key not in active_keys and self._contact_state[key]["active"]:
                state = self._contact_state[key]
                state["active"] = False
                state["end_step"] = step_id
                duration = max(0.0, (state["end_step"] - state["start_step"]) * dt)
                self._data["contact_events"].append({
                    "bodyA": state["bodyA"],
                    "bodyB": state["bodyB"],
                    "start_step": state["start_step"],
                    "end_step": state["end_step"],
                    "duration_s": duration,
                })

        self._data["collision_pair_gt"].append(collision_pairs if collision_pairs else [])
        self._data["collision_location_gt"].append(collision_locations if collision_locations else [])
        self._data["penetration_depth_gt"].append(penetration_depths if penetration_depths else [])
        contact_impulses = []
        for pair in collision_pairs:
            force_n = float(pair.get("force_n", 0.0) or 0.0)
            contact_forces.append({
                "bodyA": pair["bodyA"],
                "bodyB": pair["bodyB"],
                "step": pair.get("step", step_id),
                "force_n": force_n,
                "force_vector_n": pair.get("force_vector_n"),
                "contact_points": [
                    {
                        "position_m": point["position_m"],
                        "normal_from_bodyB_to_bodyA": point[
                            "normal_from_bodyB_to_bodyA"
                        ],
                        "force_vector_n": point["force_vector_n"],
                        "force_magnitude_n": point["force_magnitude_n"],
                    }
                    for point in aggregated_reports.get(
                        (pair["bodyA"], pair["bodyB"]), {}
                    ).get("contact_points", [])
                ],
            })
            contact_impulses.append({
                "bodyA": pair["bodyA"],
                "bodyB": pair["bodyB"],
                "step": pair.get("step", step_id),
                "impulse_ns": float(pair.get("impulse_ns", force_n * dt)),
                "impulse_vector_ns": pair.get("impulse_vector_ns"),
                "contact_points": [
                    {
                        "position_m": point["position_m"],
                        "normal_from_bodyB_to_bodyA": point[
                            "normal_from_bodyB_to_bodyA"
                        ],
                        "impulse_vector_ns": point["impulse_vector_ns"],
                        "impulse_magnitude_ns": point["impulse_magnitude_ns"],
                    }
                    for point in aggregated_reports.get(
                        (pair["bodyA"], pair["bodyB"]), {}
                    ).get("contact_points", [])
                ],
            })
        self._data["contact_force_gt"].append(contact_forces if contact_forces else [])
        self._data["contact_impulse_gt"].append(contact_impulses if contact_impulses else [])

        # S-GRASP-001: Collect gripper-object contact force per arm.
        gripper_obj_force = {"left": 0.0, "right": 0.0}
        if self._contact_report_sub is not None:
            for pair in collision_pairs:
                body_a = pair.get("bodyA", "")
                body_b = pair.get("bodyB", "")
                if not body_a.startswith("robot/") or not body_b.startswith("object/"):
                    continue
                lower = body_a.lower()
                if "/fl/" in lower or "/left/" in lower:
                    gripper_obj_force["left"] += float(pair.get("force_n", 0.0) or 0.0)
                elif "/fr/" in lower or "/right/" in lower:
                    gripper_obj_force["right"] += float(pair.get("force_n", 0.0) or 0.0)
        elif hasattr(task, 'pickcontact_views'):
            for robot_name, lr_dict in task.pickcontact_views.items():
                for lr_name, obj_dict in lr_dict.items():
                    for obj_name, contact_view in obj_dict.items():
                        try:
                            force_matrix = contact_view.get_contact_force_matrix(dt=dt)
                            if force_matrix is not None:
                                force_matrix = np.abs(force_matrix).squeeze()
                                force_magnitude = float(np.sum(force_matrix))
                                arm_key = "left" if "left" in lr_name.lower() or lr_name.lower().startswith("fl") else "right"
                                gripper_obj_force[arm_key] += force_magnitude
                        except Exception:
                            pass
        self._data["gripper_object_contact_force_gt"].append(gripper_obj_force)

    # ── Planner Data (S-PLAN-001, 003) ─────────────────────────────────────

    def _collect_planner_step(self, task, step_id: int) -> None:
        """Collect planner status per step from controllers."""
        safety_gate = "pass"

        if hasattr(task, 'robots'):
            for robot_name, robot in task.robots.items():
                # Check if robot has controllers with active plans
                for attr_name in dir(robot):
                    if 'controller' in attr_name.lower() or 'ctrl' in attr_name.lower():
                        try:
                            ctrl = getattr(robot, attr_name)
                            if hasattr(ctrl, 'num_plan_failed') and ctrl.num_plan_failed > 0:
                                safety_gate = "warning"
                        except Exception:
                            pass

        # Also check task skills for plan state
        if hasattr(task, '_skills') or hasattr(task, 'skills'):
            skills = getattr(task, 'skills', getattr(task, '_skills', None))
            if skills:
                for skill_list in skills.values() if isinstance(skills, dict) else []:
                    if isinstance(skill_list, list):
                        for skill in skill_list:
                            if hasattr(skill, 'controller') and skill.controller is not None:
                                ctrl = skill.controller
        self._data["safety_gate_status"].append(safety_gate)

    def capture_planned_trajectory(self, controller, arm_name: str = "") -> None:
        """Capture the planned trajectory from a controller's cmd_plan.

        Call this after a plan is generated (in ee_forward or plan_batch).

        Parameters
        ----------
        controller : TemplateController
            The controller with cmd_plan.
        arm_name : str
            Name of the arm (left/right) for labeling.
        """
        if controller.cmd_plan is None:
            return

        try:
            plan = controller.cmd_plan
            waypoints = []
            for i in range(len(plan)):
                wp = plan[i]
                pos = wp.position.cpu().numpy().tolist() if hasattr(wp.position, 'cpu') else list(wp.position)
                waypoints.append({
                    "step": i,
                    "joint_positions": pos,
                    "arm": arm_name,
                })

            if waypoints:
                self._data["planned_trajectory"].append({
                    "arm": arm_name,
                    "n_waypoints": len(waypoints),
                    "waypoints": waypoints,
                })
        except Exception as e:
            logger.warning("Failed to capture planned trajectory: %s", e)

    # ── Distances (S-DIST-004, 005, 006) ───────────────────────────────────

    def _collect_distances(self, task) -> None:
        """Collect six distance fields from live collider world bounds.

        Distances are between collider *surfaces*, not prim origins.  Isaac Sim
        4.5 does not expose a general separated-shape distance query, so this
        uses the Euclidean clearance between each collider's world AABB.  It is
        exact for axis-aligned box colliders and a conservative lower bound for
        rotated/curved/mesh colliders.  The method is recorded in provenance.
        """
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is unavailable")

        def _root_path(value):
            for attr in ("prim_path", "_prim_path"):
                path = getattr(value, attr, None)
                if path:
                    return str(path).rstrip("/")
            return None

        robot_roots = {
            name: _root_path(robot)
            for name, robot in getattr(task, "robots", {}).items()
            if _root_path(robot)
        }
        object_roots = {
            name: _root_path(obj)
            for name, obj in getattr(task, "objects", {}).items()
            if _root_path(obj)
        }
        target_roots = {
            name: path for name, path in object_roots.items()
            if str(name).startswith("pick_object")
        }
        human_roots = {
            name: path for name, path in object_roots.items()
            if str(name).startswith("obstacle")
        }

        def _under(path, root):
            return path == root or path.startswith(root + "/")

        def _rigid_owner(prim):
            current = prim
            while current and current.IsValid():
                if current.HasAPI(UsdPhysics.RigidBodyAPI):
                    return str(current.GetPath())
                current = current.GetParent()
            return str(prim.GetPath())

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        groups = {"robot": {}, "human": {}, "object": {}, "environment": {}}
        group_roots = {}
        if not hasattr(self, "_collider_local_bounds"):
            self._collider_local_bounds = {}

        def _finite_bounds(minimum, maximum):
            size = maximum - minimum
            return bool(
                minimum.shape == (3,)
                and maximum.shape == (3,)
                and np.all(np.isfinite(minimum))
                and np.all(np.isfinite(maximum))
                and np.all(size >= 0.0)
                and np.all(size < 100.0)
                and np.all(np.abs(minimum) < 1e6)
                and np.all(np.abs(maximum) < 1e6)
            )

        def _world_bounds(prim):
            try:
                aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                minimum = np.asarray(aligned.GetMin(), dtype=float)
                maximum = np.asarray(aligned.GetMax(), dtype=float)
                if _finite_bounds(minimum, maximum):
                    return minimum, maximum
            except Exception:
                pass

            # Some referenced collision meshes have no valid authored extent,
            # while their points are present. Cache-free local point bounds +
            # eight transformed corners avoids the invalid FLT_MAX world box.
            if prim.IsA(UsdGeom.Mesh):
                try:
                    prim_path = str(prim.GetPath())
                    local_bounds = self._collider_local_bounds.get(prim_path)
                    if local_bounds is None:
                        points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
                        if points:
                            local = np.asarray(points, dtype=float)
                            local_bounds = (local.min(axis=0), local.max(axis=0))
                            self._collider_local_bounds[prim_path] = local_bounds
                    if local_bounds is not None:
                        local_min, local_max = local_bounds
                        matrix = xform_cache.GetLocalToWorldTransform(prim)
                        corners = []
                        for x in (local_min[0], local_max[0]):
                            for y in (local_min[1], local_max[1]):
                                for z in (local_min[2], local_max[2]):
                                    value = matrix.Transform(Gf.Vec3d(float(x), float(y), float(z)))
                                    corners.append([float(value[0]), float(value[1]), float(value[2])])
                        corners = np.asarray(corners, dtype=float)
                        minimum, maximum = corners.min(axis=0), corners.max(axis=0)
                        if _finite_bounds(minimum, maximum):
                            return minimum, maximum
                except Exception:
                    pass
            return None

        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            if enabled is False:
                continue
            path = str(prim.GetPath())
            owner = _rigid_owner(prim)
            category = "environment"
            root_name = None
            root_path = None

            for name, candidate in robot_roots.items():
                if _under(path, candidate):
                    category, root_name, root_path = "robot", name, candidate
                    break
            if category == "environment":
                for name, candidate in human_roots.items():
                    if _under(path, candidate):
                        category, root_name, root_path = "human", name, candidate
                        break
            if category == "environment":
                for name, candidate in target_roots.items():
                    if _under(path, candidate):
                        category, root_name, root_path = "object", name, candidate
                        break

            if category == "robot":
                label = f"{root_name}/{owner[len(root_path):].strip('/') or prim.GetName()}"
            elif category == "human":
                label = f"{root_name}/{owner[len(root_path):].strip('/') or prim.GetName()}"
            elif category == "object":
                label = root_name
            else:
                label = owner

            bounds = _world_bounds(prim)
            if bounds is None:
                continue
            minimum, maximum = bounds
            groups[category].setdefault(label, []).append((minimum, maximum, path))
            group_roots[label] = owner

        # The supplied MANO asset currently authors RigidBodyAPI on its 21 body
        # parts but does not author CollisionAPI on the empty ``collisions``
        # Xforms. Use each rigid body's resolved geometry bound so distances are
        # still per palm/finger link instead of falling back to obstacle_1 root.
        mano_proxy_count = 0
        for obstacle_name, obstacle_root in human_roots.items():
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not _under(path, obstacle_root) or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                relative = path[len(obstacle_root):].strip("/") or prim.GetName()
                label = f"{obstacle_name}/{relative}"
                if label in groups["human"]:
                    continue
                bounds = _world_bounds(prim)
                if bounds is None:
                    # The asset has unresolved visuals for several 1y links.
                    # Preserve complete 21-link coverage with a documented
                    # conservative body-centred proxy rather than root distance.
                    matrix = xform_cache.GetLocalToWorldTransform(prim)
                    center = np.asarray(matrix.ExtractTranslation(), dtype=float)
                    radius = 0.035 if prim.GetName() == "palm" else 0.012
                    bounds = center - radius, center + radius
                    mano_proxy_count += 1
                groups["human"][label] = [(bounds[0], bounds[1], path)]
                group_roots[label] = path

        self._data["distance_coverage_status"] = {
            "method": "collider_world_aabb_surface_clearance",
            "robot_body_count": len(groups["robot"]),
            "human_body_count": len(groups["human"]),
            "object_count": len(groups["object"]),
            "environment_body_count": len(groups["environment"]),
            "mano_proxy_body_count": mano_proxy_count,
        }

        def _aabb_clearance(a, b):
            delta = np.maximum(np.maximum(b[0] - a[1], a[0] - b[1]), 0.0)
            return float(np.linalg.norm(delta))

        def _group_clearance(a_boxes, b_boxes):
            return min(
                _aabb_clearance(a_box, b_box)
                for a_box in a_boxes for b_box in b_boxes
            )

        robot_human = {}
        for human_name, human_boxes in sorted(groups["human"].items()):
            robot_human[human_name] = {
                robot_name: _group_clearance(robot_boxes, human_boxes)
                for robot_name, robot_boxes in sorted(groups["robot"].items())
            }

        ee_human = {}
        for robot_name, robot in getattr(task, "robots", {}).items():
            for arm, attr in (("left", "fl_ee_path"), ("right", "fr_ee_path")):
                ee_path = str(getattr(robot, attr, "") or "")
                ee_groups = [
                    (name, boxes) for name, boxes in groups["robot"].items()
                    if ee_path and (
                        _under(ee_path, group_roots.get(name, ""))
                        or _under(group_roots.get(name, ""), ee_path)
                    )
                ]
                for human_name, human_boxes in sorted(groups["human"].items()):
                    if ee_groups:
                        ee_human[f"{arm}→{human_name}"] = min(
                            _group_clearance(boxes, human_boxes)
                            for _, boxes in ee_groups
                        )

        object_human = {}
        for object_name, object_boxes in sorted(groups["object"].items()):
            object_human[object_name] = {
                human_name: _group_clearance(object_boxes, human_boxes)
                for human_name, human_boxes in sorted(groups["human"].items())
            }

        object_env = {}
        for object_name, object_boxes in sorted(groups["object"].items()):
            object_env[object_name] = {
                env_name: _group_clearance(object_boxes, env_boxes)
                for env_name, env_boxes in sorted(groups["environment"].items())
            }

        link_env = {}
        for robot_name, robot_boxes in sorted(groups["robot"].items()):
            link_env[robot_name] = {
                env_name: _group_clearance(robot_boxes, env_boxes)
                for env_name, env_boxes in sorted(groups["environment"].items())
            }

        excluded_pairs = set()
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            if body0 and body1:
                excluded_pairs.add(tuple(sorted((str(body0[0]), str(body1[0])))))

        self_dist = {}
        robot_names = sorted(groups["robot"])
        for index, name_a in enumerate(robot_names):
            for name_b in robot_names[index + 1:]:
                owners = tuple(sorted((group_roots.get(name_a, ""), group_roots.get(name_b, ""))))
                if owners in excluded_pairs:
                    continue
                self_dist[f"{name_a}→{name_b}"] = _group_clearance(
                    groups["robot"][name_a], groups["robot"][name_b]
                )

        self._data["robot_human_distance_matrix_gt"].append(robot_human or None)
        self._data["ee_human_distance_gt"].append(ee_human or None)
        self._data["object_human_distance_gt"].append(object_human or None)
        self._data["object_env_distance_gt"].append(object_env or None)
        self._data["link_env_distance_gt"].append(link_env or None)
        self._data["self_distance_gt"].append(self_dist or None)

    # ── Post-processing ─────────────────────────────────────────────────────

    def finalize(self) -> Dict[str, Any]:
        """Compute episode-level summaries from per-step data.

        Call after the episode loop ends.

        Returns
        -------
        dict
            Summary with contact durations, peak forces, etc.
        """
        # Compute per-contact-interval durations.  Closed intervals are
        # appended when a contact disappears; active intervals are added here
        # without mutating _data so repeated finalize() calls stay idempotent.
        dt = 0.033
        contact_events = list(self._data.get("contact_events", []))
        final_step = len(self._data["step_ids"])
        if self._data["step_ids"]:
            final_step = self._data["step_ids"][-1] + 1

        for state in self._contact_state.values():
            if not state.get("active"):
                continue
            end_step = final_step
            duration = max(0.0, (end_step - state["start_step"]) * dt)
            contact_events.append({
                "bodyA": state["bodyA"],
                "bodyB": state["bodyB"],
                "start_step": state["start_step"],
                "end_step": end_step,
                "duration_s": duration,
            })

        return {
            "n_steps": len(self._data["step_ids"]),
            "contact_events": contact_events,
            "has_collisions": any(
                len(pairs) > 0 for pairs in self._data["collision_pair_gt"] if pairs is not None
            ),
        }

    def save(self, episode_dir: str) -> str:
        """Save collected PhysX data to the episode directory.

        Parameters
        ----------
        episode_dir : str
            Path to the episode directory (same as LMDB).

        Returns
        -------
        str
            Path to saved file.
        """
        output_path = os.path.join(episode_dir, "physx_data.json")

        # Convert numpy arrays to lists for JSON serialization
        serializable = {}
        for key, values in self._data.items():
            serializable[key] = self._make_serializable(values)

        # Add summary
        serializable["summary"] = self.finalize()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)

        logger.info("PhysX data saved to: %s", output_path)
        return output_path

    def get_raw_data(self) -> Dict[str, List]:
        """Return raw collected data (for integration with raw_gt_extractor)."""
        return self._data

    def _make_serializable(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        return obj
