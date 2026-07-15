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
            # S-PLAN-004: low level command sent per step
            "low_level_command_sent": [],
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
            self._collect_contact_data(task, step_id)
        except Exception as e:
            self._data["collision_pair_gt"].append(None)
            self._data["contact_force_gt"].append(None)
            self._data["contact_impulse_gt"].append(None)

        try:
            self._collect_distances(task)
        except Exception as e:
            self._data["object_env_distance_gt"].append(None)
            self._data["link_env_distance_gt"].append(None)
            self._data["self_distance_gt"].append(None)

        # Planner data (from controllers)
        try:
            self._collect_planner_step(task, step_id)
        except Exception:
            self._data["safety_gate_status"].append(None)
            self._data["low_level_command_sent"].append(None)

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
            print(f"[safety_gate] step {step_id}: d_min={d_min_m*100:.1f}cm ({closest_link}→{closest_obj}) threshold={self.SAFETY_DISTANCE_M*100:.0f}cm")

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
                f"d={d_min_m*100:.1f}cm, TTC={ttc_s:.2f}s"
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
        dt = 0.033

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
        dt = 0.033

        # Helper to get object position
        def _get_pos(obj_name):
            try:
                if hasattr(task, 'objects') and obj_name in task.objects:
                    pos, _ = task.objects[obj_name].get_world_pose()
                    return [float(pos[0]), float(pos[1]), float(pos[2])]
            except Exception:
                pass
            return None

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
                            contact_data = contact_view.get_contact_force_data()
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
                                force_matrix = contact_view.get_contact_force_matrix()
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
                            force_matrix = contact_view.get_contact_force_matrix()
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
                            contact_data = contact_view.get_contact_force_data()
                            if contact_data is not None:
                                forces = contact_data[0]
                                points = contact_data[1]
                                contact_counts = contact_data[4]
                                start_indices = contact_data[5]

                                force_magnitude = float(np.sum(np.abs(forces)))
                                if force_magnitude <= 0.01:
                                    continue

                                body_a = f"robot/{robot_name}/{lr_name}"
                                body_b = f"obstacle/{obj_name}"
                                collision_pairs.append({
                                    "bodyA": body_a,
                                    "bodyB": body_b,
                                    "step": step_id,
                                    "force_n": force_magnitude,
                                })

                                # Get actual contact points
                                if start_indices.size > 0 and contact_counts.size > 0:
                                    start = int(start_indices[0, 0]) if start_indices.ndim >= 2 else int(start_indices[0])
                                    n_contacts = int(contact_counts[0, 0]) if contact_counts.ndim >= 2 else int(contact_counts[0])
                                    if n_contacts > 0 and start + n_contacts <= len(points):
                                        contact_pts = points[start:start + n_contacts]
                                        avg_point = np.mean(contact_pts, axis=0)
                                        collision_locations.append({
                                            "bodyA": body_a,
                                            "bodyB": body_b,
                                            "location_m": [float(avg_point[0]), float(avg_point[1]), float(avg_point[2])],
                                        })
                                    else:
                                        obj_pos = _get_pos(obj_name)
                                        collision_locations.append({
                                            "bodyA": body_a,
                                            "bodyB": body_b,
                                            "location_m": obj_pos,
                                        })
                                else:
                                    obj_pos = _get_pos(obj_name)
                                    collision_locations.append({
                                        "bodyA": body_a,
                                        "bodyB": body_b,
                                        "location_m": obj_pos,
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
                                force_matrix = contact_view.get_contact_force_matrix()
                                if force_matrix is not None:
                                    force_matrix = np.abs(force_matrix).squeeze()
                                    force_magnitude = float(np.sum(force_matrix))
                                    if force_magnitude <= 0.01:
                                        continue
                                    body_a = f"robot/{robot_name}/{lr_name}"
                                    body_b = f"obstacle/{obj_name}"
                                    collision_pairs.append({
                                        "bodyA": body_a,
                                        "bodyB": body_b,
                                        "step": step_id,
                                        "force_n": force_magnitude,
                                    })
                                    obj_pos = _get_pos(obj_name)
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

        # Targeted robot self-collision views: left arm links against right arm
        # links.  These are intentionally narrower than a scene-wide contact
        # matrix so contact recording does not perturb the rest of the pipeline.
        if hasattr(task, 'robotselfcontact_views'):
            for robot_name, source_dict in task.robotselfcontact_views.items():
                for source_link, spec in source_dict.items():
                    try:
                        contact_view = spec["view"]
                        filter_labels = spec.get("filter_labels", [])
                        matrix = contact_view.get_contact_force_matrix()
                        if matrix is None:
                            continue
                        matrix = np.asarray(matrix)
                        for filter_idx, filter_link in enumerate(filter_labels):
                            if matrix.ndim >= 3:
                                if filter_idx >= matrix.shape[1]:
                                    continue
                                force_magnitude = float(np.linalg.norm(matrix[0, filter_idx]))
                            elif matrix.ndim == 2:
                                if filter_idx >= matrix.shape[0]:
                                    continue
                                force_magnitude = float(np.linalg.norm(matrix[filter_idx]))
                            else:
                                if filter_idx != 0:
                                    continue
                                force_magnitude = float(np.linalg.norm(matrix))
                            if force_magnitude <= 0.01:
                                continue

                            body_a = f"robot/{robot_name}/{source_link}"
                            body_b = f"robot/{robot_name}/{filter_link}"
                            collision_pairs.append({
                                "bodyA": body_a,
                                "bodyB": body_b,
                                "step": step_id,
                                "force_n": force_magnitude,
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
                                "source": "contact_view.get_contact_force_matrix()",
                                "confidence": "low",
                                "num_contact_points": 0,
                                "hertzian_stiffness_n_per_m": 1e6,
                            })
                            total_impulse += force_magnitude * dt
                    except Exception as exc:
                        logger.warning(
                            "Failed to read robot self contact view %s/%s: %s",
                            robot_name, source_link, exc,
                        )

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
        for pair in collision_pairs:
            contact_forces.append({
                "bodyA": pair["bodyA"],
                "bodyB": pair["bodyB"],
                "step": pair.get("step", step_id),
                "force_n": pair.get("force_n", 0.0),
            })
        self._data["contact_force_gt"].append(contact_forces if contact_forces else [])
        self._data["contact_impulse_gt"].append(total_impulse)

        # S-GRASP-001: Collect gripper-object contact force per arm.
        gripper_obj_force = {"left": 0.0, "right": 0.0}
        if hasattr(task, 'pickcontact_views'):
            for robot_name, lr_dict in task.pickcontact_views.items():
                for lr_name, obj_dict in lr_dict.items():
                    for obj_name, contact_view in obj_dict.items():
                        try:
                            force_matrix = contact_view.get_contact_force_matrix()
                            if force_matrix is not None:
                                force_matrix = np.abs(force_matrix).squeeze()
                                force_magnitude = float(np.sum(force_matrix))
                                arm_key = "left" if "left" in lr_name.lower() or lr_name.lower().startswith("fl") else "right"
                                gripper_obj_force[arm_key] += force_magnitude
                        except Exception:
                            pass
        self._data["gripper_object_contact_force_gt"].append(gripper_obj_force)

    # ── Planner Data (S-PLAN-001, 003, 004) ────────────────────────────────

    def _collect_planner_step(self, task, step_id: int) -> None:
        """Collect planner status per step from controllers."""
        safety_gate = "pass"
        cmd_sent = False

        if hasattr(task, 'robots'):
            for robot_name, robot in task.robots.items():
                # Check if robot has controllers with active plans
                for attr_name in dir(robot):
                    if 'controller' in attr_name.lower() or 'ctrl' in attr_name.lower():
                        try:
                            ctrl = getattr(robot, attr_name)
                            if hasattr(ctrl, 'cmd_plan') and ctrl.cmd_plan is not None:
                                cmd_sent = True
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
                                if hasattr(ctrl, 'cmd_plan') and ctrl.cmd_plan is not None:
                                    cmd_sent = True

        self._data["safety_gate_status"].append(safety_gate)
        self._data["low_level_command_sent"].append(cmd_sent)

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
        """Compute distances between objects, links, and environment.

        Uses robot.get_world_pose() and known EE paths for reliable position access.
        """
        # Object-env distances: each object to ALL other objects
        obj_env_dists = {}
        if hasattr(task, 'objects'):
            for obj_name, obj in task.objects.items():
                try:
                    obj_pos, _ = obj.get_world_pose()
                    obj_dists = {}
                    for other_name, other_obj in task.objects.items():
                        if other_name != obj_name:
                            other_pos, _ = other_obj.get_world_pose()
                            d = float(np.linalg.norm(np.array(obj_pos) - np.array(other_pos)))
                            obj_dists[other_name] = d
                    obj_env_dists[obj_name] = obj_dists if obj_dists else None
                except Exception:
                    obj_env_dists[obj_name] = None

        # Link-env distances: every discovered robot link to every task object.
        # Earlier versions only reported base/fl_ee_path/fr_ee_path nearest-object
        # distances.  Use link_pose_gt collected in this same step so the output is
        # complete for all robot links and still stays easy for downstream code to
        # reduce via min(value).
        link_env_dists = {}
        latest_link_poses = self._data.get("link_pose_gt")[-1] if self._data.get("link_pose_gt") else None

        for robot_name, robot in task.robots.items():
            try:
                link_positions = {}

                if isinstance(latest_link_poses, dict):
                    robot_links = latest_link_poses.get(robot_name, {})
                    if isinstance(robot_links, dict):
                        for link_name, link_pose in robot_links.items():
                            if (isinstance(link_pose, (list, tuple)) and len(link_pose) >= 3
                                    and link_pose[0] is not None):
                                link_positions[link_name] = [float(link_pose[0]), float(link_pose[1]), float(link_pose[2])]

                # Fallbacks keep the field non-empty even if link discovery fails.
                if not link_positions:
                    try:
                        base_pos, _ = robot.get_world_pose()
                        link_positions["base"] = list(base_pos)
                    except Exception:
                        pass

                    for ee_attr in ['fl_ee_path', 'fr_ee_path']:
                        if hasattr(robot, ee_attr):
                            ee_path = getattr(robot, ee_attr)
                            if ee_path:
                                try:
                                    from omni.isaac.core.utils.prims import get_prim_at_path
                                    from omni.isaac.core.utils.xforms import get_world_pose
                                    prim = get_prim_at_path(ee_path)
                                    if prim and prim.IsValid():
                                        pos, _ = get_world_pose(ee_path)
                                        link_positions[ee_attr] = list(pos)
                                except Exception:
                                    pass

                if hasattr(task, 'objects'):
                    for link_name, link_pos in link_positions.items():
                        link_xyz = np.array(link_pos[:3], dtype=float)
                        for obj_name, obj in task.objects.items():
                            try:
                                obj_pos, _ = obj.get_world_pose()
                                d = float(np.linalg.norm(link_xyz - np.array(obj_pos, dtype=float)))
                                key = f"{robot_name}/{link_name}→{obj_name}"
                                link_env_dists[key] = d
                            except Exception:
                                pass

            except Exception:
                pass

        # Self-distance: pairwise distances between all arm links
        self_dists = {}
        for robot_name, robot in task.robots.items():
            try:
                # Get link poses from collected data
                link_poses = self._data.get("link_pose_gt")
                if not link_poses:
                    self_dists[robot_name] = None
                    continue

                # Get latest step's link poses
                step_links = link_poses[-1] if link_poses else {}
                robot_links = step_links.get(robot_name, {}) if isinstance(step_links, dict) else {}

                if not robot_links:
                    self_dists[robot_name] = None
                    continue

                # Filter to arm links only (fl/*, fr/*)
                arm_links = {}
                for link_name, link_pose in robot_links.items():
                    short = link_name.split("/")[-1] if "/" in link_name else link_name
                    if (link_name.endswith("/arm_base") or
                        link_name.endswith("/link1") or
                        link_name.endswith("/link2") or
                        link_name.endswith("/link3") or
                        link_name.endswith("/link4") or
                        link_name.endswith("/link5") or
                        link_name.endswith("/link6") or
                        link_name.endswith("/link7") or
                        link_name.endswith("/link8")):
                        if link_pose and len(link_pose) >= 3 and link_pose[0] is not None:
                            arm_links[link_name] = link_pose

                # Compute pairwise distances between all arm links
                link_names = list(arm_links.keys())
                pairwise_dists = {}

                for i in range(len(link_names)):
                    for j in range(i + 1, len(link_names)):
                        name_i = link_names[i].split("/")[-1]
                        name_j = link_names[j].split("/")[-1]
                        pose_i = arm_links[link_names[i]]
                        pose_j = arm_links[link_names[j]]
                        d = float(np.linalg.norm(
                            np.array(pose_i[:3]) - np.array(pose_j[:3])
                        ))
                        # Store with arm prefix for clarity
                        arm_i = "fl" if "/fl/" in link_names[i] else "fr"
                        arm_j = "fl" if "/fl/" in link_names[j] else "fr"
                        key = f"{arm_i}/{name_i}→{arm_j}/{name_j}"
                        pairwise_dists[key] = d

                self_dists[robot_name] = pairwise_dists
            except Exception as e:
                self_dists[robot_name] = None

        self._data["object_env_distance_gt"].append(obj_env_dists if obj_env_dists else None)
        self._data["link_env_distance_gt"].append(link_env_dists if link_env_dists else None)
        self._data["self_distance_gt"].append(self_dists if self_dists else None)

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
