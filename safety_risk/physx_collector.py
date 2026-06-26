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
            # S-ROBOT-005: link poses (per link, per step)
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
        self._contact_state: Dict[str, bool] = {}  # track ongoing contacts

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
            self._collect_link_data(task)
        except Exception as e:
            self._data["link_pose_gt"].append(None)
            self._data["link_velocity_gt"].append(None)

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

    def _collect_link_data(self, task) -> None:
        """Collect link poses and velocities from all robots.

        Uses robot.get_world_pose() for base link and the articulation's
        joint states to compute EE link poses. For split_aloha, this gives
        left/right arm EE poses per step.
        """
        all_poses = []
        all_velocities = []

        for robot_name, robot in task.robots.items():
            try:
                poses = []
                velocities = []

                # Get robot base pose
                try:
                    base_pos, base_ori = robot.get_world_pose()
                    poses.append([
                        float(base_pos[0]), float(base_pos[1]), float(base_pos[2]),
                        float(base_ori[0]), float(base_ori[1]), float(base_ori[2]), float(base_ori[3])
                    ])
                except Exception:
                    poses.append([None] * 7)

                # Get EE poses from known paths (most reliable)
                for ee_attr in ['fl_ee_path', 'fr_ee_path']:
                    if hasattr(robot, ee_attr):
                        ee_path = getattr(robot, ee_attr)
                        if ee_path:
                            try:
                                from omni.isaac.core.utils.prims import get_prim_at_path
                                from omni.isaac.core.utils.xforms import get_world_pose
                                prim = get_prim_at_path(ee_path)
                                if prim and prim.IsValid():
                                    pos, ori = get_world_pose(ee_path)
                                    poses.append([
                                        float(pos[0]), float(pos[1]), float(pos[2]),
                                        float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])
                                    ])
                                else:
                                    poses.append([None] * 7)
                            except Exception:
                                poses.append([None] * 7)

                # Velocity: compute from pose differences (stored in prev step)
                if not hasattr(self, '_prev_link_poses'):
                    self._prev_link_poses = {}
                prev = self._prev_link_poses.get(robot_name)

                for i, pose in enumerate(poses):
                    if pose is not None and prev is not None and i < len(prev) and prev[i] is not None:
                        dt = 0.033
                        vx = (pose[0] - prev[i][0]) / dt
                        vy = (pose[1] - prev[i][1]) / dt
                        vz = (pose[2] - prev[i][2]) / dt
                        velocities.append([vx, vy, vz, 0.0, 0.0, 0.0])
                    else:
                        velocities.append([0.0] * 6)

                self._prev_link_poses[robot_name] = poses

                all_poses.append(poses)
                all_velocities.append(velocities)

            except Exception:
                all_poses.append(None)
                all_velocities.append(None)

        self._data["link_pose_gt"].append(all_poses)
        self._data["link_velocity_gt"].append(all_velocities)

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

        # Check pick contact views
        if hasattr(task, 'pickcontact_views'):
            for robot_name, lr_dict in task.pickcontact_views.items():
                for lr_name, obj_dict in lr_dict.items():
                    for obj_name, contact_view in obj_dict.items():
                        try:
                            force_matrix = contact_view.get_contact_force_matrix()
                            if force_matrix is not None:
                                force_matrix = np.abs(force_matrix).squeeze()
                                if force_matrix.ndim >= 2:
                                    force_sum = np.sum(force_matrix, axis=tuple(range(force_matrix.ndim - 1)))
                                else:
                                    force_sum = force_matrix

                                force_magnitude = float(np.linalg.norm(force_sum))
                                if force_magnitude > 0.01:
                                    # Collision pair
                                    collision_pairs.append({
                                        "bodyA": f"robot/{robot_name}/{lr_name}",
                                        "bodyB": f"object/{obj_name}",
                                        "step": step_id,
                                        "force_n": force_magnitude,
                                    })

                                    # Collision location: midpoint between EE and object
                                    obj_pos = _get_pos(obj_name)
                                    if obj_pos:
                                        # Use object position as collision location (approximate)
                                        collision_locations.append({
                                            "bodyA": f"robot/{robot_name}/{lr_name}",
                                            "bodyB": f"object/{obj_name}",
                                            "location_m": obj_pos,
                                        })

                                    # Penetration depth: estimate from force
                                    # Using Hertzian approximation: F = k * delta^(3/2)
                                    # With k=1e6 N/m^(3/2) as typical stiffness
                                    # delta = (F/k)^(2/3)
                                    stiffness = 1e6  # N/m^(3/2), typical for rigid contact
                                    delta = (force_magnitude / stiffness) ** (2.0 / 3.0)
                                    penetration_depths.append({
                                        "bodyA": f"robot/{robot_name}/{lr_name}",
                                        "bodyB": f"object/{obj_name}",
                                        "depth_cm": delta * 100.0,  # m -> cm
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
                                        "depth_cm": delta * 100.0,
                                    })

                                    total_impulse += force_magnitude * dt
                        except Exception:
                            pass

        # Track contact events for duration computation
        for pair in collision_pairs:
            key = f"{pair['bodyA']}_{pair['bodyB']}"
            if key not in self._contact_state:
                self._contact_state[key] = {"start_step": step_id, "active": True}
            elif not self._contact_state[key]["active"]:
                self._contact_state[key] = {"start_step": step_id, "active": True}

        # End contacts that are no longer active
        active_keys = {f"{p['bodyA']}_{p['bodyB']}" for p in collision_pairs}
        for key in list(self._contact_state.keys()):
            if key not in active_keys and self._contact_state[key]["active"]:
                self._contact_state[key]["active"] = False
                self._contact_state[key]["end_step"] = step_id

        self._data["collision_pair_gt"].append(collision_pairs if collision_pairs else [])
        self._data["collision_location_gt"].append(collision_locations if collision_locations else [])
        self._data["penetration_depth_gt"].append(penetration_depths if penetration_depths else [])
        self._data["contact_force_gt"].append(total_force)
        self._data["contact_impulse_gt"].append(total_impulse)

        # S-GRASP-001: Collect gripper-object contact force
        gripper_obj_force = 0.0
        if hasattr(task, 'pickcontact_views'):
            for robot_name, lr_dict in task.pickcontact_views.items():
                for lr_name, obj_dict in lr_dict.items():
                    for obj_name, contact_view in obj_dict.items():
                        try:
                            force_matrix = contact_view.get_contact_force_matrix()
                            if force_matrix is not None:
                                force_matrix = np.abs(force_matrix).squeeze()
                                force_magnitude = float(np.sum(force_matrix))
                                gripper_obj_force = max(gripper_obj_force, force_magnitude)
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
        # Object-env distances
        obj_env_dists = {}
        if hasattr(task, 'objects'):
            for obj_name, obj in task.objects.items():
                try:
                    obj_pos, _ = obj.get_world_pose()
                    min_dist = float('inf')
                    for other_name, other_obj in task.objects.items():
                        if other_name != obj_name:
                            other_pos, _ = other_obj.get_world_pose()
                            d = float(np.linalg.norm(np.array(obj_pos) - np.array(other_pos)))
                            min_dist = min(min_dist, d)
                    obj_env_dists[obj_name] = min_dist * 100.0 if min_dist < float('inf') else None
                except Exception:
                    obj_env_dists[obj_name] = None

        # Link-env distances using robot EE and base positions
        link_env_dists = {}
        for robot_name, robot in task.robots.items():
            try:
                # Collect known robot positions
                link_positions = []

                # Base position
                try:
                    base_pos, _ = robot.get_world_pose()
                    link_positions.append(("base", list(base_pos)))
                except Exception:
                    pass

                # EE positions from known paths
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
                                    link_positions.append((ee_attr, list(pos)))
                            except Exception:
                                pass

                # Compute distance from each link to nearest object
                for link_name, link_pos in link_positions:
                    min_dist = float('inf')
                    if hasattr(task, 'objects'):
                        for obj_name, obj in task.objects.items():
                            try:
                                obj_pos, _ = obj.get_world_pose()
                                d = float(np.linalg.norm(np.array(link_pos) - np.array(obj_pos)))
                                min_dist = min(min_dist, d)
                            except Exception:
                                pass
                    link_env_dists[link_name] = min_dist * 100.0 if min_dist < float('inf') else None

            except Exception:
                pass

        # Self-distance (between known robot positions)
        self_dists = {}
        for robot_name, robot in task.robots.items():
            try:
                positions = []
                try:
                    base_pos, _ = robot.get_world_pose()
                    positions.append(list(base_pos))
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
                                    positions.append(list(pos))
                            except Exception:
                                pass

                min_self_dist = float('inf')
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        d = float(np.linalg.norm(np.array(positions[i]) - np.array(positions[j])))
                        min_self_dist = min(min_self_dist, d)

                self_dists[robot_name] = min_self_dist * 100.0 if min_self_dist < float('inf') else None
            except Exception:
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
        # Compute contact durations
        contact_durations = []
        dt = 0.033
        for key, state in self._contact_state.items():
            if "end_step" in state:
                duration = (state["end_step"] - state["start_step"]) * dt
            elif state["active"]:
                duration = (len(self._data["step_ids"]) - state["start_step"]) * dt
            else:
                duration = 0.0
            contact_durations.append({"contact": key, "duration_s": duration})

        return {
            "n_steps": len(self._data["step_ids"]),
            "contact_events": contact_durations,
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
