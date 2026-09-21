import os
import random
from copy import deepcopy

import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.plan_utils import (
    select_index_by_priority_dual,
    select_index_by_priority_single,
)
from core.utils.transformation_utils import poses_from_tf_matrices
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    tf_matrix_from_pose,
)


# pylint: disable=unused-argument
@register_skill
class Pick(BaseSkill):
    @staticmethod
    def _grasp_annotation_path(usd_path, npy_name):
        """Resolve a grasp annotation beside the *selected* USD asset.

        Randomized rigid objects can be selected from a sibling instance
        (e.g. ``..._003/Aligned_obj.usd``) while the task YAML still names the
        nominal instance (e.g. ``..._001``).  The annotation must come from the
        same instance; otherwise a valid grasp pose can be several centimetres
        away from the actual mesh.
        """
        usd_path = os.path.abspath(os.fspath(usd_path))
        npy_name = os.fspath(npy_name)
        if os.path.isabs(npy_name):
            return npy_name

        asset_name = os.path.basename(usd_path)
        if asset_name == "Aligned_obj.usd":
            return os.path.join(os.path.dirname(usd_path), npy_name)
        # Preserve the historical replacement behavior for custom asset names,
        # while still keeping the annotation in the selected asset directory.
        return os.path.join(os.path.dirname(usd_path), npy_name)

    def _resolve_grasp_annotation(self, object_name):
        """Return the grasp annotation belonging to the runtime object asset."""
        npy_name = self.skill_cfg.get("npy_name", "Aligned_grasp_sparse.npy")
        configured_path = [
            obj["path"]
            for obj in self.task.cfg["objects"]
            if obj["name"] == object_name
        ][0]
        configured_usd = (
            configured_path
            if os.path.isabs(configured_path)
            else os.path.join(self.task.asset_root, configured_path)
        )

        # RigidObject records the concrete path used by create_prim().  This is
        # the authoritative path after object randomization.  Keep a fallback
        # for legacy object implementations that do not expose ``usd_path``.
        runtime_usd = getattr(self.pick_obj, "usd_path", None)
        selected_usd = runtime_usd or configured_usd
        grasp_path = self._grasp_annotation_path(selected_usd, npy_name)

        if not os.path.isfile(grasp_path):
            # Do not silently use the nominal instance's annotation when a
            # randomized instance is missing one: that recreates the original
            # mismatch and produces an apparently successful approach with no
            # physical grasp.
            if runtime_usd and os.path.abspath(runtime_usd) != os.path.abspath(configured_usd):
                raise FileNotFoundError(
                    "No grasp annotation for the runtime-randomized asset: "
                    f"object={object_name!r}, usd={selected_usd!r}, "
                    f"expected={grasp_path!r}. The nominal config asset is "
                    f"{configured_usd!r}; refusing to mix annotations."
                )
            raise FileNotFoundError(
                f"Grasp annotation not found for object={object_name!r}: {grasp_path!r}"
            )

        print(
            f"[pick] object={object_name} runtime_asset={selected_usd} "
            f"grasp_annotation={grasp_path}"
        )
        return grasp_path

    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg
        object_name = self.skill_cfg["objects"][0]
        self.pick_obj = task.objects[object_name]

        # Get the annotation for the concrete runtime asset.  The object path
        # may have been randomized before this skill is constructed.
        grasp_pose_path = self._resolve_grasp_annotation(object_name)
        sparse_grasp_poses = np.load(grasp_pose_path)
        lr_arm = "right" if "right" in self.controller.robot_file else "left"
        self.T_obj_ee, self.scores = self.robot.pose_post_process_fn(
            sparse_grasp_poses,
            lr_arm=lr_arm,
            grasp_scale=self.skill_cfg.get("grasp_scale", 1),
            tcp_offset=self.skill_cfg.get("tcp_offset", self.robot.tcp_offset),
            constraints=self.skill_cfg.get("constraints", None),
        )

        # Keyposes should be generated after previous skill is done
        self.manip_list = []
        self.pickcontact_view = task.pickcontact_views[robot.name][lr_arm][object_name]
        self.process_valid = True
        self.plan_failed = False
        self.obj_init_trans = deepcopy(self.pick_obj.get_local_pose()[0])
        # Physical-grasp evidence is collected independently from CuRobo's
        # planner attachment.  ``attach_obj`` only changes the planner world;
        # it must never by itself make a pick look successful.
        self._contact_seen_frames = 0
        self._contact_samples = []
        self._first_contact_obj_z = None
        self._first_contact_ee_z = None
        self._max_contact_obj_lift = 0.0
        self._max_contact_ee_lift = 0.0
        self._close_phase_started = False
        self._strict_pick_validation = bool(
            self.skill_cfg.get("strict_pick_validation", True)
        )
        final_gripper_state = self.skill_cfg.get("final_gripper_state", -1)
        if final_gripper_state == 1:
            self.gripper_cmd = "open_gripper"
        elif final_gripper_state == -1:
            self.gripper_cmd = "close_gripper"
        else:
            raise ValueError(f"final_gripper_state must be 1 or -1, got {final_gripper_state}")
        self.fixed_orientation = self.skill_cfg.get("fixed_orientation", None)
        if self.fixed_orientation is not None:
            self.fixed_orientation = np.array(self.fixed_orientation)

    def simple_generate_manip_cmds(self):
        manip_list = []

        # Update
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        cmd = (p_base_ee_cur, q_base_ee_cur, "update_pose_cost_metric", {"hold_vec_weight": None})
        manip_list.append(cmd)

        ignore_substring = deepcopy(self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", []))
        ignore_substring.append(self.pick_obj.name)
        cmd = (
            p_base_ee_cur,
            q_base_ee_cur,
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)

        # Pre grasp
        # Small tableware assets have many annotated grasps, while only a
        # subset is reachable for the current arm/base pose.  Allow a task to
        # request a larger deterministic candidate set instead of falling
        # back to the last failed sample after the default batch is exhausted.
        grasp_candidate_count = int(
            self.skill_cfg.get("grasp_candidate_count", CUROBO_BATCH_SIZE)
        )
        T_base_ee_grasps = self.sample_ee_pose(max_length=grasp_candidate_count)  # (N, 4, 4)
        T_base_ee_pregrasps = deepcopy(T_base_ee_grasps)
        self.controller.update_specific(
            ignore_substring=ignore_substring, reference_prim_path=self.controller.reference_prim_path
        )

        if "r5a" in self.controller.robot_file:
            T_base_ee_pregrasps[:, :3, 3] -= T_base_ee_pregrasps[:, :3, 0] * self.skill_cfg.get("pre_grasp_offset", 0.1)
        else:
            T_base_ee_pregrasps[:, :3, 3] -= T_base_ee_pregrasps[:, :3, 2] * self.skill_cfg.get("pre_grasp_offset", 0.1)

        p_base_ee_pregrasps, q_base_ee_pregrasps = poses_from_tf_matrices(T_base_ee_pregrasps)
        p_base_ee_grasps, q_base_ee_grasps = poses_from_tf_matrices(T_base_ee_grasps)

        def _batch_success_mask(plan_result):
            """Return a NumPy success mask without assuming torch/NumPy type."""
            success = getattr(plan_result, "success", None)
            if success is None:
                return np.zeros(T_base_ee_grasps.shape[0], dtype=bool)
            try:
                success = success.detach().cpu().numpy()
            except AttributeError:
                success = np.asarray(success)
            return np.asarray(success, dtype=bool).reshape(-1)

        if self.controller.use_batch:
            # Check if the input arrays are exactly the same
            if np.array_equal(p_base_ee_pregrasps, p_base_ee_grasps) and np.array_equal(
                q_base_ee_pregrasps, q_base_ee_grasps
            ):
                # Inputs are identical, compute only once to avoid redundant computation
                result = self.controller.test_batch_forward(p_base_ee_grasps, q_base_ee_grasps)
                result_success = _batch_success_mask(result)
                if not result_success.any():
                    self.plan_failed = True
                    self.process_valid = False
                    self.manip_list = []
                    print(
                        f"[pick_plan_failed] object={self.pick_obj.name} "
                        f"batch_candidates={len(result_success)} stage=grasp"
                    )
                    return
                index = select_index_by_priority_single(result)
            else:
                # Inputs are different, compute separately
                pre_result = self.controller.test_batch_forward(p_base_ee_pregrasps, q_base_ee_pregrasps)
                result = self.controller.test_batch_forward(p_base_ee_grasps, q_base_ee_grasps)
                common_success = _batch_success_mask(pre_result) & _batch_success_mask(result)
                if not common_success.any():
                    self.plan_failed = True
                    self.process_valid = False
                    self.manip_list = []
                    print(
                        f"[pick_plan_failed] object={self.pick_obj.name} "
                        f"batch_candidates={len(common_success)} stage=pregrasp_and_grasp"
                    )
                    return
                index = select_index_by_priority_dual(pre_result, result)
        else:
            plan_found = False
            for index in range(T_base_ee_grasps.shape[0]):
                p_base_ee_pregrasp, q_base_ee_pregrasp = p_base_ee_pregrasps[index], q_base_ee_pregrasps[index]
                p_base_ee_grasp, q_base_ee_grasp = p_base_ee_grasps[index], q_base_ee_grasps[index]
                test_mode = self.skill_cfg.get("test_mode", "forward")
                if test_mode == "forward":
                    result_pre = self.controller.test_single_forward(p_base_ee_pregrasp, q_base_ee_pregrasp)
                elif test_mode == "ik":
                    result_pre = self.controller.test_single_ik(p_base_ee_pregrasp, q_base_ee_pregrasp)
                else:
                    raise NotImplementedError
                if self.skill_cfg.get("pre_grasp_offset", 0.1) > 0:
                    if test_mode == "forward":
                        result = self.controller.test_single_forward(p_base_ee_grasp, q_base_ee_grasp)
                    elif test_mode == "ik":
                        result = self.controller.test_single_ik(p_base_ee_grasp, q_base_ee_grasp)
                    else:
                        raise NotImplementedError
                    if result == 1 and result_pre == 1:
                        print("pick plan success")
                        plan_found = True
                        break
                else:
                    if result_pre == 1:
                        print("pick plan success")
                        plan_found = True
                        break

            if not plan_found:
                # Do not enqueue an untested grasp pose.  The caller will
                # record the failed skill/episode, preserving the distinction
                # between a planning failure and a physical grasp failure.
                self.plan_failed = True
                self.process_valid = False
                self.manip_list = []
                print(
                    f"[pick_plan_failed] object={self.pick_obj.name} "
                    f"candidates={T_base_ee_grasps.shape[0]}"
                )
                return

        if self.fixed_orientation is not None:
            q_base_ee_pregrasps[index] = self.fixed_orientation
            q_base_ee_grasps[index] = self.fixed_orientation

        print(
            f"[pick_target] object={self.pick_obj.name} "
            f"arm={'right' if 'right' in self.controller.robot_file else 'left'} "
            f"object_init={np.asarray(self.obj_init_trans).round(5).tolist()} "
            f"grasp_pos={np.asarray(p_base_ee_grasps[index]).round(5).tolist()} "
            f"pre_pos={np.asarray(p_base_ee_pregrasps[index]).round(5).tolist()}"
        )

        # Pre-grasp
        cmd = (
            p_base_ee_pregrasps[index],
            q_base_ee_pregrasps[index],
            "open_gripper",
            {"phase": "pregrasp", "force_replan": True},
        )
        manip_list.append(cmd)
        if self.skill_cfg.get("pre_grasp_hold_vec_weight", None) is not None:
            cmd = (
                p_base_ee_pregrasps[index],
                q_base_ee_pregrasps[index],
                "update_pose_cost_metric",
                {"hold_vec_weight": self.skill_cfg.get("pre_grasp_hold_vec_weight", None)},
            )
            manip_list.append(cmd)

        # Grasp
        cmd = (
            p_base_ee_grasps[index],
            q_base_ee_grasps[index],
            "open_gripper",
            {"phase": "grasp", "force_replan": True},
        )
        manip_list.append(cmd)
        gripper_steps = int(self.skill_cfg.get("gripper_change_steps", 40))
        for close_idx in range(max(1, gripper_steps)):
            # Replan once when the gripper-close phase starts.  The remaining
            # close frames hold that planned arm pose while the fingers move.
            cmd = (
                p_base_ee_grasps[index],
                q_base_ee_grasps[index],
                self.gripper_cmd,
                {
                    "phase": "gripper_close",
                    "force_replan": close_idx == 0,
                },
            )
            manip_list.append(cmd)
        ignore_substring = deepcopy(self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", []))
        cmd = (
            p_base_ee_grasps[index],
            q_base_ee_grasps[index],
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)
        cmd = (
            p_base_ee_grasps[index],
            q_base_ee_grasps[index],
            "attach_obj",
            {
                "obj_prim_path": self.pick_obj.mesh_prim_path,
                "phase": "attach",
                "force_replan": True,
            },
        )
        manip_list.append(cmd)

        # Post-grasp
        post_grasp_offset = np.random.uniform(
            self.skill_cfg.get("post_grasp_offset_min", 0.05), self.skill_cfg.get("post_grasp_offset_max", 0.05)
        )
        if post_grasp_offset:
            p_base_ee_postgrasps = deepcopy(p_base_ee_grasps)
            p_base_ee_postgrasps[index][2] += post_grasp_offset
            cmd = (
                p_base_ee_postgrasps[index],
                q_base_ee_grasps[index],
                self.gripper_cmd,
                {"phase": "lift", "force_replan": True},
            )
            manip_list.append(cmd)

        # Whether return to pre-grasp
        if self.skill_cfg.get("return_to_pregrasp", False):
            cmd = (p_base_ee_pregrasps[index], q_base_ee_pregrasps[index], self.gripper_cmd, {})
            manip_list.append(cmd)

        self.manip_list = manip_list

    def sample_ee_pose(self, max_length=CUROBO_BATCH_SIZE):
        T_base_ee = self.get_ee_poses("armbase")

        num_pose = T_base_ee.shape[0]
        flags = {
            "x": np.ones(num_pose, dtype=bool),
            "y": np.ones(num_pose, dtype=bool),
            "z": np.ones(num_pose, dtype=bool),
            "direction_to_obj": np.ones(num_pose, dtype=bool),
        }
        filter_conditions = {
            "x": {
                "forward": (0, 0, 1),  # (row, col, direction)
                "backward": (0, 0, -1),
                "upward": (2, 0, 1),
                "downward": (2, 0, -1),
            },
            "y": {"forward": (0, 1, 1), "backward": (0, 1, -1), "downward": (2, 1, -1), "upward": (2, 1, 1)},
            "z": {"forward": (0, 2, 1), "backward": (0, 2, -1), "downward": (2, 2, -1), "upward": (2, 2, 1)},
        }
        for axis in ["x", "y", "z"]:
            filter_list = self.skill_cfg.get(f"filter_{axis}_dir", None)
            if filter_list is not None:
                # direction, value = filter_list
                direction = filter_list[0]
                row, col, sign = filter_conditions[axis][direction]
                if len(filter_list) == 2:
                    value = filter_list[1]
                    cos_val = np.cos(np.deg2rad(value))
                    flags[axis] = T_base_ee[:, row, col] >= cos_val if sign > 0 else T_base_ee[:, row, col] <= cos_val
                elif len(filter_list) == 3:
                    value1, value2 = filter_list[1:]
                    cos_val1 = np.cos(np.deg2rad(value1))
                    cos_val2 = np.cos(np.deg2rad(value2))
                    if sign > 0:
                        flags[axis] = np.logical_and(
                            T_base_ee[:, row, col] >= cos_val1, T_base_ee[:, row, col] <= cos_val2
                        )
                    else:
                        flags[axis] = np.logical_and(
                            T_base_ee[:, row, col] <= cos_val1, T_base_ee[:, row, col] >= cos_val2
                        )
        if self.skill_cfg.get("direction_to_obj", None) is not None:
            direction_to_obj = self.skill_cfg["direction_to_obj"]
            T_world_obj = tf_matrix_from_pose(*self.pick_obj.get_local_pose())
            T_base_world = get_relative_transform(
                get_prim_at_path(self.task.root_prim_path), get_prim_at_path(self.controller.reference_prim_path)
            )
            T_base_obj = T_base_world @ T_world_obj
            if direction_to_obj == "right":
                flags["direction_to_obj"] = T_base_ee[:, 1, 3] <= T_base_obj[1, 3]
            elif direction_to_obj == "left":
                flags["direction_to_obj"] = T_base_ee[:, 1, 3] > T_base_obj[1, 3]
            else:
                raise NotImplementedError

        combined_flag = np.logical_and.reduce(list(flags.values()))
        if sum(combined_flag) == 0:
            idx_list = list(range(min(max_length, num_pose)))
        else:
            tmp_scores = self.scores[combined_flag]
            tmp_idxs = np.arange(num_pose)[combined_flag]
            combined = list(zip(tmp_scores, tmp_idxs))
            combined.sort()
            idx_list = [idx for (score, idx) in combined[:max_length]]
            if not self.skill_cfg.get("deterministic_grasp_selection", False):
                score_list = self.scores[idx_list]
                weights = 1.0 / (score_list + 1e-8)
                weights = weights / weights.sum()
                sampled_idx = random.choices(idx_list, weights=weights, k=min(max_length, len(idx_list)))
                sampled_scores = self.scores[sampled_idx]

                # Sort indices by their scores (ascending)
                sorted_pairs = sorted(zip(sampled_scores, sampled_idx))
                idx_list = [idx for _, idx in sorted_pairs]

        print(self.scores[idx_list])
        # print((T_base_ee[idx_list])[:, 0, 1])
        return T_base_ee[idx_list]

    def get_ee_poses(self, frame: str = "world"):
        # get grasp poses at specific frame
        if frame not in ["world", "body", "armbase"]:
            raise ValueError(
                f"poses in {frame} frame is not supported: accepted values are [world, body, armbase] only"
            )

        if frame == "body":
            return self.T_obj_ee

        T_world_obj = tf_matrix_from_pose(*self.pick_obj.get_local_pose())
        T_world_ee = T_world_obj[None] @ self.T_obj_ee

        if frame == "world":
            return T_world_ee

        if frame == "armbase":  # arm base frame
            T_world_base = get_relative_transform(
                get_prim_at_path(self.controller.reference_prim_path), get_prim_at_path(self.task.root_prim_path)
            )
            T_base_world = np.linalg.inv(T_world_base)
            T_base_ee = T_base_world[None] @ T_world_ee
            return T_base_ee

    def get_contact(self, contact_threshold=0.0):
        try:
            raw_contact = self.pickcontact_view.get_contact_force_matrix()
            contact = np.asarray(np.abs(raw_contact), dtype=float)
            if contact.ndim == 0:
                contact = contact.reshape(1)
            elif contact.ndim >= 1:
                contact = np.sum(contact, axis=-1).reshape(-1)
            indices = np.where(contact > float(contact_threshold))[0]
        except Exception as exc:  # keep the semantic result conservative
            print(f"[pick_contact] unavailable for {self.pick_obj.name}: {exc}")
            contact = np.zeros(0, dtype=float)
            indices = np.zeros(0, dtype=int)
        return contact, indices

    def _sample_physical_grasp(self):
        """Accumulate contact/follow evidence from the live PhysX state."""
        # The contact view is filtered to the gripper links, but the object
        # can still touch a finger during the open pre-grasp approach.  Only
        # begin counting evidence once the active skill command has entered
        # the actual close phase; otherwise a pre-grasp bump could be mistaken
        # for a successful grasp.
        if self.manip_list:
            active_fn = self.manip_list[0][2]
            if active_fn == self.gripper_cmd:
                self._close_phase_started = True
        if not self._close_phase_started:
            return
        threshold = float(self.skill_cfg.get("contact_force_threshold_n", 0.01))
        contact, indices = self.get_contact(contact_threshold=threshold)
        if self.gripper_cmd != "close_gripper" or len(indices) == 0:
            return

        self._contact_seen_frames += 1
        obj_pos = np.asarray(self.pick_obj.get_local_pose()[0], dtype=float)
        try:
            ee_pos, _ = self.controller.get_ee_pose()
            ee_pos = np.asarray(ee_pos, dtype=float)
            if self._first_contact_obj_z is None:
                self._first_contact_obj_z = float(obj_pos[2])
                self._first_contact_ee_z = float(ee_pos[2])
            self._max_contact_obj_lift = max(
                self._max_contact_obj_lift,
                float(obj_pos[2]) - self._first_contact_obj_z,
            )
            self._max_contact_ee_lift = max(
                self._max_contact_ee_lift,
                float(ee_pos[2]) - self._first_contact_ee_z,
            )
            self._contact_samples.append(obj_pos - ee_pos)
            # Avoid unbounded episode memory while retaining enough samples to
            # detect an object that moves independently after finger closure.
            self._contact_samples = self._contact_samples[-30:]
        except Exception as exc:
            print(f"[pick_contact] relative pose unavailable: {exc}")

    def _physical_attachment_ok(self, lift_delta):
        """Require real contact plus object motion consistent with the lift."""
        min_frames = int(self.skill_cfg.get("min_contact_frames", 2))
        if self._contact_seen_frames < max(1, min_frames):
            return False

        lift_th = float(self.skill_cfg.get("lift_th", 0.0))
        if lift_th > 0.0 and lift_delta <= lift_th:
            return False

        if self._first_contact_obj_z is not None and self._first_contact_ee_z is not None:
            # Use the maximum lift observed while contact was live.  Some
            # skills optionally return to pre-grasp after lifting; checking
            # only the final EE pose would reject an otherwise valid grasp.
            if (
                self._max_contact_obj_lift <= max(0.5 * lift_th, 0.005)
                or self._max_contact_ee_lift <= 0.0
            ):
                return False

        if len(self._contact_samples) >= 2:
            samples = np.asarray(self._contact_samples, dtype=float)
            tolerance = float(
                self.skill_cfg.get("attachment_relative_tolerance_m", 0.05)
            )
            if float(np.ptp(samples, axis=0).max()) > tolerance:
                return False
        return True

    def is_feasible(self, th=None):
        """Allow task-level planner retry budgets to reach the skill.

        Older pick skills used a hard-coded five-failure cutoff, which made
        the task YAML value ``max_consecutive_plan_failures: 20`` ineffective
        for thin or highly constrained objects.  Keep an explicit skill-level
        override, then fall back to the task budget and finally to 20.
        """
        if th is None:
            task_cfg = getattr(self.task, "cfg", {}) or {}
            data_cfg = task_cfg.get("data", {}) if hasattr(task_cfg, "get") else {}
            th = self.skill_cfg.get(
                "max_plan_failures",
                data_cfg.get("max_consecutive_plan_failures", 20),
            )
        return self.controller.num_plan_failed <= int(th)

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
        self._sample_physical_grasp()
        if len(self.manip_list) == 0:
            # Empty can mean a successful final command or an infeasible
            # pre-plan. Require the physical pick check in both cases.
            done = bool(self.is_success())
            if not done:
                self.process_valid = False
            return done
        if self.is_subtask_done(t_eps=self.skill_cfg.get("t_eps", 1e-3), o_eps=self.skill_cfg.get("o_eps", 5e-3)):
            self.manip_list.pop(0)
        return len(self.manip_list) == 0

    def is_success(self):
        flag = not self.plan_failed

        _, indices = self.get_contact(
            contact_threshold=float(self.skill_cfg.get("contact_force_threshold_n", 0.01))
        )
        contact_count = len(indices)
        lift_delta = float(self.pick_obj.get_local_pose()[0][2] - self.obj_init_trans[2])
        lift_th = float(self.skill_cfg.get("lift_th", 0.0))
        lift_ok = lift_delta > lift_th if lift_th > 0.0 else True
        planner_attachment_ok = (
            getattr(self.controller, "_last_attached_obj_path", None)
            == self.pick_obj.mesh_prim_path
        )
        attachment_ok = self._physical_attachment_ok(lift_delta)
        if self.gripper_cmd == "close_gripper":
            flag = attachment_ok if self._strict_pick_validation else contact_count >= 1

        if self.skill_cfg.get("process_valid", True):
            # Robot joint speed is an RS safety signal, not evidence that the
            # object was not picked.  Keep task-semantic success independent
            # from that instantaneous measurement.
            self.process_valid = np.max(np.abs(self.pick_obj.get_linear_velocity())) < 5
        flag = flag and self.process_valid

        if lift_th > 0.0:
            flag = flag and lift_ok

        print(
            f"[pick_result] object={self.pick_obj.name} contacts={contact_count} "
            f"contact_frames={self._contact_seen_frames} "
            f"lift_delta={lift_delta:.5f} attached={attachment_ok} "
            f"max_contact_obj_lift={self._max_contact_obj_lift:.5f} "
            f"max_contact_ee_lift={self._max_contact_ee_lift:.5f} "
            f"planner_attached={planner_attachment_ok} "
            f"process_valid={self.process_valid} success={bool(flag)}"
        )

        return flag
