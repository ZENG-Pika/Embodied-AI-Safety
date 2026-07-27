import glob
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Optional

import numpy as np
import yaml
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
)
from omni.physx import acquire_physx_interface
from tqdm import tqdm
from yaml import Loader

from deps.world_toolkit.world_recorder import WorldRecorder
from workflows.simbox.utils.task_config_parser import TaskConfigParser

from .base import NimbusWorkFlow
from .simbox.core.controllers import get_controller_cls
from .simbox.core.loggers.lmdb_logger import LmdbLogger
from .simbox.core.loggers.utils import log_dual_obs
from .simbox.core.skills import get_skill_cls
from .simbox.core.tasks import get_task_cls
from .simbox.core.utils.collision_utils import filter_collisions
from .simbox.core.utils.utils import set_random_seed


# pylint: disable=unused-argument
@NimbusWorkFlow.register("SimBoxDualWorkFlow")
class SimBoxDualWorkFlow(NimbusWorkFlow):
    def __init__(
        self,
        world,
        task_cfg_path: str,
        scene_info: str = "dining_room_scene_info",
        random_seed: int = None,
    ):
        self.scene_info = scene_info
        self.step_replay = False
        self.random_seed = random_seed
        super().__init__(world, task_cfg_path)

    def parse_task_cfgs(self, task_cfg_path: str) -> list:
        task_cfgs = TaskConfigParser(task_cfg_path).parse_tasks()
        # Merge robot configs for each task
        for task_cfg in task_cfgs:
            self._merge_robot_configs(task_cfg)
        return task_cfgs

    def _merge_robot_configs(self, task_cfg: dict):
        """Merge robot configs from robot_config_file into task_cfg['robots']."""
        robots = task_cfg.get("robots", [])

        for robot in robots:
            robot_config_file = robot.get("robot_config_file")
            if robot_config_file:
                with open(robot_config_file, "r", encoding="utf-8") as f:
                    robot_base_cfg = yaml.load(f, Loader=Loader)

                # Merge: robot_base_cfg as base, task_cfg['robots'][i] overrides
                merged_cfg = deepcopy(robot_base_cfg)
                merged_cfg.update(robot)
                robot.clear()
                robot.update(merged_cfg)

    def reset(self, need_preload: bool = True):
        # source code noted this as debug, so it could be removed later
        from omni.isaac.core.utils.viewports import set_camera_view

        set_camera_view(eye=[1.3, 0.7, 2.7], target=[0.0, 0, 1.5], camera_prim_path="/OmniverseKit_Persp")
        # Modify config — only load arena from file on the first call;
        # subsequent calls (e.g. multiple random layouts) reuse the cached arena.
        if "arena" not in self.task_cfg:
            arena_file_path = self.task_cfg.get("arena_file", None)
            with open(arena_file_path, "r", encoding="utf-8") as arena_file:
                arena = yaml.load(arena_file, Loader=Loader)
            self.task_cfg["arena"] = arena

        for obj_cfg in self.task_cfg["objects"]:
            if obj_cfg["target_class"] == "ArticulatedObject":
                if obj_cfg.get("apply_randomization", False):
                    asset_root = self.task_cfg["asset_root"]
                    art_paths = glob.glob(os.path.join(asset_root, obj_cfg["art_cat"], "*"))
                    art_paths.sort()
                    path = random.choice(art_paths)
                    info_name = obj_cfg["info_name"]
                    info_path = f"{path}/Kps/{info_name}/info.json"
                    with open(info_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    scale = info["object_scale"][:3]

                    obj_cfg["path"] = path.replace(f"{asset_root}/", "", 1) + "/instance.usd"
                    obj_cfg["category"] = path.split("/")[-2]
                    obj_cfg["obj_info_path"] = info_path.replace(f"{asset_root}/", "", 1)
                    obj_cfg["scale"] = scale
                    self.task_cfg["data"]["collect_info"] = obj_cfg["category"]

        self.task_cfg.pop("arena_file", None)
        self.task_cfg.pop("camera_file", None)
        self.task_cfg.pop("logger_file", None)
        # Modify config done
        if self.task_cfg.get("fluid", None):
            # for fluid manipulation, only gpu mode is supportive
            physx_interface = acquire_physx_interface()
            physx_interface.overwrite_gpu_setting(1)

        self.task = get_task_cls(self.task_cfg["task"])(self.task_cfg)
        self.stage = self.world.stage
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath("/World"))
        self.world.add_task(self.task)

        # # Add hidden ground plane for physics simulation
        # from omni.isaac.core.objects import GroundPlane
        # plane = GroundPlane(
        #     prim_path="/World/GroundPlane",
        #     z_position=0.0,
        #     visible=False,
        # )

        prim_paths = []  # do not collide with each other
        global_collision_paths = []  # collide with everything

        self.robots_prim_paths = []
        for robot in self.task_cfg["robots"]:
            robot_prim_path = self.task.root_prim_path + "/" + robot["name"]
            prim_paths.append(robot_prim_path)
            self.robots_prim_paths.append(robot_prim_path)
        neglect_collision_names = self.task_cfg.get("neglect_collision_names", [])
        candidates = self.task_cfg["objects"] + self.task_cfg["arena"]["fixtures"]
        for candidate in candidates:
            candidate_prim_path = self.task.root_prim_path + "/" + candidate["name"]
            global_collision_paths.append(candidate_prim_path)
            for neglect_collision_name in neglect_collision_names:
                if neglect_collision_name in candidate["name"]:
                    prim_paths.append(candidate_prim_path)
                    global_collision_paths.remove(candidate_prim_path)

        collision_root_path = "/World/collisions"
        filter_collisions(
            self.stage,
            self.world.get_physics_context().prim_path,
            collision_root_path,
            prim_paths,
            global_collision_paths,
        )
        self.world.reset()
        self.world.step(render=True)
        self.controllers = self._initialize_controllers(self.task, self.task_cfg, self.world)
        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)

        for _ in range(50):
            self._init_static_objects(self.task)
            self.world.step(render=False)

        self.logger = LmdbLogger(
            task_dir=self.task_cfg["data"]["task_dir"],
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
            collect_info=self.task_cfg["data"]["collect_info"],
            version=self.task_cfg["data"].get("version", "v1.0"),
        )
        # Motion vectors are large dense tensors; keep LMDB logging opt-in.
        self.log_motion_vectors = bool(self.task_cfg["data"].get("log_motion_vectors", False))

        # Safety risk evaluation config (optional)
        self._safety_eval_cfg = self.task_cfg.get("safety_eval", {})
        self._safety_eval_enabled = bool(self._safety_eval_cfg.get("enabled", False))

        if self.random_seed is not None:
            seed = self.random_seed
        else:
            seed = time.time_ns() % (2**32)
        self.random_seed = seed
        set_random_seed(seed)

        # while True:
        #     self.world.get_observations()
        #     # self._init_static_objects(self.task)
        #     self.world.step(render=True)

    def _initialize_skills(self, task, task_cfg, controllers, world):
        draw_points = False
        if draw_points:
            from omni.isaac.debug_draw import _debug_draw

            draw = _debug_draw.acquire_debug_draw_interface()
        else:
            draw = None

        # Initialize skills for each robot.
        skills = []
        for cfg_skill_dict in task_cfg["skills"]:
            skill_dict = defaultdict(list)
            for robot_name, robot_skill_list in cfg_skill_dict.items():
                robot = task.robots[robot_name]
                controller = controllers[robot_name]

                for lr_skill_dict in robot_skill_list:
                    skill_sequence = [
                        [
                            get_skill_cls(skill_cfg["name"])(
                                robot,
                                controller[lr_name],
                                task,
                                skill_cfg,
                                world=world,
                                draw=draw,
                            )
                            for skill_cfg in lr_skill_list
                        ]
                        for lr_name, lr_skill_list in lr_skill_dict.items()
                    ]
                    skill_dict[robot_name].append(skill_sequence)
            skills.append(skill_dict)
        return skills

    def _initialize_controllers(self, task, task_cfg, world):
        """Initialize controllers for each robot."""
        controllers = {}
        for robot in task_cfg["robots"]:
            controllers[robot["name"]] = {}
            for robot_file in robot["robot_file"]:
                controller_name = "left" if "left" in robot_file else "right"
                controllers[robot["name"]][controller_name] = get_controller_cls(robot["target_class"])(
                    name=robot["name"],
                    robot_file=robot_file,
                    constrain_grasp_approach=robot.get("constrain_grasp_approach", False),
                    collision_activation_distance=robot.get("collision_activation_distance", 0.03),
                    task=task,
                    world=world,
                    ignore_substring=robot.get("ignore_substring", ["material", "Plane", "conveyor", "scene", "table"]),
                    use_batch=robot.get("use_batch", False),
                )
                controllers[robot["name"]][controller_name].reset()
        return controllers

    def _initialize_world_recorder(self):
        """
        Initialize WorldRecorder with appropriate mode based on configuration.

        Supports two modes:
        - step_replay=False: Records prim poses for fast geometric replay (compatible with old workflow)
        - step_replay=True: Uses preprocessed joint position data for physics-accurate replay (new default)
        """
        self.world_recorder = WorldRecorder(
            self.world,
            self.task.robots,
            self.task.objects | self.task.distractors | self.task.visuals,
            step_replay=self.step_replay,
        )
        self.world_recorder.reset()

    def _reset_controllers(self, controllers):
        """Reset all controllers."""
        for _, controller in controllers.items():
            for _, ctrl in controller.items():
                ctrl.reset()

    def _init_static_objects(self, task):
        for _, obj in task.objects.items():
            try:
                init_translation = obj.init_translation
                init_orientation = obj.init_orientation
                init_parent = obj.init_parent
                if init_translation and init_orientation and init_parent:
                    parent_world_pose = get_relative_transform(
                        get_prim_at_path(task.root_prim_path + "/" + init_parent), get_prim_at_path(task.root_prim_path)
                    )
                    parent_translation, _ = pose_from_tf_matrix(parent_world_pose)
                    obj.set_local_pose(
                        translation=(parent_translation + init_translation), orientation=init_orientation
                    )
                    obj.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                    obj.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            except Exception:
                pass

    def _randomization_layout_mem(self):
        # Reset world
        self.world.reset()

        # Individual initialize
        self.task.individual_randomize_from_mem()
        self.task.post_reset()

        self.world.step(render=False)

        # Reset controllers
        self._reset_controllers(self.controllers)

        # Reset skills
        del self.skills
        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)

        # Warmup
        for _ in range(20):
            self.world.get_observations()
            self._init_static_objects(self.task)
            self.world.step(render=False)

        self._initialize_world_recorder()

        self.logger.clear(
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
        )

        # episode_stats["current_times"] += 1

    def _randomization_layout(self):
        # Reset world
        self.world.reset()

        # Individual initialize
        self.task.individual_randomize()
        self.task.post_reset()

        self.world.step(render=False)

        # Reset controllers
        if self.task_cfg.get("fluid", None):
            # Fluid, Bug, Why !!!!!!
            # For fluid manipulation, only delete controllers and reinitialize controllers can plan successfully
            if hasattr(self, "controllers"):
                del self.controllers
            self.controllers = self._initialize_controllers(self.task, self.task_cfg, self.world)

        # del self.controllers
        # self.controllers = self._initialize_controllers(self.task, self.task_cfg, self.world)
        self._reset_controllers(self.controllers)

        # Reset skills
        if hasattr(self, "skills"):
            del self.skills

        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)

        # Warmup
        for _ in range(20):
            self.world.get_observations()
            self._init_static_objects(self.task)
            self.world.step(render=False)

        if self.task_cfg.get("fluid", None):
            self.task._set_fluid()
            # Fluid need additional warmup
            for _ in range(150):
                self.world.step(render=False)

        self._initialize_world_recorder()

        self.logger.clear(
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
        )

        # episode_stats["current_times"] += 1

    def randomization(self, layout_path=None) -> bool:
        try:
            if layout_path is None:
                # Individual Reset
                self.task.individual_reset()
                self._randomization_layout()
            else:
                with open(layout_path, "rb") as f:
                    data = pickle.load(f)
                self.data = data
                self.randomization_from_mem(data)
            return True
        except Exception as e:
            raise e

    def update_skill_states(self, skills, episode_success, should_continue):
        """Update and manage skill states."""
        current_skills = skills[0]

        # Check if any skills remain
        if not any(current_skills.values()):
            skills.pop(0)
            if skills:
                should_continue = self.plan_first_skill(skills, should_continue)
            return episode_success, should_continue

        # Update each robot's skills
        for _, skill_sequences in current_skills.items():
            if not skill_sequences:
                continue

            # Update all skills first
            for lr_skill_list in skill_sequences[0]:
                if lr_skill_list:
                    start_lr_skill = lr_skill_list[0]
                    start_lr_skill.update()  # Must update regardless of completion
                    if start_lr_skill.is_done():
                        if not start_lr_skill.is_success():
                            episode_success = False
                            should_continue = False
                        lr_skill_list.remove(start_lr_skill)

                        if lr_skill_list:
                            next_skill = lr_skill_list[0]
                            next_skill.simple_generate_manip_cmds()
                            if hasattr(next_skill, "visualize_target"):
                                next_skill.visualize_target(self.world)
                            if len(next_skill.manip_list) == 0:
                                should_continue = not next_skill.is_ready()
                    if hasattr(start_lr_skill, "visualize_target"):
                        start_lr_skill.visualize_target(self.world)

            # Remove empty skill sequences
            completed_skills = []
            for lr_skill_list in skill_sequences[0]:
                if not lr_skill_list:
                    completed_skills.append(lr_skill_list)
            for completed_skill in completed_skills:
                skill_sequences[0].remove(completed_skill)

            # Move to next sequence if current is empty
            if not skill_sequences[0]:
                skill_sequences.pop(0)
                if skill_sequences:
                    for skill in skill_sequences[0]:
                        skill[0].simple_generate_manip_cmds()
                        if len(skill[0].manip_list) == 0:
                            should_continue = not skill[0].is_ready()
        return episode_success, should_continue

    def plan_first_skill(self, skills, should_continue):
        for _, robot_skill_list in skills[0].items():
            for lr_skill_list in robot_skill_list[0]:
                lr_skill_list[0].simple_generate_manip_cmds()
                if hasattr(lr_skill_list[0], "visualize_target"):
                    lr_skill_list[0].visualize_target(self.world)
                if len(lr_skill_list[0].manip_list) == 0:
                    should_continue = not lr_skill_list[0].is_ready()
        return should_continue

    def generate_seq(self) -> list:
        end = False

        # while True:
        #     obs = self.world.get_observations()
        #     # self._init_static_objects(self.task)
        #     self.world.step(render=True)

        step_id = 0
        episode_success = True
        should_continue = True
        max_episode_length = self.task_cfg["data"]["max_episode_length"]
        episode_stats = {"succeed_times": 0, "current_times": 0}

        should_continue = self.plan_first_skill(self.skills, should_continue)

        # Warmup
        for _ in range(10):
            obs = self.world.get_observations()
            # self._init_static_objects(self.task)
            self.world.step(render=False)

        while not (step_id >= max_episode_length or (not self.skills and not episode_success) or (not should_continue)):
            obs = self.world.get_observations()
            action_dict = {}
            record_flag = True
            if self.skills and should_continue:
                # Process current skills
                current_skills = self.skills[0]
                for robot_name, skill_sequences in current_skills.items():
                    if skill_sequences and skill_sequences[0]:
                        action = [
                            skill[0].controller.forward(skill[0].manip_list[0])
                            for skill in skill_sequences[0]
                            if skill[0] and skill[0].is_ready()
                        ]

                        feasible_labels = [skill[0].is_feasible() for skill in skill_sequences[0] if skill[0]]
                        record_labels = [skill[0].is_record() for skill in skill_sequences[0] if skill[0]]

                        if False in feasible_labels:
                            should_continue = False
                        if False in record_labels:
                            record_flag = False

                        if action:
                            action_dict[robot_name] = {
                                "joint_positions": np.concatenate([a["joint_positions"] for a in action]),
                                "joint_indices": np.concatenate([a["joint_indices"] for a in action]),
                                "raw_action": action,
                            }
            elif not self.skills and episode_success:
                print("Task is successful")
                end = True
                for j_idx in range(1, 7):
                    self.world.step(render=False)
                    obs = self.world.get_observations()
                    log_dual_obs(self.logger, obs, action_dict, self.controllers, step_idx=step_id + j_idx)
                    self.world_recorder.record()

                episode_stats["succeed_times"] += 1
                should_continue = False

            if record_flag:
                log_dual_obs(self.logger, obs, action_dict, self.controllers, step_idx=step_id)
                self.world_recorder.record()
            self.task.apply_action(action_dict)
            self.world.step(render=False)

            step_id += 1
            if self.skills:
                episode_success, should_continue = self.update_skill_states(
                    self.skills, episode_success, should_continue
                )

        if end:
            if self.step_replay:
                return [None] * step_id
            else:
                # Prim poses mode: return recorded poses for compatibility
                return self.world_recorder.prim_poses
        else:
            return []

    def recover_seq(self, seq_path):
        data = self.data
        return self.recover_seq_from_mem(data)

    def _record_rgb_depth(self, step_idx: int):
        for key, value in self.task.cameras.items():
            for robot_name, _ in self.task.robots.items():
                if robot_name in key:
                    camera_obs = value.get_observations()
                    rgb_img = camera_obs["color_image"]
                    # Special processing if enabled
                    camera2env_pose = camera_obs["camera2env_pose"]
                    save_camera_name = key.replace(f"{robot_name}_", "")
                    self.logger.add_color_image(
                        robot_name, "images.rgb." + save_camera_name, rgb_img, step_idx=step_idx
                    )
                    if "depth_image" in camera_obs:
                        depth_img = camera_obs["depth_image"]
                        depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
                        self.logger.add_depth_image(
                            robot_name, "images.depth." + save_camera_name, depth_img, step_idx=step_idx
                        )
                    if "semantic_mask" in camera_obs:
                        seg_mask = camera_obs["semantic_mask"]
                        self.logger.add_seg_image(
                            robot_name, "images.seg." + save_camera_name, seg_mask, step_idx=step_idx
                        )
                        if "semantic_mask_id2labels" in camera_obs:
                            self.logger.add_scalar_data(
                                robot_name,
                                "labels.seg." + save_camera_name,
                                camera_obs["semantic_mask_id2labels"],
                            )
                    if "bbox2d_tight" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name, "labels.bbox2d_tight." + save_camera_name, camera_obs["bbox2d_tight"]
                        )
                    if "bbox2d_tight_id2labels" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name,
                            "labels.bbox2d_tight_id2labels." + save_camera_name,
                            camera_obs["bbox2d_tight_id2labels"],
                        )
                    if "bbox2d_loose" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name, "labels.bbox2d_loose." + save_camera_name, camera_obs["bbox2d_loose"]
                        )
                    if "bbox2d_loose_id2labels" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name,
                            "labels.bbox2d_loose_id2labels." + save_camera_name,
                            camera_obs["bbox2d_loose_id2labels"],
                        )
                    if "bbox3d" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name, "labels.bbox3d." + save_camera_name, camera_obs["bbox3d"]
                        )
                    if "bbox3d_id2labels" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name,
                            "labels.bbox3d_id2labels." + save_camera_name,
                            camera_obs["bbox3d_id2labels"],
                        )
                    if self.log_motion_vectors and "motion_vectors" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name, "labels.motion_vectors." + save_camera_name, camera_obs["motion_vectors"]
                        )
                    self.logger.add_scalar_data(
                        robot_name, "camera2env_pose." + save_camera_name, camera2env_pose
                    )
                    if step_idx == 0:
                        save_camera_name = key.replace(f"{robot_name}_", "")
                        self.logger.add_json_data(
                            robot_name, f"{save_camera_name}_camera_params", camera_obs["camera_params"]
                        )

                    # depth_img = get_src(value, "depth")
                    # depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)

                    # # Initialize lists for new camera keys
                    # if key not in self.rgb:
                    #     self.rgb[key] = []
                    # if key not in self.depth:
                    #     self.depth[key] = []

                    # # Append current frame to the corresponding camera's list
                    # self.rgb[key].append(rgb_img)
                    # self.depth[key].append(depth_img)

    def seq_replay(self, sequence: list) -> int:
        """
        Replay recorded sequence with mode-specific data preparation.

        Returns:
            int: Number of steps replayed
        """
        if not self.step_replay:
            self.world_recorder.prim_poses = sequence

        # warmup before replay formally
        self.world_recorder.warmup()

        # Get total steps from WorldRecorder
        total_steps = self.world_recorder.get_total_steps()
        step_idx = 0

        # Unified replay loop - WorldRecorder handles rendering internally
        with tqdm(total=total_steps, desc="Replay Progress") as pbar:
            while not self.world_recorder.replay():
                # Record RGB/depth at current step
                self._record_rgb_depth(step_idx)
                step_idx += 1
                pbar.update(1)

        self.length = total_steps
        print("Replay finished.")
        return total_steps

    def get_task_name(self):
        return self.task_cfg["task"]

    def save_seq(self, save_path: str) -> int:
        ser_bytes = self.dump_plan_info()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        save_path = os.path.join(save_path, "plan")
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, f"{timestamp}.pkl")
        with open(path, "wb") as f:
            f.write(ser_bytes)
        return self.world_recorder.get_total_steps()

    def save(self, save_path: str) -> int:
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        self.logger.save(save_path, timestamp, save_img=True)

        # ── Safety risk pipeline (optional) ──
        # 串行链路: sim_raw_gt.json → sim_features.json → sim_labels.json → safety_report.json
        if self._safety_eval_enabled:
            self._run_safety_pipeline(save_path, timestamp)

        return self.length

    def _init_obstacle_origin(self):
        """Record the obstacle's initial position for round-trip movement."""
        obs_cfg = self._safety_eval_cfg.get("obstacle", {})
        obs_name = obs_cfg.get("name", "obstacle_1")
        if hasattr(self.task, 'objects') and obs_name in self.task.objects:
            pos, _ = self.task.objects[obs_name].get_world_pose()
            self._obstacle_origin = [pos[0], pos[1]]
            # heading_to_target: True = going toward target, False = going back to origin
            self._obstacle_heading_to_target = True
        else:
            self._obstacle_origin = None

    def _move_hand_obstacle(self, step_id: int):
        """Move hand obstacle during simulation.

        Config is read from task_cfg.safety_eval.obstacle:
            enabled: bool - whether to move the obstacle
            name: str - object name (default "obstacle_1")
            target: [x, y, z] - target position
            speed: float - movement speed in m/step
            fixed_z: float - fixed height
            mode: str - "once" (stop at target) or "round_trip" (bounce back and forth)
        """
        obs_cfg = self._safety_eval_cfg.get("obstacle", {})
        if not obs_cfg.get("enabled", True):
            return

        obs_name = obs_cfg.get("name", "obstacle_1")
        if not hasattr(self.task, 'objects') or obs_name not in self.task.objects:
            return

        try:
            obj = self.task.objects[obs_name]
            current_pos, current_ori = obj.get_world_pose()

            fixed_z = obs_cfg.get("fixed_z", 0.80)
            target = obs_cfg.get("target", [-0.10, -0.40, fixed_z])
            speed = obs_cfg.get("speed", 0.005)
            mode = obs_cfg.get("mode", "round_trip")
            arrive_threshold = 0.02  # 2cm

            # Determine current destination
            if mode == "round_trip":
                if not hasattr(self, '_obstacle_origin') or self._obstacle_origin is None:
                    self._init_obstacle_origin()
                origin = getattr(self, '_obstacle_origin', None)
                heading_to_target = getattr(self, '_obstacle_heading_to_target', True)

                if origin is not None:
                    if heading_to_target:
                        dest = [target[0], target[1]]
                    else:
                        dest = origin

                    # Check if arrived at destination → flip direction
                    dx = dest[0] - current_pos[0]
                    dy = dest[1] - current_pos[1]
                    if math.sqrt(dx*dx + dy*dy) < arrive_threshold:
                        self._obstacle_heading_to_target = not heading_to_target
                        # Update dest for this step
                        dest = origin if heading_to_target else [target[0], target[1]]
                else:
                    dest = [target[0], target[1]]
            else:
                dest = [target[0], target[1]]

            dx = dest[0] - current_pos[0]
            dy = dest[1] - current_pos[1]
            dist_xy = math.sqrt(dx*dx + dy*dy)

            if dist_xy > 0.005:
                nx = dx / dist_xy
                ny = dy / dist_xy
                new_pos = [
                    current_pos[0] + nx * speed,
                    current_pos[1] + ny * speed,
                    fixed_z,
                ]
                obj.set_world_pose(new_pos, current_ori)
                if hasattr(obj, 'set_linear_velocity'):
                    obj.set_linear_velocity([0.0, 0.0, 0.0])
                if hasattr(obj, 'set_angular_velocity'):
                    obj.set_angular_velocity([0.0, 0.0, 0.0])
            else:
                # At destination: hold position
                obj.set_world_pose([dest[0], dest[1], fixed_z], current_ori)
                if hasattr(obj, 'set_linear_velocity'):
                    obj.set_linear_velocity([0.0, 0.0, 0.0])
        except Exception:
            pass

    def _run_safety_pipeline(self, save_path: str, timestamp: str):
        """Run the complete safety risk pipeline: Raw_GT → Features → Labels → Report."""
        import json
        from pathlib import Path

        # Find episode directory
        first_robot = next(iter(self.logger.proprio_data_logger), None)
        if first_robot is None:
            return

        episode_dir = (
            Path(save_path) / first_robot /
            self.logger.task_dir / self.logger.collect_info / timestamp
        )
        if not (episode_dir / "meta_info.pkl").exists():
            return

        episode_id = f"{self.get_task_name()}_{timestamp}"
        report_subdir = self._safety_eval_cfg.get("output_subdir", "safety_reports")

        try:
            # ── Step 1: Sim_Raw_GT ──
            from safety_risk.raw_gt_extractor import SimRawGTExtractor

            # Build task_cfg dict for the extractor
            extractor_cfg = dict(self.task_cfg)
            extractor_cfg["random_seed"] = self.random_seed
            extractor_cfg["language_instruction"] = getattr(self.logger, "language_instruction", [""])[0] if hasattr(self.logger, "language_instruction") else ""

            raw_extractor = SimRawGTExtractor()
            raw_gt = raw_extractor.extract_from_lmdb(str(episode_dir), task_cfg=extractor_cfg)

            # Inject episode_id (timestamp-dependent, can't be in task_cfg)
            raw_gt["episode_meta"]["episode_id"] = episode_id

            # Inject every intended pick-object ID.  Retain the legacy singular
            # fields for consumers that have not migrated to dual-arm episodes.
            target_object_ids = [
                obj.get("name", "")
                for obj in self.task_cfg.get("objects", [])
                if obj.get("name", "").startswith("pick_object")
            ]
            if target_object_ids:
                raw_gt["episode_meta"]["object_id"] = target_object_ids[0]
                raw_gt["episode_meta"]["target_object_id"] = target_object_ids[0]
                raw_gt["episode_meta"]["target_object_ids"] = target_object_ids

            # ── Inject PhysX data into raw_gt ──
            if hasattr(self, '_physx_collector') and self._physx_collector is not None:
                physx_data = self._physx_collector.get_raw_data()

                raw_gt["robot_state"]["joint_torque_gt"] = physx_data.get("joint_torque_gt")
                raw_gt["robot_state"]["link_pose_gt"] = physx_data.get("link_pose_gt")
                raw_gt["robot_state"]["link_velocity_gt"] = physx_data.get("link_velocity_gt")
                raw_gt["collision_gt"]["collision_pair_gt"] = physx_data.get("collision_pair_gt")
                raw_gt["collision_gt"]["collision_location_gt"] = physx_data.get("collision_location_gt")
                raw_gt["collision_gt"]["penetration_depth_gt"] = physx_data.get("penetration_depth_gt")
                raw_gt["collision_gt"]["contact_force_gt"] = physx_data.get("contact_force_gt")
                raw_gt["collision_gt"]["contact_impulse_gt"] = physx_data.get("contact_impulse_gt")
                coverage_status = physx_data.get("contact_coverage_status", {})
                collected_steps = len(physx_data.get("step_ids", []))
                complete_safety_matrix = bool(
                    coverage_status.get("configured")
                    and collected_steps > 0
                    and coverage_status.get("failed_steps", 0) == 0
                    and coverage_status.get("successful_steps", 0) == collected_steps
                )
                object_env_complete = complete_safety_matrix
                robot_env_complete = complete_safety_matrix
                object_human_complete = complete_safety_matrix
                self_complete = complete_safety_matrix
                raw_gt["collision_gt"]["_provenance"] = {
                    "coverage": {
                        "human": "complete_robot_rigid_bodies_to_configured_human_obstacles",
                        "robot_human": "complete_robot_rigid_bodies_to_configured_human_obstacles",
                        "ee_human": "complete_robot_rigid_bodies_to_configured_human_obstacles",
                        "object_human": (
                            "complete_intended_objects_to_configured_human_obstacles"
                            if object_human_complete else "not_collected"
                        ),
                        "object_env": (
                            "complete_intended_objects_to_configured_environment"
                            if object_env_complete else "not_collected"
                        ),
                        "robot_env": (
                            "complete_robot_rigid_bodies_to_configured_environment"
                            if robot_env_complete else "not_collected"
                        ),
                        "self": (
                            "complete_unordered_robot_rigid_body_pairs"
                            if self_complete else "not_collected"
                        ),
                    },
                    "source": "single complete PhysX RigidContactView matrix",
                    "runtime_validation": coverage_status,
                    "contact_values_source": "PhysX contact-report per-point impulse vectors",
                    "contact_report_validation": physx_data.get("contact_report_status", {}),
                    "runtime_physics": physx_data.get("runtime_physics", {}),
                    "force_unit": "N (contact-report impulse vector norm divided by measured physics_dt)",
                    "impulse_unit": "N*s (direct PhysX contact-report impulse vector norm)",
                }

                # Compute contact_duration_gt from contact events
                physx_summary = self._physx_collector.finalize()
                contact_events = physx_summary.get("contact_events", [])
                if contact_events:
                    raw_gt["collision_gt"]["contact_duration_gt"] = contact_events

                raw_gt["distance_gt"]["object_env_distance_gt"] = physx_data.get("object_env_distance_gt")
                raw_gt["distance_gt"]["link_env_distance_gt"] = physx_data.get("link_env_distance_gt")
                raw_gt["distance_gt"]["self_distance_gt"] = physx_data.get("self_distance_gt")
                # The current collector computes Euclidean separation between
                # prim/link origins, not geometry-surface clearance. Record
                # this explicitly so downstream extraction cannot mistake the
                # values for exact S-DIST contract GT.
                raw_gt["distance_gt"].setdefault("_provenance", {}).update({
                    "object_env_distance_gt": {"metric": "origin_euclidean"},
                    "link_env_distance_gt": {"metric": "origin_euclidean"},
                    "self_distance_gt": {"metric": "origin_euclidean"},
                })

                # Rebuild EE poses from full per-step PhysX link_pose_gt before
                # recomputing human distances.  LMDB T_base_ee_fl/fr can be scoped
                # to manipulation segments and is often shorter than the sim.
                try:
                    raw_extractor._rebuild_ee_poses_from_link_pose_gt(raw_gt)
                    ee_left = raw_gt.get("robot_state", {}).get("ee_pose_gt") or []
                    ee_right = raw_gt.get("robot_state", {}).get("ee_pose_right_gt") or []
                    if ee_left or ee_right:
                        print(f"[safety_risk] ee_pose_gt rebuilt from link_pose_gt (left={len(ee_left)}, right={len(ee_right)})")
                except Exception as e:
                    print(f"[safety_risk] Warning: ee_pose_gt rebuild failed: {e}")

                # Recompute S-DIST-001/002/003 after PhysX link_pose_gt and rebuilt
                # EE pose have been injected. The extractor initially runs before
                # PhysX data is merged, so these distances may otherwise be stale or
                # shorter than the simulation timeline.
                try:
                    raw_extractor._compute_human_distances_from_obstacles(raw_gt)
                    matrix = raw_gt.get("distance_gt", {}).get("robot_human_distance_matrix_gt")
                    ee_human = raw_gt.get("distance_gt", {}).get("ee_human_distance_gt")
                    if matrix:
                        print(f"[safety_risk] robot_human_distance_matrix_gt recomputed ({len(matrix)} frames)")
                    if ee_human:
                        print(f"[safety_risk] ee_human_distance_gt recomputed ({len(ee_human)} frames)")
                    raw_gt["distance_gt"].setdefault("_provenance", {}).update({
                        "robot_human_distance_matrix_gt": {"metric": "origin_euclidean"},
                        "ee_human_distance_gt": {"metric": "origin_euclidean"},
                        "object_human_distance_gt": {"metric": "origin_euclidean"},
                    })
                except Exception as e:
                    print(f"[safety_risk] Warning: human distance recompute failed: {e}")

                # Planner data
                raw_gt["planner_log"]["planned_trajectory"] = physx_data.get("planned_trajectory")
                raw_gt["planner_log"]["safety_gate_status"] = physx_data.get("safety_gate_status")
                raw_gt["planner_log"]["low_level_command_sent"] = physx_data.get("low_level_command_sent")

                # Safety gate / stop event data
                raw_gt["planner_log"]["stop_success"] = physx_data.get("stop_success")
                raw_gt["planner_log"]["stop_margin_s"] = physx_data.get("stop_margin_s")
                raw_gt["planner_log"]["t_stop_s"] = physx_data.get("t_stop_s")

                # Gripper-object contact force
                raw_gt["gripper_gt"]["gripper_object_contact_force_gt"] = physx_data.get("gripper_object_contact_force_gt")

                print(f"[safety_risk] PhysX data injected into Sim_Raw_GT")

            # Joint effort limits are read from the live articulation and kept
            # with DOF indices/names.  This is required to compare the 28-value
            # measured-effort vector with the 12 arm-joint limits correctly.
            try:
                torque_limits = self._read_robot_torque_limits()
                if torque_limits:
                    raw_gt["episode_meta"].setdefault("physics_config", {})[
                        "joint_torque_limits_nm_by_index"
                    ] = torque_limits
                    print(f"[safety_risk] joint torque limits injected ({len(torque_limits)} arm DOFs)")
                gripper_widths = self._read_gripper_max_widths()
                if gripper_widths:
                    raw_gt["episode_meta"].setdefault("physics_config", {})[
                        "gripper_max_width_m_by_arm"
                    ] = gripper_widths
            except Exception as e:
                print(f"[safety_risk] Warning: joint torque limit read failed: {e}")

            # ── Inject USD physical params (S-OBJ-004) ──
            try:
                physical_params = self._read_object_physical_params()
                if physical_params:
                    raw_gt["object_state"]["object_physical_params"] = physical_params
                    print(f"[safety_risk] object_physical_params injected ({len(physical_params)} objects)")
            except Exception as e:
                print(f"[safety_risk] Warning: object_physical_params read failed: {e}")

            # ── Inject scene mesh info (S-ENV-001) ──
            try:
                scene_mesh = self._read_scene_mesh_info()
                if scene_mesh:
                    raw_gt["environment_state"]["scene_mesh_gt"] = scene_mesh
                    print(f"[safety_risk] scene_mesh_gt injected ({len(scene_mesh)} fixtures)")
            except Exception as e:
                print(f"[safety_risk] Warning: scene_mesh_gt read failed: {e}")

            # ── Inject the exact placement region used by the task planner and
            # derive final placement/stability from recorded poses. ──
            try:
                placement_region = self._read_placement_target_region()
                if placement_region:
                    raw_gt["environment_state"]["placement_target_region_gt"] = placement_region
                    placement_metrics = self._compute_final_placement_metrics(raw_gt, placement_region)
                    raw_gt["outcome_gt"].update(placement_metrics)
                    drop_metrics = self._compute_physical_grasp_drop_metrics(raw_gt)
                    raw_gt["outcome_gt"].update(drop_metrics)
                    print(
                        "[safety_risk] placement region and final-state metrics injected "
                        f"({len(placement_metrics.get('placement_error_by_object_gt', {}))} objects)"
                    )
            except Exception as e:
                print(f"[safety_risk] Warning: placement/final-state computation failed: {e}")

            # ── Compute sensor fields from segmentation (S-SENSOR-004/005/006) ──
            try:
                sensor_fields = self._compute_sensor_fields_from_seg(str(episode_dir), raw_gt)
                if sensor_fields:
                    raw_gt["sensor_gt"].update(sensor_fields)
                    n_frames = len(sensor_fields.get("instance_id_map_gt", []))
                    print(f"[safety_risk] sensor fields injected ({n_frames} frames: bbox, visibility, instance_id)")
                else:
                    print(f"[safety_risk] sensor fields: no segmentation data found")
            except Exception as e:
                print(f"[safety_risk] Warning: sensor fields computation failed: {e}")

            raw_gt_path = episode_dir / "sim_raw_gt.json"
            with open(raw_gt_path, "w", encoding="utf-8") as f:
                json.dump(raw_gt, f, indent=2, ensure_ascii=False, default=str)
            print(f"[safety_risk] 1/4 Sim_Raw_GT → {raw_gt_path}")

            # ── Step 2: Sim_Features (from Raw_GT) ──
            from safety_risk.sim_feature_extractor import SimFeatureExtractor

            feature_extractor = SimFeatureExtractor()
            features = feature_extractor.extract(raw_gt)

            features_path = episode_dir / "sim_features.json"
            with open(features_path, "w", encoding="utf-8") as f:
                json.dump(features, f, indent=2, ensure_ascii=False, default=str)
            print(f"[safety_risk] 2/4 Sim_Features → {features_path}")

            # ── Step 3: Sim_Labels (from Raw_GT + Features) ──
            from safety_risk.sim_label_extractor import SimLabelExtractor

            label_extractor = SimLabelExtractor()
            labels = label_extractor.extract(raw_gt, features)

            labels_path = episode_dir / "sim_labels.json"
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(labels, f, indent=2, ensure_ascii=False, default=str)
            print(f"[safety_risk] 3/4 Sim_Labels → {labels_path}")

            # ── Step 4: Safety Report (from Labels) ──
            report_dir = episode_dir / report_subdir
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"{episode_id}_risk.json"

            report = self._build_report_from_labels(episode_id, raw_gt, features, labels)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            print(f"[safety_risk] 4/4 Safety Report → {report_path}")

        except Exception as e:
            print(f"[safety_risk] Warning: pipeline failed: {e}")

    def _read_object_physical_params(self) -> dict:
        """S-OBJ-004: Export traceable physical parameters for task objects.

        Dynamic rigid-body mass properties come from the live PhysX tensor
        views. USD is used for explicitly authored material properties and as a
        fallback only when a runtime view is unavailable. Zero USD inertia is
        treated as an instruction for PhysX to compute inertia, not as data.

        Units: kg, dimensionless friction coefficient, kg*m^2, and meters.
        """
        from omni.isaac.core.articulations import ArticulationView
        from omni.isaac.core.prims import RigidPrimView
        from pxr import Usd, UsdGeom, UsdPhysics

        result = {}
        stage = self.world.stage
        if stage is None:
            return result

        def _to_float(value):
            try:
                if isinstance(value, bool):
                    return None
                return float(value)
            except Exception:
                return None

        def _to_float_list(value):
            if value is None:
                return None
            try:
                return [float(v) for v in value]
            except Exception:
                single = _to_float(value)
                return [single] if single is not None else None

        def _to_numpy(value):
            if value is None:
                return None
            try:
                if hasattr(value, "detach"):
                    value = value.detach()
                if hasattr(value, "cpu"):
                    value = value.cpu()
                if hasattr(value, "numpy"):
                    value = value.numpy()
                return np.asarray(value, dtype=np.float64)
            except Exception:
                return None

        def _walk_prims(root_prim):
            try:
                return list(Usd.PrimRange(root_prim))
            except Exception:
                return [root_prim]

        def _attr_value(prim, attr_names):
            for attr_name in attr_names:
                try:
                    attr = prim.GetAttribute(attr_name)
                    if attr and attr.HasValue():
                        value = attr.Get()
                        if value is not None:
                            return value, attr_name, str(prim.GetPath())
                except Exception:
                    continue
            return None, None, None

        def _first_attr(root_prim, attr_names):
            for prim in _walk_prims(root_prim):
                value, attr_name, prim_path = _attr_value(prim, attr_names)
                if value is not None:
                    return value, attr_name, prim_path
            return None, None, None

        def _valid_inertia(value):
            """Reject unset/placeholder inertia values authored as zero."""
            inertia = _to_float_list(value)
            if inertia is None or len(inertia) < 3:
                return None
            inertia = inertia[:3]
            if not all(math.isfinite(component) and component > 0.0 for component in inertia):
                return None
            return inertia

        def _read_authored_mass_kg(root_prim, obj_cfg):
            value, attr_name, prim_path = _attr_value(root_prim, ["physics:mass", "mass"])
            mass = _to_float(value)
            if mass is not None:
                return mass, f"usd:{prim_path}:{attr_name}"

            masses = []
            mass_sources = []
            for prim in _walk_prims(root_prim):
                if prim == root_prim:
                    continue
                value, attr_name, prim_path = _attr_value(prim, ["physics:mass", "mass"])
                mass = _to_float(value)
                if mass is not None:
                    masses.append(mass)
                    mass_sources.append(f"{prim_path}:{attr_name}")
            if masses:
                return float(sum(masses)), "usd_sum:" + ",".join(mass_sources)

            for key in ["mass_kg", "mass"]:
                mass = _to_float(obj_cfg.get(key))
                if mass is not None:
                    return mass, f"task_config:{key}"
            return None, "missing"

        def _read_authored_inertia(root_prim):
            value, attr_name, prim_path = _attr_value(
                root_prim, ["physics:diagonalInertia", "diagonalInertia"]
            )
            inertia = _valid_inertia(value)
            if inertia is not None:
                return inertia, f"usd:{prim_path}:{attr_name}"
            return None, "missing_or_zero_placeholder"

        def _friction_from_config(obj_cfg, component):
            physical_params = obj_cfg.get("physical_params", {}) or {}
            friction = physical_params.get("friction", {}) or {}
            return _to_float(friction.get(component))

        def _material_targets(root_prim):
            targets = []
            seen = set()
            for prim in _walk_prims(root_prim):
                try:
                    relationships = prim.GetRelationships()
                except Exception:
                    relationships = []
                for rel in relationships:
                    if "material" not in rel.GetName().lower():
                        continue
                    try:
                        rel_targets = rel.GetTargets()
                    except Exception:
                        rel_targets = []
                    for target in rel_targets:
                        target_path = str(target)
                        if target_path and target_path not in seen:
                            seen.add(target_path)
                            targets.append(target_path)
            return targets

        def _first_material_attr(root_prim, attr_names):
            value, attr_name, prim_path = _first_attr(root_prim, attr_names)
            if value is not None:
                return value, f"usd:{prim_path}:{attr_name}"
            for material_path in _material_targets(root_prim):
                material_prim = stage.GetPrimAtPath(material_path)
                if material_prim and material_prim.IsValid():
                    value, attr_name, prim_path = _attr_value(material_prim, attr_names)
                    if value is not None:
                        return value, f"usd_physics_material:{prim_path}:{attr_name}"
            return None, "unbound_or_missing"

        def _matrix_and_principal_moments(flat_matrix):
            array = _to_numpy(flat_matrix)
            if array is None or array.size < 9:
                return None, None
            matrix = array.reshape((-1, 9))[0].reshape((3, 3))
            matrix = 0.5 * (matrix + matrix.T)
            if not np.all(np.isfinite(matrix)):
                return None, None
            try:
                moments = np.linalg.eigvalsh(matrix)
            except Exception:
                return None, None
            if not np.all(np.isfinite(moments)) or np.any(moments <= 0.0):
                return None, None
            return matrix.tolist(), moments.tolist()

        def _read_runtime_rigid_body(obj, rigid_prim_path, obj_name):
            """Read the live PhysX body, rebuilding a stale wrapper view if needed."""

            def _read_view(view):
                if view is None:
                    return None
                try:
                    if not view.is_physics_handle_valid():
                        view.initialize(self.world.physics_sim_view)
                except Exception:
                    # Older Isaac Sim view variants do not expose the validity
                    # helper; initialize against the active simulation anyway.
                    view.initialize(self.world.physics_sim_view)

                masses = _to_numpy(view.get_masses())
                inertia_matrix, moments = _matrix_and_principal_moments(view.get_inertias())
                if masses is None or masses.size == 0 or inertia_matrix is None:
                    return None
                mass = float(masses.reshape(-1)[0])
                if not math.isfinite(mass) or mass <= 0.0:
                    return None

                center_of_mass = None
                principal_axes = None
                try:
                    com_result = view.get_coms()
                    if com_result is not None:
                        positions = _to_numpy(com_result[0])
                        orientations = _to_numpy(com_result[1])
                        if positions is not None and positions.size >= 3:
                            center_of_mass = positions.reshape((-1, 3))[0].tolist()
                        if orientations is not None and orientations.size >= 4:
                            principal_axes = orientations.reshape((-1, 4))[0].tolist()
                except Exception:
                    pass

                return {
                    "mass_kg": mass,
                    "inertia_kg_m2": moments,
                    "inertia_matrix_kg_m2": inertia_matrix,
                    "center_of_mass_m": center_of_mass,
                    "principal_axes_quat_wxyz": principal_axes,
                }

            errors = []
            existing_view = getattr(obj, "_rigid_prim_view", None)
            if existing_view is not None:
                try:
                    runtime_data = _read_view(existing_view)
                    if runtime_data is not None:
                        return runtime_data
                    errors.append("existing view returned no valid mass/inertia")
                except Exception as exc:
                    errors.append(f"existing view: {exc}")

            # RigidObject views are invalidated by the hard resets used during
            # layout randomization. A fresh read-only view bound to the current
            # simulation makes the exported values reflect the saved episode.
            try:
                fresh_view = RigidPrimView(
                    prim_paths_expr=rigid_prim_path,
                    name=f"physical_params_{obj_name}",
                    reset_xform_properties=False,
                    prepare_contact_sensors=False,
                )
                fresh_view.initialize(self.world.physics_sim_view)
                runtime_data = _read_view(fresh_view)
                if runtime_data is not None:
                    return runtime_data
                errors.append("fresh view returned no valid mass/inertia")
            except Exception as exc:
                errors.append(f"fresh view: {exc}")

            print(
                f"[safety_risk] Warning: PhysX rigid-body read failed for {obj_name} "
                f"at {rigid_prim_path}: {'; '.join(errors)}"
            )
            return None

        def _find_articulation_root(root_prim):
            for candidate in _walk_prims(root_prim):
                try:
                    if candidate.HasAPI(UsdPhysics.ArticulationRootAPI):
                        return candidate
                except Exception:
                    continue
            return None

        def _read_runtime_articulation(articulation_prim, obj_name):
            try:
                view = ArticulationView(
                    prim_paths_expr=str(articulation_prim.GetPath()),
                    name=f"physical_params_{obj_name}",
                )
                view.initialize(self.world.physics_sim_view)
                masses = _to_numpy(view.get_body_masses())
                inertias = _to_numpy(view.get_body_inertias())
                if masses is None or inertias is None:
                    return None
                masses = masses.reshape((view.count, view.num_bodies))[0]
                inertias = inertias.reshape((view.count, view.num_bodies, 9))[0]
                body_names = list(view.body_names or [])
                bodies = []
                for index, mass in enumerate(masses):
                    matrix, moments = _matrix_and_principal_moments(inertias[index])
                    bodies.append({
                        "name": body_names[index] if index < len(body_names) else f"body_{index}",
                        "mass_kg": float(mass),
                        "inertia_kg_m2": moments,
                        "inertia_matrix_kg_m2": matrix,
                    })
                return {
                    "mass_kg": float(np.sum(masses)),
                    "body_count": len(bodies),
                    "bodies": bodies,
                }
            except Exception as exc:
                print(f"[safety_risk] Warning: PhysX articulation read failed for {obj_name}: {exc}")
                return None

        for obj_cfg in self.task_cfg.get("objects", []):
            obj_name = obj_cfg.get("name", "")
            if not obj_name:
                continue

            prim_path_candidates = []
            obj = getattr(self.task, "objects", {}).get(obj_name) if hasattr(self, "task") else None
            obj_prim_path = getattr(obj, "prim_path", None) if obj is not None else None
            if obj_prim_path:
                prim_path_candidates.append(str(obj_prim_path))
            prim_path_candidates.extend([f"/World/task_0/{obj_name}", f"/World/{obj_name}"])

            prim = None
            for candidate in prim_path_candidates:
                candidate_prim = stage.GetPrimAtPath(candidate)
                if candidate_prim and candidate_prim.IsValid():
                    prim = candidate_prim
                    break
            if not prim or not prim.IsValid():
                continue

            size_m = None
            try:
                bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
                bbox = bbox_cache.ComputeLocalBound(prim)
                size = bbox.ComputeAlignedRange().GetSize()
                scale = obj.get_world_scale() if obj is not None else [1.0, 1.0, 1.0]
                size_m = [abs(float(size[i]) * float(scale[i])) for i in range(3)]
            except Exception:
                size_m = None

            authored_mass_kg, authored_mass_source = _read_authored_mass_kg(prim, obj_cfg)
            static_value, static_source = _first_material_attr(prim, [
                "physics:staticFriction", "physxMaterial:staticFriction", "staticFriction",
            ])
            dynamic_value, dynamic_source = _first_material_attr(prim, [
                "physics:dynamicFriction", "physxMaterial:dynamicFriction", "dynamicFriction",
            ])
            static_friction = _to_float(static_value)
            dynamic_friction = _to_float(dynamic_value)
            configured_friction = {
                "static": _friction_from_config(obj_cfg, "static"),
                "dynamic": _friction_from_config(obj_cfg, "dynamic"),
            }

            target_class = obj_cfg.get("target_class", "")
            articulation_prim = _find_articulation_root(prim)
            common = {
                "asset_path": obj_cfg.get("path"),
                "prim_path": str(prim.GetPath()),
                "friction": {
                    "static": static_friction,
                    "dynamic": dynamic_friction,
                },
                "configured_friction_not_applied": configured_friction,
                "size_m": size_m,
                "sources": {
                    "friction_static": static_source,
                    "friction_dynamic": dynamic_source,
                    "size": "usd_local_bbox_x_runtime_world_scale",
                },
            }

            if target_class == "GeometryObject":
                common.update({
                    "physical_role": "static_collider",
                    "mass_kg": None,
                    "authored_mass_kg": authored_mass_kg,
                    "inertia_kg_m2": None,
                    "inertia_matrix_kg_m2": None,
                    "inertia_method": "not_applicable_static_collider",
                })
                common["sources"].update({
                    "mass": "not_applicable_static_collider",
                    "authored_mass": authored_mass_source,
                    "inertia": "not_applicable_static_collider",
                })
                result[obj_name] = common
                continue

            if articulation_prim is not None:
                runtime_articulation = _read_runtime_articulation(articulation_prim, obj_name)
                common.update({
                    "physical_role": "articulation",
                    "mass_kg": runtime_articulation.get("mass_kg") if runtime_articulation else authored_mass_kg,
                    "inertia_kg_m2": None,
                    "inertia_matrix_kg_m2": None,
                    "inertia_method": "not_applicable_articulation_use_bodies",
                    "body_count": runtime_articulation.get("body_count") if runtime_articulation else None,
                    "bodies": runtime_articulation.get("bodies") if runtime_articulation else None,
                })
                common["sources"].update({
                    "mass": "physx_runtime_articulation" if runtime_articulation else authored_mass_source,
                    "inertia": "physx_runtime_articulation_bodies" if runtime_articulation else "unavailable",
                })
                result[obj_name] = common
                continue

            runtime_rigid = (
                _read_runtime_rigid_body(obj, str(prim.GetPath()), obj_name)
                if obj is not None
                else None
            )
            if runtime_rigid is not None:
                common.update({
                    "physical_role": "dynamic_rigid_body",
                    **runtime_rigid,
                    "inertia_method": "physx_runtime_tensor",
                })
                common["sources"].update({
                    "mass": "physx_runtime_tensor",
                    "inertia": "physx_runtime_tensor",
                    "center_of_mass": "physx_runtime_tensor",
                })
            else:
                authored_inertia, authored_inertia_source = _read_authored_inertia(prim)
                common.update({
                    "physical_role": "dynamic_rigid_body",
                    "mass_kg": authored_mass_kg,
                    "inertia_kg_m2": authored_inertia,
                    "inertia_matrix_kg_m2": (
                        np.diag(authored_inertia).tolist() if authored_inertia is not None else None
                    ),
                    "center_of_mass_m": None,
                    "principal_axes_quat_wxyz": None,
                    "inertia_method": "usd_authored" if authored_inertia is not None else "unavailable",
                })
                common["sources"].update({
                    "mass": authored_mass_source,
                    "inertia": authored_inertia_source,
                    "center_of_mass": "unavailable",
                })

            result[obj_name] = common

        return result

    def _read_gripper_max_widths(self) -> dict:
        """Read the configured physical opening width for each robot gripper."""
        result = {}
        for robot in getattr(self.task, "robots", {}).values():
            cfg = getattr(robot, "cfg", {})
            value = cfg.get("gripper_max_width") if hasattr(cfg, "get") else None
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0.0:
                if getattr(robot, "left_gripper_indices", None):
                    result["left"] = value
                if getattr(robot, "right_gripper_indices", None):
                    result["right"] = value
        return result

    def _read_scene_mesh_info(self) -> dict:
        """S-ENV-001: Read scene geometry info from USD stage.

        Returns dict with fixture names, bounding boxes, and prim paths
        for table, walls, floor, and other static scene elements.
        """
        from pxr import UsdGeom

        result = {}
        stage = self.world.stage
        if stage is None:
            return result

        # Known scene fixture patterns
        fixture_keywords = ["table", "wall", "floor", "shelf", "ground", "arena"]

        def _valid_bbox(size) -> bool:
            try:
                values = [float(size[0]), float(size[1]), float(size[2])]
            except Exception:
                return False
            return all(math.isfinite(v) and 0.0 < abs(v) < 1e6 for v in values)

        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            prim_name = prim.GetName().lower()
            prim_type = prim.GetTypeName()
            if prim_type == "PhysicsScene":
                continue

            # Skip task objects, robots, cameras
            if "/task_0/" in prim_path and any(kw in prim_name for kw in fixture_keywords):
                try:
                    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
                    bbox = bbox_cache.ComputeWorldBound(prim)
                    rng = bbox.ComputeAlignedRange()
                    if rng.GetSize() and _valid_bbox(rng.GetSize()):
                        size = rng.GetSize()
                        min_pt = rng.GetMin()
                        max_pt = rng.GetMax()
                        result[prim.GetName()] = {
                            "prim_path": prim_path,
                            "bounding_box_m": [float(size[0]), float(size[1]), float(size[2])],
                            "min_m": [float(min_pt[0]), float(min_pt[1]), float(min_pt[2])],
                            "max_m": [float(max_pt[0]), float(max_pt[1]), float(max_pt[2])],
                            "type": prim.GetTypeName(),
                        }
                except Exception:
                    pass

            # Also capture arena/scene root prims
            if "arena" in prim_path.lower() or "scene" in prim_path.lower():
                if prim_path.count("/") <= 4:  # top-level only
                    try:
                        bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
                        bbox = bbox_cache.ComputeWorldBound(prim)
                        rng = bbox.ComputeAlignedRange()
                        if rng.GetSize() and _valid_bbox(rng.GetSize()):
                            size = rng.GetSize()
                            result[prim.GetName()] = {
                                "prim_path": prim_path,
                                "bounding_box_m": [float(size[0]), float(size[1]), float(size[2])],
                                "type": prim.GetTypeName(),
                            }
                    except Exception:
                        pass

        return result

    def _read_robot_torque_limits(self) -> dict:
        """Read arm-joint effort limits from the live PhysX articulation.

        Values are indexed exactly like ``joint_torque_gt``.  Only configured
        arm DOFs are exported; gripper and passive joints are intentionally not
        mixed into the robot-load contract fields.
        """
        result = {}
        for robot_name, robot in getattr(self.task, "robots", {}).items():
            articulation_view = getattr(robot, "_articulation_view", None)
            if articulation_view is None or not hasattr(articulation_view, "get_max_efforts"):
                continue

            values = articulation_view.get_max_efforts()
            if hasattr(values, "detach"):
                values = values.detach()
            if hasattr(values, "cpu"):
                values = values.cpu()
            if hasattr(values, "numpy"):
                values = values.numpy()
            values = np.asarray(values, dtype=np.float64)
            if values.ndim > 1:
                values = values[0]
            values = values.reshape(-1)

            dof_names = list(getattr(robot, "dof_names", []) or [])
            arm_indices = list(getattr(robot, "left_joint_indices", []) or [])
            arm_indices += list(getattr(robot, "right_joint_indices", []) or [])
            for index in arm_indices:
                if index < 0 or index >= len(values):
                    continue
                limit = float(values[index])
                if not math.isfinite(limit) or limit <= 0.0:
                    continue
                result[str(index)] = {
                    "limit_nm": limit,
                    "dof_index": int(index),
                    "dof_name": dof_names[index] if index < len(dof_names) else None,
                    "robot_name": robot_name,
                    "source": "PhysX ArticulationView.get_max_efforts",
                }
        return result

    def _read_placement_target_region(self) -> Optional[dict]:
        """Record the same world AABB that the place skill uses for planning."""
        from pxr import UsdGeom

        task_objects = getattr(self.task, "objects", None) or getattr(self.task, "_task_objects", {})
        place_object = task_objects.get("place_target") if task_objects else None
        if place_object is None:
            return None
        prim = getattr(place_object, "prim", None)
        if prim is None or not prim.IsValid():
            return None

        bound = UsdGeom.Imageable(prim).ComputeWorldBound(
            0.0, UsdGeom.Tokens.default_
        ).ComputeAlignedBox()
        min_pt, max_pt = bound.GetMin(), bound.GetMax()
        bounds_min = [float(min_pt[i]) for i in range(3)]
        bounds_max = [float(max_pt[i]) for i in range(3)]
        if not all(math.isfinite(v) for v in bounds_min + bounds_max):
            return None
        if not all(bounds_max[i] > bounds_min[i] for i in range(3)):
            return None

        return {
            "target_object_id": "place_target",
            "prim_path": str(prim.GetPath()),
            "min_m": bounds_min,
            "max_m": bounds_max,
            "metric": "world_axis_aligned_bbox_xy",
            "orientation_constraint": "unconstrained",
            "source": "UsdGeom.Imageable.ComputeWorldBound; identical geometry primitive used by place skill",
        }

    def _compute_final_placement_metrics(self, raw_gt: dict, region: dict) -> dict:
        """Compute traceable placement error and end-of-episode stability.

        Position error is the XY Euclidean distance to the configured target
        region (zero means inside).  Stability is based on ten consecutive
        recorded pose intervals, using exact quaternion angular displacement.
        """
        poses = raw_gt.get("object_state", {}).get("object_pose_gt") or {}
        target_ids = raw_gt.get("episode_meta", {}).get("target_object_ids") or []
        if not target_ids:
            target_id = raw_gt.get("episode_meta", {}).get("target_object_id")
            target_ids = [target_id] if target_id else []

        bounds_min = region.get("min_m") or []
        bounds_max = region.get("max_m") or []
        if len(bounds_min) < 2 or len(bounds_max) < 2:
            return {}

        dt_text = raw_gt.get("episode_meta", {}).get("physics_config", {}).get("rendering_dt", "1/30")
        try:
            if isinstance(dt_text, str) and "/" in dt_text:
                numerator, denominator = dt_text.split("/", 1)
                dt = float(numerator) / float(denominator)
            else:
                dt = float(dt_text)
        except (TypeError, ValueError, ZeroDivisionError):
            return {}
        if not math.isfinite(dt) or dt <= 0.0:
            return {}

        def _xy_distance_to_region(position):
            dx = max(float(bounds_min[0]) - float(position[0]), 0.0, float(position[0]) - float(bounds_max[0]))
            dy = max(float(bounds_min[1]) - float(position[1]), 0.0, float(position[1]) - float(bounds_max[1]))
            return math.hypot(dx, dy)

        def _norm3(values):
            return math.sqrt(sum(float(values[i]) ** 2 for i in range(3)))

        def _angular_speed(q0, q1):
            if q0 is None or q1 is None or len(q0) < 4 or len(q1) < 4:
                return None
            a = [float(v) for v in q0[:4]]
            b = [float(v) for v in q1[:4]]
            na = math.sqrt(sum(v * v for v in a))
            nb = math.sqrt(sum(v * v for v in b))
            if na <= 0.0 or nb <= 0.0:
                return None
            dot = abs(sum(a[i] * b[i] for i in range(4)) / (na * nb))
            angle = 2.0 * math.acos(min(1.0, max(-1.0, dot)))
            return angle / dt

        error_by_object = {}
        stability_by_object = {}
        window_intervals = 10
        linear_threshold_mps = 0.01
        angular_threshold_radps = 0.10
        for object_id in target_ids:
            series = poses.get(object_id) or {}
            translations = series.get("translation_per_step") or []
            orientations = series.get("orientation_per_step") or []
            if not translations or translations[-1] is None or len(translations[-1]) < 3:
                continue
            error_by_object[object_id] = _xy_distance_to_region(translations[-1])

            if len(translations) < window_intervals + 1 or len(orientations) < window_intervals + 1:
                continue
            start = len(translations) - window_intervals - 1
            linear_speeds = []
            angular_speeds = []
            valid = True
            for index in range(start + 1, len(translations)):
                p0, p1 = translations[index - 1], translations[index]
                if p0 is None or p1 is None or len(p0) < 3 or len(p1) < 3:
                    valid = False
                    break
                delta = [float(p1[i]) - float(p0[i]) for i in range(3)]
                linear_speeds.append(_norm3(delta) / dt)
                angular_speed = _angular_speed(orientations[index - 1], orientations[index])
                if angular_speed is None:
                    valid = False
                    break
                angular_speeds.append(angular_speed)
            if valid:
                max_linear = max(linear_speeds, default=0.0)
                max_angular = max(angular_speeds, default=0.0)
                stability_by_object[object_id] = {
                    "stable": max_linear <= linear_threshold_mps and max_angular <= angular_threshold_radps,
                    "max_linear_speed_mps": max_linear,
                    "max_angular_speed_radps": max_angular,
                }

        result = {
            "placement_error_by_object_gt": error_by_object,
            "placement_error_metric": "maximum XY Euclidean distance to configured target-region AABB",
            "placement_error_rot_gt": None,
            "final_stability_evidence": {
                "objects": stability_by_object,
                "window_intervals": window_intervals,
                "window_duration_s": window_intervals * dt,
                "linear_speed_threshold_mps": linear_threshold_mps,
                "angular_speed_threshold_radps": angular_threshold_radps,
                "source": "LMDB object translation/orientation time series",
            },
        }
        if len(error_by_object) == len(target_ids) and error_by_object:
            result["placement_error_pos_gt"] = max(error_by_object.values())
        if len(stability_by_object) == len(target_ids) and stability_by_object:
            result["stable_final_gt"] = all(item["stable"] for item in stability_by_object.values())
        return result

    def _compute_physical_grasp_drop_metrics(self, raw_gt: dict) -> dict:
        """Detect confirmed grasps and unintended drops from physical evidence.

        A grasp requires at least three consecutive target/robot contact frames
        while the gripper is below its configured open width.  A drop requires
        loss of that contact while the gripper remains closed, followed by at
        least 5 cm of downward motion before re-contact.  Immediate measured
        opening identifies commanded release; opening only after the object
        has already fallen does not truncate the physical drop height.
        """
        pairs = raw_gt.get("collision_gt", {}).get("collision_pair_gt")
        poses = raw_gt.get("object_state", {}).get("object_pose_gt") or {}
        gripper = raw_gt.get("gripper_gt", {})
        target_ids = raw_gt.get("episode_meta", {}).get("target_object_ids") or []
        max_widths = raw_gt.get("episode_meta", {}).get("physics_config", {}).get(
            "gripper_max_width_m_by_arm", {}
        )
        if not isinstance(pairs, list) or len(pairs) < 2 or not target_ids:
            return {"drop_event_gt": None, "drop_height_gt": None}

        timeline_length = len(pairs)

        def _sample(values, index):
            if not isinstance(values, list) or not values:
                return None
            mapped = min(
                int(round(index * (len(values) - 1) / max(timeline_length - 1, 1))),
                len(values) - 1,
            )
            return values[mapped]

        def _scalar(value):
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _contact(frame, target):
            entries = frame if isinstance(frame, list) else [frame]
            suffix = "/" + target.strip("/")
            for pair in entries:
                if not isinstance(pair, dict):
                    continue
                a, b = str(pair.get("bodyA", "")), str(pair.get("bodyB", ""))
                if (("robot/" in a.lower() and b.endswith(suffix))
                        or ("robot/" in b.lower() and a.endswith(suffix))):
                    return True
            return False

        evidence = {}
        any_drop = False
        drop_heights = []
        all_confirmed = True
        for target in target_ids:
            arm = "right" if "right" in target.lower() else "left"
            widths = gripper.get(f"gripper_width_{arm}")
            translations = (poses.get(target) or {}).get("translation_per_step")
            max_width = max_widths.get(arm)
            if not isinstance(widths, list) or not isinstance(translations, list):
                all_confirmed = False
                continue
            try:
                max_width = float(max_width)
            except (TypeError, ValueError):
                all_confirmed = False
                continue
            close_threshold = max_width - max(0.005, 0.05 * max_width)

            width_series = [_scalar(_sample(widths, i)) for i in range(timeline_length)]
            position_series = [_sample(translations, i) for i in range(timeline_length)]
            contacts = [_contact(frame, target) for frame in pairs]
            grasp_frames = [
                contacts[i] and width_series[i] is not None and width_series[i] < close_threshold
                for i in range(timeline_length)
            ]

            longest_run = run = 0
            for active in grasp_frames:
                run = run + 1 if active else 0
                longest_run = max(longest_run, run)
            confirmed = longest_run >= 3
            all_confirmed = all_confirmed and confirmed
            object_drop = False
            object_drop_height = 0.0

            if confirmed:
                for index in range(1, timeline_length - 3):
                    if not grasp_frames[index - 1] or contacts[index]:
                        continue
                    current_width = width_series[index]
                    if current_width is None or current_width >= close_threshold:
                        continue
                    future_widths = [w for w in width_series[index:index + 4] if w is not None]
                    opening = any(
                        future_widths[j] - future_widths[0] > 0.002
                        for j in range(1, len(future_widths))
                    )
                    if opening:
                        continue
                    start_position = position_series[index - 1]
                    if not isinstance(start_position, (list, tuple)) or len(start_position) < 3:
                        continue
                    z_start = float(start_position[2])
                    z_min = z_start
                    for probe in range(index, timeline_length):
                        if contacts[probe]:
                            break
                        position = position_series[probe]
                        if isinstance(position, (list, tuple)) and len(position) >= 3:
                            z_min = min(z_min, float(position[2]))
                    fall = max(0.0, z_start - z_min)
                    if fall >= 0.05:
                        object_drop = True
                        object_drop_height = max(object_drop_height, fall)

            any_drop = any_drop or object_drop
            if object_drop:
                drop_heights.append(object_drop_height)
            evidence[target] = {
                "grasp_confirmed": confirmed,
                "longest_closed_contact_run_frames": longest_run,
                "unexpected_drop_detected": object_drop,
                "drop_height_m": object_drop_height if object_drop else None,
                "closed_width_threshold_m": close_threshold,
                "source": "PhysX robot-object contact pairs + configured gripper width + LMDB object pose",
            }

        raw_gt.setdefault("gripper_gt", {})["grasp_evidence_by_object_gt"] = evidence
        result = {
            "physical_drop_evidence": evidence,
            "drop_height_gt": max(drop_heights) if drop_heights else None,
        }
        if any_drop:
            result["drop_event_gt"] = True
        elif all_confirmed and len(evidence) == len(target_ids):
            result["drop_event_gt"] = False
        else:
            result["drop_event_gt"] = None
        return result

    def _compute_support_polygon_margin(self) -> Optional[float]:
        """S-OUT-003: Compute support polygon margin for the pick object.

        Calculates the distance from the object's center of mass projection
        to the nearest edge of the support surface (table).
        Returns margin in meters, or None if cannot be computed.
        """
        import numpy as np
        from pxr import UsdGeom

        stage = self.world.stage
        if stage is None:
            return None

        # Get pick object positions (handles pick_object, pick_object_left/right).
        if not hasattr(self.task, 'objects'):
            return None

        pick_objects = []
        for name in ['pick_object', 'pick_object_left', 'pick_object_right']:
            if name in self.task.objects:
                pick_objects.append(self.task.objects[name])
        for name, obj in self.task.objects.items():
            if name.startswith('pick_') and obj not in pick_objects:
                pick_objects.append(obj)
        if not pick_objects:
            return None

        # Find table prim and get its bounding box
        table_prim = None
        for prim in stage.Traverse():
            prim_name = prim.GetName().lower()
            prim_path = str(prim.GetPath())
            if ('table' in prim_name or 'Group_table' in prim_name) and '/task_0/' in prim_path:
                table_prim = prim
                break

        if table_prim is None:
            return None

        try:
            bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
            bbox = bbox_cache.ComputeWorldBound(table_prim)
            rng = bbox.ComputeAlignedRange()
            min_pt = rng.GetMin()
            max_pt = rng.GetMax()

            # Table boundaries
            table_min_x, table_min_y = float(min_pt[0]), float(min_pt[1])
            table_max_x, table_max_y = float(max_pt[0]), float(max_pt[1])

            margins = []
            for obj in pick_objects:
                try:
                    obj_pos, _ = obj.get_world_pose()
                    obj_x, obj_y = float(obj_pos[0]), float(obj_pos[1])
                except Exception:
                    continue

                # Distance from object projection to all four support edges.
                margins.extend([
                    obj_x - table_min_x,
                    table_max_x - obj_x,
                    obj_y - table_min_y,
                    table_max_y - obj_y,
                ])

            if not margins:
                return None
            return min(margins)

        except Exception:
            return None

    def _compute_sensor_fields_from_seg(self, episode_dir, raw_gt: dict) -> dict:
        """S-SENSOR-001..006: Build LMDB-backed sensor GT metadata and per-instance segmentation stats.

        RGB/depth/segmentation images are large, so sim_raw_gt stores LMDB
        references instead of embedding image bytes. Segmentation-derived fields
        are computed per visible instance, not as whole-scene foreground stats.
        """
        import cv2
        import numpy as np

        result = {}

        lmdb_path = os.path.join(episode_dir, "lmdb")
        if not os.path.isdir(lmdb_path):
            print(f"[safety_risk] sensor: lmdb not found at {lmdb_path}")
            return result

        import lmdb
        env = lmdb.open(lmdb_path, readonly=True, lock=False)

        def _sensor_kind(prefix: str) -> Optional[str]:
            lower = prefix.lower()
            if "rgb" in lower:
                return "rgb"
            if "depth" in lower:
                return "depth"
            if "seg" in lower:
                return "seg"
            return None

        def _camera_name(prefix: str, kind: str) -> str:
            marker = f".{kind}."
            if marker in prefix:
                return prefix.split(marker, 1)[1]
            parts = prefix.split(".")
            return parts[-1] if parts else "default"

        sensor_prefixes = {"rgb": {}, "depth": {}, "seg": {}}
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                k = key.decode("utf-8") if isinstance(key, bytes) else key
                if "/" not in k:
                    continue
                prefix = k.split("/")[0]
                kind = _sensor_kind(prefix)
                if kind is None:
                    continue
                camera = _camera_name(prefix, kind)
                sensor_prefixes[kind][camera] = prefix

        def _count_frames(txn, prefix: str) -> int:
            count = 0
            while txn.get(f"{prefix}/{str(count).zfill(4)}".encode("utf-8")) is not None:
                count += 1
            return count

        with env.begin() as txn:
            if sensor_prefixes["rgb"]:
                result["virtual_rgb"] = {
                    "storage": "lmdb",
                    "lmdb_path": os.path.join(episode_dir, "lmdb"),
                    "cameras": {
                        camera: {"key_prefix": prefix, "num_frames": _count_frames(txn, prefix)}
                        for camera, prefix in sorted(sensor_prefixes["rgb"].items())
                    },
                }
            if sensor_prefixes["depth"]:
                result["virtual_depth"] = {
                    "storage": "lmdb",
                    "lmdb_path": os.path.join(episode_dir, "lmdb"),
                    "cameras": {
                        camera: {"key_prefix": prefix, "num_frames": _count_frames(txn, prefix)}
                        for camera, prefix in sorted(sensor_prefixes["depth"].items())
                    },
                }
            if sensor_prefixes["seg"]:
                result["segmentation_mask_gt"] = {
                    "storage": "lmdb",
                    "lmdb_path": os.path.join(episode_dir, "lmdb"),
                    "cameras": {
                        camera: {"key_prefix": prefix, "num_frames": _count_frames(txn, prefix)}
                        for camera, prefix in sorted(sensor_prefixes["seg"].items())
                    },
                }

        seg_prefix = next(iter(sensor_prefixes["seg"].values()), None)
        if seg_prefix is None:
            print(f"[safety_risk] sensor: no seg keys found in LMDB")
            env.close()
            return result

        print(f"[safety_risk] sensor: found seg prefix '{seg_prefix}'")

        def _decode_seg(raw_value):
            try:
                data = pickle.loads(raw_value) if isinstance(raw_value, bytes) else raw_value
                if isinstance(data, np.ndarray):
                    if data.ndim >= 2:
                        return data
                    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                if isinstance(data, (bytes, memoryview)):
                    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            except Exception:
                return None
            return None

        # Read all seg frames from LMDB.
        instance_ids = []
        bboxes = []
        vis_ratios = []

        with env.begin() as txn:
            frame_idx = 0
            while True:
                key = f"{seg_prefix}/{str(frame_idx).zfill(4)}"
                raw = txn.get(key.encode("utf-8"))
                if raw is None:
                    break

                seg_img = _decode_seg(raw)

                if seg_img is None or seg_img.size == 0:
                    instance_ids.append({"frame": frame_idx, "visible_instance_ids": None})
                    bboxes.append({"frame": frame_idx, "instances": None})
                    vis_ratios.append({"frame": frame_idx, "instances": None})
                    frame_idx += 1
                    continue

                if seg_img.ndim == 3:
                    seg_img = seg_img[:, :, 0]

                total_px = int(seg_img.shape[0] * seg_img.shape[1])
                unique_ids = sorted(int(x) for x in np.unique(seg_img) if int(x) != 0)
                instance_ids.append({
                    "frame": frame_idx,
                    "visible_instance_ids": unique_ids,
                    "background_id": 0,
                })

                frame_bboxes = {}
                frame_ratios = {}
                for inst_id in unique_ids:
                    coords = np.where(seg_img == inst_id)
                    if len(coords[0]) == 0:
                        continue
                    y_min, y_max = int(coords[0].min()), int(coords[0].max())
                    x_min, x_max = int(coords[1].min()), int(coords[1].max())
                    key_id = str(inst_id)
                    frame_bboxes[key_id] = [x_min, y_min, x_max, y_max]
                    frame_ratios[key_id] = round(len(coords[0]) / total_px, 6)

                bboxes.append({"frame": frame_idx, "instances": frame_bboxes})
                vis_ratios.append({"frame": frame_idx, "instances": frame_ratios})

                frame_idx += 1

        env.close()

        result["instance_id_map_gt"] = instance_ids
        result["object_bbox_gt"] = bboxes
        result["visibility_ratio_gt"] = vis_ratios

        return result

    def _fix_obstacle_physics_hierarchy(self):
        """Fix MANO hand model physics hierarchy.

        Converts the parent mano prim from RigidBody to Articulation Root,
        which allows child rigid bodies to work correctly in PhysX.
        """
        try:
            from pxr import Usd, UsdPhysics
            from omni.isaac.core.utils.prims import get_prim_at_path

            stage = self.world.stage
            if stage is None or not hasattr(self.task, 'objects'):
                return

            for obj_name, obj in self.task.objects.items():
                if "obstacle" not in obj_name.lower():
                    continue
                try:
                    prim_path = getattr(obj, 'prim_path', None) or getattr(obj, 'base_prim_path', None)
                    if not prim_path:
                        continue

                    prim = get_prim_at_path(prim_path)
                    if not prim or not prim.IsValid():
                        continue

                    # Step 1: Remove RigidBodyAPI from parent
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

                    # Step 2: Remove MassAPI from parent
                    if prim.HasAPI(UsdPhysics.MassAPI):
                        prim.RemoveAPI(UsdPhysics.MassAPI)

                    # Step 3: Keep the parent as a pure Xform.  MANO already
                    # defines its articulation root inside the referenced asset.
                    # Adding another root here creates a nested articulation.

                    # Step 4: Fix collision approximation for all descendant meshes.
                    for child in Usd.PrimRange(prim):
                        if child == prim:
                            continue
                        try:
                            child_path = str(child.GetPrimPath())
                            if child.HasAPI(UsdPhysics.CollisionAPI):
                                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(child)
                                approx = mesh_api.GetApproximationAttr()
                                if not approx:
                                    approx = mesh_api.CreateApproximationAttr()
                                if approx.Get() in (None, "", "none"):
                                    approx.Set("convexHull")
                                    print(f"[safety_risk] Fixed collision approximation for {child_path}")
                        except Exception as e:
                            print(f"[safety_risk] Warning: Failed to fix collision prim {child_path}: {e}")

                except Exception as e:
                    print(f"[safety_risk] Warning: Failed to fix obstacle physics: {e}")

        except Exception as e:
            print(f"[safety_risk] Warning: Failed to fix obstacle physics hierarchy: {e}")

    def _apply_semantic_labels(self):
        """Apply semantic labels to scene objects for segmentation.

        Uses Isaac Sim's add_update_semantics API to label objects so that
        the camera's semantic segmentation annotator can distinguish them.
        """
        try:
            from omni.isaac.core.utils.semantics import add_update_semantics
            from omni.isaac.core.utils.prims import get_prim_at_path

            stage = self.world.stage
            if stage is None:
                return

            labeled_count = 0

            # Label task objects
            if hasattr(self.task, 'objects'):
                for obj_name, obj in self.task.objects.items():
                    try:
                        prim_path = getattr(obj, 'prim_path', None) or getattr(obj, 'base_prim_path', None)
                        if prim_path:
                            prim = get_prim_at_path(prim_path)
                            if prim and prim.IsValid():
                                add_update_semantics(prim, semantic_label=obj_name, type_label="class")
                                labeled_count += 1
                    except Exception:
                        pass

            # Label robot
            if hasattr(self.task, 'robots'):
                for robot_name, robot in self.task.robots.items():
                    try:
                        prim = get_prim_at_path(robot.prim_path)
                        if prim and prim.IsValid():
                            add_update_semantics(prim, semantic_label=robot_name, type_label="class")
                            labeled_count += 1
                    except Exception:
                        pass

            # Label table
            for prim in stage.Traverse():
                prim_name = prim.GetName().lower()
                prim_path = str(prim.GetPath())
                if '/task_0/' in prim_path and prim_name in ('table', 'floor', 'wall', 'arena'):
                    try:
                        add_update_semantics(prim, semantic_label=prim_name, type_label="class")
                        labeled_count += 1
                    except Exception:
                        pass

            print(f"[safety_risk] Applied semantic labels to {labeled_count} prims")

        except Exception as e:
            print(f"[safety_risk] Warning: semantic label application failed: {e}")

    def _build_report_from_labels(
        self, episode_id: str, raw_gt: dict, features: dict, labels: dict
    ) -> dict:
        """Build safety report from the three JSON layers."""
        from safety_risk.sim_label_extractor import build_safety_report
        return build_safety_report(raw_gt, features, labels)

    def plan_with_render(self):
        end = False

        step_id = 0
        length = 0
        episode_success = True
        should_continue = True
        max_episode_length = self.task_cfg["data"]["max_episode_length"]
        episode_stats = {"succeed_times": 0, "current_times": 0}

        # ── PhysX data collector (for Sim_Raw_GT) ──
        _physx_collector = None
        if self._safety_eval_enabled:
            try:
                from safety_risk.physx_collector import PhysXDataCollector
                _physx_collector = PhysXDataCollector()
                # Apply safety_gate config from YAML
                sg_cfg = self._safety_eval_cfg.get("safety_gate", {})
                _physx_collector.configure_safety_gate(sg_cfg)
            except Exception as e:
                print(f"[safety_risk] Warning: PhysX collector init failed: {e}")

        # ── Fix obstacle physics hierarchy ──
        if self._safety_eval_enabled:
            self._fix_obstacle_physics_hierarchy()

        # ── Apply semantic labels for segmentation ──
        if self._safety_eval_enabled:
            self._apply_semantic_labels()

        should_continue = self.plan_first_skill(self.skills, should_continue)
        _safety_stop_active = False

        # Record obstacle starting position for round-trip movement
        if self._safety_eval_enabled:
            self._init_obstacle_origin()

        # Warmup
        for _ in range(10):
            obs = self.world.get_observations()
            # self._init_static_objects(self.task)
            self.world.step(render=True)

        # while True:
        #     obs = self.world.get_observations()
        #     # self._init_static_objects(self.task)
        #     self.world.step(render=True)

        while not (step_id >= max_episode_length or (not self.skills and not episode_success) or (not should_continue)):
            obs = self.world.get_observations()
            action_dict = {}

            # ── Safety gate: suppress actions if stop is active ──
            if _safety_stop_active:
                for ctrl in self.controllers.values():
                    if hasattr(ctrl, 'cmd_plan'):
                        ctrl.cmd_plan = None
            record_flag = True
            if self.skills and should_continue:
                # Process current skills
                current_skills = self.skills[0]
                for robot_name, skill_sequences in current_skills.items():
                    if skill_sequences and skill_sequences[0]:
                        action = [
                            skill[0].controller.forward(skill[0].manip_list[0])
                            for skill in skill_sequences[0]
                            if skill[0] and skill[0].is_ready()
                        ]

                        feasible_labels = [skill[0].is_feasible() for skill in skill_sequences[0] if skill[0]]
                        record_labels = [skill[0].is_record() for skill in skill_sequences[0] if skill[0]]

                        if False in feasible_labels:
                            should_continue = False
                        if False in record_labels:
                            record_flag = False

                        if action:
                            action_dict[robot_name] = {
                                "joint_positions": np.concatenate([a["joint_positions"] for a in action]),
                                "joint_indices": np.concatenate([a["joint_indices"] for a in action]),
                                "raw_action": action,
                            }

                            # Capture planned trajectories from controllers
                            if _physx_collector is not None:
                                for skill_seq in skill_sequences[0]:
                                    if skill_seq and skill_seq[0] and skill_seq[0].controller:
                                        ctrl = skill_seq[0].controller
                                        if hasattr(ctrl, 'cmd_plan') and ctrl.cmd_plan is not None:
                                            lr_name = getattr(ctrl, 'lr_name', robot_name)
                                            _physx_collector.capture_planned_trajectory(ctrl, arm_name=lr_name)

            elif not self.skills and episode_success:
                print("Task is successful")
                end = True
                # Continue real physics after task completion so placement
                # stability is measured after a defensible settling window,
                # rather than from only six frames (~0.2 s).  Contact data,
                # observations, and replay frames all remain aligned.
                settle_steps = int(
                    self._safety_eval_cfg.get("final_settle_steps", 60)
                    if self._safety_eval_enabled else 6
                )
                settle_steps = max(6, settle_steps)
                for j_idx in range(settle_steps):
                    self.world.step(render=True)
                    settle_step_id = step_id + j_idx
                    if _physx_collector is not None:
                        _physx_collector.collect_step(self.task, settle_step_id)
                    obs = self.world.get_observations()
                    log_dual_obs(
                        self.logger,
                        obs,
                        action_dict,
                        self.controllers,
                        step_idx=settle_step_id,
                    )
                    self._record_rgb_depth(settle_step_id)
                    self.world_recorder.record()
                length = step_id + settle_steps
                episode_stats["succeed_times"] += 1
                should_continue = False
                break

            if record_flag:
                log_dual_obs(self.logger, obs, action_dict, self.controllers, step_idx=step_id)
                self._record_rgb_depth(step_id)
            self.task.apply_action(action_dict)
            self.world.step(render=True)

            # ── Move hand obstacle toward robot ──
            self._move_hand_obstacle(step_id)

            # ── Collect PhysX runtime data ──
            if _physx_collector is not None:
                try:
                    _physx_collector.collect_step(self.task, step_id)
                except Exception:
                    pass  # Never block the simulation loop

                # ── Safety gate: check proximity and manage stop ──
                try:
                    if step_id == 0:
                        print(f"[safety_gate] calling check_safety_gate at step {step_id}")
                        print(f"[safety_gate] task.objects keys: {list(self.task.objects.keys()) if hasattr(self.task, 'objects') else 'N/A'}")
                        print(f"[safety_gate] task.robots keys: {list(self.task.robots.keys()) if hasattr(self.task, 'robots') else 'N/A'}")
                    _safety_stop_active = _physx_collector.check_safety_gate(
                        self.task, step_id
                    )
                    if _safety_stop_active and step_id > 0:
                        print(f"[safety_gate] STOP ACTIVE at step {step_id}")
                except Exception as _sg_err:
                    print(f"[safety_gate] Error at step {step_id}: {_sg_err}")
                    import traceback; traceback.print_exc()
                    _safety_stop_active = False

            step_id += 1
            if self.skills:
                episode_success, should_continue = self.update_skill_states(
                    self.skills, episode_success, should_continue
                )

        # ── Save PhysX data ──
        self._physx_collector = _physx_collector

        self.length = length
        if end:
            return length
        else:
            return 0

    def _dump_task_cfg(self, task_cfg):
        task_cfg_copy = deepcopy(task_cfg)
        return pickle.dumps(task_cfg_copy)

    def dump_plan_info(self) -> bytes:
        logger_ser = self.logger.dump()
        cfg_ser = self._dump_task_cfg(self.task_cfg)
        ser = pickle.dumps((cfg_ser, self.world_recorder.dumps(), logger_ser))
        return ser

    def dedump_plan_info(self, ser_obj: bytes) -> object:
        res = pickle.loads(ser_obj)
        return res

    def randomization_from_mem(self, data) -> bool:
        try:
            cfg_ser, _, _ = data
            task_cfg = pickle.loads(cfg_ser)
            self.task_cfg = task_cfg
            self.task.cfg = task_cfg

            # Individual Reset
            self.task.individual_reset_from_mem()
            self._randomization_layout_mem()
            return True
        except Exception as e:
            raise e

    def recover_seq_from_mem(self, data) -> list:
        """
        Recover sequence from memory based on WorldRecorder mode.

        Returns:
            - step_replay=False: Returns prim_poses list
            - step_replay=True: Returns placeholder list (replay data is in WorldRecorder)
        """
        try:
            _, wr_ser, logger_ser = data
            self.logger.dedump(logger_ser)

            if wr_ser:
                self.world_recorder.loads(wr_ser)

            if self.step_replay:
                return [None] * self.world_recorder.num_steps
            else:
                return self.world_recorder.prim_poses

        except Exception as e:
            raise e
