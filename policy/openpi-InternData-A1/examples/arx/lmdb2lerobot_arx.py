# source /fs-computility/efm/liyang/miniconda3/etc/profile.d/conda.sh
# conda activate act

import argparse
import json
import logging
import os
import gc
import shutil
from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import torchvision
import cv2
import h5py
import lmdb
import numpy as np
import pickle
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import logging
import pdb
import os
import imageio # imageio-ffmpeg
from lerobot.common.datasets.compute_stats import auto_downsample_height_width, get_feature_stats, sample_indices
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import check_timestamps_sync, get_episode_data_index, validate_episode_buffer
import time
# import ray
# from ray.runtime_env import RuntimeEnv

"""
    Store both camera image and robot state as a combined observation. 
    Args:
        observation: images(camera), states (robot state)
        actions: joint, gripper, ee_pose
"""
FEATURES = {
    "images.rgb.head": {
        "dtype": "video",
        "shape": (368, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "images.rgb.hand_left": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "images.rgb.hand_right": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    # "states.left_joint.position": {
    #     "dtype": "float32",
    #     "shape": (6,),
    #     "names": ["left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5",],
    # },
    # "states.left_gripper.position": {
    #     "dtype": "float32",
    #     "shape": (1,),
    #     "names": ["left_gripper_0",],
    # },
    # "states.right_joint.position": {
    #     "dtype": "float32",
    #     "shape": (6,),
    #     "names": ["right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5",],
    # },
    # "states.right_gripper.position": {
    #     "dtype": "float32",
    #     "shape": (1,),
    #     "names": ["right_gripper_0",],
    # },
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),
        "names": ["left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5", "left_gripper_0",
                  "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5","right_gripper_0"],
    },
    "action": {
        "dtype": "float32",
        "shape": (14,),
        "names": ["left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5", "left_gripper_0",
                  "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5","right_gripper_0"],
    },
    # "actions.left_joint.position": {
    #     "dtype": "float32",
    #     "shape": (6,),
    #     "names": ["left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5",],
    # },
    # "actions.left_gripper.position": {
    #     "dtype": "float32",
    #     "shape": (1,),
    #     "names": ["left_gripper_0",],
    # },
    # "actions.right_joint.position": {
    #     "dtype": "float32",
    #     "shape": (6,),
    #     "names": ["right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5",],
    # },
    # "actions.right_gripper.position": {
    #     "dtype": "float32",
    #     "shape": (1,),
    #     "names": ["right_gripper_0", ],
    # },

}


import numpy as np

def filter_forbidden_frames(state_dict, position_threshold=0.001, velocity_threshold=0.005):
    """
    过滤禁止的帧，基于位置和速度阈值
    
    参数:
    - state_dict: 形状为 (n, 14) 的状态数组
    - position_threshold: 位置变化的阈值
    - velocity_threshold: 速度变化的阈值
    
    返回:
    - valid_mask: 布尔数组，True表示有效帧
    """
    # 排除夹爪列（第6和第13列，索引从0开始）
    qpos_columns = [i for i in range(14)]
    qpos_data = state_dict[:, qpos_columns]
    
    n_frames = len(state_dict)
    valid_mask = np.ones(n_frames, dtype=bool)
    # import pdb
    # pdb.set_trace()
    # 计算帧间差异（速度）
    if n_frames > 1:

        diff_sum = np.sum(np.abs(np.diff(qpos_data, axis=0)), axis=1)
        # sorted_indices = np.argsort(diff_sum)[::-1]
        # sorted_abs_sums = diff_sum[sorted_indices]
        
        # velocities = np.diff(qpos_data, axis=0)
        # 检查速度是否超过阈值
        for i in range(n_frames - 1):
            if np.any(np.abs(diff_sum[i]) > position_threshold):
                valid_mask[i] = True  # 有运动，有效帧
            else:
                valid_mask[i] = False  # 静止，可能是禁止帧
    valid_mask[i] = True
    return valid_mask

def statistical_filter(state_dict, std_multiplier=2.0):
    """
    使用统计方法检测异常（禁止）帧
    """
    # 排除夹爪列
    qpos_columns = [i for i in range(14) if i not in [6, 13]]
    qpos_data = state_dict[:, qpos_columns]
    
    # 计算每列的均值和标准差
    means = np.mean(qpos_data, axis=0)
    stds = np.std(qpos_data, axis=0)
    
    # 创建有效掩码
    valid_mask = np.ones(len(state_dict), dtype=bool)
    
    for i in range(len(state_dict)):
        # 检查每个关节位置是否在合理范围内
        deviations = np.abs(qpos_data[i] - means)
        if np.any(deviations > std_multiplier * stds):
            valid_mask[i] = False  # 异常帧
    
    return valid_mask


class ARXDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        download_videos: bool = True,
        local_files_only: bool = False,
        video_backend: str | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            local_files_only=local_files_only,
            video_backend=video_backend,
        )

    def save_episode(self, episode_data: dict | None = None, videos: dict | None = None) -> None:
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])
        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = str(video_path)  # PosixPath -> str
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)
        ep_stats = compute_episode_stats(episode_buffer, self.features)
        self._save_episode_table(episode_buffer, episode_index)
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)
        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )
        if not episode_data:
            self.episode_buffer = self.create_episode_buffer()


    def add_frame(self, frame: dict) -> None:
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()
        features = {key: value for key, value in self.features.items() if key in self.hf_features}
        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key in frame:
            if key == "task":
                self.episode_buffer["task"].append(frame["task"])
                continue
            if key not in self.features:
                print("key ", key)
                raise ValueError(f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'.")
            # import pdb
            # pdb.set_trace()
            self.episode_buffer[key].append(frame[key])
        self.episode_buffer["size"] += 1

# def crop_resize_no_padding(image, target_size=(480, 640)):
#     """
#     Crop and scale to target size (no padding)
#     :param image: input image (NumPy array)
#     :param target_size: target size (height, width)
#     :return: processed image
#     """
#     h, w = image.shape[:2]
#     target_h, target_w = target_size
#     target_ratio = target_w / target_h  # Target aspect ratio (e.g. 640/480=1.333)

#     # the original image aspect ratio and cropping direction
#     if w / h > target_ratio:  # Original image is wider → crop width
#         crop_w = int(h * target_ratio)  # Calculate crop width based on target aspect ratio
#         crop_h = h
#         start_x = (w - crop_w) // 2  # Horizontal center starting point
#         start_y = 0
#     else:  # Original image is higher → crop height
#         crop_h = int(w / target_ratio)  # Calculate clipping height according to target aspect ratio
#         crop_w = w
#         start_x = 0
#         start_y = (h - crop_h) // 2  # Vertical center starting point

#     # Perform centered cropping (to prevent out-of-bounds)
#     start_x, start_y = max(0, start_x), max(0, start_y)
#     end_x, end_y = min(w, start_x + crop_w), min(h, start_y + crop_h)
#     cropped = image[start_y:end_y, start_x:end_x]

#     # Resize to target size (bilinear interpolation)
#     resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
#     return resized


def load_lmdb_data(episode_path: Path, sava_path: Path, fps_factor: int, target_fps: int) -> Optional[Dict]:
    def load_image(txn, key):
        raw = txn.get(key)
        data = pickle.loads(raw)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # Convert to RGB if necessary
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = crop_resize_no_padding(image, target_size=(480, 640))
        return  image
    try:
        env = lmdb.open(
            str(episode_path / "lmdb"),
            readonly=True,
            lock=False,
            max_readers=128,
            readahead=False
        )
        with env.begin(write=False) as txn:
            keys = [k for k, _ in txn.cursor()]
   
            image_keys = sorted([k for k in keys if b'head' in k])
            if not image_keys:
                return None
            
            all_qpos = pickle.loads(txn.get(b'/observations/qpos'))
            
            if np.isscalar(all_qpos):
                total_steps = len(image_keys)
                all_qpos = [all_qpos] * total_steps
            else:
                total_steps = len(all_qpos)
            all_qpos = np.stack(all_qpos)
            state_action_dict = {}
            state_action_dict["states.left_joint.position"] = all_qpos[:, :6]
            state_action_dict["states.left_gripper.position"] = all_qpos[:, 6][:, None] # np.expand_dims(all_qpos[:, 6], axis=1)
            state_action_dict["states.right_joint.position"] = all_qpos[:, 7:13]
            state_action_dict["states.right_gripper.position"] = all_qpos[:, 13][:, None] # np.expand_dims(all_qpos[:, 13], axis=1)
            # state_keys = list(state_action_dict.keys())
            # for k in state_keys:
            #     state_action_dict[k.replace("states", "actions")] = np.concatenate([state_action_dict[k][1:, :], state_action_dict[k][-1, :][None,:]], axis=0)
            
            
            # action_dict = {}
            # action_dict["actions.left_joint.position"] = np.concatenate([state_dict["states.left_joint.position"][1:, :], state_dict["states.left_joint.position"][-1, :][None,:]], axis=0)
            # action_dict["actions.left_gripper.position"] = state_dict["states.left_gripper.position"][1:, :]
            # action_dict["actions.right_joint.position"] = state_dict["states.right_joint.position"][1:, :]
            # action_dict["actions.right_gripper.position"] = state_dict["states.right_gripper.position"][1:, :]

            action_dict = {}
   
            action_dict["action"] = np.concatenate([all_qpos[1:,], all_qpos[-1,].reshape(-1, 14)], axis=0)
            state_dict = {}
            state_dict["observation.state"] = all_qpos
            mask1 = filter_forbidden_frames(state_dict["observation.state"])
            # state_dict["observation.state"] = state_dict["observation.state"][mask1]
            # action_dict["actions.left_gripper.position"] = state_dict["states.left_gripper.position"][1:, :]
            # action_dict["actions.right_arm.position"] = np.concatenate([state_action_dict["states.right_joint.position"][1:, :], state_action_dict["states.right_joint.position"][-1, :][None,:]], axis=0)
            # action_dict["actions.left_arm.position"] = state_dict["states.right_gripper.position"][1:, :]

            assert total_steps == len(image_keys), "qpos length mismatch"
            selected_steps = [step for step in range(total_steps) if step % fps_factor == 0 and mask1[step]]
            frames = []
            image_observations = {
                "images.rgb.head": [],
                "images.rgb.hand_left": [],
                "images.rgb.hand_right": []
            }

            start_time = time.time()
            
            for step_index, step in enumerate(selected_steps):
                step_str = f"{step:04d}"
                head_key = f"observation/head/color_image/{step_str}".encode()
                left_key = f"observation/left_wrist/color_image/{step_str}".encode()
                right_key = f"observation/right_wrist/color_image/{step_str}".encode()
                if not (head_key in keys and left_key in keys and right_key in keys):
                    continue
                # state = all_qpos[step]
                # if step_index < len(selected_steps) - 1:
                #     action = all_qpos[selected_steps[step_index + 1]]
                # else:
                #     action = state
                data_dict = {}
                # for key, value in state_action_dict.items():
                #     data_dict[key] = value[step]
                data_dict['action'] = action_dict["action"][step]
                data_dict["task"] = " ".join(episode_path.parent.parent.name.split("_"))
                data_dict['observation.state'] = state_dict["observation.state"][step]
                # frames.append({
                #     "observation.states.joint.position": state,
                #     "actions.joint.position": action,
                #     "task": task_name,
                # })
                frames.append(data_dict)
                image_observations["images.rgb.head"].append(load_image(txn, head_key))
                image_observations["images.rgb.hand_left"].append(load_image(txn, left_key))
                image_observations["images.rgb.hand_right"].append(load_image(txn, right_key))
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"load image_observations of {episode_path}")
        env.close()
        if not frames:
            return None
        os.makedirs(sava_path, exist_ok=True)
        os.makedirs(sava_path/episode_path.name, exist_ok=True)
        imageio.mimsave(sava_path/episode_path.name/'head.mp4', image_observations["images.rgb.head"], fps=target_fps)
        imageio.mimsave(sava_path/episode_path.name/'hand_left.mp4', image_observations["images.rgb.hand_left"], fps=target_fps)
        imageio.mimsave(sava_path/episode_path.name/'hand_right.mp4', image_observations["images.rgb.hand_right"], fps=target_fps)
        print(f"imageio.mimsave time taken of {episode_path}")

        return {
            "frames": frames,
            "videos": {
                "images.rgb.head": sava_path/episode_path.name/"head.mp4",
                "images.rgb.hand_left": sava_path/episode_path.name/"hand_left.mp4",
                "images.rgb.hand_right": sava_path/episode_path.name/"hand_right.mp4",
            },
        }

    except Exception as e:
        logging.error(f"Failed to load LMDB data: {e}")
        return None


def get_all_tasks(src_path: Path, output_path: Path) -> Tuple[Path, Path]:
    src_dirs = sorted(list(src_path.glob("*"))) # "set*-*_collector*_datatime" as the conversion unit
    
    save_dirs = [output_path/_dir.parent.name/_dir.name for _dir in src_dirs]
    tasks_tuples = zip(src_dirs, save_dirs)
    for task in tasks_tuples:
        yield task

def compute_episode_stats(episode_data: Dict[str, List[str] | np.ndarray], features: Dict) -> Dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)
        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }
    return ep_stats

def sample_images(input):
    if type(input) is str:
        video_path = input
        reader = torchvision.io.VideoReader(video_path, stream="video")
        frames = [frame["data"] for frame in reader]
        frames_array = torch.stack(frames).numpy()  # Shape: [T, C, H, W]
        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)
            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)
            images[i] = img
    elif type(input) is np.ndarray:
        frames_array = input[:, None, :, :]  # Shape: [T, C, H, W]
        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)
            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)
            images[i] = img
    return images


def load_local_dataset(episode_path: str, save_path:str, origin_fps=30, target_fps=30):
    fps_factor = origin_fps // target_fps
    # print(f"fps downsample factor: {fps_factor}")
    # logging.info(f"fps downsample factor: {fps_factor}")
    # for format_str in [f"{episode_id:07d}", f"{episode_id:06d}", str(episode_id)]:
    #     episode_path = Path(src_path) / format_str
    #     save_path = Path(save_path) / format_str
    #     if episode_path.exists():
    #         break
    # else:
    #     logging.warning(f"Episode directory not found for ID {episode_id}")
    #     return None, None
    episode_path = Path(episode_path)
    if not episode_path.exists():
        logging.warning(f"{episode_path} does not exist")
        return None, None
        
    if not (episode_path / "lmdb/data.mdb").exists():
        logging.warning(f"LMDB data not found for episode {episode_path}")
        return None, None
    
    raw_dataset = load_lmdb_data(episode_path, save_path, fps_factor, target_fps)
    if raw_dataset is None:
        return None, None
    frames = raw_dataset["frames"] # states, actions, task
    
    videos = raw_dataset["videos"] # image paths
    ## check the frames
    for camera_name, video_path in videos.items():
        if not os.path.exists(video_path):
            logging.error(f"Video file {video_path} does not exist.")
            print(f"Camera {camera_name} Video file {video_path} does not exist.")
            return None, None
    return frames, videos


def save_as_lerobot_dataset(task: tuple[Path, Path], repo_id, num_threads, debug, origin_fps=30,  target_fps=30, robot_type="piper", delete_downsampled_videos=True):
    src_path, save_path = task
    print(f"**Processing collected** {src_path}")
    print(f"**saving to** {save_path}")
    if save_path.exists():
        # print(f"Output directory {save_path} already exists. Deleting it.")
        # logging.warning(f"Output directory {save_path} already exists. Deleting it.")
        # shutil.rmtree(save_path)
        print(f"Output directory {save_path} already exists.")
        return 

    dataset = ARXDataset.create(
        repo_id=f"{repo_id}",
        root=save_path,
        fps=target_fps,
        robot_type=robot_type,
        features=FEATURES,
    )
    all_episode_paths = sorted([f.as_posix() for f in src_path.glob(f"*") if f.is_dir()])
    # all_subdir_eids = [int(Path(path).name) for path in all_subdir]
    if debug:
        for i in range(1):
            # pdb.set_trace()
            frames, videos = load_local_dataset(episode_path=all_episode_paths[i], save_path=save_path, origin_fps=origin_fps, target_fps=target_fps)
            for frame_data in frames:
                dataset.add_frame(frame_data)
            dataset.save_episode(videos=videos)
            if delete_downsampled_videos:
                for _, video_path in videos.items():
                    parent_dir = os.path.dirname(video_path)
                    try:
                        shutil.rmtree(parent_dir)
                        # os.remove(video_path)
                        # print(f"Successfully deleted: {parent_dir}")
                        print(f"Successfully deleted: {video_path}")
                    except Exception as e:
                        pass  # Handle the case where the directory might not exist or is already deleted
    else:
        for batch_index in range(len(all_episode_paths)//num_threads+1):
            batch_episode_paths = all_episode_paths[batch_index*num_threads:(batch_index+1)*num_threads]
            if len(batch_episode_paths) == 0:
                continue
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for episode_path in batch_episode_paths:
                    print("starting to process episode: ", episode_path)
                    futures.append(
                        executor.submit(load_local_dataset, episode_path=episode_path, save_path=save_path, origin_fps=origin_fps, target_fps=target_fps)
                    )
                for raw_dataset in as_completed(futures):
                    frames, videos = raw_dataset.result()
                    if frames is None or videos is None:
                        print(f"Skipping episode {episode_path} due to missing data.")
                        continue
                    for frame_data in frames:
                        dataset.add_frame(frame_data)
                    dataset.save_episode(videos=videos)
                    gc.collect()
                    print(f"finishing processed {videos}")
                    if delete_downsampled_videos:
                        for _, video_path in videos.items():
                            # Get the parent directory of the video
                            parent_dir = os.path.dirname(video_path)
                            try:
                                shutil.rmtree(parent_dir)
                                print(f"Successfully deleted: {parent_dir}")
                            except Exception as e:
                                pass

def main(src_path, save_path, repo_id, num_threads=60, debug=False, origin_fps=30, target_fps=30):
    logging.info("Scanning for episodes...")
    tasks = get_all_tasks(src_path, save_path)
    # import pdb
    # pdb.set_trace()
    if debug:
        task = next(tasks)
        save_as_lerobot_dataset(task, repo_id, num_threads=num_threads, debug=debug, origin_fps=origin_fps, target_fps=target_fps)
    else:
        for task in tasks:
            save_as_lerobot_dataset(task, repo_id, num_threads=num_threads, debug=debug, origin_fps=origin_fps, target_fps=target_fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert collected data from Piper to Lerobot format.")
    parser.add_argument(
        "--src_path",
        type=str,
        # required=False,
        default="/fs-computility/efm/shared/datasets/myData-A1/real/raw_data/agilex_split_aloha/",
        help="Path to the input file containing collected data in Piper format.",
        #help="/fs-computility/efm/shared/datasets/myData-A1/real/raw_data/agilex_split_aloha/Make_a_beef_sandwich",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        # required=False,
        default="/fs-computility/efm/shared/datasets/myData-A1/real/lerobot_v2_1/agilex_split_aloha/",
        help="Path to the output file where the converted Lerobot format will be saved.",
        #help="Path to the output file where the converted Lerobot format will be saved.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with limited episodes",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=50,
        help="Number of threads per process",
    )
    # parser.add_argument(
    #     "--task_name",
    #     type=str,
    #     required=True,
    #     default="Pick_up_the_marker_and_put_it_into_the_pen_holder",
    #     help="Name of the task to be processed. Default is 'Pick_up_the_marker_and_put_it_into_the_pen_holder'.",
    # )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        # default="SplitAloha_20250714",
        help="identifier for the dataset repository.",
    )
    parser.add_argument(
        "--origin_fps",
        type=int,
        default=30,
        help="Frames per second for the obervation video. Default is 30.",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=30,
        help="Frames per second for the downsample video. Default is 30.",
    )
    args = parser.parse_args()
    assert int(args.origin_fps) % int(args.target_fps) == 0, "origin_fps must be an integer multiple of target_fps"
    start_time = time.time()
    main(
        src_path=Path(args.src_path),
        save_path=Path(args.save_path),
        repo_id=args.repo_id,
        num_threads=args.num_threads,
        debug=args.debug,
        origin_fps=args.origin_fps,
        target_fps=args.target_fps
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")
# --target_fps 10
# --src_path /fs-computility/efm/shared/datasets/myData-A1/real/raw_data/agilex_split_aloha/Put_the_bananas_in_the_basket
# --save_path /mnt/shared-storage-user/internvla/Users/liyang/data/processed_data/arx_lift2