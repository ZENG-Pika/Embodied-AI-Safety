"""Compute normalization statistics for interndata-a1 sim tasks.

This script is used to compute the normalization statistics for interndata-a1 sim tasks. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""
import os
import glob
import numpy as np
import tqdm
import tyro
import json

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.mixture_dataset as _mixture_dataset
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

### training config ###
import openpi.training.weight_loaders as weight_loaders
import openpi.models.pi0_config as pi0_config
from openpi.training.config import MultiSimGenieDataConfig, MultiSimSplitAlohaDataConfig, MultiSimFrankaDataConfig, MultiDataConfig, DataConfig, TrainConfig

from pdb import set_trace

class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _mixture_dataset.create_mixture_dataset_calculate_norm_stats(data_config, action_horizon, model_config)
    dataset = _mixture_dataset.TransformedDataset(
        dataset,
        [
            *data_config[0].repack_transforms.inputs,
            *data_config[0].data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(dataset_path, task_category, robot_name, task_name, collect_name, save_path):
    if robot_name == "lift2" or robot_name == "split_aloha":
        config = TrainConfig(
            name="lift2",
            model=pi0_config.Pi0Config(),
            data=[
                MultiSimSplitAlohaDataConfig(
                    repo_dir=dataset_path,
                    task_id=None,
                    use_gripper_aug=True,
                    gripper_aug_config={
                        "gripper_action_keys": ["master_actions.left_gripper.openness", "master_actions.right_gripper.openness"],
                        "gripper_dim": -1,
                        "gripper_threshold_method": "std_multiplier",
                        "gripper_threshold_multiplier": 1.0,
                        "gripper_min_threshold": 0.001,
                        "gripper_max_threshold": 1.0,
                    },
                    stats_dir='',
                    base_config=MultiDataConfig(
                        prompt_from_task=True,
                    ),
                    asset_id=robot_name,
                    robot_name=robot_name,
                    repack_transforms=transforms.Group(
                        inputs=[
                            transforms.RepackTransform(
                                {
                                    "state_dict": {
                                        "left_joint": "states.left_joint.position", 
                                        "right_joint": "states.right_joint.position", 
                                        "left_gripper": "states.left_gripper.position", 
                                        "right_gripper": "states.right_gripper.position"
                                    },
                                    "action_dict": {
                                        "left_joint": "actions.left_joint.position", 
                                        "right_joint": "actions.right_joint.position", 
                                        "left_gripper": "actions.left_gripper.position", 
                                        "right_gripper": "actions.right_gripper.position",
                                        "left_gripper_openness": "master_actions.left_gripper.openness",
                                        "right_gripper_openness": "master_actions.right_gripper.openness"
                                    },
                                    "prompt": "task"
                                }
                            )
                        ]
                    )
                ),
            ],
            weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/jax/pi0_base/params"),
            pytorch_weight_path="checkpoints/pytorch/pi0_base", 
            num_train_steps=30_000,
            num_workers=4,
            fsdp_devices=4,
            batch_size=8, 
        )
    elif robot_name == "genie1":
        config = TrainConfig(
            name="genie1",
            model=pi0_config.Pi0Config(),
            data=[
                MultiSimGenieDataConfig(
                    repo_dir=dataset_path,
                    task_id=None,
                    use_gripper_aug=True,
                    gripper_aug_config={
                        "gripper_action_keys": ["master_actions.left_gripper.openness", "master_actions.right_gripper.openness"],
                        "gripper_dim": -1,
                        "gripper_threshold_method": "std_multiplier",
                        "gripper_threshold_multiplier": 1.0,
                        "gripper_min_threshold": 0.001,
                        "gripper_max_threshold": 1.0,
                    },
                    stats_dir='',
                    base_config=MultiDataConfig(
                        prompt_from_task=True,
                    ),
                    asset_id=robot_name,
                    robot_name=robot_name,
                    repack_transforms=transforms.Group(
                        inputs=[
                            transforms.RepackTransform(
                                {
                                    "state_dict": {
                                        "left_joint": "states.left_joint.position", 
                                        "right_joint": "states.right_joint.position", 
                                        "left_gripper": "states.left_gripper.position", 
                                        "right_gripper": "states.right_gripper.position"
                                    },
                                    "action_dict": {
                                        "left_joint": "actions.left_joint.position", 
                                        "right_joint": "actions.right_joint.position", 
                                        "left_gripper": "actions.left_gripper.position", 
                                        "right_gripper": "actions.right_gripper.position",
                                        "left_gripper_openness": "master_actions.left_gripper.openness",
                                        "right_gripper_openness": "master_actions.right_gripper.openness"
                                    },
                                    "prompt": "task"
                                }
                            )
                        ]
                    )
                ),
            ],
            weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/jax/pi0_base/params"),
            pytorch_weight_path="checkpoints/pytorch/pi0_base", 
            num_train_steps=30_000,
            num_workers=4,
            fsdp_devices=4,
            batch_size=8, 
        )
    elif "franka" in robot_name:
        config = TrainConfig(
            name="franka",
            model=pi0_config.Pi0Config(),
            data=[
                MultiSimFrankaDataConfig(
                    repo_dir=dataset_path,
                    task_id=None,
                    use_gripper_aug=True,
                    gripper_aug_config={
                        "gripper_action_keys": ["actions.gripper.openness"],
                        "gripper_dim": -1,
                        "gripper_threshold_method": "std_multiplier",
                        "gripper_threshold_multiplier": 1.0,
                        "gripper_min_threshold": 0.001,
                        "gripper_max_threshold": 1.0,
                    },
                    stats_dir='',
                    base_config=MultiDataConfig(
                        prompt_from_task=True,
                    ),
                    asset_id=robot_name,
                    robot_name=robot_name,
                    repack_transforms=transforms.Group(
                        inputs=[
                            transforms.RepackTransform(
                                {
                                    "state_dict": {
                                        "joint_position": "states.joint.position", 
                                        "gripper_pose": "states.gripper.pose", 
                                        "gripper_position": "states.gripper.position", 
                                    },
                                    "action_dict": {
                                        "gripper_pose": "actions.gripper.pose", 
                                        "gripper_position": "actions.gripper.position", 
                                        "gripper_openness": "actions.gripper.openness", 
                                    },
                                    "prompt": "task"
                                }
                            )
                        ]
                    )
                ),
            ],
            weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/jax/pi0_base/params"),
            pytorch_weight_path="checkpoints/pytorch/pi0_base", 
            num_train_steps=30_000,
            num_workers=4,
            fsdp_devices=4,
            batch_size=8, 
        )

    data_config = config.data[0].create(config.model)
    print("done")
    output_path = os.path.join(save_path, task_category, robot_name, task_name, collect_name)
    stats_json_path = os.path.join(output_path, "norm_stats.json")
    if os.path.isfile(stats_json_path):
        with open(stats_json_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True

    data_loader, num_batches = create_torch_dataloader(
        data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames=None
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    step_id = 0
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        step_id += 1
        for key in keys:
            stats[key].update(np.asarray(batch[key]))
        if step_id > 10000:
            break

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

def check_lerobot_repo(repo_dir: str):
    if os.path.isdir(os.path.join(repo_dir, "data")) and os.path.isdir(os.path.join(repo_dir, "meta")) and os.path.isdir(os.path.join(repo_dir, "videos")):
        print(repo_dir, "true")
        return True
    else:
        print(repo_dir, "false")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir", type=str, default="data/InternData-A1/sim")
    parser.add_argument("--task_category", type=str, default="pick_and_place_tasks")
    parser.add_argument("--save_path", type=str, default="stats/sim")
    parser.add_argument("--start_ratio", type=float, default=0.0)
    parser.add_argument("--end_ratio", type=float, default=1)
    args, unknown = parser.parse_known_args()
    root_data_dir = os.path.join(args.root_data_dir, args.task_category)

    dataset_paths = glob.glob(os.path.join(root_data_dir, "*", "*"))
    dataset_paths.sort()
    valid_paths = [
                    p for p in dataset_paths
                    if check_lerobot_repo(p)
                ]
    
    start_idx = int(len(valid_paths) * args.start_ratio)
    end_idx = int(len(valid_paths) * args.end_ratio) + 1
    valid_paths = valid_paths[start_idx:end_idx]
    for dataset_path in tqdm.tqdm(valid_paths):
        task_category = dataset_path.split('/')[-3]
        robot_name = dataset_path.split('/')[-2]
        task_name = dataset_path.split('/')[-1]
        collect_name = ""
        try:
            main(dataset_path, task_category, robot_name, task_name, collect_name, args.save_path)
        except:
            print(dataset_path)

    dataset_paths_w_subtask = glob.glob(os.path.join(root_data_dir, "*", "*","*"))
    dataset_paths_w_subtask.sort()
    valid_paths_w_subtask = [
                    p for p in dataset_paths_w_subtask
                    if check_lerobot_repo(p)
                ]
    start_idx = int(len(valid_paths_w_subtask) * args.start_ratio)
    end_idx = int(len(valid_paths_w_subtask) * args.end_ratio) + 1
    valid_paths_w_subtask = valid_paths_w_subtask[start_idx:end_idx]
    for dataset_path in tqdm.tqdm(valid_paths_w_subtask):
        task_category = dataset_path.split('/')[-4]
        robot_name = dataset_path.split('/')[-3]
        task_name = dataset_path.split('/')[-2]
        collect_name = dataset_path.split('/')[-1]
        try:
            main(dataset_path, task_category, robot_name, task_name, collect_name, args.save_path)
        except:
            print(dataset_path)
