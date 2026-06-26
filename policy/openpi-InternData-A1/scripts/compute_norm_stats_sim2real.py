"""Compute normalization statistics for real-world tasks.

This script is used to compute the normalization statistics for a given real-world task. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config directory.
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
from openpi.training.config import MultiSim2RealSplitAlohaDataConfig, MultiDataConfig, DataConfig, TrainConfig

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


def main(dataset_path, robot_name, task_name, save_path):
    if robot_name == "lift2":
        config = TrainConfig(
            name="lift2",
            model=pi0_config.Pi0Config(),
            data=[
                MultiSim2RealSplitAlohaDataConfig(
                    repo_dir=dataset_path,
                    task_id=None,
                    use_gripper_aug=False,
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
            # pretrain model path
            weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/jax/pi0_base/params"),
            pytorch_weight_path="checkpoints/pytorch/pi0_base", 
            num_train_steps=30_000,
            num_workers=4,
            fsdp_devices=4,
            batch_size=8, 
        )
    
    data_config = config.data[0].create(config.model)
    print("done")
    output_path = os.path.join(save_path, robot_name, task_name)
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
    parser.add_argument("--task_path", type=str, default="data/InternData-A1/sim/long_horizon_tasks/lift2/sort_the_rubbish/*")
    parser.add_argument("--robot_name", type=str, default="lift2")
    parser.add_argument("--save_path", type=str, default="stats/sim2real")

    args, unknown = parser.parse_known_args()
    dataset_path=args.task_path
    save_path = args.save_path
    parts = dataset_path.split("/")
    robot_idx = next((i for i, p in enumerate(parts) if p == args.robot_name), None)
    if robot_idx is None:
        raise ValueError(
            f"Cannot find robot name in path. Expected {args.robot_name}, "
            f"but got path: {dataset_path}"
        )

    if robot_idx + 1 >= len(parts):
        raise ValueError(
            f"Path ends at robot name '{parts[robot_idx]}', cannot determine task_name: {local_path}"
        )
    robot_name = parts[robot_idx]
    task_name = parts[robot_idx + 1]
    try:
        main(dataset_path, robot_name, task_name, save_path)
    except:
        print(dataset_path)

