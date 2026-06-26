"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.mixture_dataset as _mixture_dataset
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
from pdb import set_trace
import openpi.training.weight_loaders as weight_loaders
import openpi.models.pi0_config as pi0_config
# from openpi.training.config import MultiSimGenieDataConfig, MultiSimSplitAlohaDataConfig, MultiSimFrankaDataConfig, MultiLeRobotReala2dDataConfig, MultiLeRobotRealArxLift2DataConfig, MultiDataConfig, DataConfig, TrainConfig
import logging
from pdb import set_trace
from typing import List
class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: List[_config.DataConfig],
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    # if data_config.repo_id is None:
    #     raise ValueError("Data config must have a repo_id")
    # dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _mixture_dataset.create_mixture_dataset_no_transform(data_config, action_horizon, model_config)
    # from pdb import set_trace; set_trace()
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config[0][0].repack_transforms.inputs,
            *data_config[0][0].data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
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

def compute_norm_stats(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_configs_list = []
    for data_config_factory in config.data:
        data_configs = data_config_factory.create(config.model)
        logging.info(f"data_config: {data_configs}")
        data_configs_list.append(data_configs)
    print("done")
    data_loader, num_batches = create_torch_dataloader(
        data_configs_list, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames=None
    )


    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    # stats = {key: normalize.OptimizedRunningStats() for key in keys}  # 新的
    # set_trace()
    step_id = 0
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        step_id += 1
        for key in keys:
            stats[key].update(np.asarray(batch[key]))
        if step_id > 10000:
            break

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    print(norm_stats)
    return norm_stats

