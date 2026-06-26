import numpy as np
from dataclasses import dataclass
from typing import SupportsIndex, Sequence, List, Dict, Any, Tuple, Optional, Union, TypeVar, Protocol

import torch

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)

import openpi.transforms as _transforms
from pdb import set_trace
import logging
T_co = TypeVar("T_co", covariant=True)
import openpi.training.config as _config
import openpi.shared.normalize as normalize

def detect_gripper_change_step(
    dataset,
    select_actions: list[str] = ["action"],
    gripper_dim: int = -1,
    threshold_method: str = "std_multiplier",
    threshold_multiplier: float = 2.0,
    min_threshold: float = 0.001,
    max_threshold: float = 1.0,
    plot_gripper_changes: bool = False,
):
    """
    Detect the step of gripper change. Only work for the self-collected dataset.
    Modifies the dataset in place by adding 'gripper_change_step_idx' attribute.
    This version uses a sliding window of size 4 centered around non_zero_idx,
    including the indices and removing duplicates.
    
    Args:
        dataset: LeRobotDataset instance
        select_actions: List of action keys to process
        gripper_dim: Dimension index for gripper in the action vector
        threshold_method: Method to calculate threshold ('std_multiplier', 'percentile', 'absolute')
        threshold_multiplier: Multiplier for std-based threshold
        min_threshold: Minimum threshold value to avoid too sensitive detection
        max_threshold: Maximum threshold value to avoid missing large changes
        plot_gripper_changes: Whether to plot gripper changes visualization
    """
    episode_lengths = [ep_dict["length"] for ep_dict in dataset.meta.episodes.values()]
    cumulative_lengths = np.cumsum(episode_lengths)

    all_window_indices = set()  # Use a set for automatic deduplication

    for action_key in select_actions:
        action_values = dataset.hf_dataset[action_key]

        delta_action = np.diff(action_values, axis=0)

        # Handle episode boundaries
        for end_idx in cumulative_lengths[:-1]:
            if end_idx - 1 < len(delta_action) and end_idx - 2 >= 0:
                delta_action[end_idx - 1] = delta_action[end_idx - 2]
            elif end_idx - 1 < len(delta_action): 
                delta_action[end_idx - 1] = 0 

        if delta_action.ndim == 1:
            delta_action = delta_action[:, np.newaxis]
        
        assert delta_action.ndim == 2

        # Extract gripper delta values
        gripper_delta = delta_action[:, gripper_dim]
        
        # Calculate threshold based on statistical properties
        if threshold_method == "std_multiplier":
            # Use standard deviation to filter out small tremors
            std_val = np.std(gripper_delta)
            threshold = threshold_multiplier * std_val
        elif threshold_method == "percentile":
            # Use percentile-based threshold (e.g., 90th percentile)
            threshold = np.percentile(np.abs(gripper_delta), 85)
        elif threshold_method == "absolute":
            # Use absolute threshold
            threshold = threshold_multiplier
        else:
            raise ValueError(f"Unknown threshold_method: {threshold_method}")

        # Clamp threshold to reasonable bounds
        threshold = np.clip(threshold, min_threshold, max_threshold)

        # Find indices where gripper change exceeds threshold
        significant_change_idx = np.where(np.abs(gripper_delta) > threshold)[0]
        
        cur_window_indices = set()
        for idx in significant_change_idx:
            # Create a sliding window of size 4 centered around idx.
            # The window should include [idx-2, idx-1, idx, idx+1].
            # This means starting 2 before and ending 1 after.
            window_start = idx - 2
            window_end = idx + 1

            # Generate indices for the current window and ensure they are non-negative
            # and within the bounds of the original action_values length.
            # The maximum index possible is len(action_values) - 1.
            # Since delta_action is len(action_values) - 1, the index refers to
            # the step *before* the change. So the max index we want is effectively
            # len(action_values) - 1, which corresponds to the last valid step index.
            # If the original index is `i`, delta_action[i] corresponds to the change
            # from step `i` to `i+1`. We want to include step `i` and its neighbors.
            # The maximum index for steps is `len(action_values) - 1`.
            # So, the window indices should not exceed `len(action_values) - 1`.
            max_possible_idx = len(action_values) - 1

            # Ensure indices are within valid range [0, max_possible_idx]
            current_window_indices = np.arange(
                max(0, window_start), min(max_possible_idx + 1, window_end + 1)
            )
            for w_idx in current_window_indices:
                cur_window_indices.add(w_idx)
                all_window_indices.add(w_idx)

        if plot_gripper_changes:
            num_episodes_to_plot = 5
            end_index_for_plot = cumulative_lengths[num_episodes_to_plot - 1] - 1
            delta_action_to_plot = delta_action[:end_index_for_plot]

            # Filter gripper_change_step_idx
            gripper_change_step_idx = np.array(sorted(list(cur_window_indices))).astype(np.int32)
            gripper_change_step_idx_to_plot = gripper_change_step_idx[gripper_change_step_idx < end_index_for_plot]

            plot_gripper_changes_in_subplots(
                delta_action_to_plot, 
                gripper_change_step_idx_to_plot, 
                episode_lengths, 
                num_episodes_to_plot, 
                gripper_dim, 
                f"{action_key}_gripper_change"
            )

    # Convert the set to a numpy array and sort it
    gripper_change_step_idx = np.array(sorted(list(all_window_indices))).astype(np.int32)

    print(f"Total unique gripper change steps: {len(gripper_change_step_idx)}, Total steps: {len(action_values)}")

    dataset.gripper_change_step_idx = gripper_change_step_idx
    # set_trace()

    return dataset

class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")

class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)

def transform_dataset(dataset: Dataset, data_config: _config.DataConfig) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )

class MixtureDataset(Dataset):
    """
    A composite dataset that combines multiple datasets, allowing for weighted sampling
    and specific handling based on training stage (e.g., pretrain, finetune) and
    gripper change detection for augmentation.

    This dataset flattens all eligible samples from its constituent datasets and assigns
    sampling weights based on configuration and heuristics (e.g., `gripper_aug_ratio`).
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        datasets_name: Sequence[str],
        datasets_meta: Sequence[LeRobotDatasetMetadata],
        datasets_weights: Dict[str, float] = None,
        gripper_aug_ratio: float = 1.0,
        shuffle: bool = True,
    ):
        """
        Initializes the MixtureDataset.

        Args:
            datasets (Sequence[Dataset]): A list of `Dataset` objects to be combined.
            datasets_name (Sequence[str]): A list of names corresponding to each dataset in `datasets`.
            datasets_meta (Sequence[LeRobotDatasetMetadata]): Metadata for each dataset,
                typically containing `num_episodes`, `num_frames`, `fps`, and `num_indices`.
            datasets_weights (Dict[str, float], optional): A dictionary mapping dataset names
                to their base sampling weights. If None, equal weights are assumed.
            is_eval (bool): If True, the dataset is configured for evaluation, potentially
                limiting the number of episodes and disabling shuffling for reproducibility.
            num_eval_episodes (int, optional): The number of episodes to select for evaluation.
                Only used if `is_eval` is True.
            stage (str): The current training stage (e.g., "stage1_pretrain_wm").
                This affects how indices are sampled from the underlying datasets.
            gripper_aug_ratio (float): A multiplier applied to the weights of samples
                that contain a detected gripper change. Useful for augmenting rare events.
            shuffle (bool): If True, the flat sample map and sampling weights are shuffled
                after initial creation. Ignored if `is_eval` is True.
        """
        self.datasets = datasets
        self.datasets_name = datasets_name
        self.meta = datasets_meta
        # Extract total number of episodes and frames for each dataset from metadata.
        self.num_episodes = [meta.info['total_episodes'] for meta in datasets_meta]
        self.num_frames = [meta.info['total_frames'] for meta in datasets_meta]


        # Compute the flattened list of (dataset_idx, sample_idx) pairs.
        # This involves sampling indices based on the stage and dataset type.
        self._compute_len(False)
        # Assign normalized sampling weights to each sample in the flattened map.
        self._get_weights(datasets_weights, gripper_aug_ratio)

        # For training, ensure the sample map and weights are consistent.
        if len(self.flat_sample_map) != len(self.sample_weights):
            raise ValueError(
                f"Mismatch in flat sample map length ({len(self.flat_sample_map)}) "
                f"and sample weights length ({len(self.sample_weights)})."
            )
        if shuffle:
            # Shuffle both the sample map and weights in the same order for training.
            # This ensures random access to samples while maintaining their assigned probabilities.
            indices = np.random.permutation(len(self.flat_sample_map))
            self.flat_sample_map = [self.flat_sample_map[i] for i in indices]
            self.sample_weights = self.sample_weights[indices]

    def __len__(self) -> int:
        """
        Returns the total number of samples in the mixture dataset (after flattening and selection).
        This length represents the effective size of the dataset for iteration.
        """
        return len(self.flat_sample_map)

    def __getitem__(self, index: SupportsIndex):
        """
        Retrieves a specific sample from one of the underlying datasets based on the
        flattened sample map.

        Args:
            index (SupportsIndex): The index in the flattened `flat_sample_map` (0 to `len(self) - 1`).

        Returns:
            Tuple[int, Any]: A tuple containing the original dataset index and the
                             sample data (dictionary) from that dataset.

        Raises:
            IndexError: If the provided index is out of bounds for the dataset.
        """
        if not (0 <= index < len(self.flat_sample_map)):
            raise IndexError(f"Index {index} is out of bounds for the dataset (size: {len(self.flat_sample_map)}).")

        # Retrieve the original dataset index and sample index from the flattened map.
        dataset_idx, sample_idx = self.flat_sample_map[index]
        return self.datasets[dataset_idx][sample_idx]

    def _compute_len(self, is_eval: bool = False):
        """
        Pre-computes and stores `all_sample_indices`, a list of episode indices sampled
        from each constituent dataset. This method prepares the data for `_create_flat_sample_map`.

        Args:
            is_eval (bool): Flag indicating if indices are being computed for an evaluation dataset.
        """
        self.all_sample_indices: List[Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor]] = []

        for i, (ds, meta) in enumerate(zip(self.datasets, self.meta)):
            # Access the underlying LeRobotDataset or MultiLeRobotDataset, bypassing TransformedDataset wrapper.
            actual_ds = ds._dataset if isinstance(ds, TransformedDataset) else ds

            # Determine the number of indices to sample for this dataset based on the current stage.
            # "stage1" typically uses a limited number of indices (`num_indices`), while other stages
            # might use all available data or a different strategy.
            num_indices = None

            if isinstance(actual_ds, MultiLeRobotDataset):
                # For MultiLeRobotDataset, iterate through its sub-datasets to get indices.
                indices_list_for_multi_ds = []
                for sub_ds in actual_ds._datasets:
                    _from = sub_ds.episode_data_index["from"]
                    _to = sub_ds.episode_data_index["to"]
                    indices = self._sample_indices(
                        _from, _to, num_indices, is_eval=is_eval, dataset_name=self.datasets_name[i]
                    )
                    indices_list_for_multi_ds.append(indices)
                self.all_sample_indices.append(indices_list_for_multi_ds)
            elif isinstance(actual_ds, LeRobotDataset):
                # For a single LeRobotDataset.
                _from = actual_ds.episode_data_index["from"]
                _to = actual_ds.episode_data_index["to"]
                indices = self._sample_indices(
                    _from, _to, num_indices, is_eval=is_eval, dataset_name=self.datasets_name[i]
                )
                self.all_sample_indices.append(indices)
            else:
                raise TypeError(f"Unsupported dataset type: {type(actual_ds)}. "
                                "Expected `LeRobotDataset` or `MultiLeRobotDataset`.")

        # After collecting all sampled episode indices, flatten them into `flat_sample_map`.
        self.flat_sample_map = self._create_flat_sample_map()

    def _create_flat_sample_map(self) -> List[Tuple[int, int]]:
        """
        Converts the potentially nested structure of `self.all_sample_indices` (which can be
        lists of lists of tensors, or lists of tensors) into a flat list of
        `(original_dataset_index, sample_index_within_original_dataset)` tuples.

        This flattened map is then used by `__getitem__` to efficiently retrieve samples.
        """
        flat_map = []
        for dataset_idx, sample_group in enumerate(self.all_sample_indices):
            # Case 1: `MultiLeRobotDataset` where `sample_group` is `List[List[torch.Tensor]]`
            if isinstance(sample_group, list) and len(sample_group) > 0 and isinstance(sample_group[0], list):
                for sub_group in sample_group: # Iterate through sub-datasets' index lists
                    for tensor_of_indices in sub_group: # Iterate through tensors of indices for episodes
                        for i in range(tensor_of_indices.numel()):
                            flat_map.append((dataset_idx, tensor_of_indices[i].item()))
            # Case 2: `LeRobotDataset` where `sample_group` is `List[torch.Tensor]`
            elif isinstance(sample_group, list) and len(sample_group) > 0 and isinstance(sample_group[0], torch.Tensor):
                for tensor_of_indices in sample_group:
                    for i in range(tensor_of_indices.numel()):
                        flat_map.append((dataset_idx, tensor_of_indices[i].item()))
            # Case 3: A rare case where `sample_group` might be a single `torch.Tensor` directly
            elif isinstance(sample_group, torch.Tensor):
                for i in range(sample_group.numel()):
                    flat_map.append((dataset_idx, sample_group[i].item()))
        return flat_map

    def _sample_indices(
        self,
        start: List[int],
        end: List[int],
        num_frames: Optional[int],
        random_pad: bool = False,
        is_eval: bool = False,
        dataset_name: str = None, # Added for potential future stage-specific logic
    ) -> List[torch.Tensor]:
        """
        Samples indices for episodes based on the current stage and dataset-specific rules.
        This function is called per episode to determine which frames to include.

        Args:
            start (List[int]): List of starting frame indices for each episode.
            end (List[int]): List of ending frame indices for each episode.
            num_frames (Optional[int]): The target number of frames to sample per episode.
                                        This is primarily used for "stage1" where sampling
                                        a fixed number of frames per episode might be desired.
            random_pad (bool): If True, and `frame_count < target_frames`, shorter episodes
                               will be padded with randomly selected indices from themselves.
            is_eval (bool): If True, adjusts indices for evaluation (e.g., shifting by 1 for stage1
                            to ensure predicted frames are not identical to observed frames).
            dataset_name (str): The name of the dataset (for debugging or future dataset-specific sampling rules).

        Returns:
            List[torch.Tensor]: A list of PyTorch tensors, where each tensor contains the
                                sampled frame indices for a single episode.
        """
        all_indices_for_episodes = []
        for _start, _end in zip(start, end):
            frame_count = _end - _start # Total frames available in this episode.
            target_frames = frame_count
            if frame_count >= target_frames:
                # If enough frames are available, linearly space the indices to sample `target_frames`.
                indices = torch.linspace(_start, _end - 1, steps=target_frames).long()
            else:
                # If fewer frames than `target_frames` are available.
                if random_pad:
                    # Pad the existing frames with randomly chosen duplicates from the episode.
                    pad_size = target_frames - frame_count
                    indices = torch.arange(_start, _end) # All available original indices
                    # Randomly sample `pad_size` indices from the existing ones.
                    pad_indices = indices[torch.randint(0, frame_count, (pad_size,))]
                    indices = torch.cat([indices, pad_indices]) # Combine original and padded indices
                    indices = indices[torch.randperm(target_frames)] # Randomly permute to mix original and padded.
                else:
                    # If not padding, simply use all available frames.
                    indices = torch.arange(_start, _end)

            all_indices_for_episodes.append(indices)

        return all_indices_for_episodes

    def _get_weights(self, datasets_weights: Dict[str, float], aug_ratio: float = 1.0):
        """
        Assigns normalized sampling weights to each individual sample in the flattened map.
        Weights are adjusted based on base dataset weights and `gripper_aug_ratio` for
        samples that have a detected gripper change.

        Args:
            datasets_weights (Dict[str, float]): A dictionary mapping dataset names to their
                                                 base sampling weights. If a dataset name is
                                                 not found, a default weight of 1.0 is used.
            aug_ratio (float): The augmentation ratio (multiplier) to apply to the base weight
                                for samples where a gripper change is detected.
        """
        self.sample_weights: List[float] = [] 
        self.datasets_weight_map: Dict[str, float] = {}

        if datasets_weights is None:
            num_datasets = len(self.datasets_name)
            datasets_weights = {name: 1.0 / num_datasets for name in self.datasets_name}

        for idx, ds_name in enumerate(self.datasets_name):
            # Access the underlying dataset to get gripper change information.
            # It might be wrapped in a TransformedDataset, so we unwrap it.
            current_base_dataset = self.datasets[idx]._dataset if isinstance(self.datasets[idx], TransformedDataset) else self.datasets[idx]
            base_weight = datasets_weights.get(ds_name, 1.0) # Get base weight for this dataset

            individual_weights_for_ds: List[float] = []

            # Logic to retrieve `gripper_change_step_idx` and assign weights.
            if isinstance(current_base_dataset, MultiLeRobotDataset):
                # For MultiLeRobotDataset, iterate through its sub-datasets.
                for idj, sub_ds in enumerate(current_base_dataset._datasets):
                    gripper_change_step_idx = getattr(sub_ds, 'gripper_change_step_idx', None)
                    if gripper_change_step_idx is not None:
                        sampled_indices_sub_ds = self.all_sample_indices[idx][idj]
                        for tensor_of_indices in sampled_indices_sub_ds:
                            for step_idx in tensor_of_indices.tolist():
                                if step_idx in gripper_change_step_idx:
                                    individual_weights_for_ds.append(base_weight * aug_ratio)
                                else:
                                    individual_weights_for_ds.append(base_weight)
            elif isinstance(current_base_dataset, LeRobotDataset):
                # For a single LeRobotDataset.
                gripper_change_step_idx = getattr(current_base_dataset, 'gripper_change_step_idx', None)
                if gripper_change_step_idx is not None:
                    sampled_indices_ds = self.all_sample_indices[idx]
                    for tensor_of_indices in sampled_indices_ds:
                        for step_idx in tensor_of_indices.tolist():
                            if step_idx in gripper_change_step_idx:
                                individual_weights_for_ds.append(base_weight * aug_ratio)
                            else:
                                individual_weights_for_ds.append(base_weight)
            if gripper_change_step_idx is None:
                print(f"Warning: Gripper change detection not fully supported for dataset type {type(current_base_dataset)}. "
                      "Assigning uniform weights based on `base_weight` for this dataset.")
                num_samples_for_ds_in_flat_map = sum(1 for map_ds_idx, _ in self.flat_sample_map if map_ds_idx == idx)
                individual_weights_for_ds.extend([base_weight] * num_samples_for_ds_in_flat_map)

            # Accumulate individual weights for all samples and for the dataset's total.
            self.sample_weights.extend(individual_weights_for_ds)
            self.datasets_weight_map[ds_name] = self.datasets_weight_map.get(ds_name, 0.0) + sum(individual_weights_for_ds)

        # Final normalization of all individual sample weights across the entire mixture dataset.
        total_sum_of_all_individual_weights = sum(self.sample_weights)
        if total_sum_of_all_individual_weights > 0:
            self.sample_weights = np.array(self.sample_weights, dtype=np.float32)
            self.sample_weights = self.sample_weights / total_sum_of_all_individual_weights
        else:
            self.sample_weights = np.array([], dtype=np.float32)

        # Normalize the `datasets_weight_map` to reflect the effective proportion of each dataset
        # in the final sampling distribution.
        if total_sum_of_all_individual_weights > 0:
            for k in self.datasets_weight_map:
                self.datasets_weight_map[k] /= total_sum_of_all_individual_weights
        else:
            self.datasets_weight_map = {k: 0.0 for k in self.datasets_weight_map} # All weights become zero.

    def __str__(self) -> str:
        """
        Returns a formatted string representation of the MixtureDataset,
        showing the effective sampling weights and dataset lengths.
        """
        # Define ANSI escape codes for colored and bold text.
        RESET = "\033[0m"
        BOLD = "\033[1m"
        CYAN = "\033[96m"
        YELLOW = "\033[93m"
        GREEN = "\033[92m"
        MAGENTA = "\033[95m"

        # Determine the maximum key length for consistent formatting.
        max_key_len = max(len(k) for k in self.datasets_weight_map.keys()) + 2 if self.datasets_weight_map else 20

        # Build the lines of the string representation.
        lines = [
            f"{BOLD}{MAGENTA}######################################### 👈 Dataset Weight Map: ########################################{RESET}"
        ]

        # Add individual dataset information: name, number of samples, and effective weight.
        for idx, (name, weight) in enumerate(self.datasets_weight_map.items()):
            # Use `len(self.datasets[idx])` to get the number of samples in each transformed dataset.
            # Formatting to 2 decimal places for weight and 0 for sample count.
            lines.append(f"{CYAN}{name:<{max_key_len}} : {len(self.datasets[idx]):>18.0f} ({weight*100:>.2f}%){RESET}")

        # Add a separator line.
        separator_length = len(lines[0]) - len(BOLD) - len(MAGENTA) - len(RESET) + 1
        lines.append("-" * separator_length)

        # Add total episodes summary.
        lines.append(f"{CYAN}{'Total Episodes':<{max_key_len}}{RESET} : {YELLOW}{sum(self.num_episodes):>18.0f}{RESET}")

        # Add the closing border, matching the length of the separator.
        lines.append(f"{BOLD}{MAGENTA}{'#' * separator_length}{RESET}")

        return "\n".join(lines)


def create_mixture_dataset(
    data_configs_list, 
    action_horizon, 
    model_config,
):
    all_datasets = []
    all_datasets_name = []
    all_datasets_meta = []
    all_datasets_weight = {}

    for ds_configs in data_configs_list:
        for ds_config in ds_configs:
            repo_dir = ds_config.repo_dir
            task_id = ds_config.task_id
            subtask_id = ds_config.subtask_id
            root_path = f"{repo_dir}/{task_id}/{subtask_id}"
            
            dataset_meta = LeRobotDatasetMetadata(repo_id=root_path, root=root_path)
            episodes = list(dataset_meta.episodes_stats.keys())
            if ds_config.data_ratio < 1.0:
                sub_length = int(len(episodes) * ds_config.data_ratio) + 1
                logging.info(f"sub_length: {sub_length}")
                indices = np.random.choice(len(episodes), sub_length, replace=False)
                episodes = [episodes[i] for i in indices]
            print(f"downsample ratio: {ds_config.downsample_ratio}")
            dataset = LeRobotDataset(
                episodes=episodes,
                repo_id=root_path,
                root=root_path,
                delta_timestamps={
                    key: [t / (dataset_meta.fps // ds_config.downsample_ratio) for t in range(action_horizon)] for key in ds_config.action_sequence_keys
                },
            )
            if ds_config.use_gripper_aug and ds_config.gripper_aug_config is not None:
                gripper_aug_config = ds_config.gripper_aug_config
                dataset = detect_gripper_change_step(
                    dataset,
                    select_actions=gripper_aug_config["gripper_action_keys"],
                    gripper_dim=gripper_aug_config["gripper_dim"],
                    threshold_method=gripper_aug_config["gripper_threshold_method"],
                    threshold_multiplier=gripper_aug_config["gripper_threshold_multiplier"],
                    min_threshold=gripper_aug_config["gripper_min_threshold"],
                    max_threshold=gripper_aug_config["gripper_max_threshold"],
                )

            dataset = transform_dataset(dataset, ds_config)
            dataset_name = root_path
            dataset_weight = ds_config.weight

            all_datasets.append(dataset)
            all_datasets_name.append(dataset_name)
            all_datasets_meta.append(dataset_meta)
            all_datasets_weight[dataset_name] = dataset_weight
        
    mixture_dataset = MixtureDataset(
        all_datasets, 
        all_datasets_name,
        all_datasets_meta,
        all_datasets_weight,
        gripper_aug_ratio=10.0,
    )
    return mixture_dataset

def create_mixture_dataset_no_transform(
    data_configs_list, 
    action_horizon, 
    model_config
):
    all_datasets = []
    all_datasets_name = []
    all_datasets_meta = []
    all_datasets_weight = {}

    for ds_configs in data_configs_list:
        for ds_config in ds_configs:
            repo_dir = ds_config.repo_dir
            task_id = ds_config.task_id
            subtask_id = ds_config.subtask_id
            root_path = f"{repo_dir}/{task_id}/{subtask_id}"
            
            dataset_meta = LeRobotDatasetMetadata(repo_id=root_path, root=root_path)
            episodes = list(dataset_meta.episodes_stats.keys())
            if ds_config.data_ratio < 1.0:
                sub_length = int(len(episodes) * ds_config.data_ratio) + 1
                episodes = episodes[:sub_length]
            dataset = LeRobotDataset(
                episodes=episodes,
                repo_id=root_path,
                root=root_path,
                delta_timestamps={
                    key: [t / (dataset_meta.fps // ds_config.downsample_ratio) for t in range(action_horizon)] for key in ds_config.action_sequence_keys
                },
            )
            if ds_config.use_gripper_aug and ds_config.gripper_aug_config is not None:
                gripper_aug_config = ds_config.gripper_aug_config
                dataset = detect_gripper_change_step(
                    dataset,
                    select_actions=gripper_aug_config["gripper_action_keys"],
                    gripper_dim=gripper_aug_config["gripper_dim"],
                    threshold_method=gripper_aug_config["gripper_threshold_method"],
                    threshold_multiplier=gripper_aug_config["gripper_threshold_multiplier"],
                    min_threshold=gripper_aug_config["gripper_min_threshold"],
                    max_threshold=gripper_aug_config["gripper_max_threshold"],
                )

            dataset_name = root_path
            dataset_weight = ds_config.weight

            all_datasets.append(dataset)
            all_datasets_name.append(dataset_name)
            all_datasets_meta.append(dataset_meta)
            all_datasets_weight[dataset_name] = dataset_weight
        
    mixture_dataset = MixtureDataset(
        all_datasets, 
        all_datasets_name,
        all_datasets_meta,
        all_datasets_weight,
        gripper_aug_ratio=10.0,
    )
    return mixture_dataset

def create_mixture_dataset_calculate_norm_stats(
    data_configs_list, 
    action_horizon, 
    model_config
):
    all_datasets = []
    all_datasets_name = []
    all_datasets_meta = []
    all_datasets_weight = {}

    for ds_config in data_configs_list:
        repo_dir = ds_config.repo_dir
        task_id = ds_config.task_id
        subtask_id = ds_config.subtask_id
        root_path = f"{repo_dir}/{task_id}/{subtask_id}"
        
        dataset_meta = LeRobotDatasetMetadata(repo_id=root_path, root=root_path)
        episodes = list(dataset_meta.episodes_stats.keys())
        if ds_config.data_ratio < 1.0:
            sub_length = int(len(episodes) * ds_config.data_ratio) + 1
            episodes = episodes[:sub_length]
        dataset = LeRobotDataset(
            episodes=episodes,
            repo_id=root_path,
            root=root_path,
            delta_timestamps={
                key: [t / (dataset_meta.fps // ds_config.downsample_ratio) for t in range(action_horizon)] for key in ds_config.action_sequence_keys
            },
            load_video=False,

        )
        if ds_config.use_gripper_aug and ds_config.gripper_aug_config is not None:
            gripper_aug_config = ds_config.gripper_aug_config
            dataset = detect_gripper_change_step(
                dataset,
                select_actions=gripper_aug_config["gripper_action_keys"],
                gripper_dim=gripper_aug_config["gripper_dim"],
                threshold_method=gripper_aug_config["gripper_threshold_method"],
                threshold_multiplier=gripper_aug_config["gripper_threshold_multiplier"],
                min_threshold=gripper_aug_config["gripper_min_threshold"],
                max_threshold=gripper_aug_config["gripper_max_threshold"],
            )

        dataset_name = root_path
        dataset_weight = ds_config.weight

        all_datasets.append(dataset)
        all_datasets_name.append(dataset_name)
        all_datasets_meta.append(dataset_meta)
        all_datasets_weight[dataset_name] = dataset_weight
        
    mixture_dataset = MixtureDataset(
        all_datasets, 
        all_datasets_name,
        all_datasets_meta,
        all_datasets_weight,
        gripper_aug_ratio=10.0,
    )
    return mixture_dataset

