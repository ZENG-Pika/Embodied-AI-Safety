from collections import deque
from typing import List, Dict, Optional, Any, Sequence, Deque, Union
import datasets
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def check_final(
    last_states: Union[Deque[Sequence[float]], Sequence[Sequence[float]], torch.Tensor],
    *,
    # 索引与初始状态
    arm_dofs: int = 6,                              # 左臂关节数（这里按你给的 6）
    gripper_index: int = -1,                        # 夹爪在向量中的索引（默认最后一维）
    mean_initial_arm_state: Optional[Sequence[float]] = (0.0107, 0.0527, 0.0463, -0.0415, 0.0187, 0.0108),
    mean_initial_gripper_state: float = 4.8438,     # 目前不参与判定，保留以便后续扩展

    # 判定阈值（角度阈值用“度”直观易调，内部会转换为弧度）
    stability_window: int = 5,                     # 最近多少帧用于判“没有太大变化”
    per_joint_range_deg: float = 2.0,               # 窗口内每个关节的最大幅度（max-min）阈值（度）
    mean_speed_deg: float = 0.5,                    # 邻帧关节差的平均 L2（每步）阈值（度/步）
    min_change_from_initial_deg: float = 15.0,      # 末帧相对初始的“至少变化量”（L2，度）
    gripper_closed_thresh: float = 0.8,             # 夹爪关闭阈值（数值越小说明越闭合）
) -> bool:
    """
    返回 True 表示“到位”：(1) 最近窗口内姿态变化不大 & (2) 夹爪关闭 & (3) 末帧与初始相差足够大。
    所有角度的阈值以“度”给出，这里会自动转弧度再比较。
    """
    # --- 数据整理为 (N, D) tensor ---
    if isinstance(last_states, torch.Tensor):
        states = last_states
    else:
        states = torch.as_tensor(list(last_states), dtype=torch.float32)

    if states.ndim != 2:
        raise ValueError(f"last_states should be 2D, got shape {tuple(states.shape)}")
    N, D = states.shape
    if D < arm_dofs:
        raise ValueError(f"Expected at least {arm_dofs} dims for arm + gripper, got {D}")
    if N < 2:
        return False  # 样本太少，无法判定稳定

    # 取最近窗口
    w = min(N, stability_window)
    window = states[-w:]                     # (w, D)
    arm = window[:, :arm_dofs]               # (w, 6)
    last_arm = arm[-1]                       # (6,)
    last_gripper = float(window[-1, gripper_index])

    # --- 1) 最近 w 帧“没有太大变化” ---
    # 两个指标：每关节range（max-min）要小、相邻帧的平均“速度”要小
    deg2rad = torch.pi / 180.0
    range_tol = per_joint_range_deg * deg2rad
    speed_tol = mean_speed_deg * deg2rad

    ranges = arm.max(dim=0).values - arm.min(dim=0).values            # (6,)
    max_range = float(ranges.abs().max())                              # 标量
    diffs = arm[1:] - arm[:-1]                                         # (w-1, 6)
    mean_speed = float(torch.linalg.norm(diffs, dim=1).mean())         # 每步的平均 L2

    stable = (max_range <= range_tol) and (mean_speed <= speed_tol)

    # --- 2) 夹爪关闭 ---
    gripper_closed = (last_gripper < gripper_closed_thresh)

    # --- 3) 末帧与“初始”差距要大 ---
    init = torch.as_tensor(mean_initial_arm_state, dtype=last_arm.dtype, device=last_arm.device)
    if init.numel() != arm_dofs:
        raise ValueError(f"mean_initial_arm_state length {init.numel()} != arm_dofs {arm_dofs}")
    dist_from_init = float(torch.linalg.norm(last_arm - init))
    far_from_init = (dist_from_init >= (min_change_from_initial_deg * deg2rad))

    # 组合判定
    return bool(stable and gripper_closed and far_from_init)
    # return bool(gripper_closed and far_from_init)


def get_last_frames(ds: LeRobotDataset, include_images: bool = False, keys=None):
    """
    Quickly fetch the last frame of each episode in a LeRobotDataset.
    - include_images=False: Return only scalar/vector fields from parquet (faster, no video decoding).
    - include_images=True : Additionally decode the corresponding image/video frame for the last frame.
    - keys: Limit the set of columns to retrieve (default: all non-image/video fields + timestamp, etc.).
    Returns: list[dict], where each element contains the last frame info of one episode.
    """
    # 1) Compute the global index of the last row for each episode.
    #    ds.episode_data_index['to'] is the exclusive end index, so last frame = to - 1.
    end_idxs = (ds.episode_data_index["to"] - 1).tolist()

    # 2) Determine which columns to load.
    #    By default, exclude video/image columns to avoid triggering slow video decoding.
    if keys is None:
        non_media_keys = [k for k, ft in ds.features.items() if ft["dtype"] not in ("image", "video")]
        keys = list(set(non_media_keys + ["timestamp", "episode_index", "task_index"]))

    # 3) Select all last-frame rows at once (does not call __getitem__, so no video decoding is triggered).
    last_rows = ds.hf_dataset.select(end_idxs)

    # 4) Build a dictionary of tensors for each requested key.
    out = []
    col = {k: last_rows[k] for k in keys}

    # Convert lists of tensors into stacked tensors for easier indexing.
    for k, v in col.items():
        # datasets.arrow_dataset.Column is the HuggingFace internal type for columns.
        if isinstance(v, datasets.arrow_dataset.Column) and len(v) > 0 and hasattr(v[0], "shape"):
            col[k] = torch.stack(v[:])

    # Iterate through each episode’s last frame and build a dict with its values.
    for i, ep_end in enumerate(end_idxs):
        item = {}
        for k in keys:
            val = col[k][i]
            # Unpack 0-dimensional tensors into Python scalars.
            if torch.is_tensor(val) and val.ndim == 0:
                val = val.item()
            item[k] = val

        # Map task_index back to the human-readable task string.
        if "task_index" in item:
            item["task"] = ds.meta.tasks[int(item["task_index"])]
        out.append(item)

    # 5) Optionally decode the actual image/video frame for each last timestamp.
    if include_images and len(ds.meta.video_keys) > 0:
        for i, ep_end in enumerate(end_idxs):
            ep_idx = int(out[i]["episode_index"])
            ts = float(out[i]["timestamp"])
            # Prepare a query dictionary: one timestamp per camera key.
            query_ts = {k: [ts] for k in ds.meta.video_keys}
            # Decode video frames at the specified timestamps for this episode.
            frames = ds._query_videos(query_ts, ep_idx)
            # Attach the decoded frame tensors to the output dictionary.
            for k, v in frames.items():
                out[i][k] = v

    return out


if __name__ == "__main__":
    # Initialize your dataset (replace with your repo ID or local path).
    ds = LeRobotDataset(repo_id="arx_lift2/pick_parcel_20250915")

    # Retrieve metadata only (timestamps, states, actions, tasks) without decoding video.
    last_infos = get_last_frames(ds, include_images=False)

    # Stack all 'observation.state' vectors into a single tensor for further processing.
    states = torch.stack([info['observation.state'] for info in last_infos])
    # Extract the left-arm joint states (first 7 values of each state vector).
    left_arm_states = states[:, 0:7]
    mean_state = torch.mean(left_arm_states, dim=0)
    std_state = torch.std(left_arm_states, dim=0)

    # Print the collected metadata for verification.
    print(last_infos)

    # --- Run check_final per episode using the last <=50 states ---

    EP_ARM_DOFS = 6                 # number of left-arm joints we use in check_final
    GRIPPER_COL_FULL = -1           # gripper is the last element in the full state vector
    STABILITY_WINDOW = 120           # must be consistent with check_final's default

    # Determine which episodes to iterate
    episode_indices = ds.episodes if ds.episodes is not None else sorted(ds.meta.episodes.keys())

    episode_flags = {}
    num_true, num_false = 0, 0

    for ep_idx in episode_indices:
        # Global index range [from_idx, to_idx) for this episode
        from_idx = int(ds.episode_data_index["from"][ep_idx])
        to_idx   = int(ds.episode_data_index["to"][ep_idx])

        if to_idx - from_idx <= 0:
            episode_flags[ep_idx] = False
            num_false += 1
            continue

        # Take the last <= STABILITY_WINDOW frames from this episode
        idxs = list(range(max(from_idx, to_idx - STABILITY_WINDOW), to_idx))
        rows = ds.hf_dataset.select(idxs)

        # Collect full "observation.state" (shape ~ [W, S])
        s_col = rows["observation.state"]
        if isinstance(s_col, datasets.arrow_dataset.Column):
            S = torch.stack(s_col[:])   # Column -> list[tensor] -> stack
        else:
            S = torch.stack(s_col)      # already a list[tensor]

        # Build the 7D small state per frame: first 6 joints + gripper
        # (Assumes the gripper signal is at the last position of the full state vector)
        small_states = torch.cat([S[:, :EP_ARM_DOFS], S[:, EP_ARM_DOFS:EP_ARM_DOFS+1]], dim=1)

        # Run your stopping logic
        ok = check_final(
            small_states,
            arm_dofs=EP_ARM_DOFS,
            gripper_index=-1,
            stability_window=STABILITY_WINDOW,
        )
        episode_flags[ep_idx] = bool(ok)
        num_true += int(ok)
        num_false += int(not ok)

    # Summary
    total_eps = len(episode_indices)
    print(f"[check_final] passed: {num_true} / {total_eps} ({(num_true/max(total_eps,1)):.1%})")

    # List some failed episodes for quick inspection
    failed_eps = [e for e, passed in episode_flags.items() if not passed]
    print("Failed episode indices (first 20):", failed_eps[:20])

