"""
Planning utilities for simbox motion generation.

This code is adapted from work by the Genie Sim Team; we thank them for
sharing their ideas and implementation.

This module provides helper functions that operate on cuRobo planning results
(`MotionGen` / `IKSolver` outputs) to:

- sort_by_difference_js: rank joint-space paths by how \"smooth\" they are,
  using cumulative joint differences (optionally weighted per joint). Used to
  pick the best trajectory among multiple candidates.
- filter_paths_by_position_error / filter_paths_by_rotation_error: apply
  simple statistical filters (threshold at mean error) on end-effector pose
  errors to reject outlier paths.
- get_prioritized_indices: combine the above filters to produce a prioritized
  list of candidate indices from a `result` object (e.g. `MotionGenResult`).
- select_index_by_priority_single: pick the single best candidate index from a
  single planning result, with safe fallback to index 0.
- select_index_by_priority_dual: pick a best index that is jointly successful
  in both a pre-planning result and a final result (dual-stage planning),
  with sensible fallbacks when intersections are empty.
"""

import random

import torch


def sort_by_difference_js(paths, weights=None):
    """
    Sorts a list of joint space paths based on the cumulative difference between consecutive waypoints.

    Args:
        paths (List(JointState): A list of JointState, each path contains a position of shape (T, D) where
                              T is the number of waypoints per path, and D is the dimensionality of each waypoint.
        weights (torch.tensor, optional): A tensor of shape (D,) representing weights for each dimension.
                                          If None, all dimensions are weighted equally.
    Returns:
        torch.tensor: Indices that would sort the paths based on the cumulative difference.
    """

    assert len(paths) > 0, "The paths list should not be empty."

    assert (
        weights is None or weights.shape[0] == paths[0].position.shape[-1]
    ), "Weights must be of shape (D,) where D is the dimensionality of each waypoint."

    device = paths[0].position.device
    if weights is None:
        weights = torch.ones(paths[0].position.shape[-1], device=device)
    else:
        weights = weights.to(device)

    # Calculate the absolute differences between consecutive waypoints
    diffs = []
    for path in paths:
        diff = torch.abs(path.position[1:, :] - path.position[:-1, :])  # Shape: (T-1, D)
        diff = diff.sum(dim=0)  # Average over waypoints, Shape: (D,)
        diffs.append(diff)
    diffs = torch.stack(diffs)  # Shape: (N, D)

    # Apply weights to the differences
    weighted_diffs = diffs * weights  # Broadcasting weights over the last dimension

    # Sum the weighted differences over all waypoints and dimensions
    cumulative_diffs = weighted_diffs.sum(dim=1)  # Shape: (N,)

    # Get the indices that would sort the paths based on cumulative differences
    sorted_indices = torch.argsort(cumulative_diffs)

    return sorted_indices


def filter_paths_by_position_error(paths, position_errors):
    """
    Filters out paths whose position error exceeds one sigma threshold.

    Args:
        paths (List(JointState): A list of JointState, each path contains a position of shape (T, D).
        position_errors (torch.tensor): A tensor of shape (N,) representing the position error for each path.

    Returns:
        List(bool): A filtered list of bool where each path's position error is below the threshold.
    """
    assert len(paths) == position_errors.shape[0], "The number of paths must match the number of position errors."

    mean_error = torch.mean(position_errors)
    torch.std(position_errors)
    threshold = mean_error  # + std_error  # one sigma threshold
    res = [position_error <= threshold for position_error in position_errors]

    return res


def filter_paths_by_rotation_error(paths, rotation_errors):
    """
    Filters out paths whose rotation error exceeds two sigma threshold.

    Args:
        paths (List(JointState): A list of JointState, each path contains a position of shape (T, D).
        rotation_errors (torch.tensor): A tensor of shape (N,) representing the rotation error for each path.

    Returns:
        List(bool): A filtered list of bool where each path's rotation error is below the threshold.
    """
    assert len(paths) == rotation_errors.shape[0], "The number of paths must match the number of rotation errors."

    mean_error = torch.mean(rotation_errors)
    torch.std(rotation_errors)
    threshold = mean_error  # + std_error  # one sigma threshold

    res = [rotation_error <= threshold for rotation_error in rotation_errors]
    return res


def get_prioritized_indices(result):
    """
    Extracts successful indices and returns them as a list ordered by priority:
    1. Filter by position and rotation errors.
    2. Sort by joint space (JS) difference.
    """
    if not torch.any(result.success):
        print("result failure")
        return []

    # Get absolute indices of successful samples
    success_indices = torch.nonzero(result.success, as_tuple=True)[0]
    print("success_indices :", success_indices)
    paths = [result.get_paths()[idx] for idx in success_indices]

    # Apply error filters
    pos_mask = filter_paths_by_position_error(paths, result.position_error[result.success])
    rot_mask = filter_paths_by_rotation_error(paths, result.rotation_error[result.success])

    # Identify local indices that pass both filters
    filtered_local_indices = [i for i, (p, r) in enumerate(zip(pos_mask, rot_mask)) if p and r]

    # Fallback to all successful paths if filtering returns nothing
    if not filtered_local_indices:
        target_paths = paths
        target_abs_indices = success_indices
    else:
        target_paths = [paths[i] for i in filtered_local_indices]
        target_abs_indices = success_indices[filtered_local_indices]

    # Sort target paths by JS difference
    sorted_rel_indices = sort_by_difference_js(target_paths)

    # Map back to absolute indices and return as a list
    return [target_abs_indices[i].item() for i in sorted_rel_indices]


def select_index_by_priority_single(result):
    """Select the best index from a single result."""
    prioritized_indices = get_prioritized_indices(result)
    return prioritized_indices[0] if prioritized_indices else 0


def select_index_by_priority_dual(pre_result, result):
    """Select the best index considering both pre_result and result."""
    # Get prioritized indices based on the final result
    prioritized_indices = get_prioritized_indices(result)
    if not prioritized_indices:
        return 0

    # Determine indices where both pre_result and result succeeded
    both_success_mask = pre_result.success & result.success
    both_success_indices = torch.nonzero(both_success_mask, as_tuple=True)[0]
    both_success_set = set(both_success_indices.cpu().tolist())

    if both_success_set:
        # Return the highest priority index that is successful in both results
        for idx in prioritized_indices:
            if idx in both_success_set:
                print("Pre and final both success, selected highest priority candidate.")
                return idx

        # Logically, the loop above should always find a match.
        # Fallback to random choice among common successes just in case.
        print("Pre and final both success, falling back to random choice.")
        return random.choice(list(both_success_set))

    # If no common successes exist, check if any final results succeeded
    if prioritized_indices:
        print("Only final success.")
        # Optionally return prioritized_indices[0] instead of random for better results
        return random.choice(prioritized_indices)

    return 0
