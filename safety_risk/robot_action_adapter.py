"""Runtime discovery of controllable SimBox robot joints for policy rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class UnsupportedRobotError(RuntimeError):
    pass


_GROUP_SPECS = (
    ("body", "body_indices", 0.01),
    ("head", "head_indices", 0.02),
    ("lift", "lift_indices", 0.005),
    ("left_arm", "left_joint_indices", 0.02),
    ("right_arm", "right_joint_indices", 0.02),
    ("left_gripper", "left_gripper_indices", 0.002),
    ("right_gripper", "right_gripper_indices", 0.002),
)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class ActionGroup:
    name: str
    indices: tuple[int, ...]
    output_slice: slice
    max_delta: float


@dataclass
class RobotActionAdapter:
    robot_name: str
    robot: Any
    groups: tuple[ActionGroup, ...]
    joint_indices: np.ndarray
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    max_deltas: np.ndarray

    @classmethod
    def discover(
        cls,
        robot_name: str,
        robot: Any,
        *,
        control_grippers: bool = False,
        group_delta_overrides: dict[str, float] | None = None,
    ) -> "RobotActionAdapter":
        overrides = group_delta_overrides or {}
        groups = []
        flat_indices: list[int] = []
        seen = set()
        for group_name, attr, default_delta in _GROUP_SPECS:
            if group_name.endswith("gripper") and not control_grippers:
                continue
            values = [int(index) for index in (getattr(robot, attr, []) or [])]
            values = [index for index in values if index not in seen]
            if not values:
                continue
            start = len(flat_indices)
            flat_indices.extend(values)
            seen.update(values)
            groups.append(ActionGroup(
                group_name,
                tuple(values),
                slice(start, len(flat_indices)),
                float(overrides.get(group_name, default_delta)),
            ))
        if not flat_indices:
            raise UnsupportedRobotError(
                f"{robot_name}: no supported controllable joint-index groups"
            )

        view = getattr(robot, "_articulation_view", None)
        if view is None or not hasattr(view, "get_dof_limits"):
            raise UnsupportedRobotError(f"{robot_name}: articulation DOF limits unavailable")
        limits = _numpy(view.get_dof_limits())
        if limits.ndim == 3:
            limits = limits[0]
        if limits.ndim != 2 or limits.shape[1] != 2:
            raise UnsupportedRobotError(
                f"{robot_name}: unexpected DOF limit shape {limits.shape}"
            )
        indices = np.asarray(flat_indices, dtype=np.int64)
        if indices.max(initial=-1) >= limits.shape[0]:
            raise UnsupportedRobotError(
                f"{robot_name}: configured joint index exceeds {limits.shape[0]} DOFs"
            )
        selected = limits[indices].astype(np.float64)
        lower, upper = selected[:, 0], selected[:, 1]
        valid = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
        if not valid.all():
            bad = indices[~valid].tolist()
            raise UnsupportedRobotError(
                f"{robot_name}: invalid or unbounded PhysX limits for DOFs {bad}"
            )
        max_deltas = np.concatenate([
            np.full(len(group.indices), group.max_delta, dtype=np.float64)
            for group in groups
        ])
        return cls(
            robot_name=robot_name,
            robot=robot,
            groups=tuple(groups),
            joint_indices=indices,
            lower_limits=lower,
            upper_limits=upper,
            max_deltas=max_deltas,
        )

    @property
    def action_dim(self) -> int:
        return int(self.joint_indices.size)

    def current_position(self) -> np.ndarray:
        state = self.robot.get_joints_state()
        positions = _numpy(state.positions).reshape(-1)
        return positions[self.joint_indices].astype(np.float64)

    def raw_action(self, target: np.ndarray, delta: np.ndarray) -> list[dict[str, Any]]:
        result = []
        for group in self.groups:
            values = target[group.output_slice].copy()
            item = {
                "lr_name": "whole",
                "control_group": group.name,
                "joint_indices": list(group.indices),
                "joint_positions": values,
                "policy_delta": delta[group.output_slice].copy(),
                "policy_name": "random-diffusion-policy",
            }
            if group.name == "left_arm":
                item.update({"lr_name": "left", "arm_action": values})
            elif group.name == "right_arm":
                item.update({"lr_name": "right", "arm_action": values})
            result.append(item)
        return result

    def schema_manifest(self) -> dict[str, Any]:
        return {
            "robot_name": self.robot_name,
            "action_dim": self.action_dim,
            "joint_indices": self.joint_indices.tolist(),
            "lower_limits": self.lower_limits.tolist(),
            "upper_limits": self.upper_limits.tolist(),
            "groups": [
                {"name": group.name, "indices": list(group.indices), "max_delta": group.max_delta}
                for group in self.groups
            ],
            "limit_source": "live_physx_articulation",
        }
