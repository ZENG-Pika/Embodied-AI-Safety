from types import SimpleNamespace

import numpy as np
import pytest

from safety_risk.robot_action_adapter import RobotActionAdapter, UnsupportedRobotError


class _View:
    def __init__(self, limits):
        self._limits = np.asarray([limits], dtype=float)

    def get_dof_limits(self):
        return self._limits


class _Robot:
    left_joint_indices = [0, 1]
    right_joint_indices = [2, 3]
    left_gripper_indices = [4]
    right_gripper_indices = [5]
    body_indices = []
    head_indices = []
    lift_indices = [6]

    def __init__(self, limits):
        self._articulation_view = _View(limits)

    def get_joints_state(self):
        return SimpleNamespace(positions=np.arange(7, dtype=float) / 10)


def test_discovers_groups_and_live_limits():
    robot = _Robot([[-1, 1]] * 7)
    adapter = RobotActionAdapter.discover("dual", robot)
    assert [group.name for group in adapter.groups] == ["lift", "left_arm", "right_arm"]
    assert adapter.action_dim == 5
    np.testing.assert_allclose(adapter.current_position(), [0.6, 0.0, 0.1, 0.2, 0.3])
    assert adapter.schema_manifest()["limit_source"] == "live_physx_articulation"


def test_optional_gripper_groups():
    adapter = RobotActionAdapter.discover(
        "dual", _Robot([[-1, 1]] * 7), control_grippers=True
    )
    assert adapter.action_dim == 7
    assert "left_gripper" in [group.name for group in adapter.groups]


def test_rejects_invalid_live_limits():
    limits = [[-1, 1]] * 7
    limits[2] = [0, 0]
    with pytest.raises(UnsupportedRobotError, match="invalid or unbounded"):
        RobotActionAdapter.discover("bad", _Robot(limits))
