from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from safety_risk.physx_collector import PhysXDataCollector
from safety_risk.raw_gt_extractor import SimRawGTExtractor
from safety_risk.schema import RiskLevel
from safety_risk.sim_feature_extractor import REQUESTED_FEATURES, SimFeatureExtractor
from safety_risk.sim_label_extractor import SimLabelExtractor


class _View:
    def __init__(self, q, dq, tau):
        self.q = np.asarray([q], dtype=float)
        self.dq = np.asarray([dq], dtype=float)
        self.tau = np.asarray([tau], dtype=float)

    def get_joint_positions(self):
        return self.q

    def get_joint_velocities(self):
        return self.dq

    def get_measured_joint_efforts(self):
        return self.tau


class _Robot:
    def __init__(self, names, q, left=(), right=(), gripper=(), lift=()):
        self.dof_names = names
        self.left_joint_indices = list(left)
        self.right_joint_indices = list(right)
        self.left_gripper_indices = list(gripper)
        self.right_gripper_indices = []
        self.lift_indices = list(lift)
        self._articulation_view = _View(q, [0.1] * len(q), [1.0] * len(q))


def test_joint_collector_flattens_multiple_robot_types_without_fixed_width():
    task = SimpleNamespace(robots=OrderedDict([
        ("franka", _Robot(["panda_joint1", "panda_joint2", "finger"], [1, 2, 0.03], left=[0, 1], gripper=[2])),
        ("lift2", _Robot(["lift", "left1", "right1"], [0.2, 3, 4], left=[1], right=[2], lift=[0])),
    ]))
    collector = PhysXDataCollector()
    collector._collect_joint_states(task)
    collector._collect_joint_torques(task)
    data = collector.get_raw_data()
    assert data["joint_position_q_gt"] == [[1.0, 2.0, 0.03, 0.2, 3.0, 4.0]]
    assert data["joint_velocity_dq_gt"][0] == [0.1] * 6
    assert data["joint_torque_gt"][0] == [1.0] * 6
    metadata = data["joint_state_metadata"]
    assert metadata["dof_names"] == [
        "franka/panda_joint1", "franka/panda_joint2", "franka/finger",
        "lift2/lift", "lift2/left1", "lift2/right1",
    ]
    assert metadata["risk_metric_indices"] == [0, 1, 3, 4, 5]
    assert metadata["joint_limit_metric_indices"] == [0, 1, 4, 5]
    assert metadata["channels"][3]["position_unit"] == "m"


def test_not_applicable_fields_do_not_reduce_feature_coverage():
    extractor = SimFeatureExtractor()
    extractor._not_applicable = {"pt.drop_flag_gt": "not a portable task"}
    sections = {
        section: {field: None for field in fields}
        for section, fields in REQUESTED_FEATURES.items()
    }
    common = extractor._extract_common(
        {"robot_state": {}}, sections["hs"], sections["pt"], sections["rs"], sections["ir"]
    )
    assert "drop_flag_gt" in common["missing_field_status"]["not_applicable"]
    assert "drop_flag_gt" not in common["missing_fields"]


def test_positive_level_is_reported_as_lower_bound_when_other_fields_missing():
    evaluation = SimpleNamespace(
        hs_level=RiskLevel.L0,
        pt_level=RiskLevel.L0,
        rs_level=RiskLevel.L2,
        ir_level=RiskLevel.L0,
        root_cause=[],
    )
    features = {
        section: {field: None for field in fields}
        for section, fields in REQUESTED_FEATURES.items()
    }
    features["field_quality"] = {
        section: {field: {"status": "unavailable"} for field in fields}
        for section, fields in REQUESTED_FEATURES.items()
    }
    labels, status = SimLabelExtractor()._extract_risk_labels(evaluation, features)
    assert labels["risk_label_RS_auto"] == "L2"
    assert status["RS"]["status"] == "lower_bound_due_to_missing_data"
    assert labels["risk_label_HS_auto"] is None


def test_ee_pose_rebuild_matches_absolute_runtime_path_to_relative_link_name():
    extractor = SimRawGTExtractor()
    raw = {
        "robot_state": {
            "link_pose_gt": [
                {"franka": {"panda_hand": [1, 2, 3, 0, 0, 0, 1]}},
                {"franka": {"panda_hand": [2, 3, 4, 0, 0, 0, 1]}},
            ],
            "joint_state_metadata": {
                "ee_link_paths": {
                    "franka": {
                        "left": "/World/task_0/franka/fr3/panda_hand",
                        "right": "",
                    }
                }
            },
        }
    }
    extractor._rebuild_ee_poses_from_link_pose_gt(raw)
    assert raw["robot_state"]["ee_pose_gt"] == [
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        [2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0],
    ]


def test_ee_pose_rebuild_keeps_dual_arm_link6_paths_distinct():
    extractor = SimRawGTExtractor()
    raw = {
        "robot_state": {
            "link_pose_gt": [{"lift2": {
                "lift2/lift2/fl/link6": [1, 0, 0, 0, 0, 0, 1],
                "lift2/lift2/fr/link6": [2, 0, 0, 0, 0, 0, 1],
            }}],
            "joint_state_metadata": {"ee_link_paths": {"lift2": {
                "left": "/World/task_0/lift2/lift2/lift2/fl/link6",
                "right": "/World/task_0/lift2/lift2/lift2/fr/link6",
            }}},
        }
    }
    extractor._rebuild_ee_poses_from_link_pose_gt(raw)
    assert raw["robot_state"]["ee_pose_gt"][0][0] == 1.0
    assert raw["robot_state"]["ee_pose_right_gt"][0][0] == 2.0
