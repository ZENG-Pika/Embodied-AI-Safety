from safety_risk.physx_collector import PhysXDataCollector


def test_human_body_pose_series_preserves_each_mano_link_and_frame():
    collector = PhysXDataCollector.__new__(PhysXDataCollector)
    collector._data = {
        "human_body_pose_gt": [
            {
                "obstacle_1": {
                    "root_prim_path": "/World/task_0/obstacle_1/mano",
                    "body_parts": {
                        "palm": {
                            "prim_path": "/World/task_0/obstacle_1/mano/palm",
                            "pose": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                        },
                        "index3": {
                            "prim_path": "/World/task_0/obstacle_1/mano/index3",
                            "pose": [0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                        },
                    },
                },
            },
            {
                "obstacle_1": {
                    "root_prim_path": "/World/task_0/obstacle_1/mano",
                    "body_parts": {
                        "palm": {
                            "prim_path": "/World/task_0/obstacle_1/mano/palm",
                            "pose": [0.01, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                        },
                        "index3": {
                            "prim_path": "/World/task_0/obstacle_1/mano/index3",
                            "pose": [0.11, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                        },
                    },
                },
            },
        ],
    }

    result = collector.build_human_body_pose_gt()

    assert result["surrogate_type"] == "articulated_mano_hand"
    assert result["coordinate_frame"] == "world"
    assert result["quaternion_order"] == "xyzw"
    assert result["position_unit"] == "m"
    assert result["num_steps"] == 2
    body_parts = result["obstacles"]["obstacle_1"]["body_parts"]
    assert sorted(body_parts) == ["index3", "palm"]
    assert len(body_parts["palm"]["pose_per_step"]) == 2
    assert body_parts["palm"]["pose_per_step"][1][0] == 0.01
    assert len(body_parts["index3"]["pose_per_step"]) == 2
