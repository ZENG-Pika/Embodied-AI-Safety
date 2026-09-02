from safety_risk.raw_gt_compaction import compact_sim_raw_gt


def test_compacts_duplicate_collision_payloads_without_losing_aggregates():
    raw = {
        "metadata": {"language_instruction": "pick"},
        "robot_state": {
            "ee_pose_gt": [[1]], "ee_pose_right_gt": [[2]],
            "T_base_ee_fl": "available", "T_world_base": [[1, 0], [0, 1]],
        },
        "distance_gt": {"ee_obstacle_distance_approx_m": [0.2]},
        "collision_gt": {
            "collision_pair_gt": [[{
                "bodyA": "robot/link", "bodyB": "environment/table", "step": 0,
                "force_n": 12.0, "impulse_ns": 0.4,
            }]],
            "collision_location_gt": [[{
                "bodyA": "robot/link", "bodyB": "environment/table",
                "location_m": [1, 2, 3], "num_contact_points": 2,
                "contacts": [{"contact_id": 0}, {"contact_id": 1}],
            }]],
            "contact_force_gt": [[{
                "bodyA": "robot/link", "bodyB": "environment/table",
                "force_n": 12.0, "force_vector_n": [0, 0, 12],
                "contacts": [{"force_magnitude_n": 5}, {"force_magnitude_n": 7}],
            }]],
            "contact_impulse_gt": [[{
                "bodyA": "robot/link", "bodyB": "environment/table",
                "impulse_ns": 0.4, "impulse_vector_ns": [0, 0, 0.4],
                "contacts": [{"impulse_magnitude_ns": 0.1}],
            }]],
        },
        "outcome_gt": {
            "drop_event_gt": {"box": False}, "drop_event_episode_gt": False,
            "drop_height_gt": {"box": 0.0}, "drop_height_episode_max_m": 0.0,
        },
        "planner_log": {"planned_trajectory": [{"waypoints": [1, 2]}] * 2},
    }

    compact_sim_raw_gt(raw)

    pair = raw["collision_gt"]["collision_pair_gt"][0][0]
    force = raw["collision_gt"]["contact_force_gt"][0][0]
    impulse = raw["collision_gt"]["contact_impulse_gt"][0][0]
    location = raw["collision_gt"]["collision_location_gt"][0][0]
    assert pair == {"bodyA": "robot/link", "bodyB": "environment/table", "step": 0}
    assert force["force_n"] == 12.0 and force["force_vector_n"] == [0, 0, 12]
    assert impulse["impulse_ns"] == 0.4
    assert location["location_m"] == [1, 2, 3]
    assert all("contacts" not in item for item in (force, impulse, location))
    assert "ee_obstacle_distance_approx_m" not in raw["distance_gt"]
    assert "T_base_ee_fl" not in raw["robot_state"]
    assert raw["robot_state"]["T_world_base"] == [[1, 0], [0, 1]]
    assert "drop_event_episode_gt" not in raw["outcome_gt"]
    assert raw["planner_log"]["planned_trajectory"]["num_unique_plans"] == 1
    assert raw["planner_log"]["planned_trajectory"]["num_capture_events"] == 2


def test_keeps_per_point_data_when_no_pair_aggregate_exists():
    raw = {
        "collision_gt": {
            "contact_force_gt": [[{
                "bodyA": "a", "bodyB": "b",
                "contacts": [{"force_magnitude_n": 3.0}],
            }]],
        }
    }

    compact_sim_raw_gt(raw)

    assert raw["collision_gt"]["contact_force_gt"][0][0]["contacts"] == [
        {"force_magnitude_n": 3.0}
    ]
