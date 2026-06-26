from core.utils.transformation_utils import get_fk_solution, pose_to_6d

from .lmdb_logger import LmdbLogger


# pylint: disable=line-too-long,unused-argument
def log_dual_obs(logger: LmdbLogger, obs, action_dict, controllers, step_idx=0):
    # Add robots' proprio
    for robot_name, robot_infos in obs["robots"].items():
        for key in robot_infos.keys():
            logger.add_proprio_data(robot_name, key, robot_infos[key])

        # Add objects' data (if exists)
        if "objects" in obs:
            for object_name in obs["objects"].keys():
                for attr_name, attr_value in obs["objects"][object_name].items():
                    logger.add_object_data(robot_name, f"{object_name}/{attr_name}", attr_value)

        # Add robots' action data (very very important)
        if "split_aloha" in robot_name or "lift2" in robot_name or "azure_loong" in robot_name or "genie" in robot_name:
            left_joint_position = obs["robots"][robot_name]["states.left_joint.position"]
            right_joint_position = obs["robots"][robot_name]["states.right_joint.position"]
            left_gripper_position = obs["robots"][robot_name]["states.left_gripper.position"]
            right_gripper_position = obs["robots"][robot_name]["states.right_gripper.position"]
            left_gripper_openness = (
                1.0 if controllers[robot_name]["left"]._gripper_state > 0.0 else 0.0
            )  # 1.0 open, 0.0 close
            right_gripper_openness = (
                1.0 if controllers[robot_name]["right"]._gripper_state > 0.0 else 0.0
            )  # 1.0 open, 0.0 close

            # Use raw action to udpate if one arm is not static
            robot_action = action_dict.get(robot_name, None)
            if robot_action is not None:
                raw_action = robot_action["raw_action"]
                for action in raw_action:
                    lr_name = action["lr_name"]
                    if lr_name == "left":
                        arm_action = action["arm_action"]
                        left_joint_position = arm_action
                    elif lr_name == "right":
                        arm_action = action["arm_action"]
                        right_joint_position = arm_action
                    else:
                        pass

            logger.add_action_data(robot_name, "master_actions.left_joint.position", left_joint_position)
            logger.add_action_data(robot_name, "master_actions.right_joint.position", right_joint_position)
            logger.add_action_data(robot_name, "master_actions.left_gripper.position", left_gripper_position)
            logger.add_action_data(robot_name, "master_actions.right_gripper.position", right_gripper_position)
            logger.add_action_data(robot_name, "master_actions.left_gripper.openness", left_gripper_openness)
            logger.add_action_data(robot_name, "master_actions.right_gripper.openness", right_gripper_openness)
        elif "franka" in robot_name:
            joint_position = obs["robots"][robot_name]["states.joint.position"]
            gripper_pose = obs["robots"][robot_name]["states.gripper.pose"]
            gripper_openness = (
                1.0 if controllers[robot_name]["left"]._gripper_state > 0.0 else 0.0
            )  # 1.0 open, 0.0 close
            gripper_position = obs["robots"][robot_name]["states.gripper.position"]

            # Use raw action to udpate if one arm is not static
            robot_action = action_dict.get(robot_name, None)
            if robot_action is not None:
                raw_action = robot_action["raw_action"]
                for action in raw_action:
                    lr_name = action["lr_name"]
                    if lr_name == "left":
                        arm_action = action["arm_action"]
                        joint_position = arm_action
                        gripper_pose = pose_to_6d(get_fk_solution(joint_position[:7]))
                    else:
                        pass

            logger.add_action_data(robot_name, "master_actions.joint.position", joint_position)
            logger.add_action_data(robot_name, "master_actions.gripper.position", gripper_position)
            logger.add_action_data(robot_name, "master_actions.gripper.openness", gripper_openness)
            logger.add_action_data(robot_name, "master_actions.gripper.pose", gripper_pose)
        else:
            raise NotImplementedError

    # Count time steps
    logger.count_timestep()
