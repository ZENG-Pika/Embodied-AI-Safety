import logging
import os
import random

import numpy as np
import torch

try:
    from pxr.Usd import Prim
except Exception:  # pylint: disable=broad-except
    print("No pxr found")
    Prim = None

os.environ["NO_PROXY"] = os.environ["NO_PROXY"] + r"\," + "localhost" if "NO_PROXY" in os.environ else "localhost"

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def set_semantic_label(prim: Prim, label):
    from omni.isaac.core.utils.semantics import add_update_semantics

    if prim.GetTypeName() == "Mesh":
        add_update_semantics(prim, semantic_label=label, type_label="class")
    all_children = prim.GetAllChildren()
    for child in all_children:
        set_semantic_label(child, label)


def set_plane_semantic_label(prim: Prim, label):
    from omni.isaac.core.utils.semantics import add_update_semantics

    if prim.GetTypeName() == "Plane":
        add_update_semantics(prim, semantic_label=label, type_label="class")
    all_children = prim.GetAllChildren()
    for child in all_children:
        set_plane_semantic_label(child, label)


def set_robot_semantic_label(robot: Prim, parent_name: str):
    from omni.isaac.core.utils.semantics import add_update_semantics

    if robot.GetTypeName() == "Mesh":
        prim_path = str(robot.GetPrimPath())
        prim_path = prim_path.replace(parent_name, "")
        if "panda_link" in prim_path:
            class_label = prim_path.split("/")[1]
        elif "mount" in prim_path:
            class_label = "mount"
        elif "Robotiq_2F_85" in prim_path:
            class_label = prim_path.split("/")[2]
            class_label = f"Robotiq_2F_85_{class_label}"
        add_update_semantics(robot, semantic_label=class_label, type_label="class")
    all_children = robot.GetAllChildren()
    for child in all_children:
        set_robot_semantic_label(child, parent_name)


def set_random_seed(seed):
    assert isinstance(
        seed,
        int,
    ), f'Expected "seed" to be an integer, but it is "{type(seed)}".'
    print(f"set seed:{seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_plane_vector_angle(x1, y1, z1, threshold_deg):
    normal = np.cross(x1, y1)

    dot_product = np.dot(normal, z1)
    norm_normal = np.linalg.norm(normal)
    norm_z1 = np.linalg.norm(z1)

    cos_theta = abs(dot_product) / (norm_normal * norm_z1)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    angle_deg = 90 - np.degrees(theta)

    return angle_deg < threshold_deg
