import json

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

R1 = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    """Author: chenxi-wang
    Create box instance with mesh representation.
    """
    box = o3d.geometry.TriangleMesh()
    vertices = np.array(
        [
            [0, 0, 0],
            [width, 0, 0],
            [0, 0, depth],
            [width, 0, depth],
            [0, height, 0],
            [width, height, 0],
            [0, height, depth],
            [width, height, depth],
        ]
    )
    vertices[:, 0] += dx
    vertices[:, 1] += dy
    vertices[:, 2] += dz
    triangles = np.array(
        [
            [4, 7, 5],
            [4, 6, 7],
            [0, 2, 4],
            [2, 6, 4],
            [0, 1, 2],
            [1, 3, 2],
            [1, 5, 7],
            [1, 7, 3],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 1],
            [1, 4, 5],
        ]
    )
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box


def create_franka_gripper_o3d(
    center,
    rot_mat,
    width,
    depth,
    depth_base=0.0584,
    score=1,
    color=None,
    return_axis=False,
):
    """
    Author: chenxi-wang

    **Input:**

    - center: numpy array of (3,), target point as gripper center

    - R: numpy array of (3,3), rotation matrix of gripper

    - width: float, gripper width

    - score: float, grasp quality score

    **Output:**

    - open3d.geometry.TriangleMesh
    """
    height = 0.004
    finger_width = 0.004
    tail_length = 0.04
    # depth_base = 0.0584

    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score  # red for high score
        color_g = 0
        color_b = 1 - score  # blue for low score

    left = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    right = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:, 0] -= depth_base + finger_width
    left_points[:, 1] -= width / 2 + finger_width
    left_points[:, 2] -= height / 2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:, 0] -= depth_base + finger_width
    right_points[:, 1] += width / 2
    right_points[:, 2] -= height / 2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:, 0] -= finger_width + depth_base
    bottom_points[:, 1] -= width / 2
    bottom_points[:, 2] -= height / 2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:, 0] -= tail_length + finger_width + depth_base
    tail_points[:, 1] -= finger_width / 2
    tail_points[:, 2] -= height / 2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    # vertices = np.concatenate([left_points], axis=0)

    # align axis for robotiq-2f-85 gripper
    #
    # R2 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    vertices = np.dot(R1, vertices.T).T
    # vertices = np.dot(R2, vertices.T).T

    vertices = np.dot(rot_mat, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([[color_r, color_g, color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)

    if return_axis:
        axis = create_rotated_coordinate_frame(size=0.1, rot=rot_mat, t=center)
        return gripper, axis
    return gripper


def create_rotated_coordinate_frame(size=1.0, rot=None, t=None):
    """
    Create a coordinate frame that can be rotated and translated.

    Args:
        size: frame size
        rot: rotation matrix (3x3) or rotation angles [rx, ry, rz] in radians
        t: translation vector [tx, ty, tz]
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

    # 如果提供了旋转参数
    if rot is not None:
        if isinstance(rot, list) or (isinstance(rot, np.ndarray) and rot.size == 3):
            rot = R.from_euler("xyz", rot).as_matrix()
        mesh_frame.rotate(rot, center=(0, 0, 0))

    if t is not None:
        mesh_frame.translate(t)

    return mesh_frame


def get_grasp_candidates(json_path):
    grasps = []
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    for _, value in data.items():
        grasps += value
    grasps = np.array(grasps)
    return grasps


def get_obj_pcd(mesh_path, unit="mm"):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)
    if unit == "mm":
        vertices = vertices / 1000.0
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    obj_pcd = mesh.sample_points_uniformly(number_of_points=200000)
    return obj_pcd


def create_coordinate_frame(origin=None, size=0.1):
    if origin is None:
        origin = [0, 0, 0]
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.translate(origin)
    return frame


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", required=True, help="Path to the object mesh file")
    parser.add_argument("--N", default=3000, type=int, help="Number of grasps to visualize")
    parser.add_argument("--unit", choices=["mm", "m"], required=True, help="Unit of the mesh (mm or m)")
    args = parser.parse_args()

    obj_path = args.obj_path
    N = args.N
    scale_unit = args.unit

    # Derive grasp path from obj_path
    sparse_grasp_path = obj_path.replace("_obj.obj", "_grasp_sparse.npy")

    sparse_grasp = np.load(sparse_grasp_path, allow_pickle=True)
    grippers = []
    print("widths :", sparse_grasp[:N, 1])
    print("sparse_grasp.shape :", sparse_grasp.shape)
    for grasp in sparse_grasp[:N]:
        grasp_score = grasp[0]
        grasp_width = grasp[1]
        grasp_depth = grasp[3]
        grasp_rot = grasp[4:13].reshape(3, 3) @ R1.T  # canonical gripper2object
        grasp_center = grasp[13:16]
        gripper_mesh = create_franka_gripper_o3d(
            grasp_center,
            grasp_rot,
            grasp_width,
            grasp_depth,
            score=grasp_score,
        )
        grippers.append(gripper_mesh)

    this_obj_pcd = get_obj_pcd(obj_path, unit=scale_unit)
    this_obj_pcd.points = o3d.utility.Vector3dVector(np.array(this_obj_pcd.points))

    obj_center = this_obj_pcd.get_center()
    coordinate_frame = create_coordinate_frame(obj_center, size=0.05)

    o3d.visualization.draw_geometries([this_obj_pcd, *grippers, coordinate_frame])
