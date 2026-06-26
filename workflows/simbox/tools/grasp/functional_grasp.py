import json

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

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
    jaw_width,
    jaw_depth,
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

    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score  # red for high score
        color_g = 0
        color_b = 1 - score  # blue for low score

    left = create_mesh_box(jaw_depth + depth_base + finger_width, finger_width, height)
    right = create_mesh_box(jaw_depth + depth_base + finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, jaw_width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:, 0] -= depth_base + finger_width
    left_points[:, 1] -= jaw_width / 2 + finger_width
    left_points[:, 2] -= height / 2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:, 0] -= depth_base + finger_width
    right_points[:, 1] += jaw_width / 2
    right_points[:, 2] -= height / 2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:, 0] -= finger_width + depth_base
    bottom_points[:, 1] -= jaw_width / 2
    bottom_points[:, 2] -= height / 2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:, 0] -= tail_length + finger_width + depth_base
    tail_points[:, 1] -= finger_width / 2
    tail_points[:, 2] -= height / 2

    vertices = np.concatenate(
        [left_points, right_points, bottom_points, tail_points],
        axis=0,
    )
    vertices = np.dot(R1, vertices.T).T
    vertices = np.dot(rot_mat, vertices.T).T + center
    triangles = np.concatenate(
        [left_triangles, right_triangles, bottom_triangles, tail_triangles],
        axis=0,
    )
    colors = np.array([[color_r, color_g, color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)

    if return_axis:
        axis = create_rotated_coordinate_frame(size=0.1, R=rot_mat, t=center)
        return gripper, axis
    return gripper


def create_rotated_coordinate_frame(size=1.0, R=None, t=None):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

    if R is not None:
        if isinstance(R, list) or (isinstance(R, np.ndarray) and R.size == 3):
            R = Rotation.from_euler("xyz", R).as_matrix()
        mesh_frame.rotate(R, center=(0, 0, 0))

    if t is not None:
        mesh_frame.translate(t)

    return mesh_frame


def get_grasp_candidates(json_path):
    grasps = []
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    for value in data.values():
        grasps += value
    grasps = np.array(grasps)
    return grasps


def get_obj_pcd(obj_path, unit="m"):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(mesh.vertices)
    if unit == "mm":
        vertices_meters = vertices / 1000.0
    else:
        vertices_meters = vertices.copy()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_meters)
    obj_pcd = mesh.sample_points_uniformly(number_of_points=20000)
    return obj_pcd


def find_indices_kdtree(points_N, points_M, distance_threshold):
    tree = KDTree(points_N)
    min_distances, _ = tree.query(points_M)
    return np.where(min_distances < distance_threshold)[0]


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

    obj_mesh_path = args.obj_path
    max_grasps = args.N
    scale_unit = args.unit

    # Derive grasp path from obj_path
    sparse_grasp_path = obj_mesh_path.replace("_obj.obj", "_grasp_sparse.npy")
    sparse_grasps = np.load(sparse_grasp_path, allow_pickle=True)

    grippers = []
    tcp_centers = []
    filter_distance_threshold = 0.02

    for grasp in sparse_grasps[:max_grasps]:
        grasp_score = grasp[0]
        grasp_width = grasp[1]
        grasp_depth = grasp[3]
        grasp_rot_mat = grasp[4:13].reshape(3, 3) @ R1.T
        grasp_center = grasp[13:16]
        tcp_center = grasp_center + grasp_rot_mat[:3, 2] * grasp_depth
        gripper_mesh = create_franka_gripper_o3d(
            grasp_center,
            grasp_rot_mat,
            grasp_width,
            grasp_depth,
            score=grasp_score,
        )
        grippers.append(gripper_mesh)
        tcp_centers.append(tcp_center)

    tcp_centers_np = np.stack(tcp_centers)

    # Cropping
    obj_pcd_full = get_obj_pcd(obj_mesh_path, unit=scale_unit)
    obj_pcd_full.points = o3d.utility.Vector3dVector(np.array(obj_pcd_full.points))

    obj_center = obj_pcd_full.get_center()
    coordinate_frame = create_coordinate_frame(obj_center, size=0.05)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(obj_pcd_full)
    vis.run()
    vis.destroy_window()

    # Filtering grasp poses
    cropped_pcd = vis.get_cropped_geometry()
    print("Selected area points:", np.asarray(cropped_pcd.points))
    cropped_np = np.asarray(cropped_pcd.points)
    index = find_indices_kdtree(cropped_np, tcp_centers_np, filter_distance_threshold)

    grippers_filter = [grippers[j] for j in index]
    o3d.visualization.draw_geometries(
        [obj_pcd_full, *grippers_filter, coordinate_frame],
    )

    filter_sparse_grasps = sparse_grasps[index]
    np.save(sparse_grasp_path, filter_sparse_grasps, allow_pickle=True)
