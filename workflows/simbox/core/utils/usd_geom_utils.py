# pylint: skip-file
# flake8: noqa
import os
import sys
from typing import List, Sequence, Tuple

import numpy as np
import open3d as o3d
from pxr import Gf, Usd, UsdGeom


def to_list(data: Sequence):
    """Convert sequence-like data to a list, returning an empty list for None."""
    if data is None:
        return []
    return list(data)


def recursive_parse_new(prim: Usd.Prim) -> Tuple[list, list, list]:
    """Recursively collect mesh vertices and face indices from a prim subtree in world coordinates."""
    points_total: List = []
    faceVertexCounts_total: List[int] = []
    faceVertexIndices_total: List[int] = []

    if prim.IsA(UsdGeom.Mesh):
        prim_imageable = UsdGeom.Imageable(prim)
        xform_world_transform = np.array(prim_imageable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

        points = prim.GetAttribute("points").Get()
        faceVertexCounts = prim.GetAttribute("faceVertexCounts").Get()
        faceVertexIndices = prim.GetAttribute("faceVertexIndices").Get()
        faceVertexCounts = to_list(faceVertexCounts)
        faceVertexIndices = to_list(faceVertexIndices)
        points = to_list(points)

        points = np.array(points)  # Nx3
        ones = np.ones((points.shape[0], 1))  # Nx1
        points_h = np.hstack([points, ones])  # Nx4
        points_transformed_h = np.dot(points_h, xform_world_transform)  # Nx4
        points_transformed = points_transformed_h[:, :3] / points_transformed_h[:, 3][:, np.newaxis]  # Nx3
        points = points_transformed.tolist()

        base_num = len(points_total)
        faceVertexIndices = np.array(faceVertexIndices)
        faceVertexIndices_total.extend((base_num + faceVertexIndices).tolist())
        faceVertexCounts_total.extend(faceVertexCounts)
        points_total.extend(points)

    children = prim.GetChildren()
    for child in children:
        child_points, child_faceVertexCounts, child_faceVertexIndices = recursive_parse_new(child)
        base_num = len(points_total)
        child_faceVertexIndices = np.array(child_faceVertexIndices)
        faceVertexIndices_total.extend((base_num + child_faceVertexIndices).tolist())
        faceVertexCounts_total.extend(child_faceVertexCounts)
        points_total.extend(child_points)

    return (
        points_total,
        faceVertexCounts_total,
        faceVertexIndices_total,
    )


def get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices):
    """Build an Open3D triangle mesh from USD mesh point and face data."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    triangles = []
    idx = 0
    for count in faceVertexCounts:
        if count == 3:
            triangles.append(faceVertexIndices[idx : idx + 3])
        idx += count
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def sample_points_from_mesh(mesh, num_points: int = 1000):
    """Uniformly sample points from an Open3D mesh."""
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd


def sample_points_from_prim(prim: Usd.Prim, num_points: int = 1000) -> np.ndarray:
    """Sample points from a prim subtree by converting meshes to a sampled point cloud."""
    points, faceVertexCounts, faceVertexIndices = recursive_parse_new(prim)
    mesh = get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices)
    pcd = sample_points_from_mesh(mesh, num_points)
    return np.asarray(pcd.points)


def compute_bbox(prim: Usd.Prim) -> Gf.Range3d:
    """Compute an axis-aligned world-space bounding box for a prim subtree."""
    imageable: UsdGeom.Imageable = UsdGeom.Imageable(prim)
    time = Usd.TimeCode.Default()
    bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
    bound_range = bound.ComputeAlignedBox()
    return bound_range


if __name__ == "__main__":
    # Simple manual test/debug hook kept from original peizhou_prim.py
    dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    models_folder = os.path.join(dir_path, "minidump_usd/0001/dest_usd/models_z/models/")
    subfolders = [f.name for f in os.scandir(models_folder) if f.is_dir()]
    for subfolder in subfolders:
        print(subfolder)
        usd_path = os.path.join(models_folder, subfolder, "instance.usd")
        stage = Usd.Stage.Open(usd_path)
        prim = stage.GetPrimAtPath("/Root/Instance")
        points, faceVertexCounts, faceVertexIndices = recursive_parse_new(prim)
        mesh = get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices)
        o3d.visualization.draw_geometries([mesh])
        pcd = sample_points_from_mesh(mesh, num_points=10000)
        o3d.visualization.draw_geometries([pcd])
