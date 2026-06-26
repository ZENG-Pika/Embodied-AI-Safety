# pylint: skip-file
# flake8: noqa
import os
import argparse
import numpy as np
import open3d as o3d
import json
from pxr import Usd, UsdGeom

def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

colors = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 0.5],
]  # red, green, blue

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config file")
args = parser.parse_args()

# Load configuration
with open(args.config, 'r') as f:
    config = json.load(f)

# Use configuration
TASK = "close_h_down"
dir_name = config['DIR']

dir_name_kps = os.path.join(config['DIR'], "Kps", TASK)
os.makedirs(dir_name_kps, exist_ok=True)
usd_file = os.path.join(dir_name, "instance.usd")
keypoint_path = os.path.join(dir_name_kps, "keypoints.json")
target_keypoint_path = os.path.join(dir_name_kps, "keypoints_final.json")

def scale_pcd_to_unit(pcd):
    # Get all points in the point cloud
    points = np.asarray(pcd.points)

    # Find min and max values
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    # Center and range
    center = (min_bound + max_bound) / 2.0
    scale = (max_bound - min_bound).max() / 2.0  # Scale using the longest edge

    # Scale point cloud to [-1, 1]
    scaled_points = (points - center) / scale
    pcd.points = o3d.utility.Vector3dVector(scaled_points)

    return pcd

def get_full_transformation(prim):
    """
    Get the full transformation matrix of the object relative to the world coordinate system,
    including transformations from all ancestor objects.
    """
    transform_matrix = np.identity(4)  # Initialize as identity matrix

    # Starting from the object, apply transformations from all ancestors level by level
    current_prim = prim
    while current_prim:
        # Get the transformation matrix of the current object
        xform = UsdGeom.Xform(current_prim)
        local_transform = xform.GetLocalTransformation()

        # Apply the current object's transformation to the accumulated transformation
        transform_matrix = np.dot(local_transform.GetTranspose(), transform_matrix)

        # Move to parent object
        current_prim = current_prim.GetParent()

    return transform_matrix

def convert_to_world_coordinates(prim, local_vertices):
    """
    Convert local coordinates to world coordinates.
    """
    # Get transformation matrix relative to world coordinate system
    transform_matrix = get_full_transformation(prim)

    # Convert local vertices to world coordinate system
    world_vertices = []
    for vertex in local_vertices:
        # Convert to homogeneous coordinates
        local_point = np.append(vertex, 1)  # [x, y, z, 1]

        # Convert to world coordinate system
        world_point = np.dot(transform_matrix, local_point)[:3]  # Take only first 3 coordinates
        world_vertices.append(world_point)

    return np.array(world_vertices)

def extract_all_geometry_from_usd(usd_file):
    # Load USD file
    stage = Usd.Stage.Open(usd_file)

    # Store geometric information for all objects
    all_meshes = []

    # Traverse all Prims
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            # Extract Mesh
            mesh = UsdGeom.Mesh(prim)

            # Get vertices
            points = mesh.GetPointsAttr().Get()  # List of vertices
            vertices = np.array([[p[0], p[1], p[2]] for p in points])

            # If conversion to world coordinate system is needed, call conversion function
            vertices = convert_to_world_coordinates(prim, vertices)

            # Get face vertex indices
            face_indices = mesh.GetFaceVertexIndicesAttr().Get()  # Indices of all face vertices
            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()  # Number of vertices per face

            # Split indices and triangulate
            faces = []
            index = 0
            for count in face_vertex_counts:
                face = face_indices[index:index + count]
                index += count
                if len(face) == 3:
                    faces.append(face)  # Already a triangle
                elif len(face) > 3:
                    for i in range(1, len(face) - 1):
                        faces.append([face[0], face[i], face[i + 1]])

            faces = np.array(faces)

            # Get normals (if available)
            if mesh.GetNormalsAttr().IsAuthored() and mesh.GetNormalsAttr().Get() is not None:
                normals = mesh.GetNormalsAttr().Get()
                normals = np.array([[n[0], n[1], n[2]] for n in normals])
            else:
                normals = None

            # Store geometric information for current object
            all_meshes.append((vertices, faces, normals))

    # If no geometry is found, raise exception
    if not all_meshes:
        raise ValueError("No geometry found in USD file.")

    return all_meshes

def visualize_geometries(meshes):
    # Build Open3D Mesh and visualize
    all_meshes = o3d.geometry.TriangleMesh()
    for idx , (vertice, face, _) in enumerate(meshes):  # Ignore normals
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertice)
        mesh.triangles = o3d.utility.Vector3iVector(face)
        mesh.paint_uniform_color(np.random.random(size=3))
        all_meshes+=mesh

    pcd = all_meshes.sample_points_uniformly(number_of_points=10000000)
    return pcd


# Extract geometric information
all_meshes = extract_all_geometry_from_usd(usd_file)
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
viewer = o3d.visualization.VisualizerWithEditing()
viewer.create_window()

# Visualize
pcd=visualize_geometries(all_meshes)
viewer.add_geometry(pcd)
opt = viewer.get_render_option()
opt.show_coordinate_frame = True

viewer.run()
viewer.destroy_window()

print("saving picked points")
picked_points = viewer.get_picked_points()

if len(picked_points) == 0:
    print("No points were picked")
    exit()

xyz = np.asarray(pcd.points)
print(picked_points)
picked_points = xyz[picked_points]
print(picked_points)
color_lists = ["red", "yellow", "blue", "green", "magenta", "purple", "orange"]

keypoint_description_file = os.path.join(dir_name_kps, "keypoints.json")
keypoint_info = {
    "keypoints": {c: p.tolist() for c, p in zip(color_lists, picked_points)},
}
with open(keypoint_description_file, "w") as f:
    json.dump(keypoint_info, f, indent=4, sort_keys=True)
print("keypoint_info saved to", keypoint_description_file)
