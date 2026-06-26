from pxr import Usd, UsdGeom
import numpy as np

import os
import os.path as osp
import trimesh
import json

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

def extract_all_geometry_from_usd(usd_file, local_frame=True):
    # Load USD file
    stage = Usd.Stage.Open(usd_file)

    # Store geometric information for all objects
    all_meshes = {
        "visuals": [],
        "collisions": [],
    }

    # Traverse all Prims
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):

            path = prim.GetPrimPath()
            curr_type = path.pathString.split("/")[-2]

            if curr_type not in list(all_meshes.keys()):
                curr_type = "visuals"

            # Extract Mesh
            mesh = UsdGeom.Mesh(prim)

            # Get vertices
            points = mesh.GetPointsAttr().Get()  # List of vertices
            vertices = np.array([[p[0], p[1], p[2]] for p in points])

            if not local_frame:
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
            all_meshes[curr_type].append((vertices, faces, normals))

    # If no geometry is found, raise exception
    if not all_meshes:
        raise ValueError("No geometry found in USD file.")

    return all_meshes

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    asset_fn = osp.join(config["DIR"], "instance.usd")

    dir_name = osp.dirname(asset_fn)

    ######## Load annotated keypoints
    keypoint_path = os.path.join(dir_name, "keypoints.json")  # in global coordinate
    keypoint_path_open = os.path.join(dir_name, "Kps/open_v", "keypoints.json")  # in global coordinate

    ######## Global

    all_meshes = extract_all_geometry_from_usd(asset_fn, local_frame=False)
    save_dir = osp.join(dir_name, "Meshes", "global_frame")

    os.makedirs(save_dir, exist_ok=True)

    for k, v in all_meshes.items():
        for i in range(len(v)):
            m = trimesh.Trimesh(vertices=v[i][0], faces=v[i][1])
            os.makedirs(os.path.join(save_dir, k), exist_ok=True)
            m.export(os.path.join(save_dir, k, "{}.obj".format(i)))

    ###### add keypoints -- Close Task
    kps = json.load(open(keypoint_path))["keypoints"]
    components = []
    radius = 0.02
    for k, v in kps.items():
        marker = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        marker.apply_translation(v)
        if k == "blue":
            marker.visual.vertex_colors[:, :3] = [0, 0, 255]
        elif k == "red":
            marker.visual.vertex_colors[:, :3] = [255, 0, 0]
        elif k == "yellow":
            marker.visual.vertex_colors[:, :3] = [255, 255, 0]
        else:
            raise NotImplementedError
        components.append(marker)
    merged_mesh = trimesh.util.concatenate(components)
    merged_mesh.export(os.path.join(save_dir, "kps_close.obj"))

    ###### add keypoints -- Open Task
    kps = json.load(open(keypoint_path_open))["keypoints"]
    components = []
    for k, v in kps.items():
        marker = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        marker.apply_translation(v)
        if k == "blue":
            marker.visual.vertex_colors[:, :3] = [0, 0, 255]
        elif k == "red":
            marker.visual.vertex_colors[:, :3] = [255, 0, 0]
        elif k == "yellow":
            marker.visual.vertex_colors[:, :3] = [255, 255, 0]
        else:
            raise NotImplementedError
        components.append(marker)
    merged_mesh = trimesh.util.concatenate(components)
    merged_mesh.export(os.path.join(save_dir, "kps_open.obj"))
