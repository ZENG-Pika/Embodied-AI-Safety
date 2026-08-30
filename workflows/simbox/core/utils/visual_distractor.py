from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import shapely
from concave_hull import concave_hull
from core.utils.dr import get_category_euler
from core.utils.usd_geom_utils import compute_bbox, recursive_parse_new
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon


def visualize_polygons(polygons: list[Polygon]):
    fig, ax = plt.subplots()
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.savefig("polygons.png")
    plt.close(fig)


def sort_points_clockwise(points: np.ndarray) -> np.ndarray:
    """Sort 2D points in clockwise order around their centroid."""
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    order = np.argsort(angles)
    return points[order]


def get_xy_contour(points, contour_type="convex_hull"):
    polygon = None
    if isinstance(points, o3d.geometry.PointCloud):
        points = np.asarray(points.points)
    if points.shape[1] == 3:
        points = points[:, :2]
    if contour_type == "convex_hull":
        xy_points = points
        hull = ConvexHull(xy_points)
        hull_points = xy_points[hull.vertices]
        sorted_points = sort_points_clockwise(hull_points)
        polygon = Polygon(sorted_points)
    elif contour_type == "concave_hull":
        xy_points = points
        concave_hull_points = concave_hull(xy_points)
        polygon = Polygon(concave_hull_points)
    return polygon


def rotate_object(obj, category):
    euler = get_category_euler(category)
    yaw = np.random.uniform(-180, 180)
    dr = R.from_euler("xyz", [0.0, 0.0, yaw], degrees=True)
    r = R.from_euler("xyz", euler, degrees=True)
    orientation = (dr * r).as_quat(scalar_first=True)
    obj.set_local_pose(orientation=orientation)
    # from pdb import set_trace
    # set_trace()


def get_pcd_from_mesh(mesh, num_points=1000):
    """Sample a mesh, returning None when the USD subtree has no triangles."""
    if mesh is None or len(mesh.vertices) < 3 or len(mesh.triangles) == 0:
        return None
    return mesh.sample_points_uniformly(number_of_points=num_points)


def mesh_from_usd_data(mesh_data):
    """Build a triangulated Open3D mesh from recursive USD mesh data."""
    points, face_counts, face_indices = mesh_data
    triangles = []
    offset = 0
    for count in face_counts:
        count = int(count)
        indices = face_indices[offset : offset + count]
        if count >= 3:
            for i in range(1, count - 1):
                triangles.append([indices[0], indices[i], indices[i + 1]])
        offset += count
    return o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(points),
        triangles=o3d.utility.Vector3iVector(triangles),
    )


def transform_pointcloud(
    pcd,
    translation,
    orientation,
) -> o3d.geometry.PointCloud:
    pcd_transformed = o3d.geometry.PointCloud()
    pcd_transformed.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).copy())

    if pcd.has_colors():
        pcd_transformed.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors).copy())
    if pcd.has_normals():
        pcd_transformed.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals).copy())

    T = np.eye(4)
    T[:3, :3] = R.from_quat(orientation, scalar_first=True).as_matrix()
    T[:3, 3] = translation

    pcd_transformed.transform(T)

    return pcd_transformed


def get_platform_available_polygon(platform_pc, pc_list, visualize=False, buffer_size=0.0):
    platform_polygon = get_xy_contour(platform_pc, contour_type="concave_hull")
    if visualize:
        polygons = []
        for pc in pc_list:
            polygons.append(get_xy_contour(pc, contour_type="concave_hull"))
        visualize_polygons(polygons + [platform_polygon])
    for pc in pc_list:
        pc_polygon = get_xy_contour(pc, contour_type="concave_hull").buffer(buffer_size)
        platform_polygon = platform_polygon.difference(pc_polygon)
    return platform_polygon


def compute_pcd_bbox(pcd):
    aabb = pcd.get_axis_aligned_bounding_box()
    return aabb


def bbox_to_polygon(x_min, y_min, x_max, y_max):
    points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    return Polygon(points)


def find_polygon_placement(large_polygon, small_polygon, buffer_thresh=0.03, max_attempts=1000):
    if large_polygon.is_empty or small_polygon.is_empty:
        return []

    safe_region = large_polygon.buffer(-buffer_thresh)
    if safe_region.is_empty:
        return []

    minx, miny, maxx, maxy = safe_region.bounds
    valid_placements = []

    for _ in range(max_attempts):
        coords = np.array(small_polygon.exterior.coords)
        small_centroid = np.mean(coords, axis=0)
        tx = np.random.uniform(minx, maxx)
        ty = np.random.uniform(miny, maxy)
        translation = np.array([tx, ty])

        transformed_polygon = shapely.affinity.translate(
            small_polygon,
            xoff=translation[0] - small_centroid[0],
            yoff=translation[1] - small_centroid[1],
        )

        if safe_region.contains(transformed_polygon):
            valid_placements.append((translation - small_centroid, 0))
            break

    return valid_placements


def randomly_place_object_on_object(
    object1_pcd,
    object2_pcd,
    object1,
    object2,
    available_polygon,  #: Polygon = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
    restrict_polygon=None,  #: Polygon = None,
):
    object1_polygon = get_xy_contour(object1_pcd, contour_type="concave_hull")
    object2_polygon = get_xy_contour(object2_pcd, contour_type="concave_hull")

    object2_polygon = object2_polygon.intersection(available_polygon)
    if restrict_polygon is not None:
        object2_polygon = object2_polygon.intersection(restrict_polygon)
    valid_placements = find_polygon_placement(object2_polygon, object1_polygon, max_attempts=10000)

    if not valid_placements:
        print("No valid placements found.")
        return 0
    else:
        rel_translation, _ = valid_placements[-1]
        translation, _ = object1.get_local_pose()

        translation[:2] += rel_translation

        bbox_obj = compute_bbox(object1.prim)
        obj_z_min = bbox_obj.min[2]
        bbox_tgt = compute_bbox(object2.prim)
        tgt_z_max = bbox_tgt.max[2]

        translation[2] = tgt_z_max + (translation[2] - obj_z_min) + 0.001  # add a small value to avoid penetration
        object1.set_local_pose(translation=translation)

        return 1


def set_distractors(
    objects,  #
    distractors,  # that need to be placed
    target,  # target object to place distractors on, default table
    distractor_cfg,
    cfgs,
):
    # Get meshes
    objects_meshes = [mesh_from_usd_data(recursive_parse_new(prim.prim)) for prim in objects.values()]
    distractor_items = list(distractors.items())
    distractor_meshes = [
        mesh_from_usd_data(recursive_parse_new(prim.prim)) for _, prim in distractor_items
    ]
    target_mesh = mesh_from_usd_data(recursive_parse_new(target.prim))

    # Get pcs
    num_points = 10000
    objects_pcds = [get_pcd_from_mesh(mesh, num_points) for mesh in objects_meshes]
    objects_pcds = [pcd for pcd in objects_pcds if pcd is not None]
    target_pcd = get_pcd_from_mesh(target_mesh, num_points)
    if target_pcd is None:
        raise RuntimeError("Target object has no valid triangle mesh for distractor placement")

    valid_distractors = []
    distractors_pcds = []
    fallback_z = distractor_cfg.get("fallback_z", -5.0)
    for (name, distractor), mesh, cfg_item in zip(distractor_items, distractor_meshes, cfgs):
        pcd = get_pcd_from_mesh(mesh, num_points)
        if pcd is None:
            trans, ori = distractor.get_local_pose()
            trans = np.asarray(trans, dtype=float)
            trans[2] = fallback_z
            distractor.set_local_pose(translation=trans, orientation=ori)
            print(f"[distractor] skipping {name}: USD subtree has no triangles")
            continue
        valid_distractors.append((name, distractor, cfg_item))
        distractors_pcds.append(pcd)
    new_distractors_pcds = []

    # Control minimum distance between distractors and main objects in XY plane (meters)
    # min_object_distance: minimum distance between objects and distractors
    # distractor_buffer: minimum distance between already placed distractors (can be 0 if not needed)
    min_object_distance = distractor_cfg.get("min_object_distance", 0.03)
    distractor_buffer = distractor_cfg.get("distractor_buffer", 0.03)
    max_attempts = distractor_cfg.get("max_attempts", 10)
    fallback_z = distractor_cfg.get("fallback_z", -5.0)  # If no valid placement, move distractor out of view

    for idx, (_, distractor, distractor_cfg_item) in enumerate(valid_distractors):
        placed = False
        for _ in range(max_attempts):
            rotate_object(distractor, distractor_cfg_item.category)
            tmp_distractor_pcd = deepcopy(distractors_pcds[idx])
            tmp_distractor_pcd = transform_pointcloud(tmp_distractor_pcd, *distractor.get_local_pose())

            # First compute available region based on already placed distractors
            available_polygon = get_platform_available_polygon(
                target_pcd,
                new_distractors_pcds,
                visualize=False,
                buffer_size=distractor_buffer,
            )

            # Then apply buffer around main objects to ensure at least min_object_distance from distractors
            if min_object_distance > 0.0:
                for obj_pcd in objects_pcds:
                    obj_polygon = get_xy_contour(obj_pcd, contour_type="concave_hull").buffer(min_object_distance)
                    available_polygon = available_polygon.difference(obj_polygon)

            pos_range = distractor_cfg.get("pos_range", None)
            if pos_range is not None:
                x_min, y_min = pos_range[0]
                x_max, y_max = pos_range[1]
                center_x, center_y = target.get_local_pose()[0][:2]
                x_min += center_x
                x_max += center_x
                y_min += center_y
                y_max += center_y
                restrict_polygon = bbox_to_polygon(x_min, y_min, x_max, y_max)
            else:
                restrict_polygon = None

            res = randomly_place_object_on_object(
                tmp_distractor_pcd,
                target_pcd,
                distractor,
                target,
                available_polygon,
                restrict_polygon,
            )

            if res == 1:
                tmp_distractor_pcd = deepcopy(distractors_pcds[idx])
                tmp_distractor_pcd = transform_pointcloud(tmp_distractor_pcd, *distractor.get_local_pose())
                new_distractors_pcds.append(tmp_distractor_pcd)
                placed = True
                break

        # If no valid placement is found within max_attempts,
        # move this distractor to a safe location (e.g. [0, 0, fallback_z])
        # and do not add it to new_distractors_pcds so it does not affect other placements.
        if not placed:
            trans, ori = distractor.get_local_pose()
            trans = np.array(trans)
            trans[0] = 0.0
            trans[1] = 0.0
            trans[2] = fallback_z
            distractor.set_local_pose(translation=trans, orientation=ori)
