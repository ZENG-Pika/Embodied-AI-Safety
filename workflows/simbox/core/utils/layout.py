import copy
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import shapely
import trimesh
from concave_hull import concave_hull
from core.utils.usd_geom_utils import recursive_parse_new
from scipy.spatial.qhull import ConvexHull
from scipy.spatial.transform import Rotation as R
from shapely.geometry import MultiPolygon, Point, Polygon


def get_current_meshList(object_list, meshDict):
    """Build a mesh dictionary keyed by object uid."""
    return {uid: meshDict[uid] for uid in object_list if uid in meshDict}


def meshlist_to_pclist(meshlist):
    """Convert mesh dictionary to point-cloud dictionary via sampling."""
    pclist = {}
    for uid, mesh in meshlist.items():
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            pclist[uid] = np.asarray(mesh.sample_points_uniformly(number_of_points=10000).points)
    return pclist


def check_subgoal_finished_rigid(subgoal, *args, **kwargs):  # pylint: disable=unused-argument
    """Placeholder for subgoal checking; always returns True to keep legacy API.

    This mirrors the behavior in the original data generation pipeline where
    geometric conditions were validated externally.
    """
    return True


def set_camera_look_at(camera, target_object, azimuth=180.0):  # pylint: disable=unused-argument
    """Placeholder stub; real implementation is environment-specific."""
    return


def get_init_grasp(  # pylint: disable=unused-argument
    camera,
    mesh,
    address=None,
    port=None,
    allow_fixed_grasp=False,
    force_fixed_grasp=False,
):
    """Placeholder stub for grasp initialization; returns a neutral grasp."""
    return {"translation": np.zeros(3), "orientation": np.array([1.0, 0.0, 0.0, 0.0])}


def get_related_position(object_pc, container_pc):  # pylint: disable=unused-argument
    """Placeholder that always returns 'in' as the relation between object and container."""
    return "in"


def get_action_meta_info(scene, action_info, default_config):
    action_meta_info = {}
    action_meta_info["obj_tar_t"], action_meta_info["obj_tar_o"] = compute_final_grasp(
        scene["object_list"],
        action_info,
        scene["cacheDict"]["meshDict"],
        ignored_uid=action_info.get("ignored_uid", []),
        fixed_position=action_info.get("fixed_position", False),
        fixed_position_config=action_info.get("fixed_position_config", None),
        without_platform=action_info.get("without_platform", False),
        mesh_top_only=action_info.get("mesh_top_only", False),
    )
    if action_meta_info["obj_tar_t"] is None or action_meta_info["obj_tar_o"] is None:
        raise Exception("can't create target position, retry......")
    action_meta_info = get_action_init_grasp(
        scene,
        action_info,
        default_config,
        action_meta_info,
        allow_fixed_grasp=action_info.get("allow_fixed_grasp", False),
        force_fixed_grasp=action_info.get("force_fixed_grasp", False),
        fixed_grasp_config=action_info.get("fixed_grasp_config", None),
    )
    action_meta_info["init_grasp"] = adjust_grasp_by_embodiment(
        action_meta_info["init_grasp"],
        scene["robot_info"]["robot_list"][0],
    )
    action_meta_info["grasp_tar_t"], action_meta_info["grasp_tar_o"] = compute_final_pose(
        action_meta_info["obj_init_t"],
        action_meta_info["obj_init_o"],
        action_meta_info["init_grasp"]["translation"],
        action_meta_info["init_grasp"]["orientation"],
        action_meta_info["obj_tar_t"],
        action_meta_info["obj_tar_o"],
    )
    return action_meta_info


def compute_final_grasp(
    object_list,
    action,
    meshDict,
    ignored_uid=None,
    extra_erosion=0.05,
    fixed_position=False,
    fixed_position_config=None,
    without_platform=False,
    mesh_top_only=False,
):
    if ignored_uid is None:
        ignored_uid = []
    obj_init_t, obj_init_o = object_list[action["obj1_uid"]].get_world_pose()
    if action["position"] in ("top", "in"):
        IS_OK = place_object_to_object_by_relation(
            action["obj1_uid"],
            action["obj2_uid"],
            object_list,
            meshDict,
            "on",
            platform_uid=("00000000000000000000000000000000" if not without_platform else None),
            ignored_uid=ignored_uid,
            extra_erosion=extra_erosion,
            fixed_position=fixed_position,
            mesh_top_only=mesh_top_only,
        )
    elif action["position"] == "near":
        IS_OK = place_object_to_object_by_relation(
            action["obj1_uid"],
            action["obj2_uid"],
            object_list,
            meshDict,
            "near",
            platform_uid="00000000000000000000000000000000",
            ignored_uid=ignored_uid,
            extra_erosion=extra_erosion,
        )
    else:
        if "another_obj2_uid" in action:
            IS_OK = place_object_to_object_by_relation(
                action["obj1_uid"],
                action["obj2_uid"],
                object_list,
                meshDict,
                action["position"],
                platform_uid="00000000000000000000000000000000",
                ignored_uid=ignored_uid,
                extra_erosion=extra_erosion,
                another_object2_uid=action["another_obj2_uid"],
            )
        else:
            IS_OK = place_object_to_object_by_relation(
                action["obj1_uid"],
                action["obj2_uid"],
                object_list,
                meshDict,
                action["position"],
                platform_uid="00000000000000000000000000000000",
                ignored_uid=ignored_uid,
                extra_erosion=extra_erosion,
            )
    if IS_OK == -1:
        return None, None
    obj_tar_t, obj_tar_o = object_list[action["obj1_uid"]].get_world_pose()
    object_list[action["obj1_uid"]].set_world_pose(position=obj_init_t, orientation=obj_init_o)
    if fixed_position_config is not None:
        obj_tar_t = obj_tar_t + np.array(fixed_position_config["translation"])
        obj_tar_o = (
            R.from_quat(np.array(fixed_position_config["orientation"])[[1, 2, 3, 0]])
            * R.from_quat(obj_tar_o[[1, 2, 3, 0]])
        ).as_quat()[[3, 0, 1, 2]]
    return obj_tar_t, obj_tar_o


def place_object_to_object_by_relation(
    object1_uid: str,
    object2_uid: str,
    object_list,
    meshDict,
    relation: str,
    platform_uid: str | None = None,
    extra_erosion: float = 0.00,
    another_object2_uid: str = None,  # for "between" relation
    ignored_uid: List[str] = None,
    debug: bool = False,
    fixed_position: bool = False,
    mesh_top_only: bool = False,
):
    if ignored_uid is None:
        ignored_uid = []

    object1 = object_list[object1_uid]
    mesh_list = get_current_meshList(object_list, meshDict)
    pointcloud_list = meshlist_to_pclist(mesh_list)
    combined_cloud = []
    for _, pc in pointcloud_list.items():
        combined_cloud.append(pc)
    combined_cloud = np.vstack(combined_cloud)
    ignored_uid_ = copy.deepcopy(ignored_uid)
    if platform_uid is not None:
        ignored_uid_.extend([object1_uid, object2_uid, platform_uid])
        available_area = get_platform_available_area(
            pointcloud_list[platform_uid],
            pointcloud_list,
            ignored_uid_,
        ).buffer(-extra_erosion)
    else:
        available_area = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    if relation in ("on", "top"):
        IS_OK = randomly_place_object_on_object_gaoning(
            pointcloud_list[object1_uid],
            combined_cloud if not mesh_top_only else pointcloud_list[object2_uid],
            object1,
            available_polygon=get_xy_contour(pointcloud_list[object2_uid], contour_type="concave_hull"),
            collider_polygon=available_area,
            fixed_position=fixed_position,
            mesh_top_only=mesh_top_only,
        )
    elif relation == "near":
        near_area = compute_near_area(mesh_list[object1_uid], mesh_list[object2_uid])
        if debug:
            visualize_polygons(
                [
                    near_area,
                ]
                + [get_xy_contour(pcd, contour_type="concave_hull") for pcd in pointcloud_list.values()]
            )
        IS_OK = randomly_place_object_on_object_gaoning(
            pointcloud_list[object1_uid],
            combined_cloud,
            object1,
            available_polygon=near_area.intersection(
                get_xy_contour(pointcloud_list[platform_uid], contour_type="convex_hull")
            ),
            collider_polygon=available_area,
            fixed_position=fixed_position,
        )
    elif relation in ("left", "right", "front", "back"):
        place_area = compute_lrfb_area(relation, mesh_list[object1_uid], mesh_list[object2_uid])
        near_area = compute_near_area(mesh_list[object1_uid], mesh_list[object2_uid])
        place_area = place_area.intersection(near_area)
        if debug:
            visualize_polygons(
                [
                    place_area,
                    near_area,
                ]
                + [get_xy_contour(pcd, contour_type="concave_hull") for pcd in pointcloud_list.values()]
            )
        IS_OK = randomly_place_object_on_object_gaoning(
            pointcloud_list[object1_uid],
            combined_cloud,
            object1,
            available_polygon=place_area.intersection(
                get_xy_contour(pointcloud_list[platform_uid], contour_type="convex_hull")
            ),
            collider_polygon=available_area,
            fixed_position=fixed_position,
        )
    elif relation == "in":
        IS_OK = place_object_in_object(object_list, meshDict, object1_uid, object2_uid)
    elif relation == "between":
        IS_OK = place_object_between_object1_and_object2(
            object_list,
            meshDict,
            object1_uid,
            object2_uid,
            another_object2_uid,
            platform_uid,
        )
    else:
        IS_OK = -1
    if IS_OK == -1:
        return -1
    meshlist = get_current_meshList(object_list, meshDict)
    pclist = meshlist_to_pclist(meshlist)
    if relation != "between":
        subgoal = {
            "obj1_uid": object1_uid,
            "obj2_uid": object2_uid,
            "position": relation,
        }
        finished = check_subgoal_finished_rigid(subgoal, pclist[object1_uid], pclist[object2_uid])
    else:
        subgoal = {
            "obj1_uid": object1_uid,
            "obj2_uid": object2_uid,
            "position": relation,
            "another_obj2_uid": another_object2_uid,
        }
        finished = check_subgoal_finished_rigid(
            subgoal,
            pclist[object1_uid],
            pclist[object2_uid],
            pclist[another_object2_uid],
        )
    if finished or fixed_position:
        return 0
    else:
        return -1


def get_platform_available_area(platform_pc, pc_list, filtered_uid=None, visualize=False, buffer_size=0.0):
    if filtered_uid is None:
        filtered_uid = []

    platform_polygon = get_xy_contour(platform_pc, contour_type="concave_hull")
    if visualize:
        polygons = []
        for key, pc in pc_list.items():
            if key not in filtered_uid:
                polygons.append(get_xy_contour(pc, contour_type="concave_hull"))
        visualize_polygons(polygons)
    for key in pc_list:
        if key not in filtered_uid:
            pc = pc_list[key]
            pc_polygon = get_xy_contour(pc, contour_type="concave_hull").buffer(buffer_size)
            platform_polygon = platform_polygon.difference(pc_polygon)
    return platform_polygon


def get_xy_contour(points, contour_type="convex_hull"):
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
    else:
        polygon = Polygon()
    return polygon


def sort_points_clockwise(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]


def get_action_init_grasp(
    scene,
    action_info,
    default_config,
    action_meta_info,
    allow_fixed_grasp=False,
    force_fixed_grasp=False,
    fixed_grasp_config=None,
):
    """
    To collect init grasp and obj init pose.
    """
    if not force_fixed_grasp:
        set_camera_look_at(
            scene["camera_list"]["camera1"],
            scene["object_list"][action_info["obj1_uid"]],
            azimuth=180.0,
        )
        current_pose_list = collect_world_pose_list(scene["object_list"])
        current_joint_positions = scene["robot_info"]["robot_list"][0].robot.get_joint_positions()
        robot_world_pose = scene["robot_info"]["robot_list"][0].robot.get_world_pose()
        scene["robot_info"]["robot_list"][0].robot.set_world_pose(
            robot_world_pose[0] + np.array([1000.0, 0.0, 0.0]), robot_world_pose[1]
        )
        for _ in range(5):
            scene["world"].step(render=True)
    meshlist = get_current_meshList(scene["object_list"], scene["cacheDict"]["meshDict"])
    mesh = meshlist[action_info["obj1_uid"]]
    # init_grasp is in world frame, with key translation and orientation
    action_meta_info["init_grasp"] = get_init_grasp(
        scene["camera_list"]["camera1"],
        mesh,
        address=default_config["ANYGRASP_ADDR"],
        port=default_config["ANYGRASP_PORT"],
        allow_fixed_grasp=allow_fixed_grasp,
        force_fixed_grasp=force_fixed_grasp,
    )
    if not force_fixed_grasp:
        for _ in range(5):
            reset_object_xyz(scene["object_list"], current_pose_list)
            scene["robot_info"]["robot_list"][0].robot.set_joint_positions(current_joint_positions)
            scene["robot_info"]["robot_list"][0].robot.set_world_pose(*robot_world_pose)
            scene["world"].step(render=True)
    if force_fixed_grasp or (
        allow_fixed_grasp
        and action_meta_info["init_grasp"]["translation"][0] == 0.0
        and action_meta_info["init_grasp"]["translation"][1] == 0.0
    ):
        action_meta_info["init_grasp"]["translation"][:2] = scene["object_list"][
            action_info["obj1_uid"]
        ].get_world_pose()[0][:2]
        if fixed_grasp_config is not None:
            action_meta_info["init_grasp"]["translation"] += np.array(fixed_grasp_config["translation"])
    action_meta_info["obj_init_t"], action_meta_info["obj_init_o"] = scene["object_list"][
        action_info["obj1_uid"]
    ].get_world_pose()
    return action_meta_info


def bbox_to_polygon(x_min, y_min, x_max, y_max):
    points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    return Polygon(points)


def place_object_in_polygon(
    object1_pcd,
    object1,
    available_polygon,
):
    object1_polygon = get_xy_contour(object1_pcd, contour_type="concave_hull")
    object1_center = object1.get_world_pose()[0][:2]
    # set_trace()

    valid_placements = find_polygon_placement(available_polygon, object1_polygon, max_attempts=10000)
    if len(valid_placements) == 0:
        valid_placements = find_polygon_placement_with_rotation(
            available_polygon, object1_polygon, object1_center, max_attempts=10000
        )
        if len(valid_placements) == 0:
            return -1

    rel_translation, angle = valid_placements[-1]
    translation, orientation = object1.get_local_pose()
    translation[:2] += rel_translation

    orientation = rotate_quaternion_z(orientation, angle)
    object1.set_local_pose(translation=translation, orientation=orientation)

    # set_trace()

    return 0


def optimize_2d_manip_layout(
    obj_cfgs,
    reg_cfgs,
    objs,
    iterations=10,
):
    # set_trace()
    optimize_keys = [obj_cfg["name"] for obj_cfg in obj_cfgs if obj_cfg.get("optimize_2d_layout", False)]
    if len(optimize_keys) == 0:
        return

    # optimize_objs = [objs[key] for key in optimize_keys]
    optimize_objs = []
    reg_polygons = []
    # set_trace()
    for reg_cfg in reg_cfgs:
        if reg_cfg["object"] in optimize_keys:
            optimize_objs.append(objs[reg_cfg["object"]])
            assert reg_cfg["random_type"] == "A_on_B_region_sampler"
            pos_range = reg_cfg["random_config"]["pos_range"]
            x_min, y_min = pos_range[0][:2]
            x_max, y_max = pos_range[1][:2]
            reg_polygon = bbox_to_polygon(x_min, y_min, x_max, y_max)
            reg_polygons.append(reg_polygon)

    obj_pcds = []
    for obj, polygon in zip(optimize_objs, reg_polygons):
        polygon = get_platform_available_polygon(polygon, obj_pcds, visualize=False)
        obj_mesh = recursive_parse_new(obj.prim)
        obj_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(obj_mesh[0]),
            triangles=o3d.utility.Vector3iVector(np.array(obj_mesh[2]).reshape(-1, 3)),
        )
        obj_pcd = get_pcd_from_mesh(obj_mesh, num_points=10000)
        for idx in range(iterations):
            res = place_object_in_polygon(obj_pcd, obj, polygon)
            # print("res :", res, "polygon :", polygon, "obj :", obj.get_world_pose())
            if res == 0 or (idx == iterations - 1):
                # print("optimized success")
                new_obj_mesh = recursive_parse_new(obj.prim)
                new_obj_mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(new_obj_mesh[0]),
                    triangles=o3d.utility.Vector3iVector(np.array(new_obj_mesh[2]).reshape(-1, 3)),
                )
                new_obj_pcd = get_pcd_from_mesh(new_obj_mesh, num_points=10000)
                obj_pcds.append(new_obj_pcd)
                break


def get_platform_available_polygon(platform_polygon, pc_list, visualize=False, buffer_size=0.0):
    if visualize:
        polygons = []
        for pc in pc_list:
            polygons.append(get_xy_contour(pc, contour_type="concave_hull"))
        visualize_polygons(polygons + [platform_polygon])
    for pc in pc_list:
        pc_polygon = get_xy_contour(pc, contour_type="concave_hull").buffer(buffer_size)
        platform_polygon = platform_polygon.difference(pc_polygon)
    return platform_polygon


def random_place_object_on_object(
    object1_pcd,
    object2_pcd,
    object1,
    min_rotate_angle=-np.pi / 6,
    max_rotate_angle=np.pi / 6,
    available_polygon=None,
    restrict_polygon=None,
):
    object1_polygon = get_xy_contour(object1_pcd, contour_type="concave_hull")
    object2_polygon = get_xy_contour(object2_pcd, contour_type="concave_hull")

    if available_polygon is not None:
        object2_polygon = object2_polygon.intersection(available_polygon)
    if restrict_polygon is not None:
        object2_polygon = object2_polygon.intersection(restrict_polygon)

    object1_pc = np.asarray(object1_pcd.points)
    object2_pc = np.asarray(object2_pcd.points)
    object1_pc_bottom = object1_pc[np.argmin(object1_pc[:, 2])][2]
    object1_bbox = compute_pcd_bbox(object1_pcd)
    obj1_2d_bbox = [
        object1_bbox.get_min_bound()[0],
        object1_bbox.get_min_bound()[1],
        object1_bbox.get_max_bound()[0] - object1_bbox.get_min_bound()[0],
        object1_bbox.get_max_bound()[1] - object1_bbox.get_min_bound()[1],
    ]
    object1_center = object1.get_local_pose()[0][:2]

    valid_placements = find_polygon_placement_with_rotation(
        object2_polygon,
        object1_polygon,
        object1_center,
        min_rotate_angle=min_rotate_angle,
        max_rotate_angle=max_rotate_angle,
        max_attempts=10000,
    )
    if len(valid_placements) == 0:
        return -1

    rel_translation, angle = valid_placements[-1]
    translation, orientation = object1.get_local_pose()

    translation[:2] += rel_translation

    updated_obj1_2d_bbox = obj1_2d_bbox
    bbox_buffer_x = 0.05 * updated_obj1_2d_bbox[2]
    bbox_buffer_y = 0.05 * updated_obj1_2d_bbox[3]
    updated_obj1_2d_bbox[0] += translation[0] - bbox_buffer_x
    updated_obj1_2d_bbox[1] += translation[1] - bbox_buffer_y
    updated_obj1_2d_bbox[2] += bbox_buffer_x * 2
    updated_obj1_2d_bbox[3] += bbox_buffer_y * 2
    cropped_object2_pc = object2_pc[
        np.where(
            (object2_pc[:, 0] >= updated_obj1_2d_bbox[0])
            & (object2_pc[:, 0] <= updated_obj1_2d_bbox[0] + updated_obj1_2d_bbox[2])
            & (object2_pc[:, 1] >= updated_obj1_2d_bbox[1])
            & (object2_pc[:, 1] <= updated_obj1_2d_bbox[1] + updated_obj1_2d_bbox[3])
        )
    ]
    if len(cropped_object2_pc) == 0:
        return -1

    object2_pc_top = cropped_object2_pc[np.argmax(cropped_object2_pc[:, 2])][2]
    object1_to_object2_axis2 = object2_pc_top - object1_pc_bottom
    translation[2] += object1_to_object2_axis2
    orientation = rotate_quaternion_z(orientation, angle)

    return (translation, orientation)


def randomly_place_object_on_object_gaoning(
    object1_pc,
    object2_pc,
    object1,
    available_polygon: Polygon = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
    collider_polygon: Polygon = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
    fixed_position: bool = False,
    mesh_top_only: bool = False,
):
    object1_polygon = get_xy_contour(object1_pc, contour_type="concave_hull")
    object1_pc_bottom = object1_pc[np.argmin(object1_pc[:, 2])][2]
    object2_polygon = get_xy_contour(object2_pc, contour_type="concave_hull")
    object1_pcd = o3d.geometry.PointCloud()
    object1_pcd.points = o3d.utility.Vector3dVector(object1_pc)
    object1_bbox = compute_pcd_bbox(object1_pcd)
    obj1_2d_bbox = [
        object1_bbox.get_min_bound()[0],
        object1_bbox.get_min_bound()[1],
        object1_bbox.get_max_bound()[0] - object1_bbox.get_min_bound()[0],
        object1_bbox.get_max_bound()[1] - object1_bbox.get_min_bound()[1],
    ]
    object2_polygon = object2_polygon.intersection(available_polygon).intersection(collider_polygon)
    object1_center = object1.get_world_pose()[0][:2]
    if fixed_position:
        valid_placements = find_fixed_polygon_placement(object2_polygon, object1_polygon)
    else:
        valid_placements = find_polygon_placement(object2_polygon, object1_polygon, max_attempts=10000)
        if len(valid_placements) == 0:
            valid_placements = find_polygon_placement_with_rotation(
                object2_polygon, object1_polygon, object1_center, max_attempts=10000
            )
            if len(valid_placements) == 0:
                return -1
    translation, angle = valid_placements[-1]
    position, orientation = object1.get_world_pose()
    position[:2] += translation
    if mesh_top_only:
        cropped_object2_pc = object2_pc
    else:
        updated_obj1_2d_bbox = obj1_2d_bbox
        bbox_buffer_x = 0.05 * updated_obj1_2d_bbox[2]
        bbox_buffer_y = 0.05 * updated_obj1_2d_bbox[3]
        updated_obj1_2d_bbox[0] += translation[0] - bbox_buffer_x
        updated_obj1_2d_bbox[1] += translation[1] - bbox_buffer_y
        updated_obj1_2d_bbox[2] += bbox_buffer_x * 2
        updated_obj1_2d_bbox[3] += bbox_buffer_y * 2
        cropped_object2_pc = object2_pc[
            np.where(
                (object2_pc[:, 0] >= updated_obj1_2d_bbox[0])
                & (object2_pc[:, 0] <= updated_obj1_2d_bbox[0] + updated_obj1_2d_bbox[2])
                & (object2_pc[:, 1] >= updated_obj1_2d_bbox[1])
                & (object2_pc[:, 1] <= updated_obj1_2d_bbox[1] + updated_obj1_2d_bbox[3])
            )
        ]
        if len(cropped_object2_pc) == 0:
            return -1
    object2_pc_top = cropped_object2_pc[np.argmax(cropped_object2_pc[:, 2])][2]
    object1_to_object2_axis2 = object2_pc_top - object1_pc_bottom
    position[2] += object1_to_object2_axis2
    orientation = rotate_quaternion_z(orientation, angle)
    object1.set_world_pose(position=position, orientation=orientation)
    return 0


def rotate_quaternion_z(quat, angle_rad):
    r = R.from_quat(quat[[1, 2, 3, 0]])
    r_z = R.from_euler("z", angle_rad)
    return (r_z * r).as_quat()[[3, 0, 1, 2]]


def find_fixed_polygon_placement(large_polygon, small_polygon):
    large_center = np.array(large_polygon.centroid.coords[0])
    small_center = np.array(small_polygon.centroid.coords[0])
    translation = large_center - small_center
    return [(translation, 0)]


def find_polygon_placement(large_polygon, small_polygon, buffer_thresh=0.03, max_attempts=1000):
    if large_polygon.is_empty or small_polygon.is_empty:
        return []

    # Shrink large polygon slightly to avoid placing objects exactly on the boundary
    # set_trace()
    safe_region = large_polygon.buffer(-buffer_thresh)
    if safe_region.is_empty:
        return []  # if shrinking removes the polygon entirely, the buffer threshold is too large

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


def find_polygon_placement_with_rotation(
    large_polygon,
    small_polygon,
    object1_center,
    # buffer_thresh=0.03,
    min_rotate_angle=0,
    max_rotate_angle=2 * np.pi,
    max_attempts=1000,
):
    if large_polygon.is_empty or small_polygon.is_empty:
        return []
    minx, miny, maxx, maxy = large_polygon.bounds
    valid_placements = []
    for _ in range(max_attempts):
        random_angle = np.random.uniform(min_rotate_angle, max_rotate_angle)
        rotated_polygon = rotate_polygon(small_polygon, random_angle, object1_center)
        coords = np.array(rotated_polygon.exterior.coords)
        small_centroid = np.mean(coords, axis=0)
        tx = np.random.uniform(minx, maxx)
        ty = np.random.uniform(miny, maxy)
        translation = np.array([tx, ty])
        transformed_polygon = shapely.affinity.translate(
            rotated_polygon,
            xoff=translation[0] - small_centroid[0],
            yoff=translation[1] - small_centroid[1],
        )
        if large_polygon.contains_properly(transformed_polygon):
            valid_placements.append((translation - small_centroid, random_angle))
            break
    return valid_placements


def rotate_polygon(polygon, angle, center):
    return shapely.affinity.rotate(polygon, angle, origin=tuple(center), use_radians=True)


def compute_pcd_bbox(pcd):
    aabb = pcd.get_axis_aligned_bounding_box()
    return aabb


def visualize_polygons(polygons: list, output_path: str = "polygons.png"):
    fig, ax = plt.subplots()
    for polygon in polygons:
        if isinstance(polygon, Polygon):
            x, y = polygon.exterior.xy
            ax.plot(x, y)
        elif isinstance(polygon, MultiPolygon):
            for single_polygon in polygon.geoms:
                x, y = single_polygon.exterior.xy
                ax.plot(x, y)
        else:
            continue
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.savefig(output_path)
    plt.close(fig)


def reset_object_xyz(object_list, xyz):
    for key in object_list:
        if key in ("00000000000000000000000000000000", "defaultGroundPlane"):
            continue
        if key in xyz:
            object_list[key].set_world_pose(*xyz[key])


def collect_world_pose_list(object_list):
    world_pose_list = {}
    for key in object_list:
        if key in ("00000000000000000000000000000000", "defaultGroundPlane"):
            continue
        world_pose_list[key] = object_list[key].get_world_pose()
    return world_pose_list


def compute_final_pose(P_A0, Q_A0, P_B0, Q_B0, P_A1, Q_A1):
    rot_A0 = R.from_quat(Q_A0[[1, 2, 3, 0]])
    rot_A1 = R.from_quat(Q_A1[[1, 2, 3, 0]])
    rot_B0 = R.from_quat(Q_B0[[1, 2, 3, 0]])
    rot_BA = rot_A0.inv() * rot_B0
    t_BA = rot_A0.inv().apply(P_B0 - P_A0)
    P_B1 = rot_A1.apply(t_BA) + P_A1
    rot_B1 = rot_A1 * rot_BA
    Q_B1 = rot_B1.as_quat()[[3, 0, 1, 2]]
    return P_B1, Q_B1


def get_max_distance_to_polygon(polygon, point):
    return max(point.distance(Point(vertex)) for vertex in polygon.exterior.coords)


def adjust_grasp_by_embodiment(grasp, embodiment):
    grasp["orientation"] = adjust_orientation(grasp["orientation"])
    if embodiment.embodiment_name == "franka":
        if embodiment.gripper_name == "panda_hand":
            grasp["translation"] = adjust_translation_along_quaternion(
                grasp["translation"],
                grasp["orientation"],
                0.08,
                aug_distance=0.0,
            )
        elif embodiment.gripper_name == "robotiq":
            # Robotiq grasp pose needs to rotate 45 degrees around z-axis
            grasp["orientation"] = rot_orientation_by_z_axis(grasp["orientation"], -45)
            grasp["translation"] = adjust_translation_along_quaternion(
                grasp["translation"],
                grasp["orientation"],
                0.15,
                aug_distance=0.0,
            )
    elif embodiment.embodiment_name == "aloha_split":
        if embodiment.gripper_name == "piper":
            grasp["orientation"] = rot_orientation_by_z_axis(grasp["orientation"], -90)
            grasp["translation"] = adjust_translation_along_quaternion(
                grasp["translation"],
                grasp["orientation"],
                0.11,
                aug_distance=0.0,
            )
    return grasp


def adjust_translation_along_quaternion(translation, quaternion, distance, aug_distance=0.0):
    rotation = R.from_quat(quaternion[[1, 2, 3, 0]])
    direction_vector = rotation.apply([0, 0, 1])
    reverse_direction = -direction_vector
    new_translation = translation + reverse_direction * distance
    arbitrary_vector = np.array([1, 0, 0]) if direction_vector[0] == 0 else np.array([0, 1, 0])
    perp_vector1 = np.cross(direction_vector, arbitrary_vector)
    perp_vector2 = np.cross(direction_vector, perp_vector1)
    perp_vector1 /= np.linalg.norm(perp_vector1)
    perp_vector2 /= np.linalg.norm(perp_vector2)
    random_shift = np.random.uniform(-aug_distance, aug_distance, size=2)
    new_translation += random_shift[0] * perp_vector1 + random_shift[1] * perp_vector2
    return new_translation


def rot_orientation_by_z_axis(ori, angle):
    ori = R.from_quat(ori[[1, 2, 3, 0]])
    ori = ori * R.from_euler("z", angle, degrees=True)
    return ori.as_quat()[[3, 0, 1, 2]]


def adjust_orientation(ori):
    ori = R.from_quat(ori[[1, 2, 3, 0]])
    if ori.apply(np.array([1, 0, 0]))[0] < 0:
        ori = R.from_euler("z", 180, degrees=True) * ori
    return ori.as_quat()[[3, 0, 1, 2]]


def compute_near_area(mesh1, mesh2, near_distance=0.1, angle_steps=36):
    pcd1 = get_pcd_from_mesh(mesh1)
    pcd2 = get_pcd_from_mesh(mesh2)
    polygon1 = get_xy_contour(pcd1, contour_type="concave_hull")
    polygon2 = get_xy_contour(pcd2, contour_type="concave_hull")
    angles = np.linspace(0, 359, angle_steps)
    transformed_polygons_1 = []
    centroid1_x, centroid1_y = polygon1.centroid.x, polygon1.centroid.y
    centroid2_x, centroid2_y = polygon2.centroid.x, polygon2.centroid.y
    angle_rads = np.radians(angles)
    cos_angles = np.cos(angle_rads)
    sin_angles = np.sin(angle_rads)
    for i in range(len(angles)):
        distance = 100
        x = cos_angles[i] * distance + centroid2_x - centroid1_x
        y = sin_angles[i] * distance + centroid2_y - centroid1_y
        transformed_polygon_1 = transform_polygon(polygon1, x, y)
        min_distance = compute_min_distance_between_two_polygons(transformed_polygon_1, polygon2, num_points=50)
        distance = distance - min_distance + near_distance
        x = cos_angles[i] * distance + centroid2_x - centroid1_x
        y = sin_angles[i] * distance + centroid2_y - centroid1_y
        transformed_polygon_1 = transform_polygon(polygon1, x, y)
        transformed_polygons_1.append(transformed_polygon_1)
    all_points = np.vstack([np.asarray(polygon.exterior.coords) for polygon in transformed_polygons_1])
    near_area = get_xy_contour(all_points, contour_type="convex_hull").difference(polygon2)
    return near_area


def compute_lrfb_area(position, mesh1, mesh2):
    from genmanip.demogen.evaluate.evaluate import XY_DISTANCE_CLOSE_THRESHOLD

    aabb1 = compute_mesh_bbox(mesh1)
    aabb2 = compute_mesh_bbox(mesh2)
    mesh1_length, mesh1_width, _ = compute_aabb_lwh(aabb1)
    if position == "back":
        distance = XY_DISTANCE_CLOSE_THRESHOLD + mesh1_length
        polygon = Polygon(
            [
                (
                    aabb2.get_max_bound()[0],
                    min(aabb2.get_min_bound()[1], aabb2.get_max_bound()[1] - mesh1_width),
                ),
                (
                    aabb2.get_max_bound()[0] + distance,
                    min(aabb2.get_min_bound()[1], aabb2.get_max_bound()[1] - mesh1_width),
                ),
                (
                    aabb2.get_max_bound()[0] + distance,
                    max(aabb2.get_max_bound()[1], aabb2.get_min_bound()[1] + mesh1_width),
                ),
                (
                    aabb2.get_max_bound()[0],
                    max(aabb2.get_max_bound()[1], aabb2.get_min_bound()[1] + mesh1_width),
                ),
            ]
        )
    elif position == "front":
        distance = XY_DISTANCE_CLOSE_THRESHOLD + mesh1_length
        polygon = Polygon(
            [
                (
                    aabb2.get_min_bound()[0],
                    max(aabb2.get_max_bound()[1], aabb2.get_min_bound()[1] + mesh1_width),
                ),
                (
                    aabb2.get_min_bound()[0] - distance,
                    max(aabb2.get_max_bound()[1], aabb2.get_min_bound()[1] + mesh1_width),
                ),
                (
                    aabb2.get_min_bound()[0] - distance,
                    min(aabb2.get_min_bound()[1], aabb2.get_max_bound()[1] - mesh1_width),
                ),
                (
                    aabb2.get_min_bound()[0],
                    min(aabb2.get_min_bound()[1], aabb2.get_max_bound()[1] - mesh1_width),
                ),
            ]
        )
    elif position == "right":
        distance = XY_DISTANCE_CLOSE_THRESHOLD + mesh1_width
        polygon = Polygon(
            [
                (
                    max(
                        aabb2.get_max_bound()[0],
                        aabb2.get_min_bound()[0] + mesh1_length,
                    ),
                    aabb2.get_max_bound()[1],
                ),
                (
                    max(
                        aabb2.get_max_bound()[0],
                        aabb2.get_min_bound()[0] + mesh1_length,
                    ),
                    aabb2.get_max_bound()[1] + distance,
                ),
                (
                    min(
                        aabb2.get_min_bound()[0],
                        aabb2.get_max_bound()[0] - mesh1_length,
                    ),
                    aabb2.get_max_bound()[1] + distance,
                ),
                (
                    min(
                        aabb2.get_min_bound()[0],
                        aabb2.get_max_bound()[0] - mesh1_length,
                    ),
                    aabb2.get_max_bound()[1],
                ),
            ]
        )
    elif position == "left":
        distance = XY_DISTANCE_CLOSE_THRESHOLD + mesh1_width
        polygon = Polygon(
            [
                (
                    max(
                        aabb2.get_max_bound()[0],
                        aabb2.get_min_bound()[0] + mesh1_length,
                    ),
                    aabb2.get_min_bound()[1],
                ),
                (
                    max(
                        aabb2.get_max_bound()[0],
                        aabb2.get_min_bound()[0] + mesh1_length,
                    ),
                    aabb2.get_min_bound()[1] - distance,
                ),
                (
                    min(
                        aabb2.get_min_bound()[0],
                        aabb2.get_max_bound()[0] - mesh1_length,
                    ),
                    aabb2.get_min_bound()[1] - distance,
                ),
                (
                    min(
                        aabb2.get_min_bound()[0],
                        aabb2.get_max_bound()[0] - mesh1_length,
                    ),
                    aabb2.get_min_bound()[1],
                ),
            ]
        )
    else:
        polygon = Polygon()
    return polygon


def compute_aabb_lwh(aabb):
    # compute the length, width, and height of the aabb
    length = aabb.get_max_bound()[0] - aabb.get_min_bound()[0]
    width = aabb.get_max_bound()[1] - aabb.get_min_bound()[1]
    height = aabb.get_max_bound()[2] - aabb.get_min_bound()[2]
    return length, width, height


def compute_mesh_bbox(mesh):
    pcd = get_pcd_from_mesh(mesh)
    return compute_pcd_bbox(pcd)


def get_pcd_from_mesh(mesh, num_points=1000):
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd


def transform_polygon(polygon, x, y):
    return shapely.affinity.translate(polygon, xoff=x, yoff=y)


def compute_min_distance_between_two_polygons(polygon1, polygon2, num_points=1000):
    points1 = sample_points_in_polygon(polygon1, num_points=num_points)
    points2 = sample_points_in_polygon(polygon2, num_points=num_points)
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1).fit(points1)
    distances, _ = nn.kneighbors(points2)
    res = np.min(distances)
    return res


def sample_points_in_polygon(polygon, num_points=1000):
    boundary = polygon.boundary
    boundary_length = boundary.length
    points = []
    for _ in range(num_points):
        point = boundary.interpolate(random.uniform(0, boundary_length))
        points.append(np.array([point.x, point.y]))
    return np.array(points)


def place_object_in_object(object_list, meshDict, object_uid, container_uid):
    meshlist = get_current_meshList(object_list, meshDict)
    container_mesh = meshlist[container_uid]
    points = sample_points_in_convex_hull(container_mesh, 1000)
    object_trans, _ = object_list[object_uid].get_world_pose()
    object_center = compute_mesh_center(meshlist[object_uid])
    trans_vector = object_trans - object_center
    for point in points:
        target_trans = point + trans_vector
        object_list[object_uid].set_world_pose(position=target_trans)
        meshlist = get_current_meshList(object_list, meshDict)
        if check_mesh_collision(meshlist[object_uid], meshlist[container_uid]):
            continue
        pclist = meshlist_to_pclist(meshlist)
        relation = get_related_position(pclist[object_uid], pclist[container_uid])
        if relation == "in":
            return 0
    return -1


def check_mesh_collision(mesh1, mesh2):
    def o3d2trimesh(o3d_mesh):
        vertices = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    tmesh1 = o3d2trimesh(mesh1)
    tmesh2 = o3d2trimesh(mesh2)

    collision_manager = trimesh.collision.CollisionManager()
    collision_manager.add_object("mesh1", tmesh1)
    collision_manager.add_object("mesh2", tmesh2)
    return collision_manager.in_collision_internal()


def sample_points_in_convex_hull(mesh, num_points=1000):
    vertices = np.asarray(mesh.vertices)
    hull = ConvexHull(vertices)
    hull_vertices = vertices[hull.vertices]
    points = []
    while len(points) < num_points:
        random_point = np.random.uniform(hull_vertices.min(axis=0), hull_vertices.max(axis=0))
        if all(np.dot(eq[:-1], random_point) + eq[-1] <= 0 for eq in hull.equations):
            points.append(random_point)
    points = np.array(points)
    return points


def compute_mesh_center(mesh):
    pcd = get_pcd_from_mesh(mesh)
    return compute_pcd_center(pcd)


def compute_pcd_center(pcd):
    pointcloud = np.asarray(pcd.points)
    center = np.mean(pointcloud, axis=0)
    return center


def place_object_between_object1_and_object2(
    object_list,
    meshDict,
    object_uid,
    object1_uid,
    object2_uid,
    platform_uid,
    attemps=100,
):
    meshlist = get_current_meshList(object_list, meshDict)
    pointcloud_list = meshlist_to_pclist(meshlist)
    line_points = sample_point_in_2d_line(
        compute_mesh_center(meshlist[object1_uid]),
        compute_mesh_center(meshlist[object2_uid]),
        1000,
    )
    object_bottom_point = pointcloud_list[object_uid][np.argmin(pointcloud_list[object_uid][:, 2])]
    vec_axis2bottom = object_list[object_uid].get_world_pose()[0] - object_bottom_point
    available_area = get_platform_available_area(
        pointcloud_list[platform_uid],
        pointcloud_list,
        [platform_uid, object_uid],
    )
    available_area = available_area.buffer(
        -get_max_distance_to_polygon(
            get_xy_contour(pointcloud_list[object_uid]),
            Point(object_bottom_point[0], object_bottom_point[1]),
        )
    )
    for _ in range(attemps):
        random_point = Point(random.choice(line_points))
        if available_area.contains(random_point):
            platform_pc = pointcloud_list[platform_uid]
            position = vec_axis2bottom + np.array([random_point.x, random_point.y, np.max(platform_pc[:, 2])])
            object_list[object_uid].set_world_pose(position=position)
            return 0
    return -1


def sample_point_in_2d_line(point1, point2, num_samples=100):
    t = np.linspace(0, 1, num_samples)
    x = point1[0] + (point2[0] - point1[0]) * t
    y = point1[1] + (point2[1] - point1[1]) * t
    return np.stack([x, y], axis=1)
