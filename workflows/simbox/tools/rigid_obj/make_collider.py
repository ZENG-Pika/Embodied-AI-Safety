# pylint: skip-file
# flake8: noqa

import argparse

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from omni.physx.scripts import physicsUtils, utils
from pxr import Usd


def process(
    usd_file,
    create_physics_mat=True,
):
    """Add collision properties to a USD file using convex decomposition.

    Args:
        usd_file: Path to the USD file.
        create_physics_mat: Whether to create physics material.
    """
    stage = Usd.Stage.Open(usd_file)
    aligned_prim = stage.GetPrimAtPath("/World/Aligned")
    child_prim = aligned_prim.GetAllChildren()[0]

    utils.setCollider(child_prim, "convexDecomposition")
    child_prim.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Set(64)
    child_prim.GetAttribute("physxConvexDecompositionCollision:hullVertexLimit").Set(64)
    child_prim.GetAttribute("physxConvexDecompositionCollision:minThickness").Set(0.001)
    child_prim.GetAttribute("physxConvexDecompositionCollision:shrinkWrap").Set(True)
    child_prim.GetAttribute("physxConvexDecompositionCollision:errorPercentage").Set(0.1)
    if create_physics_mat:
        static_friction = 1.0
        dynamic_friction = 1.0
        utils.addRigidBodyMaterial(
            stage,
            "/World/Physics_Materials",
            staticFriction=static_friction,
            dynamicFriction=dynamic_friction,
        )
        physicsUtils.add_physics_material_to_prim(
            stage,
            aligned_prim,
            "/World/Physics_Materials",
        )
    stage.Save()
    print(f"Successfully processed: {usd_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add collision properties to USD file using convex decomposition")
    parser.add_argument("--usd_path", type=str, required=True, help="Path to the USD file")
    args = parser.parse_args()

    process(args.usd_path, create_physics_mat=True)
