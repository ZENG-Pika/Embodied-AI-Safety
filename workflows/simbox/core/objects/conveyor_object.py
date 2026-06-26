import os

from core.objects.base_object import register_object
from omni.isaac.core.prims import GeometryPrim
from omni.isaac.core.utils.prims import create_prim
from pxr import Gf, UsdPhysics


@register_object
class ConveyorObject(GeometryPrim):
    def __init__(self, asset_root, root_prim_path, stage, cfg, *args, **kwargs):
        """
        Args:
            asset_root: Asset root path
            root_prim_path: Root prim path in USD stage
            stage: USD stage
            cfg: Config dict with required keys:
                - name: Object name
                - linear_velocity: [x, y, z] linear velocity in m/s
                - linear_track_list: List of linear track names in USD
                - angular_velocity: [x, y, z] angular velocity in rad/s
                - angular_track_list: List of angular track names in USD
        """
        # ===== From cfg =====
        self.asset_root = asset_root
        self.stage = stage
        prim_path = os.path.join(root_prim_path, cfg["name"])
        usd_path = os.path.join(asset_root, cfg["path"])
        self.linear_velocity = Gf.Vec3f(tuple(cfg["linear_velocity"]))
        self.linear_track_list = cfg["linear_track_list"]
        self.angular_velocity = Gf.Vec3f(tuple(cfg["angular_velocity"]))
        self.angular_track_list = cfg["angular_track_list"]

        # ===== Initialize =====
        boxActorPath = prim_path
        create_prim(usd_path=usd_path, prim_path=boxActorPath)
        super().__init__(prim_path=boxActorPath, name=cfg["name"], *args, **kwargs)

        # ===== Configure tracks (hard-coded logic) =====
        linear_dir_list = [1, -1]
        for i, linear_track in enumerate(self.linear_track_list):
            belt_path = f"{boxActorPath}/World/{linear_track}/node_"
            belt_prim = self.stage.GetPrimAtPath(belt_path)
            UsdPhysics.CollisionAPI.Apply(belt_prim)
            linarConveyor = UsdPhysics.RigidBodyAPI.Apply(belt_prim)
            linarConveyor.CreateKinematicEnabledAttr().Set(True)
            velocityAttribute = linarConveyor.GetVelocityAttr()
            velocityAttribute.Set(linear_dir_list[i] * self.linear_velocity)

        angular_dir = 1
        for i, angular_track in enumerate(self.angular_track_list):
            belt_path = f"{boxActorPath}/World/{angular_track}/validate_obj"
            belt_prim = self.stage.GetPrimAtPath(belt_path)
            UsdPhysics.CollisionAPI.Apply(belt_prim)
            angularConveyor = UsdPhysics.RigidBodyAPI.Apply(belt_prim)
            angularConveyor.CreateKinematicEnabledAttr().Set(True)
            angularvelocityAttribute = angularConveyor.GetAngularVelocityAttr()
            angularvelocityAttribute.Set(angular_dir * self.angular_velocity)
