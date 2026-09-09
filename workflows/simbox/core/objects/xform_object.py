import glob
import os
import random

import numpy as np

from core.objects.base_object import register_object
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path, is_prim_path_valid

try:
    from omni.isaac.core.materials.omni_pbr import OmniPBR  # Isaac Sim 4.1.0 / 4.2.0
except ImportError:
    from isaacsim.core.api.materials import OmniPBR  # Isaac Sim 4.5.0


@register_object
class XFormObject(XFormPrim):
    def __init__(self, asset_root, root_prim_path, cfg, *args, **kwargs):
        """
        Args:
            asset_root: Asset root path
            root_prim_path: Root prim path in USD stage
            cfg: Config dict with required keys:
                - name: Object name
                - path: USD file path relative to asset_root
        """
        # ===== From cfg =====
        self.asset_root = asset_root
        prim_path = os.path.join(root_prim_path, cfg["name"])
        usd_path = os.path.join(asset_root, cfg["path"])
        self.cfg = cfg

        # ===== Initialize =====
        create_prim(prim_path=prim_path, usd_path=usd_path)
        super().__init__(prim_path=prim_path, name=cfg["name"], *args, **kwargs)

    def get_observations(self):
        translation, orientation = self.get_local_pose()
        obs = {
            "translation": translation,
            "orientation": orientation,
        }

        return obs

    def apply_texture(self, asset_root, cfg):
        texture_name = cfg["texture_lib"]
        texture_path_list = glob.glob(os.path.join(asset_root, texture_name, "*"))
        texture_path_list.sort()
        if cfg["apply_randomization"]:
            texture_id = random.randint(0, len(texture_path_list) - 1)
        else:
            texture_id = cfg["texture_id"]
        texture_path = texture_path_list[texture_id]
        mat_prim_path = f"{self.prim_path}/Looks/Material"
        if not is_prim_path_valid(mat_prim_path):
            self.mat = OmniPBR(
                prim_path=mat_prim_path,
                name="Material",
                texture_path=texture_path,
                texture_scale=cfg.get("texture_scale"),
            )
            self.apply_visual_material(self.mat)
        else:
            self.mat.set_texture(
                texture_path,
            )


@register_object
class RoboSafeHumanObject(XFormPrim):
    """Full RoboSafe visible human with an authored compound collision body.

    The retargeted USDA contains the visual People character, skeleton and 23
    collision shapes. It is loaded as one root body: visual-mesh collision is
    disabled, while ``HumanSkeletonPhysics`` shapes remain enabled.
    """

    def __init__(self, asset_root, root_prim_path, cfg, *args, **kwargs):
        self.asset_root = asset_root
        self.cfg = cfg
        prim_path = os.path.join(root_prim_path, cfg["name"])
        usd_path = cfg["path"] if os.path.isabs(cfg["path"]) else os.path.join(asset_root, cfg["path"])
        create_prim(prim_path=prim_path, usd_path=usd_path)
        super().__init__(prim_path=prim_path, name=cfg["name"], *args, **kwargs)
        self.base_prim_path = prim_path
        self.collision_paths = []
        self._configure_physics()

    def _configure_physics(self):
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        root = get_prim_at_path(self.prim_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"RoboSafe human root does not exist: {self.prim_path}")

        rigid = UsdPhysics.RigidBodyAPI(root) if root.HasAPI(UsdPhysics.RigidBodyAPI) else UsdPhysics.RigidBodyAPI.Apply(root)
        rigid.CreateRigidBodyEnabledAttr().Set(True)
        # A fixed-pose human must not be dynamically propelled when its
        # hand reaches over a table. Kinematic bodies still participate in
        # PhysX contact queries while preserving the authored pose.
        rigid.CreateKinematicEnabledAttr().Set(bool(self.cfg.get(
            "kinematic", self.cfg.get("fixed_pose", False)
        )))
        mass_api = UsdPhysics.MassAPI(root) if root.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI.Apply(root)
        mass_api.CreateMassAttr().Set(float(self.cfg.get("collision_body_mass_kg", 75.0)))
        physx = PhysxSchema.PhysxRigidBodyAPI(root) if root.HasAPI(PhysxSchema.PhysxRigidBodyAPI) else PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        physx.CreateDisableGravityAttr().Set(True)
        physx.CreateLinearDampingAttr().Set(float(self.cfg.get("linear_damping", 0.05)))
        physx.CreateAngularDampingAttr().Set(float(self.cfg.get("angular_damping", 25.0)))

        include = tuple(str(v).lower() for v in self.cfg.get(
            "collision_include_substrings", ["/HumanSkeletonPhysics"]
        ))
        render_exclude = tuple(str(v).lower() for v in self.cfg.get(
            "render_exclude_substrings", ["/HumanSkeletonPhysics", "/CollisionDebug"]
        ))
        disabled = 0
        hidden = 0
        for prim in Usd.PrimRange(root):
            path_lower = str(prim.GetPath()).lower()
            is_shape = prim.GetTypeName() in {"Capsule", "Sphere", "Mesh", "Cube", "Cylinder"}
            included = any(token in path_lower for token in include)
            if is_shape and included:
                api = UsdPhysics.CollisionAPI(prim) if prim.HasAPI(UsdPhysics.CollisionAPI) else UsdPhysics.CollisionAPI.Apply(prim)
                api.CreateCollisionEnabledAttr().Set(True)
                self.collision_paths.append(str(prim.GetPath()))
            elif prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                disabled += 1
            if any(token in path_lower for token in render_exclude):
                try:
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden += 1
                except Exception:
                    pass

        minimum = int(self.cfg.get("minimum_collision_shapes", 23))
        if len(self.collision_paths) < minimum:
            raise RuntimeError(
                f"RoboSafe human collision skeleton incomplete: {len(self.collision_paths)} shapes"
            )
        self.collision_manifest = {
            "collision_shape_count": len(self.collision_paths),
            "disabled_visual_collision_count": disabled,
            "hidden_debug_prim_count": hidden,
            "collision_paths": sorted(self.collision_paths),
        }
        print(
            f"[robosafe_human] loaded {self.cfg.get('human_model_id', 'human')} "
            f"collision_shapes={len(self.collision_paths)} root={self.prim_path}"
        )

    def apply_fixed_pose(self):
        """Author the configured pose directly on the reference root.

        Some referenced RoboSafe stages reset their root xform during PhysX
        handle creation, so the generic XFormPrim pose setter is not enough.
        XformCommonAPI updates the root op without touching child meshes,
        materials, skeleton or collision shapes.
        """
        from pxr import Gf, UsdGeom

        root = get_prim_at_path(self.prim_path)
        if not root or not root.IsValid():
            return
        xformable = UsdGeom.Xformable(root)

        def _set_op(op_type, value, precision=UsdGeom.XformOp.PrecisionDouble):
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == op_type:
                    op.Set(value)
                    return
            if op_type == UsdGeom.XformOp.TypeTranslate:
                xformable.AddTranslateOp(precision).Set(value)
            elif op_type == UsdGeom.XformOp.TypeOrient:
                xformable.AddOrientOp(precision).Set(value)
            elif op_type == UsdGeom.XformOp.TypeScale:
                xformable.AddScaleOp(precision).Set(value)

        # The referenced RoboSafe stage has its own authored transform. The
        # generic XFormPrim setter can be reset by PhysX handle creation, so
        # re-author both translation and orientation on the reference root.
        # Keep the quaternion convention used by get_orientation: [w,x,y,z].
        from core.utils.transformation_utils import get_orientation
        orientation = get_orientation(
            self.cfg.get("euler"), self.cfg.get("quaternion")
        )
        q = [float(v) for v in orientation]
        _set_op(
            UsdGeom.XformOp.TypeOrient,
            Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3])),
        )

        translation = self.cfg.get("translation")
        if translation is not None:
            _set_op(
                UsdGeom.XformOp.TypeTranslate,
                Gf.Vec3d(*[float(v) for v in translation]),
            )
        scale = self.cfg.get("scale")
        if scale is not None:
            _set_op(
                UsdGeom.XformOp.TypeScale,
                Gf.Vec3f(*[float(v) for v in scale]),
                UsdGeom.XformOp.PrecisionFloat,
            )

    def get_observations(self):
        translation, orientation = self.get_local_pose()
        return {
            "translation": translation,
            "orientation": orientation,
            "scale": self.get_local_scale(),
            "collision_shape_count": len(self.collision_paths),
        }

    def initialize(self, physics_sim_view=None):
        """Bind the root compound body after PhysX has created its handles."""
        from omni.isaac.core.prims import RigidPrim

        self._rigid_prim = RigidPrim(
            prim_path=self.prim_path,
            name=f"{self.name}_physical_actor",
        )
        self._rigid_prim.initialize()
        view = getattr(self._rigid_prim, "_rigid_prim_view", None)
        if view is None or not view.is_physics_handle_valid():
            raise RuntimeError("RoboSafe human PhysX tensor view is invalid")
        self._rigid_prim_view = view

    def set_linear_velocity(self, velocity):
        if self.cfg.get("kinematic", self.cfg.get("fixed_pose", False)):
            return
        if hasattr(self, "_rigid_prim"):
            self._rigid_prim.set_linear_velocity(np.asarray(velocity, dtype=np.float32))

    def set_angular_velocity(self, velocity):
        if self.cfg.get("kinematic", self.cfg.get("fixed_pose", False)):
            return
        if hasattr(self, "_rigid_prim"):
            self._rigid_prim.set_angular_velocity(np.asarray(velocity, dtype=np.float32))
