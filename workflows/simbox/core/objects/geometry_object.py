import glob
import os
import random

from core.objects.base_object import register_object
from omni.isaac.core.prims import GeometryPrim
from omni.isaac.core.utils.prims import (
    create_prim,
    get_prim_at_path,
    is_prim_path_valid,
)

try:
    from omni.isaac.core.materials.omni_pbr import OmniPBR  # Isaac Sim 4.1.0 / 4.2.0
except ImportError:
    from isaacsim.core.api.materials import OmniPBR  # Isaac Sim 4.5.0


@register_object
class GeometryObject(GeometryPrim):
    def __init__(self, asset_root, root_prim_path, cfg, *args, **kwargs):
        """
        Args:
            asset_root: Asset root path
            root_prim_path: Root prim path in USD stage
            cfg: Config dict with required keys:
                - name: Object name
                - path: USD file path relative to asset_root
                - prim_path_child (optional): Child prim path suffix
        """
        # ===== From cfg =====
        self.asset_root = asset_root
        prim_path = os.path.join(root_prim_path, cfg["name"])
        usd_path = os.path.join(asset_root, cfg["path"])
        if cfg.get("prim_path_child", None):
            prim_path = os.path.join(prim_path, cfg["prim_path_child"])
        self.cfg = cfg

        # ===== Initialize =====
        create_prim(prim_path=prim_path, usd_path=usd_path)
        super().__init__(prim_path=prim_path, name=cfg["name"], *args, **kwargs)

    def get_observations(self):
        translation, orientation = self.get_local_pose()
        scale = self.get_local_scale()
        obs = {
            "translation": translation,
            "orientation": orientation,
            "scale": scale,
        }
        return obs

    def apply_texture(self, asset_root, cfg):
        def _recursive_apply(prim, mat_prim_path):
            for child in prim.GetChildren():
                rel = child.GetRelationship("material:binding")
                if rel:
                    rel.SetTargets([mat_prim_path])
                    continue
                _recursive_apply(child, mat_prim_path)

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
            target_prim_path = self.prim_path
            if cfg.get("target_prim_path"):
                target_prim_path = os.path.join(self.prim_path, cfg["target_prim_path"])
                target_prim = get_prim_at_path(target_prim_path)
            else:
                target_prim = get_prim_at_path(target_prim_path)

            _recursive_apply(target_prim, mat_prim_path)
        else:
            self.mat.set_texture(
                texture_path,
            )
