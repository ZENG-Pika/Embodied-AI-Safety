import glob
import os
import random

from core.objects.base_object import register_object
from omni.isaac.core.prims import GeometryPrim

try:
    from omni.isaac.core.materials.omni_pbr import OmniPBR  # Isaac Sim 4.1.0 / 4.2.0
except ImportError:
    from isaacsim.core.api.materials import OmniPBR  # Isaac Sim 4.5.0


@register_object
class ShapeObject(GeometryPrim):
    def __init__(self, asset_root, root_prim_path, cfg, *args, **kwargs):
        """
        Args:
            asset_root: Asset root path
            root_prim_path: Root prim path in USD stage
            cfg: Config dict with required keys:
                - name: Object name
        """
        # ===== From cfg =====
        self.asset_root = asset_root
        prim_path = f"{root_prim_path}/{cfg.get('name')}"
        self.cfg = cfg

        # ===== Initialize =====
        self.base_prim_path = prim_path
        super().__init__(prim_path=self.base_prim_path, name=cfg.get("name"), *args, **kwargs)

    def get_observations(self):
        translation, orientation = self.get_local_pose()
        obs = {
            "translation": translation,
            "orientation": orientation,
        }
        return obs

    def apply_texture(self, asset_root, cfg):
        texture_name = cfg.texture_lib
        texture_path_list = glob.glob(os.path.join(asset_root, texture_name, "*.jpg"))
        texture_path_list.sort()
        if cfg.apply_randomization:
            texture_id = random.randint(0, len(texture_path_list) - 1)
        else:
            texture_id = cfg.texture_id
        texture_path = texture_path_list[texture_id]
        mat_prim_path = f"{self.base_prim_path}/Looks/Material"
        mat = OmniPBR(
            prim_path=mat_prim_path,
            name="Material",
            texture_path=texture_path,
            texture_scale=cfg.get("texture_scale"),
        )
        self.apply_visual_material(mat)
