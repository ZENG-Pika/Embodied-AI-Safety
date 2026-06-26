import glob
import os
import random

from core.objects.base_object import register_object
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path

try:
    from omni.isaac.core.materials.omni_pbr import OmniPBR  # Isaac Sim 4.1.0 / 4.2.0
except ImportError:
    from isaacsim.core.api.materials import OmniPBR  # Isaac Sim 4.5.0


@register_object
class RigidObject(RigidPrim):
    def __init__(self, asset_root, root_prim_path, cfg, *args, **kwargs):
        """
        Args:
            asset_root: Asset root path
            root_prim_path: Root prim path in USD stage
            cfg: Config dict with required keys:
                - name: Object name
                - path: USD file path relative to asset_root
                - prim_path_child: Child prim path for rigid body
                - init_translation (optional): Initial translation
                - init_orientation (optional): Initial orientation
                - init_parent (optional): Initial parent prim
                - gap (optional): Gap parameter
                - mass (optional): Object mass
        """
        # ===== From cfg =====
        self.asset_root = asset_root
        cfg_name = cfg["name"]
        cfg_path = cfg["path"]
        prim_path = f"{root_prim_path}/{cfg_name}"
        usd_path = f"{asset_root}/{cfg_path}"
        self.init_translation = cfg.get("init_translation", None)
        self.init_orientation = cfg.get("init_orientation", None)
        self.init_parent = cfg.get("init_parent", None)
        self.gap = cfg.get("gap", None)
        self.mass = cfg.get("mass", None)
        kwargs["mass"] = cfg.get("mass", None)

        # ===== Initialize =====
        create_prim(prim_path=prim_path, usd_path=usd_path)
        self.base_prim_path = prim_path
        rigid_prim_path = os.path.join(self.base_prim_path, cfg["prim_path_child"])
        self.mesh_prim_path = str(get_prim_at_path(rigid_prim_path).GetChildren()[0].GetPrimPath())
        super().__init__(prim_path=rigid_prim_path, name=cfg["name"], *args, **kwargs)

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
        texture_name = cfg["texture_lib"]
        texture_path_list = glob.glob(os.path.join(asset_root, texture_name, "*"))
        texture_path_list.sort()
        if cfg["apply_randomization"]:
            texture_id = random.randint(0, len(texture_path_list) - 1)
        else:
            texture_id = cfg["texture_id"]
        texture_path = texture_path_list[texture_id]
        mat_prim_path = f"{self.base_prim_path}/Looks/Material"
        mat = OmniPBR(
            prim_path=mat_prim_path,
            name="Material",
            texture_path=texture_path,
            texture_scale=cfg.get("texture_scale"),
        )
        self.apply_visual_material(mat)
