#!/usr/bin/env python3
"""Build and validate a registered NuRec + MESH compositing stage.

This is intentionally a scene-level validation tool, separate from a task
rollout.  It records the exact transforms used for the WorldComposer 839875
assets and writes a scene using the native NuRec proxy contract.  NuRec
accepts at most four proxy meshes; the proxy remains visible and is made
matte so RTX can use it for depth compositing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp


MESH_TRANSFORM = {
    "translation_m": [1.293, 2.778, 0.364],
    "euler_xyz_deg": [0.0, 0.0, 0.0],
    "scale": [0.475657, 0.475657, 0.475657],
}
NUREC_TRANSFORM = {
    "translation_m": [2.376777, 2.937900, 0.360295],
    "euler_xyz_deg": [-90.0, 0.0, 90.0],
    "scale": [0.475657, 0.475657, 0.475657],
}
SUPPORT_PLANE_Z_M = 0.730


def _apply_transform(xform, transform, Gf) -> None:
    xform.AddTranslateOp().Set(Gf.Vec3d(*transform["translation_m"]))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*transform["euler_xyz_deg"]))
    xform.AddScaleOp().Set(Gf.Vec3f(*transform["scale"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nurec", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-name", default="worldcomposer_839875_registered_compositing.usda")
    parser.add_argument(
        "--proxy-mesh",
        action="append",
        default=[],
        help=(
            "source MESH prim name to bind; repeatable (maximum four). "
            "When omitted, the first source mesh is used only for a minimal "
            "renderer validation."
        ),
    )
    args = parser.parse_args()

    nurec = args.nurec.resolve()
    mesh = args.mesh.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_path = output_dir / args.stage_name
    report_path = output_dir / "registered_compositing_report.json"

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "RayTracedLighting",
            "anti_aliasing": 0,
            "multi_gpu": False,
            "extra_args": ["--enable", "omni.usd.schema.omni_nurec_types"],
        }
    )
    report: dict[str, object] = {
        "nurec_asset": str(nurec),
        "mesh_asset": str(mesh),
        "stage": str(stage_path),
        "mesh_transform": MESH_TRANSFORM,
        "nurec_transform": NUREC_TRANSFORM,
        "support_plane_z_m": SUPPORT_PLANE_Z_M,
        "registered_compositing": False,
        "proxy_target_count": 0,
        "warnings": [],
    }

    try:
        if not nurec.is_file() or not mesh.is_file():
            missing = [str(path) for path in (nurec, mesh) if not path.is_file()]
            raise FileNotFoundError(", ".join(missing))

        import carb
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

        settings = carb.settings.get_settings()
        settings.set("/rtx/post/registeredCompositing/enabled", True)
        settings.set("/rtx/post/registeredCompositing/invertColorCorrection", True)
        settings.set("/rtx/post/registeredCompositing/invertToneMap", True)
        settings.set("/rtx/post/tonemap/op", 2)
        settings.set("/rtx/post/histogram/enabled", False)
        settings.set("/rtx/matteObject/enabled", True)
        settings.set("/rtx/matteObject/visibility/secondaryRays", True)

        context = omni.usd.get_context()
        context.new_stage()
        for _ in range(8):
            app.update()
        stage = context.get_stage()
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        nurec_xform = UsdGeom.Xform.Define(stage, "/World/NativeNuRec")
        nurec_xform.GetPrim().GetReferences().AddReference(str(nurec))
        _apply_transform(nurec_xform, NUREC_TRANSFORM, Gf)

        mesh_xform = UsdGeom.Xform.Define(stage, "/World/RegisteredMesh")
        mesh_xform.GetPrim().GetReferences().AddReference(str(mesh))
        _apply_transform(mesh_xform, MESH_TRANSFORM, Gf)
        for _ in range(20):
            app.update()

        mesh_prims = [
            prim
            for prim in Usd.PrimRange(mesh_xform.GetPrim())
            if prim.IsA(UsdGeom.Mesh)
        ]
        volume_prims = [
            prim
            for prim in Usd.PrimRange(nurec_xform.GetPrim())
            if prim.GetTypeName() == "Volume"
        ]
        if not volume_prims:
            raise RuntimeError("no Volume prim found under the NuRec USDZ reference")
        if not mesh_prims:
            raise RuntimeError("no Mesh prim found under the MESH reference")

        requested_proxy_names = list(dict.fromkeys(args.proxy_mesh))
        if len(requested_proxy_names) > 4:
            raise ValueError("NuRec supports at most four proxy meshes per Volume")
        proxy_mesh_prims = (
            [prim for prim in mesh_prims if prim.GetName() in requested_proxy_names]
            if requested_proxy_names
            else [mesh_prims[0]]
        )
        missing_proxy_names = sorted(
            set(requested_proxy_names) - {prim.GetName() for prim in proxy_mesh_prims}
        )
        if not proxy_mesh_prims:
            raise RuntimeError("requested proxy mesh names did not match any source mesh")
        targets = [prim.GetPath() for prim in proxy_mesh_prims]
        # Identify source meshes that intersect the calibrated task workspace.
        # They are the only ones needed at runtime to provide tabletop depth;
        # the full list remains attached above for an auditable complete proxy.
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        )
        tabletop_meshes = []
        for prim in mesh_prims:
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lower = aligned.GetMin()
            upper = aligned.GetMax()
            bounds = {
                "name": prim.GetName(),
                "path": str(prim.GetPath()),
                "min_m": [float(lower[0]), float(lower[1]), float(lower[2])],
                "max_m": [float(upper[0]), float(upper[1]), float(upper[2])],
            }
            intersects_workspace = (
                bounds["min_m"][0] <= 0.75
                and bounds["max_m"][0] >= -0.75
                and bounds["min_m"][1] <= 0.55
                and bounds["max_m"][1] >= -0.75
                and bounds["min_m"][2] <= SUPPORT_PLANE_Z_M + 0.02
                and bounds["max_m"][2] >= SUPPORT_PLANE_Z_M - 0.02
            )
            if intersects_workspace:
                tabletop_meshes.append(bounds)
        for prim in proxy_mesh_prims:
            prim.CreateAttribute("primvars:isMatteObject", Sdf.ValueTypeNames.Bool).Set(True)
        for volume in volume_prims:
            volume.CreateRelationship("proxy", custom=False).SetTargets(targets)
            volume.CreateAttribute(
                "omni:nurec:useProxyTransform", Sdf.ValueTypeNames.Bool
            ).Set(True)

        # A proxy must remain render-visible.  Matte Object hides its direct
        # color contribution while retaining the depth/shadow information that
        # registered compositing uses.  Hiding this parent also hides all proxy
        # descendants and produces a black/invalid compositing result.
        mesh_xform.GetPrim().CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("inherited")

        # Matte compositing requires a dome light, even though the NuRec volume
        # supplies the background RGB.
        dome = UsdLux.DomeLight.Define(stage, "/World/NuRecDomeLight")
        dome.CreateIntensityAttr(500.0)
        dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        # Mark the known empty tabletop workspace.  These are authoring guides,
        # not collision or render geometry; the actual benchmark keeps its
        # table, Franka, dish and MANO hand as physical USD assets.
        guides = UsdGeom.Xform.Define(stage, "/World/TaskPlacementFrame")
        guides.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, SUPPORT_PLANE_Z_M))
        guides.GetPrim().CreateAttribute("worldcomposer:placementFrame", Sdf.ValueTypeNames.Token).Set("mesh_tabletop")
        guides.GetPrim().CreateAttribute("worldcomposer:franka_xy_m", Sdf.ValueTypeNames.Float2).Set((0.0, -0.47))
        guides.GetPrim().CreateAttribute("worldcomposer:dish_xy_m", Sdf.ValueTypeNames.Float2).Set((-0.12, -0.02))
        guides.GetPrim().CreateAttribute("worldcomposer:hand_xy_m", Sdf.ValueTypeNames.Float2).Set((0.18, -0.02))

        context.save_as_stage(str(stage_path))
        for _ in range(5):
            app.update()

        report.update(
            {
                "registered_compositing": True,
                "requested_proxy_meshes": requested_proxy_names,
                "missing_proxy_meshes": missing_proxy_names,
                "volume_paths": [str(prim.GetPath()) for prim in volume_prims],
                "proxy_target_count": len(targets),
                "proxy_targets": [str(path) for path in targets],
                "proxy_contract": {
                    "maximum_proxy_meshes": 4,
                    "mesh_visibility": "inherited",
                    "matte_object_enabled": True,
                    "dome_light": "/World/NuRecDomeLight",
                },
                "runtime_tabletop_proxy_meshes": tabletop_meshes,
                "placement_frame": {
                    "frame": "/World/TaskPlacementFrame",
                    "franka_xy_m": [0.0, -0.47],
                    "dish_xy_m": [-0.12, -0.02],
                    "hand_xy_m": [0.18, -0.02],
                },
            }
        )
    except Exception as exc:
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        app.close()


if __name__ == "__main__":
    main()
