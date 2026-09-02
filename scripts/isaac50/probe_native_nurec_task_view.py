#!/usr/bin/env python3
"""Render a plain or proxy-composited NuRec stage to an RGB validation image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--asset",
        type=Path,
        help="plain NuRec USDZ to reference using the historic task transform",
    )
    source.add_argument(
        "--stage",
        type=Path,
        help="saved USD stage, for example one containing a NuRec proxy relation",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--eye", nargs=3, type=float, default=[1.3, 0.7, 2.7])
    parser.add_argument("--target", nargs=3, type=float, default=[0.0, 0.0, 1.5])
    parser.add_argument(
        "--focal-length-mm",
        type=float,
        default=24.0,
        help="pinhole focal length for the validation camera (default: 24)",
    )
    parser.add_argument(
        "--camera-path",
        help="use an existing USD camera instead of creating /World/TaskCamera",
    )
    parser.add_argument("--nurec-exposure", type=float, default=0.0)
    args = parser.parse_args()
    asset = args.asset.resolve() if args.asset else None
    source_stage = args.stage.resolve() if args.stage else None
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    app = SimulationApp({
        "headless": True,
        "width": args.width,
        "height": args.height,
        "renderer": "RayTracedLighting",
        "anti_aliasing": 0,
        "multi_gpu": False,
        "extra_args": ["--enable", "omni.usd.schema.omni_nurec_types"],
    })
    annotator = None
    render_product = None
    report = {
        "asset": str(asset) if asset else None,
        "stage": str(source_stage) if source_stage else None,
        "native_nurec_rendered": False,
    }
    try:
        import carb
        import numpy as np
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import Gf, Sdf, Usd, UsdGeom

        settings = carb.settings.get_settings()
        settings.set("/rtx/post/registeredCompositing/enabled", True)
        settings.set("/rtx/post/registeredCompositing/invertColorCorrection", True)
        settings.set("/rtx/post/registeredCompositing/invertToneMap", True)
        settings.set("/rtx/post/histogram/enabled", False)
        settings.set("/rtx/raytracing/fractionalCutoutOpacity", False)
        settings.set("/rtx/matteObject/enabled", True)
        settings.set("/rtx/matteObject/visibility/secondaryRays", True)
        settings.set("/rtx/rtpt/gaussian/skipTonemapping/enabled", False)

        context = omni.usd.get_context()
        if source_stage:
            if not source_stage.is_file():
                raise FileNotFoundError(source_stage)
            if not context.open_stage(str(source_stage)):
                raise RuntimeError(f"Isaac Sim could not open: {source_stage}")
            for _ in range(50):
                app.update()
            stage = context.get_stage()
        else:
            if not asset or not asset.is_file():
                raise FileNotFoundError(asset)
            context.new_stage()
            for _ in range(5):
                app.update()
            stage = context.get_stage()
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            scene = UsdGeom.Xform.Define(stage, "/World/NativeNuRecBackground")
            scene.GetPrim().GetReferences().AddReference(str(asset))
            scene.AddTranslateOp().Set(Gf.Vec3d(-0.525141, 1.608039, 1.130879))
            scene.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 39.102229, -151.699244))
            scene.AddScaleOp().Set(Gf.Vec3f(0.475657, 0.475657, 0.475657))

        if stage is None:
            raise RuntimeError("stage was not available after loading")

        camera_path = args.camera_path or "/World/TaskCamera"
        if args.camera_path:
            if not stage.GetPrimAtPath(camera_path).IsA(UsdGeom.Camera):
                raise ValueError(f"camera path is not a UsdGeom.Camera: {camera_path}")
        else:
            camera = UsdGeom.Camera.Define(stage, camera_path)
            camera.CreateFocalLengthAttr(args.focal_length_mm)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
            set_camera_view(
                eye=args.eye,
                target=args.target,
                camera_prim_path=camera_path,
            )
        for _ in range(10):
            app.update()

        target_prims = [
            {"path": str(prim.GetPath()), "type_name": str(prim.GetTypeName())}
            for prim in stage.Traverse()
            if prim.GetTypeName() in {"Volume", "OmniNuRecFieldAsset"}
        ]
        visible_nurec_volumes = [
            prim
            for prim in stage.Traverse()
            if prim.GetTypeName() == "Volume"
            and prim.HasAttribute("omni:nurec:isNuRecVolume")
            and UsdGeom.Imageable(prim).ComputeVisibility() != "invisible"
        ]
        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            is_volume = prim.GetTypeName() == "Volume" and prim.HasAttribute(
                "omni:nurec:isNuRecVolume"
            )
            is_emissive_field = (
                prim.HasAttribute("fieldName")
                and prim.GetAttribute("fieldName").Get() == "emissiveColor"
            )
            if is_volume or is_emissive_field:
                prim.CreateAttribute(
                    "omni:nurec:exposure", Sdf.ValueTypeNames.Float
                ).Set(args.nurec_exposure)
        for _ in range(20):
            app.update()
        render_product = rep.create.render_product(
            camera_path, (args.width, args.height)
        )
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([render_product])
        rep.orchestrator.step(rt_subframes=8)
        rgba = np.asarray(annotator.get_data())
        rgb = rgba[:, :, :3].astype(np.float32)
        std = float(rgb.std()) if rgb.size else 0.0
        image_path = output_dir / "native_nurec_task_view.png"
        if rgb.size:
            Image.fromarray(rgba.astype(np.uint8), "RGBA").convert("RGB").save(image_path)
        report.update({
            "kit_build": str(omni.kit.app.get_app().get_build_version()),
            "image_path": str(image_path),
            "shape": list(rgba.shape),
            "rgb_std": std,
            "target_prims": target_prims,
            "visible_nurec_volume_paths": [str(prim.GetPath()) for prim in visible_nurec_volumes],
            "native_nurec_rendered": bool(visible_nurec_volumes and std > 2.0),
            "uses_backplate": False,
            "camera": {
                "path": camera_path,
                "eye": args.eye,
                "target": args.target,
                "focal_length_mm": args.focal_length_mm,
            },
        })
    except Exception as exc:
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (output_dir / "native_nurec_task_view.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)
        if annotator is not None and render_product is not None:
            try:
                annotator.detach([render_product])
            except Exception:
                pass
        try:
            if 'rep' in locals():
                rep.orchestrator.stop()
            app.update()
        except Exception:
            pass
        app.close()


if __name__ == "__main__":
    main()
