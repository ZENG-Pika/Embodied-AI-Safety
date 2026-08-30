#!/usr/bin/env python3
"""Open a saved USD stage in the Isaac Sim GUI without running a workflow."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path, help="USD stage to open")
    parser.add_argument(
        "--width", type=int, default=640, help="viewport render width (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=360, help="viewport render height (default: 360)"
    )
    parser.add_argument(
        "--camera",
        default="/World/task_0/cameras/franka_head/camera",
        help="camera prim to show after loading; pass an empty string to keep Perspective",
    )
    parser.add_argument(
        "--perspective",
        action="store_true",
        help="keep the editable Perspective view instead of switching to a task camera",
    )
    parser.add_argument(
        "--eye",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Perspective camera position; requires --target",
    )
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Perspective camera look-at point; requires --eye",
    )
    parser.add_argument(
        "--nurec",
        action="store_true",
        help="enable the Isaac Sim 5 NuRec registered-compositing renderer",
    )
    parser.add_argument(
        "--nurec-exposure",
        type=float,
        default=0.0,
        help="NuRec exposure in stops; useful only with --nurec (default: 0)",
    )
    parser.add_argument(
        "--focal-length-mm",
        type=float,
        help="set the editable Perspective camera focal length after stage loading",
    )
    args = parser.parse_args()

    if (args.eye is None) != (args.target is None):
        parser.error("--eye and --target must be provided together")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.perspective or args.eye is not None:
        args.camera = ""

    stage_path = args.stage.expanduser().resolve()
    if not stage_path.is_file():
        parser.error(f"stage does not exist: {stage_path}")
    if stage_path.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
        parser.error(f"unsupported stage extension: {stage_path.suffix}")

    # Let Kit create its initial stage first.  In Isaac Sim 5.0, setting open_usd
    # during SimulationApp construction can be overwritten by the GUI startup.
    from isaacsim import SimulationApp

    # The calibration stage contains a dense reconstructed room. Keep this
    # viewer within the 6 GB GPU budget and avoid the default DLSS allocations.
    app = SimulationApp(
        {
            "headless": False,
            "width": args.width,
            "height": args.height,
            "anti_aliasing": 0,
            "multi_gpu": False,
            "renderer": "RayTracedLighting",
        }
    )
    try:
        import omni.usd

        for _ in range(5):
            app.update()

        if args.nurec:
            import carb
            import omni.kit.app

            manager = omni.kit.app.get_app().get_extension_manager()
            manager.set_extension_enabled_immediate("omni.usd.schema.omni_nurec_types", True)
            settings = carb.settings.get_settings()
            settings.set("/rtx/post/registeredCompositing/enabled", True)
            settings.set("/rtx/post/registeredCompositing/invertColorCorrection", True)
            settings.set("/rtx/post/registeredCompositing/invertToneMap", True)
            settings.set("/rtx/post/tonemap/op", 2)
            settings.set("/rtx/post/histogram/enabled", False)
            settings.set("/rtx/matteObject/enabled", True)
            settings.set("/rtx/matteObject/visibility/secondaryRays", True)
            # Plain Volume NuRec assets need the standard Gaussian tonemap path.
            # Leaving this at the renderer default can make a valid asset appear
            # nearly black after its first Hydra synchronization.
            settings.set("/rtx/rtpt/gaussian/skipTonemapping/enabled", False)
            for _ in range(15):
                app.update()

        context = omni.usd.get_context()
        if not context.open_stage(str(stage_path)):
            raise RuntimeError(f"Isaac Sim could not open: {stage_path}")

        # Give the USD resolver and referenced assets time to populate the stage.
        for _ in range(30):
            app.update()

        stage = context.get_stage()
        if stage is None or not stage.GetPseudoRoot().GetChildren():
            raise RuntimeError(f"stage is empty after opening: {stage_path}")
        if args.nurec:
            from pxr import Sdf, Usd

            for prim in Usd.PrimRange(stage.GetPseudoRoot()):
                is_nurec_volume = (
                    prim.GetTypeName() == "Volume"
                    and prim.HasAttribute("omni:nurec:isNuRecVolume")
                )
                # This WorldComposer export stores RGB in an emissive Field3D.
                # NuRec exposure belongs on that field; keep the volume value as
                # well for exporters that author it at the parent level.
                is_emissive_field = (
                    prim.HasAttribute("fieldName")
                    and prim.GetAttribute("fieldName").Get() == "emissiveColor"
                )
                if is_nurec_volume or is_emissive_field:
                    prim.CreateAttribute(
                        "omni:nurec:exposure", Sdf.ValueTypeNames.Float
                    ).Set(args.nurec_exposure)
            for _ in range(10):
                app.update()
        if args.camera:
            camera_prim = stage.GetPrimAtPath(args.camera)
            if camera_prim.IsValid():
                from isaacsim.core.utils.viewports import set_active_viewport_camera

                set_active_viewport_camera(args.camera)
                for _ in range(5):
                    app.update()
            else:
                print(f"[open_stage_gui] camera not found; keeping Perspective: {args.camera}", flush=True)
        elif args.perspective or args.eye is not None:
            import numpy as np
            from pxr import UsdGeom
            from isaacsim.core.utils.viewports import set_camera_view

            eye = args.eye if args.eye is not None else [2.2, -2.2, 1.8]
            target = args.target if args.target is not None else [0.0, 0.0, 0.72]
            set_camera_view(
                eye=np.array(eye),
                target=np.array(target),
                camera_prim_path="/OmniverseKit_Persp",
            )
            if args.focal_length_mm is not None:
                UsdGeom.Camera(
                    stage.GetPrimAtPath("/OmniverseKit_Persp")
                ).CreateFocalLengthAttr().Set(args.focal_length_mm)
            for _ in range(5):
                app.update()
        root_children = [prim.GetPath().pathString for prim in stage.GetPseudoRoot().GetChildren()]
        print(f"[open_stage_gui] loaded: {stage_path}", flush=True)
        print(f"[open_stage_gui] root prims: {root_children}", flush=True)

        while app.is_running():
            app.update()
    finally:
        app.close()


if __name__ == "__main__":
    main()
