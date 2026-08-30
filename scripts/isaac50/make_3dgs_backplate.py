#!/usr/bin/env python3
"""Create an unlit USD backplate for a fixed Isaac Sim viewport camera."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def _vec(values: np.ndarray) -> str:
    return "(" + ", ".join(f"{float(value):.9g}" for value in values) + ")"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eye", type=float, nargs=3, required=True)
    parser.add_argument("--target", type=float, nargs=3, required=True)
    parser.add_argument("--fov-y", type=float, default=55.0)
    parser.add_argument("--aspect", type=float, default=16.0 / 9.0)
    parser.add_argument("--distance", type=float, default=60.0)
    args = parser.parse_args()

    eye = np.asarray(args.eye, dtype=np.float64)
    target = np.asarray(args.target, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    center = eye + forward * args.distance
    half_height = args.distance * math.tan(math.radians(args.fov_y) * 0.5) * 1.02
    half_width = half_height * args.aspect
    points = [
        center - right * half_width - up * half_height,
        center + right * half_width - up * half_height,
        center + right * half_width + up * half_height,
        center - right * half_width + up * half_height,
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    texture_path = args.image.resolve().as_posix()
    point_text = ", ".join(_vec(point) for point in points)
    content = f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Mesh "GaussianBackplate" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        bool doubleSided = true
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [{point_text}]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        uniform token subdivisionScheme = "none"
        rel material:binding = </World/Looks/GaussianBackground>
    }}

    def Scope "Looks"
    {{
        def Material "GaussianBackground"
        {{
            token outputs:surface.connect = </World/Looks/GaussianBackground/PreviewSurface.outputs:surface>

            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0, 0, 0)
                color3f inputs:emissiveColor.connect = </World/Looks/GaussianBackground/Texture.outputs:rgb>
                float inputs:metallic = 0
                float inputs:roughness = 1
                token outputs:surface
            }}

            def Shader "Texture"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{texture_path}@
                token inputs:sourceColorSpace = "sRGB"
                float2 inputs:st.connect = </World/Looks/GaussianBackground/Primvar.outputs:result>
                float3 outputs:rgb
            }}

            def Shader "Primvar"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }}
        }}
    }}
}}
'''
    args.output.write_text(content, encoding="ascii")
    print(f"Wrote fixed-camera 3DGS backplate: {args.output}")


if __name__ == "__main__":
    main()
