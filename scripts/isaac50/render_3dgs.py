#!/usr/bin/env python3
"""Render a Gaussian Splatting PLY or NuRec USDZ with CUDA gsplat.

This tool is intentionally independent from Isaac Sim.  It runs in the existing
CUDA 11.8 ``simgen`` environment and produces an RGB image plus an alpha image
that can be consumed by Isaac Sim 5.0.
"""

from __future__ import annotations

import argparse
import gzip
import math
from pathlib import Path
import re
import zipfile

import msgpack
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData


SH_C0 = 0.28209479177387814


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _load_ply_gaussians(path: Path) -> dict[str, np.ndarray]:
    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{path} is not a 3DGS PLY; missing properties: {missing}")

    def columns(*keys: str) -> np.ndarray:
        return np.stack([np.asarray(vertex[key], dtype=np.float32) for key in keys], axis=-1)

    means = columns("x", "y", "z")
    scales = np.exp(columns("scale_0", "scale_1", "scale_2"))
    quats = columns("rot_0", "rot_1", "rot_2", "rot_3")
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)
    opacities = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))
    colors = np.clip(0.5 + SH_C0 * columns("f_dc_0", "f_dc_1", "f_dc_2"), 0.0, 1.0)

    return {"means": means, "scales": scales, "quats": quats, "opacities": opacities, "colors": colors}


def _state_tensor(state: dict, key: str, dtype=np.float16) -> np.ndarray:
    raw = state[key]
    shape = state.get(f"{key}.shape")
    value = np.frombuffer(raw, dtype=dtype)
    if shape is not None:
        value = value.reshape(shape)
    return value.astype(np.float32)


def _load_nurec_gaussians(path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        payload_name = next((name for name in names if name.lower().endswith(".nurec")), None)
        if payload_name is None:
            raise ValueError(f"No .nurec payload found in {path}")
        raw = gzip.decompress(archive.read(payload_name))
        data = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        state = data["nre_data"]["state_dict"]

        prefixes = sorted(
            key[: -len(".positions")]
            for key in state
            if isinstance(key, str)
            and key.startswith(".gaussians_nodes.")
            and key.endswith(".positions")
        )
        if not prefixes:
            prefixes = [".gaussians_nodes.gaussians"]

        def merged(suffix: str) -> np.ndarray:
            return np.concatenate([_state_tensor(state, f"{prefix}.{suffix}") for prefix in prefixes], axis=0)

        means = merged("positions")
        quats = merged("rotations")
        scales = np.exp(merged("scales"))
        opacities = _sigmoid(merged("densities").reshape(-1))
        albedo = merged("features_albedo")
        specular = merged("features_specular")

        # NuRec stores SH coefficients feature-major: DC RGB followed by M RGB triplets.
        sh = np.concatenate((albedo[:, None, :], specular.reshape(specular.shape[0], -1, 3)), axis=1)
        sh_degree = max(0, min(3, int(round(math.sqrt(sh.shape[1]) - 1))))

        gauss_usda = next((name for name in names if name.endswith("gauss.usda")), None)
        if gauss_usda is not None:
            text = archive.read(gauss_usda).decode("utf-8", errors="replace")
            match = re.search(r"matrix4d\s+xformOp:transform\s*=\s*\(\s*\((.*?)\)\s*\)", text, re.S)
            if match:
                numbers = [float(value) for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", match.group(1))]
                if len(numbers) == 16:
                    matrix = np.asarray(numbers, dtype=np.float32).reshape(4, 4)
                    means = np.concatenate((means, np.ones((means.shape[0], 1), np.float32)), axis=1)
                    means = (means @ matrix.T)[:, :3]
                    rotation = matrix[:3, :3]
                    # The supplied NuRec transform is orthonormal; rotate each Gaussian quaternion.
                    q_volume = _rotation_matrix_to_quaternion(rotation)
                    quats = _quaternion_multiply(np.broadcast_to(q_volume, quats.shape), quats)

    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)
    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "colors": sh,
        "sh_degree": sh_degree,
    }


def _rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array([0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s], np.float32)
    axis = int(np.argmax(np.diag(matrix)))
    if axis == 0:
        s = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
        return np.array([(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s], np.float32)
    if axis == 1:
        s = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
        return np.array([(matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s], np.float32)
    s = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
    return np.array([(matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s], np.float32)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left.T
    rw, rx, ry, rz = right.T
    return np.stack((lw * rw - lx * rx - ly * ry - lz * rz, lw * rx + lx * rw + ly * rz - lz * ry, lw * ry - lx * rz + ly * rw + lz * rx, lw * rz + lx * ry - ly * rx + lz * rw), axis=1).astype(np.float32)


def load_gaussians(path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], int | None]:
    arrays = _load_nurec_gaussians(path) if path.suffix.lower() == ".usdz" else _load_ply_gaussians(path)
    sh_degree = arrays.pop("sh_degree", None)
    return {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}, sh_degree


def look_at_view(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return a world-to-camera matrix using OpenCV axes (+x right, +y down, +z forward)."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward), axis=0)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ eye
    return view


def auto_camera(
    means: torch.Tensor,
    azimuth_deg: float,
    elevation_deg: float,
    fov_y_deg: float,
    aspect: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = means.detach().cpu().numpy()
    low = np.quantile(points, 0.01, axis=0)
    high = np.quantile(points, 0.99, axis=0)
    center = (low + high) * 0.5
    extent = np.maximum(high - low, 1e-3)
    radius = float(np.linalg.norm(extent) * 0.5)
    fov_y = math.radians(fov_y_deg)
    fov_x = 2.0 * math.atan(math.tan(fov_y * 0.5) * aspect)
    distance = radius / max(math.sin(min(fov_x, fov_y) * 0.45), 1e-3)
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    direction = np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float32,
    )
    eye = center + direction * distance
    return eye, center


def _rotation_matrix_xyz(euler_deg: np.ndarray) -> np.ndarray:
    """Match scipy Rotation.from_euler("xyz", ..., degrees=True).as_matrix()."""
    x, y, z = np.radians(euler_deg)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rot_x = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)), np.float32)
    rot_y = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)), np.float32)
    rot_z = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)), np.float32)
    return rot_z @ rot_y @ rot_x


def _to_asset_camera_frame(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    translation: np.ndarray,
    euler_deg: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a calibrated Isaac-world camera into the source NuRec USDZ frame."""
    rotation_inv = _rotation_matrix_xyz(euler_deg).T
    eye = rotation_inv @ ((eye - translation) / scale)
    target = rotation_inv @ ((target - translation) / scale)
    up = rotation_inv @ up
    up /= max(float(np.linalg.norm(up)), 1e-8)
    return eye, target, up


@torch.inference_mode()
def render(args: argparse.Namespace) -> None:
    from gsplat import rasterization

    device = torch.device(args.device)
    gaussians, sh_degree = load_gaussians(args.input, device)
    aspect = args.width / args.height
    if args.eye is None:
        eye, target = auto_camera(
            gaussians["means"], args.azimuth, args.elevation, args.fov_y, aspect
        )
    else:
        eye = np.asarray(args.eye, dtype=np.float32)
        target = np.asarray(args.target, dtype=np.float32)

    up = np.asarray(args.up, dtype=np.float32)
    if args.scene_translation is not None:
        eye, target, up = _to_asset_camera_frame(
            eye,
            target,
            up,
            np.asarray(args.scene_translation, dtype=np.float32),
            np.asarray(args.scene_euler, dtype=np.float32),
            np.asarray(args.scene_scale, dtype=np.float32),
        )

    view = look_at_view(eye, target, up)
    fy = 0.5 * args.height / math.tan(math.radians(args.fov_y) * 0.5)
    fx = fy
    intrinsics = np.array(
        [[fx, 0.0, args.width * 0.5], [0.0, fy, args.height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    rgb, alpha, _ = rasterization(
        **gaussians,
        viewmats=torch.from_numpy(view).to(device)[None],
        Ks=torch.from_numpy(intrinsics).to(device)[None],
        width=args.width,
        height=args.height,
        near_plane=args.near,
        far_plane=args.far,
        radius_clip=args.radius_clip,
        backgrounds=torch.tensor(args.background, dtype=torch.float32, device=device),
        rasterize_mode="antialiased" if args.antialias else "classic",
        sh_degree=sh_degree,
    )
    rgb_u8 = (rgb[0].clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
    alpha_u8 = (alpha[0, ..., 0].clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_u8, "RGB").save(args.output)
    alpha_path = args.output.with_name(f"{args.output.stem}_alpha.png")
    Image.fromarray(alpha_u8, "L").save(alpha_path)
    print(
        f"Rendered {gaussians['means'].shape[0]} Gaussians to {args.output} "
        f"(alpha: {alpha_path}, eye={eye.tolist()}, target={target.tolist()})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fov-y", type=float, default=55.0)
    parser.add_argument("--azimuth", type=float, default=-90.0)
    parser.add_argument("--elevation", type=float, default=5.0)
    parser.add_argument("--eye", type=float, nargs=3)
    parser.add_argument("--target", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--scene-translation",
        type=float,
        nargs=3,
        help="Calibrated WorldComposer/NuRec translation in Isaac world coordinates.",
    )
    parser.add_argument(
        "--scene-euler",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Calibrated WorldComposer/NuRec xyz Euler rotation in degrees.",
    )
    parser.add_argument(
        "--scene-scale",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        help="Calibrated WorldComposer/NuRec xyz scale in Isaac world coordinates.",
    )
    parser.add_argument(
        "--up",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        help="World-space camera up vector. The supplied NuRec scene uses -Y after its USD transform.",
    )
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=1000.0)
    parser.add_argument("--radius-clip", type=float, default=0.2)
    parser.add_argument("--background", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--antialias", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
