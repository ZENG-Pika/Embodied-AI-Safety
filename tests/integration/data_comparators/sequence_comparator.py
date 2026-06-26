"""
Sequence data comparator for navigation pipeline tests.
Provides functions to compare generated navigation sequences with reference data.
"""

import json
import os
import math
import numpy as np
import cv2  # OpenCV is available per requirements
from pathlib import Path
from typing import Tuple, Any, Dict, Optional


def compare_navigation_results(generated_dir: str, reference_dir: str) -> Tuple[bool, str]:
    """Original JSON trajectory sequence comparison (unchanged logic).

    NOTE: Do not modify this function's core behavior. Image comparison is handled by a separate
    wrapper function `compare_navigation_and_images` to avoid side effects on existing tests.
    """
    # --- Enhanced logic ---
    # To support both "caller passes seq_path root directory" and "legacy call (leaf trajectory directory)" forms,
    # here we use a symmetric data.json discovery strategy for both generated and reference sides:
    # 1. If the current directory directly contains data.json, use that file.
    # 2. Otherwise, traverse one level down into subdirectories (sorted alphabetically), looking for <dir>/data.json.
    # 3. Otherwise, search within two nested levels (dir/subdir/data.json) and use the first match found.
    # 4. If not found, report an error. This is compatible with the legacy "generated=root, reference=leaf" usage,
    #    and also allows both sides to provide the root directory.

    if not os.path.isdir(generated_dir):
        return False, f"Generated directory does not exist or is not a directory: {generated_dir}"
    if not os.path.isdir(reference_dir):
        return False, f"Reference directory does not exist or is not a directory: {reference_dir}"

    try:
        generated_file = _locate_first_data_json(generated_dir)
        if generated_file is None:
            return False, f"Could not locate data.json under generated directory: {generated_dir}"
    except Exception as e:  # pylint: disable=broad-except
        return False, f"Error locating generated data file: {e}"

    try:
        reference_file = _locate_first_data_json(reference_dir)
        if reference_file is None:
            # To preserve legacy behavior, if reference_dir/data.json exists but was not detected above (should not happen in theory), check once more
            candidate = os.path.join(reference_dir, "data.json")
            if os.path.isfile(candidate):
                reference_file = candidate
            else:
                return False, f"Could not locate data.json under reference directory: {reference_dir}"
    except Exception as e:  # pylint: disable=broad-except
        return False, f"Error locating reference data file: {e}"

    return compare_trajectory_sequences(generated_file, reference_file)


def compare_navigation_and_images(
    generated_seq_dir: str,
    reference_seq_dir: str,
    generated_root_for_images: Optional[str] = None,
    reference_root_for_images: Optional[str] = None,
    rgb_abs_tolerance: int = 0,
    depth_abs_tolerance: float = 0.0,
    allowed_rgb_diff_ratio: float = 0.0,
    allowed_depth_diff_ratio: float = 0.5,
    depth_scale_auto: bool = False,
    fail_if_images_missing: bool = False,
) -> Tuple[bool, str]:
    """Wrapper that preserves original JSON comparison while optionally adding first-frame image comparison.

    Args:
        generated_seq_dir: Path to generated seq_path root used by original comparator.
        reference_seq_dir: Path to reference seq_path root.
        generated_root_for_images: Root (parent of obs_path) or the obs_path itself for generated images.
        reference_root_for_images: Same as above for reference. If None, image comparison may be skipped.
        rgb_abs_tolerance: RGB absolute per-channel tolerance.
        depth_abs_tolerance: Depth absolute tolerance.
        allowed_rgb_diff_ratio: Allowed differing RGB pixel ratio.
        allowed_depth_diff_ratio: Allowed differing depth pixel ratio.
        depth_scale_auto: Auto scale depth if uint16 millimeters.
        fail_if_images_missing: If True, treat missing obs_path as failure; otherwise skip.

    Returns:
        (success, message) combined result.
    """
    traj_ok, traj_msg = compare_navigation_results(generated_seq_dir, reference_seq_dir)

    # Determine image roots; default to parent of seq_dir if not explicitly provided
    gen_img_root = generated_root_for_images or os.path.dirname(generated_seq_dir.rstrip(os.sep))
    ref_img_root = reference_root_for_images or os.path.dirname(reference_seq_dir.rstrip(os.sep))

    img_ok = True
    img_msg = "image comparison skipped"

    if generated_root_for_images is not None or reference_root_for_images is not None:
        # User explicitly passed at least one root -> attempt compare
        img_ok, img_msg = compare_first_frame_images(
            generated_root=gen_img_root,
            reference_root=ref_img_root,
            rgb_abs_tolerance=rgb_abs_tolerance,
            depth_abs_tolerance=depth_abs_tolerance,
            allowed_rgb_diff_ratio=allowed_rgb_diff_ratio,
            allowed_depth_diff_ratio=allowed_depth_diff_ratio,
            depth_scale_auto=depth_scale_auto,
        )
    else:
        # Implicit attempt only if both obs_path exist under parent paths
        gen_obs_candidate = os.path.join(gen_img_root, "obs_path")
        ref_obs_candidate = os.path.join(ref_img_root, "obs_path")
        if os.path.isdir(gen_obs_candidate) and os.path.isdir(ref_obs_candidate):
            img_ok, img_msg = compare_first_frame_images(
                generated_root=gen_img_root,
                reference_root=ref_img_root,
                rgb_abs_tolerance=rgb_abs_tolerance,
                depth_abs_tolerance=depth_abs_tolerance,
                allowed_rgb_diff_ratio=allowed_rgb_diff_ratio,
                allowed_depth_diff_ratio=allowed_depth_diff_ratio,
                depth_scale_auto=depth_scale_auto,
            )
        else:
            if fail_if_images_missing:
                missing = []
                if not os.path.isdir(gen_obs_candidate):
                    missing.append(f"generated:{gen_obs_candidate}")
                if not os.path.isdir(ref_obs_candidate):
                    missing.append(f"reference:{ref_obs_candidate}")
                img_ok = False
                img_msg = "obs_path missing -> " + ", ".join(missing)
            else:
                img_msg = "obs_path not found in one or both roots; skipped"

    overall = traj_ok and img_ok
    message = f"trajectory: {traj_msg}; images: {img_msg}"
    return overall, message if overall else f"Mismatch - {message}"


def compare_trajectory_sequences(generated_file: str, reference_file: str, tolerance: float = 1e-6) -> Tuple[bool, str]:
    """
    Compare trajectory sequence files with numerical tolerance.

    Args:
        generated_file: Path to generated trajectory file
        reference_file: Path to reference trajectory file
        tolerance: Numerical tolerance for floating point comparisons

    Returns:
        Tuple[bool, str]: (success, message)
    """
    try:
        # Check if files exist
        if not os.path.exists(generated_file):
            return False, f"Generated file does not exist: {generated_file}"

        if not os.path.exists(reference_file):
            return False, f"Reference file does not exist: {reference_file}"

        # Load JSON files
        with open(generated_file, 'r') as f:
            generated_data = json.load(f)

        with open(reference_file, 'r') as f:
            reference_data = json.load(f)

        # Compare the JSON structures
        success, message = _compare_data_structures(generated_data, reference_data, tolerance)

        if success:
            return True, "Trajectory sequences match within tolerance"
        else:
            return False, f"Trajectory sequences differ: {message}"

    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Error comparing trajectory sequences: {e}"


def _compare_data_structures(data1: Any, data2: Any, tolerance: float, path: str = "") -> Tuple[bool, str]:
    """
    Recursively compare two data structures with numerical tolerance.

    Args:
        data1: First data structure
        data2: Second data structure
        tolerance: Numerical tolerance for floating point comparisons
        path: Current path in the data structure for error reporting

    Returns:
        Tuple[bool, str]: (success, error_message)
    """
    # Check if types are the same
    if type(data1) != type(data2):
        return False, f"Type mismatch at {path}: {type(data1)} vs {type(data2)}"

    # Handle dictionaries
    if isinstance(data1, dict):
        if set(data1.keys()) != set(data2.keys()):
            return False, f"Key mismatch at {path}: {set(data1.keys())} vs {set(data2.keys())}"

        for key in data1.keys():
            new_path = f"{path}.{key}" if path else key
            success, message = _compare_data_structures(data1[key], data2[key], tolerance, new_path)
            if not success:
                return False, message

    # Handle lists
    elif isinstance(data1, list):
        if len(data1) != len(data2):
            return False, f"List length mismatch at {path}: {len(data1)} vs {len(data2)}"

        for i, (item1, item2) in enumerate(zip(data1, data2)):
            new_path = f"{path}[{i}]" if path else f"[{i}]"
            success, message = _compare_data_structures(item1, item2, tolerance, new_path)
            if not success:
                return False, message

    # Handle numerical values
    elif isinstance(data1, (int, float)):
        if isinstance(data2, (int, float)):
            if abs(data1 - data2) > tolerance:
                return (
                    False,
                    f"Numerical difference at {path}: {data1} vs {data2} (diff: {abs(data1 - data2)}, tolerance: {tolerance})",
                )
        else:
            return False, f"Type mismatch at {path}: number vs {type(data2)}"

    # Handle strings and other exact comparison types
    elif isinstance(data1, (str, bool, type(None))):
        if data1 != data2:
            return False, f"Value mismatch at {path}: {data1} vs {data2}"

    # Handle unknown types
    else:
        if data1 != data2:
            return False, f"Value mismatch at {path}: {data1} vs {data2}"

    return True, ""

def _locate_first_data_json(root: str) -> Optional[str]:
    """Locate a data.json file under root with a shallow, deterministic strategy.

    Strategy (stop at first match to keep behavior predictable & lightweight):
      1. If root/data.json exists -> return it.
      2. Enumerate immediate subdirectories (sorted). For each d:
         - if d/data.json exists -> return it.
      3. Enumerate immediate subdirectories again; for each d enumerate its subdirectories (sorted) and
         look for d/sub/data.json -> return first match.
      4. If none found -> return None.
    """
    # 1. root/data.json
    candidate = os.path.join(root, "data.json")
    if os.path.isfile(candidate):
        return candidate

    try:
        first_level = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    except FileNotFoundError:
        return None

    first_level.sort()

    # 2. d/data.json
    for d in first_level:
        c = os.path.join(root, d, "data.json")
        if os.path.isfile(c):
            return c

    # 3. d/sub/data.json
    for d in first_level:
        d_path = os.path.join(root, d)
        try:
            second_level = [s for s in os.listdir(d_path) if os.path.isdir(os.path.join(d_path, s))]
        except FileNotFoundError:
            continue
        second_level.sort()
        for s in second_level:
            c = os.path.join(d_path, s, "data.json")
            if os.path.isfile(c):
                return c

    return None


def compare_first_frame_images(
    generated_root: str,
    reference_root: str,
    rgb_dir_name: str = "rgb",
    depth_dir_name: str = "depth",
    scene_dir: Optional[str] = None,
    traj_dir: Optional[str] = None,
    rgb_abs_tolerance: int = 0,
    depth_abs_tolerance: float = 0.0,
    allowed_rgb_diff_ratio: float = 0.0,
    allowed_depth_diff_ratio: float = 0.0,
    compute_psnr: bool = True,
    compute_mse: bool = True,
    depth_scale_auto: bool = False,
) -> Tuple[bool, str]:
    """Compare only the first frame (index 0) of RGB & depth images between generated and reference.

    This is a lightweight check to validate pipeline correctness without scanning all frames.

    Args:
        generated_root: Path to generated run root (may contain `obs_path` or be the `obs_path`).
        reference_root: Path to reference run root (same structure as generated_root).
        rgb_dir_name: Subdirectory name for RGB frames under a trajectory directory.
        depth_dir_name: Subdirectory name for depth frames under a trajectory directory.
        scene_dir: Optional explicit scene directory name (e.g. "6f"); if None will auto-pick first.
        traj_dir: Optional explicit trajectory directory (e.g. "0"); if None will auto-pick first.
        rgb_abs_tolerance: Per-channel absolute pixel tolerance (0 requires exact match).
        depth_abs_tolerance: Absolute tolerance for depth value differences (after optional scaling).
        allowed_rgb_diff_ratio: Max allowed ratio of differing RGB pixels (0.01 -> 1%).
        allowed_depth_diff_ratio: Max allowed ratio of differing depth pixels beyond tolerance.
        compute_psnr: Whether to compute PSNR metric for reporting.
        compute_mse: Whether to compute MSE metric for reporting.
        depth_scale_auto: If True, attempt simple heuristic scaling for uint16 depth (divide by 1000 if max > 10000).

    Returns:
        (success, message) summary of comparison.
    """
    try:
        gen_obs = _resolve_obs_path(generated_root)
        ref_obs = _resolve_obs_path(reference_root)
        if gen_obs is None:
            return False, f"Cannot locate obs_path under generated root: {generated_root}"
        if ref_obs is None:
            return False, f"Cannot locate obs_path under reference root: {reference_root}"

        scene_dir = scene_dir or _pick_first_subdir(gen_obs)
        if scene_dir is None:
            return False, f"No scene directory found in {gen_obs}"
        ref_scene_dir = scene_dir if os.path.isdir(os.path.join(ref_obs, scene_dir)) else _pick_first_subdir(ref_obs)
        if ref_scene_dir is None:
            return False, f"No matching scene directory in reference: {ref_obs}"

        gen_scene_path = os.path.join(gen_obs, scene_dir)
        ref_scene_path = os.path.join(ref_obs, ref_scene_dir)

        traj_dir = traj_dir or _pick_first_subdir(gen_scene_path)
        if traj_dir is None:
            return False, f"No trajectory directory in {gen_scene_path}"
        ref_traj_dir = (
            traj_dir if os.path.isdir(os.path.join(ref_scene_path, traj_dir)) else _pick_first_subdir(ref_scene_path)
        )
        if ref_traj_dir is None:
            return False, f"No trajectory directory in reference scene path {ref_scene_path}"

        gen_traj_path = os.path.join(gen_scene_path, traj_dir)
        ref_traj_path = os.path.join(ref_scene_path, ref_traj_dir)

        # RGB comparison
        rgb_result, rgb_msg = _compare_single_frame_rgb(
            gen_traj_path,
            ref_traj_path,
            rgb_dir_name,
            rgb_abs_tolerance,
            allowed_rgb_diff_ratio,
            compute_psnr,
            compute_mse,
        )

        # Depth comparison (optional if depth folder exists)
        depth_result, depth_msg = _compare_single_frame_depth(
            gen_traj_path,
            ref_traj_path,
            depth_dir_name,
            depth_abs_tolerance,
            allowed_depth_diff_ratio,
            compute_psnr,
            compute_mse,
            depth_scale_auto,
        )

        success = rgb_result and depth_result
        combined_msg = f"RGB: {rgb_msg}; Depth: {depth_msg}"
        return success, ("Images match - " + combined_msg) if success else ("Image mismatch - " + combined_msg)
    except Exception as e:  # pylint: disable=broad-except
        return False, f"Error during first-frame image comparison: {e}"


def _resolve_obs_path(root: str) -> Optional[str]:
    """Return the obs_path directory. Accept either the root itself or its child."""
    if not os.path.isdir(root):
        return None
    if os.path.basename(root) == "obs_path":
        return root
    candidate = os.path.join(root, "obs_path")
    return candidate if os.path.isdir(candidate) else None


def _pick_first_subdir(parent: str) -> Optional[str]:
    """Pick the first alphanumerically sorted subdirectory name under parent."""
    try:
        subs = [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
        if not subs:
            return None
        subs.sort()
        return subs[0]
    except FileNotFoundError:
        return None


def _find_first_frame_file(folder: str, exts: Tuple[str, ...]) -> Optional[str]:
    """Find the smallest numeral file with one of extensions; returns absolute path."""
    if not os.path.isdir(folder):
        return None
    candidates = []
    for f in os.listdir(folder):
        lower = f.lower()
        for e in exts:
            if lower.endswith(e):
                num_part = os.path.splitext(f)[0]
                if num_part.isdigit():
                    candidates.append((int(num_part), f))
                elif f.startswith("0"):  # fallback for names like 0.jpg
                    candidates.append((0, f))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return os.path.join(folder, candidates[0][1])


def _compare_single_frame_rgb(
    gen_traj_path: str,
    ref_traj_path: str,
    rgb_dir_name: str,
    abs_tol: int,
    allowed_ratio: float,
    compute_psnr: bool,
    compute_mse: bool,
) -> Tuple[bool, str]:
    rgb_gen_dir = os.path.join(gen_traj_path, rgb_dir_name)
    rgb_ref_dir = os.path.join(ref_traj_path, rgb_dir_name)
    if not os.path.isdir(rgb_gen_dir) or not os.path.isdir(rgb_ref_dir):
        return (
            False,
            f"RGB directory missing (generated: {os.path.isdir(rgb_gen_dir)}, reference: {os.path.isdir(rgb_ref_dir)})",
        )

    gen_file = _find_first_frame_file(rgb_gen_dir, (".jpg", ".png", ".jpeg"))
    ref_file = _find_first_frame_file(rgb_ref_dir, (".jpg", ".png", ".jpeg"))
    if not gen_file or not ref_file:
        return False, "First RGB frame file not found in one of the directories"

    gen_img = cv2.imread(gen_file, cv2.IMREAD_COLOR)
    ref_img = cv2.imread(ref_file, cv2.IMREAD_COLOR)
    if gen_img is None or ref_img is None:
        return False, "Failed to read RGB images"
    if gen_img.shape != ref_img.shape:
        return False, f"RGB shape mismatch {gen_img.shape} vs {ref_img.shape}"

    diff = np.abs(gen_img.astype(np.int16) - ref_img.astype(np.int16))
    diff_mask = np.any(diff > abs_tol, axis=2)
    diff_ratio = float(diff_mask.sum()) / diff_mask.size

    metrics_parts = [f"diff_pixels_ratio={diff_ratio:.4f}"]
    flag = False
    if compute_mse or compute_psnr:
        mse = float((diff**2).mean())
        if compute_mse:
            metrics_parts.append(f"mse={mse:.2f}")
        if compute_psnr:
            if mse == 0.0:
                psnr = float('inf')
                flag = True
            else:
                psnr = 10.0 * math.log10((255.0**2) / mse)
            if math.isinf(psnr):
                metrics_parts.append("psnr=inf")
                flag = True
            else:
                metrics_parts.append(f"psnr={psnr:.2f}dB")
                if psnr >= 40.0:
                    flag = True

    passed = diff_ratio <= allowed_ratio or flag
    status = "OK" if passed else "FAIL"
    return passed, f"{status} (abs_tol={abs_tol}, allowed_ratio={allowed_ratio}, {' '.join(metrics_parts)})"


def _compare_single_frame_depth(
    gen_traj_path: str,
    ref_traj_path: str,
    depth_dir_name: str,
    abs_tol: float,
    allowed_ratio: float,
    compute_psnr: bool,
    compute_mse: bool,
    auto_scale: bool,
) -> Tuple[bool, str]:
    depth_gen_dir = os.path.join(gen_traj_path, depth_dir_name)
    depth_ref_dir = os.path.join(ref_traj_path, depth_dir_name)
    if not os.path.isdir(depth_gen_dir) or not os.path.isdir(depth_ref_dir):
        return (
            False,
            f"Depth directory missing (generated: {os.path.isdir(depth_gen_dir)}, reference: {os.path.isdir(depth_ref_dir)})",
        )

    gen_file = _find_first_frame_file(depth_gen_dir, (".png", ".exr"))
    ref_file = _find_first_frame_file(depth_ref_dir, (".png", ".exr"))
    if not gen_file or not ref_file:
        return False, "First depth frame file not found in one of the directories"

    gen_img = cv2.imread(gen_file, cv2.IMREAD_UNCHANGED)
    ref_img = cv2.imread(ref_file, cv2.IMREAD_UNCHANGED)
    if gen_img is None or ref_img is None:
        return False, "Failed to read depth images"
    if gen_img.shape != ref_img.shape:
        return False, f"Depth shape mismatch {gen_img.shape} vs {ref_img.shape}"

    gen_depth = _prepare_depth_array(gen_img, auto_scale)
    ref_depth = _prepare_depth_array(ref_img, auto_scale)
    if gen_depth.shape != ref_depth.shape:
        return False, f"Depth array shape mismatch {gen_depth.shape} vs {ref_depth.shape}"

    diff = np.abs(gen_depth - ref_depth)
    diff_mask = diff > abs_tol
    diff_ratio = float(diff_mask.sum()) / diff_mask.size

    metrics_parts = [f"diff_pixels_ratio={diff_ratio:.4f}"]
    if compute_mse or compute_psnr:
        mse = float((diff**2).mean())
        if compute_mse:
            metrics_parts.append(f"mse={mse:.6f}")
        if compute_psnr:
            # Estimate dynamic range from reference depth
            dr = float(ref_depth.max() - ref_depth.min()) or 1.0
            if mse == 0.0:
                psnr = float('inf')
            else:
                psnr = 10.0 * math.log10((dr**2) / mse)
            metrics_parts.append("psnr=inf" if math.isinf(psnr) else f"psnr={psnr:.2f}dB")

    passed = diff_ratio <= allowed_ratio
    status = "OK" if passed else "FAIL"
    return passed, f"{status} (abs_tol={abs_tol}, allowed_ratio={allowed_ratio}, {' '.join(metrics_parts)})"


def _prepare_depth_array(arr: np.ndarray, auto_scale: bool) -> np.ndarray:
    """Convert raw depth image to float32 array; apply simple heuristic scaling if requested."""
    if arr.dtype == np.uint16:
        depth = arr.astype(np.float32)
        if auto_scale and depth.max() > 10000:  # likely millimeters
            depth /= 1000.0
        return depth
    if arr.dtype == np.float32:
        return arr
    # Fallback: convert to float
    return arr.astype(np.float32)
