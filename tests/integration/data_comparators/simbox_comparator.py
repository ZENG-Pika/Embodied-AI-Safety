"""
Simbox Output Comparator

This module provides functionality to compare two Simbox task output directories.
It compares both meta_info.pkl and LMDB database contents, handling different data types:
- JSON data (dict/list)
- Scalar data (numerical arrays/lists)
- Image data (encoded images)
- Proprioception data (joint states, gripper states)
- Object data (object poses and properties)
- Action data (robot actions)
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import lmdb
import numpy as np


class SimboxComparator:
    """Comparator for Simbox task output directories."""

    def __init__(self, dir1: str, dir2: str, tolerance: float = 1e-6, image_psnr_threshold: float = 30.0):
        """
        Initialize the comparator.

        Args:
            dir1: Path to the first output directory
            dir2: Path to the second output directory
            tolerance: Numerical tolerance for floating point comparisons
            image_psnr_threshold: PSNR threshold (dB) for considering images as acceptable match
        """
        self.dir1 = Path(dir1)
        self.dir2 = Path(dir2)
        self.tolerance = tolerance
        self.image_psnr_threshold = image_psnr_threshold
        self.mismatches = []
        self.warnings = []
        self.image_psnr_values: List[float] = []

    def load_directory(self, directory: Path) -> Tuple[Optional[Dict], Optional[Any], Optional[Any]]:
        """Load meta_info.pkl and LMDB database from directory."""
        meta_path = directory / "meta_info.pkl"
        lmdb_path = directory / "lmdb"

        if not directory.is_dir() or not meta_path.exists() or not lmdb_path.is_dir():
            print(f"Error: '{directory}' is not a valid output directory.")
            print("It must contain 'meta_info.pkl' and an 'lmdb' subdirectory.")
            return None, None, None

        try:
            with open(meta_path, "rb") as f:
                meta_info = pickle.load(f)
        except Exception as e:
            print(f"Error loading meta_info.pkl from {directory}: {e}")
            return None, None, None

        try:
            env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
            txn = env.begin(write=False)
        except Exception as e:
            print(f"Error opening LMDB database at {lmdb_path}: {e}")
            return None, None, None

        return meta_info, txn, env

    def compare_metadata(self, meta1: Dict, meta2: Dict) -> bool:
        """Compare high-level metadata."""
        identical = True

        if meta1.get("num_steps") != meta2.get("num_steps"):
            self.mismatches.append(f"num_steps differ: {meta1.get('num_steps')} vs {meta2.get('num_steps')}")
            identical = False

        return identical

    def get_key_categories(self, meta: Dict) -> Dict[str, set]:
        """Extract key categories from metadata."""
        key_to_category = {}
        for category, keys in meta.get("keys", {}).items():
            for key in keys:
                key_bytes = key if isinstance(key, bytes) else key.encode()
                key_to_category[key_bytes] = category

        return key_to_category

    def compare_json_data(self, key: bytes, data1: Any, data2: Any) -> bool:
        """Compare JSON/dict/list data."""
        if type(data1) != type(data2):
            self.mismatches.append(f"[{key.decode()}] Type mismatch: {type(data1).__name__} vs {type(data2).__name__}")
            return False

        if isinstance(data1, dict):
            if set(data1.keys()) != set(data2.keys()):
                self.mismatches.append(f"[{key.decode()}] Dict keys differ")
                return False
            for k in data1.keys():
                if not self.compare_json_data(key, data1[k], data2[k]):
                    return False
        elif isinstance(data1, list):
            if len(data1) != len(data2):
                self.mismatches.append(f"[{key.decode()}] List length differ: {len(data1)} vs {len(data2)}")
                return False
            # For lists, compare sample elements to avoid excessive output
            if len(data1) > 10:
                sample_indices = [0, len(data1) // 2, -1]
                for idx in sample_indices:
                    if not self.compare_json_data(key, data1[idx], data2[idx]):
                        return False
            else:
                for i, (v1, v2) in enumerate(zip(data1, data2)):
                    if not self.compare_json_data(key, v1, v2):
                        return False
        else:
            if data1 != data2:
                self.mismatches.append(f"[{key.decode()}] Value mismatch: {data1} vs {data2}")
                return False

        return True

    def compare_numerical_data(self, key: bytes, data1: Any, data2: Any) -> bool:
        """Compare numerical data (arrays, lists of numbers)."""
        # Convert to numpy arrays for comparison
        try:
            if isinstance(data1, list):
                arr1 = np.array(data1)
                arr2 = np.array(data2)
            else:
                arr1 = data1
                arr2 = data2

            if arr1.shape != arr2.shape:
                self.mismatches.append(f"[{key.decode()}] Shape mismatch: {arr1.shape} vs {arr2.shape}")
                return False

            if not np.allclose(arr1, arr2, rtol=self.tolerance, atol=self.tolerance):
                diff = np.abs(arr1 - arr2)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                self.mismatches.append(
                    f"[{key.decode()}] Numerical difference: max={max_diff:.6e}, mean={mean_diff:.6e}"
                )
                return False

        except Exception as e:
            self.warnings.append(f"[{key.decode()}] Error comparing numerical data: {e}")
            return False

        return True

    def compare_image_data(self, key: bytes, data1: np.ndarray, data2: np.ndarray) -> bool:
        """Compare image data (encoded as uint8 arrays)."""
        try:
            # Decode images
            img1 = cv2.imdecode(data1, cv2.IMREAD_UNCHANGED)
            img2 = cv2.imdecode(data2, cv2.IMREAD_UNCHANGED)

            if img1 is None or img2 is None:
                self.warnings.append(f"[{key.decode()}] Could not decode image, using binary comparison")
                return np.array_equal(data1, data2)

            # Compare shapes
            if img1.shape != img2.shape:
                self.mismatches.append(f"[{key.decode()}] Image shape mismatch: {img1.shape} vs {img2.shape}")
                return False

            # Calculate PSNR for tracking average quality
            if np.array_equal(img1, img2):
                psnr = 100.0
            else:
                diff_float = img1.astype(np.float32) - img2.astype(np.float32)
                mse = np.mean(diff_float ** 2)
                if mse == 0:
                    psnr = 100.0
                else:
                    max_pixel = 255.0
                    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

            self.image_psnr_values.append(psnr)

            try:
                print(f"[{key.decode()}] PSNR: {psnr:.2f} dB")
            except Exception:
                print(f"[<binary key>] PSNR: {psnr:.2f} dB")

            return True

        except Exception as e:
            self.warnings.append(f"[{key.decode()}] Error comparing image: {e}")
            return False

    def _save_comparison_image(self, key: bytes, img1: np.ndarray, img2: np.ndarray, diff: np.ndarray):
        """Save comparison visualization for differing images."""
        try:
            output_dir = Path("image_comparisons")
            output_dir.mkdir(exist_ok=True)

            # Normalize difference for visualization
            if len(diff.shape) == 3:
                diff_vis = np.clip(diff * 5, 0, 255).astype(np.uint8)
            else:
                diff_vis = np.clip(diff * 5, 0, 255).astype(np.uint8)

            # Ensure RGB format for concatenation
            def ensure_rgb(img):
                if len(img.shape) == 2:
                    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                return img

            img1_rgb = ensure_rgb(img1)
            img2_rgb = ensure_rgb(img2)
            diff_rgb = ensure_rgb(diff_vis)

            # Concatenate horizontally
            combined = np.hstack([img1_rgb, img2_rgb, diff_rgb])

            # Save
            safe_key = key.decode().replace("/", "_").replace("\\", "_").replace(":", "_")
            output_path = output_dir / f"diff_{safe_key}.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        except Exception as e:
            self.warnings.append(f"Failed to save comparison image for {key.decode()}: {e}")

    def compare_value(self, key: bytes, category: str, val1: bytes, val2: bytes) -> bool:
        """Compare a single key-value pair based on its category."""
        # Output the category for the current key being compared
        try:
            print(f"[{key.decode()}] category: {category}")
        except Exception:
            print(f"[<binary key>] category: {category}")

        if val1 is None and val2 is None:
            return True
        if val1 is None or val2 is None:
            self.mismatches.append(f"[{key.decode()}] Key exists in one dataset but not the other")
            return False

        try:
            data1 = pickle.loads(val1)
            data2 = pickle.loads(val2)
        except Exception as e:
            self.warnings.append(f"[{key.decode()}] Error unpickling data: {e}")
            return val1 == val2

        # Route to appropriate comparison based on category
        if category == "json_data":
            return self.compare_json_data(key, data1, data2)
        elif category in ["scalar_data", "proprio_data", "object_data", "action_data"]:
            return self.compare_numerical_data(key, data1, data2)
        elif category.startswith("images."):
            # Image data is stored as numpy uint8 array
            if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
                return self.compare_image_data(key, data1, data2)
            else:
                self.warnings.append(f"[{key.decode()}] Expected numpy array for image data")
                return False
        else:
            # Unknown category, try generic comparison
            self.warnings.append(f"[{key.decode()}] Unknown category '{category}', using binary comparison")
            return val1 == val2

    def compare(self) -> bool:
        """Execute full comparison."""
        print(f"Comparing directories:")
        print(f"  Dir1: {self.dir1}")
        print(f"  Dir2: {self.dir2}\n")

        # Load data
        meta1, txn1, env1 = self.load_directory(self.dir1)
        meta2, txn2, env2 = self.load_directory(self.dir2)

        if meta1 is None or meta2 is None:
            print("Aborting due to loading errors.")
            return False

        print("Successfully loaded data from both directories.\n")

        # Compare metadata
        print("Comparing metadata...")
        self.compare_metadata(meta1, meta2)

        # Get key categories
        key_cat1 = self.get_key_categories(meta1)
        key_cat2 = self.get_key_categories(meta2)

        keys1 = set(key_cat1.keys())
        keys2 = set(key_cat2.keys())

        # Check key sets
        if keys1 != keys2:
            missing_in_2 = sorted([k.decode() for k in keys1 - keys2])
            missing_in_1 = sorted([k.decode() for k in keys2 - keys1])
            if missing_in_2:
                self.mismatches.append(f"Keys missing in dir2: {missing_in_2[:10]}")
            if missing_in_1:
                self.mismatches.append(f"Keys missing in dir1: {missing_in_1[:10]}")

        # Compare common keys
        common_keys = sorted(list(keys1.intersection(keys2)))
        print(f"Comparing {len(common_keys)} common keys...\n")

        for i, key in enumerate(common_keys):
            if i % 100 == 0 and i > 0:
                print(f"Progress: {i}/{len(common_keys)} keys compared...")

            category = key_cat1.get(key, "unknown")
            val1 = txn1.get(key)
            val2 = txn2.get(key)

            self.compare_value(key, category, val1, val2)

        if self.image_psnr_values:
            avg_psnr = sum(self.image_psnr_values) / len(self.image_psnr_values)
            print(
                f"\nImage PSNR average over {len(self.image_psnr_values)} images: "
                f"{avg_psnr:.2f} dB (threshold {self.image_psnr_threshold:.2f} dB)"
            )
            if avg_psnr < self.image_psnr_threshold:
                self.mismatches.append(
                    f"Average image PSNR {avg_psnr:.2f} dB below threshold "
                    f"{self.image_psnr_threshold:.2f} dB"
                )
        else:
            print("\nNo image entries found for PSNR calculation.")

        # Cleanup
        env1.close()
        env2.close()

        # Print results
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)

        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings[:20]:
                print(f"  - {warning}")
            if len(self.warnings) > 20:
                print(f"  ... and {len(self.warnings) - 20} more warnings")

        if self.mismatches:
            print(f"\nMismatches found ({len(self.mismatches)}):")
            for mismatch in self.mismatches[:30]:
                print(f"  - {mismatch}")
            if len(self.mismatches) > 30:
                print(f"  ... and {len(self.mismatches) - 30} more mismatches")
            print("\n❌ RESULT: Directories are DIFFERENT")
            return False
        else:
            print("\n✅ RESULT: Directories are IDENTICAL")
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare two Simbox task output directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dir1 output/run1 --dir2 output/run2
  %(prog)s --dir1 output/run1 --dir2 output/run2 --tolerance 1e-5
  %(prog)s --dir1 output/run1 --dir2 output/run2 --image-psnr 40.0
        """,
    )
    parser.add_argument("--dir1", type=str, required=True, help="Path to the first output directory")
    parser.add_argument("--dir2", type=str, required=True, help="Path to the second output directory")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Numerical tolerance for floating point comparisons (default: 1e-6)",
    )
    parser.add_argument(
        "--image-psnr",
        type=float,
        default=37.0,
        help="PSNR threshold (dB) for considering images as matching (default: 37.0)",
    )

    args = parser.parse_args()

    comparator = SimboxComparator(args.dir1, args.dir2, tolerance=args.tolerance, image_psnr_threshold=args.image_psnr)

    success = comparator.compare()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
