"""
Integration Test Harness base class that encapsulates common test logic.
Reduces boilerplate code and provides a consistent testing interface.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .utils import set_test_seeds

# Add paths to sys.path for proper imports
sys.path.append("./")
sys.path.append("./data_engine")


# Import data_engine modules only when needed to avoid import errors during framework testing
def _import_data_engine_modules():
    """Import data_engine modules when they are actually needed."""
    try:
        from nimbus import run_data_engine
        from nimbus.utils.config_processor import ConfigProcessor
        from nimbus.utils.utils import init_env

        return init_env, run_data_engine, ConfigProcessor
    except ImportError as e:
        raise ImportError(f"Failed to import data_engine modules. Ensure dependencies are installed: {e}")


class IntegrationTestHarness:
    """
    Base class for integration tests that provides common functionality.

    This class encapsulates:
    - Configuration loading and processing
    - Output directory cleanup
    - Test pipeline execution (direct or subprocess)
    - Output validation
    - Data comparison with reference
    """

    def __init__(self, config_path: str, seed: int = 42, load_num: int = 0, random_num: int = 0, episodes: int = 0):
        """
        Initialize the test harness with configuration and seed.

        Args:
            config_path: Path to the test configuration YAML file
            seed: Random seed for reproducible results (default: 42)
        """
        self.config_path = config_path
        self.seed = seed
        self.load_num = load_num
        self.random_num = random_num
        self.episodes = episodes
        self.output_dir = None

        # Import data_engine modules
        self._init_env, self._run_data_engine, self._ConfigProcessor = _import_data_engine_modules()
        self.processor = self._ConfigProcessor()

        # Initialize environment (same as launcher.py)
        self._init_env()

        self.config = self.load_and_process_config() if config_path else None
        # Set random seeds for reproducibility
        self.modify_config()

        set_test_seeds(seed)

        from nimbus.utils.flags import set_debug_mode, set_random_seed

        set_debug_mode(True)  # Enable debug mode for better error visibility during tests
        set_random_seed(seed)

    def modify_config(self):
        """Modify configuration parameters as needed before running the pipeline."""
        if self.config and "load_stage" in self.config:
            if "layout_random_generator" in self.config.load_stage:
                if "args" in self.config.load_stage.layout_random_generator:
                    if self.random_num > 0:
                        self.config.load_stage.layout_random_generator.args.random_num = self.random_num
                    if "input_dir" in self.config.load_stage.layout_random_generator.args:
                        if "simbox" in self.config.name:
                            input_path = (
                                "/shared/smartbot_new/zhangyuchang/CI/manip/"
                                "simbox/simbox_plan_ci/seq_path/BananaBaseTask/plan"
                            )
                            self.config.load_stage.layout_random_generator.args.input_dir = input_path
        if self.config and "plan_stage" in self.config:
            if "seq_planner" in self.config.plan_stage:
                if "args" in self.config.plan_stage.seq_planner:
                    if self.episodes > 0:
                        self.config.plan_stage.seq_planner.args.episodes = self.episodes
                        if self.load_num > 0:
                            self.config.load_stage.scene_loader.args.load_num = self.load_num
        if self.config and "name" in self.config:
            self.config.name = self.config.name + "_ci"
        if self.config and "store_stage" in self.config:
            if hasattr(self.config.store_stage.writer.args, "obs_output_dir"):
                self.config.store_stage.writer.args.obs_output_dir = f"output/{self.config.name}/obs_path/"
            if hasattr(self.config.store_stage.writer.args, "seq_output_dir"):
                self.config.store_stage.writer.args.seq_output_dir = f"output/{self.config.name}/seq_path/"
            if hasattr(self.config.store_stage.writer.args, "output_dir"):
                self.config.store_stage.writer.args.output_dir = f"output/{self.config.name}/"
            self._extract_output_dir()

    def load_and_process_config(self) -> Dict[str, Any]:
        """
        Load and process the test configuration file.

        Returns:
            Processed configuration dictionary

        Raises:
            AssertionError: If config file does not exist
        """
        assert os.path.exists(self.config_path), f"Config file not found: {self.config_path}"

        self.config = self.processor.process_config(self.config_path)
        self.processor.print_final_config(self.config)

        # Extract output directory from config
        self._extract_output_dir()

        return self.config

    def _extract_output_dir(self):
        """Extract and expand the output directory path from config."""
        # Try navigation test output path
        output_dir = self.config.get("store_stage", {}).get("writer", {}).get("args", {}).get("seq_output_dir")

        # If not found, try common output_dir used in render configs
        if not output_dir:
            output_dir = self.config.get("store_stage", {}).get("writer", {}).get("args", {}).get("output_dir")

        # Process the output directory if found
        if output_dir and isinstance(output_dir, str):
            name = self.config.get("name", "test_output")
            output_dir = output_dir.replace("${name}", name)
            self.output_dir = os.path.abspath(output_dir)

    def cleanup_output_directory(self):
        """Clean up existing output directory if it exists."""
        if self.output_dir and os.path.exists(self.output_dir):
            # Use ignore_errors=True to handle NFS caching issues
            shutil.rmtree(self.output_dir, ignore_errors=True)
            # If directory still exists (NFS delay), try removing with onerror handler
            if os.path.exists(self.output_dir):

                def handle_remove_error(func, path, exc_info):  # pylint: disable=W0613
                    """Handle errors during removal, with retry for NFS issues."""
                    import time

                    time.sleep(0.1)  # Brief delay for NFS sync
                    try:
                        if os.path.isdir(path):
                            os.rmdir(path)
                        else:
                            os.remove(path)
                    except OSError:
                        pass  # Ignore if still fails

                shutil.rmtree(self.output_dir, onerror=handle_remove_error)
            print(f"Cleaned up existing output directory: {self.output_dir}")

    def run_data_engine_direct(self, **kwargs) -> None:
        """
        Run the test pipeline directly in the current process.

        Args:
            **kwargs: Additional arguments to pass to run_data_engine

        Raises:
            Exception: If pipeline execution fails
        """
        if not self.config:
            self.load_and_process_config()
        self.modify_config()
        self.cleanup_output_directory()

        try:
            self._run_data_engine(self.config, **kwargs)
        except Exception as e:
            raise e

    def run_data_engine_subprocess(
        self,
        runner_script: str,
        interpreter: str = "python",
        timeout: int = 1800,
        compare_output: bool = False,
        reference_dir: str = "",
        comparator: str = "simbox",
        comparator_args: Optional[Dict[str, Any]] = None,
    ) -> subprocess.CompletedProcess:
        """
        Run the test pipeline in a subprocess using a runner script.

        Args:
            runner_script: Path to the runner script to execute
            interpreter: Command to use for running the script (default: "python")
            timeout: Timeout in seconds for subprocess execution (default: 1800)
            compare_output: Whether to compare generated output with a reference directory
            reference_dir: Path to reference directory containing meta_info.pkl and lmdb
            comparator: Which comparator to use (default: "simbox")
            comparator_args: Optional extra arguments for comparator (e.g. {"tolerance": 1e-6, "image_psnr": 37.0})

        Returns:
            subprocess.CompletedProcess object with execution results

        Raises:
            subprocess.TimeoutExpired: If execution exceeds timeout
            AssertionError: If subprocess returns non-zero exit code
        """
        self.cleanup_output_directory()

        # Build command based on interpreter type
        if interpreter == "blenderproc":
            cmd = ["blenderproc", "run", runner_script]
            if os.environ.get("BLENDER_CUSTOM_PATH"):
                cmd.extend(["--custom-blender-path", os.environ["BLENDER_CUSTOM_PATH"]])
        elif interpreter.endswith(".sh"):
            # For scripts like /isaac-sim/python.sh
            cmd = [interpreter, runner_script]
        else:
            cmd = [interpreter, runner_script]

        if not self.output_dir:
            self._extract_output_dir()
        output_dir = self.output_dir

        print(f"Running command: {' '.join(cmd)}")
        print(f"Expected output directory: {output_dir}")

        try:
            result = subprocess.run(cmd, capture_output=False, text=True, timeout=timeout, check=False)

            # Print subprocess output for debugging
            if result.stdout:
                print("STDOUT:", result.stdout[-2000:])  # Last 2000 chars
            if result.stderr:
                print("STDERR:", result.stderr[-1000:])  # Last 1000 chars
            print("Return code:", result.returncode)

            if compare_output and result.returncode == 0:
                if not output_dir:
                    raise RuntimeError(
                        "Output directory not configured. Expected one of "
                        "store_stage.writer.args.(seq_output_dir|obs_output_dir|output_dir)."
                    )

                if output_dir:
                    for root, dirs, files in os.walk(output_dir):
                        if "lmdb" in dirs and "meta_info.pkl" in files:
                            output_dir = root
                            break

                if os.path.exists(reference_dir):
                    # Find the reference render directory
                    for root, dirs, files in os.walk(reference_dir):
                        if "lmdb" in dirs and "meta_info.pkl" in files:
                            reference_dir = root
                            break

                # Build comparator command according to requested comparator
                comp = (comparator or "simbox").lower()
                comparator_args = comparator_args or {}

                if comp == "simbox":
                    compare_cmd = [
                        "/isaac-sim/python.sh",
                        "tests/integration/data_comparators/simbox_comparator.py",
                        "--dir1",
                        output_dir,
                        "--dir2",
                        reference_dir,
                    ]
                    # Optional numeric/image thresholds
                    if "tolerance" in comparator_args and comparator_args["tolerance"] is not None:
                        compare_cmd += ["--tolerance", str(comparator_args["tolerance"])]
                    if "image_psnr" in comparator_args and comparator_args["image_psnr"] is not None:
                        compare_cmd += ["--image-psnr", str(comparator_args["image_psnr"])]

                    print(f"Running comparison: {' '.join(compare_cmd)}")
                    compare_result = subprocess.run(
                        compare_cmd, capture_output=True, text=True, timeout=600, check=False
                    )

                    print("Comparison STDOUT:")
                    print(compare_result.stdout)
                    print("Comparison STDERR:")
                    print(compare_result.stderr)

                    if compare_result.returncode != 0:
                        raise RuntimeError("Simbox comparison failed: outputs differ")

                    if "Successfully loaded data from both directories" in compare_result.stdout:
                        print("✓ Both output directories have valid structure (meta_info.pkl + lmdb)")

                    print("✓ Simbox render test completed with numeric-aligned comparison")

                else:
                    raise ValueError(f"Unknown comparator: {comp}. Use 'simbox'.")

            return result

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Test timed out after {timeout} seconds")

    def validate_output_generated(self, min_files: int = 1) -> bool:
        """
        Validate that output was generated in the expected directory.

        Args:
            min_files: Minimum number of files expected in output (default: 1)

        Returns:
            True if validation passes

        Raises:
            AssertionError: If output directory doesn't exist or has too few files
        """
        assert self.output_dir is not None, "Output directory not configured"
        assert os.path.exists(self.output_dir), f"Expected output directory was not created: {self.output_dir}"

        output_files = list(Path(self.output_dir).rglob("*"))
        assert (
            len(output_files) >= min_files
        ), f"Expected at least {min_files} files but found {len(output_files)} in: {self.output_dir}"

        print(f"✓ Pipeline completed successfully. Output generated in: {self.output_dir}")
        print(f"✓ Generated {len(output_files)} files/directories")
        return True

    def compare_with_reference(self, reference_dir: str, comparator_func, **comparator_kwargs) -> Tuple[bool, str]:
        """
        Compare generated output with reference data using provided comparator function.

        Args:
            reference_dir: Path to reference data directory
            comparator_func: Function to use for comparison
            **comparator_kwargs: Additional arguments for comparator function

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not os.path.exists(reference_dir):
            print(f"Reference directory not found, skipping comparison: {reference_dir}")
            return True, "Reference data not available"

        success, message = comparator_func(self.output_dir, reference_dir, **comparator_kwargs)

        if success:
            print(f"✓ Results match reference data: {message}")
        else:
            print(f"✗ Result comparison failed: {message}")

        return success, message

    def run_test_end_to_end(
        self,
        reference_dir: Optional[str] = None,
        comparator_func=None,
        comparator_kwargs: Optional[Dict] = None,
        min_output_files: int = 1,
        **pipeline_kwargs,
    ) -> bool:
        """
        Run a complete end-to-end test including comparison with reference.

        Args:
            reference_dir: Optional path to reference data for comparison
            comparator_func: Optional function for comparing results
            comparator_kwargs: Optional kwargs for comparator function
            min_output_files: Minimum expected output files
            **pipeline_kwargs: Additional arguments for pipeline execution

        Returns:
            True if test passes, False otherwise

        Raises:
            Exception: If any test step fails
        """
        # Load configuration
        self.load_and_process_config()
        # Run pipeline
        self.run_data_engine_direct(**pipeline_kwargs, master_seed=self.seed)

        # Validate output
        self.validate_output_generated(min_files=min_output_files)

        # Compare with reference if provided
        if reference_dir and comparator_func:
            comparator_kwargs = comparator_kwargs or {}
            success, message = self.compare_with_reference(reference_dir, comparator_func, **comparator_kwargs)
            assert success, f"Comparison with reference failed: {message}"

        return True
