"""
simbox render runner for Isaac Sim subprocess execution.
This script runs in the Isaac Sim environment and is called by the main test.
"""

import sys

from tests.integration.base.test_harness import IntegrationTestHarness

# Add paths to sys.path
sys.path.append("./")
sys.path.append("./data_engine")
sys.path.append("./tests/integration")


def run_simbox_render():
    """
    Run the simbox render test in Isaac Sim environment.
    """
    harness = IntegrationTestHarness(config_path="configs/simbox/de_render_template.yaml", seed=42)

    # Run the pipeline and validate output
    harness.run_test_end_to_end(min_output_files=1)

    print("✓ simbox render test completed successfully")


if __name__ == "__main__":
    run_simbox_render()
