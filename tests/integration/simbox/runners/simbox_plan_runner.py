"""
simbox plan runner for Isaac Sim subprocess execution.
This script runs in the Isaac Sim environment and is called by the main test.
"""

import sys

from tests.integration.base.test_harness import IntegrationTestHarness

# Add paths to sys.path
sys.path.append("./")
sys.path.append("./data_engine")
sys.path.append("./tests/integration")


def run_simbox_plan():
    """
    Run the simbox plan test in Isaac Sim environment.
    """
    harness = IntegrationTestHarness(config_path="configs/simbox/de_plan_template.yaml", seed=42, random_num=1)

    # Run the pipeline and validate output
    harness.run_test_end_to_end(min_output_files=1)

    print("✓ simbox plan test completed successfully")


if __name__ == "__main__":
    run_simbox_plan()
