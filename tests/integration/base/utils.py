"""
Utility functions for integration tests, including centralized seed setting.
"""

import random

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import open3d as o3d
except ImportError:
    o3d = None


def set_test_seeds(seed):
    """
    Set random seeds for all relevant libraries to ensure reproducible results.

    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (if available)
    - Open3D (if available)
    - PyTorch CUDA (if available)
    - PyTorch CUDNN settings for deterministic behavior

    Args:
        seed (int): The seed value to use for all random number generators
    """
    # Set Python's built-in random seed
    random.seed(seed)

    # Set NumPy seed
    np.random.seed(seed)

    # Set PyTorch seeds if available
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Configure CUDNN for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set Open3D seed if available
    if o3d is not None:
        o3d.utility.random.seed(seed)
