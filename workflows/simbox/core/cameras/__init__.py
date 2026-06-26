"""Camera module initialization."""

from core.cameras.base_camera import CAMERA_DICT

from .custom_camera import CustomCamera

__all__ = [
    "CustomCamera",
    "get_camera_cls",
    "get_camera_dict",
]


def get_camera_cls(category_name):
    """Get camera class by category name."""
    return CAMERA_DICT[category_name]


def get_camera_dict():
    """Get camera dictionary."""
    return CAMERA_DICT
