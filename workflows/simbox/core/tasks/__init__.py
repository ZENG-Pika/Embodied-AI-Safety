"""Tasks module initialization."""

from core.tasks.base_task import TASK_DICT

from .banana import BananaBaseTask

# Explicitly declare the public interface
__all__ = [
    "BananaBaseTask",
    "get_task_cls",
    "get_task_dict",
]


def get_task_cls(category_name):
    """Get task class by category name."""
    return TASK_DICT[category_name]


def get_task_dict():
    """Get task dictionary."""
    return TASK_DICT
