"""Objects module initialization."""

from core.objects.base_object import OBJECT_DICT

from .articulated_object import ArticulatedObject
from .conveyor_object import ConveyorObject
from .geometry_object import GeometryObject
from .plane_object import PlaneObject
from .rigid_object import RigidObject
from .shape_object import ShapeObject
from .xform_object import XFormObject

# Explicitly declare the public interface
__all__ = [
    "ArticulatedObject",
    "ConveyorObject",
    "GeometryObject",
    "PlaneObject",
    "RigidObject",
    "ShapeObject",
    "XFormObject",
    "get_object_cls",
    "get_object_dict",
]


def get_object_cls(category_name):
    """Get object class by category name."""
    return OBJECT_DICT[category_name]


def get_object_dict():
    """Get object dictionary."""
    return OBJECT_DICT
