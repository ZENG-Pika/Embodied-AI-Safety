"""
Robot implementations using template-based approach.
All robots inherit from TemplateRobot and are configured via yaml files.
"""
from core.robots.base_robot import ROBOT_DICT

from core.robots.template_robot import TemplateRobot
from core.robots.fr3 import FR3
from core.robots.franka_robotiq85 import FrankaRobotiq85
from core.robots.genie1 import Genie1
from core.robots.lift2 import Lift2
from core.robots.split_aloha import SplitAloha

__all__ = [
    "TemplateRobot",
    "FR3",
    "FrankaRobotiq85",
    "Genie1",
    "Lift2",
    "SplitAloha",
]

def get_robot_cls(category_name):
    """Get robot class by category name."""
    return ROBOT_DICT[category_name]