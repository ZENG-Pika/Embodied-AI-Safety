"""Template-based controllers. Import subclasses to register them with CONTROLLER_DICT."""

from core.controllers.base_controller import CONTROLLER_DICT
from core.controllers.fr3_controller import FR3Controller
from core.controllers.frankarobotiq85_controller import FrankaRobotiq85Controller
from core.controllers.genie1_controller import Genie1Controller
from core.controllers.lift2_controller import Lift2Controller
from core.controllers.splitaloha_controller import SplitAlohaController
from core.controllers.template_controller import TemplateController

__all__ = [
    "TemplateController",
    "FR3Controller",
    "FrankaRobotiq85Controller",
    "Genie1Controller",
    "Lift2Controller",
    "SplitAlohaController",
]


def get_controller_cls(category_name):
    """Get controller class by category name."""
    return CONTROLLER_DICT[category_name]


def get_controller_dict():
    """Get controller dictionary."""
    return CONTROLLER_DICT
