ROBOT_DICT = {}


def register_robot(target_class):
    """Register a robot class in the global ROBOT_DICT."""
    key = target_class.__name__
    ROBOT_DICT[key] = target_class
    return target_class
