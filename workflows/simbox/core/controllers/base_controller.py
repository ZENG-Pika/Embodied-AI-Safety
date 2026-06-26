CONTROLLER_DICT = {}


def register_controller(target_class):
    # key = "_".join(re.sub(r"([A-Z0-9])", r" \1", target_class.__name__).split()).lower()
    key = target_class.__name__
    assert key.endswith("Controller")
    key = key.removesuffix("Controller")
    # assert key not in CONTROLLER_DICT
    CONTROLLER_DICT[key] = target_class
    return target_class
