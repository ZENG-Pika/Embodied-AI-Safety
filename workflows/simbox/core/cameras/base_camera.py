CAMERA_DICT = {}


def register_camera(target_class):
    # key = "_".join(re.sub(r"([A-Z0-9])", r" \1", target_class.__name__).split()).lower()
    key = target_class.__name__
    assert key not in CAMERA_DICT
    CAMERA_DICT[key] = target_class
    return target_class
