OBJECT_DICT = {}


def register_object(target_class):
    key = target_class.__name__
    # assert key not in OBJECT_DICT
    OBJECT_DICT[key] = target_class
    return target_class
