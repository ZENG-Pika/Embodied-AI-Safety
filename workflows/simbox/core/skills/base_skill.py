import re
from abc import ABC

SKILL_DICT = {}


def register_skill(target_class):
    key = "_".join(re.sub(r"([A-Z0-9])", r" \1", target_class.__name__).split()).lower()
    # key = target_class.__name__
    # assert key not in SKILL_DICT
    SKILL_DICT[key] = target_class
    return target_class


class BaseSkill(ABC):
    def __init__(self):
        self.plan_flag = False

    def is_ready(self):
        return True

    def is_done(self):
        raise NotImplementedError

    def is_success(self):
        raise NotImplementedError

    def update(self):
        pass

    def is_feasible(self):
        return True

    def is_record(self):
        return True
