from omegaconf import DictConfig, ListConfig, OmegaConf


class AttrDict(dict):
    """A dict subclass that supports both task['key'] and task.key access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def _to_attr_dict(obj):
    if isinstance(obj, dict):
        return AttrDict({k: _to_attr_dict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attr_dict(i) for i in obj]
    return obj


class TaskConfigParser:
    """Shared utilities for workflow configuration parsing."""

    def __init__(self, task_cfg_path: str):
        self.task_cfg_path = task_cfg_path

    def parse_tasks(self):
        yaml_conf = OmegaConf.load(self.task_cfg_path)
        task_cfgs = []
        assert "tasks" in yaml_conf, f"Expected 'tasks' key in the task configuration file: {self.task_cfg_path}"
        for task in yaml_conf["tasks"]:
            if isinstance(task, (DictConfig, ListConfig)):
                cfg = OmegaConf.to_container(task, resolve=True)
            else:
                cfg = task
            task_cfgs.append(_to_attr_dict(cfg))
        return task_cfgs
