import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path


def _load_env_writer():
    nimbus = types.ModuleType("nimbus")
    components = types.ModuleType("nimbus.components")
    store = types.ModuleType("nimbus.components.store")

    class BaseWriter:
        def __init__(
            self,
            data_iter,
            seq_output_dir,
            obs_output_dir,
            batch_async=True,
            async_threshold=1,
            batch_size=1,
            failure_output_dir=None,
            max_attempts=None,
        ):
            self.data_iter = data_iter
            self.seq_output_dir = seq_output_dir
            self.obs_output_dir = obs_output_dir
            self.failure_output_dir = failure_output_dir
            self.max_attempts = max_attempts
            self.logger = types.SimpleNamespace(
                info=lambda *args, **kwargs: None,
                exception=lambda *args, **kwargs: None,
            )

    store.BaseWriter = BaseWriter
    sys.modules.update(
        {
            "nimbus": nimbus,
            "nimbus.components": components,
            "nimbus.components.store": store,
        }
    )

    source = (
        Path(__file__).parents[2]
        / "nimbus_extension"
        / "components"
        / "store"
        / "env_writer.py"
    )
    spec = importlib.util.spec_from_file_location("semantic_env_writer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EnvWriter


def _episode(root, label=None):
    episode = root / "robot" / "Task" / "collect" / "episode"
    episode.mkdir(parents=True)
    if label is not None:
        (episode / "sim_labels.json").write_text(
            json.dumps({"task_labels": {"task_semantic_success": label}}),
            encoding="utf-8",
        )
    return episode


def test_routes_semantically_successful_failure_attempt_to_output():
    EnvWriter = _load_env_writer()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output = root / "output"
        failure = root / "failure_output"
        writer = EnvWriter(None, output_dir=str(output), failure_output_dir=str(failure))
        task = types.SimpleNamespace(_safety_eval_enabled=True)
        source = failure / "task"
        episode = _episode(source, True)
        task._last_saved_episode_dirs = [episode]

        classification, semantic_success = writer._route_saved_episode(
            task, str(source), "task"
        )

        assert (classification, semantic_success) == ("semantic_success", True)
        assert (output / "task" / "robot" / "Task" / "collect" / "episode").is_dir()
        assert not episode.exists()


def test_routes_semantically_failed_output_to_failure_output():
    EnvWriter = _load_env_writer()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output = root / "output"
        failure = root / "failure_output"
        writer = EnvWriter(None, output_dir=str(output), failure_output_dir=str(failure))
        task = types.SimpleNamespace(_safety_eval_enabled=True)
        source = output / "task"
        episode = _episode(source, False)
        task._last_saved_episode_dirs = [episode]

        classification, semantic_success = writer._route_saved_episode(
            task, str(source), "task"
        )

        assert (classification, semantic_success) == ("semantic_failure", False)
        assert (failure / "task" / "robot" / "Task" / "collect" / "episode").is_dir()
        assert not episode.exists()


def test_routes_missing_semantic_label_as_unclassified():
    EnvWriter = _load_env_writer()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output = root / "output"
        failure = root / "failure_output"
        writer = EnvWriter(None, output_dir=str(output), failure_output_dir=str(failure))
        task = types.SimpleNamespace(_safety_eval_enabled=True)
        source = output / "task"
        episode = _episode(source)
        task._last_saved_episode_dirs = [episode]

        classification, semantic_success = writer._route_saved_episode(
            task, str(source), "task"
        )

        assert (classification, semantic_success) == ("unclassified", None)
        assert (failure / "task" / "robot" / "Task" / "collect" / "episode").is_dir()
        assert not episode.exists()
