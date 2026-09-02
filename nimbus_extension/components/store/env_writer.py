import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from nimbus.components.store import BaseWriter


class EnvWriter(BaseWriter):
    """
    A writer that saves generated sequences and observations to disk for environment simulations.
    This class extends the BaseWriter to provide specific implementations for handling data related
    to environment simulations.

    Args:
        data_iter (Iterator): An iterator that provides data to be written, typically containing scenes,
            sequences, and observations.
        seq_output_dir (str): The directory where generated sequences will be saved. Can be None
            if sequence output is not needed.
        obs_output_dir (str): The directory where generated observations will be saved. Can be None
            if observation output is not needed.
        batch_async (bool): If True, the writer will use asynchronous batch writing to improve performance
            when handling large amounts of data. Default is True.
        async_threshold (int): The maximum number of asynchronous write operations that can be in progress
            at the same time. If the threshold is reached, the writer will wait for the oldest operation
            to complete before starting a new one. Default is 1.
        batch_size (int): The number of data items to write in each batch when using asynchronous writing.
            Default is 1, and it will be capped at 8 to prevent potential issues with too many concurrent operations.
    """

    def __init__(
        self,
        data_iter,
        seq_output_dir=None,
        output_dir=None,
        batch_async=True,
        async_threshold=1,
        batch_size=1,
        failure_output_dir=None,
        max_attempts=None,
    ):
        super().__init__(
            data_iter,
            seq_output_dir,
            output_dir,
            batch_async=batch_async,
            async_threshold=async_threshold,
            batch_size=batch_size,
            failure_output_dir=failure_output_dir,
            max_attempts=max_attempts,
        )

    @staticmethod
    def _recorded_length(task):
        logger = getattr(task, "logger", None)
        if logger is None:
            return 0
        lengths = []
        for attr in (
            "proprio_data_logger",
            "action_data_logger",
            "object_data_logger",
            "color_image_logger",
        ):
            robot_data = getattr(logger, attr, {})
            for values_by_key in robot_data.values():
                for values in values_by_key.values():
                    try:
                        lengths.append(len(values))
                    except TypeError:
                        pass
        return max(lengths, default=0)

    @staticmethod
    def _saved_episode_dirs(task, log_dir):
        """Return episode directories created by the most recent task.save()."""
        recorded = getattr(task, "_last_saved_episode_dirs", None)
        if recorded:
            paths = [Path(path) for path in recorded]
            existing = [path for path in paths if path.is_dir()]
            if existing:
                return existing

        # Compatibility fallback for workflows that do not expose the paths.
        root = Path(log_dir)
        candidates = set()
        for marker in ("sim_labels.json", "meta_info.pkl"):
            for path in root.rglob(marker):
                candidates.add(path.parent)
        return sorted(candidates)

    @staticmethod
    def _semantic_success(task, episode_dirs):
        """Read task semantic success from the saved safety labels.

        None means that the safety pipeline did not produce a usable label and
        is intentionally kept distinct from False.
        """
        if not bool(getattr(task, "_safety_eval_enabled", False)):
            return None
        values = []
        for episode_dir in episode_dirs:
            label_path = episode_dir / "sim_labels.json"
            if not label_path.exists():
                for candidate in episode_dir.rglob("sim_labels.json"):
                    label_path = candidate
                    break
            if not label_path.exists():
                continue
            try:
                with label_path.open("r", encoding="utf-8") as stream:
                    labels = json.load(stream)
                value = labels.get("task_labels", {}).get("task_semantic_success")
            except (OSError, TypeError, ValueError):
                value = None
            if isinstance(value, bool):
                values.append(value)

        if any(value is False for value in values):
            return False
        if any(value is True for value in values):
            return True
        return None

    def _route_saved_episode(self, task, source_log_dir, scene_name):
        """Route a saved episode using task semantic success."""
        episode_dirs = self._saved_episode_dirs(task, source_log_dir)
        semantic_success = self._semantic_success(task, episode_dirs)

        if semantic_success is True:
            destination_root = self.obs_output_dir
            classification = "semantic_success"
        elif semantic_success is False or (
            semantic_success is None
            and bool(getattr(task, "_safety_eval_enabled", False))
        ):
            destination_root = self.failure_output_dir
            classification = (
                "semantic_failure" if semantic_success is False else "unclassified"
            )
        else:
            destination_root = None
            classification = "writer_success"

        source_root = Path(source_log_dir)
        if destination_root and Path(destination_root).resolve() != source_root.parent.resolve():
            destination_root = Path(destination_root)
            for episode_dir in episode_dirs:
                try:
                    relative = episode_dir.relative_to(source_root)
                except ValueError:
                    continue
                destination = destination_root / scene_name / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.move(str(episode_dir), str(destination))

        return classification, semantic_success

    def _write_semantic_manifest(
        self,
        scene_name,
        classification,
        semantic_success,
        recorded_frames,
        writer_status,
        attempt_count=None,
    ):
        if self.failure_output_dir is None:
            return
        manifest_dir = Path(self.failure_output_dir) / scene_name
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "failed" if classification != "semantic_success" else "success",
            "classification": classification,
            "writer_status": writer_status,
            "semantic_success": semantic_success,
            "attempt_count": attempt_count,
            "recorded_frames": int(recorded_frames),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with (manifest_dir / "failure_manifest.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(manifest, stream, indent=2)

    def flush_failure_to_disk(self, task, scene_name, attempt_count):
        log_dir = os.path.join(self.failure_output_dir, scene_name)
        os.makedirs(log_dir, exist_ok=True)
        recorded_length = self._recorded_length(task)
        task.length = recorded_length
        classification = "unclassified"
        semantic_success = None
        try:
            self.logger.info(f"Saving failed attempt {attempt_count} in {log_dir}")
            if recorded_length > 0:
                task.save(log_dir)
                classification, semantic_success = self._route_saved_episode(
                    task, log_dir, scene_name
                )
        except Exception as exc:
            self.logger.exception(f"Failed to save failed attempt data for scene {scene_name}: {exc}")
            classification = "unclassified"
        if classification != "semantic_success":
            self._write_semantic_manifest(
                scene_name,
                classification,
                semantic_success,
                recorded_length,
                writer_status="failed",
                attempt_count=attempt_count,
            )
        self.logger.info(f"Saved failed attempt metadata and {recorded_length} recorded frames in {log_dir}")
        return recorded_length

    def flush_to_disk(self, task, scene_name, seq, obs):
        try:
            if obs is not None and self.obs_output_dir is not None:
                log_dir = os.path.join(self.obs_output_dir, scene_name)
                self.logger.info(f"Try to save obs in {log_dir}")
                length = task.save(log_dir)
                classification, semantic_success = self._route_saved_episode(
                    task, log_dir, scene_name
                )
                if classification not in ("semantic_success", "writer_success"):
                    self._write_semantic_manifest(
                        scene_name,
                        classification,
                        semantic_success,
                        length,
                        writer_status="success",
                        attempt_count=getattr(self, "total_case", None),
                    )
                self.logger.info(f"Saved {length} obs output saved in {log_dir}")
            elif seq is not None and self.seq_output_dir is not None:
                log_dir = os.path.join(self.seq_output_dir, scene_name)
                self.logger.info(f"Try to save seq in {log_dir}")
                length = task.save_seq(log_dir)
                self.logger.info(f"Saved {length} seq output saved in {log_dir}")
            else:
                self.logger.info("Skip this storage")
            return length
        except Exception as e:
            self.logger.info(f"Failed to save data for scene {scene_name}: {e}")
            raise e
