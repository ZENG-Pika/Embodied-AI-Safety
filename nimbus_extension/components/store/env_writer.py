import json
import os
from datetime import datetime

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

    def flush_failure_to_disk(self, task, scene_name, attempt_count):
        log_dir = os.path.join(self.failure_output_dir, scene_name)
        os.makedirs(log_dir, exist_ok=True)
        recorded_length = self._recorded_length(task)
        task.length = recorded_length
        manifest = {
            "status": "failed",
            "attempt_count": attempt_count,
            "recorded_frames": recorded_length,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            self.logger.info(f"Saving failed attempt {attempt_count} in {log_dir}")
            if recorded_length > 0:
                task.save(log_dir)
        except Exception as exc:
            manifest["save_error"] = str(exc)
            self.logger.exception(f"Failed to save failed attempt data for scene {scene_name}: {exc}")
        with open(os.path.join(log_dir, "failure_manifest.json"), "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
        self.logger.info(f"Saved failed attempt metadata and {recorded_length} recorded frames in {log_dir}")
        return recorded_length

    def flush_to_disk(self, task, scene_name, seq, obs):
        try:
            scene_name = self.scene.name
            if obs is not None and self.obs_output_dir is not None:
                log_dir = os.path.join(self.obs_output_dir, scene_name)
                self.logger.info(f"Try to save obs in {log_dir}")
                length = task.save(log_dir)
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
