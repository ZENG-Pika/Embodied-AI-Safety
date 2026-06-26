import json
import pathlib

import numpy as np
import numpydantic
import pydantic


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results

class OptimizedRunningStats:
    def __init__(self, num_quantile_bins=1000):  # 减少bin数量
        self._count = 0
        self._sum = None
        self._sum_sq = None
        self._min = None
        self._max = None
        self._all_samples = []  # 用于存储采样数据
        self._sample_rate = 0.01  # 1%采样率
        self._num_quantile_bins = num_quantile_bins

    def update(self, batch: np.ndarray) -> None:
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements = batch.shape[0]
        
        # 更新基本统计量（向量化）
        if self._count == 0:
            self._sum = np.sum(batch, axis=0, dtype=np.float64)
            self._sum_sq = np.sum(batch**2, axis=0, dtype=np.float64)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
        else:
            self._sum += np.sum(batch, axis=0, dtype=np.float64)
            self._sum_sq += np.sum(batch**2, axis=0, dtype=np.float64)
            self._min = np.minimum(self._min, np.min(batch, axis=0))
            self._max = np.maximum(self._max, np.max(batch, axis=0))
        
        # 随机采样用于分位数计算（避免存储所有数据）
        if np.random.random() < self._sample_rate:
            sample_idx = np.random.randint(0, num_elements, size=min(100, num_elements))
            self._all_samples.append(batch[sample_idx])
        
        self._count += num_elements

    def get_statistics(self):
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")
        
        # 计算均值和标准差
        mean = self._sum / self._count
        variance = (self._sum_sq / self._count) - mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        
        # 基于采样数据计算分位数
        if self._all_samples:
            all_sampled = np.concatenate(self._all_samples, axis=0)
            q01 = np.quantile(all_sampled, 0.01, axis=0)
            q99 = np.quantile(all_sampled, 0.99, axis=0)
        else:
            q01 = np.zeros_like(mean)
            q99 = np.zeros_like(mean)
        
        return NormStats(mean=mean, std=stddev, q01=q01, q99=q99)
        
class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
