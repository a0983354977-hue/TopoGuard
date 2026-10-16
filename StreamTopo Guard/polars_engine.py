"""
Polars Data Pipeline Engine
-----------------------------
High-performance data processing pipeline for TopoGuard.
Responsibilities:
- Sliding window batch processing
- Rolling statistics (Rolling Baselines)
- Memory-efficient LazyFrame streaming
- Output NumPy matrices for TDA consumption

STRICT CONSTRAINTS:
- Only data processing and cleaning
- NO mathematical models or anomaly detection
- NO business logic
- Single-directional flow: Input -> Polars -> NumPy Output
"""

import polars as pl
import numpy as np
from typing import Callable, Optional, List, Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolarsDataPipeline:
    """
    High-performance data pipeline using Polars LazyFrame.
    Optimized for high-frequency streaming data (10k+ events/sec).
    """

    def __init__(
        self,
        window_size: int = 1000,
        stride: int = 500,
        max_memory_mb: float = 512
    ):
        """
        Initialize Polars data pipeline.

        Args:
            window_size: Number of samples per sliding window
            stride: Step size between windows (overlap = window_size - stride)
            max_memory_mb: Approximate cap in MB for retained Polars column data
        """
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if stride < 1:
            raise ValueError("stride must be positive")
        if max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")

        self.window_size = window_size
        self.stride = stride
        self.max_memory_mb = max_memory_mb
        self._max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._buffer_row_limit = max(
            1,
            min(window_size * 10, self._max_memory_bytes // 8),
        )
        self._lazy_frame: Optional[pl.LazyFrame] = None
        self._buffer: List[Dict[str, Any]] = []
        self._feature_names: Optional[Tuple[str, ...]] = None

    def ingest_stream(self, stream_data: List[Dict[str, Any]]) -> None:
        """
        Ingest streaming data into internal buffer.

        Args:
            stream_data: List of event dictionaries (e.g., from eBPF collector)
        """
        if not stream_data:
            return

        feature_names = tuple(stream_data[0].keys())
        if not feature_names:
            raise ValueError("Input rows must contain at least one feature")
        if self._feature_names is not None and set(feature_names) != set(self._feature_names):
            raise ValueError("Input feature schema changed during streaming")

        expected_names = self._feature_names or feature_names
        if self._feature_names is None:
            bytes_per_row = len(expected_names) * np.dtype(np.float64).itemsize
            if bytes_per_row > self._max_memory_bytes:
                raise MemoryError(
                    "A single Polars row exceeds max_memory_mb; refusing to exceed the configured limit"
                )
            self._buffer_row_limit = max(
                1,
                min(
                    self.window_size * 10,
                    self._max_memory_bytes // bytes_per_row,
                ),
            )

        for row in stream_data:
            if set(row) != set(expected_names):
                raise ValueError("Input feature schema changed within a batch")
            try:
                values = [float(row[name]) for name in expected_names]
            except (TypeError, ValueError) as exc:
                raise ValueError("Input features must be numeric") from exc
            if not np.isfinite(values).all():
                raise ValueError("Input features must be finite")

        self._feature_names = expected_names
        for row in stream_data:
            normalized = {name: float(row[name]) for name in expected_names}
            self._buffer.append(normalized)
            if len(self._buffer) >= self._buffer_row_limit:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffer to LazyFrame and clear buffer."""
        if not self._buffer:
            return

        schema = {name: pl.Float64 for name in self._feature_names or ()}
        new_frame = pl.DataFrame(self._buffer, schema=schema, strict=True)
        if self._lazy_frame is None:
            frame = new_frame
        else:
            frame = pl.concat([self._lazy_frame.collect(), new_frame])

        if frame.height > self.window_size:
            frame = frame.tail(self.window_size)
        frame = self._fit_memory_budget(frame)
        self._buffer.clear()
        self._lazy_frame = frame.lazy()

    def _fit_memory_budget(self, frame: pl.DataFrame) -> pl.DataFrame:
        estimated_bytes = frame.estimated_size()
        if estimated_bytes <= self._max_memory_bytes:
            return frame

        original_height = frame.height
        while estimated_bytes > self._max_memory_bytes:
            if frame.height <= 1:
                raise MemoryError(
                    "A single Polars row exceeds max_memory_mb; refusing to exceed the configured limit"
                )
            keep_rows = max(
                1,
                min(
                    frame.height - 1,
                    int(frame.height * self._max_memory_bytes / estimated_bytes),
                ),
            )
            frame = frame.tail(keep_rows)
            estimated_bytes = frame.estimated_size()

        logger.warning(
            "Polars memory limit reached; retained %s of %s rows",
            frame.height,
            original_height,
        )
        return frame

    def process(
        self,
        raw_stream: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Process streaming data through Polars pipeline.
        Returns NumPy matrix ready for TDA consumption.

        Args:
            raw_stream: Optional raw stream data (ingests if provided)

        Returns:
            NumPy matrix (n_samples, n_features)
        """
        if raw_stream is not None:
            self.ingest_stream(raw_stream)

        self._flush_buffer()

        if self._lazy_frame is None:
            logger.warning("No data available, returning empty matrix")
            return np.zeros((0, 0))

        # Collect LazyFrame to DataFrame
        df = self._lazy_frame.collect()

        if df.height == 0:
            return np.zeros((0, 0))

        # Convert to NumPy
        matrix = df.to_numpy().astype(np.float64, copy=False)

        # Apply rolling statistics for baseline
        matrix = self._apply_rolling_statistics(matrix)
        if not np.isfinite(matrix).all():
            raise ValueError("Processed features contain non-finite values")

        return matrix

    def _apply_rolling_statistics(self, data: np.ndarray) -> np.ndarray:
        """
        Apply rolling statistics for baseline computation.
        Uses Polars for efficient rolling operations.

        Args:
            data: Input NumPy matrix

        Returns:
            Matrix with rolling statistics features appended
        """
        if data.shape[0] < self.window_size:
            # Not enough data for rolling stats, return as-is
            return data

        data = data.astype(np.float64, copy=False)
        df = pl.DataFrame(data)

        rolling_mean = df.select(
            self._rolling_expression("rolling_mean")
        ).to_numpy()
        rolling_std = df.select(
            self._rolling_expression("rolling_std")
        ).fill_null(0.0).to_numpy()

        return np.hstack([data, rolling_mean, rolling_std])

    def _rolling_expression(self, operation: str):
        rolling = getattr(pl.all(), operation)
        try:
            return rolling(window_size=self.window_size, min_samples=1)
        except TypeError:
            return rolling(window_size=self.window_size, min_periods=1)

    def process_sliding_windows(
        self,
        process_func: Callable[[np.ndarray], Any],
        raw_stream: Optional[List[Dict[str, Any]]] = None
    ) -> List[Tuple[int, int, Any]]:
        """
        Process data in sliding windows with external function.
        Optimized for TDA computation on each window.

        Args:
            process_func: Function to apply to each window (receives NumPy matrix)
            raw_stream: Optional raw stream data

        Returns:
            List of (start_idx, end_idx, result) tuples
        """
        if raw_stream is not None:
            self.ingest_stream(raw_stream)
            self._flush_buffer()
            n_samples = len(raw_stream)
            source_frame = None
        else:
            self._flush_buffer()
            if self._lazy_frame is None:
                return []
            matrix = self._lazy_frame.collect().to_numpy()
            n_samples = matrix.shape[0]
            source_frame = None

        results = []

        for start_idx in range(0, n_samples - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            if raw_stream is not None:
                source_frame = pl.DataFrame(raw_stream[start_idx:end_idx])
                if source_frame.estimated_size() > self._max_memory_bytes:
                    raise MemoryError(
                        "A sliding window exceeds max_memory_mb; refusing unbounded processing"
                    )
                window = source_frame.to_numpy()
            else:
                window = matrix[start_idx:end_idx]

            # Apply rolling statistics to window
            window = self._apply_rolling_statistics(window)

            # Process window with external function (e.g., TDA)
            result = process_func(window)
            results.append((start_idx, end_idx, result))

        return results

    def get_window_count(self, n_samples: int) -> int:
        """Calculate number of windows for given data size."""
        return max(0, (n_samples - self.window_size) // self.stride + 1)

    def reset(self) -> None:
        """Reset pipeline state."""
        self._lazy_frame = None
        self._buffer.clear()
        self._feature_names = None
        logger.info("Polars pipeline reset")

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        buffer_size = len(self._buffer)
        frame = self._lazy_frame.collect() if self._lazy_frame is not None else None
        frame_rows = frame.height if frame is not None else 0

        return {
            'buffer_size': buffer_size,
            'frame_rows': frame_rows,
            'window_size': self.window_size,
            'stride': self.stride,
            'memory_usage_mb': self._estimate_memory_mb(frame)
        }

    def _estimate_memory_mb(self, frame: Optional[pl.DataFrame] = None) -> float:
        """Estimate Polars column data, excluding Python objects and transient copies."""
        if frame is None and self._lazy_frame is not None:
            frame = self._lazy_frame.collect()
        memory_bytes = frame.estimated_size() if frame is not None else 0
        if self._buffer:
            memory_bytes += pl.DataFrame(self._buffer).estimated_size()
        return memory_bytes / (1024 * 1024)


class RollingBaseline:
    """
    Rolling baseline computation using Polars.
    Replaces AdaptiveBaseline for high-frequency scenarios.
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize rolling baseline.

        Args:
            window_size: Rolling window size for baseline
        """
        self.window_size = window_size
        self._history: Optional[pl.DataFrame] = None

    def update(self, scores: np.ndarray) -> Dict[str, float]:
        """
        Update baseline with new scores.

        Args:
            scores: Array of anomaly scores

        Returns:
            Dict with current baseline statistics
        """
        df = pl.DataFrame({'score': scores})

        if self._history is None:
            self._history = df
        else:
            self._history = pl.concat([self._history, df])

        # Keep only recent history
        if self._history.height > self.window_size * 10:
            self._history = self._history.tail(self.window_size * 10)

        # Compute rolling statistics
        rolling_mean = self._history.select(
            self._rolling_expression('rolling_mean')
        ).to_numpy().flatten()

        rolling_std = self._history.select(
            self._rolling_expression('rolling_std')
        ).fill_null(0.0).to_numpy().flatten()

        current_mean = float(rolling_mean[-1]) if len(rolling_mean) > 0 else 0.0
        current_std = float(rolling_std[-1]) if len(rolling_std) > 0 else 0.0

        return {
            'mean': current_mean,
            'std': current_std,
            'threshold': current_mean + 3.0 * current_std
        }

    def _rolling_expression(self, operation: str):
        rolling = getattr(pl.col('score'), operation)
        try:
            return rolling(window_size=self.window_size, min_samples=1)
        except TypeError:
            return rolling(window_size=self.window_size, min_periods=1)

    def reset(self) -> None:
        """Reset baseline history."""
        self._history = None
