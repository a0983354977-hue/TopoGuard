import numpy as np
import logging
import time
from pathlib import Path
from typing import Callable, Any, Optional, List, Tuple, Dict
from collections import deque
from functools import lru_cache

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def _convert_to_dict_stream(raw_stream: List[Any] | np.ndarray) -> List[Dict[str, Any]]:
    if isinstance(raw_stream, np.ndarray):
        rows = raw_stream.reshape(-1, 1) if raw_stream.ndim == 1 else raw_stream
    elif not raw_stream:
        return []
    elif isinstance(raw_stream[0], dict):
        return list(raw_stream)
    elif isinstance(raw_stream[0], (int, float, np.number)):
        return [{"value": float(value)} for value in raw_stream]
    else:
        rows = np.asarray(raw_stream)
        if rows.ndim == 1:
            rows = rows.reshape(-1, 1)
    return [
        {f"feature_{index}": float(value) for index, value in enumerate(row)}
        for row in rows
    ]


# Import new components (lazy-loaded)
try:
    from polars_engine import PolarsDataPipeline
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    logging.warning("Polars not available")

try:
    from ebpf_collector import EBPFCollector
    EBPF_AVAILABLE = True
except ImportError:
    EBPF_AVAILABLE = False
    logging.warning("eBPF collector not available")

try:
    from hybrid_engine import HybridTdaGnnEngine
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    logging.warning("Hybrid engine not available")

try:
    from extension_plugins import Z3FormalVerificationPlugin
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logging.warning("Z3 plugin not available")


class HardCapGuard:
    DEFAULT_MAX_POINTS = 1000

    @staticmethod
    def safe_downsample(data: np.ndarray, max_points: int = DEFAULT_MAX_POINTS) -> np.ndarray:
        if max_points < 1:
            raise ValueError("max_points must be positive")
        if data is None:
            raise ValueError("Cannot downsample missing data")
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 0 or data.shape[0] == 0:
            raise ValueError("Cannot downsample empty data")

        if len(data) <= max_points:
            return data

        try:
            indices = np.random.choice(len(data), max_points, replace=False)
            downsampled = data[indices]
            logging.info(f"Downsampled from {len(data)} to {max_points} points")
            return downsampled
        except Exception as e:
            logging.error(f"Downsampling failed: {e}, using uniform binning")
            step = len(data) // max_points
            indices = np.arange(0, len(data), step)[:max_points]
            return data[indices]

    @staticmethod
    def validate_and_cap(data: np.ndarray, max_points: int = DEFAULT_MAX_POINTS,
                        max_dims: int = 100) -> np.ndarray:
        if data is None:
            raise ValueError("No data supplied")
        if max_points < 1 or max_dims < 1:
            raise ValueError("max_points and max_dims must be positive")

        try:
            data = np.asarray(data, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError("Input data must be numeric") from e

        if data.size == 0:
            raise ValueError("Input data is empty")
        if data.ndim == 0:
            raise ValueError("Input data must have at least one dimension")

        if data.ndim != 2:
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            else:
                data = data.reshape(data.shape[0], -1)

        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("Input data must contain at least one row and feature")
        if not np.isfinite(data).all():
            raise ValueError("Input data contains non-finite values")
        
        # Cap point count
        if data.shape[0] > max_points:
            data = HardCapGuard.safe_downsample(data, max_points)
        
        # Cap dimension count
        if data.shape[1] > max_dims:
            logging.warning(f"Too many dimensions {data.shape[1]}, truncating to {max_dims}")
            data = data[:, :max_dims]
        
        return data


# ============================================================================
# Defense 2: Adaptive Baseline (False Positive Prevention)
# ============================================================================

class AdaptiveBaseline:
    """
    Prevents false positive storms through adaptive baseline learning.
    Implements warm-up period and dynamic tolerance bands.
    """
    
    def __init__(self, window_size: int = 50, std_multiplier: float = 3.0):
        """
        Initialize adaptive baseline.
        
        Args:
            window_size: Number of samples for warm-up period
            std_multiplier: Standard deviation multiplier for threshold
        """
        self.window_size = window_size
        self.std_multiplier = std_multiplier
        self.history = deque(maxlen=window_size)
        self.is_warmed_up = False
        self.mean = 0.0
        self.std = 0.0
    
    def evaluate(self, current_score: float) -> tuple[bool, float, float]:
        """
        Evaluate if current score indicates true anomaly.
        
        Args:
            current_score: Current anomaly score to evaluate
        
        Returns:
            Tuple of (is_anomaly, threshold, z_score)
        """
        # Warm-up period: collect data without alerting
        if len(self.history) < self.window_size:
            self.history.append(current_score)
            self.is_warmed_up = False
            return False, 0.0, 0.0
        
        # Update baseline statistics
        self.mean = float(np.mean(self.history))
        self.std = float(np.std(self.history))
        
        # Add current score to history
        self.history.append(current_score)
        self.is_warmed_up = True
        
        # Calculate dynamic threshold
        threshold = self.mean + self.std_multiplier * self.std
        
        # Calculate z-score
        z_score = (current_score - self.mean) / (self.std + 1e-8)
        
        # Only alert if exceeds threshold
        is_anomaly = current_score > threshold
        
        return is_anomaly, threshold, z_score
    
    def reset(self):
        """Reset baseline history (use after system changes)."""
        self.history.clear()
        self.is_warmed_up = False
        self.mean = 0.0
        self.std = 0.0
    
    def get_status(self) -> dict:
        """Get current baseline status."""
        return {
            'is_warmed_up': self.is_warmed_up,
            'window_size': len(self.history),
            'mean': self.mean,
            'std': self.std,
            'threshold': self.mean + self.std_multiplier * self.std
        }


# ============================================================================
# Defense 3: Bulletproof Error Handling (Graceful Degradation)
# ============================================================================

def bulletproof_parse_adapter(parse_func: Callable, raw_input: Any,
                              log_errors: bool = True) -> np.ndarray:
    """
    Execute a parser and validate its output; report failures to the caller.
    
    Args:
        parse_func: Function to execute for parsing
        raw_input: Raw input data
        log_errors: Whether to log errors
    
    Returns:
        Parsed numeric data
    """
    try:
        result = parse_func(raw_input)
        
        # Validate result
        if result is None:
            raise ValueError("Parse function returned None")
        
        try:
            result = np.asarray(result, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError("Parser output must be numeric") from e
        if result.size == 0:
            raise ValueError("Parse function returned empty array")
        if not np.isfinite(result).all():
            raise ValueError("Parse function returned non-finite values")
        
        return result
        
    except Exception:
        if log_errors:
            logging.exception("Data parsing failed")
        raise


class BulletproofAdapter:
    """
    Wrapper class for bulletproof adapter execution.
    """
    
    def __init__(self):
        """
        Initialize bulletproof adapter.
        """
        self.error_count = 0
        self.max_errors = 10
    
    def execute(self, func: Callable, *args, **kwargs) -> np.ndarray:
        """
        Execute function with bulletproof protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        """
        try:
            result = func(*args, **kwargs)
            if result is None:
                raise ValueError("Execution returned no data")
            result = np.asarray(result, dtype=np.float64)
            if result.size == 0 or not np.isfinite(result).all():
                raise ValueError("Execution returned empty or non-finite data")
            self.error_count = 0  # Reset error count on success
            return result
        except Exception as e:
            self.error_count += 1
            logging.error(f"Execution failed (error {self.error_count}/{self.max_errors}): {e}")
            
            if self.error_count >= self.max_errors:
                logging.critical(f"Too many errors ({self.error_count}), system may be degraded")
            raise


# ============================================================================
# Defense 4: Vectorization Utilities (GIL Bypass)
# ============================================================================

class VectorizedOps:
    """
    Vectorized operations to bypass Python GIL.
    All operations use NumPy C++ backend for maximum performance.
    """

    @staticmethod
    def fast_vectorized_normalization(matrix: np.ndarray) -> np.ndarray:
        """
        Vectorized Z-score normalization (no Python loops).

        Args:
            matrix: Input data matrix

        Returns:
            Normalized matrix
        """
        mean = np.mean(matrix, axis=0, keepdims=True)
        std = np.std(matrix, axis=0, keepdims=True)
        return (matrix - mean) / (std + 1e-8)

    @staticmethod
    def fast_vectorized_distance(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
        """
        Vectorized Euclidean distance calculation.

        Args:
            matrix1: First matrix
            matrix2: Second matrix

        Returns:
            Distance matrix
        """
        diff = matrix1[:, np.newaxis, :] - matrix2[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    @staticmethod
    def fast_vectorized_rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
        """
        Vectorized rolling mean calculation.

        Args:
            data: Input data
            window: Window size

        Returns:
            Rolling mean array
        """
        cumsum = np.cumsum(data, axis=0)
        cumsum[window:] = cumsum[window:] - cumsum[:-window]
        return cumsum[window - 1:] / window

    @staticmethod
    def fast_vectorized_rolling_std(data: np.ndarray, window: int) -> np.ndarray:
        """
        Vectorized rolling standard deviation calculation.

        Args:
            data: Input data
            window: Window size

        Returns:
            Rolling std array
        """
        rolling_mean = VectorizedOps.fast_vectorized_rolling_mean(data, window)
        padded_mean = np.pad(rolling_mean, ((window - 1, 0), (0, 0)), mode='edge')
        diff_squared = (data - padded_mean) ** 2
        rolling_var = VectorizedOps.fast_vectorized_rolling_mean(diff_squared, window)
        return np.sqrt(rolling_var)

    @staticmethod
    @lru_cache(maxsize=64)
    def _cached_pca_projection(data_hash: int, n_components: int) -> np.ndarray:
        """Cached PCA projection for static data."""
        pass

    @staticmethod
    def dimensionality_reduction_pca(data: np.ndarray, n_components: int = 10) -> np.ndarray:
        """
        PCA-based dimensionality reduction for high-dimensional data.

        Args:
            data: Input data matrix (n_samples, n_features)
            n_components: Target number of components

        Returns:
            Reduced data matrix
        """
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(data)

    @staticmethod
    def random_projection(data: np.ndarray, n_components: int = 10) -> np.ndarray:
        """
        Random projection for fast dimensionality reduction.

        Args:
            data: Input data matrix (n_samples, n_features)
            n_components: Target number of components

        Returns:
            Reduced data matrix
        """
        from sklearn.random_projection import GaussianRandomProjection
        transformer = GaussianRandomProjection(n_components=n_components, random_state=42)
        return transformer.fit_transform(data)


class SlidingWindowProcessor:
    """
    Sliding window batch processor for high-frequency streaming data.
    Implements time-based batching for TDA computation.
    """

    def __init__(self, window_size: int = 1000, stride: int = 500):
        """
        Initialize sliding window processor.

        Args:
            window_size: Number of samples per window
            stride: Step size between windows (overlap = window_size - stride)
        """
        self.window_size = window_size
        self.stride = stride

    def process_streaming_data(
        self,
        data: np.ndarray,
        process_func: Callable
    ) -> List[Tuple[int, int, Any]]:
        """
        Process streaming data in sliding windows.

        Args:
            data: Input data matrix (n_samples, n_features)
            process_func: Function to apply to each window

        Returns:
            List of (start_idx, end_idx, result) tuples
        """
        results = []
        n_samples = data.shape[0]

        for start_idx in range(0, n_samples - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            window = data[start_idx:end_idx]
            result = process_func(window)
            results.append((start_idx, end_idx, result))

        return results

    def get_window_count(self, n_samples: int) -> int:
        """Calculate number of windows for given data size."""
        return max(0, (n_samples - self.window_size) // self.stride + 1)


# ============================================================================
# Unified Defensive Wrapper
# ============================================================================

class EnsembleScoring:
    """
    Hybrid decision mechanism combining TDA and Z3 verification.
    Reduces false positives by requiring both systems to agree.
    """

    def __init__(self, tda_threshold: float = 0.5, z3_required: bool = True):
        """
        Initialize ensemble scoring.

        Args:
            tda_threshold: TDA anomaly score threshold
            z3_required: Whether Z3 verification is required for high-confidence alerts
        """
        self.tda_threshold = tda_threshold
        self.z3_required = z3_required

    def evaluate(
        self,
        tda_anomaly_score: float,
        z3_result: Optional[Dict[str, Any]] = None,
        tda_stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate ensemble decision combining TDA and Z3.

        Args:
            tda_anomaly_score: TDA anomaly score (0-1)
            z3_result: Z3 verification result from Z3FormalVerificationPlugin
            tda_stats: TDA statistics for additional context

        Returns:
            Dict with ensemble decision and severity
        """
        severity = "LOW"
        is_anomaly = False
        reasons = []

        # TDA evaluation
        tda_exceeds = tda_anomaly_score > self.tda_threshold
        if tda_exceeds:
            reasons.append("TDA anomaly detected")
            severity = "MEDIUM"

        # Z3 evaluation (if provided)
        z3_violated = False
        if z3_result is not None:
            z3_violated = z3_result.get('is_violated', False)
            if z3_violated:
                reasons.append("Z3 constraint violated")
                severity = "HIGH" if tda_exceeds else "MEDIUM"

        # Ensemble decision logic
        if self.z3_required:
            # High confidence: both TDA and Z3 must agree
            is_anomaly = tda_exceeds and z3_violated
            if is_anomaly:
                severity = "CRITICAL"
            elif tda_exceeds and not z3_violated:
                reasons.append("Z3 verified as safe - downgrade")
                severity = "LOW"
                is_anomaly = False
        else:
            # TDA-only mode
            is_anomaly = tda_exceeds
            if z3_violated:
                severity = "HIGH"

        return {
            'is_anomaly': is_anomaly,
            'severity': severity,
            'reasons': reasons,
            'tda_score': tda_anomaly_score,
            'z3_violated': z3_violated if z3_result else None,
            'decision_logic': 'ensemble' if self.z3_required else 'tda_only'
        }


class DefensiveWrapper:
    """
    Unified wrapper integrating all four defensive mechanisms.
    Now with Polars integration for high-performance data pipeline.
    """

    def __init__(self, max_points: int = 1000, baseline_window: int = 50,
                 use_polars: bool = True):
        """
        Initialize defensive wrapper.

        Args:
            max_points: Maximum points for hard cap
            baseline_window: Window size for adaptive baseline
            use_polars: Whether to use Polars for data pipeline (default True)
        """
        self.hard_cap = HardCapGuard()
        self.baseline = AdaptiveBaseline(window_size=baseline_window)
        self.bulletproof = BulletproofAdapter()
        self.vectorized = VectorizedOps()
        self.sliding_window = SlidingWindowProcessor()
        self.ensemble = EnsembleScoring()
        self.use_polars = use_polars

        # Lazy-load Polars pipeline if enabled
        self._polars_pipeline = None
        if use_polars:
            try:
                from polars_engine import PolarsDataPipeline
                self._polars_pipeline = PolarsDataPipeline(
                    window_size=self.sliding_window.window_size,
                    stride=self.sliding_window.stride
                )
                logging.info("Polars pipeline initialized successfully")
            except ImportError:
                logging.warning("Polars not available, falling back to NumPy-only pipeline")
                self.use_polars = False

    def safe_execute(self, parse_func: Callable, raw_input: Any,
                    current_score: Optional[float] = None) -> tuple[np.ndarray, bool]:
        """
        Execute with full defensive protection.

        Args:
            parse_func: Function to execute
            raw_input: Raw input data
            current_score: Optional anomaly score for baseline evaluation

        Returns:
            Tuple of (result_data, is_anomaly)
        """
        # Defense 3: Bulletproof execution
        result = self.bulletproof.execute(parse_func, raw_input)

        # Defense 1: Hard cap validation
        result = self.hard_cap.validate_and_cap(result)

        # Defense 4: Vectorized normalization
        result = self.vectorized.fast_vectorized_normalization(result)

        # Defense 2: Adaptive baseline evaluation
        is_anomaly = False
        if current_score is not None:
            is_anomaly, _, _ = self.baseline.evaluate(current_score)

        return result, is_anomaly

    def process_with_polars(
        self,
        raw_stream: List[Any] | np.ndarray,
        process_func: Callable,
    ) -> List[Tuple[int, int, Any]]:
        """Process a stream in sliding windows, using Polars when available."""
        if self._polars_pipeline is not None:
            return self._polars_pipeline.process_sliding_windows(
                process_func=process_func,
                raw_stream=_convert_to_dict_stream(raw_stream),
            )
        data = np.asarray(raw_stream)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self.sliding_window.process_streaming_data(data, process_func)


# ============================================================================
# TopoGuard Unified Controller
# ============================================================================

class TopoGuardEngine:
    """
    Unified TopoGuard engine integrating all components.
    Implements the strict single-directional pipeline:
    eBPF -> Polars -> TDA+PyG -> Z3

    Usage:
        engine = TopoGuardEngine()
        result = engine.inspect_tick()
    """

    def __init__(
        self,
        use_ebpf: bool = True,
        use_polars: bool = True,
        use_hybrid: bool = True,
        use_z3: bool = True,
        window_size: int = 1000,
        stride: int = 500,
        baseline_path: Optional[str] = None,
        correlator: Optional[Any] = None,
        tda_max_dimension: int = 1,
    ):
        """
        Initialize TopoGuard engine with all components.

        Args:
            use_ebpf: Whether to use eBPF collector (optional plugin)
            use_polars: Whether to use Polars data pipeline
            use_hybrid: Whether to use hybrid TDA+PyG engine
            use_z3: Whether to use Z3 formal verification
            window_size: Sliding window size
            stride: Sliding window stride
            tda_max_dimension: Highest homology dimension computed by TDA
        """
        self.use_ebpf = use_ebpf and EBPF_AVAILABLE
        self.use_polars = use_polars and POLARS_AVAILABLE
        self.use_hybrid = use_hybrid and HYBRID_AVAILABLE
        self.use_z3 = use_z3 and Z3_AVAILABLE
        if window_size < 1 or stride < 1:
            raise ValueError("window_size and stride must be positive")
        if tda_max_dimension not in (0, 1, 2):
            raise ValueError("tda_max_dimension must be between 0 and 2")
        self.window_size = window_size
        self.stride = stride
        self.tda_max_dimension = tda_max_dimension

        # Initialize components
        self.ebpf = None
        if self.use_ebpf:
            try:
                self.ebpf = EBPFCollector(require_real=True)
                self.ebpf.start()
                logging.info("eBPF collector initialized")
            except Exception as e:
                self.ebpf = None
                logging.warning("eBPF initialization failed; collector disabled: %s", e)

        self.pipeline = None
        if self.use_polars:
            try:
                self.pipeline = PolarsDataPipeline(window_size=window_size, stride=stride)
                logging.info("Polars pipeline initialized")
            except Exception as e:
                logging.warning(f"Polars initialization failed: {e}")

        self.hybrid_ai = None
        if self.use_hybrid:
            try:
                self.hybrid_ai = HybridTdaGnnEngine(
                    tda_threshold=0.5,
                    gnn_threshold=0.6,
                    use_gnn=False,
                    use_isolation_forest=True,
                    tda_max_dimension=tda_max_dimension,
                )
                logging.info("Hybrid AI engine initialized")
            except Exception as e:
                logging.warning(f"Hybrid engine initialization failed: {e}")

        self.z3_guard = None
        if self.use_z3:
            try:
                self.z3_guard = Z3FormalVerificationPlugin
                logging.info("Z3 guard initialized")
            except Exception as e:
                logging.warning(f"Z3 initialization failed: {e}")

        # Phase 1: optional pre-trained baseline. Loaded here so Isolation Forest
        # skips the 33-sample warm-up and scores accurately from the first tick.
        self.baseline_calibrator = None
        self._trained_feature_names: tuple[str, ...] | None = None
        if baseline_path:
            snapshot_path = Path(baseline_path)
            if snapshot_path.exists() and not snapshot_path.is_file():
                raise ValueError(f"Baseline snapshot path is not a file: {baseline_path}")
            if snapshot_path.is_file():
                from baseline import BaselineCalibrator

                self.baseline_calibrator = BaselineCalibrator.load(baseline_path)
                if (
                    self.hybrid_ai is not None
                    and self.hybrid_ai.isolation_forest is not None
                    and self.baseline_calibrator.model._model is not None
                ):
                    cal = self.baseline_calibrator.model
                    self.hybrid_ai.isolation_forest._model = cal._model
                    self.hybrid_ai.isolation_forest._feature_count = cal._feature_count
                    self.hybrid_ai.isolation_forest._feature_names = cal._feature_names
                    self.hybrid_ai.isolation_forest._baseline_samples = (
                        cal._baseline_samples
                    )
                    self.hybrid_ai.isolation_forest._trained_at = cal._trained_at
                    self.hybrid_ai.isolation_forest.contamination = cal.contamination
                    self.hybrid_ai.isolation_forest.random_state = cal.random_state
                    self._trained_feature_names = tuple(
                        cal._feature_names or []
                    )
                    logging.info(
                        "Pre-trained baseline loaded: %s samples, %s features",
                        self.baseline_calibrator.sample_count,
                        len(self._trained_feature_names),
                    )
            else:
                logging.info(
                    "No baseline snapshot at %s; Isolation Forest will warm up",
                    baseline_path,
                )

        # Phase 2: optional multi-source time-window correlator.
        self.correlator = correlator

        self.defensive = DefensiveWrapper(use_polars=self.use_polars)

        logging.info(f"TopoGuard engine initialized: eBPF={self.use_ebpf}, Polars={self.use_polars}, Hybrid={self.use_hybrid}, Z3={self.use_z3}")

    def inspect_tick(
        self,
        raw_stream: Optional[List[Any]] = None,
        replace_stream: bool = False,
        source_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Single inspection tick through the full pipeline.

        Args:
            raw_stream: Optional raw stream data (if None, polls from eBPF)
            replace_stream: Whether to replace the Polars pipeline buffer
            source_name: Data source name, used for multi-source correlation

        Returns:
            Dict with inspection result and status
        """
        try:
            import numpy as np

            # Step 1: Collect data (eBPF or provided stream)
            if raw_stream is None and self.ebpf is not None:
                raw_stream = self.ebpf.poll()

            if raw_stream is None or len(raw_stream) == 0:
                return {
                    'status': 'NO_DATA',
                    'is_anomaly': False,
                    'message': 'No data available'
                }

            # Step 2: Process through Polars pipeline
            feature_names = None
            if isinstance(raw_stream, list) and raw_stream and isinstance(raw_stream[0], dict):
                feature_names = tuple(raw_stream[0].keys())
            if self.pipeline is not None:
                if replace_stream:
                    self.pipeline.reset()
                matrix = self.pipeline.process(raw_stream)
            else:
                # Fallback to NumPy
                if isinstance(raw_stream, list) and len(raw_stream) > 0 and isinstance(raw_stream[0], dict):
                    # Convert dict stream to NumPy array
                    import numpy as np
                    values = [list(d.values()) for d in raw_stream]
                    matrix = np.array(values)
                else:
                    import numpy as np
                    matrix = np.array(raw_stream) if not isinstance(raw_stream, np.ndarray) else raw_stream

            if matrix.shape[0] == 0:
                return {
                    'status': 'EMPTY_MATRIX',
                    'is_anomaly': False,
                    'message': 'Empty matrix after processing'
                }

            if feature_names is not None and matrix.shape[1] != len(feature_names):
                if matrix.shape[1] == len(feature_names) * 3:
                    feature_names = (
                        feature_names
                        + tuple(f"{name}__rolling_mean" for name in feature_names)
                        + tuple(f"{name}__rolling_std" for name in feature_names)
                    )
                else:
                    raise ValueError(
                        "Processed feature dimensions cannot be mapped to the source schema"
                    )

            # Phase 1: align the matrix to the pre-trained baseline schema so the
            # Isolation Forest scores on exactly the features it was fit on.
            if self._trained_feature_names:
                trained = list(self._trained_feature_names)
                name_to_col = {name: i for i, name in enumerate(feature_names or [])}
                columns = []
                for name in trained:
                    if name in name_to_col:
                        columns.append(matrix[:, name_to_col[name]])
                    else:
                        columns.append(np.zeros(matrix.shape[0]))
                matrix = np.column_stack(columns)
                feature_names = tuple(trained)

            # Step 3: Hybrid AI evaluation (TDA + PyG)
            if self.hybrid_ai is not None:
                score_result = self.hybrid_ai.evaluate(
                    matrix,
                    return_details=True,
                    feature_names=feature_names,
                )
                anomaly_score = score_result['anomaly_score']
                is_anomaly = score_result['is_anomaly']
                context = score_result
            else:
                # Fallback: simple statistical anomaly detection
                anomaly_score = self._simple_anomaly_score(matrix)
                is_anomaly = anomaly_score > 0.5
                context = {'anomaly_score': anomaly_score}

            # Keep the SMT solver off the hot path unless TDA confidence is high.
            z3_result = None
            if is_anomaly and anomaly_score > 0.85 and self.z3_guard is not None:
                try:
                    # Extract H0/H1 counts from context if available
                    h0_count = context.get('tda_details', {}).get('h0_count', 0)
                    h1_count = context.get('tda_details', {}).get('h1_count', 0)

                    z3_result = self.z3_guard.verify_topological_invariants(
                        h0_count=h0_count,
                        h1_count=h1_count,
                        threshold_h1=50,
                        enable_lazy=True,
                        tda_anomaly_score=anomaly_score,
                        tda_threshold=0.85
                    )

                    # Update anomaly status based on Z3
                    if z3_result['is_violated']:
                        is_anomaly = True
                        context['severity'] = 'CRITICAL'
                    else:
                        is_anomaly = False
                        context['severity'] = 'LOW'
                except Exception as e:
                    logging.warning(f"Z3 verification failed: {e}")

            # Phase 2: feed current source rows into the time-window correlator
            # and attach the latest Attack Context to the result context.
            if self.correlator is not None and raw_stream is not None:
                try:
                    now = time.time()
                    for row in raw_stream:
                        if isinstance(row, dict):
                            self.correlator.ingest(source_name or "unknown", now, **row)
                    attack_context = self.correlator.correlate()
                    if attack_context.get("correlated"):
                        context["attack_context"] = attack_context
                except Exception as e:
                    logging.warning(f"Event correlation failed: {e}")

            return {
                'status': 'SUCCESS',
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'context': context,
                'z3_result': z3_result,
                'data_points': matrix.shape[0]
            }

        except Exception as e:
            logging.error(f"Inspection tick failed: {e}")
            return {
                'status': 'ERROR',
                'is_anomaly': True,
                'severity': 'CRITICAL',
                'message': str(e)
            }

    def _simple_anomaly_score(self, matrix: np.ndarray) -> float:
        """
        Simple statistical anomaly detection as fallback.
        Uses z-score based anomaly detection.

        Args:
            matrix: Input data matrix

        Returns:
            Anomaly score (0-1)
        """
        if matrix.shape[0] == 0:
            return 0.0

        # Compute z-scores
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0) + 1e-8
        z_scores = np.abs((matrix - mean) / std)

        # Max z-score as anomaly indicator
        max_z_score = np.max(z_scores)

        # Normalize to 0-1 (assuming z-score > 3 is anomalous)
        anomaly_score = min(1.0, max_z_score / 3.0)

        return anomaly_score

    def stop(self) -> None:
        """Stop all background collectors."""
        if self.ebpf is not None:
            self.ebpf.stop()
            logging.info("eBPF collector stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        status = {
            'ebpf_active': self.ebpf._is_active if self.ebpf else False,
            'polars_available': self.pipeline is not None,
            'hybrid_available': self.hybrid_ai is not None,
            'z3_available': self.z3_guard is not None,
            'baseline_loaded': self.baseline_calibrator is not None,
            'correlator_active': self.correlator is not None,
        }

        if self.baseline_calibrator is not None:
            status['baseline'] = self.baseline_calibrator.model.get_status()

        if self.ebpf:
            status['ebpf_status'] = self.ebpf.get_status()

        if self.pipeline:
            status['polars_status'] = self.pipeline.get_status()

        if self.hybrid_ai:
            status['hybrid_status'] = self.hybrid_ai.get_status()

        return status

    def reset(self) -> None:
        """Reset all components."""
        if self.ebpf:
            self.ebpf.reset()

        if self.pipeline:
            self.pipeline.reset()

        if self.hybrid_ai:
            self.hybrid_ai.reset()

        logging.info("TopoGuard engine reset")

    def process_with_polars(
        self,
        raw_stream: List[Any],
        process_func: Callable
    ) -> List[Tuple[int, int, Any]]:
        """
        Process streaming data using Polars pipeline if available.
        Falls back to NumPy sliding window if Polars unavailable.

        Args:
            raw_stream: Raw streaming data
            process_func: Function to apply to each window

        Returns:
            List of (start_idx, end_idx, result) tuples
        """
        if self.pipeline is not None:
            # Convert raw stream to dict format for Polars
            dict_stream = self._convert_to_dict_stream(raw_stream)
            return self.pipeline.process_sliding_windows(
                process_func=process_func,
                raw_stream=dict_stream
            )
        else:
            # Fallback to NumPy sliding window
            data = np.array(raw_stream) if not isinstance(raw_stream, np.ndarray) else raw_stream
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            processor = SlidingWindowProcessor(
                window_size=self.window_size,
                stride=self.stride,
            )
            return processor.process_streaming_data(data, process_func)

    def _convert_to_dict_stream(self, raw_stream: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert raw stream to dict format for Polars ingestion.

        Args:
            raw_stream: Raw streaming data

        Returns:
            List of dictionaries
        """
        return _convert_to_dict_stream(raw_stream)
