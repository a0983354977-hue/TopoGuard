import numpy as np
import logging
from typing import Callable, Any, Optional
from collections import deque

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class HardCapGuard:
    DEFAULT_MAX_POINTS = 1000
    FALLBACK_SHAPE = (100, 3)

    @staticmethod
    def safe_downsample(data: np.ndarray, max_points: int = DEFAULT_MAX_POINTS) -> np.ndarray:
        if data is None or len(data) == 0:
            logging.warning("Empty data received, returning fallback")
            return np.zeros(HardCapGuard.FALLBACK_SHAPE)

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
            logging.warning("None data received, returning fallback")
            return np.zeros(HardCapGuard.FALLBACK_SHAPE)

        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except Exception as e:
                logging.error(f"Cannot convert to numpy array: {e}")
                return np.zeros(HardCapGuard.FALLBACK_SHAPE)

        if data.ndim != 2:
            logging.warning(f"Invalid dimensions {data.ndim}, reshaping to 2D")
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            else:
                data = data.reshape(data.shape[0], -1)
        
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
                              fallback_shape: tuple = (100, 3),
                              log_errors: bool = True) -> np.ndarray:
    """
    Execute parse function with bulletproof error handling.
    Returns safe fallback data on failure to prevent daemon crashes.
    
    Args:
        parse_func: Function to execute for parsing
        raw_input: Raw input data
        fallback_shape: Shape of fallback data on error
        log_errors: Whether to log errors
    
    Returns:
        Parsed data or fallback data on error
    """
    try:
        result = parse_func(raw_input)
        
        # Validate result
        if result is None:
            raise ValueError("Parse function returned None")
        
        if not isinstance(result, np.ndarray):
            try:
                result = np.array(result)
            except Exception as e:
                raise ValueError(f"Cannot convert result to numpy array: {e}")
        
        if result.size == 0:
            raise ValueError("Parse function returned empty array")
        
        return result
        
    except Exception as e:
        if log_errors:
            logging.warning(f"Data parsing corrupted, falling back to null-space: {str(e)}")
        
        # Return safe fallback data
        fallback_data = np.random.randn(*fallback_shape) * 0.01  # Small noise
        logging.info(f"Returned fallback data with shape {fallback_shape}")
        return fallback_data


class BulletproofAdapter:
    """
    Wrapper class for bulletproof adapter execution.
    """
    
    def __init__(self, fallback_shape: tuple = (100, 3)):
        """
        Initialize bulletproof adapter.
        
        Args:
            fallback_shape: Shape of fallback data on error
        """
        self.fallback_shape = fallback_shape
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
            Function result or fallback data
        """
        try:
            result = func(*args, **kwargs)
            self.error_count = 0  # Reset error count on success
            return result
        except Exception as e:
            self.error_count += 1
            logging.error(f"Execution failed (error {self.error_count}/{self.max_errors}): {e}")
            
            if self.error_count >= self.max_errors:
                logging.critical(f"Too many errors ({self.error_count}), system may be degraded")
            
            return np.random.randn(*self.fallback_shape) * 0.01


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
        # Pure NumPy vectorization - no Python loops
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
        # Broadcasting for vectorized distance
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
        # Using cumulative sum for O(n) complexity
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
        # Vectorized rolling variance
        rolling_mean = VectorizedOps.fast_vectorized_rolling_mean(data, window)
        
        # Pad rolling_mean to match original shape
        padded_mean = np.pad(rolling_mean, ((window - 1, 0), (0, 0)), mode='edge')
        
        # Calculate variance
        diff_squared = (data - padded_mean) ** 2
        rolling_var = VectorizedOps.fast_vectorized_rolling_mean(diff_squared, window)
        
        return np.sqrt(rolling_var)


# ============================================================================
# Unified Defensive Wrapper
# ============================================================================

class DefensiveWrapper:
    """
    Unified wrapper integrating all four defensive mechanisms.
    """
    
    def __init__(self, max_points: int = 1000, baseline_window: int = 50,
                 fallback_shape: tuple = (100, 3)):
        """
        Initialize defensive wrapper.
        
        Args:
            max_points: Maximum points for hard cap
            baseline_window: Window size for adaptive baseline
            fallback_shape: Fallback data shape
        """
        self.hard_cap = HardCapGuard()
        self.baseline = AdaptiveBaseline(window_size=baseline_window)
        self.bulletproof = BulletproofAdapter(fallback_shape=fallback_shape)
        self.vectorized = VectorizedOps()
    
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
