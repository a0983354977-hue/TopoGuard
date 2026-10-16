import numpy as np
import pandas as pd
import json
from typing import Union, Tuple
from pathlib import Path
from defensive import HardCapGuard, VectorizedOps


class DataAdapter:
    def __init__(self, max_points: int = 1000):
        self.data = None
        self.normalized_data = None
        self.hard_cap = HardCapGuard()
        self.vectorized = VectorizedOps()
        self.max_points = max_points
    
    def load_csv(self, filepath: str) -> np.ndarray:
        self.data = None
        self.normalized_data = None
        df = pd.read_csv(filepath)
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty or numeric_df.shape[1] == 0:
            raise ValueError("CSV contains no numeric telemetry columns")
        self.data = self.hard_cap.validate_and_cap(
            numeric_df.values,
            max_points=self.max_points,
        )
        self.normalized_data = None
        return self.data
    
    def load_json(self, filepath: str) -> np.ndarray:
        self.data = None
        self.normalized_data = None
        with open(filepath, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty or numeric_df.shape[1] == 0:
            raise ValueError("JSON contains no numeric telemetry columns")
        self.data = self.hard_cap.validate_and_cap(
            numeric_df.values,
            max_points=self.max_points,
        )
        self.normalized_data = None
        return self.data

    def load_numpy(self, data: np.ndarray) -> np.ndarray:
        self.data = None
        self.normalized_data = None
        self.data = self.hard_cap.validate_and_cap(
            data,
            max_points=self.max_points,
        )
        return self.data
    
    def normalize_zscore(self, data: np.ndarray = None) -> np.ndarray:
        if data is None:
            data = self.data

        if data is None:
            raise ValueError("No data loaded. Call load_csv, load_json, or load_numpy first.")

        self.normalized_data = self.vectorized.fast_vectorized_normalization(data)
        return self.normalized_data
    
    def get_shape(self) -> Tuple[int, int]:
        if self.data is None:
            raise ValueError("No data loaded")
        return self.data.shape
    
    def downsample(self, target_size: int = 500, method: str = 'random') -> np.ndarray:
        if self.data is None:
            raise ValueError("No data loaded")

        n_samples = self.data.shape[0]

        if n_samples <= target_size:
            return self.data

        if method == 'random':
            indices = np.random.choice(n_samples, target_size, replace=False)
        elif method == 'uniform':
            step = n_samples // target_size
            indices = np.arange(0, n_samples, step)[:target_size]
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        self.data = self.data[indices]
        if self.normalized_data is not None and self.normalized_data.shape[0] == n_samples:
            self.normalized_data = self.normalized_data[indices]
        else:
            self.normalized_data = None

        return self.data
    
    def sliding_window(self, window_size: int = 500, step: int = 250) -> np.ndarray:
        if self.data is None:
            raise ValueError("No data loaded")

        n_samples = self.data.shape[0]

        for start in range(0, n_samples - window_size + 1, step):
            end = start + window_size
            yield self.data[start:end]
