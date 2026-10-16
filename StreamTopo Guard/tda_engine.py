import numpy as np
import gudhi
from typing import Tuple, List, Dict
import time


class TDAEngine:
    DEFAULT_MAX_POINTS = 300

    def __init__(
        self,
        max_dimension: int = 1,
        max_edge_length: float = 1.5,
        max_points: int = DEFAULT_MAX_POINTS,
    ):
        if max_dimension not in (0, 1, 2):
            raise ValueError("max_dimension must be between 0 and 2")
        if not np.isfinite(max_edge_length) or max_edge_length <= 0:
            raise ValueError("max_edge_length must be finite and positive")
        if max_points < 2:
            raise ValueError("max_points must be at least 2")
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.max_points = max_points
        self.last_computation_time = 0

    def compute_persistence_topology(
        self,
        data: np.ndarray,
        max_edge: float | None = None,
    ) -> Tuple[np.ndarray, List[Tuple[int, Tuple[float, float]]]]:
        if max_edge is None:
            max_edge = self.max_edge_length
        if not np.isfinite(max_edge) or max_edge <= 0:
            raise ValueError("max_edge must be finite and positive")

        points = np.asarray(data, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] == 0:
            raise ValueError("TDA input must contain at least two points and one feature")
        if not np.isfinite(points).all():
            raise ValueError("TDA input must contain only finite values")
        if points.shape[0] > self.max_points:
            indices = np.linspace(
                0,
                points.shape[0] - 1,
                self.max_points,
                dtype=np.intp,
            )
            points = points[indices]
        points = np.ascontiguousarray(points, dtype=np.float64)

        start_time = time.perf_counter()

        rips = gudhi.RipsComplex(points=points, max_edge_length=max_edge)
        # Computing H_k requires simplices through dimension k + 1 so
        # k-dimensional classes can be killed by their boundaries.
        st = rips.create_simplex_tree(max_dimension=self.max_dimension + 1)
        diag = st.persistence()

        self.last_computation_time = time.perf_counter() - start_time

        return points, diag

    def extract_persistence_pairs(
        self,
        persistence_diagram: List[Tuple[int, Tuple[float, float]]]
    ) -> Dict[int, List[Tuple[float, float]]]:
        pairs = {0: [], 1: [], 2: []}
        for dim, (birth, death) in persistence_diagram:
            if dim in pairs:
                pairs[dim].append((birth, death))
        return pairs

    @staticmethod
    def _compute_intervals(
        persistence_diagram: List[Tuple[int, Tuple[float, float]]],
    ) -> Dict[int, List[float]]:
        intervals = {0: [], 1: [], 2: []}
        for dim, (birth, death) in persistence_diagram:
            if dim in intervals and not np.isinf(death):
                interval = death - birth
                if interval > 0:
                    intervals[dim].append(interval)
        return intervals

    def get_statistics(
        self,
        persistence_diagram: List[Tuple[int, Tuple[float, float]]]
    ) -> Dict[str, Dict[str, float]]:
        intervals = self._compute_intervals(persistence_diagram)

        stats = {}
        for dim in [0, 1, 2]:
            if intervals[dim]:
                arr = np.array(intervals[dim])
                stats[f"H{dim}"] = {
                    'count': len(arr),
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'max': float(np.max(arr)),
                    'min': float(np.min(arr))
                }
            else:
                stats[f"H{dim}"] = {'count': 0, 'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}

        stats['computation_time'] = self.last_computation_time
        return stats


def compute_persistence_topology(
    raw_data: np.ndarray,
    max_edge: float = 1.5,
    max_dimension: int = 1,
) -> Tuple[np.ndarray, List[Tuple[int, Tuple[float, float]]]]:
    """Convenience function for one-line TDA computation."""
    normalized_data = (raw_data - np.mean(raw_data, axis=0)) / (np.std(raw_data, axis=0) + 1e-8)
    engine = TDAEngine(max_dimension=max_dimension, max_edge_length=max_edge)
    return engine.compute_persistence_topology(normalized_data, max_edge=max_edge)
