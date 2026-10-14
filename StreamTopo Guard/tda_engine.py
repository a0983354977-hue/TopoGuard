import numpy as np
import gudhi
from typing import Tuple, List, Dict
import time


class TDAEngine:
    def __init__(self, max_dimension: int = 2, max_edge_length: float = 1.5):
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.last_computation_time = 0
    
    def compute_persistence_topology(
        self,
        data: np.ndarray,
        max_edge: float = None
    ) -> Tuple[np.ndarray, List[Tuple[int, Tuple[float, float]]]]:
        if max_edge is None:
            max_edge = self.max_edge_length

        start_time = time.time()

        rips = gudhi.RipsComplex(points=data, max_edge_length=max_edge)
        st = rips.create_simplex_tree(max_dimension=self.max_dimension)
        diag = st.persistence()

        self.last_computation_time = time.time() - start_time

        return data, diag
    
    def extract_persistence_pairs(
        self,
        persistence_diagram: List[Tuple[int, Tuple[float, float]]]
    ) -> Dict[int, List[Tuple[float, float]]]:
        pairs = {0: [], 1: [], 2: []}

        for dim, (birth, death) in persistence_diagram:
            if dim in pairs:
                pairs[dim].append((birth, death))

        return pairs
    
    def compute_persistence_intervals(
        self,
        persistence_diagram: List[Tuple[int, Tuple[float, float]]]
    ) -> Dict[int, List[float]]:
        intervals = {0: [], 1: [], 2: []}

        for dim, (birth, death) in persistence_diagram:
            if dim in intervals:
                if not np.isinf(death):
                    interval = death - birth
                    if interval > 0:
                        intervals[dim].append(interval)

        return intervals
    
    def get_statistics(
        self,
        persistence_diagram: List[Tuple[int, Tuple[float, float]]]
    ) -> Dict[str, Dict[str, float]]:
        pairs = self.extract_persistence_pairs(persistence_diagram)
        intervals = self.compute_persistence_intervals(persistence_diagram)

        stats = {}

        for dim in [0, 1, 2]:
            if len(intervals[dim]) > 0:
                stats[f"H{dim}"] = {
                    'count': len(intervals[dim]),
                    'mean': float(np.mean(intervals[dim])),
                    'std': float(np.std(intervals[dim])),
                    'max': float(np.max(intervals[dim])),
                    'min': float(np.min(intervals[dim]))
                }
            else:
                stats[f"H{dim}"] = {
                    'count': 0,
                    'mean': 0.0,
                    'std': 0.0,
                    'max': 0.0,
                    'min': 0.0
                }

        stats['computation_time'] = self.last_computation_time

        return stats


def compute_persistence_topology(
    raw_data: np.ndarray,
    max_edge: float = 1.5
) -> Tuple[np.ndarray, List[Tuple[int, Tuple[float, float]]]]:
    normalized_data = (raw_data - np.mean(raw_data, axis=0)) / (np.std(raw_data, axis=0) + 1e-8)

    rips = gudhi.RipsComplex(points=normalized_data, max_edge_length=max_edge)
    st = rips.create_simplex_tree(max_dimension=2)
    diag = st.persistence()

    return normalized_data, diag
