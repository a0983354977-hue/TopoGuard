"""
Hybrid TDA + Isolation Forest + PyG Engine
-----------------------
Combines Topological Data Analysis (TDA), a scikit-learn Isolation Forest,
and an optional PyTorch Geometric (PyG) GNN.
Responsibilities:
- Extract TDA persistence features (Betti vectors)
- Use TDA features as node attributes for PyG graphs
- Lazy-loading GNN: only trigger on TDA anomaly
- Output hybrid anomaly score

STRICT CONSTRAINTS:
- TDA provides geometric features
- PyG provides graph topology analysis
- Lazy-loading: GNN only when TDA indicates anomaly
- Single-directional flow: Data -> TDA -> (if anomaly) -> PyG -> Score
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IsolationForestBaseline:
    """Fit a bounded first-seen normal baseline and score the newest sample."""

    MIN_BASELINE_SAMPLES = 32
    MAX_BASELINE_SAMPLES = 256
    ANOMALY_THRESHOLD = 0.6

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        if not 0.0 < contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        self.contamination = contamination
        self.random_state = random_state
        self._model = None
        self._feature_count: int | None = None
        self._feature_names: tuple[str, ...] | None = None
        self._baseline_samples = 0
        self._trained_at: float | None = None

    def evaluate(
        self,
        data: np.ndarray,
        feature_names: tuple[str, ...] | None = None,
    ) -> Dict[str, Any]:
        matrix = np.ascontiguousarray(data, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] == 0 or matrix.shape[0] == 0:
            raise ValueError("Isolation Forest input must be a non-empty feature matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("Isolation Forest input must contain only finite values")
        if feature_names is not None and len(feature_names) != matrix.shape[1]:
            raise ValueError("Feature names do not match Isolation Forest input dimensions")

        if self._model is None:
            baseline_count = min(matrix.shape[0] - 1, self.MAX_BASELINE_SAMPLES)
            if baseline_count < self.MIN_BASELINE_SAMPLES:
                return {
                    "status": "warming_up",
                    "is_anomaly": False,
                    "anomaly_score": 0.0,
                    "baseline_samples": baseline_count,
                    "required_samples": self.MIN_BASELINE_SAMPLES,
                }

            from sklearn.ensemble import IsolationForest

            self._model = IsolationForest(
                n_estimators=100,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=1,
            )
            self._model.fit(matrix[:baseline_count])
            self._feature_count = matrix.shape[1]
            self._feature_names = feature_names
            self._baseline_samples = baseline_count
            self._trained_at = time.time()

        if matrix.shape[1] != self._feature_count:
            raise ValueError("Feature count changed after Isolation Forest baseline fitting")
        if self._feature_names is not None and feature_names != self._feature_names:
            raise ValueError("Feature schema changed after Isolation Forest baseline fitting")

        latest = matrix[-1:]
        decision_score = float(self._model.decision_function(latest)[0])
        is_anomaly = decision_score < 0.0
        anomaly_score = float(np.clip(0.5 - decision_score, 0.0, 1.0))
        if is_anomaly:
            anomaly_score = max(anomaly_score, self.ANOMALY_THRESHOLD + 0.01)

        return {
            "status": "active",
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "decision_score": decision_score,
            "baseline_samples": self._baseline_samples,
        }

    def reset(self) -> None:
        self._model = None
        self._feature_count = None
        self._feature_names = None
        self._baseline_samples = 0
        self._trained_at = None

    def get_status(self) -> Dict[str, Any]:
        return {
            "fitted": self._model is not None,
            "baseline_samples": self._baseline_samples,
            "required_samples": self.MIN_BASELINE_SAMPLES,
            "contamination": self.contamination,
        }


class HybridTdaGnnEngine:
    """
    Hybrid engine combining TDA, an unsupervised baseline, and optional PyG.
    """

    def __init__(
        self,
        tda_threshold: float = 0.5,
        gnn_threshold: float = 0.6,
        use_gnn: bool = False,
        gnn_layers: int = 2,
        use_isolation_forest: bool = True,
        tda_max_dimension: int = 1,
        max_tda_points: int = 300,
    ):
        """
        Initialize hybrid engine.

        Args:
            tda_threshold: TDA anomaly score threshold
            gnn_threshold: GNN anomaly score threshold
            use_gnn: Whether to opt into the experimental, untrained GNN path
            gnn_layers: Number of GNN layers (2-3 for lightweight)
        """
        self.tda_threshold = tda_threshold
        self.gnn_threshold = gnn_threshold
        self.use_gnn = use_gnn
        self.gnn_layers = gnn_layers
        self.use_isolation_forest = use_isolation_forest
        if tda_max_dimension not in (0, 1, 2):
            raise ValueError("tda_max_dimension must be between 0 and 2")
        if max_tda_points < 2:
            raise ValueError("max_tda_points must be at least 2")
        self.tda_max_dimension = tda_max_dimension
        self.max_tda_points = max_tda_points
        self.isolation_forest = (
            IsolationForestBaseline() if use_isolation_forest else None
        )

        # Lazy-load TDA engine
        self._tda_engine = None
        self._gnn_model = None
        self._pyg_available = self._check_pyg_available()

        logger.info(
            "Hybrid engine initialized: PyG available=%s, use_gnn=%s, Isolation Forest=%s",
            self._pyg_available,
            use_gnn,
            use_isolation_forest,
        )

    def _check_pyg_available(self) -> bool:
        """Check if PyTorch Geometric is available."""
        try:
            import torch
            import torch_geometric
            return True
        except ImportError:
            logger.warning("PyTorch Geometric not available, will use TDA-only mode")
            return False

    def _load_tda_engine(self):
        """Lazy-load TDA engine."""
        if self._tda_engine is None:
            from tda_engine import TDAEngine
            self._tda_engine = TDAEngine(
                max_dimension=self.tda_max_dimension,
                max_edge_length=1.5,
                max_points=self.max_tda_points,
            )
            logger.info("TDA engine loaded")

    def _load_gnn_model(self):
        """Lazy-load GNN model."""
        if self._gnn_model is None and self._pyg_available:
            import torch
            import torch.nn.functional as F
            from torch_geometric.nn import GCNConv

            class LightweightGNN(torch.nn.Module):
                """Lightweight GNN with 2-3 GCN layers."""

                def __init__(self, num_features: int, hidden_dim: int = 64, num_layers: int = 2):
                    super().__init__()
                    self.num_layers = num_layers
                    self.convs = torch.nn.ModuleList()

                    # First layer
                    self.convs.append(GCNConv(num_features, hidden_dim))

                    # Hidden layers
                    for _ in range(num_layers - 2):
                        self.convs.append(GCNConv(hidden_dim, hidden_dim))

                    # Output layer
                    self.convs.append(GCNConv(hidden_dim, 2))  # Binary classification

                def forward(self, x, edge_index):
                    for i, conv in enumerate(self.convs):
                        x = conv(x, edge_index)
                        if i < len(self.convs) - 1:
                            x = F.relu(x)
                    return F.log_softmax(x, dim=1)

            # Create model (will be trained on-the-fly or loaded from checkpoint)
            self._gnn_model = LightweightGNN(num_features=10, num_layers=self.gnn_layers)
            logger.info(f"GNN model loaded with {self.gnn_layers} layers")

    def evaluate(
        self,
        data: np.ndarray,
        return_details: bool = False,
        feature_names: tuple[str, ...] | None = None,
    ) -> Dict[str, Any]:
        """
        Evaluate data using hybrid TDA + GNN approach.

        Args:
            data: Input data matrix (n_samples, n_features)
            return_details: Whether to return detailed analysis

        Returns:
            Dict with anomaly score, severity, and decision details
        """
        start_time = time.time()

        data = np.ascontiguousarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] == 0 or data.shape[0] < 2:
            raise ValueError("TDA input must contain at least two rows and one feature")
        if not np.isfinite(data).all():
            raise ValueError("TDA input must contain only finite values")
        original_sample_count = data.shape[0]
        if original_sample_count > self.max_tda_points:
            sample_indices = np.linspace(
                0,
                original_sample_count - 1,
                self.max_tda_points,
                dtype=np.intp,
            )
            data = data[sample_indices]

        # Step 1: TDA analysis (always runs)
        self._load_tda_engine()
        tda_score, tda_details = self._run_tda_analysis(data)
        tda_details["original_sample_count"] = original_sample_count
        tda_details["analyzed_sample_count"] = int(data.shape[0])

        # Step 2: If TDA detects anomaly and GNN is available, run GNN
        gnn_score = 0.0
        gnn_details = None

        if tda_score > self.tda_threshold and self.use_gnn and self._pyg_available:
            logger.info("TDA detected anomaly, activating GNN for deep analysis")
            self._load_gnn_model()
            gnn_score, gnn_details = self._run_gnn_analysis(data, tda_details)
        else:
            logger.info("TDA score below threshold or GNN disabled, skipping GNN")

        iforest_details = (
            self.isolation_forest.evaluate(data, feature_names)
            if self.isolation_forest is not None
            else None
        )

        # Step 3: Combine independent geometric and point-level anomaly signals.
        final_score, severity = self._compute_hybrid_score(tda_score, gnn_score)
        if iforest_details is not None:
            final_score = max(final_score, iforest_details["anomaly_score"])
            iforest_severity = (
                "CRITICAL" if final_score > 0.8
                else "HIGH" if final_score > 0.6
                else "MEDIUM" if final_score > 0.4
                else "LOW"
            )
            severity = iforest_severity

        elapsed = time.time() - start_time

        result = {
            'is_anomaly': (
                final_score > max(self.tda_threshold, self.gnn_threshold)
                or bool(iforest_details and iforest_details["is_anomaly"])
            ),
            'anomaly_score': final_score,
            'severity': severity,
            'tda_score': tda_score,
            'gnn_score': gnn_score,
            'iforest_score': (
                iforest_details["anomaly_score"] if iforest_details else 0.0
            ),
            'computation_time_ms': elapsed * 1000,
            'gnn_activated': gnn_details is not None
        }

        if return_details:
            result['tda_details'] = tda_details
            result['gnn_details'] = gnn_details
            result['iforest_details'] = iforest_details

        return result

    def _run_tda_analysis(self, data: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Run TDA analysis on data.

        Args:
            data: Input data matrix

        Returns:
            Tuple of (anomaly_score, details_dict)
        """
        try:
            # Normalize data
            normalized_data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)

            # Compute persistence
            _, diag = self._tda_engine.compute_persistence_topology(normalized_data)

            # Extract statistics
            stats = self._tda_engine.get_statistics(diag)

            # Compute anomaly score based on Betti numbers
            h0_count = stats['H0']['count']
            h1_count = stats['H1']['count']
            h2_count = stats['H2']['count']

            # Simple anomaly scoring: high H1 indicates loops/holes (anomalous)
            anomaly_score = min(1.0, h1_count / 50.0)  # Normalize to 0-1

            details = {
                'h0_count': h0_count,
                'h1_count': h1_count,
                'h2_count': h2_count,
                'persistence_stats': stats
            }

            return anomaly_score, details

        except Exception as e:
            logger.exception("TDA analysis failed")
            raise RuntimeError(f"TDA analysis failed: {e}") from e

    def _run_gnn_analysis(
        self,
        data: np.ndarray,
        tda_details: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Run GNN analysis on data (lazy-loaded).

        Args:
            data: Input data matrix
            tda_details: TDA analysis results for feature extraction

        Returns:
            Tuple of (anomaly_score, details_dict)
        """
        try:
            import torch
            from torch_geometric.data import Data

            # Step 1: Build graph from data
            # Use k-nearest neighbors for edge construction
            graph_data = self._build_graph_from_data(data, tda_details)

            # Step 2: Run GNN inference
            with torch.no_grad():
                self._gnn_model.eval()
                out = self._gnn_model(graph_data.x, graph_data.edge_index)

                # Get anomaly probability (class 1)
                anomaly_prob = torch.exp(out)[:, 1].mean().item()

            details = {
                'graph_num_nodes': graph_data.num_nodes,
                'graph_num_edges': graph_data.num_edges,
                'gnn_output': out.tolist()
            }

            return anomaly_prob, details

        except Exception as e:
            logger.error(f"GNN analysis failed: {e}")
            return 0.0, {'error': str(e)}

    def _build_graph_from_data(
        self,
        data: np.ndarray,
        tda_details: Dict[str, Any]
    ):
        """
        Build PyG graph from data using TDA features as node attributes.

        Args:
            data: Input data matrix
            tda_details: TDA analysis results

        Returns:
            PyG Data object
        """
        import torch
        from torch_geometric.data import Data
        from sklearn.neighbors import NearestNeighbors

        # Downsample if too many nodes
        max_nodes = 500
        if data.shape[0] > max_nodes:
            indices = np.random.choice(data.shape[0], max_nodes, replace=False)
            data = data[indices]

        # Node features: concatenate original data with TDA features
        tda_features = np.array([
            tda_details['h0_count'],
            tda_details['h1_count'],
            tda_details['h2_count']
        ]).reshape(1, -1)

        # Broadcast TDA features to all nodes
        tda_features_broadcast = np.tile(tda_features, (data.shape[0], 1))

        # Concatenate features
        node_features = np.hstack([data, tda_features_broadcast])

        # Build k-NN graph (k=5)
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(data)
        distances, indices = nbrs.kneighbors(data)

        # Create edge list
        edge_index = []
        for i in range(len(indices)):
            for j in indices[i]:
                if i != j:  # Avoid self-loops
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(node_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index)

    def _compute_hybrid_score(
        self,
        tda_score: float,
        gnn_score: float
    ) -> Tuple[float, str]:
        """
        Compute hybrid anomaly score from TDA and GNN.

        Args:
            tda_score: TDA anomaly score (0-1)
            gnn_score: GNN anomaly score (0-1)

        Returns:
            Tuple of (final_score, severity)
        """
        if gnn_score > 0:
            # Both TDA and GNN available: weighted average
            final_score = 0.6 * tda_score + 0.4 * gnn_score
        else:
            # TDA only
            final_score = tda_score

        # Determine severity
        if final_score > 0.8:
            severity = "CRITICAL"
        elif final_score > 0.6:
            severity = "HIGH"
        elif final_score > 0.4:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        return final_score, severity

    def reset(self) -> None:
        """Reset engine state."""
        # Clear cached models
        self._tda_engine = None
        self._gnn_model = None
        if self.isolation_forest is not None:
            self.isolation_forest.reset()
        logger.info("Hybrid engine reset")

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            'tda_threshold': self.tda_threshold,
            'gnn_threshold': self.gnn_threshold,
            'use_gnn': self.use_gnn,
            'pyg_available': self._pyg_available,
            'gnn_layers': self.gnn_layers,
            'tda_max_dimension': self.tda_max_dimension,
            'max_tda_points': self.max_tda_points,
            'tda_loaded': self._tda_engine is not None,
            'gnn_loaded': self._gnn_model is not None,
            'isolation_forest': (
                self.isolation_forest.get_status()
                if self.isolation_forest is not None
                else {"enabled": False}
            ),
        }


class LightweightGNN:
    """
    Standalone lightweight GNN for graph anomaly detection.
    Can be used independently or via HybridTdaGnnEngine.
    """

    def __init__(self, num_features: int = 10, hidden_dim: int = 64, num_layers: int = 2):
        """
        Initialize lightweight GNN.

        Args:
            num_features: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of GCN layers (2-3 recommended)
        """
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self._model = None
        self._pytorch_available = self._check_pytorch_available()

    def _check_pytorch_available(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False

    def build_model(self):
        """Build the GNN model."""
        if not self._pytorch_available:
            raise ImportError("PyTorch not available")

        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv

        class GNNModel(torch.nn.Module):
            def __init__(self, num_features, hidden_dim, num_layers):
                super().__init__()
                self.convs = torch.nn.ModuleList()
                self.convs.append(GCNConv(num_features, hidden_dim))
                for _ in range(num_layers - 2):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.convs.append(GCNConv(hidden_dim, 2))

            def forward(self, x, edge_index):
                for i, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if i < len(self.convs) - 1:
                        x = F.relu(x)
                return F.log_softmax(x, dim=1)

        self._model = GNNModel(self.num_features, self.hidden_dim, self.num_layers)
        return self._model
