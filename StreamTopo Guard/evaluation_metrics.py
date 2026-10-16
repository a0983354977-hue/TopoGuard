import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter


class ClassificationMetrics:
    """
    Standardized evaluation metrics for anomaly detection.
    Computes Precision, Recall, F1-Score, and confusion matrix.
    """

    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

    def update(self, predicted: bool, actual: bool) -> None:
        """
        Update metrics with a single prediction.

        Args:
            predicted: Predicted anomaly (True) or normal (False)
            actual: Actual anomaly (True) or normal (False)
        """
        if predicted and actual:
            self.true_positives += 1
        elif predicted and not actual:
            self.false_positives += 1
        elif not predicted and actual:
            self.false_negatives += 1
        else:
            self.true_negatives += 1

    def get_metrics(self) -> Dict[str, float]:
        """
        Compute and return all metrics.

        Returns:
            Dict with precision, recall, f1, accuracy, and confusion matrix
        """
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0.0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (self.true_positives + self.true_negatives) / (self.true_positives + self.false_positives + self.true_negatives + self.false_negatives) if (self.true_positives + self.false_positives + self.true_negatives + self.false_negatives) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives
        }

    def reset(self) -> None:
        """Reset all metrics counters."""
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0


class TopologyBasedMetrics:
    """
    Topology-specific evaluation metrics for TDA-based anomaly detection.
    """

    @staticmethod
    def compute_betti_similarity(
        stats1: Dict[str, Dict[str, float]],
        stats2: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Compute similarity between two topological feature sets using Betti numbers.

        Args:
            stats1: First TDA statistics
            stats2: Second TDA statistics

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        h0_diff = abs(stats1.get('H0', {}).get('count', 0) - stats2.get('H0', {}).get('count', 0))
        h1_diff = abs(stats1.get('H1', {}).get('count', 0) - stats2.get('H1', {}).get('count', 0))
        h2_diff = abs(stats1.get('H2', {}).get('count', 0) - stats2.get('H2', {}).get('count', 0))

        total_diff = h0_diff + h1_diff + h2_diff
        max_count = max(
            stats1.get('H0', {}).get('count', 0) + stats2.get('H0', {}).get('count', 0),
            stats1.get('H1', {}).get('count', 0) + stats2.get('H1', {}).get('count', 0),
            stats1.get('H2', {}).get('count', 0) + stats2.get('H2', {}).get('count', 0)
        )

        similarity = 1.0 - (total_diff / (max_count + 1e-8))
        return max(0.0, min(1.0, similarity))

    @staticmethod
    def compute_persistence_distance(
        diag1: List[Tuple[int, Tuple[float, float]]],
        diag2: List[Tuple[int, Tuple[float, float]]]
    ) -> float:
        """
        Compute Wasserstein-like distance between persistence diagrams.

        Args:
            diag1: First persistence diagram
            diag2: Second persistence diagram

        Returns:
            Distance score (lower is more similar)
        """
        if not diag1 and not diag2:
            return 0.0

        # Simplified distance using interval statistics
        intervals1 = [(death - birth) for _, (birth, death) in diag1 if death != float('inf')]
        intervals2 = [(death - birth) for _, (birth, death) in diag2 if death != float('inf')]

        if not intervals1 and not intervals2:
            return 0.0

        mean1 = np.mean(intervals1) if intervals1 else 0.0
        mean2 = np.mean(intervals2) if intervals2 else 0.0

        return abs(mean1 - mean2)


class EnsembleMetrics:
    """
    Metrics for ensemble decision system (TDA + Z3).
    """

    def __init__(self):
        self.tda_only_correct = 0
        self.z3_only_correct = 0
        self.ensemble_correct = 0
        self.total_samples = 0

    def update(
        self,
        tda_prediction: bool,
        z3_prediction: bool,
        ensemble_prediction: bool,
        actual: bool
    ) -> None:
        """
        Update ensemble metrics.

        Args:
            tda_prediction: TDA-only prediction
            z3_prediction: Z3-only prediction
            ensemble_prediction: Ensemble prediction
            actual: Ground truth
        """
        self.total_samples += 1

        if tda_prediction == actual:
            self.tda_only_correct += 1
        if z3_prediction == actual:
            self.z3_only_correct += 1
        if ensemble_prediction == actual:
            self.ensemble_correct += 1

    def get_metrics(self) -> Dict[str, float]:
        """
        Compute ensemble-specific metrics.

        Returns:
            Dict with accuracy metrics for each decision strategy
        """
        return {
            'tda_accuracy': self.tda_only_correct / self.total_samples if self.total_samples > 0 else 0.0,
            'z3_accuracy': self.z3_only_correct / self.total_samples if self.total_samples > 0 else 0.0,
            'ensemble_accuracy': self.ensemble_correct / self.total_samples if self.total_samples > 0 else 0.0,
            'ensemble_improvement': (self.ensemble_correct - self.tda_only_correct) / self.total_samples if self.total_samples > 0 else 0.0
        }

    def reset(self) -> None:
        """Reset all counters."""
        self.tda_only_correct = 0
        self.z3_only_correct = 0
        self.ensemble_correct = 0
        self.total_samples = 0


class EvaluationSuite:
    """
    Complete evaluation suite combining all metrics.
    """

    def __init__(self):
        self.classification = ClassificationMetrics()
        self.ensemble = EnsembleMetrics()

    def evaluate_batch(
        self,
        predictions: List[bool],
        actuals: List[bool],
        tda_predictions: Optional[List[bool]] = None,
        z3_predictions: Optional[List[bool]] = None
    ) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.

        Args:
            predictions: Ensemble predictions
            actuals: Ground truth labels
            tda_predictions: Optional TDA-only predictions
            z3_predictions: Optional Z3-only predictions

        Returns:
            Dict with all metrics
        """
        for pred, actual in zip(predictions, actuals):
            self.classification.update(pred, actual)

        if tda_predictions and z3_predictions:
            for tda_pred, z3_pred, ens_pred, actual in zip(tda_predictions, z3_predictions, predictions, actuals):
                self.ensemble.update(tda_pred, z3_pred, ens_pred, actual)

        metrics = self.classification.get_metrics()
        metrics.update(self.ensemble.get_metrics())

        return metrics

    def reset(self) -> None:
        """Reset all metrics."""
        self.classification.reset()
        self.ensemble.reset()
