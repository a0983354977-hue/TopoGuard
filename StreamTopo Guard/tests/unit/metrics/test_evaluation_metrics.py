"""
Test suite for evaluation metrics module.
Validates classification metrics, topology-based metrics, and ensemble metrics.
"""

import numpy as np
from evaluation_metrics import ClassificationMetrics, TopologyBasedMetrics, EnsembleMetrics, EvaluationSuite


def test_classification_metrics():
    """Test standard classification metrics (Precision, Recall, F1)."""
    print("\n[TEST] Classification Metrics")
    metrics = ClassificationMetrics()

    # Simulate predictions
    predictions = [True, True, False, True, False, False, True, False]
    actuals = [True, False, False, True, True, False, True, False]

    for pred, actual in zip(predictions, actuals):
        metrics.update(pred, actual)

    result = metrics.get_metrics()

    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  F1-Score: {result['f1_score']:.4f}")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Confusion Matrix: TP={result['true_positives']}, FP={result['false_positives']}, FN={result['false_negatives']}, TN={result['true_negatives']}")

    # Validate basic properties
    assert 0 <= result['precision'] <= 1, "Precision out of range"
    assert 0 <= result['recall'] <= 1, "Recall out of range"
    assert 0 <= result['f1_score'] <= 1, "F1-Score out of range"
    assert 0 <= result['accuracy'] <= 1, "Accuracy out of range"

    print("[PASS] Classification metrics test")
    return True


def test_topology_based_metrics():
    """Test topology-specific metrics (Betti similarity, persistence distance)."""
    print("\n[TEST] Topology-Based Metrics")

    stats1 = {
        'H0': {'count': 100, 'mean': 0.5, 'std': 0.1, 'max': 1.0, 'min': 0.1},
        'H1': {'count': 50, 'mean': 0.3, 'std': 0.05, 'max': 0.5, 'min': 0.1},
        'H2': {'count': 10, 'mean': 0.2, 'std': 0.02, 'max': 0.3, 'min': 0.1}
    }

    stats2 = {
        'H0': {'count': 105, 'mean': 0.52, 'std': 0.11, 'max': 1.02, 'min': 0.12},
        'H1': {'count': 48, 'mean': 0.29, 'std': 0.04, 'max': 0.48, 'min': 0.09},
        'H2': {'count': 12, 'mean': 0.21, 'std': 0.03, 'max': 0.32, 'min': 0.11}
    }

    similarity = TopologyBasedMetrics.compute_betti_similarity(stats1, stats2)
    print(f"  Betti Similarity: {similarity:.4f}")

    assert 0 <= similarity <= 1, "Similarity out of range"
    assert similarity > 0.8, "Similar statistics should have high similarity"

    # Test persistence distance
    diag1 = [(0, (0.0, 0.5)), (1, (0.1, 0.6))]
    diag2 = [(0, (0.0, 0.6)), (1, (0.1, 0.7))]

    distance = TopologyBasedMetrics.compute_persistence_distance(diag1, diag2)
    print(f"  Persistence Distance: {distance:.4f}")

    assert distance >= 0, "Distance should be non-negative"

    print("[PASS] Topology-based metrics test")
    return True


def test_ensemble_metrics():
    """Test ensemble decision metrics."""
    print("\n[TEST] Ensemble Metrics")
    metrics = EnsembleMetrics()

    # Simulate ensemble decisions
    tda_preds = [True, False, True, False, True]
    z3_preds = [True, False, False, False, True]
    ensemble_preds = [True, False, True, False, True]
    actuals = [True, False, True, True, False]

    for tda, z3, ens, actual in zip(tda_preds, z3_preds, ensemble_preds, actuals):
        metrics.update(tda, z3, ens, actual)

    result = metrics.get_metrics()

    print(f"  TDA Accuracy: {result['tda_accuracy']:.4f}")
    print(f"  Z3 Accuracy: {result['z3_accuracy']:.4f}")
    print(f"  Ensemble Accuracy: {result['ensemble_accuracy']:.4f}")
    print(f"  Ensemble Improvement: {result['ensemble_improvement']:.4f}")

    assert 0 <= result['tda_accuracy'] <= 1, "TDA accuracy out of range"
    assert 0 <= result['z3_accuracy'] <= 1, "Z3 accuracy out of range"
    assert 0 <= result['ensemble_accuracy'] <= 1, "Ensemble accuracy out of range"

    print("[PASS] Ensemble metrics test")
    return True


def test_evaluation_suite():
    """Test complete evaluation suite."""
    print("\n[TEST] Evaluation Suite")
    suite = EvaluationSuite()

    predictions = [True, False, True, False, True]
    actuals = [True, False, True, True, False]
    tda_preds = [True, False, True, False, True]
    z3_preds = [True, False, False, False, True]

    result = suite.evaluate_batch(predictions, actuals, tda_preds, z3_preds)

    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  F1-Score: {result['f1_score']:.4f}")
    print(f"  Ensemble Accuracy: {result['ensemble_accuracy']:.4f}")

    assert 'precision' in result, "Missing precision metric"
    assert 'recall' in result, "Missing recall metric"
    assert 'f1_score' in result, "Missing F1-score metric"
    assert 'ensemble_accuracy' in result, "Missing ensemble accuracy"

    print("[PASS] Evaluation suite test")
    return True


def main():
    """Run all evaluation metrics tests."""
    print("=" * 60)
    print("TDA-Guard Evaluation Metrics Test Suite")
    print("=" * 60)

    try:
        test_classification_metrics()
        test_topology_based_metrics()
        test_ensemble_metrics()
        test_evaluation_suite()

        print("\n" + "=" * 60)
        print("All evaluation metrics tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
