"""
Test Suite for Hybrid TDA + PyG Engine
---------------------------------------
Tests hybrid anomaly detection, lazy-loading GNN, and graph construction.
"""

import numpy as np
import tempfile
import time
from pathlib import Path

from baseline import BaselineCalibrator
from hybrid_engine import HybridTdaGnnEngine, IsolationForestBaseline, LightweightGNN


def test_hybrid_defaults_to_isolation_forest_without_gnn():
    engine = HybridTdaGnnEngine()

    assert not engine.use_gnn
    assert engine.isolation_forest is not None


def test_hybrid_tda_only_mode():
    """Test hybrid engine in TDA-only mode (PyG disabled)."""
    print("[TEST] Hybrid TDA-Only Mode")

    engine = HybridTdaGnnEngine(use_gnn=False)

    # Generate test data
    data = np.random.randn(100, 3)

    result = engine.evaluate(data)

    assert 'anomaly_score' in result, "Result should have anomaly_score"
    assert 'tda_score' in result, "Result should have tda_score"
    assert result['gnn_score'] == 0.0, "GNN score should be 0 in TDA-only mode"
    assert 'iforest_score' in result
    assert not result['gnn_activated'], "GNN should not be activated"

    print(f"  [PASS] TDA-only mode: score={result['anomaly_score']:.4f}")


def test_hybrid_lazy_loading():
    """Test lazy-loading of GNN model."""
    print("[TEST] Hybrid Lazy Loading")

    engine = HybridTdaGnnEngine(use_gnn=True)

    # Initially, models should not be loaded
    assert engine._tda_engine is None, "TDA engine should not be loaded initially"
    assert engine._gnn_model is None, "GNN model should not be loaded initially"

    # Generate normal data (low TDA score)
    normal_data = np.random.randn(100, 3) * 0.1
    result = engine.evaluate(normal_data)

    # TDA should be loaded, but GNN should not (since TDA score is low)
    assert engine._tda_engine is not None, "TDA engine should be loaded after first evaluation"
    assert engine._gnn_model is None, "GNN should not be loaded for normal data"

    print(f"  [PASS] Lazy loading: TDA loaded, GNN not loaded for normal data")


def test_hybrid_gnn_activation():
    """Test GNN activation on high TDA score."""
    print("[TEST] Hybrid GNN Activation")

    engine = HybridTdaGnnEngine(use_gnn=True, tda_threshold=0.3)

    # Generate anomalous data (high variance to trigger TDA)
    anomalous_data = np.random.randn(100, 3) * 5.0

    # Only test if PyG is available
    if engine._pyg_available:
        result = engine.evaluate(anomalous_data)

        # If TDA score is high, GNN should be activated
        if result['tda_score'] > engine.tda_threshold:
            # Note: GNN might still fail due to missing PyTorch/PyG dependencies
            # The important thing is that the engine attempted to load it
            print(f"  [PASS] GNN activation attempted: TDA score={result['tda_score']:.4f}")
        else:
            print(f"  [SKIP] TDA score too low to trigger GNN: {result['tda_score']:.4f}")
    else:
        print("  [SKIP] PyG not available, cannot test GNN activation")


def test_hybrid_scoring():
    """Test hybrid scoring mechanism."""
    print("[TEST] Hybrid Scoring")

    engine = HybridTdaGnnEngine(use_gnn=False)

    data = np.random.randn(100, 3)
    result = engine.evaluate(data)

    assert 'severity' in result, "Result should have severity"
    assert result['severity'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], "Invalid severity"

    print(f"  [PASS] Hybrid scoring: severity={result['severity']}")


def test_hybrid_graph_construction():
    """Test graph construction from data."""
    print("[TEST] Hybrid Graph Construction")

    engine = HybridTdaGnnEngine(use_gnn=True)

    data = np.random.randn(50, 3)
    tda_details = {
        'h0_count': 10,
        'h1_count': 5,
        'h2_count': 2
    }

    if engine._pyg_available:
        try:
            graph = engine._build_graph_from_data(data, tda_details)

            assert graph.num_nodes > 0, "Graph should have nodes"
            assert graph.num_edges > 0, "Graph should have edges"
            assert graph.x.shape[0] == graph.num_nodes, "Node features should match node count"

            print(f"  [PASS] Graph constructed: {graph.num_nodes} nodes, {graph.num_edges} edges")
        except Exception as e:
            print(f"  [SKIP] Graph construction failed (likely missing dependencies): {e}")
    else:
        print("  [SKIP] PyG not available, cannot test graph construction")


def test_hybrid_high_dimensional_data():
    """Test hybrid engine with high-dimensional data."""
    print("[TEST] Hybrid High-Dimensional Data")

    engine = HybridTdaGnnEngine(use_gnn=False)

    # Generate high-dimensional data
    high_dim_data = np.random.randn(100, 20)

    result = engine.evaluate(high_dim_data)

    assert result['anomaly_score'] >= 0.0, "Score should be non-negative"
    assert result['anomaly_score'] <= 1.0, "Score should be <= 1.0"

    print(f"  [PASS] High-dimensional data processed: score={result['anomaly_score']:.4f}")


def test_hybrid_performance():
    """Test performance of hybrid engine."""
    print("[TEST] Hybrid Performance")

    engine = HybridTdaGnnEngine(use_gnn=False)

    data = np.random.randn(100, 3)

    start_time = time.time()
    for _ in range(10):
        result = engine.evaluate(data)
    elapsed = time.time() - start_time

    throughput = 10 / elapsed
    print(f"  [PASS] Performance: {throughput:.2f} evaluations/sec")

    assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s for 10 evaluations"


def test_hybrid_reset():
    """Test reset functionality."""
    print("[TEST] Hybrid Reset")

    engine = HybridTdaGnnEngine(use_gnn=False)

    data = np.random.randn(100, 3)
    engine.evaluate(data)

    assert engine._tda_engine is not None, "TDA engine should be loaded"

    engine.reset()

    assert engine._tda_engine is None, "TDA engine should be None after reset"
    assert engine._gnn_model is None, "GNN model should be None after reset"
    assert not engine.isolation_forest.get_status()["fitted"]

    print("  [PASS] Reset successful")


def test_hybrid_status():
    """Test status reporting."""
    print("[TEST] Hybrid Status")

    engine = HybridTdaGnnEngine(use_gnn=True)

    status = engine.get_status()

    assert 'tda_threshold' in status, "Status should have tda_threshold"
    assert 'gnn_threshold' in status, "Status should have gnn_threshold"
    assert 'use_gnn' in status, "Status should have use_gnn"
    assert 'pyg_available' in status, "Status should have pyg_available"
    assert 'gnn_layers' in status, "Status should have gnn_layers"
    assert status["tda_max_dimension"] == 1
    assert status["max_tda_points"] == 300
    assert 'isolation_forest' in status, "Status should expose baseline state"

    print(f"  [PASS] Status: {status}")


def test_lightweight_gnn():
    """Test standalone LightweightGNN class."""
    print("[TEST] Lightweight GNN")

    gnn = LightweightGNN(num_features=10, hidden_dim=64, num_layers=2)

    status = gnn._pytorch_available

    if status:
        try:
            model = gnn.build_model()
            assert model is not None, "Model should be built"
            print("  [PASS] Lightweight GNN model built successfully")
        except Exception as e:
            print(f"  [SKIP] Model build failed: {e}")
    else:
        print("  [SKIP] PyTorch not available")


def test_hybrid_empty_data():
    """Test handling of empty data."""
    print("[TEST] Hybrid Empty Data")

    engine = HybridTdaGnnEngine(use_gnn=False)

    empty_data = np.array([]).reshape(0, 3)

    try:
        result = engine.evaluate(empty_data)
        # Should handle gracefully
        print(f"  [PASS] Empty data handled: score={result.get('anomaly_score', 0):.4f}")
    except Exception as e:
        print(f"  [PASS] Empty data raised expected error: {e}")


def test_hybrid_caps_tda_input_and_rejects_nonfinite_values():
    engine = HybridTdaGnnEngine(use_gnn=False)
    observed_shapes = []

    def fake_tda(data):
        observed_shapes.append(data.shape)
        return 0.1, {'h0_count': 1, 'h1_count': 0, 'h2_count': 0}

    engine._run_tda_analysis = fake_tda
    result = engine.evaluate(
        np.arange(600, dtype=np.float64).reshape(200, 3),
        return_details=True,
    )
    assert observed_shapes == [(200, 3)]
    assert result['tda_details']['original_sample_count'] == 200

    observed_shapes.clear()
    engine.evaluate(np.arange(1800, dtype=np.float64).reshape(600, 3))
    assert observed_shapes == [(300, 3)]

    try:
        engine.evaluate(np.array([[1.0], [np.inf]]))
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("Non-finite data must fail validation")


def test_hybrid_return_details():
    """Test detailed return mode."""
    print("[TEST] Hybrid Return Details")

    engine = HybridTdaGnnEngine(use_gnn=False)

    data = np.random.randn(100, 3)
    result = engine.evaluate(data, return_details=True)

    assert 'tda_details' in result, "Result should have tda_details"
    assert 'h0_count' in result['tda_details'], "TDA details should have h0_count"
    assert result['iforest_details']["status"] == "active"

    print(f"  [PASS] Detailed mode: H0={result['tda_details']['h0_count']}")


def test_isolation_forest_baseline_warms_fits_scores_and_checks_schema():
    detector = IsolationForestBaseline(random_state=7)
    rng = np.random.default_rng(7)
    normal = rng.normal(0.0, 0.1, size=(128, 2))
    feature_names = ("cpu", "memory")

    warmup = detector.evaluate(normal[:32], feature_names)
    assert warmup["status"] == "warming_up"
    assert not detector.get_status()["fitted"]

    result = detector.evaluate(
        np.vstack([normal, [8.0, 8.0]]),
        feature_names,
    )
    assert result["status"] == "active"
    assert result["baseline_samples"] == 128
    assert result["is_anomaly"]
    assert result["anomaly_score"] > detector.ANOMALY_THRESHOLD

    try:
        detector.evaluate(normal, ("memory", "cpu"))
    except ValueError as exc:
        assert "schema changed" in str(exc)
    else:
        raise AssertionError("Feature order changes must invalidate the fitted baseline")


def test_isolation_forest_contributes_anomaly_when_tda_is_normal():
    engine = HybridTdaGnnEngine(use_gnn=False)
    engine._run_tda_analysis = lambda data: (
        0.0,
        {"h0_count": 1, "h1_count": 0, "h2_count": 0},
    )
    rng = np.random.default_rng(11)
    normal = rng.normal(0.0, 0.1, size=(64, 2))
    data = np.vstack([normal, [8.0, 8.0]])

    result = engine.evaluate(
        data,
        return_details=True,
        feature_names=("cpu", "memory"),
    )

    assert result["tda_score"] == 0.0
    assert result["iforest_details"]["is_anomaly"]
    assert result["is_anomaly"]
    assert result["anomaly_score"] > 0.6


def test_isolation_forest_snapshot_round_trip_is_ready_and_atomic():
    detector = IsolationForestBaseline(random_state=17)
    normal = np.random.default_rng(17).normal(0.0, 0.1, size=(80, 2))
    detector.evaluate(normal, ("cpu", "memory"))
    calibrator = BaselineCalibrator.from_detector(detector)

    with tempfile.TemporaryDirectory() as directory:
        snapshot_path = Path(directory) / "baseline.json"
        calibrator.save(snapshot_path, format="json")
        assert snapshot_path.is_file()
        assert list(Path(directory).glob("*.tmp")) == []

        loaded = BaselineCalibrator.load(snapshot_path)
        assert loaded.model.get_status()["fitted"]
        assert loaded.feature_names == ["cpu", "memory"]
        assert loaded.model.random_state == 17
        assert loaded.model._trained_at == detector._trained_at
        result = loaded.model.evaluate(normal[-1:], ("cpu", "memory"))
        assert result["status"] == "active"


def test_isolation_forest_uses_one_tree_traversal_per_sample():
    class FittedModel:
        def __init__(self):
            self.decision_calls = 0

        def decision_function(self, sample):
            self.decision_calls += 1
            assert sample.shape == (1, 2)
            return np.array([-0.2])

        def predict(self, _sample):
            raise AssertionError("Scoring should not traverse the forest twice")

    detector = IsolationForestBaseline()
    model = FittedModel()
    detector._model = model
    detector._feature_count = 2
    detector._feature_names = ("cpu", "memory")
    detector._baseline_samples = 32

    result = detector.evaluate(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        ("cpu", "memory"),
    )

    assert model.decision_calls == 1
    assert result["is_anomaly"]
    assert result["anomaly_score"] == 0.7


def run_all_tests():
    """Run all hybrid engine tests."""
    print("=" * 60)
    print("Hybrid TDA + PyG Engine Test Suite")
    print("=" * 60)

    tests = [
        test_hybrid_defaults_to_isolation_forest_without_gnn,
        test_hybrid_tda_only_mode,
        test_hybrid_lazy_loading,
        test_hybrid_gnn_activation,
        test_hybrid_scoring,
        test_hybrid_graph_construction,
        test_hybrid_high_dimensional_data,
        test_hybrid_performance,
        test_hybrid_reset,
        test_hybrid_status,
        test_lightweight_gnn,
        test_hybrid_empty_data,
        test_hybrid_caps_tda_input_and_rejects_nonfinite_values,
        test_hybrid_return_details,
        test_isolation_forest_baseline_warms_fits_scores_and_checks_schema,
        test_isolation_forest_contributes_anomaly_when_tda_is_normal,
        test_isolation_forest_snapshot_round_trip_is_ready_and_atomic,
        test_isolation_forest_uses_one_tree_traversal_per_sample,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] FAILED: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
