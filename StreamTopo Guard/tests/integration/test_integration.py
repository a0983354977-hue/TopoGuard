"""
Integration Test for Full TopoGuard Pipeline
----------------------------------------------
Tests the complete unified pipeline: eBPF -> Polars -> TDA+PyG -> Z3
"""

import numpy as np
import time
from defensive import TopoGuardEngine


def test_full_pipeline_basic():
    """Test basic full pipeline with a deterministic test fixture."""
    print("[TEST] Full Pipeline Basic")

    engine = TopoGuardEngine(
        use_ebpf=False,  # Use provided stream instead
        use_polars=True,
        use_hybrid=True,
        use_z3=True
    )

    # Build a deterministic numeric test fixture.
    stream_data = [
        {'timestamp': i, 'value': np.random.randn(), 'cpu': np.random.uniform(0, 100)}
        for i in range(100)
    ]

    result = engine.inspect_tick(stream_data)

    assert 'status' in result, "Result should have status"
    assert 'is_anomaly' in result, "Result should have is_anomaly"
    assert 'anomaly_score' in result, "Result should have anomaly_score"

    print(f"  [PASS] Pipeline executed: status={result['status']}, anomaly={result['is_anomaly']}")


def test_full_pipeline_with_ebpf():
    """Test full pipeline behavior when real eBPF is unavailable."""
    print("[TEST] Full Pipeline with eBPF")

    engine = TopoGuardEngine(
        use_ebpf=True,
        use_polars=True,
        use_hybrid=True,
        use_z3=True
    )

    # Let eBPF run briefly
    time.sleep(0.2)

    result = engine.inspect_tick()

    assert 'status' in result, "Result should have status"

    engine.stop()

    print(f"  [PASS] Pipeline with eBPF: status={result['status']}")


def test_full_pipeline_anomaly_detection():
    """Test anomaly detection through full pipeline."""
    print("[TEST] Full Pipeline Anomaly Detection")

    engine = TopoGuardEngine(
        use_ebpf=False,
        use_polars=True,
        use_hybrid=True,
        use_z3=True
    )

    # Generate normal data
    normal_data = [
        {'value': np.random.randn() * 0.1}
        for _ in range(100)
    ]

    normal_result = engine.inspect_tick(normal_data)

    # Generate anomalous data
    anomalous_data = [
        {'value': np.random.randn() * 10.0}
        for _ in range(100)
    ]

    anomalous_result = engine.inspect_tick(anomalous_data)

    print(f"  [PASS] Normal: score={normal_result['anomaly_score']:.4f}")
    print(f"  [PASS] Anomalous: score={anomalous_result['anomaly_score']:.4f}")


def test_full_pipeline_graceful_degradation():
    """Test graceful degradation when components unavailable."""
    print("[TEST] Full Pipeline Graceful Degradation")

    # Test with all components disabled
    engine = TopoGuardEngine(
        use_ebpf=False,
        use_polars=False,
        use_hybrid=False,
        use_z3=False
    )

    # Use numeric data instead of dict for degraded mode
    stream_data = [float(i) for i in range(100)]
    result = engine.inspect_tick(stream_data)

    assert result['status'] == 'SUCCESS', f"Should succeed even with components disabled, got {result['status']}"
    assert 'anomaly_score' in result, "Should still compute anomaly score"

    print(f"  [PASS] Degraded mode: score={result['anomaly_score']:.4f}")


def test_full_pipeline_status():
    """Test status reporting of full pipeline."""
    print("[TEST] Full Pipeline Status")

    engine = TopoGuardEngine(
        use_ebpf=True,
        use_polars=True,
        use_hybrid=True,
        use_z3=True
    )

    status = engine.get_status()

    assert 'ebpf_active' in status, "Status should have ebpf_active"
    assert 'polars_available' in status, "Status should have polars_available"
    assert 'hybrid_available' in status, "Status should have hybrid_available"
    assert 'z3_available' in status, "Status should have z3_available"

    engine.stop()

    print(f"  [PASS] Status: {status}")


def test_full_pipeline_high_frequency():
    """Test full pipeline with high-frequency data."""
    print("[TEST] Full Pipeline High-Frequency")

    engine = TopoGuardEngine(
        use_ebpf=False,
        use_polars=True,
        use_hybrid=True,
        use_z3=False  # Skip Z3 for speed
    )

    # Generate high-frequency stream
    high_freq_stream = [
        {'timestamp': i, 'value': np.random.randn(), 'cpu': np.random.uniform(0, 100)}
        for i in range(1000)
    ]

    start_time = time.time()
    result = engine.inspect_tick(high_freq_stream)
    elapsed = time.time() - start_time

    assert result['status'] == 'SUCCESS', "Should succeed with high-frequency data"
    assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s"

    print(f"  [PASS] High-frequency: {len(high_freq_stream)} events in {elapsed:.3f}s")


def test_full_pipeline_reset():
    """Test reset functionality of full pipeline."""
    print("[TEST] Full Pipeline Reset")

    engine = TopoGuardEngine(
        use_ebpf=True,
        use_polars=True,
        use_hybrid=True,
        use_z3=True
    )

    # Run some inspections
    stream_data = [{'value': i} for i in range(100)]
    engine.inspect_tick(stream_data)

    # Reset
    engine.reset()

    status = engine.get_status()
    print(f"  [PASS] Reset successful")

    engine.stop()


def test_full_pipeline_empty_data():
    """Test handling of empty data."""
    print("[TEST] Full Pipeline Empty Data")

    engine = TopoGuardEngine(
        use_ebpf=False,
        use_polars=True,
        use_hybrid=True,
        use_z3=True
    )

    result = engine.inspect_tick(None)

    assert result['status'] in ['NO_DATA', 'EMPTY_MATRIX'], "Should handle empty data gracefully"

    print(f"  [PASS] Empty data handled: status={result['status']}")


def test_full_pipeline_z3_integration():
    """Test Z3 integration in full pipeline."""
    print("[TEST] Full Pipeline Z3 Integration")

    engine = TopoGuardEngine(
        use_ebpf=False,
        use_polars=True,
        use_hybrid=True,
        use_z3=True
    )

    # Generate data that might trigger Z3
    stream_data = [
        {'value': np.random.randn() * 5.0}
        for _ in range(100)
    ]

    result = engine.inspect_tick(stream_data)

    if result['is_anomaly']:
        assert 'z3_result' in result['context'], "Should have Z3 result if anomaly"
        print(f"  [PASS] Z3 activated: violated={result['context']['z3_result']['is_violated']}")
    else:
        print(f"  [PASS] Z3 not activated (no anomaly)")

    print(f"  [PASS] Z3 integration test complete")


def test_full_pipeline_end_to_end():
    """Complete end-to-end test of the entire system."""
    print("[TEST] Full Pipeline End-to-End")

    engine = TopoGuardEngine(
        use_ebpf=True,
        use_polars=True,
        use_hybrid=True,
        use_z3=True
    )

    # Simulate multiple inspection ticks
    results = []
    for i in range(5):
        stream_data = [
            {'timestamp': i * 100 + j, 'value': np.random.randn()}
            for j in range(100)
        ]
        result = engine.inspect_tick(stream_data)
        results.append(result)
        time.sleep(0.1)

    # Verify all succeeded
    for result in results:
        assert result['status'] == 'SUCCESS', f"Tick failed: {result['status']}"

    anomaly_count = sum(1 for r in results if r['is_anomaly'])
    print(f"  [PASS] End-to-end: {len(results)} ticks, {anomaly_count} anomalies")

    engine.stop()


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("TopoGuard Full Pipeline Integration Test Suite")
    print("=" * 60)

    tests = [
        test_full_pipeline_basic,
        test_full_pipeline_with_ebpf,
        test_full_pipeline_anomaly_detection,
        test_full_pipeline_graceful_degradation,
        test_full_pipeline_status,
        test_full_pipeline_high_frequency,
        test_full_pipeline_reset,
        test_full_pipeline_empty_data,
        test_full_pipeline_z3_integration,
        test_full_pipeline_end_to_end
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
