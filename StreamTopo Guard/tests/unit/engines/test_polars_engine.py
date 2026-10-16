"""
Test Suite for Polars Data Pipeline Engine
-------------------------------------------
Tests high-frequency data processing, sliding windows, and memory management.
"""

import numpy as np
import time
from polars_engine import PolarsDataPipeline, RollingBaseline


def test_polars_basic_ingestion():
    """Test basic data ingestion and processing."""
    print("[TEST] Polars Basic Ingestion")

    pipeline = PolarsDataPipeline(window_size=100, stride=50)

    # Build a test fixture for streaming ingestion.
    stream_data = [
        {'timestamp': i, 'value': np.random.randn(), 'feature': i % 10}
        for i in range(500)
    ]

    pipeline.ingest_stream(stream_data)
    matrix = pipeline.process()

    assert matrix.shape[0] > 0, "Matrix should not be empty"
    assert matrix.shape[1] > 0, "Matrix should have features"
    print(f"  [PASS] Processed matrix shape: {matrix.shape}")


def test_polars_sliding_windows():
    """Test sliding window processing."""
    print("[TEST] Polars Sliding Windows")

    pipeline = PolarsDataPipeline(window_size=100, stride=50)

    # Generate high-frequency data
    stream_data = [
        {'timestamp': i, 'value': np.random.randn()}
        for i in range(1000)
    ]

    def dummy_process(window):
        return window.shape[0]

    results = pipeline.process_sliding_windows(
        process_func=dummy_process,
        raw_stream=stream_data
    )

    expected_windows = pipeline.get_window_count(1000)
    assert len(results) == expected_windows, f"Expected {expected_windows} windows, got {len(results)}"
    print(f"  [PASS] Processed {len(results)} windows")


def test_polars_rolling_statistics():
    """Test rolling statistics computation."""
    print("[TEST] Polars Rolling Statistics")

    pipeline = PolarsDataPipeline(window_size=100, stride=50)

    # Generate data with known pattern
    stream_data = [
        {'value': float(i)}
        for i in range(500)
    ]

    matrix = pipeline.process(raw_stream=stream_data)

    # Rolling statistics should add features
    assert matrix.shape[1] > 1, "Rolling statistics should add features"
    print(f"  [PASS] Matrix with rolling stats: {matrix.shape}")


def test_polars_memory_management():
    """Test memory management with large data."""
    print("[TEST] Polars Memory Management")

    pipeline = PolarsDataPipeline(window_size=1000, stride=500, max_memory_mb=10)

    # Generate large dataset (simulating high-frequency stream)
    large_stream = [
        {'timestamp': i, 'value': np.random.randn(), 'feature': i % 100}
        for i in range(10000)
    ]

    start_time = time.time()
    pipeline.ingest_stream(large_stream)
    matrix = pipeline.process()
    elapsed = time.time() - start_time

    assert elapsed < 5.0, f"Processing took too long: {elapsed:.2f}s"
    print(f"  [PASS] Processed 10k samples in {elapsed:.2f}s")
    print(f"  [PASS] Memory usage: {pipeline.get_status()['memory_usage_mb']:.2f} MB")


def test_polars_rolling_baseline():
    """Test RollingBaseline computation."""
    print("[TEST] Polars Rolling Baseline")

    baseline = RollingBaseline(window_size=100)

    # Generate scores with anomaly
    scores = np.concatenate([
        np.random.normal(0, 0.1, 500),  # Normal
        np.random.normal(2.0, 0.1, 50)  # Anomaly
    ])

    stats = baseline.update(scores)

    assert 'mean' in stats, "Stats should include mean"
    assert 'std' in stats, "Stats should include std"
    assert 'threshold' in stats, "Stats should include threshold"
    print(f"  [PASS] Baseline - Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    print(f"  [PASS] Threshold: {stats['threshold']:.4f}")


def test_polars_high_frequency_stress():
    """Stress test with high-frequency data (10k+ events/sec simulation)."""
    print("[TEST] Polars High-Frequency Stress")

    pipeline = PolarsDataPipeline(window_size=1000, stride=500)

    # Simulate 10k events
    high_freq_stream = [
        {'timestamp': i, 'value': np.random.randn(), 'cpu': np.random.uniform(0, 100)}
        for i in range(10000)
    ]

    start_time = time.time()

    # Process in batches
    batch_size = 1000
    for i in range(0, len(high_freq_stream), batch_size):
        batch = high_freq_stream[i:i+batch_size]
        pipeline.ingest_stream(batch)

    matrix = pipeline.process()
    elapsed = time.time() - start_time

    throughput = len(high_freq_stream) / elapsed
    print(f"  [PASS] Throughput: {throughput:.0f} events/sec")
    assert throughput > 1000, f"Throughput too low: {throughput:.0f} events/sec"


def test_polars_empty_data():
    """Test handling of empty data."""
    print("[TEST] Polars Empty Data Handling")

    pipeline = PolarsDataPipeline(window_size=100, stride=50)

    matrix = pipeline.process()
    assert matrix.shape == (0, 0), "Empty data should return empty matrix"
    print("  [PASS] Empty data handled correctly")


def test_polars_enforces_memory_budget():
    max_memory_mb = 0.0003
    pipeline = PolarsDataPipeline(
        window_size=100,
        stride=10,
        max_memory_mb=max_memory_mb,
    )
    pipeline.ingest_stream([
        {'value': float(index), 'feature': float(index + 1)}
        for index in range(100)
    ])
    pipeline.process()

    assert pipeline._lazy_frame is not None
    frame = pipeline._lazy_frame.collect()
    assert frame.height < 100
    assert frame.estimated_size() <= int(max_memory_mb * 1024 * 1024)
    assert pipeline.get_status()['memory_usage_mb'] <= max_memory_mb


def test_polars_rejects_invalid_memory_limits():
    for kwargs in (
        {'window_size': 0},
        {'stride': 0},
        {'max_memory_mb': 0},
    ):
        try:
            PolarsDataPipeline(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid pipeline limits must be rejected: {kwargs}")


def test_polars_fails_when_one_row_exceeds_memory_budget():
    pipeline = PolarsDataPipeline(
        window_size=10,
        stride=1,
        max_memory_mb=0.000001,
    )
    try:
        pipeline.ingest_stream([{'value': 1.0}])
    except MemoryError as exc:
        assert "single Polars row" in str(exc)
    else:
        raise AssertionError("An oversized row must not bypass the configured memory limit")


def test_polars_enforces_float64_schema_and_rejects_schema_drift():
    pipeline = PolarsDataPipeline(window_size=4, stride=2)
    pipeline.ingest_stream([
        {"cpu": index, "memory": index + 1}
        for index in range(4)
    ])
    matrix = pipeline.process()

    assert matrix.dtype == np.float64
    assert matrix.shape == (4, 6)
    try:
        pipeline.ingest_stream([{"cpu": 5, "disk": 6}])
    except ValueError as exc:
        assert "schema changed" in str(exc)
    else:
        raise AssertionError("Feature schema drift must be rejected")

    try:
        PolarsDataPipeline(window_size=4, max_memory_mb=0.000001).ingest_stream([
            {"a": 1.0, "b": 2.0}
        ])
    except MemoryError as exc:
        assert "single Polars row" in str(exc)
    else:
        raise AssertionError("An oversized feature row must be rejected before allocation")


def test_polars_reset():
    """Test pipeline reset functionality."""
    print("[TEST] Polars Reset")

    pipeline = PolarsDataPipeline(window_size=100, stride=50)

    # Add data
    stream_data = [{'value': i} for i in range(100)]
    pipeline.ingest_stream(stream_data)
    assert len(pipeline._buffer) > 0, "Buffer should have data"

    # Reset
    pipeline.reset()
    assert len(pipeline._buffer) == 0, "Buffer should be empty after reset"
    assert pipeline._lazy_frame is None, "LazyFrame should be None after reset"
    print("  [PASS] Pipeline reset successful")


def test_polars_integration_with_defensive():
    """Test integration with DefensiveWrapper."""
    print("[TEST] Polars Integration with DefensiveWrapper")

    from defensive import DefensiveWrapper

    wrapper = DefensiveWrapper(use_polars=True)

    # Generate stream data
    stream_data = [np.random.randn() for _ in range(1000)]

    def dummy_tda(window):
        return window.shape[0]

    results = wrapper.process_with_polars(stream_data, dummy_tda)

    assert len(results) > 0, "Should process windows"
    print(f"  [PASS] Integration processed {len(results)} windows")


def test_topoguard_engine_polars_and_numpy_window_paths():
    from defensive import TopoGuardEngine

    stream = [{"cpu": float(index), "memory": float(index * 2)} for index in range(10)]
    process = lambda window: window.shape

    polars_engine = TopoGuardEngine(
        use_ebpf=False,
        use_polars=True,
        use_hybrid=False,
        use_z3=False,
        window_size=4,
        stride=2,
    )
    polars_results = polars_engine.process_with_polars(stream, process)
    assert len(polars_results) == 4

    numpy_engine = TopoGuardEngine(
        use_ebpf=False,
        use_polars=False,
        use_hybrid=False,
        use_z3=False,
        window_size=4,
        stride=2,
    )
    numpy_results = numpy_engine.process_with_polars(np.arange(10), process)
    assert len(numpy_results) == 4


def run_all_tests():
    """Run all Polars engine tests."""
    print("=" * 60)
    print("Polars Data Pipeline Test Suite")
    print("=" * 60)

    tests = [
        test_polars_basic_ingestion,
        test_polars_sliding_windows,
        test_polars_rolling_statistics,
        test_polars_memory_management,
        test_polars_rolling_baseline,
        test_polars_high_frequency_stress,
        test_polars_empty_data,
        test_polars_enforces_memory_budget,
        test_polars_rejects_invalid_memory_limits,
        test_polars_fails_when_one_row_exceeds_memory_budget,
        test_polars_enforces_float64_schema_and_rejects_schema_drift,
        test_polars_reset,
        test_polars_integration_with_defensive,
        test_topoguard_engine_polars_and_numpy_window_paths,
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
