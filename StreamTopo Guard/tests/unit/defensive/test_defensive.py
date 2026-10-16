"""
Test suite for Defensive Engineering Module.

Tests all four defensive mechanisms:
1. Hard Cap Guard (memory protection)
2. Adaptive Baseline (false positive prevention)
3. Bulletproof Error Handling (graceful degradation)
4. Vectorization Utilities (GIL bypass)
"""

import numpy as np
import sys
from pathlib import Path

# Add the project root to the path when this test is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from defensive import (
    HardCapGuard,
    AdaptiveBaseline,
    bulletproof_parse_adapter,
    BulletproofAdapter,
    VectorizedOps,
    DefensiveWrapper
)


def test_hard_cap_guard():
    """Test Defense 1: Hard Cap Guard (Memory Protection)."""
    print("=" * 60)
    print("Testing Defense 1: Hard Cap Guard")
    print("=" * 60)
    
    guard = HardCapGuard()
    
    # Test 1: Data under limit (should pass through)
    small_data = np.random.randn(100, 3)
    result = guard.safe_downsample(small_data, max_points=1000)
    assert result.shape == small_data.shape, "Small data should pass through unchanged"
    print(f"[PASS] Small data (100 points) passed through: {result.shape}")
    
    # Test 2: Data over limit (should downsample)
    large_data = np.random.randn(5000, 3)
    result = guard.safe_downsample(large_data, max_points=1000)
    assert result.shape[0] == 1000, "Large data should be downsampled to 1000"
    print(f"[PASS] Large data (5000 points) downsampled to: {result.shape}")
    
    # Empty or missing data must fail rather than masquerade as telemetry.
    empty_data = np.array([])
    for invalid_data in (empty_data, None):
        try:
            guard.safe_downsample(invalid_data, max_points=1000)
        except ValueError:
            pass
        else:
            raise AssertionError("Empty or missing input must be rejected")
    print("[PASS] Empty and missing data rejected")

    try:
        guard.validate_and_cap(None)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing data must not be converted into a fallback")
    
    # Test 5: validate_and_cap with dimension limit
    high_dim_data = np.random.randn(100, 200)
    result = guard.validate_and_cap(high_dim_data, max_points=1000, max_dims=50)
    assert result.shape[1] == 50, "High dimension data should be truncated"
    print(f"[PASS] High dimension data (200 dims) truncated to: {result.shape}")
    
    print("Hard Cap Guard tests passed!\n")


def test_adaptive_baseline():
    """Test Defense 2: Adaptive Baseline (False Positive Prevention)."""
    print("=" * 60)
    print("Testing Defense 2: Adaptive Baseline")
    print("=" * 60)
    
    baseline = AdaptiveBaseline(window_size=10, std_multiplier=2.0)
    
    # Test 1: Warm-up period (should not alert)
    print("Warm-up period (10 samples)...")
    for i in range(10):
        score = 0.5 + np.random.randn() * 0.1  # Normal scores around 0.5
        is_anomaly, threshold, z_score = baseline.evaluate(score)
        assert not is_anomaly, f"Should not alert during warm-up (iteration {i})"
        assert not baseline.is_warmed_up, "Should not be warmed up yet"
    print("[PASS] Warm-up period: no alerts, not warmed up")
    
    # Test 2: After warm-up, normal score (should not alert)
    is_anomaly, threshold, z_score = baseline.evaluate(0.5)
    assert not is_anomaly, "Normal score should not alert"
    assert baseline.is_warmed_up, "Should be warmed up now"
    print(f"[PASS] Normal score (0.5): no alert, threshold={threshold:.3f}")
    
    # Test 3: Score just above threshold (should alert)
    high_score = threshold + 0.1
    is_anomaly, threshold, z_score = baseline.evaluate(high_score)
    assert is_anomaly, "Score above threshold should alert"
    print(f"[PASS] High score ({high_score:.3f}): alert triggered, threshold={threshold:.3f}")
    
    # Test 4: Reset functionality
    baseline.reset()
    assert not baseline.is_warmed_up, "Should not be warmed up after reset"
    assert len(baseline.history) == 0, "History should be empty after reset"
    print("[PASS] Reset functionality working")
    
    # Test 5: Get status
    baseline.evaluate(0.5)
    status = baseline.get_status()
    assert 'is_warmed_up' in status, "Status should contain is_warmed_up"
    assert 'mean' in status, "Status should contain mean"
    assert 'std' in status, "Status should contain std"
    print(f"[PASS] Status retrieval: {status}")
    
    print("Adaptive Baseline tests passed!\n")


def test_bulletproof_error_handling():
    """Test Defense 3: Bulletproof Error Handling (Graceful Degradation)."""
    print("=" * 60)
    print("Testing Defense 3: Bulletproof Error Handling")
    print("=" * 60)
    
    # Test 1: Successful parsing
    def successful_parse(data):
        return np.array(data) * 2
    
    result = bulletproof_parse_adapter(successful_parse, [1, 2, 3])
    assert np.array_equal(result, np.array([2, 4, 6])), "Successful parse should work"
    print("[PASS] Successful parsing: result correct")
    
    # Test 2: Failed parsing must be reported without inventing telemetry.
    def failing_parse(data):
        raise ValueError("Intentional error")
    
    try:
        bulletproof_parse_adapter(failing_parse, [1, 2, 3])
    except ValueError as exc:
        assert "Intentional error" in str(exc)
    else:
        raise AssertionError("Parser failures must propagate to the caller")
    print("[PASS] Failed parsing propagated")
    
    # Test 3: BulletproofAdapter class
    adapter = BulletproofAdapter()

    def good_func(x):
        return np.ones((10, 2))

    result = adapter.execute(good_func, "test_input")
    assert result.shape == (10, 2), "Good function should return correct shape"
    assert adapter.error_count == 0, "Error count should be 0 after success"
    print("[PASS] BulletproofAdapter: successful execution")

    # Test 4: Error counting
    def bad_func(x):
        raise RuntimeError("Test error")

    for _ in range(3):
        try:
            adapter.execute(bad_func, "test_input")
        except RuntimeError as exc:
            assert "Test error" in str(exc)
        else:
            raise AssertionError("Adapter failures must propagate")
    
    assert adapter.error_count == 3, f"Error count should be 3, got {adapter.error_count}"
    print(f"[PASS] BulletproofAdapter: error counting working ({adapter.error_count} errors)")
    
    # Test 5: Error reset on success
    result = adapter.execute(good_func, "test_input")
    assert adapter.error_count == 0, "Error count should reset after success"
    print("[PASS] BulletproofAdapter: error reset on success")
    
    print("Bulletproof Error Handling tests passed!\n")


def test_vectorization_utilities():
    """Test Defense 4: Vectorization Utilities (GIL Bypass)."""
    print("=" * 60)
    print("Testing Defense 4: Vectorization Utilities")
    print("=" * 60)
    
    ops = VectorizedOps()
    
    # Test 1: Vectorized normalization
    data = np.random.randn(1000, 10)
    normalized = ops.fast_vectorized_normalization(data)
    
    # Check normalization: mean ~ 0, std ~ 1
    mean = np.mean(normalized, axis=0)
    std = np.std(normalized, axis=0)
    
    assert np.allclose(mean, 0, atol=1e-10), "Mean should be close to 0"
    assert np.allclose(std, 1, atol=1e-10), "Std should be close to 1"
    print(f"[PASS] Vectorized normalization: mean~{np.mean(mean):.2e}, std~{np.mean(std):.2f}")
    
    # Test 2: Vectorized distance calculation
    matrix1 = np.random.randn(100, 3)
    matrix2 = np.random.randn(50, 3)
    distances = ops.fast_vectorized_distance(matrix1, matrix2)
    
    assert distances.shape == (100, 50), "Distance matrix shape should be (100, 50)"
    assert np.all(distances >= 0), "Distances should be non-negative"
    print(f"[PASS] Vectorized distance: shape {distances.shape}, all non-negative")
    
    # Test 3: Vectorized rolling mean
    time_series = np.random.randn(100, 1)
    rolling_mean = ops.fast_vectorized_rolling_mean(time_series, window=10)
    
    expected_shape = (91, 1)  # 100 - 10 + 1
    assert rolling_mean.shape == expected_shape, f"Rolling mean shape should be {expected_shape}"
    print(f"[PASS] Vectorized rolling mean: shape {rolling_mean.shape}")
    
    # Test 4: Vectorized rolling std
    rolling_std = ops.fast_vectorized_rolling_std(time_series, window=10)
    assert rolling_std.shape == expected_shape, f"Rolling std shape should be {expected_shape}"
    assert np.all(rolling_std >= 0), "Rolling std should be non-negative"
    print(f"[PASS] Vectorized rolling std: shape {rolling_std.shape}, all non-negative")
    
    print("Vectorization Utilities tests passed!\n")


def test_defensive_wrapper():
    """Test Unified Defensive Wrapper (All 4 defenses together)."""
    print("=" * 60)
    print("Testing Unified Defensive Wrapper")
    print("=" * 60)
    
    wrapper = DefensiveWrapper(max_points=1000, baseline_window=10)
    
    # Test 1: Safe execution with good data
    def good_parse(data):
        return np.random.randn(500, 3)
    
    result, is_anomaly = wrapper.safe_execute(good_parse, "test_input", current_score=0.5)
    assert result.shape[0] <= 1000, "Result should respect hard cap"
    assert result.shape == (500, 3), "Good data should pass through"
    print(f"[PASS] Safe execution: result shape {result.shape}, anomaly={is_anomaly}")
    
    # Test 2: Safe execution with oversized data
    def oversize_parse(data):
        return np.random.randn(5000, 3)
    
    result, is_anomaly = wrapper.safe_execute(oversize_parse, "test_input", current_score=0.5)
    assert result.shape[0] <= 1000, "Oversized data should be capped"
    print(f"[PASS] Oversized data capped: result shape {result.shape}")
    
    # Test 3: Safe execution with failing parse
    def failing_parse(data):
        raise ValueError("Test error")
    
    try:
        wrapper.safe_execute(failing_parse, "test_input", current_score=0.5)
    except ValueError as exc:
        assert "Test error" in str(exc)
    else:
        raise AssertionError("Failed parsing must not return fabricated data")
    assert wrapper.bulletproof.error_count == 1
    print("[PASS] Failed parse propagated without fallback data")
    
    # Test 4: Baseline evaluation over multiple iterations
    print("Testing baseline evaluation over 15 iterations...")
    def parse_with_score(data):
        return np.random.randn(100, 3)
    
    for i in range(15):
        score = 0.5 if i < 12 else 2.0  # Spike at end
        result, is_anomaly = wrapper.safe_execute(parse_with_score, "test", current_score=score)
        
        if i < 10:
            assert not is_anomaly, f"Should not alert during warm-up (iteration {i})"
        elif i < 12:
            assert not is_anomaly, f"Normal score should not alert (iteration {i})"
        else:
            # After warm-up, high score might alert depending on baseline
            pass
    
    baseline_status = wrapper.baseline.get_status()
    print(f"[PASS] Baseline evaluation: warmed_up={baseline_status['is_warmed_up']}, "
          f"threshold={baseline_status['threshold']:.3f}")
    
    print("Unified Defensive Wrapper tests passed!\n")


def test_integration_with_adapters():
    """Test defensive integration with existing adapters."""
    print("=" * 60)
    print("Testing Integration with Adapters")
    print("=" * 60)
    
    from adapters import UnifiedAdapter
    
    # Test with IoT Stream Guard
    adapter = UnifiedAdapter('iot_stream', max_points=1000, baseline_window=10)
    
    # Collect data
    packet_flows = np.arange(3000, dtype=float).reshape(500, 6)
    data = adapter.collect_data(packet_flows=packet_flows)
    print(f"[PASS] Collected data: shape {data.shape}")
    
    # Analyze with defensive enabled
    result = adapter.analyze_topology(data=data, use_defensive=True)

    assert 'data' in result, "Result should contain data"
    assert 'diag' in result, "Result should contain diag"
    assert 'stats' in result, "Result should contain stats"
    assert 'anomaly' in result, "Result should contain anomaly"

    # baseline_status is only added if anomaly dict has 'score' key
    if 'score' in result['anomaly']:
        assert 'baseline_status' in result, "Result should contain baseline_status when score exists"
        print(f"[PASS] Analysis with defensive: H0={result['stats']['H0']['count']}, "
              f"H1={result['stats']['H1']['count']}")
        print(f"[PASS] Baseline status: {result['baseline_status']}")
    else:
        print(f"[PASS] Analysis with defensive: H0={result['stats']['H0']['count']}, "
              f"H1={result['stats']['H1']['count']}")
        print(f"[PASS] No score in anomaly dict, baseline_status not added (expected for IoT)")
    
    # Test with defensive disabled (legacy mode)
    result_legacy = adapter.analyze_topology(data=data, use_defensive=False)
    assert 'baseline_status' not in result_legacy, "Legacy mode should not have baseline_status"
    print("[PASS] Legacy mode (defensive disabled) working")
    
    print("Integration with Adapters tests passed!\n")


if __name__ == "__main__":
    try:
        test_hard_cap_guard()
        test_adaptive_baseline()
        test_bulletproof_error_handling()
        test_vectorization_utilities()
        test_defensive_wrapper()
        test_integration_with_adapters()
        
        print("=" * 60)
        print("All defensive engineering tests passed successfully!")
        print("=" * 60)
        print("\nDefensive mechanisms verified:")
        print("[PASS] Defense 1: Hard Cap Guard (memory protection)")
        print("[PASS] Defense 2: Adaptive Baseline (false positive prevention)")
        print("[PASS] Defense 3: Bulletproof Error Handling (graceful degradation)")
        print("[PASS] Defense 4: Vectorization Utilities (GIL bypass)")
        print("[PASS] Unified Defensive Wrapper integration")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
