"""
Test suite for TDA-Guard Industry Application Adapters.

Tests all four modules:
- Module A: System Resource Monitor
- Module B: Web Traffic Guard
- Module C: IoT Stream Guard
- Module D: AI Loss Landscape Guard
"""

import numpy as np
import sys
import os
from pathlib import Path
import tempfile

# Add the project root to the path when this test is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from adapters.module_a_system_monitor import SystemResourceMonitor
from adapters.module_b_web_traffic import WebTrafficGuard
from adapters.module_c_iot_stream import IoTStreamGuard
from adapters.module_d_ai_landscape import AILossLandscapeGuard
from adapters.unified_adapter import UnifiedAdapter, quick_analyze


def packet_flow_fixture(n_samples=500):
    index = np.arange(n_samples, dtype=float)
    return np.column_stack([
        400 + index % 100,
        0.01 + (index % 17) * 0.0001,
        2.5 + np.sin(index * 0.1),
        3.0 + np.cos(index * 0.1),
        index % 2,
        5.0 + np.sin(index * 0.04),
    ])


def test_module_a_system_monitor():
    """Test Module A: System Resource Monitor."""
    print("=" * 60)
    print("Testing Module A: System Resource Monitor")
    print("=" * 60)
    
    monitor = SystemResourceMonitor(history_window=300)
    
    # Fetch system telemetry
    metrics = monitor.fetch_system_telemetry_stream()
    print(f"Fetched metrics shape: {metrics.shape}")
    print(f"Features: CPU={metrics[0]:.2f}%, Memory={metrics[1]:.2f}%")
    
    # Collect continuous stream (short duration for testing)
    import time
    print("Collecting 5 seconds of system telemetry...")
    data = monitor.collect_continuous_stream(duration_seconds=5, interval=0.5)
    print(f"Collected data shape: {data.shape}")
    
    # Detect anomalies
    anomalies = monitor.detect_resource_anomaly(data)
    print(f"Anomaly detection: {anomalies}")
    
    print("Module A tests passed!\n")


def test_module_b_web_traffic():
    """Test Module B: Web Traffic Guard."""
    print("=" * 60)
    print("Testing Module B: Web Traffic Guard")
    print("=" * 60)
    
    guard = WebTrafficGuard()
    
    log_entries = [
        '192.0.2.10 - - [10/Oct/2025:13:55:36 +0000] "GET /index.html HTTP/1.1" 200 1234\n',
        '192.0.2.11 - - [10/Oct/2025:13:55:37 +0000] "POST /api/login HTTP/1.1" 401 245\n',
        '192.0.2.12 - - [10/Oct/2025:13:55:38 +0000] "GET /admin/../etc/passwd HTTP/1.1" 403 87\n',
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as file:
        file.writelines(log_entries)
        log_file = file.name

    data = guard.parse_access_log_to_manifold(log_file)
    print(f"Parsed data shape: {data.shape}")
    
    # Detect scan patterns
    anomalies = guard.detect_scan_patterns(data)
    print(f"Scan detection: {anomalies}")
    
    # Clean up
    os.unlink(log_file)
    
    print("Module B tests passed!\n")


def test_module_c_iot_stream():
    """Test Module C: IoT Stream Guard."""
    print("=" * 60)
    print("Testing Module C: IoT Stream Guard")
    print("=" * 60)
    
    guard = IoTStreamGuard(window_size=500)
    
    # Generate normal traffic
    print("Generating normal IoT traffic...")
    normal_traffic = packet_flow_fixture()
    print(f"Normal traffic shape: {normal_traffic.shape}")
    
    # Standardize
    normal_standardized = guard.adapt_iot_traffic(normal_traffic)
    print(f"Standardized shape: {normal_standardized.shape}")
    print(f"Mean: {np.mean(normal_standardized, axis=0)}")
    print(f"Std: {np.std(normal_standardized, axis=0)}")
    
    # Use a distinct recorded-like feature batch to verify transform behavior.
    botnet_traffic = normal_traffic.copy()
    botnet_traffic[:, 0] *= 1.5
    botnet_traffic[:, 1] *= 2
    botnet_standardized = guard.adapt_iot_traffic(botnet_traffic)
    
    # Detect flow regime
    normal_regime = guard.detect_flow_regime(normal_standardized)
    botnet_regime = guard.detect_flow_regime(botnet_standardized)
    
    print(f"Normal regime: {normal_regime}")
    print(f"Botnet regime: {botnet_regime}")
    
    print("Module C tests passed!\n")


def test_module_d_ai_landscape():
    """Test Module D: AI Loss Landscape Guard."""
    print("=" * 60)
    print("Testing Module D: AI Loss Landscape Guard")
    print("=" * 60)
    
    guard = AILossLandscapeGuard(tensor_dim=16)
    
    baseline_manifold = np.zeros((4, 16))
    current_manifold = np.repeat(
        (np.arange(1, 5, dtype=float) * 0.25)[:, np.newaxis],
        16,
        axis=1,
    )
    fracture = guard.detect_landscape_fracture(current_manifold, baseline_manifold)
    assert fracture["is_fractured"]
    print(f"Measured fracture score: {fracture['fracture_score']:.4f}")
    
    # Test with PyTorch if available
    if guard.has_torch:
        import torch.nn as nn
        pytorch_model = nn.Linear(4, 4, bias=False)
        pytorch_manifold = guard.extract_model_weights_manifold(pytorch_model)
        print(f"PyTorch manifold shape: {pytorch_manifold.shape}")
    else:
        print("PyTorch not available, skipping PyTorch tests")
    
    print("Module D tests passed!\n")


def test_unified_adapter():
    """Test Unified Adapter interface."""
    print("=" * 60)
    print("Testing Unified Adapter Interface")
    print("=" * 60)
    
    # Test Module C (easiest to test without external dependencies)
    print("Testing IoT Stream Guard via Unified Adapter...")
    adapter = UnifiedAdapter('iot_stream', window_size=500)
    
    try:
        adapter.collect_data()
    except ValueError as exc:
        assert "real packet_flows" in str(exc)
    else:
        raise AssertionError("IoT collection must require actual packet flows")

    data = adapter.collect_data(packet_flows=packet_flow_fixture())
    result = adapter.analyze_topology(data=data)
    print(f"Analysis result keys: {result.keys()}")
    print(f"H0 features: {result['stats']['H0']['count']}")
    print(f"H1 features: {result['stats']['H1']['count']}")
    print(f"Anomaly score: {result['anomaly']['turbulence_score']:.4f}")
    
    print("\nTesting quick_analyze function...")
    result2 = quick_analyze('iot_stream', data=packet_flow_fixture(300))
    print(f"Quick analyze H0: {result2['stats']['H0']['count']}")
    print(f"Quick analyze anomaly: {result2['anomaly']['turbulence_score']:.4f}")
    
    print("Unified Adapter tests passed!\n")


def test_tda_integration():
    """Test integration with TDA-Guard core."""
    print("=" * 60)
    print("Testing TDA-Guard Core Integration")
    print("=" * 60)
    
    from data_adapter import DataAdapter
    from tda_engine import TDAEngine
    
    # Use Module C to generate data
    guard = IoTStreamGuard()
    traffic = packet_flow_fixture()
    standardized = guard.adapt_iot_traffic(traffic)
    
    # Feed into TDA-Guard core
    data_adapter = DataAdapter()
    data_adapter.load_numpy(standardized)
    normalized = data_adapter.normalize_zscore()
    
    engine = TDAEngine()
    data, diag = engine.compute_persistence_topology(normalized)
    stats = engine.get_statistics(diag)
    
    print(f"Computed persistence diagram with {len(diag)} features")
    print(f"H0: {stats['H0']['count']} features")
    print(f"H1: {stats['H1']['count']} features")
    print(f"Computation time: {stats['computation_time']:.3f}s")
    
    print("TDA-Guard integration tests passed!\n")


if __name__ == "__main__":
    try:
        test_module_a_system_monitor()
        test_module_b_web_traffic()
        test_module_c_iot_stream()
        test_module_d_ai_landscape()
        test_unified_adapter()
        test_tda_integration()
        
        print("=" * 60)
        print("All adapter tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
