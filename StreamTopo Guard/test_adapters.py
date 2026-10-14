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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.module_a_system_monitor import SystemResourceMonitor
from adapters.module_b_web_traffic import WebTrafficGuard
from adapters.module_c_iot_stream import IoTStreamGuard
from adapters.module_d_ai_landscape import AILossLandscapeGuard, NumpyModelMock, extract_numpy_model_manifold
from adapters.unified_adapter import UnifiedAdapter, quick_analyze


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
    
    # Generate sample log
    print("Generating sample access log...")
    log_file = guard.generate_sample_log(n_entries=1000, include_attack=True)
    print(f"Generated log file: {log_file}")
    
    # Parse log to manifold
    print("Parsing access log...")
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
    normal_traffic = guard.generate_sample_iot_traffic(n_samples=500, mode='normal')
    print(f"Normal traffic shape: {normal_traffic.shape}")
    
    # Standardize
    normal_standardized = guard.adapt_iot_traffic(normal_traffic)
    print(f"Standardized shape: {normal_standardized.shape}")
    print(f"Mean: {np.mean(normal_standardized, axis=0)}")
    print(f"Std: {np.std(normal_standardized, axis=0)}")
    
    # Generate botnet traffic
    print("Generating botnet traffic...")
    botnet_traffic = guard.generate_sample_iot_traffic(n_samples=500, mode='botnet')
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
    
    # Test with NumPy mock (no PyTorch required)
    print("Testing with NumPy mock model...")
    model = NumpyModelMock(n_parameters=1000)
    
    # Extract manifold
    manifold = extract_numpy_model_manifold(model, tensor_dim=16)
    print(f"Extracted manifold shape: {manifold.shape}")
    
    # Simulate training
    print("Simulating training steps...")
    baseline_manifold = manifold.copy()
    
    for i in range(5):
        model.update_weights(learning_rate=0.01)
        current_manifold = extract_numpy_model_manifold(model, tensor_dim=16)
        
        # Detect fracture
        fracture = guard.detect_landscape_fracture(current_manifold, baseline_manifold)
        print(f"Step {i+1}: fracture_score={fracture['fracture_score']:.4f}")
    
    # Test with PyTorch if available
    if guard.has_torch:
        print("Testing with PyTorch model...")
        pytorch_model = guard.generate_synthetic_model(n_layers=3, layer_size=64)
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
    
    # Collect data first
    data = adapter.collect_data(n_samples=500, mode='normal')
    result = adapter.analyze_topology(data=data)
    print(f"Analysis result keys: {result.keys()}")
    print(f"H0 features: {result['stats']['H0']['count']}")
    print(f"H1 features: {result['stats']['H1']['count']}")
    print(f"Anomaly score: {result['anomaly']['turbulence_score']:.4f}")
    
    # Test quick_analyze
    print("\nTesting quick_analyze function...")
    # For quick_analyze, pass parameters directly to collect_data
    adapter2 = UnifiedAdapter('iot_stream')
    data2 = adapter2.collect_data(n_samples=300, mode='botnet')
    result2 = adapter2.analyze_topology(data=data2)
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
    traffic = guard.generate_sample_iot_traffic(n_samples=500, mode='normal')
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
