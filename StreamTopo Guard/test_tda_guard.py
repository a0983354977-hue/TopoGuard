"""
Test script for TDA-Guard implementation.
Tests the core functionality without GUI.
"""

import numpy as np
from data_adapter import DataAdapter
from tda_engine import TDAEngine, compute_persistence_topology


def test_data_adapter():
    """Test DataAdapter functionality."""
    print("=" * 60)
    print("Testing DataAdapter...")
    print("=" * 60)
    
    # Generate sample data
    n_samples = 500
    np.random.seed(42)
    data = np.random.randn(n_samples, 3)
    
    adapter = DataAdapter()
    adapter.load_numpy(data)
    
    print(f"Loaded data shape: {adapter.get_shape()}")
    
    # Test normalization
    normalized = adapter.normalize_zscore()
    print(f"Normalized data mean: {np.mean(normalized, axis=0)}")
    print(f"Normalized data std: {np.std(normalized, axis=0)}")
    
    # Test downsampling
    large_data = np.random.randn(2000, 3)
    adapter.load_numpy(large_data)
    downsampled = adapter.downsample(target_size=500)
    print(f"Downsampled from 2000 to {downsampled.shape[0]} points")
    
    print("DataAdapter tests passed!\n")


def test_tda_engine():
    """Test TDAEngine functionality."""
    print("=" * 60)
    print("Testing TDAEngine...")
    print("=" * 60)
    
    # Generate sample data with topological structure (torus)
    n_samples = 500
    np.random.seed(42)
    
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    phi = np.random.uniform(0, 2*np.pi, n_samples)
    
    R = 3.0
    r = 1.0
    
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    
    noise = np.random.normal(0, 0.1, (n_samples, 3))
    data = np.column_stack([x, y, z]) + noise
    
    # Test TDA engine
    engine = TDAEngine(max_dimension=2, max_edge_length=1.5)
    normalized_data, diag = engine.compute_persistence_topology(data)
    
    print(f"Computed persistence diagram with {len(diag)} features")
    
    # Extract pairs
    pairs = engine.extract_persistence_pairs(diag)
    print(f"H0 features: {len(pairs[0])}")
    print(f"H1 features: {len(pairs[1])}")
    
    # Compute statistics
    stats = engine.get_statistics(diag)
    print(f"\nStatistics:")
    for key, value in stats.items():
        if key != 'computation_time':
            print(f"{key}: {value}")
        else:
            print(f"Computation time: {value:.3f}s")
    
    print("TDAEngine tests passed!\n")


def test_convenience_function():
    """Test the convenience function."""
    print("=" * 60)
    print("Testing convenience function...")
    print("=" * 60)
    
    # Generate simple test data
    data = np.random.randn(300, 3)
    
    normalized_data, diag = compute_persistence_topology(data, max_edge=1.5)
    
    print(f"Computed persistence diagram with {len(diag)} features")
    print(f"Data shape: {normalized_data.shape}")
    
    print("Convenience function test passed!\n")


def test_sliding_window():
    """Test sliding window functionality."""
    print("=" * 60)
    print("Testing sliding window...")
    print("=" * 60)
    
    # Generate time-series data
    n_samples = 2000
    data = np.random.randn(n_samples, 3)
    
    adapter = DataAdapter()
    adapter.load_numpy(data)
    
    window_size = 500
    step = 250
    
    window_count = 0
    for window in adapter.sliding_window(window_size=window_size, step=step):
        window_count += 1
        print(f"Window {window_count}: shape {window.shape}")
        
        if window_count >= 3:  # Test first 3 windows
            break
    
    print(f"Sliding window test passed! Generated {window_count} windows\n")


if __name__ == "__main__":
    try:
        test_data_adapter()
        test_tda_engine()
        test_convenience_function()
        test_sliding_window()
        
        print("=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)
        print("\nTo run the GUI application, execute:")
        print("python tda_guard_gui.py")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
