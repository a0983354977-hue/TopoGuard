# TDA-Guard: Topological Anomaly Detection Tool

A commercial-grade TDA (Topological Data Analysis) tool for detecting anomalies in high-dimensional data using persistent homology. Built with Python, Gudhi, and Tkinter/Matplotlib for real-time visualization.

## Features

### Phase 1: Core Math Engine & Data Adapter
- **Data Adapter**: Converts heterogeneous data (CSV/JSON/NumPy) to standardized float64 matrices
- **Z-score Normalization**: Prevents topological distortion from different numerical scales
- **Gudhi Integration**: Computes persistent homology using RipsComplex and simplex trees
- **Downsampling**: Automatic downsampling to 300-500 points to prevent O(N³) combinatorial explosion
- **Sliding Window**: Continuous time-series scanning for large datasets

### Phase 2: Asynchronous GUI & Rendering
- **Non-blocking Computation**: Background threading prevents UI freezing
- **Dual Visualization**: Interactive 3D point cloud + persistence diagram
- **Safe UI Updates**: Callback pattern via `root.after()` for thread-safe rendering
- **Memory Leak Prevention**: Automatic Matplotlib buffer clearing
- **Real-time Statistics**: Feature counts, persistence intervals, computation time

### Industry Application Adapters

Four modular adapters for real-world security and monitoring scenarios:

#### 🟢 Module A: System Resource Monitor
- **Purpose**: Detect resource collapse and crypto-mining malware
- **Technology**: psutil + pandas
- **Detection**: Identifies geometric collapse in CPU/RAM/Disk metrics before threshold alerts
- **Use Case**: Early warning for system overload and hidden mining processes

#### 🔵 Module B: Web Traffic Guard
- **Purpose**: Detect malicious scanning (SQL injection, directory traversal)
- **Technology**: pandas + regex
- **Detection**: Analyzes HTTP request frequency, status codes, IP entropy, path complexity
- **Use Case**: Pre-attack detection in web server access logs

#### 🟣 Module C: IoT Stream Guard
- **Purpose**: Detect botnet infection in IoT/P2P networks
- **Technology**: scikit-learn StandardScaler
- **Detection**: Identifies transition from laminar flow to turbulent flow
- **Use Case**: Real-time monitoring of IoT device traffic patterns

#### 🟠 Module D: AI Loss Landscape Guard
- **Purpose**: Detect model collapse and hallucination during training
- **Technology**: PyTorch + Gudhi
- **Detection**: Analyzes weight manifold topology for geometric fractures
- **Use Case**: Training stability monitoring for LLMs and neural networks

### Defensive Engineering

Four production-grade defensive mechanisms for enterprise stability:

#### 🛡️ Defense 1: Hard Cap Guard (Memory Protection)
- **Purpose**: Prevent memory exhaustion from massive data bursts
- **Technology**: Smart downsampling with hard thresholds
- **Protection**: Automatic data capping at MAX_POINTS (default: 1000)
- **Use Case**: Handling 100K+ log entries or IoT packet floods

#### 📊 Defense 2: Adaptive Baseline (False Positive Prevention)
- **Purpose**: Eliminate false positive storms from normal fluctuations
- **Technology**: Dynamic tolerance bands with warm-up learning
- **Protection**: Only alerts when deviation exceeds Mean + 3×Std
- **Use Case**: Filtering day/night cycles and business peak noise

#### 🔧 Defense 3: Bulletproof Error Handling (Graceful Degradation)
- **Purpose**: Prevent daemon crashes from malformed input
- **Technology**: Try-except wrappers with null-space fallbacks
- **Protection**: Background processes never crash, always respond
- **Use Case**: Handling log format mutations and corrupted data

#### ⚡ Defense 4: Vectorization Utilities (GIL Bypass)
- **Purpose**: Maximize multi-core performance on enterprise servers
- **Technology**: Pure NumPy C++ operations, no Python loops
- **Protection**: Gudhi C++ core releases GIL automatically
- **Use Case**: High-throughput deployment on multi-core systems

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- numpy >= 1.24.0
- gudhi >= 3.8.0
- matplotlib >= 3.7.0
- scipy >= 1.10.0
- pandas >= 2.0.0
- psutil >= 5.9.0 (Module A)
- scikit-learn >= 1.3.0 (Module C)
- torch >= 2.0.0 (Module D, optional)

## Usage

### Command Line Testing

Run the test suite to verify core functionality:

```bash
python test_tda_guard.py
```

### GUI Application

Launch the interactive GUI:

```bash
python tda_guard_gui.py
```

#### GUI Controls

1. **Load Data**:
   - `Load CSV`: Import CSV files with numeric columns
   - `Load JSON`: Import JSON files with numeric fields
   - `Generate Sample Data`: Create synthetic torus data for testing

2. **Parameters**:
   - `Max Edge Length`: Maximum edge length for Rips complex (default: 1.5)
   - `Downsample Size`: Target point count for large datasets (default: 500)

3. **Computation**:
   - `Compute Topology`: Start async TDA computation
   - Progress bar shows computation status
   - Status displays completion time

4. **Visualization**:
   - Left panel: Interactive 3D point cloud (rotate/zoom)
   - Right panel: Persistence diagram (H0=red, H1=blue)
   - Statistics panel shows feature counts and intervals

### Defensive Engineering Usage

```python
from defensive import DefensiveWrapper, AdaptiveBaseline

# Initialize defensive wrapper
wrapper = DefensiveWrapper(max_points=1000, baseline_window=50, fallback_shape=(100, 3))

# Safe execution with all defenses
result, is_anomaly = wrapper.safe_execute(parse_func, raw_input, current_score=0.5)

# Adaptive baseline for false positive prevention
baseline = AdaptiveBaseline(window_size=50, std_multiplier=3.0)
is_anomaly, threshold, z_score = baseline.evaluate(current_score)
```

**Enable defensive mode in adapters**:
```python
from adapters import UnifiedAdapter

adapter = UnifiedAdapter('system_monitor', max_points=1000, baseline_window=50)
result = adapter.analyze_topology(data=data, use_defensive=True)

# Result includes baseline_status when defensive enabled
print(f"Baseline threshold: {result['baseline_status']['threshold']}")
print(f"Is true anomaly: {result['baseline_status']['is_true_anomaly']}")
```

### Programmatic Usage

```python
from data_adapter import DataAdapter
from tda_engine import TDAEngine

# Load and normalize data
adapter = DataAdapter()
adapter.load_csv('data.csv')
normalized_data = adapter.normalize_zscore()

# Compute persistent homology
engine = TDAEngine()
data, diag = engine.compute_persistence_topology(normalized_data)

# Extract statistics
stats = engine.get_statistics(diag)
print(f"H0 features: {stats['H0']['count']}")
print(f"H1 features: {stats['H1']['count']}")
```

### Industry Adapter Usage

#### Using Unified Adapter Interface

```python
from adapters import UnifiedAdapter

# System Resource Monitor (Module A)
adapter = UnifiedAdapter('system_monitor', history_window=300)
result = adapter.analyze_topology(duration_seconds=60, interval=0.5)
print(f"Anomaly score: {result['anomaly']['score']}")

# Web Traffic Guard (Module B)
adapter = UnifiedAdapter('web_traffic')
result = adapter.analyze_topology(n_entries=1000, include_attack=True)
print(f"Scan detected: {result['anomaly']['high_frequency_ip']}")

# IoT Stream Guard (Module C)
adapter = UnifiedAdapter('iot_stream', window_size=500)
result = adapter.analyze_topology(n_samples=500, mode='botnet')
print(f"Turbulence score: {result['anomaly']['turbulence_score']}")

# AI Loss Landscape Guard (Module D)
adapter = UnifiedAdapter('ai_landscape', tensor_dim=16)
result = adapter.analyze_topology(n_layers=3, layer_size=64)
print(f"Fracture score: {result['anomaly']['fracture_score']}")
```

#### Quick One-Line Analysis

```python
from adapters import quick_analyze

# Analyze system resources in one line
result = quick_analyze('system_monitor', duration_seconds=30)

# Analyze web traffic
result = quick_analyze('web_traffic', log_file='access.log')

# Analyze IoT traffic
result = quick_analyze('iot_stream', n_samples=500, mode='normal')
```

### Testing

Run core tests:
```bash
python test_tda_guard.py
```

Run adapter tests:
```bash
python test_adapters.py
```

Run defensive engineering tests:
```bash
python test_defensive.py
```

## Architecture

### Core Components

1. **data_adapter.py**: Data loading, normalization, downsampling (with defensive integration)
2. **tda_engine.py**: Persistent homology computation, statistics
3. **tda_guard_gui.py**: Tkinter GUI with async threading
4. **defensive.py**: Four defensive mechanisms (Hard Cap, Adaptive Baseline, Bulletproof, Vectorization)
5. **test_tda_guard.py**: Unit tests for validation
6. **test_defensive.py**: Defensive engineering tests

### Industry Adapters

Located in `adapters/` directory:

1. **module_a_system_monitor.py**: System resource telemetry (psutil + pandas)
2. **module_b_web_traffic.py**: Web access log parsing (pandas + regex)
3. **module_c_iot_stream.py**: IoT traffic standardization (scikit-learn)
4. **module_d_ai_landscape.py**: Neural network weight extraction (PyTorch)
5. **unified_adapter.py**: High-level interface for all adapters (with defensive integration)

### Defensive Engineering

Located in `defensive.py`:

1. **HardCapGuard**: Memory protection with smart downsampling
2. **AdaptiveBaseline**: False positive prevention with warm-up learning
3. **BulletproofAdapter**: Graceful degradation with fallback mechanisms
4. **VectorizedOps**: GIL bypass through NumPy C++ operations
5. **DefensiveWrapper**: Unified interface integrating all four defenses

### Data Flow

```
Raw Data (System/Logs/IoT/Model)
    ↓
Industry Adapter (Standardization)
    ↓
DefensiveWrapper (Bulletproof + Hard Cap + Vectorization)
    ↓
DataAdapter (Z-score normalization, downsampling)
    ↓
TDAEngine (RipsComplex → SimplexTree → Persistence)
    ↓
AdaptiveBaseline (False positive filtering)
    ↓
GUI Thread (3D visualization + persistence diagram)
```

## Performance Considerations

- **Memory**: Hard Cap Guard limits point cloud to 1000 points (configurable)
- **Computation**: RipsComplex has O(N³) complexity; use sliding window for large datasets
- **Threading**: Background computation prevents UI blocking
- **Visualization**: Matplotlib buffers cleared on each redraw to prevent leaks
- **Vectorization**: NumPy C++ operations bypass Python GIL for multi-core performance
- **False Positives**: Adaptive Baseline filters noise with 3σ threshold

## Testing

The test suite validates:
- Data loading and normalization
- Downsampling and sliding window
- TDA engine computation
- Statistics extraction
- Convenience functions

All tests pass with synthetic torus data (500 points, H0=499, H1=146 features).

## Defensive Engineering Documentation

For detailed defensive engineering specifications, see [DEFENSIVE_ENGINEERING.md](DEFENSIVE_ENGINEERING.md).

## License

Commercial-grade tool for security research and anomaly detection.
