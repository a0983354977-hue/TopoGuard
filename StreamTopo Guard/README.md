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
- **Technology**: Logged exceptions and explicit error propagation
- **Protection**: Invalid data never becomes a successful analysis result
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

### Standalone downloads

Run `setup_and_build.bat` on Windows to install Python 3.12 if needed, create an isolated `.venv-build` environment, install the project and build dependencies, run the core tests, and create the GUI executable. The script uses winget when available and falls back to the official Python installer.

The Windows executable is created at `downloads/Windows/TopoGuard.exe`. It includes its Python runtime and required GUI dependencies, so Python does not need to be installed on a computer that only runs the EXE.

Linux and macOS downloads must be built on their native operating systems. The **Build platform downloads** GitHub Actions workflow builds all three platforms on each push to `main`, or can be started manually from the Actions tab. Download the `TopoGuard-<platform>-<architecture>` artifact from a completed workflow run. Linux downloads are `.tar.gz` archives; macOS downloads are `.zip` archives containing the app bundle. Workflow artifacts are retained for 30 days.

To build locally on Linux or macOS, install `requirements.txt` and `requirements-build.txt`, run `python -m tests.unit.core.test_tda_guard`, then run `python build_download.py`. The output is placed in the matching `downloads/<platform>` directory.

PyTorch is optional for the separate AI landscape adapter and is not installed or bundled by this GUI build.

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
python -m tests.unit.core.test_tda_guard
```

The complete test suite is organized by project area in [`tests/`](tests/README.md); run the relevant suite with the `python -m tests...` commands listed there.

### Live and Recorded Data Sources

Run the CLI without `--source` to analyze live local system metrics. Select one source per run:

On Windows, double-click `run_topoguard.bat` to start a 100-tick live system
monitor without an interactive mode menu. On Linux/macOS, use
`bash ./run_topoguard.sh` for the same direct-start behavior.

```bash
python topoguard_cli.py monitor --source system --ticks 20
sudo python topoguard_cli.py monitor --source ebpf --ticks 20
python topoguard_cli.py audit --source web-log --input /var/log/nginx/access.log
python topoguard_cli.py audit --source csv --input metrics.csv --batch-size 500
python topoguard_cli.py audit --source json --input metrics.json
python topoguard_cli.py audit --source jsonl --input fluent-bit.jsonl --batch-size 500
python topoguard_cli.py audit --source zeek --input conn.log --batch-size 500
python topoguard_cli.py monitor --source prometheus --input http://localhost:9100/metrics --ticks 20
```

The eBPF source requires Linux, BCC, and the permissions needed to attach kernel probes. Web log mode analyzes newly appended access-log lines; CSV/JSON/JSONL modes replay recorded data in bounded batches. JSONL accepts one JSON record per line, including nested numeric objects such as Fluent Bit or Zeek JSON output. Nested numeric features retain dotted names (for example, `metrics.cpu_usage`); timestamps and non-numeric metadata are excluded. Rows with non-finite numeric values are logged and skipped; a batch with no finite numeric features is rejected. This is file replay, not a live Fluent Bit or Zeek connection. If a real source cannot start or its input format is invalid, the CLI reports the error rather than silently substituting synthetic data. Runtime collectors and launchers do not provide synthetic/mock telemetry.
Prometheus mode scrapes a live HTTP(S) `/metrics` endpoint on each tick, such as
the node_exporter endpoint at `http://localhost:9100/metrics`. It accepts
Prometheus text exposition samples, retains label values in stable feature
names, rejects malformed/non-finite samples, and bounds each response to 4 MiB
and 1,000 series. Scrapes use a five-second timeout and require an endpoint
reachable without URL-embedded credentials or custom authentication headers.
It uses only Python's standard library; it does not install or configure
Prometheus or node_exporter.
Zeek `conn.log` and `http.log` must be configured for JSON output (for example,
`@load policy/tuning/json-logs.zeek`). Select `--source zeek` explicitly because
Zeek `.log` files share an extension with ordinary access logs. Zeek replay
keeps only recognized numeric fields such as ports, duration, byte/packet
counts, HTTP status, and body lengths; addresses, UIDs, methods, and other
categorical metadata are excluded. This is offline JSON-log replay, not live
Zeek control or packet capture.

The production hybrid engine combines TDA with scikit-learn Isolation Forest;
untrained PyG/GNN inference is disabled by default.
It waits until it has at least 33 rows, fits on the first 32 (up to 256) as the
normal baseline, and scores the newest row. Start with a known-clean baseline;
warm-up samples are not independently labeled as safe. Feature count/order
changes after fitting fail explicitly and require restarting the engine with a
new baseline. Isolation Forest can also be disabled through the Python engine
API; the CLI and GUI use it by default.

Slack and Discord alerts are opt-in and send only the source, tick, score, and
severity (not raw telemetry or IP addresses). Configure the HTTPS incoming
webhook URL in the `TOPOGUARD_WEBHOOK_URL` environment variable, then add
`--notify` to `topoguard-cli monitor` or `audit`; the Streamlit GUI has a
separate notification checkbox. Delivery errors are reported instead of being silently ignored.
The webhook URL contains a secret token: provide it through a secret manager or
protected environment configuration and never commit it or pass it as a CLI
argument. Only Slack and Discord incoming webhook endpoints are accepted.
The current engine reports and notifies; it does not invoke Fail2ban or change
firewall state. Anomaly scores are not yet mapped to reliably attributable
client IPs, so automatic blocking is intentionally unavailable.

### Production CLI

Install the command from the project directory with `pip install -e .`:

```bash
topoguard-cli monitor --source system --ticks 20
sudo topoguard-cli monitor --source ebpf
topoguard-cli monitor --source prometheus --input http://localhost:9100/metrics --ticks 20 --interval 5
topoguard-cli monitor --source system --tda-max-dim 2
topoguard-cli export-tda --source csv --input metrics.csv --tda-max-dim 2 --output topology.json
topoguard-cli monitor --source system --baseline state/baseline.json
topoguard-cli audit --input /var/log/nginx/access.log
topoguard-cli audit --input capture.pcap --state-json security-state.json
topoguard-cli audit --input fluent-bit.jsonl
topoguard-cli audit --source zeek --input conn.log
topoguard-cli export-tda --source csv --input metrics.csv --output topology.json
topoguard-cli export-tda --source prometheus --input http://localhost:9100/metrics --interval 1 --max-points 20 --output topology.json
topoguard-cli monitor --source system --ticks 20 --notify
```

`monitor` runs continuously unless `--ticks` is supplied. If eBPF is unavailable
or fails while attaching, it reports the reason and continues with real `psutil`
host metrics; it never substitutes generated events. Offline `audit` supports
access logs, numeric CSV/JSON/JSONL replay, and IPv4 PCAP/PCAPNG metadata. JSONL
files may contain one structured record per line; nested numeric fields are
flattened to stable dotted feature names. The optional
`--state-json` must contain `cpu`, `sockets`, and `su_attempts`; Z3 evaluates
that boundary only when the TDA score is above `0.85`. `export-tda` writes
persistence intervals and sampled Betti curves as JSON, or the curves as CSV.
TDA defaults to maximum homology dimension 1 (H0/H1); use `--tda-max-dim 2`
to include H2. H2 requires building one higher simplex dimension and can grow
substantially in cost, so enable it only when needed. The Rips point cloud is
made C-contiguous and deterministically downsampled to at most 300 points before
topology computation. The export point limit defaults to 300 and is capped at 500.
H2 results are exposed for inspection/export but are not yet part of the anomaly
score; scoring behavior remains based on H1 and Isolation Forest.
Prometheus scraping is available to live `monitor` and sampled `export-tda`;
offline `audit` remains limited to finite logs, replays, and packet captures.

Live monitoring collects and analyzes on separate worker threads through a
bounded eight-item queue. All sources apply backpressure when analysis falls
behind; queued samples are not dropped, at the cost of a slower effective
sampling rate under sustained load. Polars ingestion holds a bounded Float64
window, enforces a stable feature schema, and rejects a row that alone would
exceed its configured memory budget.

Live `monitor` skips malformed numeric samples and retryable source errors while
logging each rejection. Five consecutive invalid/failed samples open a circuit
breaker, emit a CRITICAL event to configured webhook/SIEM outputs, and end the
run with a failure exit status. A successfully inspected sample resets the
consecutive-failure count; non-retryable source and internal computation errors
remain fail-closed.

Pass `--baseline <path>` to `monitor` to load a snapshot at startup when that
file exists, save once the Isolation Forest baseline is trained, checkpoint it
hourly, and save again on normal exit. If the path does not yet exist, the
detector warms up and the first trained model is saved there. For example,
`--baseline state/baseline.json` uses JSON metadata; other extensions use
pickle. Without a pre-trained snapshot, the existing Isolation Forest behavior
still fits its initial warm-up window from live samples; it does not continually
retrain from later observations. For environments where those initial samples
must be explicitly approved as normal, create the snapshot with
`train-baseline` from trusted known-normal history and supply it to `monitor`.
Snapshots contain a serialized scikit-learn model; load only files from trusted
sources because deserialization can execute code. `audit --baseline` remains
read-only and loads a previously trained model without writing snapshots.

#### Linux eBPF container

Build from the project directory with `docker build -f Dockerfile.ebpf -t topoguard-ebpf .`.
The image includes Ubuntu's BCC Python bindings; BCC still compiles probes
against the **host** kernel, so run it with the host module/header paths and
required privileges:

```bash
docker run --rm --privileged --pid=host \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  topoguard-ebpf monitor --source ebpf
```

Use only on Linux hosts with BCC-compatible kernel headers installed. The
container cannot make an incompatible or unsupported host kernel eBPF-capable.

### GUI Application

Launch the desktop TDA visualization with `python tda_guard_gui.py`. Launch
the real-data monitoring dashboard with `streamlit run run_simulation_gui.py`.

```bash
python tda_guard_gui.py
```

#### GUI Controls

1. **Load Data**:
   - `Load CSV`: Import CSV files with numeric columns
   - `Load JSON`: Import JSON files with numeric fields
   - Load CSV/JSON data files for analysis

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
wrapper = DefensiveWrapper(max_points=1000, baseline_window=50)

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
metrics = adapter.collect_data(duration_seconds=60, interval=0.5)
result = adapter.analyze_topology(data=metrics)
print(f"Anomaly score: {result['anomaly']['score']}")

# Web Traffic Guard (Module B)
adapter = UnifiedAdapter('web_traffic')
log_features = adapter.collect_data(log_file='access.log')
result = adapter.analyze_topology(data=log_features)
print(f"Scan detected: {result['anomaly']['high_frequency_ip']}")

# IoT Stream Guard (Module C)
adapter = UnifiedAdapter('iot_stream', window_size=500)
packet_flows = packet_collector.read_feature_batch()
features = adapter.collect_data(packet_flows=packet_flows)
result = adapter.analyze_topology(data=features)
print(f"Turbulence score: {result['anomaly']['turbulence_score']}")

# AI Loss Landscape Guard (Module D)
adapter = UnifiedAdapter(
    'ai_landscape',
    tensor_dim=16,
    baseline_weights=baseline_weights,
)
weights = adapter.collect_data(model=trained_model)
result = adapter.analyze_topology(data=weights)
print(f"Fracture score: {result['anomaly']['fracture_score']}")
```

#### Quick One-Line Analysis

```python
from adapters import quick_analyze

# Analyze real features already collected from a source
metrics = UnifiedAdapter('system_monitor').collect_data(duration_seconds=30)
result = quick_analyze('system_monitor', data=metrics)

# Analyze real access-log features
log_features = UnifiedAdapter('web_traffic').collect_data(log_file='access.log')
result = quick_analyze('web_traffic', data=log_features)
```

### Testing

Run core tests:
```bash
python -m tests.unit.core.test_tda_guard
```

Run adapter tests:
```bash
python -m tests.unit.adapters.test_adapters
```

Run defensive engineering tests:
```bash
python -m tests.unit.defensive.test_defensive
```

## Architecture

### Core Components

1. **data_adapter.py**: Data loading, normalization, downsampling (with defensive integration)
2. **tda_engine.py**: Persistent homology computation, statistics
3. **tda_guard_gui.py**: Tkinter GUI with async threading
4. **defensive.py**: Four defensive mechanisms (Hard Cap, Adaptive Baseline, Bulletproof, Vectorization)
5. **tests/unit/core/test_tda_guard.py**: Core TDA tests
6. **tests/unit/defensive/test_defensive.py**: Defensive engineering tests

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
3. **BulletproofAdapter**: Error tracking with fail-closed propagation
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

Topology tests use controlled local fixtures; runtime collection requires real or explicitly supplied recorded data.

## Defensive Engineering Documentation

For detailed defensive engineering specifications, see [DEFENSIVE_ENGINEERING.md](DEFENSIVE_ENGINEERING.md).

## License

Commercial-grade tool for security research and anomaly detection.
