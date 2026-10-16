# TDA-Guard Extension Plugins

## Overview

This document describes the extension plugins for TDA-Guard that provide cross-domain symbol verification and physical phase transition analysis. These plugins follow the **Open-Closed Principle** - they extend functionality without modifying the core codebase.

## Architecture

```
TDA-Guard Core (tda_engine.py, data_adapter.py)
    ↓
Extension Plugins (extension_plugins.py)
    ↓
    ├── Z3FormalVerificationPlugin (Symbolic Logic Verification)
    └── PhysicsPhaseTransitionPlugin (Statistical Physics Entropy)
```

## Plugin 1: Z3 Formal Verification

### Purpose

Uses Z3 SMT solver to perform formal verification of topological invariants with mathematical and logical proofs.

### Features

- Formal constraint verification for H0 and H1 feature counts
- Threshold-based safety violation detection
- Mathematical proof generation

### Usage

```python
from extension_plugins import Z3FormalVerificationPlugin

# Verify topological invariants
result = Z3FormalVerificationPlugin.verify_topological_invariants(
    h0_count=469,
    h1_count=268,
    threshold_h1=50
)

print(result)
# Output: [h1_count = 268, is_violated = True]
```

### Parameters

- `h0_count` (int): Number of H0 (connected components) features
- `h1_count` (int): Number of H1 (loops) features
- `threshold_h1` (int, default=50): Safety threshold for H1 features

### Interpretation

- `is_violated = True`: H1 count exceeds safety threshold (potential anomaly)
- `is_violated = False`: H1 count within safe range

## Plugin 2: Physics Phase Transition

### Purpose

Applies statistical physics phase transition theory to calculate Shannon entropy of persistence diagrams, capturing pre-collapse indicators.

### Features

- Shannon entropy calculation from persistence lifespans
- Physical phase transition detection
- System collapse precursor identification

### Usage

```python
from extension_plugins import PhysicsPhaseTransitionPlugin

# Calculate manifold entropy from persistence diagram
diag = [
    (0, (0.0, 0.5)),
    (0, (0.1, 0.6)),
    (1, (0.2, 0.8)),
    (1, (0.3, 0.9))
]

entropy = PhysicsPhaseTransitionPlugin.calculate_manifold_entropy(diag)
print(f"Manifold Entropy: {entropy:.4f} bits")
```

### Parameters

- `diag_persistence` (list): Persistence diagram from TDA engine

### Interpretation

- **Low entropy (< 1.0 bits)**: Stable system, homogeneous lifespans
- **Medium entropy (1.0-2.0 bits)**: Normal fluctuations
- **High entropy (> 2.0 bits)**: Potential phase transition, diverse lifespans
- **Very high entropy (> 5.0 bits)**: Critical state, collapse imminent

## Integration with TDA-Guard Core

### Complete Workflow

```python
from data_adapter import DataAdapter
from tda_engine import TDAEngine
from extension_plugins import Z3FormalVerificationPlugin, PhysicsPhaseTransitionPlugin

# 1. Load and normalize data
adapter = DataAdapter(max_points=1000)
adapter.load_csv('data.csv')
normalized_data = adapter.normalize_zscore()

# 2. Compute topology
engine = TDAEngine()
data, diag = engine.compute_persistence_topology(normalized_data, max_edge=1.5)
stats = engine.get_statistics(diag)

# 3. Apply Z3 formal verification
z3_result = Z3FormalVerificationPlugin.verify_topological_invariants(
    h0_count=stats['H0']['count'],
    h1_count=stats['H1']['count'],
    threshold_h1=50
)
print(f"Z3 Verification: {z3_result}")

# 4. Calculate manifold entropy for phase transition detection
entropy = PhysicsPhaseTransitionPlugin.calculate_manifold_entropy(diag)
print(f"Manifold Entropy: {entropy:.4f} bits")

# 5. Interpret results
if "is_violated = True" in z3_result:
    print("WARNING: H1 count exceeds safety threshold!")
if entropy > 5.0:
    print("CRITICAL: High entropy indicates potential phase transition!")
```

### Integration with Industry Adapters

```python
from adapters import UnifiedAdapter
from extension_plugins import Z3FormalVerificationPlugin, PhysicsPhaseTransitionPlugin

# Use system monitor adapter
adapter = UnifiedAdapter('system_monitor', max_points=1000)
data = adapter.collect_data(duration_seconds=60, interval=0.5)
result = adapter.analyze_topology(data=data, use_defensive=True)

# Apply extensions
z3_result = Z3FormalVerificationPlugin.verify_topological_invariants(
    h0_count=result['stats']['H0']['count'],
    h1_count=result['stats']['H1']['count'],
    threshold_h1=50
)

entropy = PhysicsPhaseTransitionPlugin.calculate_manifold_entropy(result['diag'])
print(f"System Resource Anomaly Detection:")
print(f"  Z3 Verification: {z3_result}")
print(f"  Manifold Entropy: {entropy:.4f} bits")
```

## Installation

### Requirements

Add `z3-solver` to your dependencies:

```bash
pip install z3-solver>=4.12.0
```

Or update requirements.txt:

```
z3-solver>=4.12.0
```

## Testing

Run the extension plugin test suite:

```bash
python -m tests.unit.plugins.test_extension_plugins
```

### Test Coverage

- Z3 formal verification with normal and high H1 counts
- Physics phase transition with various persistence diagrams
- Integration with TDA-Guard core engine
- Empty and infinite value handling

### Expected Output

```
============================================================
TDA-Guard Extension Plugins Test Suite
============================================================

============================================================
Testing Z3 Formal Verification Plugin
============================================================
[PASS] Test 1: H0=469, H1=268, Threshold=50
  Result: [h1_count = 268, is_violated = True]
[PASS] Test 2: H0=469, H1=100, Threshold=50
  Result: [h1_count = 100, is_violated = True]

[PASS] Z3 Formal Verification Plugin tests completed

============================================================
Testing Physics Phase Transition Plugin
============================================================
[PASS] Test 1: Normal persistence diagram
  Entropy: 1.9940 bits
[PASS] Test 2: Empty persistence diagram
  Entropy: 0.0000 bits
[PASS] Test 3: High entropy persistence diagram
  Entropy: 1.9302 bits
[PASS] Test 4: Persistence diagram with infinite death values
  Entropy: 0.9940 bits

[PASS] Physics Phase Transition Plugin tests completed

============================================================
Testing Integration with TDA-Guard Core
============================================================
[PASS] TDA Engine computed successfully
  H0 count: 99
  H1 count: 24
[PASS] Z3 Verification: [h1_count = 24, is_violated = False]
[PASS] Manifold Entropy: 6.6627 bits

[PASS] Integration test completed successfully

============================================================
All extension plugin tests completed!
============================================================
```

## Design Principles

### Open-Closed Principle

- **Closed for modification**: Core TDA engine remains unchanged
- **Open for extension**: New functionality added via plugins

### Zero Core Impact

- No modifications to `tda_engine.py`
- No modifications to `data_adapter.py`
- No modifications to `tda_guard_gui.py`
- Pure extension through import and call

### Modular Architecture

- Each plugin is self-contained
- Plugins can be used independently
- Easy to add new plugins following the same pattern

## Use Cases

### System Security

```python
# Detect resource collapse with formal verification
z3_result = Z3FormalVerificationPlugin.verify_topological_invariants(
    h0_count=resource_stats['H0']['count'],
    h1_count=resource_stats['H1']['count'],
    threshold_h1=30  # Stricter threshold for system resources
)
```

### Web Security

```python
# Detect SQL injection attacks with entropy analysis
entropy = PhysicsPhaseTransitionPlugin.calculate_manifold_entropy(web_diag)
if entropy > 3.0:
    alert("Potential attack pattern detected")
```

### IoT Security

```python
# Detect botnet infection with combined analysis
z3_result = Z3FormalVerificationPlugin.verify_topological_invariants(...)
entropy = PhysicsPhaseTransitionPlugin.calculate_manifold_entropy(...)
if "is_violated = True" in z3_result and entropy > 4.0:
    alert("Critical: Botnet infection detected")
```

### AI Safety

```python
# Detect model collapse with phase transition
entropy = PhysicsPhaseTransitionPlugin.calculate_manifold_entropy(ai_diag)
if entropy > 5.0:
    alert("WARNING: Model approaching collapse phase")
```

## Advanced Configuration

### Custom Thresholds

```python
# Adjust thresholds based on domain requirements
z3_result = Z3FormalVerificationPlugin.verify_topological_invariants(
    h0_count=stats['H0']['count'],
    h1_count=stats['H1']['count'],
    threshold_h1=100  # Higher threshold for AI models
)
```

### Multi-Stage Analysis

```python
# Combine both plugins for comprehensive analysis
z3_result = Z3FormalVerificationPlugin.verify_topological_invariants(...)
entropy = PhysicsPhaseTransitionPlugin.calculate_manifold_entropy(...)

# Decision logic
if "is_violated = True" in z3_result:
    if entropy > 5.0:
        severity = "CRITICAL"
    elif entropy > 3.0:
        severity = "HIGH"
    else:
        severity = "MEDIUM"
else:
    if entropy > 4.0:
        severity = "MEDIUM"
    else:
        severity = "LOW"
```

## Troubleshooting

### Z3 Solver Not Found

```bash
pip install z3-solver
```

### Import Error

Ensure `extension_plugins.py` is in the same directory as your script.

### Empty Persistence Diagram

The plugin handles empty diagrams gracefully, returning 0.0 entropy.

### Infinite Death Values

The plugin automatically filters out infinite death values (representing features that never die).

## Future Extensions

Potential future plugins following the same pattern:

- **Statistical Learning Plugin**: Machine learning-based anomaly classification
- **Time Series Plugin**: Temporal pattern analysis for streaming data
- **Graph Theory Plugin**: Network topology analysis
- **Information Theory Plugin**: Mutual information and complexity measures

## Conclusion

The extension plugins provide a powerful, non-invasive way to add cross-domain analysis capabilities to TDA-Guard. By following the Open-Closed Principle, they ensure the core system remains stable while enabling flexible expansion for specific use cases.
