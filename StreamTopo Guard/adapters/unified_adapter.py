"""
Unified Adapter Interface for TDA-Guard

Provides a high-level interface to seamlessly integrate all industry adapters
with the TDA-Guard core engine.

Integrated with defensive engineering:
- Hard Cap Guard for memory protection
- Adaptive Baseline for false positive prevention
- Bulletproof error handling for graceful degradation
- Vectorized operations for GIL bypass

Usage:
    from adapters import UnifiedAdapter
    from tda_engine import TDAEngine
    
    adapter = UnifiedAdapter('system_monitor')
    data = adapter.collect_data()
    
    engine = TDAEngine()
    normalized_data, diag = engine.compute_persistence_topology(data)
"""

import numpy as np
from typing import Optional, Dict, Any
from data_adapter import DataAdapter
from tda_engine import TDAEngine
import sys
import os

# Add parent directory to path for defensive module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from defensive import DefensiveWrapper, AdaptiveBaseline

# Import all module adapters
from .module_a_system_monitor import SystemResourceMonitor
from .module_b_web_traffic import WebTrafficGuard
from .module_c_iot_stream import IoTStreamGuard
from .module_d_ai_landscape import AILossLandscapeGuard


class UnifiedAdapter:
    """
    Unified interface for all TDA-Guard industry adapters.
    
    Supported modes:
    - 'system_monitor': Module A - System Resource Monitor
    - 'web_traffic': Module B - Web Traffic Guard
    - 'iot_stream': Module C - IoT Stream Guard
    - 'ai_landscape': Module D - AI Loss Landscape Guard
    """
    
    def __init__(self, mode: str, **kwargs):
        """
        Initialize unified adapter with defensive engineering.
        
        Args:
            mode: Adapter mode ('system_monitor', 'web_traffic', 'iot_stream', 'ai_landscape')
            **kwargs: Additional arguments for specific adapters
        """
        self.mode = mode
        self.kwargs = kwargs
        self.adapter = self._create_adapter()
        self.data_adapter = DataAdapter(max_points=kwargs.get('max_points', 1000))
        self.tda_engine = TDAEngine()
        
        # Initialize defensive wrapper
        self.defensive = DefensiveWrapper(
            max_points=kwargs.get('max_points', 1000),
            baseline_window=kwargs.get('baseline_window', 50),
            fallback_shape=kwargs.get('fallback_shape', (100, 3))
        )
        
        # Initialize adaptive baseline for anomaly scores
        self.baseline = AdaptiveBaseline(window_size=kwargs.get('baseline_window', 50))
    
    def _create_adapter(self):
        """Create the appropriate adapter based on mode."""
        # Filter out defensive-only kwargs before passing to adapters
        adapter_kwargs = self.kwargs.copy()
        defensive_keys = {'max_points', 'baseline_window', 'fallback_shape'}
        for key in defensive_keys:
            adapter_kwargs.pop(key, None)
        
        if self.mode == 'system_monitor':
            return SystemResourceMonitor(**adapter_kwargs)
        elif self.mode == 'web_traffic':
            return WebTrafficGuard(**adapter_kwargs)
        elif self.mode == 'iot_stream':
            return IoTStreamGuard(**adapter_kwargs)
        elif self.mode == 'ai_landscape':
            return AILossLandscapeGuard(**adapter_kwargs)
        else:
            raise ValueError(f"Unknown adapter mode: {self.mode}")
    
    def collect_data(self, **collect_kwargs) -> np.ndarray:
        """
        Collect data using the selected adapter.
        
        Args:
            **collect_kwargs: Arguments for data collection (adapter-specific)
        
        Returns:
            NumPy array of collected data
        """
        if self.mode == 'system_monitor':
            return self.adapter.collect_continuous_stream(**collect_kwargs)
        elif self.mode == 'web_traffic':
            log_file = collect_kwargs.get('log_file')
            if log_file:
                return self.adapter.parse_access_log_to_manifold(log_file)
            else:
                # Generate sample log for testing
                sample_log = self.adapter.generate_sample_log(**collect_kwargs)
                return self.adapter.parse_access_log_to_manifold(sample_log)
        elif self.mode == 'iot_stream':
            n_samples = collect_kwargs.get('n_samples', 500)
            mode = collect_kwargs.get('mode', 'normal')
            traffic = self.adapter.generate_sample_iot_traffic(n_samples, mode)
            return self.adapter.adapt_iot_traffic(traffic)
        elif self.mode == 'ai_landscape':
            model = collect_kwargs.get('model')
            if model:
                return self.adapter.extract_model_weights_manifold(model)
            else:
                # Generate synthetic model for testing
                model = self.adapter.generate_synthetic_model(**collect_kwargs)
                return self.adapter.extract_model_weights_manifold(model)
        else:
            raise ValueError(f"Data collection not implemented for mode: {self.mode}")
    
    def analyze_topology(self, data: Optional[np.ndarray] = None,
                        max_edge: float = 1.5, use_defensive: bool = True) -> Dict[str, Any]:
        """
        Perform complete TDA analysis on collected data with defensive protection.
        
        Args:
            data: Optional pre-collected data (if None, will collect)
            max_edge: Maximum edge length for Rips complex
            use_defensive: Whether to use defensive wrapper (default: True)
        
        Returns:
            Dictionary with analysis results:
            - 'data': Normalized data
            - 'diag': Persistence diagram
            - 'stats': Topological statistics
            - 'anomaly': Quick anomaly detection results
            - 'baseline_status': Adaptive baseline status (if defensive enabled)
        """
        # Collect data if not provided
        if data is None:
            data = self.collect_data()
        
        if use_defensive:
            # Use defensive wrapper for bulletproof execution
            normalized_data, _ = self.defensive.safe_execute(
                lambda x: self.data_adapter.normalize_zscore(x),
                data
            )
        else:
            # Normalize using DataAdapter (legacy mode)
            self.data_adapter.load_numpy(data)
            normalized_data = self.data_adapter.normalize_zscore()
        
        # Downsample if needed (already handled by hard cap in defensive mode)
        if not use_defensive and normalized_data.shape[0] > 500:
            normalized_data = self.data_adapter.downsample(target_size=500)
        
        # Compute persistence topology
        final_data, diag = self.tda_engine.compute_persistence_topology(
            normalized_data,
            max_edge=max_edge
        )
        
        # Get statistics
        stats = self.tda_engine.get_statistics(diag)
        
        # Quick anomaly detection (adapter-specific)
        anomaly = self._detect_anomaly(data)
        
        # Evaluate anomaly score with adaptive baseline (Defense 2)
        is_true_anomaly = False
        baseline_status = None
        
        if use_defensive and 'score' in anomaly:
            anomaly_score = anomaly['score']
            is_true_anomaly, threshold, z_score = self.baseline.evaluate(anomaly_score)
            baseline_status = {
                'is_warmed_up': self.baseline.is_warmed_up,
                'threshold': threshold,
                'z_score': z_score,
                'is_true_anomaly': is_true_anomaly
            }
            anomaly['is_true_anomaly'] = is_true_anomaly
            anomaly['baseline_threshold'] = threshold
        
        result = {
            'data': final_data,
            'diag': diag,
            'stats': stats,
            'anomaly': anomaly
        }
        
        if baseline_status is not None:
            result['baseline_status'] = baseline_status
        
        return result
    
    def _detect_anomaly(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform adapter-specific anomaly detection."""
        if self.mode == 'system_monitor':
            return self.adapter.detect_resource_anomaly(data)
        elif self.mode == 'web_traffic':
            return self.adapter.detect_scan_patterns(data)
        elif self.mode == 'iot_stream':
            return self.adapter.detect_flow_regime(data)
        elif self.mode == 'ai_landscape':
            # For AI landscape, need baseline
            baseline = self.kwargs.get('baseline_weights')
            if baseline:
                return self.adapter.detect_landscape_fracture(data, baseline)
            else:
                return {'fracture_score': 0.0, 'is_fractured': False}
        else:
            return {'score': 0.0}
    
    def continuous_monitor(self, data_stream: np.ndarray, 
                          step: int = 50, max_edge: float = 1.5) -> list:
        """
        Continuous monitoring with sliding window.
        
        Args:
            data_stream: Full data stream
            step: Step size for sliding window
            max_edge: Maximum edge length for Rips complex
        
        Returns:
            List of analysis results for each window
        """
        results = []
        n_samples = data_stream.shape[0]
        window_size = 500
        
        for start in range(0, n_samples - window_size + 1, step):
            end = start + window_size
            window = data_stream[start:end]
            
            # Analyze window
            result = self.analyze_topology(data=window, max_edge=max_edge)
            results.append(result)
        
        return results


def quick_analyze(mode: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function for quick one-line analysis.
    
    Args:
        mode: Adapter mode
        **kwargs: Arguments for data collection and analysis
    
    Returns:
        Analysis results dictionary
    """
    adapter = UnifiedAdapter(mode, **kwargs)
    return adapter.analyze_topology(**kwargs)
