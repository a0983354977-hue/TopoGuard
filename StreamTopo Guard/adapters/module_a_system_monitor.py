"""
Module A: System Resource Monitor Adapter
Detects resource collapse and crypto-mining malware via topological analysis.

Architecture:
Raw System Metrics (psutil) → Pandas Time-Series → TDA-Guard Core → Alert
"""

import psutil
import pandas as pd
import numpy as np
from typing import List, Tuple
from collections import deque


class SystemResourceMonitor:
    """
    Adapter for converting system telemetry into topological features.
    Detects anomalies before traditional thresholds (e.g., 90% CPU) trigger.
    """
    
    def __init__(self, history_window: int = 300):
        """
        Initialize system monitor.
        
        Args:
            history_window: Number of samples to keep for sliding window analysis
        """
        self.history_window = history_window
        self.metrics_history = deque(maxlen=history_window)
        self.feature_names = ['cpu_percent', 'memory_percent', 'disk_read_bytes', 
                            'disk_write_bytes', 'network_sent', 'network_recv']
    
    def fetch_system_telemetry_stream(self) -> np.ndarray:
        """
        Fetch current system metrics using psutil.
        
        Returns:
            NumPy array of shape (6,) containing:
            - CPU percent
            - Memory percent
            - Disk read bytes
            - Disk write bytes
            - Network bytes sent
            - Network bytes received
        """
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        
        disk_io = psutil.disk_io_counters()
        disk_read = disk_io.read_bytes if disk_io else 0
        disk_write = disk_io.write_bytes if disk_io else 0
        
        net_io = psutil.net_io_counters()
        net_sent = net_io.bytes_sent if net_io else 0
        net_recv = net_io.bytes_recv if net_io else 0
        
        metrics = np.array([cpu, memory, disk_read, disk_write, net_sent, net_recv])
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_sliding_window_matrix(self) -> np.ndarray:
        """
        Convert history deque to pandas DataFrame, then to NumPy matrix.
        
        Returns:
            NumPy array of shape (N, 6) for TDA analysis
        """
        if len(self.metrics_history) < 10:
            raise ValueError("Insufficient data for analysis (need at least 10 samples)")
        
        df = pd.DataFrame(list(self.metrics_history), columns=self.feature_names)
        
        # Normalize disk and network metrics to rates (bytes per sample)
        df['disk_read_rate'] = df['disk_read_bytes'].diff().fillna(0)
        df['disk_write_rate'] = df['disk_write_bytes'].diff().fillna(0)
        df['net_sent_rate'] = df['network_sent'].diff().fillna(0)
        df['net_recv_rate'] = df['network_recv'].diff().fillna(0)
        
        # Select features for TDA
        features = df[['cpu_percent', 'memory_percent', 'disk_read_rate', 
                      'disk_write_rate', 'net_sent_rate', 'net_recv_rate']].values
        
        return features
    
    def collect_continuous_stream(self, duration_seconds: int = 60, 
                                  interval: float = 0.5) -> np.ndarray:
        """
        Collect continuous system telemetry stream.
        
        Args:
            duration_seconds: Total collection time
            interval: Sampling interval in seconds
        
        Returns:
            NumPy array of time-series metrics
        """
        import time
        
        n_samples = int(duration_seconds / interval)
        
        for _ in range(n_samples):
            self.fetch_system_telemetry_stream()
            time.sleep(interval)
        
        return self.get_sliding_window_matrix()
    
    def detect_resource_anomaly(self, data: np.ndarray = None) -> dict:
        """
        Quick anomaly detection using statistical thresholds.
        Full topological analysis should be done via TDA-Guard core.
        
        Args:
            data: Optional pre-collected data matrix
        
        Returns:
            Dictionary with anomaly scores
        """
        if data is None:
            data = self.get_sliding_window_matrix()
        
        # Compute statistics
        cpu_mean = np.mean(data[:, 0])
        cpu_std = np.std(data[:, 0])
        memory_mean = np.mean(data[:, 1])
        
        # Anomaly indicators
        anomalies = {
            'cpu_spike': cpu_mean > 80,
            'cpu_volatility': cpu_std > 20,
            'memory_pressure': memory_mean > 85,
            'score': (cpu_mean / 100) * 0.4 + (memory_mean / 100) * 0.4 + (cpu_std / 100) * 0.2
        }
        
        return anomalies


# Convenience function for quick integration
def fetch_system_telemetry_stream(history_window: int = 300) -> np.ndarray:
    """
    Convenience function: Instant system telemetry fetch.
    
    Args:
        history_window: Number of samples to collect
    
    Returns:
        NumPy array ready for TDA-Guard
    """
    monitor = SystemResourceMonitor(history_window)
    
    # Quick collection (10 samples for demo)
    import time
    for _ in range(10):
        monitor.fetch_system_telemetry_stream()
        time.sleep(0.1)
    
    return monitor.get_sliding_window_matrix()
