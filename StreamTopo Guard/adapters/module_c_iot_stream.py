"""
Module C: IoT Stream Guard Adapter
Detects botnet infection by analyzing traffic flow topology (laminar vs turbulent).

Architecture:
Raw Packet Flows → Scikit-Learn StandardScaler → TDA-Guard Core → Alert
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
from collections import deque


class IoTStreamGuard:
    """
    Adapter for IoT/P2P network traffic analysis.
    Detects transition from stable laminar flow to chaotic turbulent flow (botnet).
    """
    
    def __init__(self, window_size: int = 500):
        """
        Initialize IoT stream guard.
        
        Args:
            window_size: Size of sliding window for continuous monitoring
        """
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.traffic_history = deque(maxlen=window_size)
        self.is_fitted = False
    
    def adapt_iot_traffic(self, packet_flows: np.ndarray) -> np.ndarray:
        """
        Standardize high-dimensional packet flows using scikit-learn.
        Ensures geometric equivalence across different magnitude scales.
        
        Args:
            packet_flows: Raw packet flow data of shape (N, D)
                         Features may include: packet_size, inter_arrival_time,
                         src_port_entropy, dst_port_entropy, protocol_type, etc.
        
        Returns:
            Standardized NumPy array ready for TDA-Guard
        """
        if not self.is_fitted:
            # Fit scaler on first batch
            standardized = self.scaler.fit_transform(packet_flows)
            self.is_fitted = True
        else:
            # Transform using fitted scaler
            standardized = self.scaler.transform(packet_flows)
        
        return standardized
    
    def generate_sample_iot_traffic(self, n_samples: int = 500, 
                                   mode: str = 'normal') -> np.ndarray:
        """
        Generate synthetic IoT traffic for testing.
        
        Args:
            n_samples: Number of samples to generate
            mode: 'normal' (laminar flow) or 'botnet' (turbulent flow)
        
        Returns:
            NumPy array of shape (n_samples, 6)
        """
        np.random.seed(42)
        
        if mode == 'normal':
            # Laminar flow: stable, predictable patterns
            packet_size = np.random.normal(500, 50, n_samples)
            inter_arrival = np.random.normal(0.01, 0.001, n_samples)
            src_entropy = np.random.normal(2.5, 0.2, n_samples)
            dst_entropy = np.random.normal(3.0, 0.3, n_samples)
            protocol_type = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
            flow_duration = np.random.normal(5.0, 0.5, n_samples)
        else:
            # Turbulent flow (botnet): chaotic, high variance
            packet_size = np.random.normal(1200, 300, n_samples)
            inter_arrival = np.random.exponential(0.05, n_samples)
            src_entropy = np.random.uniform(0.5, 4.0, n_samples)
            dst_entropy = np.random.uniform(1.0, 4.5, n_samples)
            protocol_type = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])
            flow_duration = np.random.exponential(2.0, n_samples)
        
        # Combine features
        traffic = np.column_stack([
            packet_size,
            inter_arrival,
            src_entropy,
            dst_entropy,
            protocol_type,
            flow_duration
        ])
        
        return traffic
    
    def add_traffic_sample(self, sample: np.ndarray):
        """
        Add single traffic sample to history.
        
        Args:
            sample: Single sample of shape (D,)
        """
        self.traffic_history.append(sample)
    
    def get_window_matrix(self) -> np.ndarray:
        """
        Get current window as matrix for TDA analysis.
        
        Returns:
            NumPy array of shape (N, D)
        """
        if len(self.traffic_history) < 10:
            raise ValueError("Insufficient data in window (need at least 10 samples)")
        
        matrix = np.array(list(self.traffic_history))
        return matrix
    
    def detect_flow_regime(self, data: np.ndarray) -> dict:
        """
        Quick detection of flow regime (laminar vs turbulent).
        Full topological analysis should be done via TDA-Guard core.
        
        Args:
            data: Standardized traffic matrix
        
        Returns:
            Dictionary with flow regime indicators
        """
        # Compute statistical features
        packet_size_std = np.std(data[:, 0])
        inter_arrival_cv = np.std(data[:, 1]) / (np.mean(data[:, 1]) + 1e-8)
        entropy_variance = np.var(data[:, 2])
        
        # Turbulence indicators
        regime = {
            'packet_size_volatility': packet_size_std > 100,
            'arrival_chaos': inter_arrival_cv > 0.5,
            'entropy_instability': entropy_variance > 0.5,
            'turbulence_score': (packet_size_std / 100) * 0.4 + 
                               inter_arrival_cv * 0.3 + 
                               entropy_variance * 0.3
        }
        
        return regime
    
    def continuous_monitor(self, data_stream: np.ndarray, 
                          step: int = 50) -> List[dict]:
        """
        Continuous monitoring with sliding window.
        
        Args:
            data_stream: Full traffic stream
            step: Step size for sliding window
        
        Returns:
            List of regime detections for each window
        """
        n_samples = data_stream.shape[0]
        detections = []
        
        for start in range(0, n_samples - self.window_size + 1, step):
            end = start + self.window_size
            window = data_stream[start:end]
            
            # Standardize window
            standardized = self.adapt_iot_traffic(window)
            
            # Detect regime
            regime = self.detect_flow_regime(standardized)
            detections.append(regime)
        
        return detections


# Convenience function for quick integration
def adapt_iot_traffic(packet_flows: np.ndarray) -> np.ndarray:
    """
    Convenience function: Standardize IoT traffic flows.
    
    Args:
        packet_flows: Raw packet flow data
    
    Returns:
        Standardized NumPy array ready for TDA-Guard
    """
    guard = IoTStreamGuard()
    return guard.adapt_iot_traffic(packet_flows)
