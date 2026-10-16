"""
Module B: Web Traffic Guard Adapter
Detects malicious scanning (SQL injection, directory traversal) via topological analysis.

Architecture:
Raw Access Log → Pandas Parsing → Numerical Feature Matrix → TDA-Guard Core → Alert
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict
from collections import Counter


class WebTrafficGuard:
    """
    Adapter for converting web access logs into topological features.
    Detects anomalous traffic patterns before attacks fully materialize.
    """
    
    def __init__(self):
        """Initialize web traffic guard."""
        self.common_log_pattern = re.compile(
            r'(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<time>.+?)\] '
            r'"(?P<method>\w+)\s+(?P<path>.+?)\s+HTTP/[\d.]+"\s+(?P<status>\d+)\s+.*'
        )
    
    def parse_access_log_to_manifold(self, log_file_path: str) -> np.ndarray:
        """
        Parse access log and convert to numerical feature matrix.
        
        Args:
            log_file_path: Path to access log file
        
        Returns:
            NumPy array of shape (N, D) containing numerical features
        """
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        return self.parse_access_log_lines(lines)

    def parse_access_log_lines(self, lines: List[str]) -> np.ndarray:
        """Parse access log lines into numeric features."""
        # Parse lines into structured data
        records = []
        for line in lines:
            match = self.common_log_pattern.search(line)
            if match:
                records.append(match.groupdict())
        
        if not records:
            raise ValueError("No valid log entries found")
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Extract numerical features
        features = self._extract_features(df)
        
        return features
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract numerical features from parsed log DataFrame.
        
        Features:
        - Request frequency (per IP)
        - Status code distribution
        - Method distribution
        - Path entropy
        - URL length statistics
        """
        # Feature 1: Request frequency per IP
        ip_counts = df['ip'].value_counts()
        df['ip_frequency'] = df['ip'].map(ip_counts)
        
        # Feature 2: Status code (numeric)
        df['status_code'] = df['status'].astype(int)
        
        # Feature 3: Method encoding (GET=0, POST=1, PUT=2, DELETE=3, OTHER=4)
        method_map = {'GET': 0, 'POST': 1, 'PUT': 2, 'DELETE': 3}
        df['method_encoded'] = df['method'].map(method_map).fillna(4)
        
        # Feature 4: Path length
        df['path_length'] = df['path'].str.len()
        
        # Feature 5: Path complexity (number of special characters)
        df['path_complexity'] = df['path'].str.count(r'[^a-zA-Z0-9/]')
        
        # Feature 6: IP entropy (entropy of IP distribution in sliding window)
        df['ip_entropy'] = self._compute_rolling_entropy(df['ip'], window=100)
        
        # Feature 7: Error rate (rolling window)
        df['error_rate'] = self._compute_rolling_error_rate(df['status_code'], window=50)
        
        # Select features for TDA
        feature_cols = ['ip_frequency', 'status_code', 'method_encoded', 
                       'path_length', 'path_complexity', 'ip_entropy', 'error_rate']
        
        features = df[feature_cols].values
        
        return features
    
    def _compute_rolling_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """Compute rolling entropy of IP distribution."""
        entropy_values = []
        
        for i in range(len(series)):
            start = max(0, i - window + 1)
            window_data = series.iloc[start:i+1]
            
            # Compute probability distribution
            counts = Counter(window_data)
            total = len(window_data)
            
            if total == 0:
                entropy_values.append(0.0)
                continue
            
            probs = [count / total for count in counts.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            entropy_values.append(entropy)
        
        return pd.Series(entropy_values)
    
    def _compute_rolling_error_rate(self, status_series: pd.Series, window: int) -> pd.Series:
        """Compute rolling error rate (4xx/5xx status codes)."""
        error_rates = []
        
        for i in range(len(status_series)):
            start = max(0, i - window + 1)
            window_data = status_series.iloc[start:i+1]
            
            if len(window_data) == 0:
                error_rates.append(0.0)
                continue
            
            errors = sum(1 for s in window_data if s >= 400)
            error_rates.append(errors / len(window_data))
        
        return pd.Series(error_rates)
    
    def detect_scan_patterns(self, data: np.ndarray) -> Dict[str, float]:
        """
        Quick detection of scanning patterns using statistical thresholds.
        Full topological analysis should be done via TDA-Guard core.
        
        Args:
            data: Feature matrix from parse_access_log_to_manifold
        
        Returns:
            Dictionary with anomaly scores
        """
        # Extract features
        ip_freq = data[:, 0]
        status_codes = data[:, 1]
        path_complexity = data[:, 4]
        error_rate = data[:, 6]
        
        # Anomaly indicators
        anomalies = {
            'high_frequency_ip': np.max(ip_freq) > 100,
            'high_error_rate': np.mean(error_rate) > 0.3,
            'path_complexity_spike': np.mean(path_complexity) > 5,
            'status_code_anomaly': np.std(status_codes) > 50,
            'score': (np.mean(error_rate) * 0.4 + 
                     (np.max(ip_freq) / 100) * 0.3 + 
                     (np.mean(path_complexity) / 10) * 0.3)
        }
        
        return anomalies
    
# Convenience function for quick integration
def parse_access_log_to_manifold(log_file_path: str) -> np.ndarray:
    """
    Convenience function: Parse access log to feature matrix.
    
    Args:
        log_file_path: Path to access log file
    
    Returns:
        NumPy array ready for TDA-Guard
    """
    guard = WebTrafficGuard()
    return guard.parse_access_log_to_manifold(log_file_path)
