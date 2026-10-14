"""
TDA-Guard Industry Application Adapters

Unified interface for all industry-specific adapters:
- Module A: System Resource Monitor (psutil + pandas)
- Module B: Web Traffic Guard (pandas + regex)
- Module C: IoT Stream Guard (scikit-learn)
- Module D: AI Loss Landscape Guard (PyTorch)

All adapters follow the pattern:
Raw Data → Standardization → TDA-Guard Core → Alert
"""

from .module_a_system_monitor import SystemResourceMonitor, fetch_system_telemetry_stream
from .module_b_web_traffic import WebTrafficGuard, parse_access_log_to_manifold
from .module_c_iot_stream import IoTStreamGuard, adapt_iot_traffic
from .module_d_ai_landscape import AILossLandscapeGuard, extract_model_weights_manifold
from .unified_adapter import UnifiedAdapter, quick_analyze

__all__ = [
    'SystemResourceMonitor',
    'fetch_system_telemetry_stream',
    'WebTrafficGuard',
    'parse_access_log_to_manifold',
    'IoTStreamGuard',
    'adapt_iot_traffic',
    'AILossLandscapeGuard',
    'extract_model_weights_manifold',
    'UnifiedAdapter',
    'quick_analyze',
]
