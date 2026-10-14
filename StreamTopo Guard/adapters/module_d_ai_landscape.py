"""
Module D: AI Loss Landscape Guard Adapter
Detects model collapse and hallucination by analyzing weight manifold topology.

Architecture:
PyTorch Model Weights → High-Dimensional Point Cloud → TDA-Guard Core → Alert
"""

import numpy as np
from typing import List, Tuple, Optional


class AILossLandscapeGuard:
    """
    Adapter for extracting and analyzing neural network weight manifolds.
    Detects geometric fractures in loss landscape that indicate collapse/hallucination.
    """
    
    def __init__(self, tensor_dim: int = 16):
        """
        Initialize AI loss landscape guard.
        
        Args:
            tensor_dim: Dimension to reshape weights into for point cloud
        """
        self.tensor_dim = tensor_dim
        self.weight_history = []
        self.has_torch = self._check_torch()
    
    def _check_torch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def extract_model_weights_manifold(self, model) -> np.ndarray:
        """
        Extract neural network weights and reshape into high-dimensional point cloud.
        
        Args:
            model: PyTorch model (torch.nn.Module)
        
        Returns:
            NumPy array of shape (N, tensor_dim) representing weight manifold
        """
        if not self.has_torch:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")
        
        import torch
        
        # Flatten all parameters into single vector
        weights = []
        for param in model.parameters():
            weights.append(param.data.flatten())
        
        if not weights:
            raise ValueError("Model has no parameters")
        
        # Concatenate all weights
        flat_weights = torch.cat(weights).cpu().numpy()
        
        # Reshape into high-dimensional tensor space
        # Pad if necessary to match tensor_dim
        n_elements = flat_weights.shape[0]
        n_points = n_elements // self.tensor_dim
        
        if n_points == 0:
            raise ValueError(f"Model has too few parameters ({n_elements}) for tensor_dim={self.tensor_dim}")
        
        # Truncate to exact multiple
        flat_weights = flat_weights[:n_points * self.tensor_dim]
        
        # Reshape to (N, tensor_dim)
        manifold = flat_weights.reshape(n_points, self.tensor_dim)
        
        return manifold
    
    def extract_weight_snapshots(self, model, n_snapshots: int = 10) -> List[np.ndarray]:
        """
        Extract multiple weight snapshots during training for temporal analysis.
        
        Args:
            model: PyTorch model
            n_snapshots: Number of snapshots to collect
        
        Returns:
            List of weight manifolds
        """
        snapshot = self.extract_model_weights_manifold(model)
        self.weight_history.append(snapshot)
        
        # Keep only last n_snapshots
        if len(self.weight_history) > n_snapshots:
            self.weight_history.pop(0)
        
        return self.weight_history
    
    def detect_landscape_fracture(self, current_weights: np.ndarray, 
                                  baseline_weights: np.ndarray) -> dict:
        """
        Detect geometric fractures in loss landscape.
        Full topological analysis should be done via TDA-Guard core.
        
        Args:
            current_weights: Current weight manifold
            baseline_weights: Baseline (healthy) weight manifold
        
        Returns:
            Dictionary with fracture indicators
        """
        # Compute distance between manifolds
        if current_weights.shape != baseline_weights.shape:
            # Resample to match sizes
            min_size = min(current_weights.shape[0], baseline_weights.shape[0])
            current_weights = current_weights[:min_size]
            baseline_weights = baseline_weights[:min_size]
        
        # Euclidean distance between corresponding points
        distances = np.linalg.norm(current_weights - baseline_weights, axis=1)
        
        # Fracture indicators
        fracture = {
            'mean_distance': np.mean(distances),
            'max_distance': np.max(distances),
            'distance_variance': np.var(distances),
            'fracture_score': np.mean(distances) / (np.std(distances) + 1e-8),
            'is_fractured': np.mean(distances) > 0.5 and np.var(distances) > 0.1
        }
        
        return fracture
    
    def generate_synthetic_model(self, n_layers: int = 3, 
                                layer_size: int = 64) -> object:
        """
        Generate synthetic PyTorch model for testing.
        
        Args:
            n_layers: Number of layers
            layer_size: Size of each layer
        
        Returns:
            PyTorch model
        """
        if not self.has_torch:
            raise ImportError("PyTorch is not installed")
        
        import torch
        import torch.nn as nn
        
        layers = []
        input_size = layer_size
        
        for i in range(n_layers):
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.ReLU())
            input_size = layer_size
        
        model = nn.Sequential(*layers)
        
        # Initialize with random weights
        for param in model.parameters():
            nn.init.normal_(param, mean=0.0, std=0.1)
        
        return model
    
    def simulate_training_step(self, model, learning_rate: float = 0.01):
        """
        Simulate a training step by perturbing weights.
        
        Args:
            model: PyTorch model
            learning_rate: Learning rate for weight updates
        """
        if not self.has_torch:
            raise ImportError("PyTorch is not installed")
        
        import torch
        
        with torch.no_grad():
            for param in model.parameters():
                # Simulate gradient descent with random gradients
                gradient = torch.randn_like(param) * 0.1
                param.data -= learning_rate * gradient


# Convenience function for quick integration
def extract_model_weights_manifold(model, tensor_dim: int = 16) -> np.ndarray:
    """
    Convenience function: Extract model weights to manifold.
    
    Args:
        model: PyTorch model
        tensor_dim: Dimension for reshaping
    
    Returns:
        NumPy array ready for TDA-Guard
    """
    guard = AILossLandscapeGuard(tensor_dim=tensor_dim)
    return guard.extract_model_weights_manifold(model)


# Fallback: NumPy-only version for when PyTorch is not available
class NumpyModelMock:
    """
    Mock model using NumPy for testing without PyTorch.
    """
    
    def __init__(self, n_parameters: int = 1000):
        """Initialize mock model with random weights."""
        self.weights = np.random.randn(n_parameters)
    
    def get_weights(self) -> np.ndarray:
        """Get weight vector."""
        return self.weights
    
    def update_weights(self, learning_rate: float = 0.01):
        """Simulate training step."""
        gradient = np.random.randn(*self.weights.shape) * 0.1
        self.weights -= learning_rate * gradient


def extract_numpy_model_manifold(model: NumpyModelMock, 
                                tensor_dim: int = 16) -> np.ndarray:
    """
    Extract weights from NumPy mock model.
    
    Args:
        model: NumpyModelMock instance
        tensor_dim: Dimension for reshaping
    
    Returns:
        NumPy array ready for TDA-Guard
    """
    weights = model.get_weights()
    n_elements = weights.shape[0]
    n_points = n_elements // tensor_dim
    
    if n_points == 0:
        raise ValueError(f"Model has too few parameters ({n_elements}) for tensor_dim={tensor_dim}")
    
    flat_weights = weights[:n_points * tensor_dim]
    manifold = flat_weights.reshape(n_points, tensor_dim)
    
    return manifold
