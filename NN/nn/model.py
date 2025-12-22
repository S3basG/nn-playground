import numpy as np
from abc import ABC, abstractmethod
from typing import Dict


def set_seed(seed: int) -> None:
    """Set random seed for reproducible initialization."""
    np.random.seed(seed)


class Layer(ABC):
    """Base class for all neural network layers."""
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass (backpropagation).
        
        Args:
            grad: Gradient from the next layer
            
        Returns:
            Gradient to pass to the previous layer
        """
        pass
    
    @abstractmethod
    def params(self) -> Dict[str, np.ndarray]:
        """
        Get layer parameters (weights, biases, etc.).
        
        Returns:
            Dictionary mapping parameter names to numpy arrays
        """
        pass
    
    @abstractmethod
    def grads(self) -> Dict[str, np.ndarray]:
        """
        Get gradients for layer parameters.
        
        Returns:
            Dictionary mapping parameter names to their gradients
        """
        pass

