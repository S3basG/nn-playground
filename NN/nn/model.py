import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Generator, Tuple


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


class Sequential:
    """Container for stacking layers sequentially."""
    
    def __init__(self, layers: List[Layer]):
        """
        Initialize sequential model.
        
        Args:
            layers: List of Layer instances to stack
        """
        self.layers = layers
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: feed input through each layer in order.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Feed through each layer sequentially
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass: backpropagate through layers in reverse order.
        
        Args:
            dout: Gradient from loss function
            
        Returns:
            Gradient w.r.t. input
        """
        # Backpropagate through layers in reverse order
        grad = dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def params_and_grads(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generator yielding (param, grad) pairs for all parameters in all layers.
        
        Yields:
            Tuple of (parameter array, gradient array)
        """
        # Iterate through all layers
        for layer in self.layers:
            params = layer.params()
            grads = layer.grads()
            # Yield each parameter and its corresponding gradient
            for param_name in params:
                yield params[param_name], grads[param_name]


if __name__ == "__main__":
    # Self-test for Sequential model
    from .layers import Linear
    from .activations import Tanh
    
    np.random.seed(42)
    
    # Build sequential model: Linear(2,4) -> Tanh() -> Linear(4,3)
    model = Sequential([
        Linear(2, 4),
        Tanh(),
        Linear(4, 3)
    ])
    
    # Forward pass with random input
    x = np.random.randn(5, 2)
    out = model.forward(x)
    assert out.shape == (5, 3), f"Expected output shape (5, 3), got {out.shape}"
    
    # Backward pass with random gradient
    dout = np.random.randn(5, 3)
    dx = model.backward(dout)
    assert dx.shape == x.shape, f"Expected dx shape {x.shape}, got {dx.shape}"
    
    # Check params_and_grads generator
    param_count = 0
    for param, grad in model.params_and_grads():
        assert param.shape == grad.shape, f"Param and grad shapes must match: {param.shape} vs {grad.shape}"
        param_count += 1
    assert param_count > 0, "Should have at least one parameter"
    
    print("sequential ok")

