import numpy as np
from typing import Dict
from .model import Layer


class ReLU(Layer):
    """Rectified Linear Unit activation: max(0, x)."""
    
    def __init__(self):
        """Initialize ReLU layer."""
        # Store input for backward pass
        self.x = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = max(0, x)
        
        Args:
            x: Input tensor
            
        Returns:
            Activated output
        """
        # Store input for backward
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass: dx = dout * (x > 0)
        
        Args:
            grad: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input
        """
        # Gradient is zero where input was negative, otherwise pass through
        dx = grad * (self.x > 0)
        return dx
    
    def params(self) -> Dict[str, np.ndarray]:
        """ReLU has no parameters."""
        return {}
    
    def grads(self) -> Dict[str, np.ndarray]:
        """ReLU has no gradients."""
        return {}


class Tanh(Layer):
    """Hyperbolic tangent activation."""
    
    def __init__(self):
        """Initialize Tanh layer."""
        # Store output for backward pass
        self.y = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = tanh(x)
        
        Args:
            x: Input tensor
            
        Returns:
            Activated output
        """
        # Compute and store output for backward
        self.y = np.tanh(x)
        return self.y
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass: dx = dout * (1 - tanh(x)^2) = dout * (1 - y^2)
        
        Args:
            grad: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input
        """
        # Derivative of tanh: 1 - tanh(x)^2
        dx = grad * (1 - self.y ** 2)
        return dx
    
    def params(self) -> Dict[str, np.ndarray]:
        """Tanh has no parameters."""
        return {}
    
    def grads(self) -> Dict[str, np.ndarray]:
        """Tanh has no gradients."""
        return {}


class Sigmoid(Layer):
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    
    def __init__(self):
        """Initialize Sigmoid layer."""
        # Store output for backward pass
        self.y = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = 1 / (1 + exp(-x))
        
        Args:
            x: Input tensor
            
        Returns:
            Activated output
        """
        # Compute and store output for backward
        # Use numerical stability: clip x to avoid overflow
        x_clipped = np.clip(x, -500, 500)
        self.y = 1.0 / (1.0 + np.exp(-x_clipped))
        return self.y
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass: dx = dout * y * (1 - y)
        
        Args:
            grad: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input
        """
        # Derivative of sigmoid: y * (1 - y)
        dx = grad * self.y * (1 - self.y)
        return dx
    
    def params(self) -> Dict[str, np.ndarray]:
        """Sigmoid has no parameters."""
        return {}
    
    def grads(self) -> Dict[str, np.ndarray]:
        """Sigmoid has no gradients."""
        return {}


if __name__ == "__main__":
    # Self-test for activation layers
    np.random.seed(42)
    
    # Create random input and gradient
    x = np.random.randn(3, 5)
    dout = np.random.randn(3, 5)
    
    # Test ReLU
    relu = ReLU()
    relu_out = relu.forward(x)
    relu_dx = relu.backward(dout)
    assert relu_dx.shape == x.shape, f"ReLU: Expected dx shape {x.shape}, got {relu_dx.shape}"
    assert relu.params() == {}, "ReLU params() should return empty dict"
    assert relu.grads() == {}, "ReLU grads() should return empty dict"
    
    # Test Tanh
    tanh = Tanh()
    tanh_out = tanh.forward(x)
    tanh_dx = tanh.backward(dout)
    assert tanh_dx.shape == x.shape, f"Tanh: Expected dx shape {x.shape}, got {tanh_dx.shape}"
    assert tanh.params() == {}, "Tanh params() should return empty dict"
    assert tanh.grads() == {}, "Tanh grads() should return empty dict"
    
    # Test Sigmoid
    sigmoid = Sigmoid()
    sigmoid_out = sigmoid.forward(x)
    sigmoid_dx = sigmoid.backward(dout)
    assert sigmoid_dx.shape == x.shape, f"Sigmoid: Expected dx shape {x.shape}, got {sigmoid_dx.shape}"
    assert sigmoid.params() == {}, "Sigmoid params() should return empty dict"
    assert sigmoid.grads() == {}, "Sigmoid grads() should return empty dict"
    
    print("activations ok")

