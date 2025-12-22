import numpy as np
from typing import Dict
from .model import Layer


class Linear(Layer):
    """Fully connected (linear) layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias term
        """
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        # Xavier/Glorot uniform initialization
        # Limit: sqrt(6 / (fan_in + fan_out))
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        
        # Initialize bias to zero if present
        if bias:
            self.b = np.zeros(out_features)
        else:
            self.b = None
        
        # Storage for forward pass input (needed for backward)
        self.x = None
        
        # Storage for gradients
        self.dW = None
        self.db = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = x @ W + b
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Store input for backward pass
        self.x = x
        
        # Compute output: x @ W + b
        out = x @ self.W
        if self.bias:
            out = out + self.b
        
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute gradients and return input gradient.
        
        Args:
            grad: Gradient from next layer, shape (batch_size, out_features)
            
        Returns:
            Gradient w.r.t. input, shape (batch_size, in_features)
        """
        # Gradient w.r.t. input: dx = dout @ W.T
        dx = grad @ self.W.T
        
        # Gradient w.r.t. weights: dW = x.T @ dout
        self.dW = self.x.T @ grad
        
        # Gradient w.r.t. bias: db = sum(dout, axis=0)
        if self.bias:
            self.db = np.sum(grad, axis=0)
        else:
            self.db = None
        
        return dx
    
    def params(self) -> Dict[str, np.ndarray]:
        """
        Get layer parameters.
        
        Returns:
            Dictionary with 'W' and optionally 'b'
        """
        params = {"W": self.W}
        if self.bias:
            params["b"] = self.b
        return params
    
    def grads(self) -> Dict[str, np.ndarray]:
        """
        Get layer gradients.
        
        Returns:
            Dictionary with 'W' gradient and optionally 'b' gradient
        """
        grads = {"W": self.dW}
        if self.bias:
            grads["b"] = self.db
        return grads


if __name__ == "__main__":
    # Self-test for Linear layer
    np.random.seed(42)
    
    # Create layer: 10 input features, 5 output features, with bias
    layer = Linear(in_features=10, out_features=5, bias=True)
    
    # Create random input: batch_size=3, in_features=10
    x = np.random.randn(3, 10)
    
    # Forward pass
    out = layer.forward(x)
    
    # Check output shape
    assert out.shape == (3, 5), f"Expected output shape (3, 5), got {out.shape}"
    
    # Create random gradient from next layer
    dout = np.random.randn(3, 5)
    
    # Backward pass
    dx = layer.backward(dout)
    
    # Check input gradient shape
    assert dx.shape == (3, 10), f"Expected dx shape (3, 10), got {dx.shape}"
    
    # Check weight gradient shape
    assert layer.dW.shape == (10, 5), f"Expected dW shape (10, 5), got {layer.dW.shape}"
    
    # Check bias gradient shape
    assert layer.db.shape == (5,), f"Expected db shape (5,), got {layer.db.shape}"
    
    # Check params and grads dicts
    params = layer.params()
    assert "W" in params and "b" in params, "params() should contain 'W' and 'b'"
    
    grads = layer.grads()
    assert "W" in grads and "b" in grads, "grads() should contain 'W' and 'b'"
    
    print("linear ok")

