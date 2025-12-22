import numpy as np
from typing import Generator, Tuple, Dict


class SGD:
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, lr: float = 1e-2):
        """
        Initialize SGD optimizer.
        
        Args:
            lr: Learning rate
        """
        self.lr = lr
    
    def step(self, params_and_grads: Generator[Tuple[np.ndarray, np.ndarray], None, None]) -> None:
        """
        Perform one optimization step.
        
        Args:
            params_and_grads: Generator yielding (param, grad) tuples
        """
        # Update each parameter: p -= lr * g
        for param, grad in params_and_grads:
            param -= self.lr * grad


class Adam:
    """Adam optimizer with bias correction."""
    
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            lr: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            eps: Small constant for numerical stability
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Per-parameter state: m (first moment) and v (second moment)
        # Keyed by id(param) to track each parameter array
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        
        # Timestep counter
        self.t = 0
    
    def step(self, params_and_grads: Generator[Tuple[np.ndarray, np.ndarray], None, None]) -> None:
        """
        Perform one Adam optimization step with bias correction.
        
        Args:
            params_and_grads: Generator yielding (param, grad) tuples
        """
        # Increment timestep
        self.t += 1
        
        # Process each parameter
        for param, grad in params_and_grads:
            param_id = id(param)
            
            # Initialize state for new parameters
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(param)
                self.v[param_id] = np.zeros_like(param)
            
            # Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate: v = beta2 * v + (1 - beta2) * g^2
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
            
            # Update parameter: p -= lr * m_hat / (sqrt(v_hat) + eps)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


if __name__ == "__main__":
    # Self-test for optimizers
    np.random.seed(42)
    
    # Create fake parameter and gradient
    p_sgd = np.random.randn(3, 4).astype(np.float64)
    p_adam = np.random.randn(3, 4).astype(np.float64)
    g = np.random.randn(3, 4).astype(np.float64)
    
    # Store initial values
    p_sgd_initial = p_sgd.copy()
    p_adam_initial = p_adam.copy()
    
    # Test SGD
    sgd = SGD(lr=0.01)
    for _ in range(5):
        def sgd_gen():
            yield (p_sgd, g)
        sgd.step(sgd_gen())
    
    # Check SGD: parameter should change and remain finite
    assert not np.allclose(p_sgd, p_sgd_initial), "SGD: Parameter should change"
    assert np.all(np.isfinite(p_sgd)), "SGD: Parameter should remain finite"
    
    # Test Adam
    adam = Adam(lr=0.001)
    for _ in range(5):
        def adam_gen():
            yield (p_adam, g)
        adam.step(adam_gen())
    
    # Check Adam: parameter should change and remain finite
    assert not np.allclose(p_adam, p_adam_initial), "Adam: Parameter should change"
    assert np.all(np.isfinite(p_adam)), "Adam: Parameter should remain finite"
    
    print("optim ok")

