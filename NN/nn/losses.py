import numpy as np
from typing import Tuple


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Mean Squared Error loss.
    
    Args:
        y_pred: Predictions, shape (N, D) or (N,)
        y_true: Ground truth, shape (N, D) or (N,)
        
    Returns:
        Tuple of (loss: float, gradient w.r.t. y_pred: np.ndarray)
    """
    # Compute squared differences
    diff = y_pred - y_true
    squared_diff = diff ** 2
    
    # Mean over all elements
    loss = np.mean(squared_diff)
    
    # Gradient: d/dy_pred of mean((y_pred - y_true)^2)
    # = 2 * (y_pred - y_true) / N (mean normalization)
    dypred = 2 * diff / diff.size
    
    return loss, dypred


def softmax_cross_entropy(logits: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Softmax Cross-Entropy loss (numerically stable).
    
    Args:
        logits: Unnormalized log probabilities, shape (N, C)
        y_true: Integer class labels, shape (N,)
        
    Returns:
        Tuple of (loss: float, gradient w.r.t. logits: np.ndarray)
    """
    N, C = logits.shape
    
    # Numerical stability: subtract max from each row
    # This doesn't change the softmax but prevents overflow
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    
    # Compute exp of shifted logits
    exp_logits = np.exp(logits_shifted)
    
    # Softmax: exp(logits) / sum(exp(logits))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Compute cross-entropy loss
    # For each sample, take -log(softmax[correct_class])
    # Use log-sum-exp trick: log(sum(exp(logits))) = max + log(sum(exp(logits - max)))
    log_sum_exp = np.log(np.sum(exp_logits, axis=1))
    correct_logits = logits_shifted[np.arange(N), y_true]
    loss = np.mean(log_sum_exp - correct_logits)
    
    # Gradient: softmax - one_hot(y_true)
    # For mean loss, divide by N
    one_hot = np.zeros_like(softmax_probs)
    one_hot[np.arange(N), y_true] = 1.0
    dlogits = (softmax_probs - one_hot) / N
    
    return loss, dlogits


if __name__ == "__main__":
    # Self-test for loss functions
    np.random.seed(42)
    
    # Test MSE loss
    y_pred = np.random.randn(10, 5)
    y_true = np.random.randn(10, 5)
    mse, dypred = mse_loss(y_pred, y_true)
    assert np.isfinite(mse), "MSE loss should be finite"
    assert dypred.shape == y_pred.shape, f"MSE: Expected dypred shape {y_pred.shape}, got {dypred.shape}"
    
    # Test MSE with 1D arrays
    y_pred_1d = np.random.randn(10)
    y_true_1d = np.random.randn(10)
    mse_1d, dypred_1d = mse_loss(y_pred_1d, y_true_1d)
    assert np.isfinite(mse_1d), "MSE loss (1D) should be finite"
    assert dypred_1d.shape == y_pred_1d.shape, f"MSE (1D): Expected dypred shape {y_pred_1d.shape}, got {dypred_1d.shape}"
    
    # Test Softmax Cross-Entropy
    N, C = 20, 5
    logits = np.random.randn(N, C)
    y_true = np.random.randint(0, C, size=N)
    
    loss, dlogits = softmax_cross_entropy(logits, y_true)
    assert np.isfinite(loss), "Softmax cross-entropy loss should be finite"
    assert dlogits.shape == logits.shape, f"Expected dlogits shape {logits.shape}, got {dlogits.shape}"
    
    print("losses ok")

