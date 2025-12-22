#!/usr/bin/env python3
"""
Standalone training script to prove the from-scratch neural net can learn.
"""

import numpy as np
from nn.model import Sequential, set_seed
from nn.layers import Linear
from nn.activations import Tanh
from nn.losses import softmax_cross_entropy
from nn.optim import Adam


def generate_blobs_dataset(n_samples: int = 1000, noise: float = 0.1, seed: int = 42) -> tuple:
    """
    Generate a 2D classification dataset with two Gaussian blobs.
    
    Args:
        n_samples: Total number of samples (split between classes)
        noise: Standard deviation of the blobs
        seed: Random seed
        
    Returns:
        Tuple of (X, y) where X is (N, 2) and y is (N,) with labels 0/1
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2
    
    # Generate two Gaussian blobs
    # Class 0: centered at (-1, -1)
    X0 = np.random.randn(n_per_class, 2) * noise + np.array([-1.0, -1.0])
    y0 = np.zeros(n_per_class, dtype=np.int64)
    
    # Class 1: centered at (1, 1)
    X1 = np.random.randn(n_per_class, 2) * noise + np.array([1.0, 1.0])
    y1 = np.ones(n_per_class, dtype=np.int64)
    
    # Combine and shuffle
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def compute_accuracy(model: Sequential, X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        model: Trained Sequential model
        X: Input features, shape (N, 2)
        y: True labels, shape (N,)
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    # Forward pass to get logits
    logits = model.forward(X)
    
    # Predict class with highest logit
    predictions = np.argmax(logits, axis=1)
    
    # Compute accuracy
    accuracy = np.mean(predictions == y)
    return accuracy


def main():
    """Main training function."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Generate dataset
    print("Generating dataset...")
    X, y = generate_blobs_dataset(n_samples=1000, noise=0.3)
    n_samples = X.shape[0]
    print(f"Dataset: {n_samples} samples, 2 features, 2 classes")
    
    # Create model: Linear(2, 16) -> Tanh -> Linear(16, 16) -> Tanh -> Linear(16, 2)
    print("Creating model...")
    model = Sequential([
        Linear(2, 16),
        Tanh(),
        Linear(16, 16),
        Tanh(),
        Linear(16, 2)
    ])
    
    # Initialize optimizer
    optimizer = Adam(lr=1e-3)
    
    # Training hyperparameters
    epochs = 300
    batch_size = 32
    print_every = 25
    
    # Training loop
    print(f"Training for {epochs} epochs with batch_size={batch_size}...")
    loss_history = []
    
    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_losses = []
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            # Get batch
            end_idx = min(i + batch_size, n_samples)
            X_batch = X_shuffled[i:end_idx]
            y_batch = y_shuffled[i:end_idx]
            
            # Forward pass
            logits = model.forward(X_batch)
            loss, dlogits = softmax_cross_entropy(logits, y_batch)
            epoch_losses.append(loss)
            
            # Backward pass
            model.backward(dlogits)
            
            # Optimizer step
            optimizer.step(model.params_and_grads())
        
        # Track average loss for epoch
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        
        # Print progress
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Compute final accuracy
    print("\nComputing accuracy...")
    accuracy = compute_accuracy(model, X, y)
    accuracy_pct = accuracy * 100
    print(f"Final accuracy: {accuracy_pct:.2f}%")
    
    # Check if training was successful
    # For Gaussian blobs, expect >= 90% accuracy
    if accuracy >= 0.90:
        print("train ok")
    else:
        print(f"Warning: Accuracy {accuracy_pct:.2f}% is below 90% threshold")


if __name__ == "__main__":
    main()

