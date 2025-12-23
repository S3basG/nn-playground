"""
Toy training script to prove the from-scratch NumPy neural net learns.

Run:
  python3 NN/train_toy.py
"""

from __future__ import annotations

import numpy as np

from NN.nn import set_seed
from NN.nn.model import Sequential
from NN.nn.layers import Linear
from NN.nn.activations import Tanh
from NN.nn.losses import softmax_cross_entropy
from NN.nn.optim import Adam


def make_blobs(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Two Gaussian blobs in 2D.
    Returns:
      X: (N,2) float64
      y: (N,) int64 labels {0,1}
    """
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0

    mean0 = np.array([-1.0, -1.0], dtype=np.float64)
    mean1 = np.array([+1.0, +1.0], dtype=np.float64)
    cov = np.array([[0.35, 0.0], [0.0, 0.35]], dtype=np.float64)

    x0 = rng.multivariate_normal(mean0, cov, size=n0).astype(np.float64)
    x1 = rng.multivariate_normal(mean1, cov, size=n1).astype(np.float64)

    X = np.vstack([x0, x1]).astype(np.float64)
    y = np.concatenate([np.zeros(n0, dtype=np.int64), np.ones(n1, dtype=np.int64)])

    # shuffle
    idx = rng.permutation(n)
    X = X[idx]
    y = y[idx]
    return X, y


def accuracy_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))


def clip_grads_elementwise(model: Sequential, clip_value: float = 5.0) -> None:
    """
    Elementwise clip all gradients in-place to [-clip_value, clip_value].
    """
    for _, grad in model.params_and_grads():
        np.clip(grad, -clip_value, clip_value, out=grad)


def assert_all_finite(model: Sequential) -> None:
    """
    Portable correctness check. If anything becomes NaN/Inf, fail loudly.
    """
    for param, grad in model.params_and_grads():
        if not np.isfinite(grad).all():
            raise ValueError("NONFINITE GRAD detected (NaN/Inf)")
        if not np.isfinite(param).all():
            raise ValueError("NONFINITE PARAM detected (NaN/Inf)")


def main() -> None:
    # NOTE: do NOT use np.seterr(all="raise") here.
    # On some macOS BLAS backends, matmul may raise spurious FloatingPointError.
    np.seterr(all="warn")

    print("Generating dataset...")
    set_seed(0)
    X, y = make_blobs(n=1000, seed=0)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, 2 classes")
    print(f"  X dtype: {X.dtype}, max abs: {float(np.max(np.abs(X))):.6f}")
    print(f"  Label distribution: class 0={(y==0).sum()}, class 1={(y==1).sum()}")

    # Safety: ensure dataset is finite
    if not np.isfinite(X).all():
        raise ValueError("Dataset X contains NaN/Inf")

    print("Creating model...")
    model = Sequential(
        [
            Linear(2, 16, bias=True),
            Tanh(),
            Linear(16, 16, bias=True),
            Tanh(),
            Linear(16, 2, bias=True),
        ]
    )

    # Optimizer
    optimizer = Adam(lr=1e-3)

    # Training hyperparams
    epochs = 300
    batch_size = 32
    grad_clip = 5.0

    print(f"Training for {epochs} epochs with batch_size={batch_size}...")

    n = X.shape[0]
    loss_history: list[float] = []

    rng = np.random.default_rng(0)

    for epoch in range(1, epochs + 1):
        # Shuffle each epoch
        perm = rng.permutation(n)

        epoch_loss_sum = 0.0
        num_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            X_batch = X[idx]
            y_batch = y[idx]

            # Forward
            logits = model.forward(X_batch)

            # Loss + gradient w.r.t logits
            loss, dlogits = softmax_cross_entropy(logits, y_batch)

            # Backward
            model.backward(dlogits)

            # Clip grads (helps prevent exploding updates)
            clip_grads_elementwise(model, clip_value=grad_clip)

            # Hard finite check (portable + real)
            assert_all_finite(model)

            # Update
            optimizer.step(model.params_and_grads())

            # Check again after update
            assert_all_finite(model)

            epoch_loss_sum += float(loss)
            num_batches += 1

        epoch_loss = epoch_loss_sum / max(1, num_batches)
        loss_history.append(epoch_loss)

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

    print("\nComputing accuracy...")
    logits_full = model.forward(X)
    acc = accuracy_from_logits(logits_full, y)
    print(f"Final accuracy: {acc*100:.2f}%")

    # For blobs we expect it to be very high
    if acc >= 0.90:
        print("train ok")
    else:
        print("train weak (try more epochs or bigger hidden size)")


if __name__ == "__main__":
    main()