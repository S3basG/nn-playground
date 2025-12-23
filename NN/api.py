"""
FastAPI backend for training and inference with the from-scratch NumPy neural network.

Endpoints:
  POST /train - Train the neural network
  POST /predict - Run inference on the trained model
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from NN.nn import set_seed
from NN.nn.model import Sequential
from NN.nn.layers import Linear
from NN.nn.activations import Tanh
from NN.nn.losses import softmax_cross_entropy
from NN.nn.optim import Adam


app = FastAPI(title="Neural Network Playground API")

# Add CORS middleware before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state: store trained model
_trained_model: Optional[Sequential] = None
_trained_dataset: Optional[tuple[np.ndarray, np.ndarray]] = None


class TrainRequest(BaseModel):
    """Request model for training."""
    n_samples: int = Field(default=1000, ge=100, le=10000, description="Number of training samples")
    epochs: int = Field(default=300, ge=1, le=1000, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=1, le=256, description="Batch size for training")
    learning_rate: float = Field(default=1e-3, gt=0, le=1.0, description="Learning rate")
    hidden_size: int = Field(default=16, ge=4, le=128, description="Hidden layer size")
    seed: int = Field(default=0, description="Random seed for reproducibility")
    grad_clip: float = Field(default=5.0, gt=0, description="Gradient clipping value")
    # Aliases for frontend compatibility
    lr: Optional[float] = Field(default=None, gt=0, le=1.0, description="Learning rate (alias)")
    hidden: Optional[int] = Field(default=None, ge=4, le=128, description="Hidden layer size (alias)")
    grid_size: Optional[int] = Field(default=100, ge=50, le=200, description="Decision boundary grid resolution")


class TrainResponse(BaseModel):
    """Response model for training."""
    loss_history: list[float] = Field(description="Loss values for each epoch")
    final_accuracy: float = Field(description="Final accuracy on training set")
    dataset_points: list[dict] = Field(description="Dataset points with coordinates and labels")
    decision_boundary: dict = Field(description="Decision boundary grid data")
    epochs_trained: int = Field(description="Number of epochs trained")
    # Frontend-friendly aliases
    losses: list[float] = Field(description="Loss values (alias for loss_history)")
    X: list[list[float]] = Field(description="Dataset X coordinates as list of [x,y]")
    y: list[int] = Field(description="Dataset labels")


class PredictRequest(BaseModel):
    """Request model for prediction."""
    points: list[list[float]] = Field(description="List of 2D points to predict")


class PredictResponse(BaseModel):
    """Response model for prediction."""
    predictions: list[int] = Field(description="Predicted class labels")
    probabilities: list[list[float]] = Field(description="Class probabilities for each point")


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
    """Compute accuracy from logits and true labels."""
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))


def clip_grads_elementwise(model: Sequential, clip_value: float = 5.0) -> None:
    """Elementwise clip all gradients in-place to [-clip_value, clip_value]."""
    for _, grad in model.params_and_grads():
        np.clip(grad, -clip_value, clip_value, out=grad)


def assert_all_finite(model: Sequential) -> None:
    """Portable correctness check. If anything becomes NaN/Inf, fail loudly."""
    for param, grad in model.params_and_grads():
        if not np.isfinite(grad).all():
            raise ValueError("NONFINITE GRAD detected (NaN/Inf)")
        if not np.isfinite(param).all():
            raise ValueError("NONFINITE PARAM detected (NaN/Inf)")


def compute_decision_boundary(
    model: Sequential,
    x_min: float = -3.0,
    x_max: float = 3.0,
    y_min: float = -3.0,
    y_max: float = 3.0,
    resolution: int = 100
) -> dict:
    """
    Compute decision boundary grid for visualization.
    
    Returns:
        Dictionary with grid coordinates and predictions
    """
    # Create grid
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Flatten grid for prediction
    grid_points = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    
    # Get predictions
    logits = model.forward(grid_points)
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    predictions = np.argmax(logits, axis=1)
    
    # Reshape to grid
    pred_grid = predictions.reshape(xx.shape)
    prob_grid = probs[:, 1].reshape(xx.shape)  # Probability of class 1
    
    # Flatten probabilities for frontend
    flat_probs = prob_grid.flatten().tolist()
    
    return {
        "x_grid": xx.tolist(),
        "y_grid": yy.tolist(),
        "predictions": pred_grid.tolist(),
        "probabilities": prob_grid.tolist(),
        "x_range": [float(x_min), float(x_max)],
        "y_range": [float(y_min), float(y_max)],
        "resolution": resolution,
        # Frontend-friendly fields
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "grid_size": resolution,
        "probs_class1": flat_probs
    }


@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest) -> TrainResponse:
    """
    Train the neural network on a 2D classification dataset.
    
    Returns loss history, final accuracy, dataset points, and decision boundary.
    """
    global _trained_model, _trained_dataset
    
    # Set error handling
    np.seterr(all="warn")
    
    # Handle aliases
    learning_rate = request.lr if request.lr is not None else request.learning_rate
    hidden_size = request.hidden if request.hidden is not None else request.hidden_size
    grid_size = request.grid_size if request.grid_size is not None else 100
    
    # Generate dataset
    set_seed(request.seed)
    X, y = make_blobs(n=request.n_samples, seed=request.seed)
    
    # Safety: ensure dataset is finite
    if not np.isfinite(X).all():
        raise HTTPException(status_code=500, detail="Dataset X contains NaN/Inf")
    
    # Create model
    model = Sequential(
        [
            Linear(2, hidden_size, bias=True),
            Tanh(),
            Linear(hidden_size, hidden_size, bias=True),
            Tanh(),
            Linear(hidden_size, 2, bias=True),
        ]
    )
    
    # Optimizer
    optimizer = Adam(lr=learning_rate)
    
    # Training
    n = X.shape[0]
    loss_history: list[float] = []
    rng = np.random.default_rng(request.seed)
    
    for epoch in range(1, request.epochs + 1):
        # Shuffle each epoch
        perm = rng.permutation(n)
        
        epoch_loss_sum = 0.0
        num_batches = 0
        
        for start in range(0, n, request.batch_size):
            idx = perm[start : start + request.batch_size]
            X_batch = X[idx]
            y_batch = y[idx]
            
            # Forward
            logits = model.forward(X_batch)
            
            # Loss + gradient w.r.t logits
            loss, dlogits = softmax_cross_entropy(logits, y_batch)
            
            # Backward
            model.backward(dlogits)
            
            # Clip grads
            clip_grads_elementwise(model, clip_value=request.grad_clip)
            
            # Hard finite check
            assert_all_finite(model)
            
            # Update
            optimizer.step(model.params_and_grads())
            
            # Check again after update
            assert_all_finite(model)
            
            epoch_loss_sum += float(loss)
            num_batches += 1
        
        epoch_loss = epoch_loss_sum / max(1, num_batches)
        loss_history.append(epoch_loss)
    
    # Compute final accuracy
    logits_full = model.forward(X)
    final_accuracy = accuracy_from_logits(logits_full, y)
    
    # Prepare dataset points for response
    dataset_points = [
        {"x": float(X[i, 0]), "y": float(X[i, 1]), "label": int(y[i])}
        for i in range(len(X))
    ]
    
    # Compute decision boundary with requested grid size
    decision_boundary = compute_decision_boundary(model, resolution=grid_size)
    
    # Store model and dataset
    _trained_model = model
    _trained_dataset = (X, y)
    
    # Prepare response with frontend-friendly format
    X_list = [[float(X[i, 0]), float(X[i, 1])] for i in range(len(X))]
    y_list = [int(y[i]) for i in range(len(y))]
    
    return TrainResponse(
        loss_history=loss_history,
        final_accuracy=final_accuracy,
        dataset_points=dataset_points,
        decision_boundary=decision_boundary,
        epochs_trained=request.epochs,
        losses=loss_history,
        X=X_list,
        y=y_list
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Run inference on the trained model.
    
    Requires that /train has been called first.
    """
    global _trained_model
    
    if _trained_model is None:
        raise HTTPException(
            status_code=400,
            detail="No trained model available. Please call /train first."
        )
    
    # Validate input
    if not request.points:
        raise HTTPException(status_code=400, detail="No points provided")
    
    # Convert to numpy array
    try:
        points_array = np.array(request.points, dtype=np.float64)
        if points_array.shape[1] != 2:
            raise HTTPException(
                status_code=400,
                detail="Each point must be 2D (x, y coordinates)"
            )
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid point format: {str(e)}"
        )
    
    # Run inference
    logits = _trained_model.forward(points_array)
    
    # Compute probabilities (softmax)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Get predictions
    predictions = np.argmax(logits, axis=1).tolist()
    
    return PredictResponse(
        predictions=predictions,
        probabilities=probabilities.tolist()
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Neural Network Playground API",
        "endpoints": {
            "POST /train": "Train the neural network",
            "POST /predict": "Run inference on trained model",
            "GET /": "API information"
        },
        "model_trained": _trained_model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

