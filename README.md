# Neural Network Playground

An interactive web application for training and visualizing a neural network built from scratch using only NumPy. This project demonstrates core deep learning concepts including forward/backward propagation, optimization algorithms, and numerical stability techniques.

## What It Does

This project implements a complete neural network library in pure NumPy (no PyTorch/TensorFlow) and provides a web interface to:
- Train a neural network on a 2D classification dataset (two Gaussian blobs)
- Visualize the decision boundary as a heatmap
- View the training loss curve
- Adjust hyperparameters (epochs, learning rate, hidden layer size) via interactive sliders

The neural network learns to classify 2D points into two classes, and you can see how the decision boundary evolves during training.

## Tech Stack

- **Backend**: FastAPI (Python web framework)
- **Neural Network**: Pure NumPy implementation
  - Layers: Linear (fully connected)
  - Activations: Tanh
  - Loss: Softmax Cross-Entropy
  - Optimizer: Adam with bias correction
- **Frontend**: Vanilla HTML/CSS/JavaScript (no frameworks)
- **Visualization**: HTML5 Canvas API

## How to Run

### Prerequisites

- Python 3.9+
- pip

### Backend Setup

1. Install dependencies:
```bash
cd NN
pip install fastapi uvicorn numpy
```

2. Start the FastAPI server:
```bash
python api.py
```

The API will be available at `http://127.0.0.1:8000`

### Frontend Setup

1. Open `frontend/index.html` in a web browser
   - You can use a simple HTTP server if needed:
   ```bash
   # Python 3
   cd frontend
   python -m http.server 5173
   ```
   Then open `http://127.0.0.1:5173` in your browser

2. Click "Train Model" to start training
3. Adjust sliders to change hyperparameters and retrain

## Screenshots

![Neural Network Playground Interface](screenshot.png)

*The interface shows the decision boundary heatmap (left) and training loss curve (right), with interactive controls for adjusting training hyperparameters.*

## Project Structure

```
NN/
├── nn/                 # Neural network implementation
│   ├── layers.py      # Linear layer (forward/backward)
│   ├── activations.py # Activation functions (ReLU, Tanh, Sigmoid)
│   ├── losses.py      # Loss functions (MSE, Softmax Cross-Entropy)
│   ├── optim.py       # Optimizers (SGD, Adam)
│   └── model.py       # Sequential container and Layer base class
├── api.py             # FastAPI backend with /train and /predict endpoints
├── train_toy.py       # Standalone training script for testing
└── frontend/
    └── index.html     # Single-page web interface
```

## Key Features

- **From Scratch Implementation**: No ML frameworks - pure NumPy
- **Numerical Stability**: Gradient clipping, NaN/Inf checks, stable softmax
- **Interactive Visualization**: Real-time decision boundary and loss plots
- **Production-Ready**: Error handling, CORS support, input validation

