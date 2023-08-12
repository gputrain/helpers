import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """
    Plots the decision boundary for a given model and data.

    Args:
        model (torch.nn.Module): The trained model for which the decision boundary will be plotted.
        X (torch.Tensor): The input features, a tensor of shape (n_samples, n_features).
        y (torch.Tensor): The target labels, a tensor of shape (n_samples,).

    Note:
        This function supports both binary and multi-class classification.
    """
    
    # Move model and data to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Determine the minimum and maximum values for the plot boundaries
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # Create a mesh grid for the plot
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Convert the mesh grid to a PyTorch tensor
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Set the model to evaluation mode
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)  # Compute logits

    # Determine if the problem is multi-class or binary and convert logits to predictions
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # Multi-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # Binary

    # Reshape the predictions and plot the decision boundary
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
