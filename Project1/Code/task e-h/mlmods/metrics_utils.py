import numpy as np

def mse(y, yhat):
    """Plain mean squared error."""
    y = np.asarray(y).reshape(-1,1)
    yhat = np.asarray(yhat).reshape(-1,1)
    return float(np.mean((y - yhat)**2))

def train_mse_over_traj(X, y, traj, mse_fn):
    """
    Compute plain train MSE along a parameter trajectory list.
    """
    return [mse_fn(y, X @ th) for th in traj]
