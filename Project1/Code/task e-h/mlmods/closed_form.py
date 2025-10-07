import numpy as np

def ols_closed_form(X, y):
    """Closed-form OLS: pinv(X) @ y (robust to rank deficiency)."""
    return np.linalg.pinv(X) @ y

def ridge_closed_form(X, y, lam):
    """Closed-form Ridge with (intercept unpenalized via D[0,0]=0)."""
    n, p = X.shape
    D = np.eye(p); D[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + n * lam * D, X.T @ y)
