import numpy as np

def ols_grad_full_builder(X, y):
    """
    Full-batch gradient for OLS loss. 
    """
    n = X.shape[0]
    def grad(theta):
        return (2.0/n) * (X.T @ (X @ theta - y))
    return grad

def ols_grad_minibatch(theta, Xb, yb):
    """Mini-batch OLS gradient with m = batch."""
    m = Xb.shape[0]
    return (2.0/m) * (Xb.T @ (Xb @ theta - yb))

def ridge_grad_full_builder(X, y, lam, intercept_free=True):
    """
    Full batch gradient for Ridge loss.
    If intercept_free is true the first coefficient is not penalized.
    """
    n = X.shape[0]
    def grad(theta):
        g = (2.0/n) * (X.T @ (X @ theta - y))
        if lam != 0.0:
            g_reg = 2.0 * lam * theta
            if intercept_free:
                g_reg[0] = 0.0
            g = g + g_reg
        return g
    return grad

def ridge_grad_minibatch(theta, Xb, yb, lam, intercept_free=True):
    """Mini-batch Ridge gradient"""
    m = Xb.shape[0]
    g = (2.0/m) * (Xb.T @ (Xb @ theta - yb))
    if lam != 0.0:
        g_reg = 2.0 * lam * theta
        if intercept_free:
            g_reg[0] = 0.0
        g = g + g_reg
    return g

def lasso_grad_full_builder(X, y, lam, intercept_free=True):
    """
    Full batch subgradient for LASSO loss.
    Uses the sign of the parameters as the subgradient for the L1 part.
    If intercept_free is true the first coefficient is not penalized.
    """
    n = X.shape[0]
    def grad(theta):
        g = (2.0/n) * (X.T @ (X @ theta - y))
        if lam != 0.0:
            s = np.sign(theta)
            if intercept_free:
                s[0] = 0.0
            g = g + lam * s
        return g
    return grad

def lasso_grad_minibatch(theta, Xb, yb, lam, intercept_free=True):
    """Mini-batch LASSO subgradient."""
    m = Xb.shape[0]
    g = (2.0/m) * (Xb.T @ (Xb @ theta - yb))
    if lam != 0.0:
        s = np.sign(theta)
        if intercept_free:
            s[0] = 0.0
        g = g + lam * s
    return g
