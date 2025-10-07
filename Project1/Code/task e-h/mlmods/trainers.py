# trainers.py
import numpy as np
from .optims import gd, gd_momentum, adagrad, rmsprop, adam
from .optims import sgd as sgd_plain, sgd_momentum as sgd_mom
from .optims import sgd_adagrad as sgd_ada, sgd_rmsprop as sgd_rms, sgd_adam as sgd_adm

def train_full_batch_family(grad_builder, X, y, lr, iters, theta0=None):
    """
    Train GD family (GD, Momentum, AdaGrad, Adam, RMSprop) with the same base lr.
    Returns dict name (theta_final, trajectory)
    """
    if theta0 is None:
        theta0 = np.zeros((X.shape[1], 1))
    grad = grad_builder
    out = {}
    out["GD"]       = gd(grad, theta0, lr=lr, iters=iters)
    out["Momentum"] = gd_momentum(grad, theta0, lr=lr, iters=iters)
    out["AdaGrad"]  = adagrad(grad, theta0, lr=lr, iters=iters)
    out["Adam"]     = adam(grad, theta0, lr=lr, iters=iters)
    out["RMSprop"]  = rmsprop(grad, theta0, lr=lr, iters=iters)
    return out

def train_sgd_family(grad_mb, X, y, lr, epochs, batch_size, theta0=None, seed=3155):
    """
    Train SGD family (SGD, Momentum, AdaGrad, Adam, RMSprop) with the SAME base lr.
    Snapshot trajectory once per epoch.
    """
    if theta0 is None:
        theta0 = np.zeros((X.shape[1], 1))
    out = {}
    out["SGD"]      = sgd_plain(grad_mb, theta0, X, y, lr=lr, epochs=epochs, batch_size=batch_size, seed=seed)
    out["Momentum"] = sgd_mom(grad_mb,   theta0, X, y, lr=lr, epochs=epochs, batch_size=batch_size, seed=seed)
    out["AdaGrad"]  = sgd_ada(grad_mb,   theta0, X, y, lr=lr, epochs=epochs, batch_size=batch_size, seed=seed)
    out["Adam"]     = sgd_adm(grad_mb,   theta0, X, y, lr=lr, epochs=epochs, batch_size=batch_size, seed=seed)
    out["RMSprop"]  = sgd_rms(grad_mb,   theta0, X, y, lr=lr, epochs=epochs, batch_size=batch_size, seed=seed)
    return out
