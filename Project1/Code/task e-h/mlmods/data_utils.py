import numpy as np
from sklearn.model_selection import train_test_split

def runge(x):
    """Runge function: 1 / (1 + 25 x^2)."""
    x = np.asarray(x)
    return 1.0 / (1.0 + 25.0 * x**2)

def polynomial_features(x, degree, intercept=False):
    """
    Build polynomial features with degrees 1 to degree.
    If intercept=True, adds a column of ones.
    """
    x = np.asarray(x).reshape(-1, 1)
    X = np.hstack([x ** p for p in range(1, degree + 1)]) if degree > 0 else np.empty((x.shape[0], 0))
    if intercept:
        X = np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])
    return X

def add_intercept(X):
    """Adds a column of ones at the front of the design matrix."""
    X = np.asarray(X, float)
    return np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])

def standardize_train_test(
    X_tr, X_te, y_tr=None, y_te=None,
    mode="zscore",
    center_y=False,
    eps=1e-12,
):
    """

    Standardizes or centers features using only statistics from the 
    training data.

    mode can be zscore or center or none.
    zscore subtracts the mean and divides by the standard deviation.
    center only subtracts the mean.
    none keeps features as they are.
    If y values are given and center_y is true, y is treated the 
    same way as the chosen mode.
    If center_y is false, y is returned unchanged.
    Returns either four or five values depending on whether y is 
    provided.
    When y is missing it returns transformed X train, transformed X 
    test, mean of X train, scale of X train.
    When y is present it returns transformed X train, transformed X 
    test, transformed y train, transformed y test, a small stats 
    dictionary.

    """
    X_tr = np.asarray(X_tr, float)
    X_te = None if X_te is None else np.asarray(X_te, float)

    X_mu = X_tr.mean(axis=0, keepdims=True)
    if mode == "zscore":
        X_sd = X_tr.std(axis=0, keepdims=True) + eps
        X_tr_t = (X_tr - X_mu) / X_sd
        X_te_t = None if X_te is None else (X_te - X_mu) / X_sd
    elif mode == "center":
        X_sd = np.ones_like(X_mu)
        X_tr_t = X_tr - X_mu
        X_te_t = None if X_te is None else (X_te - X_mu)
    elif mode == "none":
        X_sd = np.ones_like(X_mu)
        X_tr_t, X_te_t = X_tr, X_te
    else:
        raise ValueError(f"Unknown mode: (expected 'zscore', 'center', or 'none')")

    if y_tr is None:
        return X_tr_t, X_te_t, X_mu, X_sd

    y_tr = np.asarray(y_tr, float)
    y_te = None if y_te is None else np.asarray(y_te, float)

    if not center_y:
        y_mu, y_sd = 0.0, 1.0
        y_tr_t, y_te_t = y_tr, y_te
    else:
        y_mu = float(y_tr.mean())
        if mode == "zscore":
            y_sd = float(y_tr.std() + eps)
            y_tr_t = (y_tr - y_mu) / y_sd
            y_te_t = None if y_te is None else (y_te - y_mu) / y_sd
        else:
            y_sd = 1.0
            y_tr_t = y_tr - y_mu
            y_te_t = None if y_te is None else (y_te - y_mu)

    stats = {"mode": mode, "X_mu": X_mu, "X_sd": X_sd, "y_mu": y_mu, "y_sd": y_sd}
    return X_tr_t, X_te_t, y_tr_t, y_te_t, stats

def make_dataset(
    N, degree, seed=3155, noise_std=1.0,
    split=0.2, x_sampling="linspace"
):
    """
    Returns raw train and test splits only.
    No intercept added and no scaling performed.

    Returns:
        X_tr_raw, X_te_raw, y_tr, y_te
        X_* are polynomial features without intercept
        y_* are column vectors
    """
    rng = np.random.default_rng(seed)
    if x_sampling == "uniform":
        x = rng.uniform(-1.0, 1.0, size=N)
    else:
        x = np.linspace(-1.0, 1.0, N)

    y = runge(x) + rng.normal(0.0, noise_std, size=N)
    X = polynomial_features(x, degree, intercept=False)

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=split, shuffle=True, random_state=seed
    )

    return X_tr_raw, X_te_raw, y_tr.reshape(-1, 1), y_te.reshape(-1, 1)

def prepare_design_from_indices(x, y, degree, tr_idx, te_idx, mode="center", center_y=False):
    """
    Builds features for a given train and test split given by index 
    arrays.
    We build features without intercept first.
    We standardize using only training rows.
    We add an intercept at the end.
    Returns X train, X test, y train, y test.
    """
    X_all = polynomial_features(x, degree, intercept=False)
    X_tr_raw, X_te_raw = X_all[tr_idx], X_all[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    X_tr_t, X_te_t, _, _ = standardize_train_test(X_tr_raw, X_te_raw, mode=mode)
    X_tr = add_intercept(X_tr_t)
    X_te = add_intercept(X_te_t)
    return X_tr, X_te, y_tr, y_te

