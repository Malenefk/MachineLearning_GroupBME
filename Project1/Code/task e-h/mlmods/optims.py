import numpy as np

def gd(grad, th0, lr=1e-2, iters=1000):
    """
    Plain full batch gradient descent.
    This subtracts a learning rate times the gradient at each step.
    """
    th = th0.copy(); traj = [th.copy()]
    for _ in range(iters):
        th -= lr * grad(th)
        traj.append(th.copy())
    return th, traj
    
def gd_momentum(grad, th0, lr=1e-2, iters=1000, beta=0.9):
    """
    Gradient descent with momentum in the classic style.
    v stores a running update that mixes the previous update with the current
    gradient. lr is included inside v. The parameter is moved by subtracting v.
    """
    th = th0.copy(); v = np.zeros_like(th); traj = [th.copy()]
    for _ in range(iters):
        g = grad(th)
        v = beta * v + lr * g
        th -= v
        traj.append(th.copy())
    return th, traj

def adagrad(grad, th0, lr=1e-2, iters=1000, eps=1e-8):
    """
    AdaGrad keeps a running sum of squared gradients and shrinks the step per 
    coordinate.
    The epsilon value avoids division by zero.
    """
    th = th0.copy(); G = np.zeros_like(th); traj = [th.copy()]
    for _ in range(iters):
        g = grad(th)
        G += g*g
        th -= (lr/(np.sqrt(G) + eps)) * g
        traj.append(th.copy())
    return th, traj

def rmsprop(grad, th0, lr=1e-3, iters=1000, beta=0.9, eps=1e-8):
    """
    RMSprop keeps an exponential moving average of squared gradients.
    The step is the gradient divided by the root of that average.
    """
    th = th0.copy(); S = np.zeros_like(th); traj = [th.copy()]
    for _ in range(iters):
        g = grad(th)
        S = beta*S + (1.0 - beta)*(g*g)
        th -= lr * g / (np.sqrt(S) + eps)
        traj.append(th.copy())
    return th, traj

def adam(grad, th0, lr=1e-3, iters=1000, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam keeps two moving averages.
    m tracks the average gradient.
    v tracks the average squared gradient.
    """
    th = th0.copy(); m = np.zeros_like(th); v = np.zeros_like(th); traj = [th.copy()]
    for t in range(1, iters+1):
        g = grad(th)
        m = beta1*m + (1.0 - beta1)*g
        v = beta2*v + (1.0 - beta2)*(g*g)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        th -= lr * m_hat / (np.sqrt(v_hat) + eps)
        traj.append(th.copy())
    return th, traj
    
# SGD

def _iterate_minibatches(X, y, batch_size, seed):
    """
    Yields batches of rows from X and y.
    The order is shuffled once per epoch using the seed for repeatability.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    for start in range(0, n, batch_size):
        bi = idx[start:start+batch_size]
        yield X[bi], y[bi]

def _sgd_core(grad_mb, th0, X, y, lr, epochs, batch_size, seed, update_rule):
    """  
    Core training loop for SGD and its variants.
    grad_mb is a function that returns a gradient for a mini batch.
    update_rule decides how to use that gradient to change the parameters.
    We record one point per epoch for plotting.
    """
    th = th0.copy()
    traj = [th.copy()]  # snapshot once per epoch
    state = {}
    for ep in range(epochs):
        for Xb, yb in _iterate_minibatches(X, y, batch_size, seed + ep):
            th = update_rule(th, Xb, yb, grad_mb, lr, state)
        traj.append(th.copy())
    return th, traj

def _upd_sgd(th, Xb, yb, grad_mb, lr, state):
    """Plain SGD update using the mini batch gradient and a learning rate."""
    g = grad_mb(th, Xb, yb); return th - lr*g

def _upd_mom(th, Xb, yb, grad_mb, lr, state):
    """
    SGD with momentum using the same style as full batch momentum.
    v stores the running update. The parameter is moved by subtracting v.
    """
    v = state.setdefault("v", np.zeros_like(th))
    beta = 0.9
    g = grad_mb(th, Xb, yb)
    v[:] = beta * v + lr * g
    return th - v

def _upd_ada(th, Xb, yb, grad_mb, lr, state):
    """
    AdaGrad style SGD. Keeps a running sum of squared gradients per coordinate.
    """
    if "G" not in state: state["G"] = np.zeros_like(th)
    eps = 1e-8
    g = grad_mb(th, Xb, yb)
    state["G"] += g*g
    return th - (lr/(np.sqrt(state["G"])+eps)) * g

def _upd_rms(th, Xb, yb, grad_mb, lr, state):
    """
    RMSprop style SGD. Keeps a moving average of squared gradients.
    """
    if "S" not in state: state["S"] = np.zeros_like(th)
    beta, eps = 0.9, 1e-8
    g = grad_mb(th, Xb, yb)
    state["S"] = beta*state["S"] + (1.0 - beta)*(g*g)
    return th - lr * g / (np.sqrt(state["S"]) + eps)

def _upd_adam(th, Xb, yb, grad_mb, lr, state):
    """
    Adam style SGD with moving averages.
    """
    if "m" not in state: state["m"] = np.zeros_like(th)
    if "v" not in state: state["v"] = np.zeros_like(th)
    if "t" not in state: state["t"] = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    g = grad_mb(th, Xb, yb)
    state["t"] += 1
    state["m"] = beta1*state["m"] + (1.0 - beta1)*g
    state["v"] = beta2*state["v"] + (1.0 - beta2)*(g*g)
    m_hat = state["m"] / (1.0 - beta1**state["t"])
    v_hat = state["v"] / (1.0 - beta2**state["t"])
    return th - lr * m_hat / (np.sqrt(v_hat) + eps)
    


def sgd(grad_mb, th0, X, y, lr, epochs, batch_size, seed=0):
    return _sgd_core(grad_mb, th0, X, y, lr, epochs, batch_size, seed, _upd_sgd)

def sgd_momentum(grad_mb, th0, X, y, lr, epochs, batch_size, seed=0):
    return _sgd_core(grad_mb, th0, X, y, lr, epochs, batch_size, seed, _upd_mom)

def sgd_adagrad(grad_mb, th0, X, y, lr, epochs, batch_size, seed=0):
    return _sgd_core(grad_mb, th0, X, y, lr, epochs, batch_size, seed, _upd_ada)

def sgd_rmsprop(grad_mb, th0, X, y, lr, epochs, batch_size, seed=0):
    return _sgd_core(grad_mb, th0, X, y, lr, epochs, batch_size, seed, _upd_rms)

def sgd_adam(grad_mb, th0, X, y, lr, epochs, batch_size, seed=0):
    return _sgd_core(grad_mb, th0, X, y, lr, epochs, batch_size, seed, _upd_adam)
