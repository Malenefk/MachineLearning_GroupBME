def kfold_indices(n, k, shuffle=True, seed=None):
    """
    Return a list of k arrays, each the indices of one fold.
    If shuffle=True, indices are shuffled once before splitting.)
    """
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    return np.array_split(idx, k)
