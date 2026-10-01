import numpy as np


def generate_sparse_signal(
    N: int,
    s: int,
    coeff_dist: str = "gaussian",
    rng: np.random.Generator | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate an s-sparse signal x of length N with support chosen uniformly at random.

    Parameters:
    -----------
    N : int
        Ambient dimension of the signal.
    s : int
        Sparsity level (number of non-zero entries).
    coeff_dist : str
        Distribution for non-zero coefficients:
        - "gaussian": N(loc, scale) with realistic default loc=20.0, scale=6.0 (or specified via kwargs).
        - "rademacher": Uniform choice of +1 or -1.
        - "uniform": Uniform in [-b, -a] U [a, b] (default a=0.5, b=2.0).
        - "decay": Power-law magnitude decay k^(-alpha) with random signs.
    rng : np.random.Generator | None
        NumPy random generator for reproducibility.

    Returns:
    --------
    x : np.ndarray
        Dense vector of length N containing s non-zero entries.
    support : np.ndarray
        Sorted 1D array containing indices of the non-zero entries.
    """
    if rng is None:
        rng = np.random.default_rng()

    if s > N:
        raise ValueError(f"Sparsity s={s} cannot exceed signal dimension N={N}")

    support = np.sort(rng.choice(N, size=s, replace=False))
    x = np.zeros(N, dtype=float)

    if coeff_dist == "gaussian":
        loc = kwargs.get("loc", 20.0)
        scale = kwargs.get("scale", 6.0)
        values = rng.normal(loc=loc, scale=scale, size=s)
    elif coeff_dist == "rademacher":
        values = rng.choice([-1.0, 1.0], size=s)
    elif coeff_dist == "uniform":
        a = kwargs.get("a", 0.5)
        b = kwargs.get("b", 2.0)
        mags = rng.uniform(a, b, size=s)
        signs = rng.choice([-1.0, 1.0], size=s)
        values = mags * signs
    elif coeff_dist == "decay":
        alpha = kwargs.get("alpha", 1.0)
        ranks = np.arange(1, s + 1)
        mags = ranks ** (-alpha)
        signs = rng.choice([-1.0, 1.0], size=s)
        values = mags * signs
    else:
        raise ValueError(f"Unknown coefficient distribution: {coeff_dist}")

    x[support] = values
    return x, support


def sparsity_of(v: np.ndarray, tol: float = 1e-8) -> int:
    """Return the number of entries with absolute value greater than tol."""
    return int(np.sum(np.abs(v) > tol))
