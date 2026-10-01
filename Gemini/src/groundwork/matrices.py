"""
Unified Measurement Matrix Generation & Noise Engine.

Provides unified generation for both Sensing Matrices (Phi in R^{m x N})
and Validation Projection Matrices (Psi in R^{r x N}), guaranteeing consistent
mathematical representations across dense Gaussian, dense Rademacher, and
sparse Rademacher distributions.
"""

import numpy as np


def generate_measurement_matrix(
    rows: int,
    cols: int,
    dist: str = "sparse_rademacher",
    q: float = 1.0,
    normalize: str = "column",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Unified generator for compressed sensing measurement matrices (Phi or Psi).

    Parameters:
    -----------
    rows : int
        Number of measurements / rows (m for sensing, r for validation).
    cols : int
        Signal ambient dimension / columns (N).
    dist : str
        Distribution family:
        - "gaussian": Gaussian entries.
        - "rademacher" or "dense_rademacher": +/- 1 with equal probability.
        - "sparse_rademacher": Bernoulli-Rademacher with density q in (0, 1].
        - "sparse_gaussian": Bernoulli-Gaussian with density q in (0, 1].
    q : float
        Density parameter in (0, 1]. Expected fraction of non-zero entries.
        (For q=1.0, sparse_rademacher is identical to dense_rademacher).
    normalize : str
        Normalization mode:
        - "column": Normalize each column to have Euclidean norm 1 (||a_j||_2 = 1).
          Standard for sensing matrices Phi in OMP.
        - "variance": Scale matrix so entries have expected variance 1/rows (E[||a_j||_2^2] = 1).
          Guarantees E[||A x||_2^2] = ||x||_2^2. Standard for validation matrices Psi.
        - "none": Raw unscaled entries.
    rng : np.random.Generator | None
        NumPy random number generator for reproducibility.

    Returns:
    --------
    A : np.ndarray of shape (rows, cols)
    """
    if rng is None:
        rng = np.random.default_rng()

    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid dimensions: rows={rows}, cols={cols}")

    if not (0.0 < q <= 1.0):
        raise ValueError(f"Density parameter q must be in (0, 1], got {q}")

    dist_clean = dist.lower()

    # 1. Base random entry generation
    if dist_clean in ["gaussian", "dense_gaussian"]:
        A = rng.normal(loc=0.0, scale=1.0, size=(rows, cols))
        if q < 1.0:
            mask = rng.uniform(0.0, 1.0, size=(rows, cols)) < q
            A = np.where(mask, A, 0.0)

    elif dist_clean in ["rademacher", "dense_rademacher"]:
        signs = rng.choice([-1.0, 1.0], size=(rows, cols))
        if q < 1.0:
            mask = rng.uniform(0.0, 1.0, size=(rows, cols)) < q
            A = np.where(mask, signs, 0.0)
        else:
            A = signs

    elif dist_clean == "sparse_rademacher":
        signs = rng.choice([-1.0, 1.0], size=(rows, cols))
        if q < 1.0:
            mask = rng.uniform(0.0, 1.0, size=(rows, cols)) < q
            A = np.where(mask, signs, 0.0)
        else:
            A = signs

    elif dist_clean == "sparse_gaussian":
        gauss = rng.normal(loc=0.0, scale=1.0, size=(rows, cols))
        mask = rng.uniform(0.0, 1.0, size=(rows, cols)) < q
        A = np.where(mask, gauss, 0.0)

    else:
        raise ValueError(f"Unknown measurement matrix distribution: {dist}")

    # Guard: For column normalization (e.g. sensing matrix Phi), ensure no completely empty columns
    # so that division by column norm is well-defined.
    if normalize == "column":
        col_zero_mask = np.all(A == 0, axis=0)
        if np.any(col_zero_mask):
            empty_indices = np.where(col_zero_mask)[0]
            for c in empty_indices:
                rand_row = rng.integers(0, rows)
                A[rand_row, c] = rng.choice([-1.0, 1.0])

    # 2. Normalization
    if normalize == "column":
        col_norms = np.linalg.norm(A, axis=0)
        col_norms = np.where(col_norms == 0, 1.0, col_norms)
        A = A / col_norms

    elif normalize == "variance":
        # Scale so that Var(A_ij) = 1/rows, guaranteeing unbiased norm estimation
        scale = 1.0 / np.sqrt(q * rows)
        A = A * scale

    elif normalize == "none":
        pass

    else:
        raise ValueError(f"Unknown normalization mode: {normalize}")

    return A


def generate_sensing_matrix(
    m: int,
    N: int,
    dist: str = "gaussian",
    normalize: str = "column",
    q: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate an (m x N) sensing matrix Phi (backward-compatible wrapper).
    """
    return generate_measurement_matrix(
        rows=m, cols=N, dist=dist, q=q, normalize=normalize, rng=rng
    )


def generate_projection_matrix(
    r: int,
    N: int,
    matrix_type: str = "sparse_rademacher",
    q: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate an (r x N) validation projection matrix Psi (backward-compatible wrapper).
    """
    return generate_measurement_matrix(
        rows=r, cols=N, dist=matrix_type, q=q, normalize="variance", rng=rng
    )


def compute_matrix_stats(A: np.ndarray, tol: float = 1e-12) -> dict:
    """
    Compute sparsity, non-zeros, and variance statistics for a measurement matrix.
    """
    rows, cols = A.shape
    total_elements = rows * cols
    nnz = int(np.sum(np.abs(A) > tol))
    empirical_density = nnz / total_elements
    empirical_var = float(np.var(A))

    return {
        "shape": (rows, cols),
        "total_elements": total_elements,
        "nnz": nnz,
        "empirical_density": empirical_density,
        "empirical_variance": empirical_var,
        "expected_variance": 1.0 / rows,
    }


def compute_projection_stats(Psi: np.ndarray, tol: float = 1e-12) -> dict:
    """Backward-compatible alias for compute_matrix_stats."""
    return compute_matrix_stats(Psi, tol=tol)


def budget_matched_dense_rows(r_sparse: int, q: float) -> int:
    """
    Calculate the number of rows for a dense matrix with approximately the same
    number of non-zero entries as an (r_sparse x N) matrix with density q.
    """
    return max(1, int(round(q * r_sparse)))


def add_noise(
    y: np.ndarray,
    sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add zero-mean Gaussian noise with standard deviation sigma to measurement vector y.
    """
    if rng is None:
        rng = np.random.default_rng()

    if sigma <= 0.0:
        noise = np.zeros_like(y)
    else:
        noise = rng.normal(loc=0.0, scale=sigma, size=y.shape)

    return y + noise, noise


def add_noise_snr(
    y: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Add AWGN to signal y to achieve a target Signal-to-Noise Ratio (SNR) in decibels.
    SNR_dB = 10 * log10( ||y||_2^2 / ||noise||_2^2 )
    """
    signal_power = np.mean(y ** 2)
    if signal_power <= 0:
        sigma = 1e-6
    else:
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        sigma = np.sqrt(noise_power)

    return add_noise(y, sigma, rng=rng) + (sigma,)
