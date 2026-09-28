import numpy as np


def generate_projection_matrix(
    r: int,
    N: int,
    matrix_type: str = "sparse_rademacher",
    q: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate an (r x N) validation projection matrix Psi.

    Parameters:
    -----------
    r : int
        Number of validation measurements (rows).
    N : int
        Signal dimension (columns).
    matrix_type : str
        Type of projection matrix:
        - "gaussian": Dense Gaussian N(0, 1/r).
        - "dense_rademacher": Entries +/- 1/sqrt(r) with probability 1/2 each.
        - "sparse_rademacher":
            0 with probability 1 - q,
            +1/sqrt(q * r) with probability q / 2,
            -1/sqrt(q * r) with probability q / 2.
            Note: For q=1.0, this is identical to dense_rademacher.
            Every entry has Var(Psi_ij) = 1/r, guaranteeing unbiased norm estimation.
    q : float
        Density parameter in (0, 1]. Expected fraction of non-zero entries.
    rng : np.random.Generator | None
        NumPy random generator.

    Returns:
    --------
    Psi : np.ndarray of shape (r, N)
    """
    if rng is None:
        rng = np.random.default_rng()

    if r <= 0 or N <= 0:
        raise ValueError(f"Invalid dimensions: r={r}, N={N}")

    if matrix_type == "gaussian":
        Psi = rng.normal(loc=0.0, scale=1.0 / np.sqrt(r), size=(r, N))

    elif matrix_type == "dense_rademacher":
        signs = rng.choice([-1.0, 1.0], size=(r, N))
        Psi = signs / np.sqrt(r)

    elif matrix_type == "sparse_rademacher":
        if not (0.0 < q <= 1.0):
            raise ValueError(f"Density parameter q must be in (0, 1], got {q}")

        # Choose nonzero locations with probability q
        mask = rng.uniform(0.0, 1.0, size=(r, N)) < q
        signs = rng.choice([-1.0, 1.0], size=(r, N))

        scale = 1.0 / np.sqrt(q * r)
        Psi = np.where(mask, signs * scale, 0.0)

    else:
        raise ValueError(f"Unknown projection matrix type: {matrix_type}")

    return Psi


def compute_projection_stats(Psi: np.ndarray, tol: float = 1e-12) -> dict:
    """
    Compute sparsity and variance statistics for a projection matrix.
    """
    r, N = Psi.shape
    total_elements = r * N
    nnz = int(np.sum(np.abs(Psi) > tol))
    empirical_density = nnz / total_elements
    empirical_var = float(np.var(Psi))

    return {
        "shape": (r, N),
        "total_elements": total_elements,
        "nnz": nnz,
        "empirical_density": empirical_density,
        "empirical_variance": empirical_var,
        "expected_variance": 1.0 / r,
    }


def budget_matched_dense_rows(r_sparse: int, q: float) -> int:
    """
    Calculate the number of rows for a dense matrix with approximately the same
    number of non-zero entries as an (r_sparse x N) matrix with density q.

    nnz(sparse) ≈ r_sparse * q * N
    nnz(dense)  ≈ r_dense * N
    => r_dense ≈ round(q * r_sparse)
    """
    return max(1, int(round(q * r_sparse)))
