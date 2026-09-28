import numpy as np
from scipy import stats


def compute_cv_errors(
    Psi: np.ndarray,
    y_cv: np.ndarray,
    iterates: np.ndarray,
) -> np.ndarray:
    """
    Compute cross-validation squared error estimates E_hat_j = ||y_cv - Psi @ x^j||_2^2
    for each iterate x^j in the trajectory.

    Parameters:
    -----------
    Psi : np.ndarray of shape (r, N)
        Validation projection matrix.
    y_cv : np.ndarray of shape (r,)
        Validation measurements.
    iterates : np.ndarray of shape (p + 1, N)
        Trajectory of iterates.

    Returns:
    --------
    E_hat : np.ndarray of shape (p + 1,)
    """
    # Vectorized computation: iterates @ Psi.T is shape (p + 1, r)
    projected_iterates = iterates @ Psi.T  # (p + 1, r)
    residuals = y_cv - projected_iterates  # (p + 1, r)
    E_hat = np.sum(residuals ** 2, axis=1)  # (p + 1,)
    return E_hat


def relative_norm_errors(
    E_true: np.ndarray,
    E_hat: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute relative error |E_hat_j - E_j| / E_j for each iterate.
    """
    return np.abs(E_hat - E_true) / np.maximum(E_true, eps)


def error_ratios(
    E_true: np.ndarray,
    E_hat: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute ratio E_hat_j / E_j for each iterate.
    """
    return E_hat / np.maximum(E_true, eps)


def kendall_tau(
    E_true: np.ndarray,
    E_hat: np.ndarray,
) -> float:
    """
    Compute Kendall's rank correlation coefficient tau between true and estimated errors.
    Returns value in [-1, 1] (or 0.0 if degenerate).
    """
    if len(E_true) < 2:
        return 1.0
    res = stats.kendalltau(E_true, E_hat)
    tau = float(res.statistic) if not np.isnan(res.statistic) else 0.0
    return tau


def spearman_rho(
    E_true: np.ndarray,
    E_hat: np.ndarray,
) -> float:
    """
    Compute Spearman's rank correlation coefficient rho between true and estimated errors.
    """
    if len(E_true) < 2:
        return 1.0
    res = stats.spearmanr(E_true, E_hat)
    rho = float(res.statistic) if not np.isnan(res.statistic) else 0.0
    return rho


def pairwise_concordance(
    E_true: np.ndarray,
    E_hat: np.ndarray,
) -> float:
    """
    Compute the fraction of correctly ordered pairs:
    For every pair i < j:
        Concordant if (E_i < E_j and E_hat_i < E_hat_j) or (E_i > E_j and E_hat_i > E_hat_j).
    """
    n = len(E_true)
    if n < 2:
        return 1.0

    concordant = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            d_true = E_true[i] - E_true[j]
            d_hat = E_hat[i] - E_hat[j]

            # If true errors are not strictly identical
            if d_true != 0:
                total_pairs += 1
                if (d_true * d_hat) > 0:
                    concordant += 1

    return float(concordant / total_pairs) if total_pairs > 0 else 1.0


def best_iterate_selection(
    E_true: np.ndarray,
    E_hat: np.ndarray,
    eps: float = 1e-12,
) -> dict:
    """
    Evaluate best-iterate selection metrics.

    Returns:
    --------
    dict containing:
        - "j_true_star": index of true optimal iterate argmin(E_true)
        - "j_hat_star": index of estimated optimal iterate argmin(E_hat)
        - "is_exact_match": bool indicating if j_hat_star == j_true_star
        - "excess_error": (E_true[j_hat_star] - E_true[j_true_star]) / max(E_true[j_true_star], eps)
    """
    j_true_star = int(np.argmin(E_true))
    j_hat_star = int(np.argmin(E_hat))

    true_min_error = float(E_true[j_true_star])
    selected_true_error = float(E_true[j_hat_star])

    excess_error = (selected_true_error - true_min_error) / max(true_min_error, eps)

    return {
        "j_true_star": j_true_star,
        "j_hat_star": j_hat_star,
        "is_exact_match": bool(j_true_star == j_hat_star),
        "excess_error": float(excess_error),
    }


def jl_distortion_violation_rate(
    error_vectors: np.ndarray,
    Psi: np.ndarray,
    epsilon: float = 0.2,
) -> float:
    """
    Measure empirical JL distortion violation rate over a collection of error vectors.

    For each vector e, check whether:
        (1 - epsilon) * ||e||_2^2 <= ||Psi @ e||_2^2 <= (1 + epsilon) * ||e||_2^2

    Returns the fraction of vectors where this condition FAILS.
    """
    if len(error_vectors) == 0:
        return 0.0

    true_sq_norms = np.sum(error_vectors ** 2, axis=1)  # (K,)
    # Filter out zero-norm vectors
    nonzero_mask = true_sq_norms > 1e-12
    if not np.any(nonzero_mask):
        return 0.0

    error_vectors = error_vectors[nonzero_mask]
    true_sq_norms = true_sq_norms[nonzero_mask]

    projected = error_vectors @ Psi.T  # (K, r)
    proj_sq_norms = np.sum(projected ** 2, axis=1)  # (K,)

    ratios = proj_sq_norms / true_sq_norms
    violations = np.sum((ratios < (1.0 - epsilon)) | (ratios > (1.0 + epsilon)))

    return float(violations / len(error_vectors))
