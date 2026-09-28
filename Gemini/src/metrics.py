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


def jl_isometry_stats(
    error_vectors: np.ndarray,
    Psi: np.ndarray,
    epsilons: tuple[float, ...] = (0.2, 0.5),
) -> dict:
    """
    Compute classical Johnson-Lindenstrauss distortion statistics on a set of vectors.

    Returns:
    --------
    dict containing:
        - "mean_ratio": mean of ||Psi @ e||_2^2 / ||e||_2^2
        - "std_ratio": standard deviation of ratio (measures spread / tail risk)
        - "success_rates": dict mapping epsilon to success rate (fraction within 1 +/- epsilon)
    """
    if len(error_vectors) == 0:
        return {"mean_ratio": 1.0, "std_ratio": 0.0, "success_rates": {eps: 1.0 for eps in epsilons}}

    true_sq_norms = np.sum(error_vectors ** 2, axis=1)
    nonzero_mask = true_sq_norms > 1e-12
    if not np.any(nonzero_mask):
        return {"mean_ratio": 1.0, "std_ratio": 0.0, "success_rates": {eps: 1.0 for eps in epsilons}}

    error_vectors = error_vectors[nonzero_mask]
    true_sq_norms = true_sq_norms[nonzero_mask]

    projected = error_vectors @ Psi.T
    proj_sq_norms = np.sum(projected ** 2, axis=1)
    ratios = proj_sq_norms / true_sq_norms

    success_rates = {}
    for eps in epsilons:
        in_bounds = (ratios >= (1.0 - eps)) & (ratios <= (1.0 + eps))
        success_rates[eps] = float(np.mean(in_bounds))

    return {
        "mean_ratio": float(np.mean(ratios)),
        "std_ratio": float(np.std(ratios)),
        "median_ratio": float(np.median(ratios)),
        "success_rates": success_rates,
    }


def noise_corrected_cv_errors(
    Psi: np.ndarray,
    y_cv: np.ndarray,
    iterates: np.ndarray,
    sigma_noise: float = 0.0,
) -> np.ndarray:
    """
    Compute noise-corrected CV error estimates:
    E_hat_corrected_j = max(0, E_hat_j - sigma_noise^2)
    This removes the additive noise floor bias at late iterations.
    """
    raw_errors = compute_cv_errors(Psi, y_cv, iterates)
    corrected = np.maximum(0.0, raw_errors - (sigma_noise ** 2))
    return corrected


def pre_convergence_relative_errors(
    E_true: np.ndarray,
    E_hat: np.ndarray,
    j_star: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute relative error restricted to the pre-convergence regime (j <= j_star),
    where true signal error strictly dominates the noise floor.
    """
    j_cutoff = max(1, min(j_star + 1, len(E_true)))
    return relative_norm_errors(E_true[:j_cutoff], E_hat[:j_cutoff], eps=eps)


def ward_bound_stats(
    E_true: np.ndarray,
    E_hat: np.ndarray,
    eps: float = 1e-12,
    lower_factor: float = 4.0 / 5.0,
    upper_factor: float = 4.0 / 3.0,
) -> dict:
    """
    Evaluate Rachel Ward's (2009) compressed sensing cross-validation theoretical bounds:
    For any iterate j, the true error norm is bounded by:
        lower_factor * ||Psi e^j||_2 <= ||e^j||_2 <= upper_factor * ||Psi e^j||_2
    where lower_factor = 4/5 (0.80) and upper_factor = 4/3 (~1.333).

    In terms of estimated and true squared errors (E_hat_j and E_j):
        (3/4)^2 * E_j = 0.5625 * E_j <= E_hat_j <= 1.5625 * E_j = (5/4)^2 * E_j.

    Parameters:
    -----------
    E_true : np.ndarray of shape (p + 1,)
        True squared reconstruction errors ||e^j||_2^2.
    E_hat : np.ndarray of shape (p + 1,)
        Estimated squared errors ||Psi e^j||_2^2.
    eps : float
        Numerical zero threshold to ignore near-zero errors.

    Returns:
    --------
    dict containing:
        - "ward_simultaneous_success": bool (True if bound holds across ALL non-trivial iterates)
        - "ward_fraction_satisfied": float (fraction of iterates satisfying the bound)
        - "per_iterate_satisfied": list[bool]
        - "norm_ratios": np.ndarray (||Psi e^j||_2 / ||e^j||_2)
    """
    # Non-trivial iterates where true error is positive
    valid_mask = E_true > eps
    if not np.any(valid_mask):
        return {
            "ward_simultaneous_success": True,
            "ward_fraction_satisfied": 1.0,
            "per_iterate_satisfied": [],
            "norm_ratios": np.array([]),
        }

    e_true_norms = np.sqrt(E_true[valid_mask])
    e_hat_norms = np.sqrt(np.maximum(0.0, E_hat[valid_mask]))

    # Ward's exact condition: (4/5) * ||Psi e^j|| <= ||e^j|| <= (4/3) * ||Psi e^j||
    in_ward_bounds = (e_true_norms >= lower_factor * e_hat_norms) & (
        e_true_norms <= upper_factor * e_hat_norms
    )

    norm_ratios = e_hat_norms / np.maximum(e_true_norms, eps)

    return {
        "ward_simultaneous_success": bool(np.all(in_ward_bounds)),
        "ward_fraction_satisfied": float(np.mean(in_ward_bounds)),
        "per_iterate_satisfied": in_ward_bounds.tolist(),
        "norm_ratios": norm_ratios,
    }

