from dataclasses import dataclass
import numpy as np


@dataclass
class OMPTrajectory:
    """
    Holds the complete sequence of iterates and diagnostics produced by OMP.

    Attributes:
    -----------
    iterates : np.ndarray
        Array of shape (p + 1, N) where row j corresponds to iterate x^j.
    supports : list[list[int]]
        List of length p + 1 containing active support indices at each iteration.
    residuals : np.ndarray
        Array of shape (p + 1, m) containing residual vectors r^j = y - Phi @ x^j.
    true_errors : np.ndarray | None
        Array of shape (p + 1,) containing true squared errors E_j = ||x - x^j||_2^2.
    error_sparsities : np.ndarray | None
        Array of shape (p + 1,) containing ||x - x^j||_0 for each iteration.
    p : int
        Total number of iterations performed.
    """
    iterates: np.ndarray
    supports: list[list[int]]
    residuals: np.ndarray
    true_errors: np.ndarray | None
    error_sparsities: np.ndarray | None
    p: int


def run_omp_trajectory(
    Phi: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    x_true: np.ndarray | None = None,
    tol: float = 1e-12,
) -> OMPTrajectory:
    """
    Run Orthogonal Matching Pursuit (OMP) and record the complete trajectory of iterates.

    Parameters:
    -----------
    Phi : np.ndarray of shape (m, N)
        Sensing / measurement matrix.
    y : np.ndarray of shape (m,)
        Measurement vector.
    max_iter : int
        Maximum number of iterations (p).
    x_true : np.ndarray | None
        Ground truth sparse vector (if available) for computing true errors E_j.
    tol : float
        Tolerance for residual norm to terminate early if residual drops to near zero.

    Returns:
    --------
    trajectory : OMPTrajectory
    """
    m, N = Phi.shape
    p = min(max_iter, m, N)

    iterates = np.zeros((p + 1, N), dtype=float)
    residuals = np.zeros((p + 1, m), dtype=float)
    supports: list[list[int]] = [[]]

    # Iteration 0: x^0 = 0, residual = y
    residuals[0] = y.copy()

    true_errors = np.zeros(p + 1, dtype=float) if x_true is not None else None
    error_sparsities = np.zeros(p + 1, dtype=int) if x_true is not None else None

    if x_true is not None:
        e0 = x_true - iterates[0]
        true_errors[0] = float(np.sum(e0 ** 2))
        error_sparsities[0] = int(np.sum(np.abs(e0) > 1e-8))

    current_support: list[int] = []
    support_set: set[int] = set()
    current_residual = y.copy()

    actual_steps = p
    for j in range(1, p + 1):
        if np.linalg.norm(current_residual) < tol:
            actual_steps = j - 1
            break

        # Step 1: Compute correlations with residual for columns not yet chosen
        correlations = Phi.T @ current_residual
        if support_set:
            # Mask already selected columns so we don't pick them again
            mask = np.zeros(N, dtype=bool)
            mask[list(support_set)] = True
            correlations[mask] = 0.0

        best_index = int(np.argmax(np.abs(correlations)))
        current_support.append(best_index)
        support_set.add(best_index)

        # Step 2: Least squares over active support
        Phi_support = Phi[:, current_support]
        x_support, _, _, _ = np.linalg.lstsq(Phi_support, y, rcond=None)

        # Step 3: Form full iterate vector
        x_j = np.zeros(N, dtype=float)
        x_j[current_support] = x_support

        # Step 4: Update residual
        current_residual = y - Phi_support @ x_support

        # Record state
        iterates[j] = x_j
        residuals[j] = current_residual
        supports.append(list(current_support))

        if x_true is not None:
            e_j = x_true - x_j
            true_errors[j] = float(np.sum(e_j ** 2))
            error_sparsities[j] = int(np.sum(np.abs(e_j) > 1e-8))

    # Slice arrays if terminated early
    if actual_steps < p:
        iterates = iterates[: actual_steps + 1]
        residuals = residuals[: actual_steps + 1]
        supports = supports[: actual_steps + 1]
        if true_errors is not None:
            true_errors = true_errors[: actual_steps + 1]
            error_sparsities = error_sparsities[: actual_steps + 1]

    return OMPTrajectory(
        iterates=iterates,
        supports=supports,
        residuals=residuals,
        true_errors=true_errors,
        error_sparsities=error_sparsities,
        p=actual_steps,
    )
