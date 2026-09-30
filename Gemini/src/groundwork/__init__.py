"""
Groundwork package for Compressed Sensing Cross-Validation.
Contains fundamental mathematical building blocks:
- signals: sparse signal generation
- matrices: unified measurement matrix generation (sensing Phi & validation Psi) and noise models
- omp: Orthogonal Matching Pursuit trajectory tracking
- metrics: validation metrics, Kendall's tau, distortion variance, Ward bounds
"""

from .signals import generate_sparse_signal, sparsity_of
from .matrices import (
    generate_measurement_matrix,
    generate_sensing_matrix,
    generate_projection_matrix,
    compute_matrix_stats,
    compute_projection_stats,
    budget_matched_dense_rows,
    add_noise,
    add_noise_snr,
)
from .omp import run_omp_trajectory, OMPTrajectory
from .metrics import (
    compute_cv_errors,
    relative_norm_errors,
    error_ratios,
    kendall_tau,
    spearman_rho,
    pairwise_concordance,
    best_iterate_selection,
    jl_distortion_violation_rate,
    jl_isometry_stats,
    noise_corrected_cv_errors,
    pre_convergence_relative_errors,
    ward_bound_stats,
)

__all__ = [
    "generate_sparse_signal",
    "sparsity_of",
    "generate_measurement_matrix",
    "generate_sensing_matrix",
    "generate_projection_matrix",
    "compute_matrix_stats",
    "compute_projection_stats",
    "budget_matched_dense_rows",
    "add_noise",
    "add_noise_snr",
    "run_omp_trajectory",
    "OMPTrajectory",
    "compute_cv_errors",
    "relative_norm_errors",
    "error_ratios",
    "kendall_tau",
    "spearman_rho",
    "pairwise_concordance",
    "best_iterate_selection",
    "jl_distortion_violation_rate",
    "jl_isometry_stats",
    "noise_corrected_cv_errors",
    "pre_convergence_relative_errors",
    "ward_bound_stats",
]
