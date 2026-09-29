"""
Groundwork package for Compressed Sensing Cross-Validation.
Contains fundamental mathematical building blocks:
- signals: sparse signal generation
- sensing: sensing matrices and additive noise
- omp: Orthogonal Matching Pursuit trajectory tracking
- projections: Gaussian, Rademacher, and Sparse Rademacher validation matrices
- metrics: validation metrics, Kendall's tau, distortion variance, Ward bounds
"""

from .signals import generate_sparse_signal, sparsity_of
from .sensing import generate_sensing_matrix, add_noise, add_noise_snr
from .omp import run_omp_trajectory, OMPTrajectory
from .projections import (
    generate_projection_matrix,
    compute_projection_stats,
    budget_matched_dense_rows,
)
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
    "generate_sensing_matrix",
    "add_noise",
    "add_noise_snr",
    "run_omp_trajectory",
    "OMPTrajectory",
    "generate_projection_matrix",
    "compute_projection_stats",
    "budget_matched_dense_rows",
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
