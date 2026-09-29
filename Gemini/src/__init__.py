"""
Gemini Compressed Sensing Library

Organised into:
- groundwork: Fundamental mathematical modules (signals, sensing, omp, projections, metrics)
- experiments: Experimental stages and comparison sweeps
- run_pipeline: Master execution pipeline
"""

from . import groundwork
from . import experiments

# Expose groundwork directly at package level
from .groundwork.signals import generate_sparse_signal, sparsity_of
from .groundwork.sensing import generate_sensing_matrix, add_noise, add_noise_snr
from .groundwork.omp import run_omp_trajectory, OMPTrajectory
from .groundwork.projections import (
    generate_projection_matrix,
    compute_projection_stats,
    budget_matched_dense_rows,
)
from .groundwork.metrics import (
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
    "groundwork",
    "experiments",
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
