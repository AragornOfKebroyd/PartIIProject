"""
Experiments package for Compressed Sensing Cross-Validation.
Contains experimental stages and evaluation routines:
- plot_single_trajectory: Stage 2 single trajectory visualizer
- run_trajectory_ensemble: Stage 2 ensemble trajectory statistics
- run_single_experiment: Single run test harness
- evaluate_three_notions: Stage 3 norm error vs ordering vs selection
- compare_matrix_families: Stage 4 Gaussian vs Rademacher vs Sparse Rademacher
- compare_budgets: Stage 5 iso-compute non-zero budget trade-offs
- compare_jl_hierarchy: Stage 6 5-level task hierarchy & Ward bounds
- analyze_separation_reversal: Stage 6b candidate error separation vs reversals
- generate_heatmaps: Stage 7 2D phase maps & dimensional scaling
- compare_joint_sparsity: Joint sensing & validation matrix sparsity cross-product
"""

from .plot_single_trajectory import run_and_plot_single_trajectory
from .run_trajectory_ensemble import run_ensemble
from .evaluate_three_notions import evaluate_three_notions
from .compare_matrix_families import compare_matrix_families
from .compare_budgets import run_budget_experiments
from .compare_jl_hierarchy import run_jl_hierarchy_experiments, compute_all_pairs_jl
from .analyze_separation_reversal import run_separation_reversal_analysis
from .generate_heatmaps import run_heatmap_grid, run_regime_scaling_sweep
from .compare_joint_sparsity import run_joint_sparsity_cross_product

__all__ = [
    "run_and_plot_single_trajectory",
    "run_ensemble",
    "evaluate_three_notions",
    "compare_matrix_families",
    "run_budget_experiments",
    "run_jl_hierarchy_experiments",
    "compute_all_pairs_jl",
    "run_separation_reversal_analysis",
    "run_heatmap_grid",
    "run_regime_scaling_sweep",
    "run_joint_sparsity_cross_product",
]
