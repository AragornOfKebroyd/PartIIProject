"""
Prototype script demonstrating Phase 1 pipeline:
1. Generates an s-sparse signal and measurements.
2. Runs OMP and records full iterate trajectory.
3. Computes cross-validation estimates using both Dense Gaussian and Sparse Rademacher projections.
4. Evaluates the 3 notions of success: relative error, Kendall's tau, and best-iterate selection.
5. Saves visualization and metrics table to Gemini/output/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local imports work cleanly
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from groundwork.signals import generate_sparse_signal
from groundwork.matrices import (
    generate_sensing_matrix,
    generate_projection_matrix,
    compute_projection_stats,
    add_noise_snr,
)
from groundwork.omp import run_omp_trajectory
from groundwork.metrics import (
    compute_cv_errors,
    relative_norm_errors,
    kendall_tau,
    pairwise_concordance,
    best_iterate_selection,
)


def main():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Configuration parameters
    N = 512       # Signal dimension
    s = 10        # Sparsity
    m = 120       # Sensing measurements
    r = 35        # Validation measurements
    snr_db = 25.0 # SNR in dB
    max_iter = 2 * s
    rng = np.random.default_rng(2026)

    print("=" * 60)
    print("PHASE 1: RUNNING COMPRESSED SENSING & CV PIPELINE")
    print(f"Parameters: N={N}, s={s}, m={m}, r={r}, SNR={snr_db} dB, max_iter={max_iter}")
    print("=" * 60)

    # 2. Generate ground-truth signal
    x_true, support = generate_sparse_signal(N, s, coeff_dist="gaussian", rng=rng)

    # 3. Generate sensing matrix & measurements
    Phi = generate_sensing_matrix(m, N, dist="gaussian", normalize="column", rng=rng)
    y_clean = Phi @ x_true
    y, noise, sigma_noise = add_noise_snr(y_clean, snr_db=snr_db, rng=rng)

    # 4. Run OMP and record full trajectory
    traj = run_omp_trajectory(Phi, y, max_iter=max_iter, x_true=x_true)
    E_true = traj.true_errors
    p = traj.p
    print(f"OMP completed {p} iterations.")
    print(f"True minimum error reached at iteration j* = {np.argmin(E_true)} (Error = {np.min(E_true):.4e})")

    # 5. Generate Validation Projections
    # Baseline: Dense Gaussian
    Psi_dense = generate_projection_matrix(r, N, matrix_type="gaussian", rng=rng)
    y_cv_dense, _, _ = add_noise_snr(Psi_dense @ x_true, snr_db=snr_db, rng=rng)
    E_hat_dense = compute_cv_errors(Psi_dense, y_cv_dense, traj.iterates)

    # Sparse Rademacher (density q = 0.2, i.e. 80% zeros)
    q = 0.2
    Psi_sparse = generate_projection_matrix(r, N, matrix_type="sparse_rademacher", q=q, rng=rng)
    sparse_stats = compute_projection_stats(Psi_sparse)
    y_cv_sparse, _, _ = add_noise_snr(Psi_sparse @ x_true, snr_db=snr_db, rng=rng)
    E_hat_sparse = compute_cv_errors(Psi_sparse, y_cv_sparse, traj.iterates)

    print(f"\nSparse Rademacher (target q={q}): empirical density = {sparse_stats['empirical_density']:.3f} "
          f"({sparse_stats['nnz']}/{sparse_stats['total_elements']} nonzeros)")

    # 6. Evaluate the Three Notions of "Works"
    metrics_summary = []
    for name, E_hat in [("Dense Gaussian", E_hat_dense), (f"Sparse Rademacher (q={q})", E_hat_sparse)]:
        rel_errors = relative_norm_errors(E_true, E_hat)
        tau = kendall_tau(E_true, E_hat)
        concordance = pairwise_concordance(E_true, E_hat)
        selection = best_iterate_selection(E_true, E_hat)

        metrics_summary.append({
            "Projection": name,
            "Median_Rel_Error": float(np.median(rel_errors)),
            "Mean_Rel_Error": float(np.mean(rel_errors)),
            "Kendall_Tau": tau,
            "Pairwise_Concordance": concordance,
            "True_Optimal_j*": selection["j_true_star"],
            "Selected_j_hat": selection["j_hat_star"],
            "Exact_Match": selection["is_exact_match"],
            "Excess_Error": selection["excess_error"],
        })

    df_summary = pd.DataFrame(metrics_summary)
    print("\nEvaluation Summary:")
    print(df_summary.to_string(index=False))

    # Save summary table
    summary_csv_path = os.path.join(output_dir, "phase1_demo_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"\nSaved metrics summary to: {summary_csv_path}")

    # 7. Generate Trajectory Plot
    iterations = np.arange(p + 1)
    plt.figure(figsize=(10, 6))
    plt.semilogy(iterations, E_true, color="black", linestyle="-", marker="o", label=r"True Error $E_j = \|\mathbf{x} - \mathbf{x}^j\|_2^2$", linewidth=2.5)
    plt.semilogy(iterations, E_hat_dense, color="blue", linestyle="--", marker="s", label=r"Dense Gaussian $\widehat{E}_j$", alpha=0.8)
    plt.semilogy(iterations, E_hat_sparse, color="red", linestyle="-.", marker="^", label=f"Sparse Rademacher ($q={q}$) " + r"$\widehat{E}_j$", alpha=0.8)

    plt.axvline(selection["j_true_star"], color="gray", linestyle=":", label=f"True optimal $j^*={selection['j_true_star']}$")
    plt.xlabel("OMP Iteration $j$", fontsize=12)
    plt.ylabel(r"Squared Error ($\ell_2^2$)", fontsize=12)
    plt.title(f"Reconstruction Error Trajectory vs CV Estimates (N={N}, s={s}, m={m}, r={r})", fontsize=13)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "phase1_trajectory_plot.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved trajectory plot to: {fig_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
