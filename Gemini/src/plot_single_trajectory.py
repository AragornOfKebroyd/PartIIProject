"""
Stage 2 - Single Trajectory Visualizer
Generates a single compressed sensing problem instance, runs OMP,
computes cross-validation estimates across multiple projection types,
and plots E_j vs E_hat_j with iteration diagnostics.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src modules can be imported
sys.path.insert(0, os.path.dirname(__file__))

from signals import generate_sparse_signal
from sensing import generate_sensing_matrix, add_noise_snr
from omp import run_omp_trajectory
from projections import generate_projection_matrix, compute_projection_stats
from metrics import (
    compute_cv_errors,
    relative_norm_errors,
    kendall_tau,
    pairwise_concordance,
    best_iterate_selection,
)


def run_and_plot_single_trajectory(
    N: int = 512,
    s: int = 10,
    m: int = 120,
    r: int = 40,
    q: float = 0.2,
    snr_db: float = 25.0,
    max_iter: int = 25,
    seed: int = 42,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Run a single problem instance and save trajectory plot and metrics table.
    """
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)

    # 1. Signal and sensing measurements
    x_true, support = generate_sparse_signal(N, s, coeff_dist="gaussian", rng=rng)
    Phi = generate_sensing_matrix(m, N, dist="gaussian", normalize="column", rng=rng)
    y_clean = Phi @ x_true
    y, _, _ = add_noise_snr(y_clean, snr_db=snr_db, rng=rng)

    # 2. OMP Trajectory
    traj = run_omp_trajectory(Phi, y, max_iter=max_iter, x_true=x_true)
    E_true = traj.true_errors
    p = traj.p
    iterations = np.arange(p + 1)
    j_true_star = int(np.argmin(E_true))

    # 3. Validation Projections
    # a. Dense Gaussian
    Psi_gauss = generate_projection_matrix(r, N, matrix_type="gaussian", rng=rng)
    y_cv_gauss, _, _ = add_noise_snr(Psi_gauss @ x_true, snr_db=snr_db, rng=rng)
    E_hat_gauss = compute_cv_errors(Psi_gauss, y_cv_gauss, traj.iterates)

    # b. Dense Rademacher
    Psi_rad = generate_projection_matrix(r, N, matrix_type="dense_rademacher", rng=rng)
    y_cv_rad, _, _ = add_noise_snr(Psi_rad @ x_true, snr_db=snr_db, rng=rng)
    E_hat_rad = compute_cv_errors(Psi_rad, y_cv_rad, traj.iterates)

    # c. Sparse Rademacher (density q)
    Psi_sparse = generate_projection_matrix(r, N, matrix_type="sparse_rademacher", q=q, rng=rng)
    sparse_stats = compute_projection_stats(Psi_sparse)
    y_cv_sparse, _, _ = add_noise_snr(Psi_sparse @ x_true, snr_db=snr_db, rng=rng)
    E_hat_sparse = compute_cv_errors(Psi_sparse, y_cv_sparse, traj.iterates)

    # 4. Compile iteration-level data
    df_trajectory = pd.DataFrame({
        "iteration": iterations,
        "true_error": E_true,
        "error_sparsity": traj.error_sparsities,
        "cv_gauss": E_hat_gauss,
        "cv_rademacher": E_hat_rad,
        "cv_sparse": E_hat_sparse,
        "rel_err_gauss": relative_norm_errors(E_true, E_hat_gauss),
        "rel_err_rad": relative_norm_errors(E_true, E_hat_rad),
        "rel_err_sparse": relative_norm_errors(E_true, E_hat_sparse),
    })

    csv_path = os.path.join(output_dir, "single_trajectory_data.csv")
    df_trajectory.to_csv(csv_path, index=False)

    # 5. Evaluate summary metrics
    models = [
        ("Dense Gaussian", E_hat_gauss),
        ("Dense Rademacher", E_hat_rad),
        (f"Sparse Rademacher (q={q})", E_hat_sparse),
    ]

    print("\n--- Single Trajectory Summary ---")
    print(f"Problem: N={N}, s={s}, m={m}, r={r}, SNR={snr_db} dB, true optimal j*={j_true_star}")
    print(f"Sparse Rademacher nonzeros: {sparse_stats['nnz']}/{sparse_stats['total_elements']} ({sparse_stats['empirical_density']:.1%})")
    print("-" * 65)
    for name, E_hat in models:
        tau = kendall_tau(E_true, E_hat)
        concordance = pairwise_concordance(E_true, E_hat)
        best_info = best_iterate_selection(E_true, E_hat)
        med_rel = float(np.median(relative_norm_errors(E_true, E_hat)))
        print(f"{name:28s} | Tau: {tau:.3f} | Conc: {concordance:.3f} | Sel j^: {best_info['j_hat_star']:2d} (Match: {str(best_info['is_exact_match']):5s}) | MedRel: {med_rel:.3f}")
    print("-" * 65)

    # 6. Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

    # Top panel: Error curves
    ax1.semilogy(iterations, E_true, color="black", linestyle="-", marker="o", linewidth=2.5, label=r"True Error $E_j = \|\mathbf{x} - \mathbf{x}^j\|_2^2$")
    ax1.semilogy(iterations, E_hat_gauss, color="royalblue", linestyle="--", marker="s", alpha=0.85, label=r"Dense Gaussian $\widehat{E}_j$")
    ax1.semilogy(iterations, E_hat_rad, color="forestgreen", linestyle=":", marker="d", alpha=0.85, label=r"Dense Rademacher $\widehat{E}_j$")
    ax1.semilogy(iterations, E_hat_sparse, color="crimson", linestyle="-.", marker="^", alpha=0.85, label=f"Sparse Rademacher ($q={q}$) " + r"$\widehat{E}_j$")

    ax1.axvline(j_true_star, color="black", linestyle="--", alpha=0.5, label=f"True optimal $j^*={j_true_star}$")
    ax1.set_ylabel(r"Squared Error ($\ell_2^2$)", fontsize=11)
    ax1.set_title(f"True Error vs Cross-Validation Estimates along OMP Trajectory\n(N={N}, s={s}, m={m}, r={r}, SNR={snr_db} dB, sparse density q={q})", fontsize=12)
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=10)

    # Bottom panel: Error vector sparsity ||x - x^j||_0
    ax2.plot(iterations, traj.error_sparsities, color="purple", marker="o", linestyle="-", label=r"$\|\mathbf{x} - \mathbf{x}^j\|_0$")
    ax2.plot(iterations, s + iterations, color="gray", linestyle=":", label=r"Theoretical bound $s + j$")
    ax2.set_xlabel("OMP Iteration $j$", fontsize=11)
    ax2.set_ylabel(r"Error Sparsity", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "single_trajectory.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved trajectory plot to: {plot_path}")
    print(f"Saved trajectory data to: {csv_path}\n")

    return df_trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot single OMP-CV trajectory")
    parser.add_argument("--N", type=int, default=512, help="Signal dimension")
    parser.add_argument("--s", type=int, default=10, help="Sparsity")
    parser.add_argument("--m", type=int, default=120, help="Sensing measurements")
    parser.add_argument("--r", type=int, default=40, help="Validation measurements")
    parser.add_argument("--q", type=float, default=0.2, help="Sparse Rademacher density")
    parser.add_argument("--snr", type=float, default=25.0, help="SNR in dB")
    parser.add_argument("--max_iter", type=int, default=25, help="OMP iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_and_plot_single_trajectory(
        N=args.N,
        s=args.s,
        m=args.m,
        r=args.r,
        q=args.q,
        snr_db=args.snr,
        max_iter=args.max_iter,
        seed=args.seed,
    )
