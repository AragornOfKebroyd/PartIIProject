"""
Stage 4 - Comparing Matrix Distribution Families
Compares:
  1. Dense Gaussian: N(0, 1/r), kurtosis parameter gamma = 2/r^2
  2. Dense Rademacher: +/- 1/sqrt(r), kurtosis parameter gamma = 0
  3. Sparse Rademacher (q=0.5): gamma = 1/r^2 (sub-Gaussian)
  4. Sparse Rademacher (q=0.2): gamma = 4/r^2
  5. Sparse Rademacher (q=0.05): gamma = 19/r^2

Sweeps validation measurement budget r in [10, 20, 30, 40, 60, 80, 100].
Tests Zhang et al.'s (2014) prediction on CV variance and evaluates model selection accuracy.

Saves summary tables and multi-panel figures to Gemini/output/.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src modules can be imported
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from groundwork.signals import generate_sparse_signal
from groundwork.matrices import (
    generate_sensing_matrix,
    generate_projection_matrix,
    add_noise_snr,
)
from groundwork.omp import run_omp_trajectory
from groundwork.metrics import (
    compute_cv_errors,
    pre_convergence_relative_errors,
    kendall_tau,
    pairwise_concordance,
    best_iterate_selection,
    jl_isometry_stats,
)


def compare_matrix_families(
    r_values: list[int] | None = None,
    num_trials: int = 100,
    N: int = 512,
    s: int = 10,
    m: int = 120,
    snr_db: float = 25.0,
    max_iter: int = 25,
    base_seed: int = 4040,
    output_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare Dense Gaussian, Dense Rademacher, and Sparse Rademacher across validation measurements r.
    """
    if r_values is None:
        r_values = [10, 20, 30, 40, 60, 80, 100]

    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("STAGE 4: MATRIX DISTRIBUTION FAMILIES COMPARISON ACROSS VALIDATION BUDGET r")
    print(f"Problem: N={N}, s={s}, m={m}, SNR={snr_db} dB, max_iter={max_iter}")
    print(f"Validation rows r: {r_values}")
    print(f"Trials per r: {num_trials}")
    print("=" * 80)

    families = [
        ("dense_gauss", "Dense Gaussian", "gaussian", 1.0),
        ("dense_rad", "Dense Rademacher", "dense_rademacher", 1.0),
        ("sparse_rad_05", "Sparse Rad (q=0.5)", "sparse_rademacher", 0.5),
        ("sparse_rad_02", "Sparse Rad (q=0.2)", "sparse_rademacher", 0.2),
        ("sparse_rad_005", "Sparse Rad (q=0.05)", "sparse_rademacher", 0.05),
    ]

    p = min(max_iter, m, N)

    # Pre-generate problems so that all families and r values are tested on identical problems
    print("Generating problem instances...")
    problems = []
    for t in range(num_trials):
        rng = np.random.default_rng(base_seed + t)
        x_true, support = generate_sparse_signal(N, s, coeff_dist="gaussian", rng=rng)
        Phi = generate_sensing_matrix(m, N, dist="gaussian", normalize="column", rng=rng)
        y_clean = Phi @ x_true
        y, _, sigma_n = add_noise_snr(y_clean, snr_db=snr_db, rng=rng)
        traj = run_omp_trajectory(Phi, y, max_iter=p, x_true=x_true)
        error_vectors = x_true[None, :] - traj.iterates  # (p+1, N)
        problems.append((x_true, traj, error_vectors, sigma_n))

    all_trial_records = []

    for r in r_values:
        print(f"\nEvaluating validation measurements r = {r}...")
        for family_key, family_name, matrix_type, q in families:
            for t, (x_true, traj, error_vectors, sigma_n) in enumerate(problems):
                rng_val = np.random.default_rng(base_seed + 100000 * r + int(q * 1000) + t)
                Psi = generate_projection_matrix(r, N, matrix_type=matrix_type, q=q, rng=rng_val)
                y_cv, _, _ = add_noise_snr(Psi @ x_true, snr_db=snr_db, rng=rng_val)
                E_hat = compute_cv_errors(Psi, y_cv, traj.iterates)

                E_true = traj.true_errors
                sel = best_iterate_selection(E_true, E_hat)
                near_match = abs(sel["j_hat_star"] - sel["j_true_star"]) <= 1

                # JL distortion stats on error vectors
                jl_stats = jl_isometry_stats(error_vectors, Psi, epsilons=(0.2, 0.5))

                # Pre-convergence relative norm error
                pre_rel = pre_convergence_relative_errors(E_true, E_hat, sel["j_true_star"])

                tau = kendall_tau(E_true, E_hat)
                conc = pairwise_concordance(E_true, E_hat)

                all_trial_records.append({
                    "trial": t,
                    "r": r,
                    "family_key": family_key,
                    "family_name": family_name,
                    "matrix_type": matrix_type,
                    "q": q,
                    "jl_eps20_success": jl_stats["success_rates"][0.2],
                    "jl_eps50_success": jl_stats["success_rates"][0.5],
                    "jl_distortion_std": jl_stats["std_ratio"],
                    "pre_conv_rel_error": float(np.median(pre_rel)),
                    "kendall_tau": tau,
                    "pairwise_concordance": conc,
                    "j_true_star": sel["j_true_star"],
                    "j_hat_star": sel["j_hat_star"],
                    "is_exact_match": sel["is_exact_match"],
                    "is_near_match": near_match,
                    "excess_error": sel["excess_error"],
                })

    df_trials = pd.DataFrame(all_trial_records)
    trials_csv_path = os.path.join(output_dir, "matrix_families_trials.csv")
    df_trials.to_csv(trials_csv_path, index=False)

    # Compute summary table grouped by (r, family_key)
    summary_rows = []
    for r in r_values:
        for family_key, family_name, _, q in families:
            sub = df_trials[(df_trials["r"] == r) & (df_trials["family_key"] == family_key)]
            summary_rows.append({
                "r": r,
                "family_key": family_key,
                "Family": family_name,
                "q": q,
                "JL_Success_Eps20": float(sub["jl_eps20_success"].mean()),
                "JL_Success_Eps50": float(sub["jl_eps50_success"].mean()),
                "Distortion_Std": float(sub["jl_distortion_std"].mean()),
                "PreConv_Rel_Error": float(sub["pre_conv_rel_error"].median()),
                "Mean_Kendall_Tau": float(sub["kendall_tau"].mean()),
                "Mean_Concordance": float(sub["pairwise_concordance"].mean()),
                "Selection_Accuracy": float(sub["is_exact_match"].mean()),
                "Near_Selection_Accuracy": float(sub["is_near_match"].mean()),
                "Mean_Excess_Error": float(sub["excess_error"].mean()),
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "matrix_families_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 95)
    print("STAGE 4 SUMMARY: SELECTION ACCURACY AND DISTORTION VARIANCE BY MATRIX FAMILY")
    print("=" * 95)
    pivot_acc = df_summary.pivot(index="r", columns="Family", values="Selection_Accuracy") * 100
    pivot_std = df_summary.pivot(index="r", columns="Family", values="Distortion_Std")
    print("\n--- Selection Accuracy P(j_hat == j*) (%) ---")
    print(pivot_acc.round(1).to_string())
    print("\n--- Distortion Standard Deviation (Lower = Tighter Bounds) ---")
    print(pivot_std.round(3).to_string())
    print("=" * 95)

    # Multi-panel visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_std, ax_acc = axes[0, 0], axes[0, 1]
    ax_tau, ax_rel = axes[1, 0], axes[1, 1]

    family_styles = {
        "dense_gauss": ("royalblue", "-", "o"),
        "dense_rad": ("forestgreen", "--", "s"),
        "sparse_rad_05": ("darkorange", "-.", "^"),
        "sparse_rad_02": ("crimson", ":", "d"),
        "sparse_rad_005": ("purple", "-", "x"),
    }

    # Panel 1: Distortion Spread / Standard Deviation vs r
    for key, name, _, _ in families:
        color, ls, m_mark = family_styles[key]
        sub = df_summary[df_summary["family_key"] == key]
        ax_std.plot(sub["r"], sub["Distortion_Std"], color=color, linestyle=ls, marker=m_mark, linewidth=2.0, label=name)
    ax_std.set_xlabel("Validation Measurements $r$", fontsize=11)
    ax_std.set_ylabel(r"Distortion Std $\sigma(\frac{\|\Psi \mathbf{e}\|^2}{\|\mathbf{e}\|^2})$", fontsize=11)
    ax_std.set_title("Variance of Error Estimation (Testing Zhang et al. $\\gamma$)", fontsize=12, fontweight="bold")
    ax_std.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_std.legend(fontsize=9)

    # Panel 2: Best-Iterate Selection Accuracy vs r
    for key, name, _, _ in families:
        color, ls, m_mark = family_styles[key]
        sub = df_summary[df_summary["family_key"] == key]
        ax_acc.plot(sub["r"], sub["Selection_Accuracy"] * 100, color=color, linestyle=ls, marker=m_mark, linewidth=2.0, label=name)
    ax_acc.set_xlabel("Validation Measurements $r$", fontsize=11)
    ax_acc.set_ylabel("Selection Accuracy $P(\\hat{j}=j^*)$ (%)", fontsize=11)
    ax_acc.set_ylim(20, 100)
    ax_acc.set_title("Model Selection Accuracy Rate vs Validation Budget $r$", fontsize=12, fontweight="bold")
    ax_acc.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_acc.legend(fontsize=9, loc="lower right")

    # Panel 3: Kendall's Tau vs r
    for key, name, _, _ in families:
        color, ls, m_mark = family_styles[key]
        sub = df_summary[df_summary["family_key"] == key]
        ax_tau.plot(sub["r"], sub["Mean_Kendall_Tau"], color=color, linestyle=ls, marker=m_mark, linewidth=2.0, label=name)
    ax_tau.set_xlabel("Validation Measurements $r$", fontsize=11)
    ax_tau.set_ylabel("Mean Kendall's $\\tau$", fontsize=11)
    ax_tau.set_ylim(0.70, 1.00)
    ax_tau.set_title("Ordering Preservation (Kendall's $\\tau$) vs Validation Budget $r$", fontsize=12, fontweight="bold")
    ax_tau.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_tau.legend(fontsize=9, loc="lower right")

    # Panel 4: Pre-convergence Relative Error vs r
    for key, name, _, _ in families:
        color, ls, m_mark = family_styles[key]
        sub = df_summary[df_summary["family_key"] == key]
        ax_rel.plot(sub["r"], sub["PreConv_Rel_Error"], color=color, linestyle=ls, marker=m_mark, linewidth=2.0, label=name)
    ax_rel.set_xlabel("Validation Measurements $r$", fontsize=11)
    ax_rel.set_ylabel(r"Median Relative Error $\frac{|\widehat{E}_j - E_j|}{E_j}$", fontsize=11)
    ax_rel.set_title("Signal Norm Estimation Error vs Validation Budget $r$", fontsize=12, fontweight="bold")
    ax_rel.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_rel.legend(fontsize=9)

    fig.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.08, hspace=0.30, wspace=0.25)
    plot_path = os.path.join(output_dir, "matrix_families_comparison.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"\nSaved matrix families figure to: {plot_path}")
    print(f"Saved summary metrics to: {summary_csv_path}")
    print(f"Saved trials metrics to: {trials_csv_path}\n")

    return df_summary, df_trials


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare matrix distribution families across validation budget r")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials per r")
    parser.add_argument("--N", type=int, default=512, help="Signal dimension")
    parser.add_argument("--s", type=int, default=10, help="Sparsity")
    parser.add_argument("--m", type=int, default=120, help="Sensing measurements")
    parser.add_argument("--snr", type=float, default=25.0, help="SNR in dB")
    parser.add_argument("--max_iter", type=int, default=25, help="OMP iterations")
    parser.add_argument("--seed", type=int, default=4040, help="Base random seed")
    args = parser.parse_args()

    compare_matrix_families(
        num_trials=args.trials,
        N=args.N,
        s=args.s,
        m=args.m,
        snr_db=args.snr,
        max_iter=args.max_iter,
        base_seed=args.seed,
    )
