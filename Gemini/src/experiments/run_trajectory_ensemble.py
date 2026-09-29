"""
Stage 2 - Trajectory Ensemble Runner
Repeats OMP-CV over an ensemble of T independent random problem instances.
Calculates summary statistics across trials:
- Median and IQR (25th-75th percentile) error trajectories.
- Kendall's tau distributions.
- Best-iterate selection accuracy rates P(j_hat == j*).
- Relative norm error distributions.
Saves multi-panel figures and summary CSVs to Gemini/output/.
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
from groundwork.sensing import generate_sensing_matrix, add_noise_snr
from groundwork.omp import run_omp_trajectory
from groundwork.projections import generate_projection_matrix
from groundwork.metrics import (
    compute_cv_errors,
    relative_norm_errors,
    kendall_tau,
    pairwise_concordance,
    best_iterate_selection,
)


def run_ensemble(
    num_trials: int = 100,
    N: int = 512,
    s: int = 10,
    m: int = 120,
    r: int = 40,
    q_sparse: float = 0.2,
    q_ultra_sparse: float = 0.05,
    snr_db: float = 25.0,
    max_iter: int = 25,
    base_seed: int = 1000,
    output_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run an ensemble of T random problem instances and compute trajectory distributions.
    """
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("STAGE 2: RUNNING TRAJECTORY ENSEMBLE OVER RANDOM PROBLEMS")
    print(f"Trials: {num_trials}, N: {N}, s: {s}, m: {m}, r: {r}, SNR: {snr_db} dB")
    print(f"Testing projections: Dense Gaussian, Dense Rademacher, Sparse (q={q_sparse}), Ultra-Sparse (q={q_ultra_sparse})")
    print("=" * 70)

    model_keys = [
        ("dense_gauss", "Dense Gaussian"),
        ("dense_rad", "Dense Rademacher"),
        ("sparse_rad", f"Sparse Rad (q={q_sparse})"),
        ("ultra_sparse_rad", f"Sparse Rad (q={q_ultra_sparse})"),
    ]

    # Arrays to record per-iteration trajectory values across all trials: shape (T, p + 1)
    p = min(max_iter, m, N)
    E_true_all = np.zeros((num_trials, p + 1))
    E_hat_all = {key: np.zeros((num_trials, p + 1)) for key, _ in model_keys}

    trial_records = []

    for t in range(num_trials):
        rng = np.random.default_rng(base_seed + t)

        # 1. Signal & sensing setup
        x_true, support = generate_sparse_signal(N, s, coeff_dist="gaussian", rng=rng)
        Phi = generate_sensing_matrix(m, N, dist="gaussian", normalize="column", rng=rng)
        y_clean = Phi @ x_true
        y, _, _ = add_noise_snr(y_clean, snr_db=snr_db, rng=rng)

        # 2. OMP Trajectory
        traj = run_omp_trajectory(Phi, y, max_iter=p, x_true=x_true)
        E_true = traj.true_errors
        E_true_all[t] = E_true
        j_true_star = int(np.argmin(E_true))

        # 3. Validation Projections & CV Estimates
        # a. Dense Gaussian
        Psi_dg = generate_projection_matrix(r, N, matrix_type="gaussian", rng=rng)
        y_cv_dg, _, _ = add_noise_snr(Psi_dg @ x_true, snr_db=snr_db, rng=rng)
        E_hat_dg = compute_cv_errors(Psi_dg, y_cv_dg, traj.iterates)
        E_hat_all["dense_gauss"][t] = E_hat_dg

        # b. Dense Rademacher
        Psi_dr = generate_projection_matrix(r, N, matrix_type="dense_rademacher", rng=rng)
        y_cv_dr, _, _ = add_noise_snr(Psi_dr @ x_true, snr_db=snr_db, rng=rng)
        E_hat_dr = compute_cv_errors(Psi_dr, y_cv_dr, traj.iterates)
        E_hat_all["dense_rad"][t] = E_hat_dr

        # c. Sparse Rademacher (q_sparse)
        Psi_sr = generate_projection_matrix(r, N, matrix_type="sparse_rademacher", q=q_sparse, rng=rng)
        y_cv_sr, _, _ = add_noise_snr(Psi_sr @ x_true, snr_db=snr_db, rng=rng)
        E_hat_sr = compute_cv_errors(Psi_sr, y_cv_sr, traj.iterates)
        E_hat_all["sparse_rad"][t] = E_hat_sr

        # d. Ultra-Sparse Rademacher (q_ultra_sparse)
        Psi_usr = generate_projection_matrix(r, N, matrix_type="sparse_rademacher", q=q_ultra_sparse, rng=rng)
        y_cv_usr, _, _ = add_noise_snr(Psi_usr @ x_true, snr_db=snr_db, rng=rng)
        E_hat_usr = compute_cv_errors(Psi_usr, y_cv_usr, traj.iterates)
        E_hat_all["ultra_sparse_rad"][t] = E_hat_usr

        # 4. Evaluate Trial Metrics
        estimates_dict = {
            "dense_gauss": E_hat_dg,
            "dense_rad": E_hat_dr,
            "sparse_rad": E_hat_sr,
            "ultra_sparse_rad": E_hat_usr,
        }

        for key, name in model_keys:
            E_hat = estimates_dict[key]
            tau = kendall_tau(E_true, E_hat)
            concordance = pairwise_concordance(E_true, E_hat)
            best_info = best_iterate_selection(E_true, E_hat)
            rel_errs = relative_norm_errors(E_true, E_hat)

            trial_records.append({
                "trial": t,
                "model_key": key,
                "model_name": name,
                "kendall_tau": tau,
                "pairwise_concordance": concordance,
                "j_true_star": j_true_star,
                "j_hat_star": best_info["j_hat_star"],
                "is_exact_match": best_info["is_exact_match"],
                "excess_error": best_info["excess_error"],
                "median_rel_error": float(np.median(rel_errs)),
                "mean_rel_error": float(np.mean(rel_errs)),
            })

        if (t + 1) % 25 == 0 or t == num_trials - 1:
            print(f"Processed {t + 1}/{num_trials} trials...")

    df_trials = pd.DataFrame(trial_records)
    trials_csv_path = os.path.join(output_dir, "ensemble_trials.csv")
    df_trials.to_csv(trials_csv_path, index=False)

    # 5. Compute summary statistics across trials
    summary_list = []
    for key, name in model_keys:
        sub = df_trials[df_trials["model_key"] == key]
        summary_list.append({
            "Model": name,
            "Selection_Accuracy": float(sub["is_exact_match"].mean()),
            "Mean_Excess_Error": float(sub["excess_error"].mean()),
            "Median_Excess_Error": float(sub["excess_error"].median()),
            "Mean_Kendall_Tau": float(sub["kendall_tau"].mean()),
            "Median_Kendall_Tau": float(sub["kendall_tau"].median()),
            "Mean_Concordance": float(sub["pairwise_concordance"].mean()),
            "Mean_Rel_Error": float(sub["mean_rel_error"].mean()),
            "Median_Rel_Error": float(sub["median_rel_error"].median()),
        })

    df_summary = pd.DataFrame(summary_list)
    summary_csv_path = os.path.join(output_dir, "ensemble_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 70)
    print("ENSEMBLE SUMMARY STATISTICS ACROSS ALL TRIALS:")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print("=" * 70)

    # 6. Generate Publication Figure
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    iterations = np.arange(p + 1)

    # Subplot 1: Trajectory Bands (Median & 25-75th Percentile)
    ax1 = fig.add_subplot(gs[0, :])
    # True error ribbon
    true_med = np.median(E_true_all, axis=0)
    true_p25 = np.percentile(E_true_all, 25, axis=0)
    true_p75 = np.percentile(E_true_all, 75, axis=0)
    ax1.semilogy(iterations, true_med, color="black", linestyle="-", linewidth=2.5, label="True Error $E_j$ (Median)")
    ax1.fill_between(iterations, true_p25, true_p75, color="black", alpha=0.15)

    colors = {
        "dense_gauss": ("royalblue", "--"),
        "dense_rad": ("forestgreen", ":"),
        "sparse_rad": ("crimson", "-."),
        "ultra_sparse_rad": ("darkorange", "--"),
    }

    for key, name in model_keys:
        color, ls = colors[key]
        arr = E_hat_all[key]
        med = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        ax1.semilogy(iterations, med, color=color, linestyle=ls, linewidth=1.8, label=f"{name} (Median)")
        ax1.fill_between(iterations, p25, p75, color=color, alpha=0.08)

    ax1.axvline(s, color="gray", linestyle=":", alpha=0.7, label=f"True Sparsity s={s}")
    ax1.set_xlabel("OMP Iteration $j$", fontsize=11)
    ax1.set_ylabel(r"Squared Error ($\ell_2^2$)", fontsize=11)
    ax1.set_title(f"Trajectory Ensemble Distributions over {num_trials} Random Realizations (Median & IQR Bands)\n(N={N}, s={s}, m={m}, r={r}, SNR={snr_db} dB)", fontsize=12)
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=9, ncol=2)

    # Subplot 2: Kendall's Tau Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    tau_data = [df_trials[df_trials["model_key"] == key]["kendall_tau"].values for key, _ in model_keys]
    labels = [name for _, name in model_keys]
    bp = ax2.boxplot(tau_data, tick_labels=labels, patch_artist=True)
    box_colors = ["lightblue", "lightgreen", "salmon", "wheat"]
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c)
    ax2.set_ylabel("Kendall's Rank Correlation $\\tau$", fontsize=11)
    ax2.set_title("Ordering Preservation Distribution (Kendall's $\\tau$)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.tick_params(axis="x", rotation=15)

    # Subplot 3: Selection Accuracy Comparison Bar Chart
    ax3 = fig.add_subplot(gs[1, 1])
    accuracies = [df_summary[df_summary["Model"] == name]["Selection_Accuracy"].values[0] * 100 for _, name in model_keys]
    bars = ax3.bar(labels, accuracies, color=box_colors, edgecolor="black", width=0.55)
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f"{height:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight="bold")
    ax3.set_ylabel("Selection Accuracy $P(\\hat{j} = j^*)$ (%)", fontsize=11)
    ax3.set_ylim(0, 105)
    ax3.set_title("Best-Iterate Model Selection Accuracy Rate", fontsize=12)
    ax3.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax3.tick_params(axis="x", rotation=15)

    fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.10, hspace=0.35, wspace=0.25)
    plot_path = os.path.join(output_dir, "ensemble_trajectories.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Saved ensemble visualization to: {plot_path}")
    print(f"Saved summary metrics to: {summary_csv_path}")
    print(f"Saved trial data to: {trials_csv_path}\n")

    return df_summary, df_trials


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OMP-CV trajectory ensemble")
    parser.add_argument("--trials", type=int, default=100, help="Number of random trials")
    parser.add_argument("--N", type=int, default=512, help="Signal dimension")
    parser.add_argument("--s", type=int, default=10, help="Sparsity")
    parser.add_argument("--m", type=int, default=120, help="Sensing measurements")
    parser.add_argument("--r", type=int, default=40, help="Validation measurements")
    parser.add_argument("--q_sparse", type=float, default=0.2, help="Sparse Rademacher density")
    parser.add_argument("--q_ultra_sparse", type=float, default=0.05, help="Ultra-sparse Rademacher density")
    parser.add_argument("--snr", type=float, default=25.0, help="SNR in dB")
    parser.add_argument("--max_iter", type=int, default=25, help="Max OMP iterations")
    parser.add_argument("--seed", type=int, default=1000, help="Base random seed")
    args = parser.parse_args()

    run_ensemble(
        num_trials=args.trials,
        N=args.N,
        s=args.s,
        m=args.m,
        r=args.r,
        q_sparse=args.q_sparse,
        q_ultra_sparse=args.q_ultra_sparse,
        snr_db=args.snr,
        max_iter=args.max_iter,
        base_seed=args.seed,
    )
