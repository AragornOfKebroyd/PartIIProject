"""
Stage 3 - Evaluating the Three Notions of "Works" (Refined)
Measures how validation matrix sparsity (density q) affects:
  1. Classical JL Isometry: Success rates for (1 +/- eps) and distortion standard deviation.
  2. Notion A (Norm Estimation): Pre-convergence relative error (j <= j*), avoiding noise-floor contamination.
  3. Notion B (Ordering Preservation): Kendall's tau and pairwise concordance.
  4. Notion C (Model Selection): Best iterate selection rate P(j_hat == j*), near selection, and excess error.

Saves summary tables and multi-panel figures to Gemini/output/.
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
from projections import generate_projection_matrix
from metrics import (
    compute_cv_errors,
    relative_norm_errors,
    pre_convergence_relative_errors,
    kendall_tau,
    pairwise_concordance,
    best_iterate_selection,
    jl_isometry_stats,
    ward_bound_stats,
)


def evaluate_three_notions(
    q_values: list[float] | None = None,
    num_trials: int = 100,
    N: int = 512,
    s: int = 10,
    m: int = 120,
    r: int = 40,
    snr_db: float = 25.0,
    max_iter: int = 25,
    base_seed: int = 2026,
    output_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sweep sparsity density q across multiple trials and measure the refined notions.
    """
    if q_values is None:
        q_values = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]

    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("STAGE 3 (REFINED): EVALUATING THE VALIDATION HIERARCHY ACROSS SPARSITY (q)")
    print(f"Problem: N={N}, s={s}, m={m}, r={r}, SNR={snr_db} dB, max_iter={max_iter}")
    print(f"Sparsity grid q: {q_values}")
    print(f"Trials per q: {num_trials}")
    print("=" * 80)

    all_trial_records = []
    p = min(max_iter, m, N)

    # Pre-generate problems so that all q values are evaluated on the exact same problem instances
    print("Generating problem instances...")
    problems = []
    for t in range(num_trials):
        rng = np.random.default_rng(base_seed + t)
        x_true, support = generate_sparse_signal(N, s, coeff_dist="gaussian", rng=rng)
        Phi = generate_sensing_matrix(m, N, dist="gaussian", normalize="column", rng=rng)
        y_clean = Phi @ x_true
        y, _, sigma_n = add_noise_snr(y_clean, snr_db=snr_db, rng=rng)
        traj = run_omp_trajectory(Phi, y, max_iter=p, x_true=x_true)
        # Precompute true error vectors e^j = x - x^j
        error_vectors = x_true[None, :] - traj.iterates  # (p+1, N)
        problems.append((x_true, traj, error_vectors, sigma_n))

    # Dense Gaussian reference benchmark
    gauss_records = []
    for t, (x_true, traj, error_vectors, sigma_n) in enumerate(problems):
        rng_val = np.random.default_rng(base_seed + 10000 + t)
        Psi_g = generate_projection_matrix(r, N, matrix_type="gaussian", rng=rng_val)
        y_cv_g, _, _ = add_noise_snr(Psi_g @ x_true, snr_db=snr_db, rng=rng_val)
        E_hat_g = compute_cv_errors(Psi_g, y_cv_g, traj.iterates)

        E_true = traj.true_errors
        sel = best_iterate_selection(E_true, E_hat_g)
        jl_stats = jl_isometry_stats(error_vectors, Psi_g, epsilons=(0.2, 0.5))
        pre_rel = pre_convergence_relative_errors(E_true, E_hat_g, sel["j_true_star"])
        proj_error_g_sq = np.sum((error_vectors @ Psi_g.T) ** 2, axis=1)
        ward_g = ward_bound_stats(E_true, proj_error_g_sq)

        gauss_records.append({
            "jl_eps20": jl_stats["success_rates"][0.2],
            "jl_eps50": jl_stats["success_rates"][0.5],
            "jl_std": jl_stats["std_ratio"],
            "ward_success": ward_g["ward_simultaneous_success"],
            "ward_fraction": ward_g["ward_fraction_satisfied"],
            "pre_rel_med": float(np.median(pre_rel)),
            "tau": kendall_tau(E_true, E_hat_g),
            "conc": pairwise_concordance(E_true, E_hat_g),
            "exact_match": sel["is_exact_match"],
            "near_match": abs(sel["j_hat_star"] - sel["j_true_star"]) <= 1,
            "excess_err": sel["excess_error"],
        })
    df_gauss = pd.DataFrame(gauss_records)
    gauss_summary = {
        "jl_eps20": float(df_gauss["jl_eps20"].mean()),
        "jl_eps50": float(df_gauss["jl_eps50"].mean()),
        "jl_std": float(df_gauss["jl_std"].mean()),
        "ward_success": float(df_gauss["ward_success"].mean()),
        "ward_fraction": float(df_gauss["ward_fraction"].mean()),
        "pre_rel": float(df_gauss["pre_rel_med"].median()),
        "tau": float(df_gauss["tau"].mean()),
        "conc": float(df_gauss["conc"].mean()),
        "acc": float(df_gauss["exact_match"].mean()),
        "near_acc": float(df_gauss["near_match"].mean()),
    }

    # Sweep over density q
    for q in q_values:
        print(f"Evaluating density q = {q:5.2f} (expected nonzeros/row: {q * N:5.1f} / {N})...")
        for t, (x_true, traj, error_vectors, sigma_n) in enumerate(problems):
            rng_val = np.random.default_rng(base_seed + int(q * 100000) + t)
            Psi = generate_projection_matrix(r, N, matrix_type="sparse_rademacher", q=q, rng=rng_val)
            y_cv, _, _ = add_noise_snr(Psi @ x_true, snr_db=snr_db, rng=rng_val)
            E_hat = compute_cv_errors(Psi, y_cv, traj.iterates)

            E_true = traj.true_errors
            sel = best_iterate_selection(E_true, E_hat)
            near_match = abs(sel["j_hat_star"] - sel["j_true_star"]) <= 1

            # Classical JL metrics on actual error vectors
            jl_stats = jl_isometry_stats(error_vectors, Psi, epsilons=(0.2, 0.5))

            # Rachel Ward (2009) theoretical cross-validation error bounds (4/5 <= ||e|| / ||Psi e|| <= 4/3)
            proj_error_sq = np.sum((error_vectors @ Psi.T) ** 2, axis=1)
            ward_res = ward_bound_stats(E_true, proj_error_sq)

            # Pre-convergence relative error (avoids noise floor dominating the median)
            pre_rel = pre_convergence_relative_errors(E_true, E_hat, sel["j_true_star"])

            # Full trajectory relative errors
            full_rel = relative_norm_errors(E_true, E_hat)

            tau = kendall_tau(E_true, E_hat)
            conc = pairwise_concordance(E_true, E_hat)

            all_trial_records.append({
                "trial": t,
                "q": q,
                "nnz_per_row": q * N,
                "jl_eps20_success": jl_stats["success_rates"][0.2],
                "jl_eps50_success": jl_stats["success_rates"][0.5],
                "jl_distortion_std": jl_stats["std_ratio"],
                "ward_success": ward_res["ward_simultaneous_success"],
                "ward_fraction": ward_res["ward_fraction_satisfied"],
                "pre_conv_rel_error": float(np.median(pre_rel)),
                "full_rel_error_med": float(np.median(full_rel)),
                "kendall_tau": tau,
                "pairwise_concordance": conc,
                "j_true_star": sel["j_true_star"],
                "j_hat_star": sel["j_hat_star"],
                "is_exact_match": sel["is_exact_match"],
                "is_near_match": near_match,
                "excess_error": sel["excess_error"],
            })

    df_trials = pd.DataFrame(all_trial_records)
    trials_csv_path = os.path.join(output_dir, "three_notions_trials.csv")
    df_trials.to_csv(trials_csv_path, index=False)

    # Compute summary by q
    summary_rows = []
    for q in q_values:
        sub = df_trials[df_trials["q"] == q]
        summary_rows.append({
            "q": q,
            "nnz_per_row": q * N,
            # JL Isometry
            "JL_Success_Eps20": float(sub["jl_eps20_success"].mean()),
            "JL_Success_Eps50": float(sub["jl_eps50_success"].mean()),
            "JL_Distortion_Std": float(sub["jl_distortion_std"].mean()),
            # Ward Bounds (4/5 <= ||e|| / ||Psi e|| <= 4/3)
            "Ward_Bound_Success": float(sub["ward_success"].mean()),
            "Ward_Bound_Fraction": float(sub["ward_fraction"].mean()),
            # Norm Estimation (Pre-convergence and Full)
            "PreConv_Rel_Error_Median": float(sub["pre_conv_rel_error"].median()),
            "PreConv_Rel_Error_IQR25": float(np.percentile(sub["pre_conv_rel_error"], 25)),
            "PreConv_Rel_Error_IQR75": float(np.percentile(sub["pre_conv_rel_error"], 75)),
            "Median_Rel_Error": float(sub["full_rel_error_med"].median()),
            # Ordering
            "Mean_Kendall_Tau": float(sub["kendall_tau"].mean()),
            "Median_Kendall_Tau": float(sub["kendall_tau"].median()),
            "Mean_Concordance": float(sub["pairwise_concordance"].mean()),
            # Selection
            "Selection_Accuracy": float(sub["is_exact_match"].mean()),
            "Near_Selection_Accuracy": float(sub["is_near_match"].mean()),
            "Mean_Excess_Error": float(sub["excess_error"].mean()),
            "Median_Excess_Error": float(sub["excess_error"].median()),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "three_notions_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 90)
    print("STAGE 3 REFINED SUMMARY TABLE:")
    print("=" * 90)
    display_cols = [
        "q", "nnz_per_row", "JL_Success_Eps20", "Ward_Bound_Success", "JL_Distortion_Std",
        "PreConv_Rel_Error_Median", "Mean_Kendall_Tau", "Selection_Accuracy", "Near_Selection_Accuracy"
    ]
    print(df_summary[display_cols].to_string(index=False))
    print(f"\n[Dense Gaussian Benchmark] -> JL(eps=0.2): {gauss_summary['jl_eps20']:.1%}, Std: {gauss_summary['jl_std']:.2f}, PreConv RelErr: {gauss_summary['pre_rel']:.2f}, Tau: {gauss_summary['tau']:.3f}, Acc: {gauss_summary['acc']:.1%}")
    print("=" * 90)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_jl, ax_std = axes[0, 0], axes[0, 1]
    ax_task, ax_hierarchy = axes[1, 0], axes[1, 1]

    # Panel 1: Classical JL Isometry Success Rate vs Sparsity q
    ax_jl.plot(df_summary["q"], df_summary["JL_Success_Eps20"] * 100, "o-", color="darkred", linewidth=2.2, label=r"JL Success ($\varepsilon = 0.20$)")
    ax_jl.plot(df_summary["q"], df_summary["JL_Success_Eps50"] * 100, "s--", color="orange", linewidth=2.0, label=r"JL Success ($\varepsilon = 0.50$)")
    ax_jl.axhline(gauss_summary["jl_eps20"] * 100, color="darkred", linestyle=":", alpha=0.6, label=f"Dense Gauss eps=0.2 ({gauss_summary['jl_eps20']*100:.1f}%)")
    ax_jl.set_xscale("log")
    ax_jl.set_xlabel("Matrix Density $q$ (log scale)", fontsize=11)
    ax_jl.set_ylabel("JL Success Rate (%)", fontsize=11)
    ax_jl.set_ylim(15, 105)
    ax_jl.set_title(r"Level 1: General JL Isometry $(1 \pm \varepsilon)$ Preservation", fontsize=12, fontweight="bold")
    ax_jl.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_jl.legend(fontsize=9)

    # Panel 2: Distortion Spread / Standard Deviation vs Sparsity q
    ax_std.plot(df_summary["q"], df_summary["JL_Distortion_Std"], "d-", color="crimson", linewidth=2.2, label=r"$\mathrm{Std}(\|\Psi \mathbf{e}\|_2^2 / \|\mathbf{e}\|_2^2)$")
    ax_std.axhline(gauss_summary["jl_std"], color="royalblue", linestyle="--", label=f"Dense Gauss Std ({gauss_summary['jl_std']:.2f})")
    ax_std.set_xscale("log")
    ax_std.set_xlabel("Matrix Density $q$ (log scale)", fontsize=11)
    ax_std.set_ylabel(r"Distortion Standard Deviation $\sigma$", fontsize=11)
    ax_std.set_title("Projection Tail Variance Expansion (Higher = Unstable)", fontsize=12, fontweight="bold")
    ax_std.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_std.legend(fontsize=9)

    # Panel 3: Pre-Convergence Relative Error vs Model Selection Accuracy
    ax_task.errorbar(
        df_summary["q"],
        df_summary["PreConv_Rel_Error_Median"],
        yerr=[
            df_summary["PreConv_Rel_Error_Median"] - df_summary["PreConv_Rel_Error_IQR25"],
            df_summary["PreConv_Rel_Error_IQR75"] - df_summary["PreConv_Rel_Error_Median"],
        ],
        fmt="^-",
        color="purple",
        linewidth=2.0,
        capsize=4,
        label=r"Pre-Convergence Rel Error ($j \leq j^*$)",
    )
    ax_task.set_xscale("log")
    ax_task.set_xlabel("Matrix Density $q$ (log scale)", fontsize=11)
    ax_task.set_ylabel(r"Relative Error $\frac{|\widehat{E}_j - E_j|}{E_j}$", fontsize=11, color="purple")
    ax_task.tick_params(axis="y", labelcolor="purple")
    ax_task.grid(True, which="both", linestyle="--", alpha=0.4)

    # Twin axis for selection accuracy
    ax_task2 = ax_task.twinx()
    ax_task2.plot(df_summary["q"], df_summary["Selection_Accuracy"] * 100, "o-", color="darkblue", linewidth=2.2, label=r"Selection Acc $P(\hat{j}=j^*)$")
    ax_task2.plot(df_summary["q"], df_summary["Near_Selection_Accuracy"] * 100, "d--", color="teal", linewidth=1.8, label=r"Near Match $P(|\hat{j}-j^*| \leq 1)$")
    ax_task2.set_ylabel("Selection Accuracy (%)", fontsize=11, color="darkblue")
    ax_task2.tick_params(axis="y", labelcolor="darkblue")
    ax_task2.set_ylim(20, 105)
    ax_task.set_title("Pre-Convergence Norm Error vs Model Selection", fontsize=12, fontweight="bold")

    # Combine legends for twin axes
    lines1, labels1 = ax_task.get_legend_handles_labels()
    lines2, labels2 = ax_task2.get_legend_handles_labels()
    ax_task.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, loc="center left")

    # Panel 4: The Hierarchy Comparison (Retention Relative to Dense Matrix)
    # Relative performance normalized to q=1.0
    base_jl = df_summary[df_summary["q"] == 1.0]["JL_Success_Eps20"].values[0]
    base_ward = df_summary[df_summary["q"] == 1.0]["Ward_Bound_Success"].values[0]
    base_pre_rel = df_summary[df_summary["q"] == 1.0]["PreConv_Rel_Error_Median"].values[0]
    base_tau = df_summary[df_summary["q"] == 1.0]["Mean_Kendall_Tau"].values[0]
    base_acc = df_summary[df_summary["q"] == 1.0]["Selection_Accuracy"].values[0]

    jl_retention = df_summary["JL_Success_Eps20"] / base_jl
    ward_retention = df_summary["Ward_Bound_Success"] / max(1e-6, base_ward)
    norm_retention = (1.0 / (1.0 + df_summary["PreConv_Rel_Error_Median"])) / (1.0 / (1.0 + base_pre_rel))
    tau_retention = df_summary["Mean_Kendall_Tau"] / base_tau
    acc_retention = df_summary["Selection_Accuracy"] / base_acc

    ax_hierarchy.plot(df_summary["q"], acc_retention, "o-", color="darkblue", linewidth=2.5, label="Level 4: Best-Iterate Selection")
    ax_hierarchy.plot(df_summary["q"], tau_retention, "s-", color="forestgreen", linewidth=2.2, label=r"Level 3: Kendall's $\tau$ (Ordering)")
    ax_hierarchy.plot(df_summary["q"], norm_retention, "^--", color="purple", linewidth=2.0, label="Level 2: Norm Estimation Retention")
    ax_hierarchy.plot(df_summary["q"], ward_retention, "v-", color="darkorange", linewidth=2.0, label="Level 2b: Ward Bound (4/5..4/3)")
    ax_hierarchy.plot(df_summary["q"], jl_retention, "x-.", color="darkred", linewidth=2.2, label=r"Level 1: General JL Isometry ($\varepsilon=0.2$)")

    ax_hierarchy.set_xscale("log")
    ax_hierarchy.set_xlabel("Matrix Density $q$ (log scale)", fontsize=11)
    ax_hierarchy.set_ylabel("Retention Factor Relative to Dense ($q=1.0$)", fontsize=11)
    ax_hierarchy.set_title("The Hierarchy of Validation Requirements", fontsize=12, fontweight="bold")
    ax_hierarchy.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_hierarchy.legend(fontsize=8.5, loc="lower left")

    fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.08, hspace=0.30, wspace=0.28)
    plot_path = os.path.join(output_dir, "three_notions_vs_sparsity.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"\nSaved refined figure to: {plot_path}")
    print(f"Saved summary metrics to: {summary_csv_path}")
    print(f"Saved trials metrics to: {trials_csv_path}\n")

    return df_summary, df_trials


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate refined three notions of works vs validation sparsity")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials per q")
    parser.add_argument("--N", type=int, default=512, help="Signal dimension")
    parser.add_argument("--s", type=int, default=10, help="Sparsity")
    parser.add_argument("--m", type=int, default=120, help="Sensing measurements")
    parser.add_argument("--r", type=int, default=40, help="Validation measurements")
    parser.add_argument("--snr", type=float, default=25.0, help="SNR in dB")
    parser.add_argument("--max_iter", type=int, default=25, help="OMP iterations")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed")
    args = parser.parse_args()

    evaluate_three_notions(
        num_trials=args.trials,
        N=args.N,
        s=args.s,
        m=args.m,
        r=args.r,
        snr_db=args.snr,
        max_iter=args.max_iter,
        base_seed=args.seed,
    )
