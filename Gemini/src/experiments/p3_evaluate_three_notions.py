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
        q_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]

    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))
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
            "jl_eps20": jl_stats["simultaneous_rates"][0.2],
            "jl_eps50": jl_stats["simultaneous_rates"][0.5],
            "jl_eps20_frac": jl_stats["fractional_rates"][0.2],
            "jl_eps50_frac": jl_stats["fractional_rates"][0.5],
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
        "jl_eps20_frac": float(df_gauss["jl_eps20_frac"].mean()),
        "jl_eps50_frac": float(df_gauss["jl_eps50_frac"].mean()),
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
                "jl_eps20_simultaneous": jl_stats["simultaneous_rates"][0.2],
                "jl_eps50_simultaneous": jl_stats["simultaneous_rates"][0.5],
                "jl_eps20_fraction": jl_stats["fractional_rates"][0.2],
                "jl_eps50_fraction": jl_stats["fractional_rates"][0.5],
                "jl_distortion_std": jl_stats["std_ratio"],
                "ward_simultaneous": ward_res["ward_simultaneous_success"],
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
    # trials_csv_path = os.path.join(output_dir, "three_notions_trials.csv")
    # df_trials.to_csv(trials_csv_path, index=False)

    # Compute summary by q
    summary_rows = []
    for q in q_values:
        sub = df_trials[df_trials["q"] == q]
        summary_rows.append({
            "q": q,
            "nnz_per_row": q * N,
            # JL Isometry (Both Simultaneous Trajectory & Fractional Iterates)
            "JL_Simultaneous_Eps20": float(sub["jl_eps20_simultaneous"].mean()),
            "JL_Simultaneous_Eps50": float(sub["jl_eps50_simultaneous"].mean()),
            "JL_Fraction_Eps20": float(sub["jl_eps20_fraction"].mean()),
            "JL_Fraction_Eps50": float(sub["jl_eps50_fraction"].mean()),
            "JL_Success_Eps20": float(sub["jl_eps20_simultaneous"].mean()),
            "JL_Success_Eps50": float(sub["jl_eps50_simultaneous"].mean()),
            "JL_Distortion_Std": float(sub["jl_distortion_std"].mean()),
            # Ward Bounds (Both Simultaneous & Fractional)
            "Ward_Bound_Success": float(sub["ward_simultaneous"].mean()),
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
    # summary_csv_path = os.path.join(output_dir, "three_notions_summary.csv")
    # df_summary.to_csv(summary_csv_path, index=False)

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

    # -------------------------------------------------------------------------
    # Visualization: Save individual standalone PNGs matching presentation slides
    # -------------------------------------------------------------------------

    # 1. Graph 3a: Measure 1 - Norm Preservation (Simultaneous vs Fractional)
    fig1, ax1 = plt.subplots(figsize=(9.2, 6.0))
    # Simultaneous Trajectory Success (All Iterates In Bounds) - Solid Lines
    # Fractional Iterates In Bounds (Average % of Iterates Preserved) - Dashed Lines

    ax1.plot(df_summary["q"], df_summary["JL_Simultaneous_Eps50"] * 100, "s-", color="darkorange", linewidth=2.2, label=r"Simultaneous JL ($\varepsilon=0.50$, all iterates preserved)")
    ax1.plot(df_summary["q"], df_summary["JL_Fraction_Eps50"] * 100, "s--", color="darkorange", alpha=0.75, linewidth=1.8, label=r"Fraction of Iterates in $[1 \pm 0.50]$ (mean %)")
    ax1.plot(df_summary["q"], df_summary["JL_Simultaneous_Eps20"] * 100, "o-", color="darkred", linewidth=2.2, label=r"Simultaneous JL ($\varepsilon=0.20$, all iterates preserved)")
    ax1.plot(df_summary["q"], df_summary["JL_Fraction_Eps20"] * 100, "o--", color="darkred", alpha=0.75, linewidth=1.8, label=r"Fraction of Iterates in $[1 \pm 0.20]$ (mean %)")
    ax1.plot(df_summary["q"], df_summary["Ward_Bound_Success"] * 100, "v-", color="purple", linewidth=2.2, label=r"Simultaneous Ward Bound ($4/5 \leq \mathrm{ratio} \leq 4/3$)")
    ax1.plot(df_summary["q"], df_summary["Ward_Bound_Fraction"] * 100, "v--", color="purple", alpha=0.75, linewidth=1.8, label=r"Fraction of Iterates in Ward Bounds (mean %)")

    ax1.set_xscale("log")
    ax1.set_xlabel(r"Matrix Density $q$ (probability of non-zero entry, log scale)", fontsize=11)
    ax1.set_ylabel(r"Rate / Percentage (%)", fontsize=11)
    ax1.set_ylim(-2, 105)
    ax1.set_title(r"Measure 1: Error Norm Preservation ($\varepsilon_{\mathrm{cv}}^n \approx \varepsilon_x^n$)" + f"  [Validation Budget $r={r}$]\n" + r"Solid = Simultaneous Trajectory Guarantee ($\forall n$)  |  Dashed = Mean Fraction of Iterates Preserved", fontsize=11, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.02, 0.42), framealpha=0.9)
    fig1.tight_layout()
    p1_path = os.path.join(output_dir, "p3_1_norm_preservation.png")
    fig1.savefig(p1_path, dpi=200)
    plt.close(fig1)

    # 2. Graph 3b: Measure 2 - Ordering Preservation (Kendall's tau & Concordance)
    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    ax2.plot(df_summary["q"], df_summary["Mean_Kendall_Tau"], "s-", color="forestgreen", linewidth=2.2, label=r"Kendall's $\tau$ (Rank Correlation)")
    ax2.plot(df_summary["q"], df_summary["Mean_Concordance"], "^--", color="teal", linewidth=2.0, label=r"Pairwise Concordance $\mathbb{P}(\varepsilon_{\mathrm{cv}}^p < \varepsilon_{\mathrm{cv}}^q \mid \varepsilon_x^p < \varepsilon_x^q)$")
    ax2.axhline(gauss_summary["tau"], color="forestgreen", linestyle=":", alpha=0.7, label=f"Dense Gaussian Kendall's $\\tau$ ({gauss_summary['tau']:.3f})")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Matrix Density $q$ (probability of non-zero entry, log scale)", fontsize=11)
    ax2.set_ylabel(r"Correlation / Concordance", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title(r"Measure 2: Ordering Preservation vs Matrix Density $q$" + f"  [Validation Budget $r={r}$]\n" + r"Does $\varepsilon_x^p < \varepsilon_x^q \Rightarrow \varepsilon_{\mathrm{cv}}^p < \varepsilon_{\mathrm{cv}}^q$?", fontsize=12, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9.5, loc="lower right")
    fig2.tight_layout()
    p2_path = os.path.join(output_dir, "p3_2_ordering_preservation.png")
    fig2.savefig(p2_path, dpi=200)
    plt.close(fig2)

    # 3. Graph 3c: Measures 3 & 4 - Model Selection Accuracy P(n^ = n*)
    fig3, ax3 = plt.subplots(figsize=(8, 5.5))
    ax3.plot(df_summary["q"], df_summary["Selection_Accuracy"] * 100, "o-", color="darkblue", linewidth=2.2, label=r"Exact Selection $\mathbb{P}(\hat{n} = n^*)$")
    ax3.plot(df_summary["q"], df_summary["Near_Selection_Accuracy"] * 100, "d--", color="royalblue", linewidth=2.0, label=r"Near Selection $\mathbb{P}(|\hat{n} - n^*| \leq 1)$")
    ax3.axhline(gauss_summary["acc"] * 100, color="darkblue", linestyle=":", alpha=0.7, label=f"Dense Gaussian Exact Match ({gauss_summary['acc']*100:.1f}%)")
    ax3.set_xscale("log")
    ax3.set_xlabel(r"Matrix Density $q$ (probability of non-zero entry, log scale)", fontsize=11)
    ax3.set_ylabel(r"Selection Accuracy (%)", fontsize=11)
    ax3.set_ylim(-2, 105)
    ax3.set_title(r"Measures 3 & 4: Best-Iterate Selection Accuracy vs Density $q$" + f"  [Validation Budget $r={r}$]\n" + r"Oracle Minimizer $n^*$ vs Cross-Validation Minimizer $\hat{n}$", fontsize=12, fontweight="bold")
    ax3.grid(True, which="both", linestyle="--", alpha=0.4)
    ax3.legend(fontsize=9.5, loc="lower right")
    fig3.tight_layout()
    p3_path = os.path.join(output_dir, "p3_3_selection_accuracy.png")
    fig3.savefig(p3_path, dpi=200)
    plt.close(fig3)

    # 4. Graph 3d: The Hierarchy of Validation Requirements (Absolute Performance Rates)
    fig4, ax4 = plt.subplots(figsize=(8.8, 5.8))
    ax4.plot(df_summary["q"], df_summary["Selection_Accuracy"] * 100, "o-", color="darkblue", linewidth=2.5, label=r"Level 4: Best-Iterate Selection $\mathbb{P}(\hat{n} = n^*)$")
    ax4.plot(df_summary["q"], df_summary["Near_Selection_Accuracy"] * 100, "d--", color="royalblue", linewidth=2.0, label=r"Level 4b: Near Selection $\mathbb{P}(|\hat{n} - n^*| \leq 1)$")
    ax4.plot(df_summary["q"], df_summary["Mean_Kendall_Tau"] * 100, "s-", color="forestgreen", linewidth=2.2, label=r"Level 3: Kendall's $\tau$ (Rank Correlation $\times 100$)")
    ax4.plot(df_summary["q"], df_summary["Ward_Bound_Success"] * 100, "v-", color="purple", linewidth=2.0, label=r"Level 2: Simultaneous Ward Bound ($4/5 \leq \mathrm{ratio} \leq 4/3$)")
    ax4.plot(df_summary["q"], df_summary["JL_Simultaneous_Eps50"] * 100, "s-", color="darkorange", linewidth=2.0, label=r"Level 1: Simultaneous JL ($\varepsilon=0.50$, all iterates)")
    ax4.plot(df_summary["q"], df_summary["JL_Fraction_Eps50"] * 100, "s--", color="darkorange", alpha=0.7, linewidth=1.8, label=r"Level 1b: Fraction of Preserved Iterates ($\varepsilon=0.50$)")
    ax4.set_xscale("log")
    ax4.set_xlabel(r"Matrix Density $q$ (probability of non-zero entry, log scale)", fontsize=11)
    ax4.set_ylabel(r"Absolute Success / Quality Rate (%)", fontsize=11)
    ax4.set_ylim(-2, 105)
    ax4.set_title(f"The Hierarchy of Validation Requirements  [Validation Budget $r={r}$]\nSelection Survives Even When Simultaneous JL Isometry Collapses", fontsize=12, fontweight="bold")
    ax4.grid(True, which="both", linestyle="--", alpha=0.4)
    ax4.legend(fontsize=8.5, loc="lower left")
    fig4.tight_layout()
    p4_path = os.path.join(output_dir, "p3_4_hierarchy.png")
    fig4.savefig(p4_path, dpi=200)
    plt.close(fig4)

    # 5. Combined Multi-Panel Overview (without projection variance)
    fig_all, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Top-Left: Measure 1 Norm Preservation (Simultaneous & Fractional)
    axes[0, 0].plot(df_summary["q"], df_summary["JL_Simultaneous_Eps50"] * 100, "s-", color="darkorange", linewidth=2.0, label=r"Simul. JL ($\varepsilon=0.50$)")
    axes[0, 0].plot(df_summary["q"], df_summary["JL_Fraction_Eps50"] * 100, "s--", color="darkorange", alpha=0.7, linewidth=1.5, label=r"Frac. in $[1 \pm 0.50]$")
    axes[0, 0].plot(df_summary["q"], df_summary["JL_Simultaneous_Eps20"] * 100, "o-", color="darkred", linewidth=2.0, label=r"Simul. JL ($\varepsilon=0.20$)")
    axes[0, 0].plot(df_summary["q"], df_summary["JL_Fraction_Eps20"] * 100, "o--", color="darkred", alpha=0.7, linewidth=1.5, label=r"Frac. in $[1 \pm 0.20]$")
    axes[0, 0].plot(df_summary["q"], df_summary["Ward_Bound_Success"] * 100, "v-", color="purple", linewidth=2.0, label=r"Simul. Ward ($4/5..4/3$)")
    axes[0, 0].plot(df_summary["q"], df_summary["Ward_Bound_Fraction"] * 100, "v--", color="purple", alpha=0.7, linewidth=1.5, label=r"Frac. in Ward Bounds")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel(r"Matrix Density $q$", fontsize=10)
    axes[0, 0].set_ylabel("Rate / Percentage (%)", fontsize=10)
    axes[0, 0].set_ylim(-2, 105)
    axes[0, 0].set_title(r"1. Error Norm Preservation (Solid=Simul., Dashed=Frac.)", fontsize=11, fontweight="bold")
    axes[0, 0].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[0, 0].legend(fontsize=7.5, loc="upper left")

    # Top-Right: Measure 2 Ordering Preservation
    axes[0, 1].plot(df_summary["q"], df_summary["Mean_Kendall_Tau"], "s-", color="forestgreen", linewidth=2.2, label=r"Kendall's $\tau$")
    axes[0, 1].plot(df_summary["q"], df_summary["Mean_Concordance"], "^--", color="teal", linewidth=2.0, label="Concordance")
    axes[0, 1].axhline(gauss_summary["tau"], color="forestgreen", linestyle=":", alpha=0.6, label="Dense Gauss")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel(r"Matrix Density $q$", fontsize=10)
    axes[0, 1].set_ylabel(r"$\tau$ / Concordance", fontsize=10)
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].set_title(r"2. Ordering Preservation (Kendall's $\tau$)", fontsize=11, fontweight="bold")
    axes[0, 1].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[0, 1].legend(fontsize=8.5)

    # Bottom-Left: Measures 3 & 4 Selection Accuracy
    axes[1, 0].plot(df_summary["q"], df_summary["Selection_Accuracy"] * 100, "o-", color="darkblue", linewidth=2.2, label=r"Exact $\mathbb{P}(\hat{n} = n^*)$")
    axes[1, 0].plot(df_summary["q"], df_summary["Near_Selection_Accuracy"] * 100, "d--", color="royalblue", linewidth=2.0, label=r"Near $\mathbb{P}(|\hat{n} - n^*| \leq 1)$")
    axes[1, 0].axhline(gauss_summary["acc"] * 100, color="darkblue", linestyle=":", alpha=0.6, label="Dense Gauss")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel(r"Matrix Density $q$", fontsize=10)
    axes[1, 0].set_ylabel("Selection Accuracy (%)", fontsize=10)
    axes[1, 0].set_ylim(-2, 105)
    axes[1, 0].set_title(r"3 & 4. Selection Accuracy $\mathbb{P}(\hat{n} = n^*)$", fontsize=11, fontweight="bold")
    axes[1, 0].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[1, 0].legend(fontsize=8.5)

    # Bottom-Right: The Hierarchy (Absolute Success Rates)
    axes[1, 1].plot(df_summary["q"], df_summary["Selection_Accuracy"] * 100, "o-", color="darkblue", linewidth=2.2, label=r"Level 4: Selection $\mathbb{P}(\hat{n}=n^*)$")
    axes[1, 1].plot(df_summary["q"], df_summary["Near_Selection_Accuracy"] * 100, "d--", color="royalblue", linewidth=1.8, label=r"Level 4b: Near Selection")
    axes[1, 1].plot(df_summary["q"], df_summary["Mean_Kendall_Tau"] * 100, "s-", color="forestgreen", linewidth=2.0, label=r"Level 3: Kendall's $\tau \times 100$")
    axes[1, 1].plot(df_summary["q"], df_summary["Ward_Bound_Success"] * 100, "v-", color="purple", linewidth=1.8, label=r"Level 2: Simul. Ward ($4/5..4/3$)")
    axes[1, 1].plot(df_summary["q"], df_summary["JL_Simultaneous_Eps50"] * 100, "s-", color="darkorange", linewidth=2.0, label=r"Level 1b: Simul. JL ($\varepsilon=0.5$)")
    axes[1, 1].plot(df_summary["q"], df_summary["JL_Fraction_Eps50"] * 100, "s--", color="darkorange", alpha=0.7, linewidth=1.6, label=r"Level 1: Frac. Preserved ($\varepsilon=0.5$)")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel(r"Matrix Density $q$", fontsize=10)
    axes[1, 1].set_ylabel("Absolute Rate (%)", fontsize=10)
    axes[1, 1].set_ylim(-2, 105)
    axes[1, 1].set_title(f"Hierarchy of Requirements ($r={r}$)", fontsize=11, fontweight="bold")
    axes[1, 1].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[1, 1].legend(fontsize=7.5, loc="upper left")

    fig_all.suptitle(f"Measures of Validation Success vs Matrix Density $q$ (Validation Budget $r={r}$)", fontsize=13, fontweight="bold")
    fig_all.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.08, hspace=0.30, wspace=0.28)
    combined_path = os.path.join(output_dir, "p3_three_notions_vs_sparsity.png")
    fig_all.savefig(combined_path, dpi=200)
    plt.close(fig_all)

    print(f"\nSaved individual graphs to:")
    print(f"  - {p1_path}")
    print(f"  - {p2_path}")
    print(f"  - {p3_path}")
    print(f"  - {p4_path}")
    print(f"Saved combined overview to: {combined_path}\n")

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
