"""
Stage 5 - Budget-Matched Comparisons: Fixed r vs Fixed Nonzeros (nnz)
Investigates the core practical engineering trade-off:
  1. Fixed Measurement Budget (r fixed):
     Measures performance loss when thinning matrix entries at constant measurement count r.
  2. Fixed Computational / Nonzero Budget (nnz fixed):
     Tests whether spending a fixed FLOP/hardware budget B on a "wide-and-sparse" matrix
     (r_sparse = B / (q*N)) is superior to a "narrow-and-dense" matrix (r_dense = B / N).

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
    compute_projection_stats,
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


def run_budget_experiments(
    num_trials: int = 100,
    N: int = 512,
    s: int = 10,
    m: int = 120,
    snr_db: float = 25.0,
    max_iter: int = 25,
    base_seed: int = 5050,
    output_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run both Fixed-r and Fixed-nnz budget matched comparisons.
    """
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 85)
    print("STAGE 5: BUDGET-MATCHED EXPERIMENTS (FIXED r vs FIXED NONZEROS)")
    print(f"Problem: N={N}, s={s}, m={m}, SNR={snr_db} dB, max_iter={max_iter}")
    print(f"Trials per setting: {num_trials}")
    print("=" * 85)

    p = min(max_iter, m, N)

    # 1. Pre-generate problem ensemble
    print("Generating problem instances...")
    problems = []
    for t in range(num_trials):
        rng = np.random.default_rng(base_seed + t)
        x_true, support = generate_sparse_signal(N, s, coeff_dist="gaussian", rng=rng)
        Phi = generate_sensing_matrix(m, N, dist="gaussian", normalize="column", rng=rng)
        y_clean = Phi @ x_true
        y, _, sigma_n = add_noise_snr(y_clean, snr_db=snr_db, rng=rng)
        traj = run_omp_trajectory(Phi, y, max_iter=p, x_true=x_true)
        error_vectors = x_true[None, :] - traj.iterates
        problems.append((x_true, traj, error_vectors, sigma_n))

    # =========================================================================
    # PART A: FIXED MEASUREMENTS r (Thinning entries at constant r)
    # =========================================================================
    r_fixed_grid = [10, 20, 30, 40, 60, 80, 100]
    q_grid = [1.0, 0.5, 0.2, 0.1, 0.05]

    print("\n[Part A] Running Fixed-r Experiments...")
    fixed_r_records = []
    for r in r_fixed_grid:
        for q in q_grid:
            m_type = "dense_rademacher" if q == 1.0 else "sparse_rademacher"
            for t, (x_true, traj, error_vectors, sigma_n) in enumerate(problems):
                rng_val = np.random.default_rng(base_seed + 10000 * r + int(q * 1000) + t)
                Psi = generate_projection_matrix(r, N, matrix_type=m_type, q=q, rng=rng_val)
                y_cv, _, _ = add_noise_snr(Psi @ x_true, snr_db=snr_db, rng=rng_val)
                E_hat = compute_cv_errors(Psi, y_cv, traj.iterates)

                E_true = traj.true_errors
                sel = best_iterate_selection(E_true, E_hat)
                tau = kendall_tau(E_true, E_hat)
                conc = pairwise_concordance(E_true, E_hat)
                pre_rel = pre_convergence_relative_errors(E_true, E_hat, sel["j_true_star"])

                fixed_r_records.append({
                    "trial": t,
                    "r": r,
                    "q": q,
                    "expected_nnz": int(round(r * q * N)),
                    "selection_match": sel["is_exact_match"],
                    "near_match": abs(sel["j_hat_star"] - sel["j_true_star"]) <= 1,
                    "kendall_tau": tau,
                    "concordance": conc,
                    "pre_rel_error": float(np.median(pre_rel)),
                })

    df_fixed_r_trials = pd.DataFrame(fixed_r_records)

    # Summary table for Fixed r
    summary_r_list = []
    for r in r_fixed_grid:
        for q in q_grid:
            sub = df_fixed_r_trials[(df_fixed_r_trials["r"] == r) & (df_fixed_r_trials["q"] == q)]
            summary_r_list.append({
                "r": r,
                "q": q,
                "expected_nnz": int(round(r * q * N)),
                "Selection_Accuracy": float(sub["selection_match"].mean()),
                "Near_Selection_Accuracy": float(sub["near_match"].mean()),
                "Mean_Kendall_Tau": float(sub["kendall_tau"].mean()),
                "Mean_Concordance": float(sub["concordance"].mean()),
                "PreConv_Rel_Error": float(sub["pre_rel_error"].median()),
            })
    df_fixed_r_summary = pd.DataFrame(summary_r_list)

    # =========================================================================
    # PART B: FIXED NONZEROS BUDGET B (Iso-Compute / Iso-FLOPs)
    # =========================================================================
    # Choose budgets B corresponding to equivalent dense row counts:
    # B = b_dense * N where b_dense in [10, 20, 30, 40, 60, 80]
    b_dense_grid = [10, 15, 20, 30, 40, 60, 80]
    fixed_nnz_budgets = [b * N for b in b_dense_grid]

    print("\n[Part B] Running Fixed-nnz Budget Matched Experiments...")
    fixed_nnz_records = []
    for B in fixed_nnz_budgets:
        b_dense = B // N
        for q in q_grid:
            # Calculate row count for this density to match budget B:
            # r_sparse * q * N = B => r_sparse = B / (q * N) = b_dense / q
            r_matched = max(2, int(round(B / (q * N))))
            m_type = "dense_rademacher" if q == 1.0 else "sparse_rademacher"

            for t, (x_true, traj, error_vectors, sigma_n) in enumerate(problems):
                rng_val = np.random.default_rng(base_seed + 200000 + B + int(q * 1000) + t)
                Psi = generate_projection_matrix(r_matched, N, matrix_type=m_type, q=q, rng=rng_val)
                y_cv, _, _ = add_noise_snr(Psi @ x_true, snr_db=snr_db, rng=rng_val)
                E_hat = compute_cv_errors(Psi, y_cv, traj.iterates)

                E_true = traj.true_errors
                sel = best_iterate_selection(E_true, E_hat)
                tau = kendall_tau(E_true, E_hat)
                conc = pairwise_concordance(E_true, E_hat)
                pre_rel = pre_convergence_relative_errors(E_true, E_hat, sel["j_true_star"])

                fixed_nnz_records.append({
                    "trial": t,
                    "budget_nnz": B,
                    "dense_equiv_rows": b_dense,
                    "q": q,
                    "r_matched": r_matched,
                    "selection_match": sel["is_exact_match"],
                    "near_match": abs(sel["j_hat_star"] - sel["j_true_star"]) <= 1,
                    "kendall_tau": tau,
                    "concordance": conc,
                    "pre_rel_error": float(np.median(pre_rel)),
                })

    df_fixed_nnz_trials = pd.DataFrame(fixed_nnz_records)

    summary_nnz_list = []
    for B in fixed_nnz_budgets:
        for q in q_grid:
            sub = df_fixed_nnz_trials[(df_fixed_nnz_trials["budget_nnz"] == B) & (df_fixed_nnz_trials["q"] == q)]
            r_val = sub["r_matched"].iloc[0]
            summary_nnz_list.append({
                "budget_nnz": B,
                "dense_equiv_rows": B // N,
                "q": q,
                "r_matched": r_val,
                "Selection_Accuracy": float(sub["selection_match"].mean()),
                "Near_Selection_Accuracy": float(sub["near_match"].mean()),
                "Mean_Kendall_Tau": float(sub["kendall_tau"].mean()),
                "Mean_Concordance": float(sub["concordance"].mean()),
                "PreConv_Rel_Error": float(sub["pre_rel_error"].median()),
            })
    df_fixed_nnz_summary = pd.DataFrame(summary_nnz_list)

    # Save CSVs
    fixed_r_csv = os.path.join(output_dir, "budget_fixed_r_summary.csv")
    fixed_nnz_csv = os.path.join(output_dir, "budget_fixed_nnz_summary.csv")
    df_fixed_r_summary.to_csv(fixed_r_csv, index=False)
    df_fixed_nnz_summary.to_csv(fixed_nnz_csv, index=False)

    print("\n" + "=" * 90)
    print("STAGE 5 SUMMARY: FIXED-NNZ COMPARISON (IS A WIDE-AND-SPARSE MATRIX BETTER?)")
    print("=" * 90)
    pivot_nnz_acc = df_fixed_nnz_summary.pivot(index="budget_nnz", columns="q", values="Selection_Accuracy") * 100
    pivot_nnz_tau = df_fixed_nnz_summary.pivot(index="budget_nnz", columns="q", values="Mean_Kendall_Tau")
    print("\n--- Selection Accuracy P(j_hat == j*) (%) vs Total Nonzero Budget B ---")
    print(pivot_nnz_acc.round(1).to_string())
    print("\n--- Kendall's Tau vs Total Nonzero Budget B ---")
    print(pivot_nnz_tau.round(3).to_string())
    print("=" * 90)

    # Multi-panel visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_r_acc, ax_nnz_acc = axes[0, 0], axes[0, 1]
    ax_r_tau, ax_nnz_tau = axes[1, 0], axes[1, 1]

    colors = {
        1.0: ("black", "-", "o", "Dense (q=1.0)"),
        0.5: ("royalblue", "--", "s", "Sparse (q=0.5)"),
        0.2: ("forestgreen", "-.", "^", "Sparse (q=0.2)"),
        0.1: ("crimson", ":", "d", "Sparse (q=0.1)"),
        0.05: ("purple", "-", "x", "Sparse (q=0.05)"),
    }

    # Panel 1: Fixed r -> Selection Accuracy vs r
    for q in q_grid:
        c, ls, mk, label = colors[q]
        sub = df_fixed_r_summary[df_fixed_r_summary["q"] == q]
        ax_r_acc.plot(sub["r"], sub["Selection_Accuracy"] * 100, color=c, linestyle=ls, marker=mk, linewidth=2.0, label=label)
    ax_r_acc.set_xlabel("Validation Measurements $r$", fontsize=11)
    ax_r_acc.set_ylabel("Selection Accuracy $P(\\hat{j}=j^*)$ (%)", fontsize=11)
    ax_r_acc.set_ylim(20, 100)
    ax_r_acc.set_title("Fixed Measurements $r$: Selection Accuracy vs $r$", fontsize=12, fontweight="bold")
    ax_r_acc.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_r_acc.legend(fontsize=9, loc="lower right")

    # Panel 2: Fixed nnz -> Selection Accuracy vs Nonzero Budget B
    for q in q_grid:
        c, ls, mk, label = colors[q]
        sub = df_fixed_nnz_summary[df_fixed_nnz_summary["q"] == q]
        ax_nnz_acc.plot(sub["budget_nnz"], sub["Selection_Accuracy"] * 100, color=c, linestyle=ls, marker=mk, linewidth=2.0, label=label)
    ax_nnz_acc.set_xlabel("Total Non-Zero Entries $\\mathrm{nnz}(\\Psi)$ (Budget $B$)", fontsize=11)
    ax_nnz_acc.set_ylabel("Selection Accuracy $P(\\hat{j}=j^*)$ (%)", fontsize=11)
    ax_nnz_acc.set_ylim(20, 100)
    ax_nnz_acc.set_title("Fixed Nonzeros: Selection Accuracy vs Budget $\\mathrm{nnz}(\\Psi)$", fontsize=12, fontweight="bold")
    ax_nnz_acc.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_nnz_acc.legend(fontsize=9, loc="lower right")

    # Panel 3: Fixed r -> Kendall's Tau vs r
    for q in q_grid:
        c, ls, mk, label = colors[q]
        sub = df_fixed_r_summary[df_fixed_r_summary["q"] == q]
        ax_r_tau.plot(sub["r"], sub["Mean_Kendall_Tau"], color=c, linestyle=ls, marker=mk, linewidth=2.0, label=label)
    ax_r_tau.set_xlabel("Validation Measurements $r$", fontsize=11)
    ax_r_tau.set_ylabel("Mean Kendall's $\\tau$", fontsize=11)
    ax_r_tau.set_ylim(0.70, 0.98)
    ax_r_tau.set_title("Fixed Measurements $r$: Ordering (Kendall's $\\tau$) vs $r$", fontsize=12, fontweight="bold")
    ax_r_tau.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_r_tau.legend(fontsize=9, loc="lower right")

    # Panel 4: Fixed nnz -> Kendall's Tau vs Nonzero Budget B
    for q in q_grid:
        c, ls, mk, label = colors[q]
        sub = df_fixed_nnz_summary[df_fixed_nnz_summary["q"] == q]
        ax_nnz_tau.plot(sub["budget_nnz"], sub["Mean_Kendall_Tau"], color=c, linestyle=ls, marker=mk, linewidth=2.0, label=label)
    ax_nnz_tau.set_xlabel("Total Non-Zero Entries $\\mathrm{nnz}(\\Psi)$ (Budget $B$)", fontsize=11)
    ax_nnz_tau.set_ylabel("Mean Kendall's $\\tau$", fontsize=11)
    ax_nnz_tau.set_ylim(0.70, 0.98)
    ax_nnz_tau.set_title("Fixed Nonzeros: Ordering (Kendall's $\\tau$) vs Budget $\\mathrm{nnz}(\\Psi)$", fontsize=12, fontweight="bold")
    ax_nnz_tau.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_nnz_tau.legend(fontsize=9, loc="lower right")

    fig.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.08, hspace=0.30, wspace=0.25)
    plot_path = os.path.join(output_dir, "budget_matched_comparison.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"\nSaved budget-matched visualization to: {plot_path}")
    print(f"Saved Fixed-r summary to: {fixed_r_csv}")
    print(f"Saved Fixed-nnz summary to: {fixed_nnz_csv}\n")

    return df_fixed_r_summary, df_fixed_nnz_summary, df_fixed_nnz_trials


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run budget matched cross validation experiments")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials per configuration")
    parser.add_argument("--N", type=int, default=512, help="Signal dimension")
    parser.add_argument("--s", type=int, default=10, help="Sparsity")
    parser.add_argument("--m", type=int, default=120, help="Sensing measurements")
    parser.add_argument("--snr", type=float, default=25.0, help="SNR in dB")
    parser.add_argument("--max_iter", type=int, default=25, help="OMP iterations")
    parser.add_argument("--seed", type=int, default=5050, help="Base random seed")
    args = parser.parse_args()

    run_budget_experiments(
        num_trials=args.trials,
        N=args.N,
        s=args.s,
        m=args.m,
        snr_db=args.snr,
        max_iter=args.max_iter,
        base_seed=args.seed,
    )
