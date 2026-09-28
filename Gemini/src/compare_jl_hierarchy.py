"""
Stage 6 - General JL Requirement vs Task-Specific Hierarchy
Evaluates the hierarchy of guarantees on the actual finite point set of OMP error vectors:
  P = {0, e^1, ..., e^p}, where e^j = x - x^j and ||e^j||_0 <= s + j.

Hierarchy Levels Tested:
  Level 1 (General JL): All pairs (u, v) in P satisfy (1 +/- eps) simultaneously.
  Level 2 (Norm Preservation): All error norms ||e^j|| satisfy (1 +/- eps) simultaneously.
  Level 3 (CV Ordering): Kendall's tau >= 0.90 across iterates.
  Level 4 (Task Objective): Best iterate selection argmin(E_hat) == argmin(E_true).

Also investigates how distortion correlates with error vector sparsity ||e^j||_0 = s + j.

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
    kendall_tau,
    pairwise_concordance,
    best_iterate_selection,
    ward_bound_stats,
)


def compute_all_pairs_jl(
    error_vectors: np.ndarray,
    Psi: np.ndarray,
    epsilon: float = 0.20,
) -> dict:
    """
    Test classical JL guarantee on the point set P = {0, e^1, ..., e^p}.

    Parameters:
    -----------
    error_vectors : np.ndarray of shape (p + 1, N)
        Includes e^0 = x - x^0 = x, down to e^p = x - x^p.
    Psi : np.ndarray of shape (r, N)
        Validation projection matrix.
    epsilon : float
        JL distortion tolerance parameter (default 0.20, i.e. 20%).

    Returns:
    --------
    dict containing:
        - "all_pairs_simultaneous_success": bool (True if ALL pairs in P satisfy 1 +/- eps)
        - "all_pairs_fraction_satisfied": float (fraction of pairs satisfying 1 +/- eps)
        - "norm_simultaneous_success": bool (True if ALL error norms satisfy 1 +/- eps)
        - "norm_fraction_satisfied": float (fraction of error norms satisfying 1 +/- eps)
        - "max_pair_distortion": float
        - "median_pair_distortion": float
    """
    # P has points: 0, e^0, e^1, ..., e^p
    # Total points K = p + 2
    N = error_vectors.shape[1]
    origin = np.zeros((1, N), dtype=float)
    P = np.vstack([origin, error_vectors])
    K = len(P)

    # Compute all pairwise differences
    # Pairs (i, k) for 0 <= i < k < K
    diff_vectors = []
    true_sq_dists = []
    is_norm_pair = []  # True if i == 0 (distance to origin)

    for i in range(K):
        for k in range(i + 1, K):
            diff = P[i] - P[k]
            d_sq = float(np.sum(diff ** 2))
            if d_sq > 1e-12:  # Ignore near-zero distances
                diff_vectors.append(diff)
                true_sq_dists.append(d_sq)
                is_norm_pair.append(bool(i == 0))

    if len(diff_vectors) == 0:
        return {
            "all_pairs_simultaneous_success": True,
            "all_pairs_fraction_satisfied": 1.0,
            "norm_simultaneous_success": True,
            "norm_fraction_satisfied": 1.0,
            "max_pair_distortion": 0.0,
            "median_pair_distortion": 0.0,
        }

    diff_arr = np.array(diff_vectors)
    true_dists_arr = np.array(true_sq_dists)
    is_norm_arr = np.array(is_norm_pair)

    # Project all difference vectors through Psi
    proj_diffs = diff_arr @ Psi.T
    proj_dists_arr = np.sum(proj_diffs ** 2, axis=1)

    ratios = proj_dists_arr / true_dists_arr
    distortions = np.abs(ratios - 1.0)

    in_bounds = (ratios >= (1.0 - epsilon)) & (ratios <= (1.0 + epsilon))

    all_pairs_simultaneous = bool(np.all(in_bounds))
    all_pairs_fraction = float(np.mean(in_bounds))

    norm_in_bounds = in_bounds[is_norm_arr]
    norm_simultaneous = bool(np.all(norm_in_bounds)) if len(norm_in_bounds) > 0 else True
    norm_fraction = float(np.mean(norm_in_bounds)) if len(norm_in_bounds) > 0 else 1.0

    return {
        "all_pairs_simultaneous_success": all_pairs_simultaneous,
        "all_pairs_fraction_satisfied": all_pairs_fraction,
        "norm_simultaneous_success": norm_simultaneous,
        "norm_fraction_satisfied": norm_fraction,
        "max_pair_distortion": float(np.max(distortions)),
        "median_pair_distortion": float(np.median(distortions)),
    }


def run_jl_hierarchy_experiments(
    q_values: list[float] | None = None,
    r_values: list[int] | None = None,
    num_trials: int = 100,
    N: int = 512,
    s: int = 10,
    m: int = 120,
    epsilon: float = 0.20,
    snr_db: float = 25.0,
    max_iter: int = 25,
    base_seed: int = 6060,
    output_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run empirical tests of the 4-level validation hierarchy.
    """
    if q_values is None:
        q_values = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    if r_values is None:
        r_values = [10, 20, 30, 40, 60, 80, 100]

    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 85)
    print("STAGE 6: GENERAL JL REQUIREMENT VS TASK-SPECIFIC HIERARCHY")
    print(f"Problem: N={N}, s={s}, m={m}, SNR={snr_db} dB, max_iter={max_iter}, JL epsilon={epsilon}")
    print(f"Sparsity grid q: {q_values}")
    print(f"Measurements grid r: {r_values}")
    print(f"Trials per setting: {num_trials}")
    print("=" * 85)

    p = min(max_iter, m, N)

    # 1. Pre-generate problems
    print("Generating problem ensemble...")
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

    # =========================================================================
    # PART 1: HIERARCHY VS SPARSITY (q) AT FIXED r = 50
    # =========================================================================
    r_fixed = 50
    print(f"\n[Part 1] Testing Hierarchy vs Matrix Density q at r={r_fixed}...")
    hierarchy_q_records = []
    distortion_by_sparsity_records = []

    for q in q_values:
        m_type = "dense_rademacher" if q == 1.0 else "sparse_rademacher"
        for t, (x_true, traj, error_vectors, sigma_n) in enumerate(problems):
            rng_val = np.random.default_rng(base_seed + 100000 + int(q * 10000) + t)
            Psi = generate_projection_matrix(r_fixed, N, matrix_type=m_type, q=q, rng=rng_val)
            y_cv, _, _ = add_noise_snr(Psi @ x_true, snr_db=snr_db, rng=rng_val)
            E_hat = compute_cv_errors(Psi, y_cv, traj.iterates)

            E_true = traj.true_errors
            sel = best_iterate_selection(E_true, E_hat)
            tau = kendall_tau(E_true, E_hat)

            # Test Level 1 and Level 2 (on actual error vector projections)
            jl_res = compute_all_pairs_jl(error_vectors, Psi, epsilon=epsilon)
            proj_error_sq = np.sum((error_vectors @ Psi.T) ** 2, axis=1)
            ward_res = ward_bound_stats(E_true, proj_error_sq)

            hierarchy_q_records.append({
                "trial": t,
                "q": q,
                "nnz_per_row": q * N,
                "level1_all_pairs_simultaneous": jl_res["all_pairs_simultaneous_success"],
                "level1_all_pairs_fraction": jl_res["all_pairs_fraction_satisfied"],
                "level2_norm_simultaneous": jl_res["norm_simultaneous_success"],
                "level2_norm_fraction": jl_res["norm_fraction_satisfied"],
                "level2b_ward_simultaneous": ward_res["ward_simultaneous_success"],
                "level2b_ward_fraction": ward_res["ward_fraction_satisfied"],
                "level3_ordering_tau90": bool(tau >= 0.90),
                "level3_kendall_tau": tau,
                "level4_selection_match": sel["is_exact_match"],
                "level4_near_match": abs(sel["j_hat_star"] - sel["j_true_star"]) <= 1,
            })

            # Record per-iteration distortion vs error vector sparsity ||e^j||_0
            if q in [1.0, 0.2, 0.05]:
                for j in range(len(error_vectors)):
                    e_j = error_vectors[j]
                    e_norm_sq = float(np.sum(e_j ** 2))
                    if e_norm_sq > 1e-10:
                        proj_norm_sq = float(np.sum((Psi @ e_j) ** 2))
                        distort = abs((proj_norm_sq / e_norm_sq) - 1.0)
                        sparsity_e = traj.error_sparsities[j]
                        distortion_by_sparsity_records.append({
                            "q": q,
                            "iteration": j,
                            "error_sparsity": sparsity_e,
                            "distortion": distort,
                        })

    df_hierarchy_q_trials = pd.DataFrame(hierarchy_q_records)
    df_distort_sparsity = pd.DataFrame(distortion_by_sparsity_records)

    # Summary by q
    summary_q_list = []
    for q in q_values:
        sub = df_hierarchy_q_trials[df_hierarchy_q_trials["q"] == q]
        summary_q_list.append({
            "q": q,
            "nnz_per_row": q * N,
            "Level1_AllPairs_Strict": float(sub["level1_all_pairs_simultaneous"].mean()),
            "Level1_AllPairs_Fraction": float(sub["level1_all_pairs_fraction"].mean()),
            "Level2_Norms_Strict": float(sub["level2_norm_simultaneous"].mean()),
            "Level2_Norms_Fraction": float(sub["level2_norm_fraction"].mean()),
            "Level2B_Ward_Strict": float(sub["level2b_ward_simultaneous"].mean()),
            "Level2B_Ward_Fraction": float(sub["level2b_ward_fraction"].mean()),
            "Level3_Ordering_Tau90": float(sub["level3_ordering_tau90"].mean()),
            "Level3_Mean_Tau": float(sub["level3_kendall_tau"].mean()),
            "Level4_Selection_Acc": float(sub["level4_selection_match"].mean()),
            "Level4_Near_Acc": float(sub["level4_near_match"].mean()),
        })
    df_hierarchy_q_summary = pd.DataFrame(summary_q_list)

    # =========================================================================
    # PART 2: HIERARCHY VS MEASUREMENTS (r) AT q = 0.20
    # =========================================================================
    q_target = 0.20
    print(f"\n[Part 2] Testing Hierarchy vs Measurement Budget r at q={q_target}...")
    hierarchy_r_records = []
    for r in r_values:
        for t, (x_true, traj, error_vectors, sigma_n) in enumerate(problems):
            rng_val = np.random.default_rng(base_seed + 300000 + 100 * r + t)
            Psi = generate_projection_matrix(r, N, matrix_type="sparse_rademacher", q=q_target, rng=rng_val)
            y_cv, _, _ = add_noise_snr(Psi @ x_true, snr_db=snr_db, rng=rng_val)
            E_hat = compute_cv_errors(Psi, y_cv, traj.iterates)

            E_true = traj.true_errors
            sel = best_iterate_selection(E_true, E_hat)
            tau = kendall_tau(E_true, E_hat)
            jl_res = compute_all_pairs_jl(error_vectors, Psi, epsilon=epsilon)
            proj_error_sq = np.sum((error_vectors @ Psi.T) ** 2, axis=1)
            ward_res = ward_bound_stats(E_true, proj_error_sq)

            hierarchy_r_records.append({
                "trial": t,
                "r": r,
                "level1_all_pairs_simultaneous": jl_res["all_pairs_simultaneous_success"],
                "level1_all_pairs_fraction": jl_res["all_pairs_fraction_satisfied"],
                "level2_norm_simultaneous": jl_res["norm_simultaneous_success"],
                "level2_norm_fraction": jl_res["norm_fraction_satisfied"],
                "level2b_ward_simultaneous": ward_res["ward_simultaneous_success"],
                "level2b_ward_fraction": ward_res["ward_fraction_satisfied"],
                "level3_ordering_tau90": bool(tau >= 0.90),
                "level4_selection_match": sel["is_exact_match"],
                "level4_near_match": abs(sel["j_hat_star"] - sel["j_true_star"]) <= 1,
            })

    df_hierarchy_r_trials = pd.DataFrame(hierarchy_r_records)

    summary_r_list = []
    for r in r_values:
        sub = df_hierarchy_r_trials[df_hierarchy_r_trials["r"] == r]
        summary_r_list.append({
            "r": r,
            "Level1_AllPairs_Strict": float(sub["level1_all_pairs_simultaneous"].mean()),
            "Level1_AllPairs_Fraction": float(sub["level1_all_pairs_fraction"].mean()),
            "Level2_Norms_Strict": float(sub["level2_norm_simultaneous"].mean()),
            "Level2_Norms_Fraction": float(sub["level2_norm_fraction"].mean()),
            "Level2B_Ward_Strict": float(sub["level2b_ward_simultaneous"].mean()),
            "Level2B_Ward_Fraction": float(sub["level2b_ward_fraction"].mean()),
            "Level3_Ordering_Tau90": float(sub["level3_ordering_tau90"].mean()),
            "Level4_Selection_Acc": float(sub["level4_selection_match"].mean()),
            "Level4_Near_Acc": float(sub["level4_near_match"].mean()),
        })
    df_hierarchy_r_summary = pd.DataFrame(summary_r_list)

    # Save summaries
    csv_q_path = os.path.join(output_dir, "jl_hierarchy_q_summary.csv")
    csv_r_path = os.path.join(output_dir, "jl_hierarchy_r_summary.csv")
    df_hierarchy_q_summary.to_csv(csv_q_path, index=False)
    df_hierarchy_r_summary.to_csv(csv_r_path, index=False)

    print("\n" + "=" * 95)
    print("STAGE 6 SUMMARY TABLE: THE 4-LEVEL HIERARCHY SUCCESS RATES (%) VS DENSITY q (r=50)")
    print("=" * 95)
    disp_cols = [
        "q", "nnz_per_row", "Level1_AllPairs_Strict", "Level1_AllPairs_Fraction",
        "Level2_Norms_Strict", "Level2B_Ward_Strict", "Level3_Ordering_Tau90",
        "Level4_Selection_Acc", "Level4_Near_Acc"
    ]
    summary_display = df_hierarchy_q_summary[disp_cols].copy()
    for col in disp_cols[2:]:
        summary_display[col] = (summary_display[col] * 100).round(1)
    print(summary_display.to_string(index=False))
    print("=" * 95)

    # Multi-panel visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_q, ax_r = axes[0, 0], axes[0, 1]
    ax_sparsity, ax_levels_bar = axes[1, 0], axes[1, 1]

    # Panel 1: Hierarchy Success Rates vs Sparsity q
    ax_q.plot(df_hierarchy_q_summary["q"], df_hierarchy_q_summary["Level4_Near_Acc"] * 100, "d--", color="teal", linewidth=2.0, label=r"Level 4b: Near Selection $(\pm 1)$")
    ax_q.plot(df_hierarchy_q_summary["q"], df_hierarchy_q_summary["Level4_Selection_Acc"] * 100, "o-", color="darkblue", linewidth=2.2, label=r"Level 4: Best-Iterate Selection $\hat{j}=j^*$")
    ax_q.plot(df_hierarchy_q_summary["q"], df_hierarchy_q_summary["Level3_Ordering_Tau90"] * 100, "s-", color="forestgreen", linewidth=2.0, label=r"Level 3: Ordering ($\tau \geq 0.90$)")
    ax_q.plot(df_hierarchy_q_summary["q"], df_hierarchy_q_summary["Level2B_Ward_Strict"] * 100, "v-", color="darkorange", linewidth=2.0, label=r"Level 2b: Ward Bound ($4/5 \leq \|e\|/\|\Psi e\| \leq 4/3$)")
    ax_q.plot(df_hierarchy_q_summary["q"], df_hierarchy_q_summary["Level2_Norms_Strict"] * 100, "^-", color="purple", linewidth=1.8, label=r"Level 2: Strict Norms ($\varepsilon=0.20$)")
    ax_q.plot(df_hierarchy_q_summary["q"], df_hierarchy_q_summary["Level1_AllPairs_Strict"] * 100, "x-", color="darkred", linewidth=1.8, label=r"Level 1: Strict All-Pairs JL ($\varepsilon=0.20$)")

    ax_q.set_xscale("log")
    ax_q.set_xlabel("Matrix Density $q$ (log scale)", fontsize=11)
    ax_q.set_ylabel("Success Rate (%)", fontsize=11)
    ax_q.set_ylim(-2, 105)
    ax_q.set_title("Hierarchy Success Rate vs Validation Sparsity $q$ (at $r=50$)", fontsize=12, fontweight="bold")
    ax_q.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_q.legend(fontsize=8.0, loc="lower left")

    # Panel 2: Hierarchy Success Rates vs Measurement Budget r (at q=0.20)
    ax_r.plot(df_hierarchy_r_summary["r"], df_hierarchy_r_summary["Level4_Near_Acc"] * 100, "d--", color="teal", linewidth=2.0, label=r"Level 4b: Near Selection $(\pm 1)$")
    ax_r.plot(df_hierarchy_r_summary["r"], df_hierarchy_r_summary["Level4_Selection_Acc"] * 100, "o-", color="darkblue", linewidth=2.2, label=r"Level 4: Best-Iterate Selection $\hat{j}=j^*$")
    ax_r.plot(df_hierarchy_r_summary["r"], df_hierarchy_r_summary["Level3_Ordering_Tau90"] * 100, "s-", color="forestgreen", linewidth=2.0, label=r"Level 3: Ordering ($\tau \geq 0.90$)")
    ax_r.plot(df_hierarchy_r_summary["r"], df_hierarchy_r_summary["Level2B_Ward_Strict"] * 100, "v-", color="darkorange", linewidth=2.0, label=r"Level 2b: Ward Bound ($4/5 \leq \|e\|/\|\Psi e\| \leq 4/3$)")
    ax_r.plot(df_hierarchy_r_summary["r"], df_hierarchy_r_summary["Level2_Norms_Strict"] * 100, "^-", color="purple", linewidth=1.8, label=r"Level 2: Strict Norms ($\varepsilon=0.20$)")
    ax_r.plot(df_hierarchy_r_summary["r"], df_hierarchy_r_summary["Level1_AllPairs_Strict"] * 100, "x-", color="darkred", linewidth=1.8, label=r"Level 1: Strict All-Pairs JL ($\varepsilon=0.20$)")

    ax_r.set_xlabel("Validation Measurements $r$", fontsize=11)
    ax_r.set_ylabel("Success Rate (%)", fontsize=11)
    ax_r.set_ylim(-2, 105)
    ax_r.set_title("Hierarchy Success Rate vs Measurements $r$ (at $q=0.20$)", fontsize=12, fontweight="bold")
    ax_r.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_r.legend(fontsize=8.0, loc="lower right")

    # Panel 3: Distortion vs Error Vector Sparsity ||e^j||_0
    for q_plot, c, label in [(1.0, "black", "Dense (q=1.0)"), (0.2, "royalblue", "Sparse (q=0.2)"), (0.05, "crimson", "Ultra-Sparse (q=0.05)")]:
        sub_d = df_distort_sparsity[df_distort_sparsity["q"] == q_plot]
        grouped = sub_d.groupby("error_sparsity")["distortion"].agg(["mean", "std"]).reset_index()
        ax_sparsity.plot(grouped["error_sparsity"], grouped["mean"], "-o", color=c, label=label, linewidth=1.8)

    ax_sparsity.set_xlabel(r"Error Vector Sparsity $\|\mathbf{e}^j\|_0 = s + j$", fontsize=11)
    ax_sparsity.set_ylabel(r"Mean Distortion $|\frac{\|\Psi \mathbf{e}\|^2}{\|\mathbf{e}\|^2} - 1|$", fontsize=11)
    ax_sparsity.set_title(r"Error Structure: Distortion vs Sparsity $\|\mathbf{e}^j\|_0$", fontsize=12, fontweight="bold")
    ax_sparsity.grid(True, linestyle="--", alpha=0.4)
    ax_sparsity.legend(fontsize=9)

    # Panel 4: Required Measurements r to reach 75% Success Rate
    target_thresh = 0.75
    levels = ["L1: Strict JL", "L2: Norms (ε=0.2)", "L2b: Ward (4/5..4/3)", "L3: Tau>=0.9", "L4: Selection"]
    req_r = []

    for key, name in [
        ("Level1_AllPairs_Strict", "Level 1"),
        ("Level2_Norms_Strict", "Level 2"),
        ("Level2B_Ward_Strict", "Level 2b"),
        ("Level3_Ordering_Tau90", "Level 3"),
        ("Level4_Selection_Acc", "Level 4"),
    ]:
        qual = df_hierarchy_r_summary[df_hierarchy_r_summary[key] >= target_thresh]
        min_r = qual["r"].min() if len(qual) > 0 else 120  # cap if not reached
        req_r.append(min_r)

    bar_colors = ["darkred", "purple", "darkorange", "forestgreen", "darkblue"]
    bars = ax_levels_bar.bar(levels, req_r, color=bar_colors, edgecolor="black", width=0.55)
    for bar in bars:
        h = bar.get_height()
        label_text = f"r > {r_values[-1]}" if h > r_values[-1] else f"r = {int(h)}"
        ax_levels_bar.annotate(label_text, xy=(bar.get_x() + bar.get_width() / 2, h),
                               xytext=(0, 3), textcoords="offset points",
                               ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_levels_bar.set_ylabel("Min Rows $r$ Needed (for 75% Success)", fontsize=11)
    ax_levels_bar.set_ylim(0, 135)
    ax_levels_bar.set_title("Required Validation Budget by Hierarchy Level (at $q=0.20$)", fontsize=12, fontweight="bold")
    ax_levels_bar.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_levels_bar.tick_params(axis="x", rotation=15)

    fig.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.10, hspace=0.35, wspace=0.25)
    plot_path = os.path.join(output_dir, "jl_hierarchy_comparison.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"\nSaved JL hierarchy visualization to: {plot_path}")
    print(f"Saved summary tables to: {csv_q_path} and {csv_r_path}\n")

    return df_hierarchy_q_summary, df_hierarchy_r_summary, df_hierarchy_q_trials


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run JL hierarchy comparison experiments")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials per configuration")
    parser.add_argument("--N", type=int, default=512, help="Signal dimension")
    parser.add_argument("--s", type=int, default=10, help="Sparsity")
    parser.add_argument("--m", type=int, default=120, help="Sensing measurements")
    parser.add_argument("--eps", type=float, default=0.20, help="JL distortion tolerance epsilon")
    parser.add_argument("--snr", type=float, default=25.0, help="SNR in dB")
    parser.add_argument("--max_iter", type=int, default=25, help="OMP iterations")
    parser.add_argument("--seed", type=int, default=6060, help="Base random seed")
    args = parser.parse_args()

    run_jl_hierarchy_experiments(
        num_trials=args.trials,
        N=args.N,
        s=args.s,
        m=args.m,
        epsilon=args.eps,
        snr_db=args.snr,
        max_iter=args.max_iter,
        base_seed=args.seed,
    )
