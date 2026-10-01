"""
Stage 7 - 2D Phase Transitions and Parameter Sweeps: (q, r) -> Performance
Generates the complete 2D operational phase map of sparse cross-validation:
  - Density q in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
  - Validation rows r in [10, 20, 30, 40, 60, 80, 100, 150, 200]
  - Tracks: Selection Accuracy P(j_hat == j*), Near Selection, Kendall's tau, and Relative Error.
  - Also examines scaling across ambient dimensions N in {256, 512, 1024} and sparsity s in {5, 10, 20}.

Produces 2D contour heatmaps with empirical phase transition boundaries (80%, 90% success).
Saves summary CSVs and publication figures to Gemini/output/.
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
)


def run_heatmap_grid(
    q_values: list[float] | None = None,
    r_values: list[int] | None = None,
    num_trials: int = 100,
    N: int = 512,
    s: int = 10,
    m: int = 120,
    snr_db: float = 25.0,
    max_iter: int = 25,
    base_seed: int = 7070,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Run 2D grid sweep over (q, r) and compute phase transition statistics.
    """
    if q_values is None:
        # Doubled resolution across matrix density q: 14 logarithmically spaced points
        q_values = [1.0, 0.70, 0.50, 0.35, 0.25, 0.18, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02, 0.014, 0.01]
    if r_values is None:
        # Doubled resolution across validation measurements budget r: 18 points
        r_values = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]

    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 85)
    print("STAGE 7: 2D PHASE TRANSITIONS (q, r) -> SELECTION ACCURACY & ORDERING")
    print(f"Problem: N={N}, s={s}, m={m}, SNR={snr_db} dB, max_iter={max_iter}")
    print(f"Sparsity grid q ({len(q_values)}): {q_values}")
    print(f"Measurements grid r ({len(r_values)}): {r_values}")
    print(f"Total grid cells: {len(q_values) * len(r_values)} | Trials per cell: {num_trials}")
    print("=" * 85)

    p = min(max_iter, m, N)

    # 1. Pre-generate problems: OMP is solved once per trial, then cached!
    print("Generating problem ensemble and pre-computing OMP trajectories...")
    problems = []
    for t in range(num_trials):
        rng = np.random.default_rng(base_seed + t)
        x_true, support = generate_sparse_signal(N, s, coeff_dist="gaussian", rng=rng)
        Phi = generate_sensing_matrix(m, N, dist="gaussian", normalize="column", rng=rng)
        y_clean = Phi @ x_true
        y, _, sigma_n = add_noise_snr(y_clean, snr_db=snr_db, rng=rng)
        traj = run_omp_trajectory(Phi, y, max_iter=p, x_true=x_true)
        problems.append((x_true, traj, sigma_n))

    # 2. Evaluate 2D Grid
    grid_results = []
    total_cells = len(r_values) * len(q_values)
    cell_idx = 0

    for r in r_values:
        for q in q_values:
            cell_idx += 1
            m_type = "dense_rademacher" if q == 1.0 else "sparse_rademacher"

            matches = []
            near_matches = []
            taus = []
            concordances = []
            pre_rels = []
            excesses = []

            for t, (x_true, traj, sigma_n) in enumerate(problems):
                rng_val = np.random.default_rng(base_seed + 100000 * r + int(q * 10000) + t)
                Psi = generate_projection_matrix(r, N, matrix_type=m_type, q=q, rng=rng_val)
                y_cv, _, _ = add_noise_snr(Psi @ x_true, snr_db=snr_db, rng=rng_val)
                E_hat = compute_cv_errors(Psi, y_cv, traj.iterates)

                E_true = traj.true_errors
                sel = best_iterate_selection(E_true, E_hat)
                tau = kendall_tau(E_true, E_hat)
                conc = pairwise_concordance(E_true, E_hat)
                pre_rel = pre_convergence_relative_errors(E_true, E_hat, sel["j_true_star"])

                matches.append(sel["is_exact_match"])
                near_matches.append(abs(sel["j_hat_star"] - sel["j_true_star"]) <= 1)
                taus.append(tau)
                concordances.append(conc)
                pre_rels.append(float(np.median(pre_rel)))
                excesses.append(sel["excess_error"])

            grid_results.append({
                "r": r,
                "q": q,
                "expected_nnz_per_row": q * N,
                "total_nnz": int(round(r * q * N)),
                "selection_accuracy": float(np.mean(matches)),
                "near_selection_accuracy": float(np.mean(near_matches)),
                "mean_kendall_tau": float(np.mean(taus)),
                "median_kendall_tau": float(np.median(taus)),
                "mean_concordance": float(np.mean(concordances)),
                "pre_conv_rel_error_med": float(np.median(pre_rels)),
                "mean_excess_error": float(np.mean(excesses)),
            })

            if cell_idx % 25 == 0 or cell_idx == total_cells:
                print(f"Evaluated {cell_idx}/{total_cells} grid cells...")

    df_grid = pd.DataFrame(grid_results)

    # =========================================================================
    # PLOTTING 2D HEATMAPS (Clean Color Fields, No Contour Lines)
    # =========================================================================
    pivot_acc = df_grid.pivot(index="r", columns="q", values="selection_accuracy") * 100
    pivot_near = df_grid.pivot(index="r", columns="q", values="near_selection_accuracy") * 100
    pivot_tau = df_grid.pivot(index="r", columns="q", values="mean_kendall_tau")
    pivot_rel = df_grid.pivot(index="r", columns="q", values="pre_conv_rel_error_med")

    q_arr = np.array(q_values)
    r_arr = np.array(r_values)
    Q, R = np.meshgrid(q_arr, r_arr)

    # -------------------------------------------------------------------------
    # Figure 1: 2-Panel Selection Accuracy & Near Selection Heatmaps
    # -------------------------------------------------------------------------
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Exact Best-Iterate Selection Accuracy
    im1 = ax1.pcolormesh(Q, R, pivot_acc.values, cmap="viridis", shading="auto", vmin=30, vmax=100, edgecolors="none")
    cbar1 = fig1.colorbar(im1, ax=ax1)
    cbar1.set_label(r"Selection Accuracy $\mathbb{P}(\hat{n} = n^*)$ (%)", fontsize=11, fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_xlabel("Matrix Density $q$ (log scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Validation Measurements Budget $r$", fontsize=11, fontweight="bold")
    ax1.set_title(r"Exact Selection Accuracy $\mathbb{P}(\hat{n} = n^*)$ Phase Transition", fontsize=12, fontweight="bold")
    ax1.grid(False)

    # Panel 2: Near Match Selection Accuracy (within +/- 1 iterate)
    im2 = ax2.pcolormesh(Q, R, pivot_near.values, cmap="viridis", shading="auto", vmin=50, vmax=100, edgecolors="none")
    cbar2 = fig1.colorbar(im2, ax=ax2)
    cbar2.set_label(r"Near Selection Accuracy $\mathbb{P}(|\hat{n} - n^*| \leq 1)$ (%)", fontsize=11, fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_xlabel("Matrix Density $q$ (log scale)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Validation Measurements Budget $r$", fontsize=11, fontweight="bold")
    ax2.set_title(r"Near-Optimal Selection Accuracy $\mathbb{P}(|\hat{n} - n^*| \leq 1)$", fontsize=12, fontweight="bold")
    ax2.grid(False)

    fig1.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.12, wspace=0.25)
    fig1_path = os.path.join(output_dir, "p7_heatmap_selection_accuracy.png")
    fig1.savefig(fig1_path, dpi=200)
    plt.close(fig1)
    print(f"Saved Selection Accuracy Heatmap to: {fig1_path}")

    # -------------------------------------------------------------------------
    # Figure 2: 2-Panel Kendall's Tau & Pre-convergence Relative Error
    # -------------------------------------------------------------------------
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 7))

    im3 = ax3.pcolormesh(Q, R, pivot_tau.values, cmap="plasma", shading="auto", vmin=0.70, vmax=1.0, edgecolors="none")
    cbar3 = fig2.colorbar(im3, ax=ax3)
    cbar3.set_label(r"Mean Kendall's $\tau$ (between $[\varepsilon_x^n]$ and $[\varepsilon_{\mathrm{cv}}^n]$)", fontsize=11, fontweight="bold")
    ax3.set_xscale("log")
    ax3.set_xlabel("Matrix Density $q$ (log scale)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Validation Measurements Budget $r$", fontsize=11, fontweight="bold")
    ax3.set_title(r"Ordering Preservation (Mean Kendall's $\tau$) Operational Regime", fontsize=12, fontweight="bold")
    ax3.grid(False)

    im4 = ax4.pcolormesh(Q, R, pivot_rel.values, cmap="magma_r", shading="auto", vmin=0.10, vmax=0.70, edgecolors="none")
    cbar4 = fig2.colorbar(im4, ax=ax4)
    cbar4.set_label(r"Pre-Convergence Relative Error $\frac{|\varepsilon_{\mathrm{cv}}^n - \varepsilon_x^n|}{\varepsilon_x^n}$", fontsize=11, fontweight="bold")
    ax4.set_xscale("log")
    ax4.set_xlabel("Matrix Density $q$ (log scale)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Validation Measurements Budget $r$", fontsize=11, fontweight="bold")
    ax4.set_title(r"Signal Norm Estimation Relative Error ($n \leq n^*$)", fontsize=12, fontweight="bold")
    ax4.grid(False)

    fig2.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.12, wspace=0.25)
    fig2_path = os.path.join(output_dir, "p7_heatmap_kendall_tau.png")
    fig2.savefig(fig2_path, dpi=200)
    plt.close(fig2)
    print(f"Saved Kendall's Tau Heatmap to: {fig2_path}")

    # -------------------------------------------------------------------------
    # Standalone Individual Heatmaps (Single Panel per Metric)
    # -------------------------------------------------------------------------
    standalone_specs = [
        ("p7_1_selection_accuracy_heatmap.png", pivot_acc.values, "viridis", 30, 100,
         r"Exact Selection Accuracy $\mathbb{P}(\hat{n} = n^*)$ Phase Transition",
         r"Selection Accuracy $\mathbb{P}(\hat{n}=n^*)$ (%)"),
        ("p7_2_near_selection_heatmap.png", pivot_near.values, "viridis", 50, 100,
         r"Near-Optimal Selection Accuracy $\mathbb{P}(|\hat{n} - n^*| \leq 1)$ Phase Transition",
         r"Near Selection Accuracy $\mathbb{P}(|\hat{n} - n^*| \leq 1)$ (%)"),
        ("p7_3_kendall_tau_heatmap.png", pivot_tau.values, "plasma", 0.70, 1.0,
         r"Ordering Preservation (Mean Kendall's $\tau$) Operational Regime",
         r"Mean Kendall's $\tau$ (between $[\varepsilon_x^n]$ and $[\varepsilon_{\mathrm{cv}}^n]$)"),
        ("p7_4_norm_error_heatmap.png", pivot_rel.values, "magma_r", 0.10, 0.70,
         r"Signal Norm Estimation Relative Error ($n \leq n^*$)",
         r"Median Relative Error $\frac{|\varepsilon_{\mathrm{cv}}^n - \varepsilon_x^n|}{\varepsilon_x^n}$"),
    ]

    for fname, data_vals, cmap_name, vmin_val, vmax_val, title_str, cbar_str in standalone_specs:
        fig_s, ax_s = plt.subplots(figsize=(8.5, 6.5))
        im_s = ax_s.pcolormesh(Q, R, data_vals, cmap=cmap_name, shading="auto", vmin=vmin_val, vmax=vmax_val, edgecolors="none")
        cbar_s = fig_s.colorbar(im_s, ax=ax_s)
        cbar_s.set_label(cbar_str, fontsize=11, fontweight="bold")
        ax_s.set_xscale("log")
        ax_s.set_xlabel("Matrix Density $q$ (log scale)", fontsize=11, fontweight="bold")
        ax_s.set_ylabel("Validation Measurements Budget $r$", fontsize=11, fontweight="bold")
        ax_s.set_title(title_str, fontsize=12, fontweight="bold", pad=12)
        ax_s.grid(False)
        fig_s.tight_layout()
        s_path = os.path.join(output_dir, fname)
        fig_s.savefig(s_path, dpi=200)
        plt.close(fig_s)
        print(f"Saved standalone heatmap: {s_path}")

    return df_grid


def run_regime_scaling_sweep(
    N_values: list[int] | None = None,
    s_values: list[int] | None = None,
    num_trials: int = 50,
    r: int = 60,
    q_values: list[float] | None = None,
    snr_db: float = 25.0,
    base_seed: int = 8080,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Test how the validation sparsity threshold scales across ambient dimension N and sparsity s.
    """
    if N_values is None:
        N_values = [256, 512, 1024]
    if s_values is None:
        s_values = [5, 10, 20]
    if q_values is None:
        q_values = [1.0, 0.2, 0.05, 0.02]

    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 85)
    print("STAGE 7 (PART B): SCALING ACROSS AMBIENT DIMENSION N AND SPARSITY s")
    print(f"Dimensions N: {N_values} | Sparsities s: {s_values} | Densities q: {q_values}")
    print(f"Validation budget fixed at r={r} | Trials per setting: {num_trials}")
    print("=" * 85)

    scaling_records = []
    idx = 0

    for N in N_values:
        for s in s_values:
            # Set sensing measurements proportional to s * log(N/s)
            m = min(int(round(3.5 * s * np.log(N / s))), N // 2)
            max_iter = int(round(2.5 * s))

            # Pre-generate problems for this (N, s, m)
            problems = []
            for t in range(num_trials):
                rng = np.random.default_rng(base_seed + 1000 * N + 10 * s + t)
                x_true, support = generate_sparse_signal(N, s, coeff_dist="gaussian", rng=rng)
                Phi = generate_sensing_matrix(m, N, dist="gaussian", normalize="column", rng=rng)
                y_clean = Phi @ x_true
                y, _, _ = add_noise_snr(y_clean, snr_db=snr_db, rng=rng)
                traj = run_omp_trajectory(Phi, y, max_iter=max_iter, x_true=x_true)
                problems.append((x_true, traj))

            for q in q_values:
                idx += 1
                m_type = "dense_rademacher" if q == 1.0 else "sparse_rademacher"
                matches = []
                near_matches = []
                taus = []

                for t, (x_true, traj) in enumerate(problems):
                    rng_val = np.random.default_rng(base_seed + 500000 + 100 * N + 10 * s + int(q * 1000) + t)
                    Psi = generate_projection_matrix(r, N, matrix_type=m_type, q=q, rng=rng_val)
                    y_cv, _, _ = add_noise_snr(Psi @ x_true, snr_db=snr_db, rng=rng_val)
                    E_hat = compute_cv_errors(Psi, y_cv, traj.iterates)

                    E_true = traj.true_errors
                    sel = best_iterate_selection(E_true, E_hat)
                    tau = kendall_tau(E_true, E_hat)

                    matches.append(sel["is_exact_match"])
                    near_matches.append(abs(sel["j_hat_star"] - sel["j_true_star"]) <= 1)
                    taus.append(tau)

                scaling_records.append({
                    "N": N,
                    "s": s,
                    "m": m,
                    "r": r,
                    "q": q,
                    "expected_nonzeros_row": q * N,
                    "selection_accuracy": float(np.mean(matches)),
                    "near_selection_accuracy": float(np.mean(near_matches)),
                    "mean_kendall_tau": float(np.mean(taus)),
                })

    df_scaling = pd.DataFrame(scaling_records)
    # scaling_csv = os.path.join(output_dir, "heatmap_regime_scaling.csv")
    # df_scaling.to_csv(scaling_csv, index=False)

    # Plotting scaling curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Dynamic palette for arbitrary N
    cmap = plt.cm.tab10
    colors_N = {N_val: cmap(i % 10) for i, N_val in enumerate(N_values)}

    # Target q to plot across sparsity s
    q_target = 0.05 if 0.05 in q_values else q_values[-1]
    for N in N_values:
        sub = df_scaling[(df_scaling["N"] == N) & (df_scaling["q"] == q_target)]
        if len(sub) > 0:
            ax1.plot(sub["s"], sub["selection_accuracy"] * 100, "-o", color=colors_N[N], linewidth=2.0, label=f"N={N} (q={q_target})")
    ax1.set_xlabel("Signal Sparsity $s$", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Selection Accuracy $\mathbb{P}(\hat{n} = n^*)$ (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(40, 105)
    ax1.set_title(f"Scaling with Signal Sparsity $s$ (Sparsity $q={q_target}$)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9)

    # Panel 2: Selection Accuracy vs Matrix Density q across Dimensions N (at median s)
    s_target = s_values[len(s_values) // 2]
    for N in N_values:
        sub = df_scaling[(df_scaling["N"] == N) & (df_scaling["s"] == s_target)]
        if len(sub) > 0:
            ax2.plot(sub["q"], sub["selection_accuracy"] * 100, "-s", color=colors_N[N], linewidth=2.0, label=f"N={N} (s={s_target})")
    ax2.set_xscale("log")
    ax2.set_xlabel("Matrix Density $q$ (log scale)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Selection Accuracy $\mathbb{P}(\hat{n} = n^*)$ (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(40, 105)
    ax2.set_title(f"Scaling with Ambient Dimension $N$ ($s={s_target}$)", fontsize=12, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9)

    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12, wspace=0.25)
    scaling_plot = os.path.join(output_dir, "p7_heatmap_regime_scaling.png")
    plt.savefig(scaling_plot, dpi=200)
    plt.close()
    print(f"Saved Scaling Plot to: {scaling_plot}")

    return df_scaling


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2D Phase Transition Heatmaps and Scaling Sweeps")
    parser.add_argument("--trials", type=int, default=100, help="Trials per grid cell")
    parser.add_argument("--N", type=int, default=512, help="Signal dimension")
    parser.add_argument("--s", type=int, default=10, help="Sparsity")
    parser.add_argument("--m", type=int, default=120, help="Sensing measurements")
    parser.add_argument("--snr", type=float, default=25.0, help="SNR in dB")
    parser.add_argument("--seed", type=int, default=7070, help="Base random seed")
    parser.add_argument("--run_scaling", action="store_true", help="Also run N and s scaling sweep")
    args = parser.parse_args()

    run_heatmap_grid(
        num_trials=args.trials,
        N=args.N,
        s=args.s,
        m=args.m,
        snr_db=args.snr,
        base_seed=args.seed,
    )

    if args.run_scaling:
        run_regime_scaling_sweep(
            num_trials=50,
            base_seed=args.seed,
        )
