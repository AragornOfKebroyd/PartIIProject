"""
Stage Joint Sparsity: Cross-Product Interplay of Sensing and Validation Matrix Sparsity.

Evaluates the joint parameter space (q_Phi, q_Psi) in a 2D Cartesian cross-product grid:
- q_Phi : Sparsity density of sensing matrix Phi in R^{m x N}
- q_Psi : Sparsity density of validation matrix Psi in R^{r x N}

Addresses whether:
1. Sparse sensing matrices change the error trajectory structure and affect validation.
2. Sparse validation matrices Psi maintain model selection accuracy when Phi is also sparse.
3. Fully sparse compressed sensing pipelines (sparse Phi AND sparse Psi) remain viable.
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src modules can be imported
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from groundwork.signals import generate_sparse_signal
from groundwork.matrices import generate_measurement_matrix, add_noise, add_noise_snr
from groundwork.omp import run_omp_trajectory
from groundwork.metrics import (
    compute_cv_errors,
    kendall_tau,
    pairwise_concordance,
    best_iterate_selection,
)


def run_joint_sparsity_cross_product(
    q_phi_values: list[float] | None = None,
    q_psi_values: list[float] | None = None,
    num_trials: int = 50,
    N: int = 512,
    s: int = 10,
    m: int = 120,
    r: int = 40,
    snr_db: float = 25.0,
    max_iter: int = 25,
    base_seed: int = 8080,
    output_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run 2D cross-product grid of sensing sparsity q_Phi x validation sparsity q_Psi.

    Parameters:
    -----------
    q_phi_values : list[float]
        Densities for sensing matrix Phi.
    q_psi_values : list[float]
        Densities for validation projection matrix Psi.
    num_trials : int
        Number of Monte Carlo trials per grid cell.
    """
    if q_phi_values is None:
        q_phi_values = [1.0, 0.5, 0.2, 0.1, 0.05]
    if q_psi_values is None:
        q_psi_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 85)
    print("JOINT SPARSITY CROSS-PRODUCT: SENSING MATRIX (q_Phi) x VALIDATION MATRIX (q_Psi)")
    print(f"Problem: N={N}, s={s}, m={m}, r={r}, SNR={snr_db} dB, max_iter={max_iter}")
    print(f"Sensing densities q_Phi ({len(q_phi_values)}): {q_phi_values}")
    print(f"Validation densities q_Psi ({len(q_psi_values)}): {q_psi_values}")
    print(f"Total grid cells: {len(q_phi_values) * len(q_psi_values)} | Trials per cell: {num_trials}")
    print("=" * 85)

    trial_records = []
    t_start = time.time()

    # Pre-generate signals for consistency across trials
    signals = []
    for t in range(num_trials):
        rng_sig = np.random.default_rng(base_seed + t)
        x_true, _ = generate_sparse_signal(N, s, coeff_dist="gaussian", rng=rng_sig)
        signals.append(x_true)

    total_cells = len(q_phi_values) * len(q_psi_values)
    cell_idx = 0

    for i_phi, q_phi in enumerate(q_phi_values):
        for i_psi, q_psi in enumerate(q_psi_values):
            cell_idx += 1
            cell_start = time.time()

            for t in range(num_trials):
                trial_seed = base_seed + 100_000 * (i_phi + 1) + 1_000 * (i_psi + 1) + t
                rng = np.random.default_rng(trial_seed)
                x_true = signals[t]

                # 1. Sensing matrix Phi (density q_phi)
                Phi = generate_measurement_matrix(
                    rows=m, cols=N, dist="sparse_rademacher", q=q_phi, normalize="column", rng=rng
                )
                y_clean = Phi @ x_true
                y, noise, sigma = add_noise_snr(y_clean, snr_db=snr_db, rng=rng)

                # 2. OMP Trajectory
                traj = run_omp_trajectory(Phi, y, max_iter=max_iter, x_true=x_true)
                E_true = traj.true_errors
                j_star = int(np.argmin(E_true))
                min_oracle_err = float(E_true[j_star])

                # 3. Validation matrix Psi (density q_psi)
                Psi = generate_measurement_matrix(
                    rows=r, cols=N, dist="sparse_rademacher", q=q_psi, normalize="variance", rng=rng
                )
                # Validation measurements
                y_val_clean = Psi @ x_true
                y_val, _ = add_noise(y_val_clean, sigma=sigma, rng=rng)

                # 4. Cross-validation error estimates
                E_hat = compute_cv_errors(Psi, y_val, traj.iterates)

                # 5. Metrics evaluation
                sel_stats = best_iterate_selection(E_true, E_hat)
                j_hat = sel_stats["j_hat_star"]
                is_exact = float(sel_stats["is_exact_match"])
                near_match = float(abs(j_hat - j_star) <= 1)
                tau = kendall_tau(E_true, E_hat)
                conc = pairwise_concordance(E_true, E_hat)
                abs_excess = float(E_true[j_hat] - E_true[j_star])

                trial_records.append({
                    "trial": t,
                    "q_phi": q_phi,
                    "q_psi": q_psi,
                    "j_star": j_star,
                    "j_hat": j_hat,
                    "exact_selection": is_exact,
                    "near_selection": near_match,
                    "excess_error": abs_excess,
                    "relative_excess_error": sel_stats["excess_error"],
                    "kendall_tau": tau,
                    "concordance": conc,
                    "oracle_error": min_oracle_err,
                })

            pct = (cell_idx / total_cells) * 100
            print(f"  [{cell_idx:2d}/{total_cells}] Completed (q_Phi={q_phi:5.2f}, q_Psi={q_psi:5.2f}) in {time.time() - cell_start:.2f}s ({pct:4.1f}%)")

    df_trials = pd.DataFrame(trial_records)

    # Aggregate by (q_phi, q_psi)
    summary_list = []
    for (q_phi, q_psi), group in df_trials.groupby(["q_phi", "q_psi"]):
        summary_list.append({
            "q_phi": q_phi,
            "q_psi": q_psi,
            "nnz_phi_per_row": q_phi * N,
            "nnz_psi_per_row": q_psi * N,
            "selection_accuracy": group["exact_selection"].mean(),
            "near_accuracy": group["near_selection"].mean(),
            "kendall_tau_mean": group["kendall_tau"].mean(),
            "kendall_tau_median": group["kendall_tau"].median(),
            "concordance_mean": group["concordance"].mean(),
            "oracle_error_mean": group["oracle_error"].mean(),
            "excess_error_mean": group["excess_error"].mean(),
            "excess_error_median": group["excess_error"].median(),
            "relative_excess_mean": group["relative_excess_error"].mean(),
        })

    df_summary = pd.DataFrame(summary_list).sort_values(by=["q_phi", "q_psi"], ascending=[False, False])

    # Save CSVs
    # csv_trials_path = os.path.join(output_dir, "joint_sparsity_cross_product_trials.csv")
    # csv_summary_path = os.path.join(output_dir, "joint_sparsity_cross_product_summary.csv")
    # df_trials.to_csv(csv_trials_path, index=False)
    # df_summary.to_csv(csv_summary_path, index=False)
    # print(f"\nSaved trial data to: {csv_trials_path}")
    # print(f"Saved summary data to: {csv_summary_path}")

    # Plot comprehensive 4-panel visual comparison
    _plot_joint_sparsity_results(df_summary, q_phi_values, q_psi_values, output_dir, N, s, m, r, snr_db)

    print(f"\nJoint sparsity sweep completed in {time.time() - t_start:.2f} seconds!")
    return df_summary, df_trials


def _plot_joint_sparsity_results(
    df_summary: pd.DataFrame,
    q_phi_values: list[float],
    q_psi_values: list[float],
    output_dir: str,
    N: int,
    s: int,
    m: int,
    r: int,
    snr_db: float,
):
    """Generate 4-panel cross-product publication visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Pivot tables for 2D heatmaps
    piv_acc = df_summary.pivot(index="q_phi", columns="q_psi", values="selection_accuracy") * 100.0
    piv_tau = df_summary.pivot(index="q_phi", columns="q_psi", values="kendall_tau_mean")

    # Sort descending so top-left is dense (1.0, 1.0)
    piv_acc = piv_acc.sort_index(ascending=False).sort_index(axis=1, ascending=False)
    piv_tau = piv_tau.sort_index(ascending=False).sort_index(axis=1, ascending=False)

    q_phi_labels = [f"{q:.2f}" for q in piv_acc.index]
    q_psi_labels = [f"{q:.2f}" for q in piv_acc.columns]

    # Panel 1: Selection Accuracy Heatmap
    # Panel 1: Selection Accuracy Heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(piv_acc.values, cmap="viridis", aspect="auto", vmin=30, vmax=100)
    ax1.set_xticks(range(len(q_psi_labels)))
    ax1.set_xticklabels(q_psi_labels, fontsize=10)
    ax1.set_yticks(range(len(q_phi_labels)))
    ax1.set_yticklabels(q_phi_labels, fontsize=10)
    ax1.set_xlabel(r"Validation Matrix Density $q_{\Psi}$", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Sensing Matrix Density $q_{\Phi}$", fontsize=11, fontweight="bold")
    ax1.set_title(r"(A) Model Selection Accuracy $\mathbb{P}(\hat{n} = n^*)$ (%)", fontsize=12, fontweight="bold")
    plt.colorbar(im1, ax=ax1, label=r"Accuracy $\mathbb{P}(\hat{n} = n^*)$ (%)")

    # Annotate values inside heatmap
    for i in range(len(q_phi_labels)):
        for j in range(len(q_psi_labels)):
            val = piv_acc.values[i, j]
            color = "white" if val < 70 else "black"
            ax1.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    # Panel 2: Kendall's Tau Heatmap
    ax2 = axes[0, 1]
    im2 = ax2.imshow(piv_tau.values, cmap="magma", aspect="auto", vmin=0.6, vmax=1.0)
    ax2.set_xticks(range(len(q_psi_labels)))
    ax2.set_xticklabels(q_psi_labels, fontsize=10)
    ax2.set_yticks(range(len(q_phi_labels)))
    ax2.set_yticklabels(q_phi_labels, fontsize=10)
    ax2.set_xlabel(r"Validation Matrix Density $q_{\Psi}$", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Sensing Matrix Density $q_{\Phi}$", fontsize=11, fontweight="bold")
    ax2.set_title(r"(B) Kendall's Rank Correlation $\tau$ (Trajectory Ordering)", fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=ax2, label=r"Kendall's $\tau$")

    for i in range(len(q_phi_labels)):
        for j in range(len(q_psi_labels)):
            val = piv_tau.values[i, j]
            color = "white" if val < 0.8 else "black"
            ax2.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    # Panel 3: Slices of Accuracy vs q_Psi across different q_Phi
    ax3 = axes[1, 0]
    markers = ["o", "s", "^", "D", "v"]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(q_phi_values)))

    for idx, q_phi in enumerate(sorted(q_phi_values, reverse=True)):
        sub = df_summary[df_summary["q_phi"] == q_phi].sort_values("q_psi", ascending=False)
        ax3.plot(
            sub["q_psi"],
            sub["selection_accuracy"] * 100.0,
            marker=markers[idx % len(markers)],
            linewidth=2,
            label=rf"Sensing $q_{{\Phi}}={q_phi:.2f}$",
            color=colors[idx],
        )

    ax3.set_xscale("log")
    ax3.set_xlabel(r"Validation Matrix Density $q_{\Psi}$ (log scale)", fontsize=11, fontweight="bold")
    ax3.set_ylabel(r"Selection Accuracy $\mathbb{P}(\hat{n} = n^*)$ (%)", fontsize=11, fontweight="bold")
    ax3.set_title(r"(C) Validation Sparsity Tolerance Across Sensing Densities", fontsize=12, fontweight="bold")
    ax3.grid(True, which="both", linestyle="--", alpha=0.4)
    ax3.legend(fontsize=9, loc="lower left")
    ax3.set_ylim(20, 105)

    # Panel 4: OMP Oracle Error vs Validation Excess Error
    ax4 = axes[1, 1]
    oracle_by_phi = df_summary.groupby("q_phi")["oracle_error_mean"].mean().sort_index(ascending=False)
    q_psi_max = df_summary["q_psi"].max()
    q_psi_min = df_summary["q_psi"].min()
    excess_by_phi_dense = df_summary[(df_summary["q_psi"] == q_psi_max)].set_index("q_phi")["excess_error_mean"]
    excess_by_phi_sparse = df_summary[(df_summary["q_psi"] == q_psi_min)].set_index("q_phi")["excess_error_mean"]

    x_vals = np.arange(len(oracle_by_phi))
    width = 0.25

    b1 = ax4.bar(x_vals - width, oracle_by_phi.values, width, label=r"Oracle Error $\varepsilon_{\mathrm{or}}$", color="#2b5c8f", alpha=0.85)
    b2 = ax4.bar(x_vals, excess_by_phi_dense.loc[oracle_by_phi.index].values, width, label=rf"Excess Error ($q_{{\Psi}}={q_psi_max:.2f}$)", color="#3caea3", alpha=0.85)
    b3 = ax4.bar(x_vals + width, excess_by_phi_sparse.loc[oracle_by_phi.index].values, width, label=rf"Excess Error ($q_{{\Psi}}={q_psi_min:.2f}$)", color="#ed553b", alpha=0.85)

    for bar in list(b1) + list(b2) + list(b3):
        h = bar.get_height()
        if h > 0:
            ax4.text(bar.get_x() + bar.get_width() / 2.0, h * 1.15, f"{h:.2f}" if h < 10 else f"{h:.1f}",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax4.set_xticks(x_vals)
    ax4.set_xticklabels([f"{q:.2f}" for q in oracle_by_phi.index], fontsize=10)
    ax4.set_xlabel(r"Sensing Matrix Density $q_{\Phi}$", fontsize=11, fontweight="bold")
    ax4.set_ylabel(r"Mean Squared Error (log scale)", fontsize=11, fontweight="bold")
    ax4.set_yscale("log")
    ax4.set_ylim(0.08, 600)
    ax4.set_title(r"(D) Oracle Error $\varepsilon_{\mathrm{or}}$ vs Excess Error $\varepsilon_x^{\hat{n}} - \varepsilon_x^{n^*}$", fontsize=12, fontweight="bold")
    ax4.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
    ax4.legend(fontsize=8.5, loc="upper right")

    fig.suptitle(
        f"Joint Sparsity Interplay: Sensing Matrix $\\Phi$ vs Validation Matrix $\\Psi$ ($N={N}, s={s}, m={m}, r={r}$)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()

    plot_path = os.path.join(output_dir, "p8_joint_sparsity_comparison.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved joint sparsity figure to: {plot_path}")

    # Also save standalone single-panel figures for slides
    # Panel 1 standalone
    fig1, a1 = plt.subplots(figsize=(7, 6))
    im1_s = a1.imshow(piv_acc.values, cmap="viridis", aspect="auto", vmin=30, vmax=100)
    a1.set_xticks(range(len(q_psi_labels)))
    a1.set_xticklabels(q_psi_labels, fontsize=10)
    a1.set_yticks(range(len(q_phi_labels)))
    a1.set_yticklabels(q_phi_labels, fontsize=10)
    a1.set_xlabel(r"Validation Matrix Density $q_{\Psi}$", fontsize=11, fontweight="bold")
    a1.set_ylabel(r"Sensing Matrix Density $q_{\Phi}$", fontsize=11, fontweight="bold")
    a1.set_title(r"Model Selection Accuracy $\mathbb{P}(\hat{n} = n^*)$ (%)", fontsize=12, fontweight="bold")
    plt.colorbar(im1_s, ax=a1, label=r"Accuracy $\mathbb{P}(\hat{n} = n^*)$ (%)")
    for i in range(len(q_phi_labels)):
        for j in range(len(q_psi_labels)):
            val = piv_acc.values[i, j]
            color = "white" if val < 70 else "black"
            a1.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=9, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "p8_1_selection_accuracy.png"), dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # Panel 2 standalone
    fig2, a2 = plt.subplots(figsize=(7, 6))
    im2_s = a2.imshow(piv_tau.values, cmap="magma", aspect="auto", vmin=0.6, vmax=1.0)
    a2.set_xticks(range(len(q_psi_labels)))
    a2.set_xticklabels(q_psi_labels, fontsize=10)
    a2.set_yticks(range(len(q_phi_labels)))
    a2.set_yticklabels(q_phi_labels, fontsize=10)
    a2.set_xlabel(r"Validation Matrix Density $q_{\Psi}$", fontsize=11, fontweight="bold")
    a2.set_ylabel(r"Sensing Matrix Density $q_{\Phi}$", fontsize=11, fontweight="bold")
    a2.set_title(r"Kendall's Rank Correlation $\tau$ (Trajectory Ordering)", fontsize=12, fontweight="bold")
    plt.colorbar(im2_s, ax=a2, label=r"Kendall's $\tau$")
    for i in range(len(q_phi_labels)):
        for j in range(len(q_psi_labels)):
            val = piv_tau.values[i, j]
            color = "white" if val < 0.8 else "black"
            a2.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "p8_2_kendall_tau.png"), dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # Panel 3 standalone
    fig3, a3 = plt.subplots(figsize=(7, 5))
    for idx, q_phi in enumerate(sorted(q_phi_values, reverse=True)):
        sub = df_summary[df_summary["q_phi"] == q_phi].sort_values("q_psi", ascending=False)
        a3.plot(
            sub["q_psi"],
            sub["selection_accuracy"] * 100.0,
            marker=markers[idx % len(markers)],
            linewidth=2,
            label=rf"Sensing $q_{{\Phi}}={q_phi:.2f}$",
            color=colors[idx],
        )
    a3.set_xscale("log")
    a3.set_xlabel(r"Validation Matrix Density $q_{\Psi}$ (log scale)", fontsize=11, fontweight="bold")
    a3.set_ylabel(r"Selection Accuracy $\mathbb{P}(\hat{n} = n^*)$ (%)", fontsize=11, fontweight="bold")
    a3.set_title(r"Validation Sparsity Tolerance Across Sensing Densities", fontsize=12, fontweight="bold")
    a3.grid(True, which="both", linestyle="--", alpha=0.4)
    a3.legend(fontsize=9, loc="lower left")
    a3.set_ylim(20, 105)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "p8_3_sparsity_tolerance.png"), dpi=200, bbox_inches="tight")
    plt.close(fig3)

    # Panel 4 standalone
    fig4, a4 = plt.subplots(figsize=(7, 5))
    b1_s = a4.bar(x_vals - width, oracle_by_phi.values, width, label=r"Oracle Error $\varepsilon_{\mathrm{or}}$", color="#2b5c8f", alpha=0.85)
    b2_s = a4.bar(x_vals, excess_by_phi_dense.loc[oracle_by_phi.index].values, width, label=rf"Excess Error ($q_{{\Psi}}={q_psi_max:.2f}$)", color="#3caea3", alpha=0.85)
    b3_s = a4.bar(x_vals + width, excess_by_phi_sparse.loc[oracle_by_phi.index].values, width, label=rf"Excess Error ($q_{{\Psi}}={q_psi_min:.2f}$)", color="#ed553b", alpha=0.85)
    for bar in list(b1_s) + list(b2_s) + list(b3_s):
        h = bar.get_height()
        if h > 0:
            a4.text(bar.get_x() + bar.get_width() / 2.0, h * 1.15, f"{h:.2f}" if h < 10 else f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    a4.set_xticks(x_vals)
    a4.set_xticklabels([f"{q:.2f}" for q in oracle_by_phi.index], fontsize=10)
    a4.set_xlabel(r"Sensing Matrix Density $q_{\Phi}$", fontsize=11, fontweight="bold")
    a4.set_ylabel(r"Mean Squared Error (log scale)", fontsize=11, fontweight="bold")
    a4.set_yscale("log")
    a4.set_ylim(0.08, 600)
    a4.set_title(r"Oracle Error $\varepsilon_{\mathrm{or}}$ vs Excess Error $\varepsilon_x^{\hat{n}} - \varepsilon_x^{n^*}$", fontsize=12, fontweight="bold")
    a4.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
    a4.legend(fontsize=8.5, loc="upper right")
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, "p8_4_oracle_vs_excess.png"), dpi=200, bbox_inches="tight")
    plt.close(fig4)


def main():
    parser = argparse.ArgumentParser(description="Joint Sensing & Validation Matrix Sparsity Cross-Product Sweep")
    parser.add_argument("--trials", type=int, default=30, help="Number of Monte Carlo trials per grid cell")
    parser.add_argument("--N", type=int, default=512, help="Signal dimension")
    parser.add_argument("--s", type=int, default=10, help="Signal sparsity")
    parser.add_argument("--m", type=int, default=120, help="Number of sensing measurements")
    parser.add_argument("--r", type=int, default=40, help="Number of validation measurements")
    parser.add_argument("--snr", type=float, default=25.0, help="Signal-to-noise ratio in dB")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save artifacts")
    args = parser.parse_args()

    run_joint_sparsity_cross_product(
        num_trials=args.trials,
        N=args.N,
        s=args.s,
        m=args.m,
        r=args.r,
        snr_db=args.snr,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
