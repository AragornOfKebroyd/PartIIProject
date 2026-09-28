"""
Candidate Error Separation vs. Pairwise Ordering Reversal Analysis
Investigates how the relative separation between candidate errors:
    Delta_rel = |E_i - E_j| / min(E_i, E_j)
affects the empirical probability that the validation matrix Psi reverses their ordering:
    P(Reversal | Delta_rel)

This turns the qualitative observation "ordering is more tolerant than norm preservation"
into a rigorous, quantitative function.
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
from metrics import compute_cv_errors


def run_separation_reversal_analysis(
    q_values: list[float] | None = None,
    num_trials: int = 100,
    N: int = 512,
    s: int = 10,
    m: int = 120,
    r: int = 40,
    snr_db: float = 25.0,
    max_iter: int = 25,
    base_seed: int = 7070,
    num_bins: int = 14,
    output_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Measure pairwise reversal probability P(Reversal | Delta_rel) across candidate error pairs.
    """
    if q_values is None:
        q_values = [1.0, 0.5, 0.2, 0.05, 0.01]

    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("PAIRWISE REVERSAL PROBABILITY VS RELATIVE ERROR SEPARATION")
    print(f"Problem: N={N}, s={s}, m={m}, r={r}, SNR={snr_db} dB, max_iter={max_iter}")
    print(f"Densities q: {q_values} | Trials: {num_trials}")
    print("=" * 80)

    p = min(max_iter, m, N)
    bins = np.logspace(np.log10(0.01), np.log10(10.0), num_bins + 1)

    all_pair_records = []

    for t in range(num_trials):
        rng = np.random.default_rng(base_seed + t)
        x_true, _ = generate_sparse_signal(N, s, coeff_dist="gaussian", rng=rng)
        Phi = generate_sensing_matrix(m, N, dist="gaussian", normalize="column", rng=rng)
        y, _, _ = add_noise_snr(Phi @ x_true, snr_db=snr_db, rng=rng)
        traj = run_omp_trajectory(Phi, y, max_iter=p, x_true=x_true)
        E_true = traj.true_errors

        for q in q_values:
            rng_val = np.random.default_rng(base_seed + 500000 + int(q * 10000) + t)
            m_type = "dense_rademacher" if q == 1.0 else "sparse_rademacher"
            Psi = generate_projection_matrix(r, N, matrix_type=m_type, q=q, rng=rng_val)
            y_cv, _, _ = add_noise_snr(Psi @ x_true, snr_db=snr_db, rng=rng_val)
            E_hat = compute_cv_errors(Psi, y_cv, traj.iterates)

            for i in range(len(E_true)):
                for j in range(i + 1, len(E_true)):
                    e_min = min(E_true[i], E_true[j])
                    if e_min > 1e-8:
                        delta_rel = abs(E_true[i] - E_true[j]) / e_min
                        is_reversed = (E_true[i] < E_true[j] and E_hat[i] > E_hat[j]) or (
                            E_true[i] > E_true[j] and E_hat[i] < E_hat[j]
                        )
                        all_pair_records.append({
                            "trial": t,
                            "q": q,
                            "delta_rel": delta_rel,
                            "is_reversed": int(is_reversed),
                        })

    df_pairs = pd.DataFrame(all_pair_records)
    df_pairs["bin"] = pd.cut(df_pairs["delta_rel"], bins)

    # Bin midpoint calculation
    bin_centers = [np.sqrt(bins[k] * bins[k + 1]) for k in range(len(bins) - 1)]
    bin_map = {b: c for b, c in zip(pd.cut(bin_centers, bins).categories, bin_centers)}

    # Summary table
    grouped = df_pairs.groupby(["q", "bin"], observed=False)["is_reversed"].agg(["mean", "count"]).reset_index()
    grouped["bin_center"] = grouped["bin"].map(bin_map)
    grouped.rename(columns={"mean": "reversal_probability", "count": "pair_count"}, inplace=True)

    summary_csv_path = os.path.join(output_dir, "pairwise_reversal_summary.csv")
    grouped.to_csv(summary_csv_path, index=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {
        1.0: "black",
        0.5: "darkgreen",
        0.2: "royalblue",
        0.05: "crimson",
        0.01: "purple",
    }
    labels = {
        1.0: "Dense Rademacher (q=1.0)",
        0.5: "Sparse (q=0.50)",
        0.2: "Sparse (q=0.20)",
        0.05: "Ultra-Sparse (q=0.05)",
        0.01: "Ultra-Sparse (q=0.01)",
    }

    for q in q_values:
        sub = grouped[grouped["q"] == q].dropna(subset=["reversal_probability"])
        ax.plot(
            sub["bin_center"],
            sub["reversal_probability"] * 100,
            "-o",
            color=colors.get(q, "grey"),
            label=labels.get(q, f"q={q}"),
            linewidth=2.0,
            markersize=5,
        )

    ax.axhline(50.0, color="gray", linestyle=":", alpha=0.7, label="Coin Flip (50% Reversal)")
    ax.set_xscale("log")
    ax.set_xlabel(r"Relative Error Separation $\Delta_{\mathrm{rel}} = \frac{|E_i - E_j|}{\min(E_i, E_j)}$", fontsize=11)
    ax.set_ylabel("Pairwise Reversal Probability (%)", fontsize=11)
    ax.set_title(r"Pairwise Reversal Probability vs Error Separation $\Delta_{\mathrm{rel}}$ ($r=40$)", fontsize=12, fontweight="bold")
    ax.set_ylim(-2, 60)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "pairwise_reversal_vs_separation.png")
    pdf_path = os.path.join(output_dir, "pairwise_reversal_vs_separation.pdf")
    plt.savefig(plot_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()

    print(f"Saved reversal plot to: {plot_path}")
    print(f"Saved reversal data to: {summary_csv_path}")

    return grouped, df_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze pairwise ordering reversal vs error separation")
    parser.add_argument("--trials", type=int, default=50, help="Number of random trials")
    parser.add_argument("--N", type=int, default=512, help="Signal dimension")
    parser.add_argument("--s", type=int, default=10, help="Sparsity")
    parser.add_argument("--m", type=int, default=120, help="Measurement rows")
    parser.add_argument("--r", type=int, default=40, help="Validation rows")
    parser.add_argument("--snr", type=float, default=25.0, help="SNR in dB")
    args = parser.parse_args()

    run_separation_reversal_analysis(
        num_trials=args.trials,
        N=args.N,
        s=args.s,
        m=args.m,
        r=args.r,
        snr_db=args.snr,
    )
