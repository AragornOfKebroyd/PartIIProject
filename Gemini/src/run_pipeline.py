"""
Master Pipeline Automation Script
Runs all stages of the Compressed Sensing Cross-Validation research pipeline.

Modes:
  --mode fast : Runs quick verification of all stages (~10 seconds).
  --mode full : Runs full statistical reproduction (100 trials per experiment).
  --stage N   : Runs a specific stage (2 through 7).
"""

import os
import sys
import argparse
import time

# Ensure src directory is in sys.path
SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from experiments.plot_single_trajectory import run_and_plot_single_trajectory
from experiments.run_trajectory_ensemble import run_ensemble
from experiments.evaluate_three_notions import evaluate_three_notions
from experiments.compare_matrix_families import compare_matrix_families
from experiments.compare_budgets import run_budget_experiments
from experiments.compare_jl_hierarchy import run_jl_hierarchy_experiments
from experiments.generate_heatmaps import run_heatmap_grid, run_regime_scaling_sweep


def run_pipeline(mode: str = "fast", target_stage: int | None = None):
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    trials = 5 if mode == "fast" else 100
    N = 128 if mode == "fast" else 512
    s = 4 if mode == "fast" else 10
    m = 40 if mode == "fast" else 120
    max_iter = 10 if mode == "fast" else 25

    print("=" * 80)
    print(f"STARTING COMPRESSED SENSING RESEARCH PIPELINE (MODE: {mode.upper()})")
    print(f"Base Parameters: N={N}, s={s}, m={m}, max_iter={max_iter}, trials={trials}")
    print("=" * 80)

    start_time = time.time()

    # Stage 2: Single Trajectory & Trajectory Ensemble
    if target_stage is None or target_stage == 2:
        print("\n>>> [STAGE 2] Running Single Trajectory & Ensemble Experiments...")
        t0 = time.time()
        run_and_plot_single_trajectory(N=N, s=s, m=m, r=40, q=0.2, snr_db=25.0, max_iter=max_iter, output_dir=output_dir)
        run_ensemble(num_trials=trials, N=N, s=s, m=m, r=40, q_sparse=0.2, q_ultra_sparse=0.05, max_iter=max_iter, output_dir=output_dir)
        print(f"Stage 2 completed in {time.time() - t0:.2f}s")

    # Stage 3: Three Notions of "Works"
    if target_stage is None or target_stage == 3:
        print("\n>>> [STAGE 3] Running Three Notions of Works vs Sparsity...")
        t0 = time.time()
        q_vals = [1.0, 0.5, 0.2, 0.05] if mode == "fast" else [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
        evaluate_three_notions(q_values=q_vals, num_trials=trials, N=N, s=s, m=m, r=40, max_iter=max_iter, output_dir=output_dir)
        print(f"Stage 3 completed in {time.time() - t0:.2f}s")

    # Stage 4: Matrix Families
    if target_stage is None or target_stage == 4:
        print("\n>>> [STAGE 4] Comparing Matrix Distribution Families...")
        t0 = time.time()
        r_vals = [20, 40] if mode == "fast" else [10, 20, 30, 40, 60, 80, 100]
        compare_matrix_families(r_values=r_vals, num_trials=trials, N=N, s=s, m=m, max_iter=max_iter, output_dir=output_dir)
        print(f"Stage 4 completed in {time.time() - t0:.2f}s")

    # Stage 5: Budget Matched Experiments
    if target_stage is None or target_stage == 5:
        print("\n>>> [STAGE 5] Running Budget Matched Comparisons (Fixed r vs Fixed nnz)...")
        t0 = time.time()
        run_budget_experiments(num_trials=trials, N=N, s=s, m=m, max_iter=max_iter, output_dir=output_dir)
        print(f"Stage 5 completed in {time.time() - t0:.2f}s")

    # Stage 6: JL Hierarchy
    if target_stage is None or target_stage == 6:
        print("\n>>> [STAGE 6] Testing General JL Requirement vs Task Hierarchy...")
        t0 = time.time()
        q_vals = [1.0, 0.2] if mode == "fast" else [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
        r_vals = [20, 40] if mode == "fast" else [10, 20, 30, 40, 60, 80, 100]
        run_jl_hierarchy_experiments(q_values=q_vals, r_values=r_vals, num_trials=trials, N=N, s=s, m=m, max_iter=max_iter, output_dir=output_dir)
        print(f"Stage 6 completed in {time.time() - t0:.2f}s")

    # Stage 7: 2D Phase Transitions
    if target_stage is None or target_stage == 7:
        print("\n>>> [STAGE 7] Computing 2D Phase Transition Heatmaps...")
        t0 = time.time()
        q_vals = [1.0, 0.2, 0.05] if mode == "fast" else [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        r_vals = [20, 40, 80] if mode == "fast" else [10, 20, 30, 40, 60, 80, 100, 150, 200]
        run_heatmap_grid(q_values=q_vals, r_values=r_vals, num_trials=trials, N=N, s=s, m=m, max_iter=max_iter, output_dir=output_dir)

        if mode == "full":
            run_regime_scaling_sweep(num_trials=50, output_dir=output_dir)
        print(f"Stage 7 completed in {time.time() - t0:.2f}s")

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed:.2f} SECONDS!")
    print(f"All figures and CSV results are saved in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master automation runner for CS cross-validation pipeline")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast", help="Execution mode (fast for testing, full for production)")
    parser.add_argument("--stage", type=int, choices=[2, 3, 4, 5, 6, 7], default=None, help="Run a specific stage only")
    args = parser.parse_args()

    run_pipeline(mode=args.mode, target_stage=args.stage)
