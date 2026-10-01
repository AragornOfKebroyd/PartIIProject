import unittest
import numpy as np
import pandas as pd
import sys
import os

# Ensure Gemini/src is in sys.path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from experiments.p2_plot_single_trajectory import run_and_plot_single_trajectory
from experiments.p2_run_trajectory_ensemble import run_ensemble


class TestStage2(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "test_stage2"))
        os.makedirs(self.test_output_dir, exist_ok=True)

    def test_single_trajectory_runner(self):
        df = run_and_plot_single_trajectory(
            N=128,
            s=4,
            m=40,
            r=20,
            q=0.3,
            snr_db=30.0,
            max_iter=10,
            seed=123,
            output_dir=self.test_output_dir,
        )
        self.assertIsInstance(df, pd.DataFrame)
        expected_cols = ["iteration", "true_error", "error_sparsity", "cv_gauss", "cv_rademacher", "cv_sparse"]
        for col in expected_cols:
            self.assertIn(col, df.columns)
        self.assertEqual(len(df), 11)  # 0 to 10 iterations

        # Check plot exists
        plot_path = os.path.join(self.test_output_dir, "p2_single_trajectory.png")
        self.assertTrue(os.path.exists(plot_path))

    def test_ensemble_runner_small(self):
        df_summary, df_trials = run_ensemble(
            num_trials=5,
            N=128,
            s=4,
            m=40,
            r=20,
            q_sparse=0.3,
            q_ultra_sparse=0.1,
            snr_db=30.0,
            max_iter=8,
            base_seed=999,
            output_dir=self.test_output_dir,
        )
        self.assertEqual(len(df_trials), 5 * 4)  # 5 trials * 4 models
        self.assertEqual(len(df_summary), 4)

        for _, row in df_summary.iterrows():
            self.assertGreaterEqual(row["Selection_Accuracy"], 0.0)
            self.assertLessEqual(row["Selection_Accuracy"], 1.0)
            self.assertGreaterEqual(row["Mean_Kendall_Tau"], -1.0)
            self.assertLessEqual(row["Mean_Kendall_Tau"], 1.0)

        # Check output plot exists
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "p2_ensemble_trajectories.png")) or os.path.exists(os.path.join(self.test_output_dir, "ensemble_trajectories.png")))


if __name__ == "__main__":
    unittest.main()
