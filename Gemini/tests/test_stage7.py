import unittest
import numpy as np
import pandas as pd
import sys
import os

# Ensure Gemini/src is in sys.path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from experiments.generate_heatmaps import run_heatmap_grid, run_regime_scaling_sweep


class TestStage7(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "test_stage7"))
        os.makedirs(self.test_output_dir, exist_ok=True)

    def test_heatmap_runner_small(self):
        q_vals = [1.0, 0.2]
        r_vals = [15, 30]
        df_grid = run_heatmap_grid(
            q_values=q_vals,
            r_values=r_vals,
            num_trials=3,
            N=128,
            s=4,
            m=40,
            snr_db=30.0,
            max_iter=8,
            base_seed=1313,
            output_dir=self.test_output_dir,
        )

        self.assertEqual(len(df_grid), len(q_vals) * len(r_vals))
        for col in ["selection_accuracy", "near_selection_accuracy", "mean_kendall_tau"]:
            self.assertIn(col, df_grid.columns)

        # Check outputs exist
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "heatmap_grid_results.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "heatmap_selection_accuracy.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "heatmap_kendall_tau.png")))

    def test_scaling_runner_small(self):
        df_scale = run_regime_scaling_sweep(
            N_values=[128],
            s_values=[4],
            num_trials=2,
            r=20,
            q_values=[1.0, 0.2],
            snr_db=30.0,
            base_seed=1414,
            output_dir=self.test_output_dir,
        )
        self.assertEqual(len(df_scale), 2)
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "heatmap_regime_scaling.png")))


if __name__ == "__main__":
    unittest.main()
