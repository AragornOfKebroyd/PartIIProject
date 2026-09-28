import unittest
import numpy as np
import pandas as pd
import sys
import os

# Ensure Gemini/src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from compare_budgets import run_budget_experiments


class TestStage5(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "test_stage5"))
        os.makedirs(self.test_output_dir, exist_ok=True)

    def test_budget_comparison_runner(self):
        df_r, df_nnz, _ = run_budget_experiments(
            num_trials=3,
            N=128,
            s=4,
            m=40,
            snr_db=30.0,
            max_iter=8,
            base_seed=1212,
            output_dir=self.test_output_dir,
        )

        self.assertGreater(len(df_r), 0)
        self.assertGreater(len(df_nnz), 0)

        # Check required columns exist
        self.assertIn("r", df_r.columns)
        self.assertIn("budget_nnz", df_nnz.columns)
        self.assertIn("Selection_Accuracy", df_nnz.columns)
        self.assertIn("Mean_Kendall_Tau", df_nnz.columns)

        # Check outputs exist
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "budget_matched_comparison.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "budget_fixed_r_summary.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "budget_fixed_nnz_summary.csv")))


if __name__ == "__main__":
    unittest.main()
