import unittest
import numpy as np
import pandas as pd
import sys
import os

# Ensure Gemini/src is in sys.path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from experiments.evaluate_three_notions import evaluate_three_notions


class TestStage3(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "test_stage3"))
        os.makedirs(self.test_output_dir, exist_ok=True)

    def test_three_notions_runner(self):
        q_vals = [1.0, 0.5, 0.1]
        df_summary, df_trials = evaluate_three_notions(
            q_values=q_vals,
            num_trials=4,
            N=128,
            s=4,
            m=40,
            r=20,
            snr_db=30.0,
            max_iter=8,
            base_seed=555,
            output_dir=self.test_output_dir,
        )

        self.assertEqual(len(df_summary), len(q_vals))
        self.assertEqual(len(df_trials), 4 * len(q_vals))

        expected_summary_cols = [
            "q", "nnz_per_row", "Median_Rel_Error", "Mean_Kendall_Tau",
            "Mean_Concordance", "Selection_Accuracy", "Near_Selection_Accuracy"
        ]
        for col in expected_summary_cols:
            self.assertIn(col, df_summary.columns)

        # Check output plots and CSVs exist
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "three_notions_vs_sparsity.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "three_notions_summary.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "three_notions_trials.csv")))


if __name__ == "__main__":
    unittest.main()
