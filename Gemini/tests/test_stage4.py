import unittest
import numpy as np
import pandas as pd
import sys
import os

# Ensure Gemini/src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from compare_matrix_families import compare_matrix_families


class TestStage4(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "test_stage4"))
        os.makedirs(self.test_output_dir, exist_ok=True)

    def test_compare_matrix_families_runner(self):
        r_vals = [10, 20]
        df_summary, df_trials = compare_matrix_families(
            r_values=r_vals,
            num_trials=3,
            N=128,
            s=4,
            m=40,
            snr_db=30.0,
            max_iter=8,
            base_seed=777,
            output_dir=self.test_output_dir,
        )

        # 5 families * len(r_vals)
        self.assertEqual(len(df_summary), 5 * len(r_vals))
        self.assertEqual(len(df_trials), 3 * 5 * len(r_vals))

        expected_cols = [
            "r", "family_key", "Family", "q", "JL_Success_Eps20",
            "Distortion_Std", "PreConv_Rel_Error", "Mean_Kendall_Tau",
            "Selection_Accuracy", "Near_Selection_Accuracy"
        ]
        for col in expected_cols:
            self.assertIn(col, df_summary.columns)

        # Check outputs
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "matrix_families_comparison.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "matrix_families_summary.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "matrix_families_trials.csv")))


if __name__ == "__main__":
    unittest.main()
