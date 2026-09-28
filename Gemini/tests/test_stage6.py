import unittest
import numpy as np
import pandas as pd
import sys
import os

# Ensure Gemini/src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from compare_jl_hierarchy import compute_all_pairs_jl, run_jl_hierarchy_experiments


class TestStage6(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "test_stage6"))
        os.makedirs(self.test_output_dir, exist_ok=True)

    def test_compute_all_pairs_jl(self):
        N = 32
        error_vectors = np.random.normal(size=(5, N))
        Psi = np.random.normal(scale=1.0 / np.sqrt(20), size=(20, N))

        res = compute_all_pairs_jl(error_vectors, Psi, epsilon=0.25)
        self.assertIn("all_pairs_simultaneous_success", res)
        self.assertIn("all_pairs_fraction_satisfied", res)
        self.assertIn("norm_simultaneous_success", res)
        self.assertIn("norm_fraction_satisfied", res)

        self.assertGreaterEqual(res["all_pairs_fraction_satisfied"], 0.0)
        self.assertLessEqual(res["all_pairs_fraction_satisfied"], 1.0)
        self.assertGreaterEqual(res["norm_fraction_satisfied"], 0.0)
        self.assertLessEqual(res["norm_fraction_satisfied"], 1.0)

    def test_hierarchy_runner(self):
        df_q, df_r, _ = run_jl_hierarchy_experiments(
            q_values=[1.0, 0.2],
            r_values=[15, 30],
            num_trials=3,
            N=128,
            s=4,
            m=40,
            epsilon=0.20,
            snr_db=30.0,
            max_iter=8,
            base_seed=888,
            output_dir=self.test_output_dir,
        )

        self.assertEqual(len(df_q), 2)
        self.assertEqual(len(df_r), 2)

        # Check required columns
        for col in [
            "Level1_AllPairs_Strict",
            "Level2_Norms_Strict",
            "Level2B_Ward_Strict",
            "Level3_Ordering_Tau90",
            "Level4_Selection_Acc",
        ]:
            self.assertIn(col, df_q.columns)
            self.assertIn(col, df_r.columns)

        # Check outputs exist
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "jl_hierarchy_comparison.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "jl_hierarchy_q_summary.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "jl_hierarchy_r_summary.csv")))

    def test_ward_bound_stats(self):
        from metrics import ward_bound_stats
        # True norms = 1.0, so E_true = 1.0
        E_true = np.ones(5)
        # Perfect estimate: E_hat = 1.0 -> ratio = 1.0
        res = ward_bound_stats(E_true, E_true)
        self.assertTrue(res["ward_simultaneous_success"])
        self.assertEqual(res["ward_fraction_satisfied"], 1.0)

        # Boundary test:
        # Lower bound: ||e|| >= 0.8 * ||Psi e|| => ||Psi e|| <= 1.25 * ||e|| => E_hat <= 1.5625
        # Upper bound: ||e|| <= (4/3) * ||Psi e|| => ||Psi e|| >= 0.75 * ||e|| => E_hat >= 0.5625
        E_hat_inside = np.array([0.60, 0.80, 1.0, 1.20, 1.50])
        res_inside = ward_bound_stats(E_true, E_hat_inside)
        self.assertTrue(res_inside["ward_simultaneous_success"])

        E_hat_outside = np.array([0.50, 0.80, 1.0, 1.20, 1.50])  # 0.50 is outside (needs >= 0.5625)
        res_outside = ward_bound_stats(E_true, E_hat_outside)
        self.assertFalse(res_outside["ward_simultaneous_success"])
        self.assertEqual(res_outside["ward_fraction_satisfied"], 0.8)


if __name__ == "__main__":
    unittest.main()
