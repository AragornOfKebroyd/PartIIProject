import unittest
import numpy as np
import pandas as pd
import sys
import os

# Ensure Gemini/src is in sys.path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from groundwork.matrices import (
    generate_measurement_matrix,
    generate_sensing_matrix,
    generate_projection_matrix,
    compute_matrix_stats,
    add_noise_snr,
)
from experiments.compare_joint_sparsity import run_joint_sparsity_cross_product


class TestJointSparsity(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "output", "test_joint_sparsity")
        )
        os.makedirs(self.test_output_dir, exist_ok=True)

    def test_unified_measurement_matrix_properties(self):
        # 1. Test Gaussian
        A_gauss = generate_measurement_matrix(rows=50, cols=100, dist="gaussian", normalize="column")
        self.assertEqual(A_gauss.shape, (50, 100))
        np.testing.assert_allclose(np.linalg.norm(A_gauss, axis=0), 1.0, atol=1e-5)

        # 2. Test Sparse Rademacher with variance normalization
        q = 0.1
        A_sparse = generate_measurement_matrix(rows=100, cols=200, dist="sparse_rademacher", q=q, normalize="variance")
        self.assertEqual(A_sparse.shape, (100, 200))
        stats = compute_matrix_stats(A_sparse)
        self.assertAlmostEqual(stats["empirical_density"], q, delta=0.05)
        # Expected variance = 1/rows = 0.01
        self.assertAlmostEqual(stats["empirical_variance"], 0.01, delta=0.005)

        # 3. Test backward-compatible aliases
        Phi = generate_sensing_matrix(30, 80, dist="gaussian")
        self.assertEqual(Phi.shape, (30, 80))
        Psi = generate_projection_matrix(25, 80, matrix_type="sparse_rademacher", q=0.2)
        self.assertEqual(Psi.shape, (25, 80))

    def test_joint_sparsity_cross_product_runner(self):
        # Fast runner test with minimal grid
        q_phi_vals = [1.0, 0.2]
        q_psi_vals = [1.0, 0.1]

        df_summary, df_trials = run_joint_sparsity_cross_product(
            q_phi_values=q_phi_vals,
            q_psi_values=q_psi_vals,
            num_trials=2,
            N=64,
            s=3,
            m=25,
            r=15,
            snr_db=25.0,
            max_iter=5,
            output_dir=self.test_output_dir,
        )

        self.assertIsInstance(df_summary, pd.DataFrame)
        self.assertIsInstance(df_trials, pd.DataFrame)
        self.assertEqual(len(df_summary), len(q_phi_vals) * len(q_psi_vals))
        self.assertIn("selection_accuracy", df_summary.columns)
        self.assertIn("kendall_tau_mean", df_summary.columns)

        # Check that artifacts were written
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "joint_sparsity_comparison.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "joint_sparsity_cross_product_summary.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, "joint_sparsity_cross_product_trials.csv")))


if __name__ == "__main__":
    unittest.main()
