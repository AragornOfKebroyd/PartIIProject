import unittest
import numpy as np
import sys
import os

# Ensure Gemini/src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from signals import generate_sparse_signal, sparsity_of
from sensing import generate_sensing_matrix, add_noise, add_noise_snr
from omp import run_omp_trajectory
from projections import (
    generate_projection_matrix,
    compute_projection_stats,
    budget_matched_dense_rows,
)
from metrics import (
    compute_cv_errors,
    relative_norm_errors,
    kendall_tau,
    pairwise_concordance,
    best_iterate_selection,
    jl_distortion_violation_rate,
)


class TestPhase1(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_signals(self):
        N = 100
        s = 5
        x, support = generate_sparse_signal(N, s, coeff_dist="gaussian", rng=self.rng)
        self.assertEqual(len(x), N)
        self.assertEqual(len(support), s)
        self.assertEqual(sparsity_of(x), s)
        self.assertTrue(np.all(x[support] != 0))

        # Rademacher coefficients
        x_rad, _ = generate_sparse_signal(N, s, coeff_dist="rademacher", rng=self.rng)
        nonzeros = x_rad[x_rad != 0]
        self.assertTrue(np.all(np.isin(nonzeros, [-1.0, 1.0])))

    def test_sensing_and_noise(self):
        m, N = 40, 100
        Phi = generate_sensing_matrix(m, N, dist="gaussian", normalize="column", rng=self.rng)
        self.assertEqual(Phi.shape, (m, N))
        col_norms = np.linalg.norm(Phi, axis=0)
        np.testing.assert_allclose(col_norms, 1.0, atol=1e-10)

        # Noise SNR test
        y = np.ones(m)
        y_noisy, noise, sigma = add_noise_snr(y, snr_db=20.0, rng=self.rng)
        self.assertEqual(y_noisy.shape, (m,))
        self.assertGreater(sigma, 0.0)

    def test_omp_noiseless_exact_recovery(self):
        N = 64
        s = 4
        m = 25
        x_true, support = generate_sparse_signal(N, s, coeff_dist="gaussian", rng=self.rng)
        Phi = generate_sensing_matrix(m, N, dist="gaussian", normalize="column", rng=self.rng)
        y = Phi @ x_true

        traj = run_omp_trajectory(Phi, y, max_iter=2 * s, x_true=x_true)
        # Check trajectory lengths
        self.assertGreaterEqual(traj.p, s)
        self.assertEqual(len(traj.iterates), traj.p + 1)
        self.assertEqual(len(traj.residuals), traj.p + 1)
        self.assertIsNotNone(traj.true_errors)

        # At iteration s, error should be near zero for well-conditioned random Gaussian matrix
        min_err_idx = int(np.argmin(traj.true_errors))
        self.assertLess(traj.true_errors[min_err_idx], 1e-10)

        # Check sparsity constraint of error vectors ||x - x^j||_0 <= s + j
        for j in range(traj.p + 1):
            self.assertLessEqual(traj.error_sparsities[j], s + j)

    def test_projections_unbiasedness_and_density(self):
        r, N = 50, 100
        # Test Gaussian
        Psi_g = generate_projection_matrix(r, N, matrix_type="gaussian", rng=self.rng)
        self.assertEqual(Psi_g.shape, (r, N))

        # Test Sparse Rademacher
        q = 0.2
        Psi_s = generate_projection_matrix(r, N, matrix_type="sparse_rademacher", q=q, rng=self.rng)
        stats = compute_projection_stats(Psi_s)
        self.assertAlmostEqual(stats["empirical_density"], q, delta=0.05)

        # Check unbiased norm preservation: E[ ||Psi v||_2^2 ] = ||v||_2^2
        v = self.rng.normal(size=N)
        true_norm_sq = float(np.sum(v ** 2))
        proj_norm_sqs = []
        for _ in range(500):
            Psi = generate_projection_matrix(r, N, matrix_type="sparse_rademacher", q=q, rng=self.rng)
            proj_norm_sqs.append(float(np.sum((Psi @ v) ** 2)))

        mean_proj = np.mean(proj_norm_sqs)
        self.assertAlmostEqual(mean_proj / true_norm_sq, 1.0, delta=0.08)

    def test_budget_matching(self):
        r_sparse = 100
        q = 0.1
        r_dense = budget_matched_dense_rows(r_sparse, q)
        self.assertEqual(r_dense, 10)

    def test_metrics(self):
        E_true = np.array([10.0, 5.0, 2.0, 0.5, 0.1, 0.3])
        # Monotonically consistent estimate
        E_hat = np.array([11.0, 5.2, 2.1, 0.6, 0.15, 0.35])

        rel_err = relative_norm_errors(E_true, E_hat)
        self.assertEqual(len(rel_err), len(E_true))
        self.assertLess(rel_err[0], 0.2)

        tau = kendall_tau(E_true, E_hat)
        self.assertAlmostEqual(tau, 1.0, places=5)

        concordance = pairwise_concordance(E_true, E_hat)
        self.assertAlmostEqual(concordance, 1.0, places=5)

        best_sel = best_iterate_selection(E_true, E_hat)
        self.assertEqual(best_sel["j_true_star"], 4)
        self.assertEqual(best_sel["j_hat_star"], 4)
        self.assertTrue(best_sel["is_exact_match"])
        self.assertAlmostEqual(best_sel["excess_error"], 0.0)

        # Test JL violation rate
        errors = np.array([[1.0, 0.0], [0.0, 1.0]])
        Psi_id = np.array([[1.0, 0.0], [0.0, 1.0]])
        viol = jl_distortion_violation_rate(errors, Psi_id, epsilon=0.1)
        self.assertEqual(viol, 0.0)


if __name__ == "__main__":
    unittest.main()
