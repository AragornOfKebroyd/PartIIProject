# Gemini Compressed Sensing (CV-CS) Library & Experiment Suite

A modular, readable, and reproducible Python framework for researching **sparse validation projection matrices in compressed sensing cross-validation (OMP-CV)**.

---

## 1. Quickstart & Environment

All code is developed and verified using the project's virtual environment:
```bash
# Python interpreter location
/home/aragorn/Documents/PartIIProject/venv/bin/python
```

### Running Unit Tests
All unit tests are located in `Gemini/tests/` and cover all mathematical modules:
```bash
venv/bin/python -m unittest discover Gemini/tests
```

### Running the Complete Automated Pipeline
To reproduce all stages with a single command:
```bash
venv/bin/python Gemini/src/run_pipeline.py --mode full
```
Or for a fast test run (reduced trials):
```bash
venv/bin/python Gemini/src/run_pipeline.py --mode fast
```

---

## 2. Project Directory Structure

```
PartIIProject/
├── OMP_CV.py                          # UNTOUCHED legacy script
├── venv/                              # Python 3.14 virtual environment
└── Gemini/                            # All new prototype & experiment code
    ├── ARCHITECTURE.md                # System design & mathematical formulation
    ├── IMPLEMENTATION_PLAN.md         # Phased roadmap & specifications
    ├── FINDINGS.md                    # Comprehensive research writeup & results
    ├── README.md                      # This user guide
    ├── run_all.sh                     # Shell automation script
    ├── src/                           # Core library & experiment scripts
    │   ├── groundwork/                # Mathematical foundation modules
    │   │   ├── signals.py             # Sparse vector generator
    │   │   ├── matrices.py            # Unified sensing Phi & validation Psi generator
    │   │   ├── omp.py                 # OMP retaining complete iterate trajectory
    │   │   └── metrics.py             # Norm error, Kendall's tau, selection acc, Ward bounds
    │   ├── experiments/               # Evaluation scripts and parameter sweeps
    │   │   ├── plot_single_trajectory.py  # Stage 2: Single trajectory visualizer
    │   │   ├── run_trajectory_ensemble.py # Stage 2: Ensemble trajectory statistics
    │   │   ├── evaluate_three_notions.py  # Stage 3: Norm error vs Ordering vs Selection
    │   │   ├── compare_matrix_families.py # Stage 4: Matrix distribution families
    │   │   ├── compare_budgets.py         # Stage 5: Fixed-r vs Fixed-nnz (Iso-Compute)
    │   │   ├── compare_jl_hierarchy.py    # Stage 6: All-pairs JL vs Task hierarchy
    │   │   ├── analyze_separation_reversal.py # Stage 6b: Candidate error separation vs reversals
    │   │   ├── generate_heatmaps.py       # Stage 7: 2D Phase Transitions & Scaling
    │   │   └── compare_joint_sparsity.py  # Stage 7b: Joint sensing & validation sparsity cross-product
    │   └── run_pipeline.py            # Master pipeline runner
    ├── output/                        # All generated figures (.png) and data (.csv)
    └── tests/                         # Unit tests (test_phase1 to test_joint_sparsity)
```

---

## 3. Groundwork Modules (`Gemini/src/groundwork/`)

### `signals.py`
- `generate_sparse_signal(N, s, coeff_dist="gaussian", rng=None, **kwargs)`:
  Generates an $s$-sparse vector of length $N$. Supports `"gaussian"`, `"rademacher"`, `"uniform"` $[-b, -a] \cup [a, b]$, and `"decay"` (power-law decay $k^{-\alpha}$).
- `sparsity_of(v, tol=1e-8)`: Returns the number of non-zero entries.

### `matrices.py` (Unified Matrix Engine)
- `generate_measurement_matrix(rows, cols, dist="sparse_rademacher", q=1.0, normalize="column", rng=None)`:
  Unified mathematical generator for sensing matrices $\Phi \in \mathbb{R}^{m \times N}$ and validation projection matrices $\Psi \in \mathbb{R}^{r \times N}$.
  Supports `"gaussian"`, `"rademacher"`, `"sparse_rademacher"`, and `"sparse_gaussian"` with arbitrary non-zero density $q \in (0, 1]$.
  Supports column normalization (`normalize="column"`, $\|\mathbf{a}_j\|_2 = 1$) and variance normalization (`normalize="variance"`, $\mathbb{E}[A_{ij}^2] = 1/\text{rows}$).
- `generate_sensing_matrix(...)` & `generate_projection_matrix(...)`: Backward-compatible wrappers for $\Phi$ and $\Psi$.
- `add_noise(y, sigma, rng=None)` & `add_noise_snr(y, snr_db, rng=None)`: Adds Gaussian noise with specified standard deviation or target SNR in dB.
- `compute_matrix_stats(A)`: Computes empirical density, non-zeros, and variance.
- `budget_matched_dense_rows(r_sparse, q)`: Calculates $r_{\text{dense}} = \max(1, \text{round}(q \cdot r_{\text{sparse}}))$.

### `omp.py`
- `run_omp_trajectory(Phi, y, max_iter, x_true=None, tol=1e-12)`:
  Solves OMP and preserves the **entire sequence** of states:
  - `traj.iterates`: shape $(p+1, N)$ containing $\mathbf{x}^0 = \mathbf{0}, \mathbf{x}^1, \dots, \mathbf{x}^p$.
  - `traj.true_errors`: squared error $E_j = \|\mathbf{x} - \mathbf{x}^j\|_2^2$.
  - `traj.error_sparsities`: active support size of error $\|\mathbf{x} - \mathbf{x}^j\|_0 \le s + j$.
  - `traj.residuals`: residual vectors $\mathbf{r}^j = \mathbf{y} - \Phi \mathbf{x}^j$.

### `metrics.py`
- `compute_cv_errors(Psi, y_cv, iterates)`: Computes $\widehat{E}_j = \|\mathbf{y}_\Psi - \Psi \mathbf{x}^j\|_2^2$ vectorized.
- `relative_norm_errors(E_true, E_hat)`: Computes $|\widehat{E}_j - E_j| / E_j$.
- `pre_convergence_relative_errors(E_true, E_hat, j_star)`: Relative error restricted to $j \le j^*$ (avoids noise floor).
- `kendall_tau(E_true, E_hat)` & `pairwise_concordance(E_true, E_hat)`: Measures rank correlation and pairwise monotonicity.
- `best_iterate_selection(E_true, E_hat)`: Evaluates exact match $\mathbf{1}[\hat{j} == j^*]$, near match $|\hat{j} - j^*| \le 1$, and excess oracle error.
- `ward_bound_stats(E_true, E_hat)`: Evaluates Rachel Ward's (2009) theoretical cross-validation bounds ($4/5 \|\Psi \mathbf{e}^j\|_2 \le \|\mathbf{e}^j\|_2 \le 4/3 \|\Psi \mathbf{e}^j\|_2$) across all iterates.
- `jl_isometry_stats(error_vectors, Psi, epsilons=(0.2, 0.5))`: Tests classical $(1 \pm \varepsilon)$ isometry and distortion spread $\operatorname{Std}(\|\Psi \mathbf{e}\|^2 / \|\mathbf{e}\|^2)$.

---

## 4. Running Individual Experiment Scripts

### Stage 2: Single Trajectory & Ensemble Statistics
```bash
# 1. Single Trajectory Visualizer (Generates single_trajectory.png)
venv/bin/python Gemini/src/experiments/plot_single_trajectory.py --N 512 --s 10 --m 120 --r 40 --q 0.2 --snr 25.0

# 2. Ensemble Statistics across 100 trials (Generates ensemble_trajectories.png)
venv/bin/python Gemini/src/experiments/run_trajectory_ensemble.py --trials 100 --N 512 --s 10 --m 120 --r 40 --q_sparse 0.2 --q_ultra_sparse 0.05
```

### Stage 3: The Three Notions of "Works" vs Sparsity $q$
```bash
# Sweeps q in [1.0 ... 0.01] (Generates three_notions_vs_sparsity.png)
venv/bin/python Gemini/src/experiments/evaluate_three_notions.py --trials 100 --N 512 --s 10 --m 120 --r 40 --snr 25.0
```

### Stage 4: Matrix Distribution Families Comparison
```bash
# Compares Gaussian vs Rademacher vs Sparse Rademacher across r in [10 ... 100] (Generates matrix_families_comparison.png)
venv/bin/python Gemini/src/experiments/compare_matrix_families.py --trials 100 --N 512 --s 10 --m 120 --snr 25.0
```

### Stage 5: Budget-Matched Experiments (Fixed $r$ vs Fixed $\operatorname{nnz}$)
```bash
# Tests whether a "wide-and-sparse" matrix beats a "narrow-and-dense" matrix at equal FLOP budget (Generates budget_matched_comparison.png)
venv/bin/python Gemini/src/experiments/compare_budgets.py --trials 100 --N 512 --s 10 --m 120 --snr 25.0
```

### Stage 6: General JL Requirement vs Task-Specific Hierarchy
```bash
# Evaluates All-pairs JL vs Norms vs Ordering vs Model Selection on P={0, e^1, ..., e^p} (Generates jl_hierarchy_comparison.png)
venv/bin/python Gemini/src/experiments/compare_jl_hierarchy.py --trials 100 --N 512 --s 10 --m 120 --eps 0.20 --snr 25.0
```

### Stage 6b: Candidate Error Separation vs Pairwise Ordering Reversal
```bash
# Quantifies reversal probability P(Reversal | Delta_rel) where Delta_rel = |E_i - E_j| / min(E_i, E_j)
venv/bin/python Gemini/src/experiments/analyze_separation_reversal.py --trials 50 --N 512 --s 10 --m 120 --r 40 --snr 25.0
```

### Stage 7: 2D Phase Transition Heatmaps and Scaling
```bash
# Computes 63-cell (q, r) phase diagram + N & s scaling sweep (Generates heatmap_selection_accuracy.png, heatmap_kendall_tau.png, heatmap_regime_scaling.png)
venv/bin/python Gemini/src/experiments/generate_heatmaps.py --trials 100 --N 512 --s 10 --m 120 --snr 25.0 --run_scaling
```

### Stage 7b: Joint Sensing & Validation Sparsity Cross-Product
```bash
# Evaluates 35-cell cross-product grid q_Phi in [1.0 .. 0.05] x q_Psi in [1.0 .. 0.01] (Generates joint_sparsity_comparison.png)
venv/bin/python Gemini/src/experiments/compare_joint_sparsity.py --trials 50 --N 512 --s 10 --m 120 --r 40 --snr 25.0
```

---

## 5. Output Deliverables Directory (`Gemini/output/`)

| Filename | Description |
| :--- | :--- |
| [`joint_sparsity_comparison.png`](file:///home/aragorn/Documents/PartIIProject/Gemini/output/joint_sparsity_comparison.png) | 4-panel cross-product comparison: Sensing density $q_\Phi$ vs Validation density $q_\Psi$ |
| [`single_trajectory.png`](file:///home/aragorn/Documents/PartIIProject/Gemini/output/single_trajectory.png) | Single trajectory plot ($E_j$ vs $\widehat{E}_j$) with error sparsity panel |
| [`ensemble_trajectories.png`](file:///home/aragorn/Documents/PartIIProject/Gemini/output/ensemble_trajectories.png) | 100-trial ensemble median & IQR ribbon trajectories, Kendall $\tau$ boxplots, accuracy bars |
| [`three_notions_vs_sparsity.png`](file:///home/aragorn/Documents/PartIIProject/Gemini/output/three_notions_vs_sparsity.png) | 4-panel comparison of Norm Error vs Ordering vs Selection vs Hierarchy Retention |
| [`matrix_families_comparison.png`](file:///home/aragorn/Documents/PartIIProject/Gemini/output/matrix_families_comparison.png) | Comparison of Gaussian vs Rademacher vs Sparse Rademacher verifying kurtosis $\gamma$ |
| [`budget_matched_comparison.png`](file:///home/aragorn/Documents/PartIIProject/Gemini/output/budget_matched_comparison.png) | The "Wide-and-Sparse" advantage at fixed FLOP budget $\operatorname{nnz}(\Psi)$ |
| [`jl_hierarchy_comparison.png`](file:///home/aragorn/Documents/PartIIProject/Gemini/output/jl_hierarchy_comparison.png) | 5-level hierarchy success rates including Rachel Ward's bound and required measurement budget comparison |
| [`pairwise_reversal_vs_separation.png`](file:///home/aragorn/Documents/PartIIProject/Gemini/output/pairwise_reversal_vs_separation.png) | Pairwise reversal probability vs candidate error separation $\Delta_{\text{rel}} = \frac{\|E_i - E_j\|}{\min(E_i, E_j)}$ |
| [`heatmap_selection_accuracy.png`](file:///home/aragorn/Documents/PartIIProject/Gemini/output/heatmap_selection_accuracy.png) | 2D $(q, r) \mapsto \text{Accuracy}$ contour phase map for Exact and Near Selection |
| [`heatmap_kendall_tau.png`](file:///home/aragorn/Documents/PartIIProject/Gemini/output/heatmap_kendall_tau.png) | 2D $(q, r) \mapsto \tau$ contour phase map for rank correlation |
| [`heatmap_regime_scaling.png`](file:///home/aragorn/Documents/PartIIProject/Gemini/output/heatmap_regime_scaling.png) | Scaling curves across dimensions $N \in \{256, 512, 1024\}$ and sparsities $s \in \{5, 10, 20\}$ |
| `*.csv` | Structured data tables containing numerical trial records and summary aggregations |

