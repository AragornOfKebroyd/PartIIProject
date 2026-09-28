# Gemini Compressed Sensing Implementation Plan

## 1. Executive Summary

This document specifies the step-by-step implementation plan for the Compressed Sensing Cross-Validation project in the `Gemini/` workspace. The plan is organized into progressive phases that move from foundational numerical modules to reproducible experiment pipelines and visualization suites.

All code will use the project's existing virtual environment:
```bash
/home/aragorn/Documents/PartIIProject/venv/bin/python
```
The legacy script `OMP_CV.py` remains strictly untouched.

---

## 2. Phased Roadmap

```
Phase 1: Core Library Foundation
  └── Signals, Sensing, OMP Trajectory, Projections, Metrics
Phase 2: Single Problem & Baseline Validation
  └── Experiment 1: E_j vs E_hat_j trajectory visualizer
Phase 3: The Three Notions of "Works"
  └── Experiment 2: Norm Relative Error, Kendall's τ, Best-Iterate Selection
Phase 4: Systematic Matrix Comparison
  └── Experiment 3: Dense Gaussian vs Dense Rademacher vs Sparse Rademacher (q-sweep)
Phase 5: Budget-Matched Analysis (Iso-Compute)
  └── Experiment 4: Fixed r vs Fixed nnz (r_dense ≈ q * r)
Phase 6: The JL Distortion Hierarchy
  └── Experiment 5: Isometry vs Pairwise Distance vs CV Ranking vs Model Selection
Phase 7: Comprehensive Sweeps & 2D Phase Transitions
  └── Experiment 6: (q, r) Heatmaps across N, s, SNR, and error sparsity ||e^j||_0
Phase 8: CLI Automation & Export Engine
  └── Reproducible batch runners, CSV/Parquet export, publication figures
```

---

## 3. Phase-by-Phase Specification

### Phase 1: Core Library Foundation (`Gemini/gemini_cs`)

**Objective**: Build a modular, well-tested core Python library implementing the mathematical primitives.

1. **`signals.py`**:
   - `generate_sparse_signal(N: int, s: int, coeff_dist: str = 'gaussian', rng: np.random.Generator) -> (np.ndarray, np.ndarray)`
   - Support for normal coefficients, Rademacher $\pm 1$, and dynamic range decay (e.g., power-law decay $k^{-\alpha}$).
   - Helper function `sparsity_of(vector, tol=1e-8) -> int`.
2. **`sensing.py`**:
   - `generate_sensing_matrix(M: int, N: int, dist: str = 'gaussian', normalize: str = 'column', rng: np.random.Generator) -> np.ndarray`
   - AWGN generator: `add_noise_snr(y, snr_db, rng)` and `add_noise_sigma(y, sigma, rng)`.
3. **`reconstruction/omp.py`**:
   - `run_omp_trajectory(Phi: np.ndarray, y: np.ndarray, max_iter: int, x_true: Optional[np.ndarray]) -> ReconstructionTrajectory`
   - Retains the entire trajectory: iterates $\mathbf{x}^0, \mathbf{x}^1, \dots, \mathbf{x}^p$, residual history $\mathbf{r}^j$, supports $S^j$, true errors $E_j = \|\mathbf{x} - \mathbf{x}^j\|_2^2$, and error vector sparsity $\|\mathbf{e}^j\|_0$.
4. **`projections.py`**:
   - `generate_projection_matrix(r: int, N: int, matrix_type: str, q: float = 1.0, rng: np.random.Generator) -> np.ndarray`
   - Supports:
     - `'gaussian'`: $\mathcal{N}(0, 1/r)$
     - `'rademacher'`: $\pm 1/\sqrt{r}$
     - `'sparse_rademacher'`: entries in $\{0, \pm 1/\sqrt{qr}\}$ with density $q$
5. **`metrics.py`**:
   - `relative_norm_error(E_true: np.ndarray, E_hat: np.ndarray) -> np.ndarray`
   - `kendall_tau_score(E_true: np.ndarray, E_hat: np.ndarray) -> float`
   - `pairwise_concordance(E_true: np.ndarray, E_hat: np.ndarray) -> float`
   - `best_iterate_match(E_true: np.ndarray, E_hat: np.ndarray) -> bool`
   - `oracle_excess_error(E_true: np.ndarray, E_hat: np.ndarray) -> float`

**Verification Criteria**:
- Write unit tests in `Gemini/tests/test_core.py`.
- Run `venv/bin/python -m unittest discover Gemini/tests` to verify shapes, variances, unbiasedness, and OMP exact recovery on noiseless $s$-sparse signals.

---

### Phase 2: Single Trajectory & Summary Baseline

**Objective**: Implement the first visual diagnostic comparing true error $E_j$ against estimated error $\widehat{E}_j$ along an OMP trajectory.

1. **Script `Gemini/scripts/01_single_trajectory.py`**:
   - Synthesizes problem ($N=512$, $s=10$, $m=120$, $r=30$).
   - Computes $E_j$ and $\widehat{E}_j$ for $j \in \{0, \dots, 2s\}$.
   - Plots dual-curve visualization (x-axis: OMP iteration $j$, y-axis: squared error $\ell_2^2$, curves: $E_j$ vs $\widehat{E}_j$).
2. **Batch Summary Script `Gemini/scripts/01b_trajectory_ensemble.py`**:
   - Runs $T=100$ independent random trials.
   - Plots median trajectory with interquartile range (IQR, 25th–75th percentile) ribbon.

**Deliverables**:
- Generated figures: `Gemini/figures/fig1_single_trajectory.png` and `Gemini/figures/fig1b_ensemble_trajectory.png`.

---

### Phase 3: The Three Notions of "Works"

**Objective**: Quantify the fundamental gap between norm estimation, ordering preservation, and best-iterate selection.

1. **Script `Gemini/scripts/02_three_notions_evaluation.py`**:
   - Fixes problem size ($N=512, s=10, m=120, r=40$).
   - Sweeps validation matrix density $q \in [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]$.
   - Runs $T=200$ trials per $q$.
   - Computes:
     - **Notion A**: Median relative norm error $\text{median}_j |\widehat{E}_j - E_j| / E_j$.
     - **Notion B**: Kendall's rank correlation $\tau$ and pairwise concordance fraction.
     - **Notion C**: Best iterate selection rate $P(\hat{j} = j^*)$ and oracle excess error.
2. **Visualization**:
   - Multi-panel plot showing the three metrics side-by-side as a function of $q$.
   - Demonstrates empirically if selection accuracy remains resilient even when norm error deteriorates.

**Deliverables**:
- Data table: `Gemini/results/three_notions_results.csv`.
- Figure: `Gemini/figures/fig2_three_notions_vs_sparsity.png`.

---

### Phase 4: Matrix Distribution Family Comparison

**Objective**: Compare Dense Gaussian vs Dense Rademacher vs Sparse Rademacher under varying sparsity $q$.

1. **Script `Gemini/scripts/03_matrix_families.py`**:
   - Matrix distributions evaluated:
     1. Dense Gaussian ($\Psi_{ij} \sim \mathcal{N}(0, 1/r)$)
     2. Dense Rademacher ($\Psi_{ij} \in \{\pm 1/\sqrt{r}\}$)
     3. Sparse Rademacher ($q \in \{0.5, 0.2, 0.1, 0.05\}$)
   - Evaluates whether sub-Gaussian tail properties (Rademacher having kurtosis parameter $\gamma = 0$ vs Gaussian $\gamma = 2/m^2$) manifest as tighter CV bounds as predicted by Zhang et al. (2014).
2. **Analysis**:
   - Variance of $\widehat{E}_j$ conditional on $E_j$.
   - Concordance and selection accuracy curves across validation measurements $r \in [10, 100]$.

**Deliverables**:
- Figure: `Gemini/figures/fig3_matrix_families_comparison.png`.

---

### Phase 5: Budget-Matched Comparison (Iso-Cost: Fixed $r$ vs Fixed $\operatorname{nnz}$)

**Objective**: Answer the core practical question: *If compute or hardware constraints fix the number of nonzero elements, is a sparse projection matrix actually advantageous over a smaller dense matrix?*

1. **Formulation**:
   - Given a sparse matrix $\Psi_{\text{sparse}} \in \mathbb{R}^{r \times N}$ with density $q$:
     $$\operatorname{nnz}(\Psi_{\text{sparse}}) \approx r \cdot q N$$
   - Budget-matched dense matrix $\Psi_{\text{dense}} \in \mathbb{R}^{r_{\text{dense}} \times N}$ where:
     $$r_{\text{dense}} \approx \lceil q \cdot r \rceil \implies \operatorname{nnz}(\Psi_{\text{dense}}) \approx r_{\text{dense}} N \approx rqN$$
2. **Script `Gemini/scripts/04_budget_matching.py`**:
   - Generates two comparison plots:
     1. **Plot 5A (Fixed $r$)**: x-axis: $r$, curves: different densities $q$. (Tests dimensional reduction capacity).
     2. **Plot 5B (Fixed $\operatorname{nnz}$)**: x-axis: total non-zeros, curves: Sparse Rademacher vs Dense Gaussian / Dense Rademacher. (Tests computational efficiency).

**Deliverables**:
- Figure: `Gemini/figures/fig4_budget_matched_comparison.png`.

---

### Phase 6: General JL Requirement vs Task-Specific Hierarchy

**Objective**: Empirically evaluate the hierarchy of guarantees on the actual error set $P = \{\mathbf{0}, \mathbf{e}^1, \dots, \mathbf{e}^p\}$.

1. **Script `Gemini/scripts/05_jl_hierarchy_check.py`**:
   - Tests:
     - **Level 1 (All Pairs JL)**: Fraction of pairs $(\mathbf{u}, \mathbf{v}) \in P \times P$ satisfying $(1-\varepsilon)\|\mathbf{u}-\mathbf{v}\|_2^2 \le \|\Psi(\mathbf{u}-\mathbf{v})\|_2^2 \le (1+\varepsilon)\|\mathbf{u}-\mathbf{v}\|_2^2$.
     - **Level 2 (Error Norms)**: Fraction of iterates $j$ satisfying $(1-\varepsilon)E_j \le \widehat{E}_j \le (1+\varepsilon)E_j$.
     - **Level 3 (Order Preservation)**: Kendall's $\tau > 0.9$.
     - **Level 4 (Argmin Match)**: $\hat{j} = j^*$.
   - Tracks minimum $r$ (or maximum sparsity $1-q$) required to reach 95% success for each of the 4 levels.
2. **Structure Analysis**:
   - Evaluates whether sparsity tolerance correlates with error vector sparsity $\|\mathbf{e}^j\|_0 \le s + j$.

**Deliverables**:
- Figure: `Gemini/figures/fig5_jl_vs_task_hierarchy.png`.

---

### Phase 7: Parameter Space Sweeps & 2D Phase Transitions

**Objective**: Map out the complete operational regime of sparse CV projections via 2D heatmaps.

1. **Script `Gemini/scripts/06_heatmap_sweep.py`**:
   - Grid parameters:
     - Density $q \in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]$
     - Validation rows $r \in [10, 20, 30, 40, 60, 80, 100, 150, 200]$
     - Problem sizes: $N \in \{256, 512, 1024\}$, $s \in \{5, 10, 20\}$
     - Noise regimes: Noiseless ($\sigma=0$), Moderate SNR (20 dB), Low SNR (10 dB)
   - Evaluated metrics: Best-iterate selection rate $P(\hat{j} = j^*)$, mean Kendall's $\tau$, median relative error.
2. **Parallel execution**:
   - Utilizes `ProcessPoolExecutor` with multi-core batching to compute thousands of trials efficiently.
3. **Visualization**:
   - Heatmap plots showing $(q, r) \mapsto \text{Accuracy}$ with empirical 90% and 95% contour boundaries.

**Deliverables**:
- Data: `Gemini/results/heatmap_sweep_grid.parquet` (or `.csv`).
- Figures: `Gemini/figures/fig6_heatmap_selection_accuracy.png` and `Gemini/figures/fig6b_heatmap_kendall_tau.png`.

---

## 4. Verification & Testing Strategy

1. **Numerical Invariance Tests**:
   - Ensure expected row variance of $\Psi$ is strictly $1/r$ regardless of density $q$:
     $$\mathbb{E}\left[\sum_{j=1}^N \Psi_{ij}^2\right] = \frac{N}{r}$$
2. **OMP Trajectory Sanity**:
   - For $s$-sparse noiseless $\mathbf{x}$ and RIP measurement matrix $\Phi$ ($m \ge 4s$), confirm that at iteration $j = s$, $E_j \approx 0$ and argmin is reached at $j^* = s$.
3. **Reproducibility Guarantee**:
   - Every script accepts an explicit `--seed` parameter and produces identical numerical outputs when rerun with the same seed.

---

## 5. Execution Summary Table

| Phase | Script Name | Primary Output | Research Question Addressed |
| :--- | :--- | :--- | :--- |
| **Phase 1** | `Gemini/tests/test_core.py` | Unit test suite | Mathematical correctness of primitives |
| **Phase 2** | `01_single_trajectory.py` | Trajectory plot | Does CV follow error trajectory? |
| **Phase 3** | `02_three_notions_evaluation.py` | 3-notions curve | Norm accuracy vs Ordering vs Selection |
| **Phase 4** | `03_matrix_families.py` | Distribution comparison | Gaussian vs Rademacher vs Sparse |
| **Phase 5** | `04_budget_matching.py` | Iso-compute comparison | Fixed $r$ vs Fixed $\operatorname{nnz}$ (compute budget) |
| **Phase 6** | `05_jl_hierarchy_check.py` | Hierarchy bar/line chart | General JL bound vs CV task demand |
| **Phase 7** | `06_heatmap_sweep.py` | 2D Phase heatmaps | Phase transition of validation sparsity $(q, r)$ |
