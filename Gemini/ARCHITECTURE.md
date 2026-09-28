# Gemini Compressed Sensing Architecture (CV-CS)

## 1. Overview & Research Objective

This architecture provides an extensible, modular, and mathematically rigorous framework for investigating cross-validation (CV) in Compressed Sensing (CS), with a specific focus on **sparse validation projection matrices**.

### 1.1 The Core Research Question
> **How does exploiting the structure of reconstruction errors affect the sparsity required of the validation projection to reliably estimate or rank recovery errors?**

In compressive sensing, an unknown sparse vector $\mathbf{x} \in \mathbb{R}^N$ ($\|\mathbf{x}\|_0 = s \ll N$) is measured via:
$$\mathbf{y} = \Phi \mathbf{x} + \mathbf{n}, \quad \Phi \in \mathbb{R}^{m \times N}, \quad \mathbf{n} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$$

Iterative recovery algorithms such as Orthogonal Matching Pursuit (OMP) produce a trajectory of candidate reconstructions $\mathbf{x}^1, \mathbf{x}^2, \dots, \mathbf{x}^p$. Because the true sparsity $s$ and noise level $\sigma$ are typically unknown in practice, model selection requires estimating the true reconstruction error:
$$E_j \triangleq \|\mathbf{x} - \mathbf{x}^j\|_2^2 \quad \text{for } j \in \{1, \dots, p\}$$

Using an independent validation projection matrix $\Psi \in \mathbb{R}^{r \times N}$ and validation measurements $\mathbf{y}_{\Psi} = \Psi \mathbf{x} + \mathbf{n}_{\Psi}$, we compute the computable cross-validation proxy:
$$\widehat{E}_j \triangleq \|\mathbf{y}_{\Psi} - \Psi \mathbf{x}^j\|_2^2$$

### 1.2 The Validation Hierarchy
General-purpose dimensionality reduction (Johnson-Lindenstrauss lemma) demands uniform pairwise distance preservation over arbitrary point sets. However, the cross-validation task exhibits a natural hierarchy of decreasing stringency:

```
[Level 1: General JL Isometry]
(1 - ε) ||e||_2^2 <= ||Ψ e||_2^2 <= (1 + ε) ||e||_2^2 for all e in R^N
                          │
                          ▼  (Relaxed: Only error vectors e^j = x - x^j)
[Level 2: Error Vector Norm Preservation]
|E_hat_j - E_j| / E_j <= ε
                          │
                          ▼  (Relaxed: Magnitude not required, only monotonicity)
[Level 3: Global & Pairwise Ordering Preservation]
Kendall's τ between (E_1, ..., E_p) and (E_hat_1, ..., E_hat_p) ≈ 1
                          │
                          ▼  (Relaxed: Ordering over all iterates not required)
[Level 4: Best-Iterate Selection]
arg min_j E_hat_j == arg min_j E_j (or minimal oracle excess error)
```

Furthermore, OMP error vectors have strict structural constraints:
$$\mathbf{e}^j = \mathbf{x} - \mathbf{x}^j \implies \|\mathbf{e}^j\|_0 \le s + j$$

This project investigates whether the required density $q$ of nonzeros in $\Psi$ can be drastically lower when moving from Level 1 down to Level 4, and how structural features of $\mathbf{e}^j$ (sparsity $s+j$, coordinate concentration, decay) govern this transition.

---

## 2. System Architecture & Component Breakdown

The architecture decouples signal generation, sensing, reconstruction, validation projections, metric computation, and visualization into isolated, testable modules.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Experiment Runner                              │
│         (Parameter Grids, Multi-processing, Trial Aggregation)           │
└──────┬──────────────────────┬──────────────────────┬─────────────────────┘
       │                      │                      │
       ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────────────────────┐
│ Signal Engine│      │ Sensing &    │      │ Recovery Trajectory Engine   │
│ - Sparse x   │      │ Noise Model  │      │ - OMP (full trajectory)      │
│ - Support/Mag│      │ - Phi matrix │      │ - Residuals & Support sets   │
└──────┬───────┘      └──────┬───────┘      └──────────────┬───────────────┘
       │                     │                             │
       └─────────────────────┼─────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ Validation Projection Factory│
              │ - Dense Gaussian             │
              │ - Dense Rademacher           │
              │ - Sparse Rademacher (q)      │
              │ - Budget Matching (nnz vs r) │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ Metric & Evaluation Engine   │
              │ - Relative Norm Error        │
              │ - Kendall's Tau / Spearman   │
              │ - Best-Iterate Acc P(j*=j^)  │
              │ - Empirical JL Violations    │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ Analysis & Plotting Engine   │
              │ - Trajectory Plots           │
              │ - Heatmaps (q vs r)          │
              │ - Budget Iso-Cost Curves     │
              └──────────────────────────────┘
```

---

## 3. Detailed Component Specifications

### 3.1 Signals & Sparsity (`gemini_cs.signals`)
Responsible for generating ground-truth sparse vectors $\mathbf{x} \in \mathbb{R}^N$ with controllable structure.

- **Classes / Functions**:
  - `SparseSignalConfig`: Dataclass specifying $N$, $s$, support distribution (uniform random, clustered, harmonic), and nonzero coefficient distribution (standard normal, Rademacher $\pm 1$, uniform $[-b, -a] \cup [a, b]$, geometrically decaying).
  - `generate_sparse_signal(config, rng) -> (x, support)`: Pure function using explicit `np.random.Generator`.
  - `compute_signal_properties(x) -> dict`: Computes dynamic range, $\ell_1/\ell_2$ ratio, $\ell_\infty/\ell_2$ coordinate concentration.

### 3.2 Sensing & Noise Models (`gemini_cs.sensing`)
Generates measurement matrix $\Phi \in \mathbb{R}^{m \times N}$ and noisy observations.

- **Normalization Modes**:
  - Column normalization: $\|\varphi_i\|_2 = 1$ (standard for RIP and coherence analysis).
  - Entry-scaled: $\Phi_{ij} \sim \mathcal{N}(0, 1/m)$ (standard in asymptotic CS and JL literature).
- **Noise Generators**:
  - Additive White Gaussian Noise (AWGN) specified by SNR (dB) or exact standard deviation $\sigma_n$.
  - Generates $\mathbf{y} = \Phi \mathbf{x} + \mathbf{n}$.

### 3.3 Reconstruction Trajectory Engine (`gemini_cs.reconstruction`)
Rather than returning only the final estimate, the engine returns the **entire history of iterates** to study the error evolution.

- **Data Model**:
  ```python
  @dataclass
  class ReconstructionTrajectory:
      iterates: np.ndarray      # Shape (p + 1, N), iterate x^j for j=0..p
      supports: list[set[int]]  # Support set history S^j
      residuals: np.ndarray     # Residual vectors r^j = y - Phi x^j
      true_errors: np.ndarray   # True squared L2 errors ||x - x^j||_2^2
      error_sparsities: np.ndarray # ||x - x^j||_0 for each j
  ```
- **Algorithms**:
  - `OMPTrajectory`: Efficient least-squares solver (using QR / Cholesky rank-1 updates or `np.linalg.lstsq` on active supports).
  - Trajectory collection: Stores $\mathbf{x}^0 = \mathbf{0}$ through $\mathbf{x}^p$ ($p$ typically $2s$ to $4s$).
  - Easily extensible to CoSaMP, Subspace Pursuit, or LASSO path.

### 3.4 Validation Projection Factory (`gemini_cs.projections`)
Generates $\Psi \in \mathbb{R}^{r \times N}$ under varying sparsity and distribution families:

1. **Dense Gaussian**:
   $$\Psi_{ij} \sim \mathcal{N}\left(0, \frac{1}{r}\right)$$
2. **Dense Rademacher**:
   $$\Psi_{ij} = \pm \frac{1}{\sqrt{r}} \quad \text{with probability } \frac{1}{2}$$
3. **Sparse Rademacher (Kane-Nelson / Matoušek style)**:
   For density parameter $q \in (0, 1]$ (expected fraction of nonzeros):
   $$\Psi_{ij} = \begin{cases} 
   0 & \text{with probability } 1 - q \\ 
   +\frac{1}{\sqrt{q r}} & \text{with probability } \frac{q}{2} \\ 
   -\frac{1}{\sqrt{q r}} & \text{with probability } \frac{q}{2} 
   \end{cases}$$
   *Property*: $\mathbb{E}[\Psi_{ij}] = 0$, $\operatorname{Var}(\Psi_{ij}) = \frac{1}{r}$.
4. **Column-wise / Hash Sparse Projections (CountSketch)**:
   Each column has exactly $d$ non-zero entries (e.g., $d=1$ or $d=O(\log(1/\varepsilon))$).
5. **Budget Matching Module (`BudgetMatcher`)**:
   Enables iso-computational / iso-hardware comparisons:
   - Total non-zeros budget $B \approx r \cdot q N$.
   - Dense equivalent: $r_{\text{dense}} = \lceil q \cdot r \rceil$.
   - Allows evaluating: *For a fixed compute budget of matrix-vector multiply $\Psi \mathbf{x}^j$, is a wide-and-sparse $\Psi \in \mathbb{R}^{r \times N}$ better or worse than a narrow-and-dense $\Psi_{\text{dense}} \in \mathbb{R}^{r_{\text{dense}} \times N}$?*

### 3.5 Evaluation & Metric Suite (`gemini_cs.metrics`)
Evaluates the validation output $\widehat{E}_j = \|\mathbf{y}_\Psi - \Psi \mathbf{x}^j\|_2^2$ against ground-truth $E_j = \|\mathbf{x} - \mathbf{x}^j\|_2^2$.

| Metric Category | Mathematical Definition | Interpretation / Research Relevance |
| :--- | :--- | :--- |
| **A. Norm Relative Error** | $\frac{|\widehat{E}_j - E_j|}{E_j}$ or $\frac{\widehat{E}_j}{E_j}$ | Evaluates isometric embedding accuracy across iterations. |
| **B. Pairwise Concordance** | $\frac{1}{\binom{p}{2}} \sum_{i < k} \mathbf{1}[(E_i - E_k)(\widehat{E}_i - \widehat{E}_k) > 0]$ | Probability of correctly ranking two arbitrary reconstruction states. |
| **C. Global Rank Correlation** | $\tau = \text{Kendall's }\tau(E, \widehat{E})$, $\rho = \text{Spearman's }\rho(E, \widehat{E})$ | Monotonic alignment between estimated and true error trajectories. |
| **D. Best Iterate Selection** | $\mathbf{1}[\hat{j} = j^*]$, where $j^* = \arg\min_j E_j, \hat{j} = \arg\min_j \widehat{E}_j$ | Success rate of CV stopping / iterate selection rule. |
| **E. Oracle Excess Error** | $\frac{E_{\hat{j}} - E_{j^*}}{E_{j^*}}$ | Suboptimality penalty incurred when CV selects a suboptimal iterate. |
| **F. JL $\varepsilon$-Violation Rate** | $\mathbb{P}\left(\left|\frac{\|\Psi \mathbf{e}\|_2}{\|\mathbf{e}\|_2} - 1\right| > \varepsilon\right)$ | Direct test of the classical Johnson-Lindenstrauss distortion bound. |

### 3.6 Experiment Execution Engine (`gemini_cs.experiments`)
- **Trial-based execution**: Runs $T$ independent Monte-Carlo trials per parameter configuration.
- **Reproducibility**: Master seed generates independent streams for $(\mathbf{x}, \Phi, \mathbf{n}, \Psi, \mathbf{n}_\Psi)$ via `numpy.random.SeedSequence`.
- **Parallelization**: `ProcessPoolExecutor` with batching over trials to maximize multi-core CPU utilization.
- **Serialization**: Results stored in structured Parquet / CSV / JSON formats with complete metadata (parameters, git commit hash, timestamps).

---

## 4. Proposed Directory Layout

```
PartIIProject/
├── OMP_CV.py                          # UNTOUCHED legacy script
├── venv/                              # Existing virtual environment
└── Gemini/                            # All new modular code & prototypes
    ├── ARCHITECTURE.md                # This document
    ├── IMPLEMENTATION_PLAN.md         # Phased implementation roadmap
    ├── gemini_cs/                     # Core reusable library
    │   ├── __init__.py
    │   ├── signals.py                 # Ground truth sparse signal generation
    │   ├── sensing.py                 # Measurement matrices & noise
    │   ├── reconstruction/            # Iterative recovery algorithms
    │   │   ├── __init__.py
    │   │   ├── base.py                # Abstract Trajectory base class
    │   │   └── omp.py                 # OMP with full iterate preservation
    │   ├── projections.py             # Gaussian, Rademacher, Sparse Rademacher
    │   ├── metrics.py                 # Norm error, Kendall's tau, selection acc
    │   ├── budget.py                  # Fixed-r vs Fixed-nnz budget matching
    │   └── runner.py                  # Parallel Monte-Carlo experiment engine
    ├── scripts/                       # Executable experiment scripts
    │   ├── 01_single_trajectory.py    # Plot single trajectory: E_j vs E_hat_j
    │   ├── 02_norm_vs_ordering.py     # Compare the 3 notions of success
    │   ├── 03_sparsity_sweep.py       # Sweep q in [1.0 ... 0.01]
    │   ├── 04_budget_comparison.py    # Fixed r vs Fixed nnz comparison
    │   ├── 05_heatmap_q_r.py          # 2D Heatmap (q, r) -> selection accuracy
    │   └── 06_jl_bounds_check.py      # Empirical JL vs task performance
    ├── results/                       # Output metrics & CSVs
    └── figures/                       # Publication-quality plots (.png, .pdf)
```

---

## 5. Design Decisions & Rationale

1. **Preserving Full Trajectories**:
   Standard OMP-CV implementations (like `OMP_CV.py`) update the best iterate online and discard the sequence. By storing the full sequence $(\mathbf{x}^0, \dots, \mathbf{x}^p)$, we can evaluate pairwise ordering, Kendall's $\tau$, and error vectors $\mathbf{e}^j$ without re-running OMP for different validation matrices $\Psi$.
2. **Decoupling $\Phi$ and $\Psi$**:
   In hardware (e.g. RMPI), the validation measurements might come from dedicated physical channels or distinct modulation sequences. Decoupling the generation of $\Phi$ and $\Psi$ allows testing mismatched distributions (e.g. dense Gaussian $\Phi$ for sensing, but ultra-sparse Rademacher $\Psi$ for cheap CV computation).
3. **Explicit Random Generator Streams**:
   Never use global `np.random.seed()`. Use independent `np.random.default_rng(subseed)` for signal, sensing matrix, noise, and validation matrix so that varying validation parameters does not alter the underlying recovery problem.
4. **Pure Python + NumPy/SciPy/Pandas/Matplotlib**:
   All dependencies are already present in `venv` (Python 3.14.4, NumPy 2.5.3, SciPy 1.18.1, Pandas 3.0.6, Matplotlib 3.11.2). No heavy external C++ or non-standard packages required.
