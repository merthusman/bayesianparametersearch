# Project Prometheus V

*A GPU-Accelerated Physics Simulation for Discovering Emergent Particle Spectra using Clifford Algebra and Multi-Scale Extrapolation*

---

## 🚀 Overview

**Project Prometheus V** is a computational physics model designed to discover a fundamental set of universal laws—referred to as a "constitution"—from which the particle mass spectrum of the Standard Model can emerge. The simulation is built upon the mathematical framework of **Geometric Algebra**, specifically the Clifford Algebra $Cl(1,8)$.

The core of the project is the `WarpDriveExplorer`, a sophisticated optimization framework that employs the **"DaVinci" methodology**. This method involves running simulations across multiple grid scales (e.g., 64x64, 96x96, 128x128), tracking the behavior of emergent quasi-particles, and extrapolating their properties to the continuum limit ($1/L^2 \to 0$). This robust approach filters out grid-dependent artifacts and identifies truly fundamental particles.

The system uses Bayesian optimization (`scikit-optimize`) to intelligently search the vast parameter space of physical constants (`lambda` values), seeking a "champion constitution" that best reproduces the known particle masses provided by the Particle Data Group (PDG).

---

## ⚡ Key Technical Optimizations

The simulation engine (`PrometheusV_Engine`) is heavily optimized for performance on NVIDIA GPUs using CUDA and CuPy. The architecture is based on a formal optimization report and incorporates several key strategies:

1.  **Fused Computational Kernel**: All core model dynamics—the Laplacian, potential fields, and damping forces—are consolidated into a single, massive CUDA kernel (`calculate_force_kernel`). This minimizes memory bandwidth usage by avoiding the need to write intermediate results to and from slow global GPU memory, greatly enhancing data locality.

2.  **Tiling with Shared Memory**: **This is the most critical performance enhancement.** The simulation loads multivector data for a spacetime point and its neighbors into a high-speed shared memory "tile" at the start of the kernel. All subsequent calculations, especially the stencil-based Laplacian, are performed on this fast tile, reducing global memory access by a factor of approximately five and effectively hiding memory latency.

3.  **Occupancy Management**: The CUDA kernel is compiled with `__launch_bounds__(512, 2)`, a directive that guides the compiler to limit register usage. This ensures that at least two thread blocks can reside concurrently on each Streaming Multiprocessor (SM), which is crucial for masking the high latency of HBM2 memory on architectures like the NVIDIA P100.

4.  **Robust Resource Management**: The relaxation and evolution functions include "intelligent early stopping" mechanisms to detect and terminate unstable simulations that diverge or collapse. Furthermore, `try...finally` blocks are used to guarantee that all GPU memory is deallocated, preventing memory leaks even if errors occur.

---

## 🔬 Methodology

The project's objective function aims to maximize a score based on three criteria:

1.  **Particle Coverage**: How many particles from the PDG list are successfully identified in the extrapolated spectrum.
2.  **Particle Diversity**: How many distinct *types* of particles (Leptons, Mesons, Baryons, etc.) are found.
3.  **Axiomatic Consistency**: How well the model's fundamental parameters (`lambda_d`, `lambda_p_g0`) predict the fine-structure constant, $\alpha$. The predicted value is calculated as:
    $$
    \alpha_{\text{predicted}} = \frac{\lambda_{p, g0} / \lambda_d}{4\pi}
    $$
    The score increases as this value approaches the real fine-structure constant, $\alpha \approx 1/137.036$.



---

## 📋 Requirements

* **Python 3.8+**
* An **NVIDIA GPU** with CUDA Toolkit 11.x or 12.x installed.
* **Python Libraries**:
    * `cupy-cuda11x` or `cupy-cuda12x`
    * `scikit-optimize`
    * `numpy`
    * `scipy`
    * `tqdm`

You can install all dependencies with pip:
```bash
pip install cupy-cuda12x scikit-optimize numpy scipy tqdm
```
*(Note: Replace `cupy-cuda12x` with the version matching your installed CUDA Toolkit, e.g., `cupy-cuda11x`)*

---

## ▶️ How to Run

The main script is designed to run out-of-the-box. It will load a "champion constitution" (a promising set of parameters) and perform a fine-tuning optimization run to see if it can be improved.

1.  Clone the repository.
2.  Navigate to the project directory.
3.  Run the script:
    ```bash
    python prometheus_v.py
    ```
    *(Replace `prometheus_v.py` with the actual filename)*

The script will launch a Bayesian optimization process, displaying its progress with a `tqdm` progress bar. Each trial involves running the multi-scale simulation and evaluating the resulting particle spectrum.

---

## 📊 Output

The script will produce the following:

1.  **Console Logs**: Detailed output for each optimization trial, including the score, the number of particles found, and diagnostic information. New "best" constitutions are highlighted when found.
2.  **`prometheus_run_v6_skopt_object.pkl`**: A pickle file containing the `scikit-optimize` result object. This can be used for further analysis of the optimization landscape.
3.  **`prometheus_run_v6_detailed_results.pkl`**: A list of dictionaries, where each dictionary contains the detailed parameters, score, and score breakdown for every trial run.
4.  **Final Report**: At the end of the run, a summary of the best-found constitution is printed to the console.

---

## 🏛️ Code Structure

* `prometheus_v.py`: The main executable script that configures and runs the optimization.
* **`CliffordAlgebra`**: A class that sets up the mathematical foundation for $Cl(1,8)$, including blade definitions and grades.
* **`PrometheusV_Engine`**: The core simulation engine. It contains the raw CUDA C++ code, manages GPU memory, and executes the highly optimized kernels for state relaxation and time evolution.
* **`EmergentPhenomenaAnalyzer`**: A class containing static methods to analyze the simulation's time-series output. It performs Fourier transforms and peak-finding to identify emergent particle frequencies.
* **`WarpDriveExplorer`**: The high-level optimization manager. It implements the "DaVinci" multi-scale methodology, defines the objective function for `scikit-optimize`, and orchestrates the entire search for the ideal constitution.

---

## 🎯 Target Particle Spectrum (PDG Data)

The model is benchmarked against the following established particles and their masses (in MeV/c²).

| Particle Name | Mass (MeV/c²) | Type                  |
| :------------ | :------------ | :-------------------- |
| **Leptons** |               |                       |
| Electron      | 0.511         | Lepton (Fermion)      |
| Muon          | 105.7         | Lepton (Fermion)      |
| Tau           | 1777          | Lepton (Fermion)      |
| **Mesons** |               |                       |
| Pion (π⁰)     | 135.0         | Meson                 |
| Pion (π⁺)     | 139.6         | Meson                 |
| Kaon (K⁺)     | 493.7         | Meson                 |
| Kaon (K⁰)     | 497.6         | Meson                 |
| Eta (η)       | 547.9         | Meson                 |
| Rho (ρ)       | 775.3         | Meson                 |
| Omega (ω)     | 782.7         | Meson                 |
| J/psi (J/ψ)   | 3096.9        | Meson (Quarkonium)    |
| **Baryons** |               |                       |
| Proton        | 938.3         | Baryon (Fermion)      |
| Neutron       | 939.6         | Baryon (Fermion)      |
| Lambda (Λ⁰)   | 1115.7        | Baryon (Fermion)      |
| Sigma (Σ⁺)    | 1189.4        | Baryon (Fermion)      |
| Delta (Δ⁺⁺)   | 1232          | Baryon (Fermion)      |
| **Bosons** |               |                       |
| W Boson       | 80379         | Vector Boson          |
| Z Boson       | 91187         | Vector Boson          |
| Higgs Boson   | 125100        | Scalar Boson          |
