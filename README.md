# ⚛️ Quantum Algorithms Comparative Study

A comprehensive implementation and benchmarking suite for **six foundational quantum algorithms**, comparing circuit complexity, gate requirements, qubit usage, noise resilience, and classical vs quantum speedup — built with **Qiskit** and fully reproducible.

> **Portfolio project demonstrating quantum computing fundamentals from first principles.**

---

## 📌 Algorithms Covered

| Algorithm | Problem | Classical Complexity | Quantum Complexity | Speedup |
|-----------|---------|---------------------|-------------------|---------|
| **Deutsch-Jozsa** | Constant vs balanced function | O(2^(n−1) + 1) | **O(1)** | Exponential |
| **Bernstein-Vazirani** | Find hidden bit-string | O(n) | **O(1)** | Linear |
| **Grover's Search** | Search unsorted database | O(N) | **O(√N)** | Quadratic |
| **QFT** | Discrete Fourier Transform | O(n · 2^n) | **O(n²)** | Exponential |
| **Shor's Factoring** | Integer factorization | O(exp((log N)^⅓ · (log log N)^⅔)) | **O((log N)³)** | Super-polynomial |
| **QAOA** | Combinatorial optimization (Max-Cut) | O(1.1383^n) exact | **O(p · \|E\|)** | Heuristic |

---

## 📊 Benchmark Results

Running `main.py` produces the following circuit metrics:

| Algorithm | Qubits | Depth | Gates | Runtime (s) |
|-----------|--------|-------|-------|-------------|
| Deutsch-Jozsa | 5 | 8 | 22 | 0.037 |
| Bernstein-Vazirani | 5 | 6 | 20 | 0.007 |
| Grover's Search | 4 | 32 | 96 | 0.014 |
| QFT | 4 | 10 | 21 | 0.007 |
| Shor's Factoring | 12 | 26 | 68 | 0.284 |
| QAOA | 4 | 14 | 31 | 0.246 |

### Generated Figures

| Figure | Description |
|--------|-------------|
| `plots/01_circuit_metrics.png` | Circuit depth and gate count comparison |
| `plots/02_resource_usage.png` | Qubit count and runtime |
| `plots/03_speedup_radar.png` | Quantum speedup profile (radar chart) |
| `plots/04_complexity_table.png` | Complexity summary |
| `plots/05_grover_distribution.png` | Grover measurement distribution |

---

## 🔬 Noise Analysis

`noise_analysis.py` simulates four physically motivated noise models and measures fidelity degradation across increasing noise strengths.

**Noise models:**
- **Depolarizing** — random Pauli errors after every gate
- **Thermal relaxation** — T1/T2 decoherence over gate time
- **Readout / SPAM** — bit-flip errors during measurement
- **Combined (realistic)** — all three simultaneously, modelling a real superconducting device

**Key findings from simulation:**

| Algorithm | Depolarizing threshold (p₁) | Noise resilience |
|-----------|----------------------------|-----------------|
| Bernstein-Vazirani | > 0.05 | ✅ Most resilient (shallow circuit, depth 6) |
| Grover's Search | ~0.05 | ✅ Robust — amplitude amplification tolerates moderate errors |
| Deutsch-Jozsa | ~0.005 | ⚠️ Sensitive — relies on precise destructive interference |
| QFT | Flat TVD across all levels | ℹ️ Uniform output; noise doesn't change distribution shape |

Figures are saved to `./figures/` via `visualize_results.py`.

---

## 🗂️ Project Structure

```
quantum-algorithms-comparison/
├── algorithms/
│   ├── __init__.py
│   ├── deutsch_jozsa.py       # Deutsch-Jozsa: O(1) oracle classification
│   ├── bernstein_vazirani.py  # BV: recover n-bit secret in 1 query
│   ├── grover.py              # Grover: O(√N) amplitude amplification
│   ├── qft.py                 # QFT: O(n²) gate Fourier transform
│   ├── shor.py                # Shor: QPE-based period finding (N=15)
│   └── qaoa.py                # QAOA p=2: variational Max-Cut solver
├── benchmarks/
│   └── benchmark.py           # Timing, gate count, depth harness
├── figures/                   # Noise analysis plots (from visualize_results.py)
├── plots/                     # Circuit comparison charts (from main.py)
├── results/
│   ├── comparison_results.json
│   └── noise_results.json
├── main.py                    # Entry point — runs all algorithms + benchmarks
├── noise_analysis.py          # Noise model sweep across 4 error types
├── visualize_results.py       # Publication-quality figures from noise results
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

```
Python >= 3.10
```

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/quantum-algorithms-comparison.git
cd quantum-algorithms-comparison
pip install -r requirements.txt
```

### Run Everything

```bash
# Run all 6 algorithms and generate comparison plots
python main.py

# Run noise analysis (saves results/noise_results.json)
python noise_analysis.py

# Generate publication figures from noise results
python visualize_results.py
```

### Use Individual Algorithms

```python
from algorithms import Grover, QFT, QAOA

# Grover's search on 4-qubit space
g = Grover(n=4, target="1011")
result = g.run(shots=1024)
print(result["found"], result["prob_target"])

# QFT on a specific input state
q = QFT(n=5, input_state="10110")
result = q.run()
print(result["probabilities"])

# QAOA for Max-Cut (p=3 layers)
qaoa = QAOA(edges=[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)], p=3)
result = qaoa.run()
print(result["best_cut"], result["approx_ratio"])
```

---

## 🔬 Algorithm Deep-Dives

### Deutsch-Jozsa
Places all input qubits in superposition via Hadamard gates, applies a single oracle query that encodes phase information via kickback, then applies a final Hadamard layer. Constructive interference produces |0…0⟩ for constant functions; destructive interference produces anything else for balanced functions — a **deterministic answer in one shot** versus exponentially many classical queries.

### Bernstein-Vazirani
Structurally identical to Deutsch-Jozsa but the oracle encodes `f(x) = s·x mod 2`. The Hadamard sandwich + phase kickback reveals **all n bits of secret s simultaneously** in a single measurement, compared to n separate classical queries.

### Grover's Search
The oracle phase-flips the target state. The diffusion operator (inversion about the mean) then **amplifies the target amplitude** by a small amount each iteration. After ⌊π/4 · √N⌋ iterations the target dominates with probability > 0.5, achieving a provably optimal **quadratic speedup** over classical linear search.

### Quantum Fourier Transform (QFT)
Decomposes into n Hadamard gates and O(n²) controlled-phase (CPhase) rotations, computing the DFT in **O(n²) gates vs O(n · 2^n)** for the classical FFT. It is the core subroutine in Shor's algorithm and quantum phase estimation (QPE).

### Shor's Factoring
Reduces integer factorization to period-finding of `f(x) = aˣ mod N`. **Quantum Phase Estimation** (built on QFT) finds the period r exponentially faster than any known classical method. Given r, classical GCD arithmetic yields the factors. This implementation demonstrates the full QPE circuit for N=15, a=7, recovering period r=4 and factors (3, 5).

### QAOA
A **variational quantum-classical hybrid** algorithm. A parameterized circuit alternates between a phase separator (encodes the cost Hamiltonian via ZZ interactions) and a mixer (Rx rotations). A classical optimizer (COBYLA) tunes angles γ and β to maximize ⟨H_C⟩. As depth p → ∞, QAOA converges to the optimal Max-Cut solution.

---

## ⚙️ Implementation Notes

- **Framework**: [Qiskit 2.x](https://www.ibm.com/quantum/qiskit) with Aer statevector/noise simulator
- **Shor's**: Uses hand-coded controlled-U gates for N=15 (pedagogical; generalizing requires O((log N)³) qubit arithmetic circuits)
- **QAOA**: Classical optimizer is `scipy.optimize.COBYLA` with 150 iterations and random restarts
- **Grover's**: Uses optimal iteration count ⌊π/4 · √N⌋ for maximum success probability
- **Noise models**: Loosely inspired by superconducting qubit devices (IBM Falcon r5 class), scaled for fast simulation

---

## 📚 References

1. Deutsch & Jozsa (1992). *Rapid solution of problems by quantum computation.* Proc. R. Soc. Lond. A.
2. Bernstein & Vazirani (1997). *Quantum complexity theory.* SIAM Journal on Computing.
3. Grover (1996). *A fast quantum mechanical algorithm for database search.* STOC '96.
4. Coppersmith (1994). *An approximate Fourier transform useful in quantum factoring.* IBM Research Report.
5. Shor (1994). *Algorithms for quantum computation: discrete logarithms and factoring.* FOCS '94.
6. Farhi, Goldstone & Gutmann (2014). *A quantum approximate optimization algorithm.* arXiv:1411.4028.
7. Nielsen & Chuang (2010). *Quantum Computation and Quantum Information.* Cambridge University Press.

---

## 📄 License

MIT License — free to use, fork, and build upon.



