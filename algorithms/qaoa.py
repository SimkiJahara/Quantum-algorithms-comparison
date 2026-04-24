"""
Quantum Approximate Optimization Algorithm (QAOA)
==================================================
Problem : Find an approximate solution to combinatorial optimization problems.
          Here we solve Max-Cut on a weighted graph: partition vertices into
          two sets S and S̄ to maximize the number of edges between them.

Classical best (Max-Cut) : O(1.1383^n)  — exact;  O(n²) approximation ratio 0.878
Quantum QAOA (p layers)  : O(p · |E|)   — variational, depth scales with layers p

Key idea: Encode the cost function as a quantum Hamiltonian H_C.  Alternating
          layers of H_C (phase separator) and H_B (mixer/driver) create a
          parameterized state |γ,β>.  Classical optimizer tunes (γ,β) to
          maximize <H_C>, approaching the optimal cut value as p → ∞.
"""

import numpy as np
from itertools import combinations
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize


class QAOA:
    """QAOA for Max-Cut on a small graph."""

    CLASSICAL_COMPLEXITY = "O(1.1383^n) exact / O(n²) approx"
    QUANTUM_COMPLEXITY   = "O(p · |E|)"
    SPEEDUP              = "Heuristic (problem-dependent)"

    def __init__(self, edges: list = None, p: int = 2):
        """
        Parameters
        ----------
        edges : list of (u, v, weight) tuples.  Defaults to a 4-node graph.
        p     : number of QAOA layers (depth parameter)
        """
        if edges is None:
            # 4-node ring graph with unit weights
            edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0), (0, 2, 1.0)]
        self.edges = edges
        self.n = max(max(u, v) for u, v, *_ in edges) + 1
        self.p = p
        self._best_params = None
        self._best_cost = None

    # ------------------------------------------------------------------
    # QAOA circuit for given parameters
    # ------------------------------------------------------------------
    def _build_circuit(self, gamma: list, beta: list) -> QuantumCircuit:
        qc = QuantumCircuit(self.n, self.n, name=f"QAOA p={self.p}")

        # Initial state: equal superposition
        qc.h(range(self.n))

        for layer in range(self.p):
            qc.barrier()
            # Phase separator: e^(-i·gamma·H_C)
            for u, v, w in self.edges:
                qc.rzz(2 * gamma[layer] * w, u, v)

            qc.barrier()
            # Mixer: e^(-i·beta·H_B)
            for i in range(self.n):
                qc.rx(2 * beta[layer], i)

        qc.barrier()
        qc.measure(range(self.n), range(self.n))
        return qc

    # ------------------------------------------------------------------
    # Classical expectation value (used during optimization)
    # ------------------------------------------------------------------
    def _cut_value(self, bitstring: str) -> float:
        """Compute cut value for a given assignment."""
        bits = [int(b) for b in bitstring]
        return sum(w for u, v, w in self.edges if bits[u] != bits[v])

    def _expected_cost(self, params: np.ndarray, shots: int = 512) -> float:
        gamma = params[:self.p]
        beta  = params[self.p:]
        qc = self._build_circuit(gamma, beta)
        sim = AerSimulator()
        tqc = transpile(qc, sim)
        result = sim.run(tqc, shots=shots).result()
        counts = result.get_counts()
        exp_val = sum(
            (count / shots) * self._cut_value(bs[::-1])
            for bs, count in counts.items()
        )
        return -exp_val   # minimize negative → maximize cost

    # ------------------------------------------------------------------
    # Optimize parameters with COBYLA
    # ------------------------------------------------------------------
    def optimize(self, shots: int = 512) -> tuple:
        x0 = np.random.uniform(0, np.pi, 2 * self.p)
        result = minimize(
            self._expected_cost,
            x0,
            args=(shots,),
            method="COBYLA",
            options={"maxiter": 150, "rhobeg": 0.5},
        )
        self._best_params = result.x
        self._best_cost = -result.fun
        return self._best_params, self._best_cost

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, shots: int = 2048) -> dict:
        params, cost = self.optimize(shots=512)
        gamma = params[:self.p]
        beta  = params[self.p:]

        # Final high-shot run with optimized params
        qc = self._build_circuit(gamma, beta)
        sim = AerSimulator()
        tqc = transpile(qc, sim)
        result = sim.run(tqc, shots=shots).result()
        counts = result.get_counts()

        # Best solution from measurement outcomes
        best_bs = max(counts, key=lambda bs: self._cut_value(bs[::-1]))
        best_cut = self._cut_value(best_bs[::-1])

        # Theoretical maximum cut for this graph
        max_cut = sum(w for _, _, w in self.edges)   # upper bound
        approx_ratio = best_cut / max_cut if max_cut > 0 else 0

        return {
            "algorithm"     : "QAOA",
            "p_layers"      : self.p,
            "edges"         : self.edges,
            "best_bitstring": best_bs[::-1],
            "best_cut"      : best_cut,
            "approx_ratio"  : round(approx_ratio, 4),
            "exp_cost"      : round(cost, 4),
            "gamma"         : list(np.round(gamma, 4)),
            "beta"          : list(np.round(beta, 4)),
            "counts"        : counts,
            "num_qubits"    : qc.num_qubits,
            "depth"         : qc.depth(),
            "gate_count"    : sum(qc.count_ops().values()),
            "classical_complexity": self.CLASSICAL_COMPLEXITY,
            "quantum_complexity"  : self.QUANTUM_COMPLEXITY,
            "speedup"             : self.SPEEDUP,
        }
