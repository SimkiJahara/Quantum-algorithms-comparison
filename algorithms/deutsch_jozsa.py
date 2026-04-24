"""
Deutsch-Jozsa Algorithm
=======================
Problem : Determine if f:{0,1}^n -> {0,1} is CONSTANT (always 0 or always 1)
          or BALANCED (returns 0 for exactly half inputs, 1 for the other half).

Classical complexity : O(2^(n-1) + 1)  — worst case needs half + 1 queries
Quantum complexity   : O(1)             — single oracle query always suffices

Key idea: Hadamard transforms put the input register into superposition,
          the oracle encodes phase information, and a final Hadamard layer
          causes constructive/destructive interference that reveals the answer
          deterministically from a single measurement.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


class DeutschJozsa:
    """Implements the Deutsch-Jozsa algorithm for n-qubit functions."""

    CLASSICAL_COMPLEXITY = "O(2^(n-1) + 1)"
    QUANTUM_COMPLEXITY   = "O(1)"
    SPEEDUP              = "Exponential"

    def __init__(self, n: int = 4, oracle_type: str = "balanced"):
        """
        Parameters
        ----------
        n           : number of input qubits
        oracle_type : 'balanced' | 'constant'
        """
        self.n = n
        self.oracle_type = oracle_type
        self.circuit = self._build_circuit()

    # ------------------------------------------------------------------
    # Oracle builders
    # ------------------------------------------------------------------
    def _constant_oracle(self) -> QuantumCircuit:
        """Constant oracle: f(x) = 0 for all x (ancilla untouched)."""
        qc = QuantumCircuit(self.n + 1, name="Constant Oracle")
        # f(x)=0 → do nothing;  f(x)=1 → flip ancilla
        # Here we use f(x)=0 variant
        return qc

    def _balanced_oracle(self) -> QuantumCircuit:
        """Balanced oracle: f(x) = x_0 XOR x_1 XOR … XOR x_{n-1}."""
        qc = QuantumCircuit(self.n + 1, name="Balanced Oracle")
        for i in range(self.n):
            qc.cx(i, self.n)          # CNOT from each input qubit to ancilla
        return qc

    # ------------------------------------------------------------------
    # Full circuit
    # ------------------------------------------------------------------
    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n + 1, self.n, name="Deutsch-Jozsa")

        # Step 1: put ancilla in |-> = H|1>
        qc.x(self.n)
        qc.barrier()

        # Step 2: Hadamard on all qubits
        qc.h(range(self.n + 1))
        qc.barrier()

        # Step 3: Oracle
        if self.oracle_type == "constant":
            oracle = self._constant_oracle()
        else:
            oracle = self._balanced_oracle()
        qc.compose(oracle, inplace=True)
        qc.barrier()

        # Step 4: Hadamard on input register
        qc.h(range(self.n))
        qc.barrier()

        # Step 5: Measure input register
        qc.measure(range(self.n), range(self.n))
        return qc

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, shots: int = 1024) -> dict:
        sim = AerSimulator()
        tqc = transpile(self.circuit, sim)
        result = sim.run(tqc, shots=shots).result()
        counts = result.get_counts()

        # All-zeros → constant, anything else → balanced
        all_zeros = "0" * self.n
        total = sum(counts.values())
        verdict = "constant" if counts.get(all_zeros, 0) / total > 0.99 else "balanced"

        return {
            "algorithm"  : "Deutsch-Jozsa",
            "oracle_type": self.oracle_type,
            "verdict"    : verdict,
            "correct"    : verdict == self.oracle_type,
            "counts"     : counts,
            "num_qubits" : self.circuit.num_qubits,
            "depth"      : self.circuit.depth(),
            "gate_count" : sum(self.circuit.count_ops().values()),
            "classical_complexity": self.CLASSICAL_COMPLEXITY,
            "quantum_complexity"  : self.QUANTUM_COMPLEXITY,
            "speedup"             : self.SPEEDUP,
        }
