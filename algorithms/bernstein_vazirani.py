"""
Bernstein-Vazirani Algorithm
============================
Problem : Given a black-box oracle for f(x) = s·x mod 2  (dot product mod 2),
          find the hidden secret string s ∈ {0,1}^n.

Classical complexity : O(n)  — need n separate queries, one bit at a time
Quantum complexity   : O(1)  — one oracle call recovers all n bits simultaneously

Key idea: Essentially the same circuit as Deutsch-Jozsa but the oracle now
          encodes a specific inner product.  The Hadamard sandwich + phase
          kickback reveals every bit of s in parallel from a single shot.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


class BernsteinVazirani:
    """Implements the Bernstein-Vazirani algorithm."""

    CLASSICAL_COMPLEXITY = "O(n)"
    QUANTUM_COMPLEXITY   = "O(1)"
    SPEEDUP              = "Linear"

    def __init__(self, secret: str = "1011"):
        """
        Parameters
        ----------
        secret : bit-string secret s (e.g. '1011').  Length determines n.
        """
        self.secret = secret
        self.n = len(secret)
        self.circuit = self._build_circuit()

    # ------------------------------------------------------------------
    # Oracle: computes f(x) = s · x  mod 2
    # ------------------------------------------------------------------
    def _build_oracle(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n + 1, name="BV Oracle")
        for i, bit in enumerate(reversed(self.secret)):   # LSB first
            if bit == "1":
                qc.cx(i, self.n)
        return qc

    # ------------------------------------------------------------------
    # Full circuit
    # ------------------------------------------------------------------
    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n + 1, self.n, name="Bernstein-Vazirani")

        # Ancilla into |->
        qc.x(self.n)
        qc.barrier()

        # Hadamard on all qubits
        qc.h(range(self.n + 1))
        qc.barrier()

        # Oracle
        qc.compose(self._build_oracle(), inplace=True)
        qc.barrier()

        # Hadamard on input register
        qc.h(range(self.n))
        qc.barrier()

        # Measure: result should be s
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

        # Most frequent bitstring is the recovered secret.
        # No reversal needed: the oracle already uses enumerate(reversed(secret)),
        # so Qiskit's output string (MSB = qubit n-1) maps directly to the secret.
        recovered_str = max(counts, key=counts.get)

        return {
            "algorithm"   : "Bernstein-Vazirani",
            "secret"      : self.secret,
            "recovered"   : recovered_str,
            "correct"     : recovered_str == self.secret,
            "counts"      : counts,
            "num_qubits"  : self.circuit.num_qubits,
            "depth"       : self.circuit.depth(),
            "gate_count"  : sum(self.circuit.count_ops().values()),
            "classical_complexity": self.CLASSICAL_COMPLEXITY,
            "quantum_complexity"  : self.QUANTUM_COMPLEXITY,
            "speedup"             : self.SPEEDUP,
        }