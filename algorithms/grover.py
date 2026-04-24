"""
Grover's Search Algorithm
=========================
Problem : Search an unsorted database of N = 2^n items for a marked target.

Classical complexity : O(N)      — linear scan
Quantum complexity   : O(√N)     — quadratic speedup via amplitude amplification

Key idea: The oracle flips the phase of the target state.  The diffusion
          operator (inversion about average) then amplifies its amplitude.
          Repeating ~π/4 · √N times makes the target state dominate,
          so measurement finds it with high probability.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


class Grover:
    """Implements Grover's search algorithm for a single marked state."""

    CLASSICAL_COMPLEXITY = "O(N)"
    QUANTUM_COMPLEXITY   = "O(√N)"
    SPEEDUP              = "Quadratic"

    def __init__(self, n: int = 4, target: str = None):
        """
        Parameters
        ----------
        n      : number of qubits  (search space = 2^n)
        target : target bit-string to search for (random if None)
        """
        self.n = n
        self.N = 2 ** n
        if target is None:
            target = format(np.random.randint(0, self.N), f"0{n}b")
        self.target = target
        self.iterations = max(1, int(np.floor(np.pi / 4 * np.sqrt(self.N))))
        self.circuit = self._build_circuit()

    # ------------------------------------------------------------------
    # Oracle: flips phase of |target>
    # ------------------------------------------------------------------
    def _phase_oracle(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n, name="Phase Oracle")
        # Flip qubits where target bit is '0' so target becomes |11…1>
        for i, bit in enumerate(reversed(self.target)):
            if bit == "0":
                qc.x(i)
        # Multi-controlled Z (phase flip on |11…1>)
        qc.h(self.n - 1)
        qc.mcx(list(range(self.n - 1)), self.n - 1)
        qc.h(self.n - 1)
        # Undo bit flips
        for i, bit in enumerate(reversed(self.target)):
            if bit == "0":
                qc.x(i)
        return qc

    # ------------------------------------------------------------------
    # Diffusion (Grover diffusion operator): inversion about average
    # ------------------------------------------------------------------
    def _diffusion(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n, name="Diffusion")
        qc.h(range(self.n))
        qc.x(range(self.n))
        qc.h(self.n - 1)
        qc.mcx(list(range(self.n - 1)), self.n - 1)
        qc.h(self.n - 1)
        qc.x(range(self.n))
        qc.h(range(self.n))
        return qc

    # ------------------------------------------------------------------
    # Full circuit
    # ------------------------------------------------------------------
    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n, self.n, name="Grover")

        # Uniform superposition
        qc.h(range(self.n))
        qc.barrier()

        # Grover iterations
        oracle = self._phase_oracle()
        diffusion = self._diffusion()
        for _ in range(self.iterations):
            qc.compose(oracle, inplace=True)
            qc.compose(diffusion, inplace=True)
            qc.barrier()

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

        found = max(counts, key=counts.get)[::-1]
        prob_target = counts.get(self.target[::-1], 0) / shots

        return {
            "algorithm"    : "Grover's Search",
            "target"       : self.target,
            "found"        : found,
            "correct"      : found == self.target,
            "prob_target"  : round(prob_target, 4),
            "iterations"   : self.iterations,
            "counts"       : counts,
            "num_qubits"   : self.circuit.num_qubits,
            "depth"        : self.circuit.depth(),
            "gate_count"   : sum(self.circuit.count_ops().values()),
            "classical_complexity": self.CLASSICAL_COMPLEXITY,
            "quantum_complexity"  : self.QUANTUM_COMPLEXITY,
            "speedup"             : self.SPEEDUP,
        }
