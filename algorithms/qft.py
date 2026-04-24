"""
Quantum Fourier Transform (QFT)
================================
Problem : Compute the Discrete Fourier Transform of the amplitudes of a
          quantum state |x> → (1/√N) Σ_k  e^(2πi·xk/N) |k>.

Classical FFT complexity : O(n · 2^n)
Quantum QFT complexity   : O(n²)         — exponentially fewer operations

Key idea: The QFT can be decomposed into n Hadamard gates and O(n²) controlled
          rotation gates (CPhase), making it far more efficient than any
          classical FFT.  It is the core subroutine in Shor's and many other
          quantum algorithms.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


class QFT:
    """Implements the Quantum Fourier Transform."""

    CLASSICAL_COMPLEXITY = "O(n · 2^n)"
    QUANTUM_COMPLEXITY   = "O(n²)"
    SPEEDUP              = "Exponential"

    def __init__(self, n: int = 4, input_state: str = None, inverse: bool = False):
        """
        Parameters
        ----------
        n           : number of qubits
        input_state : computational basis state to transform (random if None)
        inverse     : if True, compute the inverse QFT
        """
        self.n = n
        self.N = 2 ** n
        if input_state is None:
            val = np.random.randint(0, self.N)
            input_state = format(val, f"0{n}b")
        self.input_state = input_state
        self.inverse = inverse
        self.circuit = self._build_circuit()

    # ------------------------------------------------------------------
    # QFT core sub-circuit (applied to all n qubits)
    # ------------------------------------------------------------------
    def _qft_core(self, qc: QuantumCircuit, n: int):
        """Add QFT gates to circuit in-place."""
        for j in range(n - 1, -1, -1):
            qc.h(j)
            for k in range(j - 1, -1, -1):
                angle = np.pi / (2 ** (j - k))
                qc.cp(angle, k, j)
        # Swap qubits to match conventional bit-ordering
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)

    def _iqft_core(self, qc: QuantumCircuit, n: int):
        """Add inverse QFT gates to circuit in-place."""
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)
        for j in range(n):
            for k in range(j - 1, -1, -1):
                angle = -np.pi / (2 ** (j - k))
                qc.cp(angle, k, j)
            qc.h(j)

    # ------------------------------------------------------------------
    # Full circuit
    # ------------------------------------------------------------------
    def _build_circuit(self) -> QuantumCircuit:
        label = "IQFT" if self.inverse else "QFT"
        qc = QuantumCircuit(self.n, self.n, name=label)

        # Prepare input state |input_state>
        for i, bit in enumerate(reversed(self.input_state)):
            if bit == "1":
                qc.x(i)
        qc.barrier()

        # Apply QFT or IQFT
        if self.inverse:
            self._iqft_core(qc, self.n)
        else:
            self._qft_core(qc, self.n)
        qc.barrier()

        qc.measure(range(self.n), range(self.n))
        return qc

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, shots: int = 4096) -> dict:
        sim = AerSimulator()
        tqc = transpile(self.circuit, sim)
        result = sim.run(tqc, shots=shots).result()
        counts = result.get_counts()

        # Convert counts to probability distribution
        probs = {k: v / shots for k, v in sorted(counts.items())}

        return {
            "algorithm"    : "QFT",
            "input_state"  : self.input_state,
            "inverse"      : self.inverse,
            "counts"       : counts,
            "probabilities": probs,
            "num_qubits"   : self.circuit.num_qubits,
            "depth"        : self.circuit.depth(),
            "gate_count"   : sum(self.circuit.count_ops().values()),
            "classical_complexity": self.CLASSICAL_COMPLEXITY,
            "quantum_complexity"  : self.QUANTUM_COMPLEXITY,
            "speedup"             : self.SPEEDUP,
        }
