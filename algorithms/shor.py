"""
Shor's Factoring Algorithm
===========================
Problem : Factor an integer N into its prime factors.

Classical best     : O(exp( (log N)^(1/3) · (log log N)^(2/3) ))  — GNFS
Quantum complexity : O((log N)³)                                    — polynomial!

Key idea: Factoring reduces to finding the period r of  f(x) = a^x mod N.
          Quantum Phase Estimation (which uses QFT) finds r exponentially faster
          than any known classical algorithm.  Given r, classical math gives
          factors:  gcd(a^(r/2) ± 1,  N).

Implementation note:
    A fully general Shor circuit requires O((log N)³) qubits, which is infeasible
    on today's simulators for large N.  This implementation demonstrates the
    quantum period-finding circuit for N=15 (a=7), the textbook example,
    using 8 counting qubits + 4 target qubits.
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from fractions import Fraction


class Shor:
    """
    Simplified Shor's algorithm for N=15, a=7.
    Demonstrates the full Quantum Phase Estimation + QFT structure.
    """

    CLASSICAL_COMPLEXITY = "O(exp((logN)^(1/3)(loglogN)^(2/3)))"
    QUANTUM_COMPLEXITY   = "O((log N)³)"
    SPEEDUP              = "Super-polynomial"

    def __init__(self, N: int = 15, a: int = 7):
        self.N = N
        self.a = a
        # counting register size (accuracy qubits)
        self.n_count = 8
        # target register size: ceil(log2(N))
        self.n_target = math.ceil(math.log2(N))
        self.circuit = self._build_circuit()

    # ------------------------------------------------------------------
    # Modular exponentiation oracle  a^x mod N  for N=15, a=7
    # Hand-coded controlled-U gates based on the known period r=4
    # ------------------------------------------------------------------
    def _c_amod15(self, a: int, power: int) -> QuantumCircuit:
        """Controlled modular exponentiation: multiply by a^2^power mod 15."""
        if a not in [2, 4, 7, 8, 11, 13]:
            raise ValueError("a must be coprime to 15 and in [2,4,7,8,11,13]")
        U = QuantumCircuit(4, name=f"U^{2**power} a={a}")
        for _ in range(2 ** power):
            if a in [2, 13]:
                U.swap(0, 1); U.swap(1, 2); U.swap(2, 3)
            if a in [7, 8]:
                U.swap(2, 3); U.swap(1, 2); U.swap(0, 1)
            if a in [4, 11]:
                U.swap(1, 3); U.swap(0, 2)
            if a in [7, 11, 13]:
                for q in range(4): U.x(q)
        # Wrap as controlled gate
        c_U = U.to_gate().control(1)
        qc = QuantumCircuit(5, name=f"cU^{2**power}")
        qc.append(c_U, range(5))
        return qc

    # ------------------------------------------------------------------
    # QFT inverse (for QPE readout)
    # ------------------------------------------------------------------
    def _iqft(self, n: int) -> QuantumCircuit:
        qc = QuantumCircuit(n, name="IQFT")
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)
        for j in range(n):
            for k in range(j):
                qc.cp(-np.pi / 2 ** (j - k), k, j)
            qc.h(j)
        return qc

    # ------------------------------------------------------------------
    # Full QPE circuit
    # ------------------------------------------------------------------
    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(
            self.n_count + self.n_target,
            self.n_count,
            name="Shor QPE"
        )

        # Initialize counting register in superposition
        qc.h(range(self.n_count))

        # Initialize target register to |1>
        qc.x(self.n_count)
        qc.barrier()

        # Apply controlled-U^(2^j) for each counting qubit
        for j in range(self.n_count):
            try:
                c_U = self._c_amod15(self.a, j)
                qc.append(
                    c_U.to_gate(),
                    [j] + list(range(self.n_count, self.n_count + self.n_target))
                )
            except Exception:
                pass
        qc.barrier()

        # Inverse QFT on counting register
        iqft = self._iqft(self.n_count)
        qc.compose(iqft, qubits=range(self.n_count), inplace=True)
        qc.barrier()

        # Measure counting register
        qc.measure(range(self.n_count), range(self.n_count))
        return qc

    # ------------------------------------------------------------------
    # Classical post-processing: extract factors from measured phase
    # ------------------------------------------------------------------
    def _phase_to_period(self, measured: int) -> int | None:
        phase = measured / (2 ** self.n_count)
        frac = Fraction(phase).limit_denominator(self.N)
        r = frac.denominator
        return r if r % 2 == 0 and pow(self.a, r, self.N) == 1 else None

    def _find_factors(self, r: int) -> tuple[int, int] | None:
        if r is None or r % 2 != 0:
            return None
        guess1 = math.gcd(self.a ** (r // 2) + 1, self.N)
        guess2 = math.gcd(self.a ** (r // 2) - 1, self.N)
        factors = [g for g in (guess1, guess2) if 1 < g < self.N]
        if len(factors) == 2:
            return tuple(sorted(factors))
        return None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, shots: int = 2048) -> dict:
        sim = AerSimulator()
        tqc = transpile(self.circuit, sim)
        result = sim.run(tqc, shots=shots).result()
        counts = result.get_counts()

        factors_found = None
        period_found = None
        for measured_str, freq in sorted(counts.items(), key=lambda x: -x[1]):
            measured_int = int(measured_str, 2)
            r = self._phase_to_period(measured_int)
            if r:
                factors = self._find_factors(r)
                if factors:
                    period_found = r
                    factors_found = factors
                    break

        return {
            "algorithm"     : "Shor's Factoring",
            "N"             : self.N,
            "a"             : self.a,
            "period"        : period_found,
            "factors"       : factors_found,
            "correct"       : factors_found == (3, 5),
            "counts"        : counts,
            "num_qubits"    : self.circuit.num_qubits,
            "depth"         : self.circuit.depth(),
            "gate_count"    : sum(self.circuit.count_ops().values()),
            "classical_complexity": self.CLASSICAL_COMPLEXITY,
            "quantum_complexity"  : self.QUANTUM_COMPLEXITY,
            "speedup"             : self.SPEEDUP,
        }
