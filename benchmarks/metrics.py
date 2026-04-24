"""
Benchmarks — Circuit Metrics & Speedup Analysis
================================================
Collects, structures and compares all algorithm metrics in one place.
"""

from __future__ import annotations
import time
import pandas as pd
import numpy as np


class BenchmarkSuite:
    """
    Runs all quantum algorithms, collects metrics, and produces
    structured DataFrames for analysis and plotting.
    """

    def __init__(self, algorithms: list):
        """
        Parameters
        ----------
        algorithms : list of algorithm instances (each must implement .run() and .get_metrics())
        """
        self.algorithms = algorithms
        self.metrics: list[dict] = []
        self.results: list[dict] = []

    def run_all(self, shots: int = 1024, verbose: bool = True) -> None:
        """Execute all algorithms and collect metrics."""
        for algo in self.algorithms:
            if verbose:
                print(f"  ▶  Running {algo.name} ...", end="", flush=True)
            t0 = time.perf_counter()
            result = algo.run(shots=shots)
            elapsed = time.perf_counter() - t0

            metrics = algo.get_metrics()
            metrics["simulation_time_s"] = round(elapsed, 4)
            self.metrics.append(metrics)
            self.results.append(result)

            if verbose:
                print(f"  done ({elapsed:.2f}s)")

    # ------------------------------------------------------------------
    # DataFrames
    # ------------------------------------------------------------------
    def circuit_metrics_df(self) -> pd.DataFrame:
        """Return a DataFrame with circuit-level metrics for each algorithm."""
        rows = []
        for m in self.metrics:
            rows.append(
                {
                    "Algorithm": m["algorithm"],
                    "Qubits": m["n_qubits"],
                    "Circuit Depth": m["circuit_depth"],
                    "Gate Count": m["gate_count"],
                    "Simulation Time (s)": m.get("simulation_time_s", float("nan")),
                }
            )
        return pd.DataFrame(rows).set_index("Algorithm")

    def complexity_df(self) -> pd.DataFrame:
        """Return a DataFrame with complexity and speedup information."""
        rows = []
        for m in self.metrics:
            rows.append(
                {
                    "Algorithm": m["algorithm"],
                    "Classical Complexity": m["classical_complexity"],
                    "Quantum Complexity": m["quantum_complexity"],
                    "Quantum Advantage": m["quantum_advantage"],
                }
            )
        return pd.DataFrame(rows).set_index("Algorithm")

    def speedup_data(self) -> dict[str, dict]:
        """
        Compute concrete speedup numbers for n in {4,8,16,20} qubits
        where applicable (for bar/line chart).
        """
        n_values = [4, 8, 12, 16]
        speedups = {}

        for n in n_values:
            N = 2 ** n
            speedups[n] = {
                "n": n,
                "N": N,
                "Deutsch-Jozsa classical": 2 ** (n - 1) + 1,
                "Deutsch-Jozsa quantum": 1,
                "Bernstein-Vazirani classical": n,
                "Bernstein-Vazirani quantum": 1,
                "Grover classical": N,
                "Grover quantum": int(np.pi / 4 * N ** 0.5),
                "QFT classical (FFT)": N * n,
                "QFT quantum": n ** 2,
            }

        return speedups

    def speedup_df(self) -> pd.DataFrame:
        """Return a long-form DataFrame for classical vs quantum query counts."""
        rows = []
        for n, data in self.speedup_data().items():
            N = data["N"]
            rows.append({"n (qubits)": n, "N": N, "Algorithm": "Deutsch-Jozsa",
                          "Classical": data["Deutsch-Jozsa classical"],
                          "Quantum": data["Deutsch-Jozsa quantum"]})
            rows.append({"n (qubits)": n, "N": N, "Algorithm": "Bernstein-Vazirani",
                          "Classical": data["Bernstein-Vazirani classical"],
                          "Quantum": data["Bernstein-Vazirani quantum"]})
            rows.append({"n (qubits)": n, "N": N, "Algorithm": "Grover's Search",
                          "Classical": data["Grover classical"],
                          "Quantum": data["Grover quantum"]})
            rows.append({"n (qubits)": n, "N": N, "Algorithm": "QFT",
                          "Classical": data["QFT classical (FFT)"],
                          "Quantum": data["QFT quantum"]})
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Return a formatted text summary of all results."""
        lines = [
            "=" * 65,
            "  QUANTUM ALGORITHMS COMPARATIVE STUDY — RESULTS SUMMARY",
            "=" * 65,
        ]
        for r in self.results:
            lines.append(f"\n▶ {r['algorithm']}")
            lines.append(f"  Qubits       : {r['n_qubits']}")
            lines.append(f"  Circuit Depth: {r.get('depth', r.get('circuit_depth', 'N/A'))}")
            lines.append(f"  Gate Count   : {r.get('gate_count', 'N/A')}")
            lines.append(f"  Classical    : {r.get('classical_complexity', '')}")
            lines.append(f"  Quantum      : {r.get('quantum_complexity', '')}")

            # Algorithm-specific highlights
            if "verdict" in r:
                lines.append(f"  DJ Verdict   : {r['verdict']} ({'✓' if r['correct'] else '✗'})")
            if "recovered_secret" in r:
                lines.append(f"  Secret found : {r['recovered_secret']} ({'✓' if r['correct'] else '✗'})")
            if "correct" in r and "target" in r:
                lines.append(f"  Target found : {r.get('found')} (target={r['target']}) — "
                              f"P={r.get('success_probability', '?')}")
            if "factors" in r:
                lines.append(f"  Factors of {r['N']} : {r.get('factors')} ({'✓' if r.get('correct') else '✗'})")
            if "approximation_ratio" in r:
                lines.append(f"  MaxCut ratio : {r['approximation_ratio']:.2%} "
                              f"(quantum={r['quantum_cut']}, optimal={r['classical_cut']})")

        lines.append("\n" + "=" * 65)
        return "\n".join(lines)
