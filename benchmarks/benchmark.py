"""
Benchmarking harness for all quantum algorithms.
Collects: runtime, circuit depth, gate count, qubit count, correctness.
"""

import time
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algorithms import DeutschJozsa, BernsteinVazirani, Grover, QFT, Shor, QAOA


class Benchmarker:
    """Runs all algorithms and collects comparative metrics."""

    def __init__(self, n: int = 4):
        self.n = n
        self.results = {}

    def run_all(self) -> dict:
        runs = [
            ("Deutsch-Jozsa",       self._run_dj),
            ("Bernstein-Vazirani",  self._run_bv),
            ("Grover's Search",     self._run_grover),
            ("QFT",                 self._run_qft),
            ("Shor's Factoring",    self._run_shor),
            ("QAOA",                self._run_qaoa),
        ]

        for name, fn in runs:
            print(f"  ▶ Running {name}...", end=" ", flush=True)
            t0 = time.perf_counter()
            result = fn()
            elapsed = time.perf_counter() - t0
            result["runtime_s"] = round(elapsed, 3)
            self.results[name] = result
            print(f"done  ({elapsed:.2f}s)")

        return self.results

    # ------------------------------------------------------------------
    def _run_dj(self):
        algo = DeutschJozsa(n=self.n, oracle_type="balanced")
        return algo.run()

    def _run_bv(self):
        secret = "1" * (self.n // 2) + "0" * (self.n - self.n // 2)
        algo = BernsteinVazirani(secret=secret)
        return algo.run()

    def _run_grover(self):
        algo = Grover(n=self.n)
        return algo.run()

    def _run_qft(self):
        algo = QFT(n=self.n)
        return algo.run()

    def _run_shor(self):
        algo = Shor(N=15, a=7)
        return algo.run()

    def _run_qaoa(self):
        algo = QAOA(p=2)
        return algo.run()

    # ------------------------------------------------------------------
    def summary_table(self) -> list[dict]:
        rows = []
        for name, r in self.results.items():
            rows.append({
                "Algorithm"          : name,
                "Qubits"             : r.get("num_qubits", "—"),
                "Depth"              : r.get("depth", "—"),
                "Gates"              : r.get("gate_count", "—"),
                "Runtime (s)"        : r.get("runtime_s", "—"),
                "Classical Complexity": r.get("classical_complexity", "—"),
                "Quantum Complexity"  : r.get("quantum_complexity", "—"),
                "Speedup"            : r.get("speedup", "—"),
            })
        return rows

    def save_results(self, path: str = "results/comparison_results.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Counts dicts can be large; slim them before saving
        slim = {}
        for k, v in self.results.items():
            slim[k] = {key: val for key, val in v.items() if key != "counts"}
        with open(path, "w") as f:
            json.dump(slim, f, indent=2, default=str)
        print(f"\n  Results saved → {path}")
