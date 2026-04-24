"""
main.py — Quantum Algorithms Comparative Study
===============================================
Entry point: instantiates all algorithms, runs benchmarks, prints
the summary table, and saves results to ./results/

Usage
-----
    python main.py
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

from algorithms import (
    DeutschJozsa,
    BernsteinVazirani,
    Grover,
    QFT,
    Shor,
    QAOA,
)
from benchmarks import Benchmarker


def print_table(rows: list[dict]) -> None:
    """Pretty-print a list of dicts as a fixed-width table."""
    if not rows:
        return
    col_widths = {k: len(k) for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            col_widths[k] = max(col_widths[k], len(str(v)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths.values()) + "-+"
    header = "| " + " | ".join(k.ljust(col_widths[k]) for k in col_widths) + " |"
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(row[k]).ljust(col_widths[k]) for k in col_widths) + " |")
    print(sep)


def main():
    print("\n" + "=" * 65)
    print("  ⚛   QUANTUM ALGORITHMS COMPARATIVE STUDY")
    print("      Framework : Qiskit + Qiskit-Aer")
    print("=" * 65 + "\n")

    print("Running algorithms...\n")
    suite = Benchmarker(n=4)
    suite.run_all()

    print("\n📋  Circuit Metrics & Complexity Table")
    print("-" * 65)
    print_table(suite.summary_table())

    print("\n✅  Saving results...")
    suite.save_results("results/comparison_results.json")

    print("\n🎉  Done! Results saved to ./results/\n")


if __name__ == "__main__":
    main()