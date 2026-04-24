"""
noise_analysis.py — Quantum Noise Models & Fidelity Study
==========================================================
Simulates four physically motivated noise models and measures how each
degrades algorithm performance across increasing noise strengths.

Noise models covered
--------------------
1. Depolarizing noise      — random Pauli errors (X, Y, Z) after every gate
2. Thermal relaxation      — energy decay (T1) and dephasing (T2) over gate time
3. Readout / SPAM error    — bit-flip during measurement (symmetric)
4. Combined (realistic)    — all three simultaneously, modelling a real device

Metrics collected per run
-------------------------
• fidelity        — overlap between noisy and ideal output distributions (TVD-based)
• success_rate    — fraction of shots giving the correct / expected answer
• tvd             — Total Variation Distance from ideal distribution

Usage
-----
    python noise_analysis.py

Results are printed as a table and saved to results/noise_results.json
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)

from algorithms import DeutschJozsa, BernsteinVazirani, Grover, QFT


# ══════════════════════════════════════════════════════════════════════
# Noise model factories
# ══════════════════════════════════════════════════════════════════════

def make_depolarizing_model(p1: float, p2: float) -> NoiseModel:
    """
    Depolarizing noise model.

    Parameters
    ----------
    p1 : single-qubit gate error probability  (0 ≤ p1 ≤ 1)
    p2 : two-qubit gate error probability      (0 ≤ p2 ≤ 1)

    Physics: After every gate, with probability p the qubit is replaced
    by the maximally mixed state ρ → (1-p)ρ + p·I/2.  Models imperfect
    gate pulses and environmental coupling equally on all qubits.
    """
    noise = NoiseModel()
    # Single-qubit gates
    err1 = depolarizing_error(p1, 1)
    noise.add_all_qubit_quantum_error(err1, ["h", "x", "rx", "ry", "rz", "u"])
    # Two-qubit gates
    err2 = depolarizing_error(p2, 2)
    noise.add_all_qubit_quantum_error(err2, ["cx", "cz", "cp", "rzz", "swap", "mcx"])
    return noise


def make_thermal_model(t1_us: float, t2_us: float, gate_time_ns: float) -> NoiseModel:
    """
    Thermal relaxation noise model.

    Parameters
    ----------
    t1_us       : T1 relaxation time in microseconds   (energy decay)
    t2_us       : T2 dephasing time   in microseconds   (phase coherence)
    gate_time_ns: gate execution time in nanoseconds

    Physics: T1 is the timescale for |1⟩ → |0⟩ decay (amplitude damping).
    T2 ≤ 2·T1 captures additional pure dephasing on top of T1.
    Longer gates relative to T1/T2 cause more decoherence.

    Constraint enforced: T2 ≤ 2·T1 (physical requirement).
    """
    t2_us = min(t2_us, 2 * t1_us)
    t1_ns = t1_us * 1_000
    t2_ns = t2_us * 1_000

    noise = NoiseModel()

    # Single-qubit gate error
    err1q = thermal_relaxation_error(t1_ns, t2_ns, gate_time_ns)
    noise.add_all_qubit_quantum_error(err1q, ["h", "x", "rx", "ry", "rz", "u"])

    # Two-qubit gate error (applied to each qubit independently)
    err2q = thermal_relaxation_error(t1_ns, t2_ns, gate_time_ns * 10)
    err2q_tensor = err2q.expand(err2q)
    noise.add_all_qubit_quantum_error(err2q_tensor, ["cx", "cz", "cp", "rzz", "swap"])

    return noise


def make_readout_model(p_meas: float) -> NoiseModel:
    """
    Symmetric readout / SPAM error model.

    Parameters
    ----------
    p_meas : probability of a bit-flip during measurement (0 → 1 or 1 → 0)

    Physics: State-preparation-and-measurement (SPAM) errors arise from
    imperfect initialisation and detector noise.  The confusion matrix is:
        P(0|0) = 1 - p_meas,   P(1|0) = p_meas
        P(1|1) = 1 - p_meas,   P(0|1) = p_meas
    """
    noise = NoiseModel()
    confusion = [[1 - p_meas, p_meas],
                 [p_meas,     1 - p_meas]]
    err = ReadoutError(confusion)
    noise.add_all_qubit_readout_error(err)
    return noise


def make_combined_model(
    p1: float     = 0.002,
    p2: float     = 0.01,
    t1_us: float  = 50.0,
    t2_us: float  = 70.0,
    gate_ns: float= 50.0,
    p_meas: float = 0.02,
) -> NoiseModel:
    """
    Realistic combined noise model (depolarizing + thermal + readout).

    Default parameters are loosely inspired by superconducting qubit devices
    (e.g. IBM Falcon r5 class), but scaled for fast simulation.
    """
    t2_us = min(t2_us, 2 * t1_us)
    t1_ns = t1_us * 1_000
    t2_ns = t2_us * 1_000

    noise = NoiseModel()

    # ── Gate errors: thermal relaxation + depolarizing ──
    therm1 = thermal_relaxation_error(t1_ns, t2_ns, gate_ns)
    dep1   = depolarizing_error(p1, 1)
    combined1 = therm1.compose(dep1)
    noise.add_all_qubit_quantum_error(combined1, ["h", "x", "rx", "ry", "rz", "u"])

    therm2 = thermal_relaxation_error(t1_ns, t2_ns, gate_ns * 10)
    therm2_tensor = therm2.expand(therm2)
    dep2   = depolarizing_error(p2, 2)
    combined2 = therm2_tensor.compose(dep2)
    noise.add_all_qubit_quantum_error(combined2, ["cx", "cz", "cp", "rzz", "swap"])

    # ── Readout error ──
    confusion = [[1 - p_meas, p_meas], [p_meas, 1 - p_meas]]
    noise.add_all_qubit_readout_error(ReadoutError(confusion))

    return noise


# ══════════════════════════════════════════════════════════════════════
# Metric helpers
# ══════════════════════════════════════════════════════════════════════

def total_variation_distance(p: dict, q: dict) -> float:
    """
    TVD = 0.5 · Σ |p(x) - q(x)|  over all basis states.
    TVD = 0 means identical distributions; TVD = 1 means disjoint.
    """
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def fidelity_from_tvd(tvd: float) -> float:
    """Approximate output-distribution fidelity as 1 - TVD."""
    return max(0.0, 1.0 - tvd)


def counts_to_probs(counts: dict, shots: int) -> dict:
    return {k: v / shots for k, v in counts.items()}


def run_circuit(circuit: QuantumCircuit, noise_model: NoiseModel | None,
                shots: int = 2048) -> dict:
    """Transpile and run a circuit, returning raw counts."""
    sim = AerSimulator()
    kwargs = {"noise_model": noise_model} if noise_model else {}
    tqc = transpile(circuit, sim)
    result = sim.run(tqc, shots=shots, **kwargs).result()
    return result.get_counts()


# ══════════════════════════════════════════════════════════════════════
# Per-algorithm noise sweep
# ══════════════════════════════════════════════════════════════════════

SHOTS = 2048

# Algorithms to test (name, instance, "ideal" answer key + expected value)
ALGORITHMS = [
    {
        "name"   : "Deutsch-Jozsa",
        "algo"   : DeutschJozsa(n=4, oracle_type="balanced"),
        "correct": lambda counts, shots: counts.get("0000", 0) / shots < 0.01,
        # Balanced → all-zeros probability should be ~0
    },
    {
        "name"   : "Bernstein-Vazirani",
        "algo"   : BernsteinVazirani(secret="1100"),
        "correct": lambda counts, shots: (
            max(counts, key=counts.get)[::-1] == "1100"
        ),
    },
    {
        "name"   : "Grover's Search",
        "algo"   : Grover(n=4, target="1010"),
        "correct": lambda counts, shots: (
            max(counts, key=counts.get)[::-1] == "1010"
        ),
    },
    {
        "name"   : "QFT",
        "algo"   : QFT(n=4),
        "correct": None,   # QFT has no single "correct" answer; use TVD only
    },
]

# Noise sweep levels
DEPOLARIZING_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]   # p1 (p2 = 5×p1)
THERMAL_T1_LEVELS   = [500, 200, 100, 50, 20, 10]              # T1 in µs (T2 = 0.8·T1)
READOUT_LEVELS      = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]      # p_meas


def sweep_noise(algo_entry: dict, noise_fn, levels: list, level_label: str) -> list[dict]:
    """
    Run an algorithm under increasing noise, collect TVD + fidelity + success.

    Parameters
    ----------
    algo_entry  : entry from ALGORITHMS list
    noise_fn    : callable(level) → NoiseModel | None
    levels      : list of noise parameter values to sweep
    level_label : human-readable name of the noise parameter
    """
    algo    = algo_entry["algo"]
    circuit = algo.circuit
    correct_fn = algo_entry["correct"]

    # Ideal (noiseless) distribution
    ideal_counts = run_circuit(circuit, noise_model=None, shots=SHOTS * 2)
    ideal_probs  = counts_to_probs(ideal_counts, SHOTS * 2)

    rows = []
    for level in levels:
        noise = noise_fn(level)
        noisy_counts = run_circuit(circuit, noise_model=noise, shots=SHOTS)
        noisy_probs  = counts_to_probs(noisy_counts, SHOTS)

        tvd = total_variation_distance(ideal_probs, noisy_probs)
        fid = fidelity_from_tvd(tvd)
        success = correct_fn(noisy_counts, SHOTS) if correct_fn else None

        rows.append({
            "algorithm"     : algo_entry["name"],
            "noise_type"    : level_label,
            "noise_level"   : level,
            "tvd"           : round(tvd, 4),
            "fidelity"      : round(fid, 4),
            "success"       : success,
        })

    return rows


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def print_sweep_table(rows: list[dict]) -> None:
    header = f"{'Level':>10}  {'TVD':>6}  {'Fidelity':>8}  {'Success':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        succ = ("✓" if r["success"] else "✗") if r["success"] is not None else "—"
        print(f"  {str(r['noise_level']):>8}  {r['tvd']:>6.4f}  {r['fidelity']:>8.4f}  {succ:>7}")


def main():
    print("\n" + "=" * 65)
    print("  ⚛   QUANTUM NOISE ANALYSIS")
    print("=" * 65)

    all_results = {}

    for algo_entry in ALGORITHMS:
        name = algo_entry["name"]
        print(f"\n{'─'*65}")
        print(f"  Algorithm: {name}")
        print(f"{'─'*65}")

        algo_results = {}

        # ── 1. Depolarizing noise ──────────────────────────────────────
        print("\n  [1] Depolarizing Noise  (p1 = single-qubit error rate)")
        rows_dep = sweep_noise(
            algo_entry,
            noise_fn   = lambda p: make_depolarizing_model(p, p * 5) if p > 0 else None,
            levels     = DEPOLARIZING_LEVELS,
            level_label= "depolarizing_p1",
        )
        print_sweep_table(rows_dep)
        algo_results["depolarizing"] = rows_dep

        # ── 2. Thermal relaxation ──────────────────────────────────────
        print("\n  [2] Thermal Relaxation  (T1 in µs, gate=50ns, T2=0.8·T1)")
        rows_therm = sweep_noise(
            algo_entry,
            noise_fn   = lambda t1: make_thermal_model(t1, t1 * 0.8, 50.0) if t1 < 1e6 else None,
            levels     = THERMAL_T1_LEVELS,
            level_label= "thermal_T1_us",
        )
        print_sweep_table(rows_therm)
        algo_results["thermal"] = rows_therm

        # ── 3. Readout error ───────────────────────────────────────────
        print("\n  [3] Readout / SPAM Error  (p_meas = bit-flip probability)")
        rows_ro = sweep_noise(
            algo_entry,
            noise_fn   = lambda p: make_readout_model(p) if p > 0 else None,
            levels     = READOUT_LEVELS,
            level_label= "readout_p",
        )
        print_sweep_table(rows_ro)
        algo_results["readout"] = rows_ro

        # ── 4. Combined realistic noise ────────────────────────────────
        print("\n  [4] Combined Noise  (depolarizing + thermal + readout, scaling factor)")
        scale_levels = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
        rows_comb = sweep_noise(
            algo_entry,
            noise_fn   = lambda s: make_combined_model(
                p1=0.002*s, p2=0.01*s, t1_us=max(1, 50/max(s,0.01)),
                t2_us=max(0.5, 35/max(s,0.01)), gate_ns=50, p_meas=0.02*s
            ) if s > 0 else None,
            levels     = scale_levels,
            level_label= "combined_scale",
        )
        print_sweep_table(rows_comb)
        algo_results["combined"] = rows_comb

        all_results[name] = algo_results

    # ── Save results ───────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = "results/noise_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*65}")
    print(f"  ✅  Noise analysis complete. Results saved → {out_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()