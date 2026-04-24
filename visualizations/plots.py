"""
Visualisations
==============
Generates all plots for the comparative study:
  1. Circuit metrics bar chart (depth & gate count per algorithm)
  2. Classical vs Quantum query complexity (log scale)
  3. Grover search probability histogram
  4. QAOA MaxCut results (measurement distribution)
  5. Simulation time comparison
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Style ──────────────────────────────────────────────────────────────────
COLORS = {
    "quantum": "#4C72B0",
    "classical": "#DD8452",
    "accent1": "#55A868",
    "accent2": "#C44E52",
    "accent3": "#8172B2",
    "accent4": "#937860",
    "bg": "#F8F9FA",
    "grid": "#DEE2E6",
}
ALGO_COLORS = [
    "#4C72B0", "#DD8452", "#55A868",
    "#C44E52", "#8172B2", "#937860",
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": COLORS["bg"],
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.8,
    }
)


class Plotter:
    """Generates and saves all comparison plots."""

    def __init__(self, benchmark_suite, output_dir: str = "results"):
        self.suite = benchmark_suite
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Circuit metrics (depth & gate count)
    # ------------------------------------------------------------------
    def plot_circuit_metrics(self) -> str:
        df = self.suite.circuit_metrics_df()
        algos = df.index.tolist()
        x = np.arange(len(algos))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Circuit Metrics Comparison", fontsize=16, fontweight="bold", y=1.01)

        # Depth
        bars1 = ax1.bar(x, df["Circuit Depth"], width * 2, color=ALGO_COLORS, edgecolor="white", linewidth=0.8)
        ax1.set_title("Circuit Depth", fontsize=13)
        ax1.set_xticks(x)
        ax1.set_xticklabels(algos, rotation=35, ha="right", fontsize=9)
        ax1.set_ylabel("Depth (gate layers)")
        for bar in bars1:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

        # Gate count
        bars2 = ax2.bar(x, df["Gate Count"], width * 2, color=ALGO_COLORS, edgecolor="white", linewidth=0.8)
        ax2.set_title("Total Gate Count", fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(algos, rotation=35, ha="right", fontsize=9)
        ax2.set_ylabel("Number of gates")
        for bar in bars2:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "circuit_metrics.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 2. Classical vs Quantum complexity (log scale)
    # ------------------------------------------------------------------
    def plot_speedup_analysis(self) -> str:
        df = self.suite.speedup_df()
        algorithms = df["Algorithm"].unique()
        n_values = sorted(df["n (qubits)"].unique())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Classical vs Quantum Query Complexity  (log scale)",
            fontsize=16, fontweight="bold",
        )

        for ax, algo in zip(axes.flat, algorithms):
            sub = df[df["Algorithm"] == algo]
            ns = sub["n (qubits)"].values
            classical = sub["Classical"].values
            quantum = sub["Quantum"].values

            ax.semilogy(ns, classical, "o-", color=COLORS["classical"],
                        linewidth=2.5, markersize=8, label="Classical")
            ax.semilogy(ns, quantum, "s--", color=COLORS["quantum"],
                        linewidth=2.5, markersize=8, label="Quantum")
            ax.fill_between(ns, quantum, classical, alpha=0.12, color=COLORS["accent1"],
                            label="Speedup region")
            ax.set_title(algo, fontsize=12, fontweight="bold")
            ax.set_xlabel("n (qubits)")
            ax.set_ylabel("Queries / Operations")
            ax.legend(fontsize=9)
            ax.set_xticks(ns)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "speedup_analysis.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 3. Grover probability histogram
    # ------------------------------------------------------------------
    def plot_grover_results(self) -> str:
        grover_result = next(
            r for r in self.suite.results if "Grover" in r["algorithm"]
        )
        counts = grover_result["counts"]
        target_bs = grover_result["target_bits"]
        n = grover_result["n_qubits"]
        total = sum(counts.values())

        # Sort states; highlight target
        states = sorted(counts.keys())
        probs = [counts.get(s, 0) / total for s in states]
        colors = [COLORS["accent2"] if s == target_bs else COLORS["quantum"] for s in states]

        fig, ax = plt.subplots(figsize=(14, 5))
        bars = ax.bar(states, probs, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_title(
            f"Grover's Search — Measurement Probabilities\n"
            f"n={n} qubits  |  Target = |{target_bs}⟩  |  "
            f"Iterations = {grover_result['n_iterations']}",
            fontsize=13,
        )
        ax.set_xlabel("Basis State")
        ax.set_ylabel("Probability")
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels(states, rotation=90, fontsize=7)

        target_patch = mpatches.Patch(color=COLORS["accent2"], label=f"Target |{target_bs}⟩")
        other_patch = mpatches.Patch(color=COLORS["quantum"], label="Other states")
        ax.legend(handles=[target_patch, other_patch])

        plt.tight_layout()
        path = os.path.join(self.output_dir, "grover_results.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 4. QAOA MaxCut distribution
    # ------------------------------------------------------------------
    def plot_qaoa_results(self) -> str:
        qaoa_result = next(
            r for r in self.suite.results if "QAOA" in r["algorithm"]
        )
        counts = qaoa_result["counts"]
        total = sum(counts.values())
        edges = qaoa_result["edges"]

        def cut_value(bs):
            assignment = {i: int(b) for i, b in enumerate(reversed(bs))}
            return sum(1 for u, v in edges if assignment[u] != assignment[v])

        states = sorted(counts.keys(), key=cut_value, reverse=True)
        probs = [counts[s] / total for s in states]
        cuts = [cut_value(s) for s in states]
        opt_cut = qaoa_result["classical_cut"]
        colors = [COLORS["accent1"] if c == opt_cut else COLORS["quantum"] for c in cuts]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"QAOA MaxCut Results  (p={qaoa_result['p_layers']} layers, "
            f"n={qaoa_result['n_qubits']} nodes)\n"
            f"Approximation ratio = {qaoa_result['approximation_ratio']:.1%}",
            fontsize=13, fontweight="bold",
        )

        # Measurement distribution
        ax1.bar(states, probs, color=colors, edgecolor="white")
        ax1.set_title("Measurement Distribution")
        ax1.set_xlabel("Bitstring partition")
        ax1.set_ylabel("Probability")
        ax1.tick_params(axis="x", rotation=45)
        optimal_patch = mpatches.Patch(color=COLORS["accent1"], label=f"Optimal cut = {opt_cut}")
        other_patch = mpatches.Patch(color=COLORS["quantum"], label="Suboptimal")
        ax1.legend(handles=[optimal_patch, other_patch])

        # Cut value histogram
        cut_counts: dict[int, float] = {}
        for s, p in zip(states, probs):
            c = cut_value(s)
            cut_counts[c] = cut_counts.get(c, 0) + p
        ax2.bar(cut_counts.keys(), cut_counts.values(),
                color=[COLORS["accent1"] if k == opt_cut else COLORS["classical"]
                       for k in cut_counts], edgecolor="white")
        ax2.set_title("Cut Value Distribution")
        ax2.set_xlabel("Cut value")
        ax2.set_ylabel("Probability mass")
        ax2.set_xticks(list(cut_counts.keys()))

        plt.tight_layout()
        path = os.path.join(self.output_dir, "qaoa_results.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 5. Simulation time comparison
    # ------------------------------------------------------------------
    def plot_simulation_times(self) -> str:
        df = self.suite.circuit_metrics_df()
        algos = df.index.tolist()
        times = df["Simulation Time (s)"].values

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(algos, times, color=ALGO_COLORS, edgecolor="white")
        ax.set_title("Classical Simulation Time per Algorithm", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (seconds)")
        for bar, t in zip(bars, times):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{t:.2f}s", va="center", fontsize=9)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "simulation_times.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 6. Summary dashboard
    # ------------------------------------------------------------------
    def plot_dashboard(self) -> str:
        """Combine all key plots into one dashboard figure."""
        df_circuit = self.suite.circuit_metrics_df()
        df_speed = self.suite.speedup_df()
        algos = df_circuit.index.tolist()
        x = np.arange(len(algos))

        grover_result = next(r for r in self.suite.results if "Grover" in r["algorithm"])
        counts_g = grover_result["counts"]
        target_bs = grover_result["target_bits"]
        total_g = sum(counts_g.values())
        states_g = sorted(counts_g.keys())
        probs_g = [counts_g.get(s, 0) / total_g for s in states_g]
        colors_g = [COLORS["accent2"] if s == target_bs else COLORS["quantum"] for s in states_g]

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(
            "Quantum Algorithms — Comparative Study Dashboard",
            fontsize=18, fontweight="bold", y=0.98,
        )
        gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # (a) Circuit Depth
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(x, df_circuit["Circuit Depth"], color=ALGO_COLORS, edgecolor="white")
        ax1.set_title("(a) Circuit Depth", fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(algos, rotation=40, ha="right", fontsize=8)
        ax1.set_ylabel("Depth")

        # (b) Gate Count
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(x, df_circuit["Gate Count"], color=ALGO_COLORS, edgecolor="white")
        ax2.set_title("(b) Gate Count", fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(algos, rotation=40, ha="right", fontsize=8)
        ax2.set_ylabel("Gates")

        # (c) Simulation Time
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.barh(algos, df_circuit["Simulation Time (s)"].values, color=ALGO_COLORS, edgecolor="white")
        ax3.set_title("(c) Simulation Time", fontweight="bold")
        ax3.set_xlabel("Seconds")

        # (d) Grover probability
        ax4 = fig.add_subplot(gs[1, :2])
        ax4.bar(states_g, probs_g, color=colors_g, edgecolor="white", linewidth=0.5)
        ax4.set_title(
            f"(d) Grover Search — target |{target_bs}⟩ (n={grover_result['n_qubits']})",
            fontweight="bold",
        )
        ax4.set_xlabel("State")
        ax4.set_ylabel("Probability")
        ax4.set_xticks(range(len(states_g)))
        ax4.set_xticklabels(states_g, rotation=90, fontsize=6)

        # (e) Speedup (DJ as example)
        ax5 = fig.add_subplot(gs[1, 2])
        dj = df_speed[df_speed["Algorithm"] == "Deutsch-Jozsa"]
        gr = df_speed[df_speed["Algorithm"] == "Grover's Search"]
        ax5.semilogy(dj["n (qubits)"], dj["Classical"], "o-", color=COLORS["classical"],
                     label="DJ Classical")
        ax5.semilogy(dj["n (qubits)"], dj["Quantum"], "s--", color=COLORS["quantum"],
                     label="DJ Quantum")
        ax5.semilogy(gr["n (qubits)"], gr["Classical"], "o-", color=COLORS["accent2"],
                     label="Grover Classical")
        ax5.semilogy(gr["n (qubits)"], gr["Quantum"], "s--", color=COLORS["accent1"],
                     label="Grover Quantum")
        ax5.set_title("(e) Query Complexity (log)", fontweight="bold")
        ax5.set_xlabel("n (qubits)")
        ax5.set_ylabel("Queries")
        ax5.legend(fontsize=7)

        path = os.path.join(self.output_dir, "dashboard.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------
    def generate_all(self, verbose: bool = True) -> dict[str, str]:
        """Generate and save all plots. Returns {name: filepath}."""
        plots = {}
        steps = [
            ("circuit_metrics", self.plot_circuit_metrics),
            ("speedup_analysis", self.plot_speedup_analysis),
            ("grover_results", self.plot_grover_results),
            ("qaoa_results", self.plot_qaoa_results),
            ("simulation_times", self.plot_simulation_times),
            ("dashboard", self.plot_dashboard),
        ]
        for name, fn in steps:
            if verbose:
                print(f"  📊  Plotting {name} ...", end="", flush=True)
            path = fn()
            plots[name] = path
            if verbose:
                print(f"  saved → {path}")
        return plots
