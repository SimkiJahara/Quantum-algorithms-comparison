"""
visualize_results.py — Quantum Noise Analysis Visualization
============================================================
Generates publication-quality figures from noise_results.json.

Produces:
  figures/fig1_fidelity_degradation.png  — 4×4 fidelity curves (main figure)
  figures/fig2_noise_threshold.png       — per-algorithm threshold bar chart
  figures/fig3_circuit_complexity.png    — circuit metrics comparison

Usage
-----
    python visualize_results.py

Requires: matplotlib, numpy
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"       : "serif",
    "font.size"         : 9,
    "axes.titlesize"    : 10,
    "axes.labelsize"    : 9,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.25,
    "grid.linewidth"    : 0.5,
    "legend.frameon"    : False,
    "figure.dpi"        : 150,
    "savefig.dpi"       : 300,
    "savefig.bbox"      : "tight",
})

ALGO_COLORS = {
    "Deutsch-Jozsa"    : "#2563EB",   # blue
    "Bernstein-Vazirani": "#16A34A",  # green
    "Grover's Search"  : "#DC2626",   # red
    "QFT"              : "#9333EA",   # purple
}

ALGO_MARKERS = {
    "Deutsch-Jozsa"    : "o",
    "Bernstein-Vazirani": "s",
    "Grover's Search"  : "^",
    "QFT"              : "D",
}

NOISE_LABELS = {
    "depolarizing" : "Depolarizing Noise  (p₁)",
    "thermal"      : "Thermal Relaxation  (T₁, µs)",
    "readout"      : "Readout / SPAM Error  (p_meas)",
    "combined"     : "Combined Realistic Noise  (scale factor)",
}

NOISE_KEYS = ["depolarizing", "thermal", "readout", "combined"]

os.makedirs("figures", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════

def load_results(path: str = "results/noise_results.json") -> dict:
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════
# Figure 1 — Fidelity Degradation (4 × 4 grid)
# ══════════════════════════════════════════════════════════════════════

def fig1_fidelity_degradation(data: dict):
    fig = plt.figure(figsize=(13, 9))
    fig.suptitle(
        "Quantum Algorithm Fidelity Under Increasing Noise Intensity",
        fontsize=13, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.32)

    for col, noise_key in enumerate(NOISE_KEYS):
        ax = fig.add_subplot(gs[col // 2, col % 2])
        ax.set_title(NOISE_LABELS[noise_key], pad=6)
        ax.set_ylabel("Output Fidelity  (1 − TVD)")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.9, color="gray", lw=0.8, ls="--", alpha=0.6, label="F = 0.90 threshold")

        for algo_name, algo_data in data.items():
            if noise_key not in algo_data:
                continue
            rows = algo_data[noise_key]
            levels = [r["noise_level"] for r in rows]
            fids   = [r["fidelity"]    for r in rows]

            # For thermal: invert x-axis (higher T1 = less noise)
            if noise_key == "thermal":
                levels = levels[::-1]
                fids   = fids[::-1]

            ax.plot(
                levels, fids,
                marker=ALGO_MARKERS[algo_name],
                color=ALGO_COLORS[algo_name],
                linewidth=1.6,
                markersize=5,
                label=algo_name,
            )

            # Mark where fidelity first crosses 0.9
            for i, f in enumerate(fids):
                if f < 0.90:
                    ax.axvline(levels[i], color=ALGO_COLORS[algo_name],
                               lw=0.6, ls=":", alpha=0.4)
                    break

        if noise_key == "thermal":
            ax.set_xlabel("T₁ (µs)  ← more noise")
            ax.invert_xaxis()
        else:
            ax.set_xlabel(ax.get_title().split("(")[1].rstrip(")") if "(" in ax.get_title() else "Noise Level")

    # Shared legend
    handles = [
        Line2D([0], [0], color=ALGO_COLORS[a], marker=ALGO_MARKERS[a],
               linewidth=1.6, markersize=5, label=a)
        for a in ALGO_COLORS
    ] + [Line2D([0], [0], color="gray", lw=0.8, ls="--", label="F = 0.90 threshold")]

    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.04), fontsize=8.5)

    path = "figures/fig1_fidelity_degradation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓  Saved {path}")


# ══════════════════════════════════════════════════════════════════════
# Figure 2 — Noise Threshold Bar Chart
# ══════════════════════════════════════════════════════════════════════

def _find_threshold(rows: list, fid_threshold: float = 0.90) -> float | None:
    """Return the noise level at which fidelity first drops below threshold."""
    levels = [r["noise_level"] for r in rows]
    fids   = [r["fidelity"]    for r in rows]
    for lv, fid in zip(levels, fids):
        if fid < fid_threshold:
            return lv
    return None   # never crossed


def fig2_noise_thresholds(data: dict):
    noise_display = {
        "depolarizing": "Depolarizing (p₁)",
        "readout"     : "Readout Error (p_meas)",
        "combined"    : "Combined (scale)",
    }

    algos = list(data.keys())
    x = np.arange(len(algos))
    width = 0.25
    n_groups = len(noise_display)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Noise Threshold at F = 0.90  (higher = more resilient)", fontweight="bold")
    ax.set_ylabel("Noise Level at F = 0.90 Crossover")
    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels([a.replace(" ", "\n") for a in algos], fontsize=8)

    palette = ["#2563EB", "#DC2626", "#F59E0B"]
    for gi, (noise_key, noise_label) in enumerate(noise_display.items()):
        thresholds = []
        for algo_name in algos:
            rows = data[algo_name].get(noise_key, [])
            t = _find_threshold(rows)
            thresholds.append(t if t is not None else 0)

        bars = ax.bar(
            x + gi * width, thresholds, width,
            label=noise_label, color=palette[gi], alpha=0.82,
            edgecolor="white", linewidth=0.6,
        )
        for bar, val in zip(bars, thresholds):
            if val:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.legend(fontsize=8)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    path = "figures/fig2_noise_threshold.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓  Saved {path}")


# ══════════════════════════════════════════════════════════════════════
# Figure 3 — Circuit Complexity Comparison
# ══════════════════════════════════════════════════════════════════════

def fig3_circuit_complexity():
    # Hard-coded from main.py output (or load from comparison_results.json)
    try:
        with open("results/comparison_results.json") as f:
            comp = json.load(f)
        algos  = [r["Algorithm"]    for r in comp["results"]]
        depths = [int(r["Depth"])   for r in comp["results"]]
        gates  = [int(r["Gates"])   for r in comp["results"]]
        qubits = [int(r["Qubits"])  for r in comp["results"]]
    except Exception:
        # Fallback to values from the terminal output
        algos  = ["Deutsch-Jozsa", "Bernstein-Vazirani", "Grover's Search",
                  "QFT", "Shor's Factoring", "QAOA"]
        depths = [8,  6,  38, 10, 26, 14]
        gates  = [22, 20, 96, 21, 68, 31]
        qubits = [5,  5,  4,  4,  12,  4]

    x = np.arange(len(algos))
    colors = ["#2563EB","#16A34A","#DC2626","#9333EA","#F59E0B","#0891B2"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("Circuit Complexity Metrics Across Algorithms", fontsize=12, fontweight="bold")

    for ax, vals, title, ylabel in zip(
        axes,
        [depths, gates, qubits],
        ["Circuit Depth", "Total Gate Count", "Qubit Count"],
        ["Depth (layers)", "Gates", "Qubits"],
    ):
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.7)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace(" ", "\n").replace("'s", "'s\n") for a in algos], fontsize=7.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = "figures/fig3_circuit_complexity.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓  Saved {path}")


# ══════════════════════════════════════════════════════════════════════
# Figure 4 — TVD Heatmap (bonus: research-style)
# ══════════════════════════════════════════════════════════════════════

def fig4_tvd_heatmap(data: dict):
    """Heatmap: rows = algorithms, cols = noise levels, color = TVD."""
    noise_key = "depolarizing"
    algos = list(data.keys())
    levels = [r["noise_level"] for r in next(iter(data.values()))[noise_key]]

    matrix = np.array([
        [r["tvd"] for r in data[algo][noise_key]]
        for algo in algos
    ])

    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.8)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([str(l) for l in levels], fontsize=8)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos, fontsize=9)
    ax.set_xlabel("Depolarizing Error Rate  p₁")
    ax.set_title("Total Variation Distance (TVD) Under Depolarizing Noise\n"
                 "darker red = further from ideal distribution", fontweight="bold")

    for i in range(len(algos)):
        for j in range(len(levels)):
            val = matrix[i, j]
            color = "white" if val > 0.4 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7.5, color=color)

    fig.colorbar(im, ax=ax, label="TVD", fraction=0.03, pad=0.02)
    plt.tight_layout()

    path = "figures/fig4_tvd_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓  Saved {path}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 55)
    print("  ⚛   QUANTUM NOISE — FIGURE GENERATION")
    print("=" * 55)

    data = load_results()

    print("\n  Generating figures...")
    fig1_fidelity_degradation(data)
    fig2_noise_thresholds(data)
    fig3_circuit_complexity()
    fig4_tvd_heatmap(data)

    print(f"\n  ✅  All figures saved to ./figures/")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()