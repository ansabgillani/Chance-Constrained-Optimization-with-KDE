"""Shared figure style for all matplotlib generators of the thesis.

Usage at the top of a generator:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from thesis_style import apply_style, COLORS, WBOX, width_in
    apply_style()

Sizes are in printed points: draw each figure at the width LaTeX includes it
(width_in(fraction of \\textwidth)) so that 1 pt in code is 1 pt on paper.
"""
import matplotlib.pyplot as plt

TEXTWIDTH_IN = 15 / 2.54  # \textwidth = 15 cm

# One colour per method and per role, used by every figure in the thesis.
COLORS = {
    "nominal": "#7a7a7a",    # nominal / mean-value / deterministic
    "scenario": "#d9842b",   # scenario approach / robust
    "unbiased": "#1f4e79",   # unbiased (symmetric) KDE
    "biased": "#2e7d4f",     # biased (shifted Epanechnikov) KDE
    "truth": "#222222",      # true quantities, indicators, references
    "violation": "#b3392e",  # violation / infeasible / unsafe
    "safe": "#2e7d4f",       # safe / feasible (same green as the biased KDE)
    "stochastic": "#6a4c93", # stochastic programming (taxonomy only)
    "chance": "#1f4e79",     # chance-constrained paradigm (taxonomy, spectrum)
    "annotation": "#333333",
    "muted": "#555555",
}

# White backplate for the rare label that must sit over plotted content.
WBOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)


def width_in(fraction=1.0):
    """Printed width in inches of a figure included at fraction*\\textwidth."""
    return TEXTWIDTH_IN * fraction


def apply_style():
    plt.rcParams.update({
        "font.family": "Charter",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Charter",
        "mathtext.it": "Charter:italic",
        "mathtext.bf": "Charter:bold",
        "mathtext.fallback": "stix",
        "pdf.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "normal",
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
