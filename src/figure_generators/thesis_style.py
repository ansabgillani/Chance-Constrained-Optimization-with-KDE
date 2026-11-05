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
# Hues are kept; lightness is spaced so that roles drawn together print as
# clearly different grays in black and white (printed gray in the comment).
COLORS = {
    "nominal": "#999999",    # 0.60  nominal / mean-value / deterministic
    "scenario": "#e4a666",   # 0.70  scenario approach / robust
    "unbiased": "#1f4e79",   # 0.27  unbiased (symmetric) KDE
    "biased": "#3da669",     # 0.50  biased (shifted Epanechnikov) KDE
    "truth": "#222222",      # 0.13  true quantities, indicators, references
    "violation": "#b3392e",  # 0.36  violation / infeasible / unsafe
    "safe": "#3da669",       # 0.50  safe / feasible (same green as the biased KDE)
    "stochastic": "#8160ad", # 0.45  stochastic programming (taxonomy only)
    "chance": "#1f4e79",     # 0.27  chance-constrained paradigm (taxonomy, spectrum)
    "annotation": "#333333",
    "muted": "#555555",
}

# Print-safe line per role: the same pattern and marker in every figure, so a
# legend still reads when the thesis is printed in black and white.
LINE = {
    "truth": dict(color=COLORS["truth"], ls="-"),
    "unbiased": dict(color=COLORS["unbiased"], ls="-", marker="o"),
    "biased": dict(color=COLORS["biased"], ls="--", marker="^"),
    "nominal": dict(color=COLORS["nominal"], ls=":", marker="D"),
    "scenario": dict(color=COLORS["scenario"], ls="-.", marker="s"),
    "violation": dict(color=COLORS["violation"], ls=(0, (3, 1.2)), marker="x"),
}
LINE["safe"] = LINE["biased"]
LINE["chance"] = LINE["unbiased"]

# Unsafe / violation regions are hatched; safe regions use a plain light fill.
HATCH_UNSAFE = "////"
# Light region fills. Safe regions use SAFE_FILL plain; unsafe regions use
# UNSAFE_FILL under HATCH_UNSAFE. Both are light enough for curves on top to
# stay clear in black-and-white print.
SAFE_FILL = "#cfe9da"    # 0.88 printed gray
UNSAFE_FILL = "#ecceca"  # 0.84 printed gray

# Dark text shade per role (printed gray <= 0.40), so labels drawn in a role
# colour stay legible in black-and-white print. Lines and fills keep COLORS.
TEXT = {
    "truth": "#222222", "annotation": "#333333", "muted": "#555555",
    "unbiased": "#1f4e79", "chance": "#1f4e79",
    "violation": "#b3392e",
    "biased": "#2c774c", "safe": "#2c774c",
    "nominal": "#555555",
    "scenario": "#895218",
    "stochastic": "#694b91",
}


def style(role, every=0.12):
    """Line2D kwargs for a role.

    every: marker spacing as a fraction of the axes diagonal (continuous
    curves); None marks every data point (short series); False drops the marker.
    """
    kw = dict(LINE[role])
    if every is False:
        kw.pop("marker", None)
    elif "marker" in kw and every is not None:
        kw["markevery"] = every
    return kw


def gray(hex_color):
    """Printed gray value of a hex colour (0 = black, 1 = white)."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b

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
        "hatch.linewidth": 0.6,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
if __name__ == "__main__":
    # Roles drawn together must print as clearly different grays, no line may
    # be too light to see on paper, and every role needs its own pattern.
    roles = ["truth", "unbiased", "violation", "biased", "nominal", "scenario"]
    levels = sorted(gray(COLORS[r]) for r in roles)
    assert all(b - a >= 0.09 for a, b in zip(levels, levels[1:])), levels
    assert levels[-1] <= 0.70, levels
    assert len({(LINE[r]["ls"], LINE[r].get("marker")) for r in roles}) == len(roles)
    assert style("unbiased")["markevery"] == 0.12 and "marker" not in style("biased", every=False)
    assert gray(SAFE_FILL) >= 0.85 and gray(UNSAFE_FILL) >= 0.80
    assert all(gray(c) <= 0.40 for c in TEXT.values()), TEXT
    print("thesis_style ok:", [f"{v:.2f}" for v in levels])
