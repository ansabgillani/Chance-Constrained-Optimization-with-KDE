#!/usr/bin/env python3
"""
make_chapter4_figures.py
========================
Generates the three figures for Chapter 4 (the unbiased-KDE chapter) of the thesis
"Chance-Constrained Optimization with Kernel Density Estimation".

Usage (from any directory):
    pip install numpy scipy matplotlib
    python make_chapter4_figures.py

Outputs five PDFs and matching PNG previews in ./figures/.

Figures produced
----------------
fig_pipeline       -- KDE reformulation computational loop
fig_smoothing      -- KDE smoothed constraint and its derivative (standalone lognormal scalar law)
fig_optimism       -- unbiased KDE certificate failure rate (standalone lognormal scalar law)
(fig_staircase and fig_scenariogrowth moved to the Chapter 2 survey script)

House style
-----------
- Serif fonts, no top/right spines, dpi=200
- All in-axes labels use a semi-opaque white backplate so that curves
  and lines never strike through text
- Annotations point at features from clear space with short arrows
- LaTeX is NOT required
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")        # remove this line for an interactive window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thesis_style import apply_style, COLORS, WBOX, width_in

apply_style()
BLUE, GREEN, RED = COLORS["unbiased"], COLORS["biased"], COLORS["violation"]
GRAY, TRUTH, ARROW = COLORS["nominal"], COLORS["truth"], COLORS["muted"]


def tint(c, a=0.14):
    return tuple(1 - a + a * v for v in to_rgb(c))


OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Fixed seed for reproducibility
rng = np.random.default_rng(21)

# Standalone scalar lognormal law used by diagnostic figures; this is not
# the dispatch experiment ledger.
LAW  = stats.lognorm(s=0.18, scale=100)
EPS  = 0.05
TRUE_Q = LAW.ppf(1 - EPS)


def finish(fig, name):
    """Save PDF and PNG preview then close."""
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ===========================================================================
# Figure 1  fig_staircase  (Chapter 4)
# Left panel: empirical indicator is piecewise constant.
# Right panel: three sample draws near the feasibility boundary.
# ===========================================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(width_in(0.95), 2.2))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def pbox(x, y, w, h, color, title, body):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.25",
            fc=tint(color), ec=color, lw=0.8))
        ax.text(x + w/2, y + h*0.68, title, ha="center", va="center",
                fontsize=9, fontweight="bold", color=COLORS["annotation"])
        ax.text(x + w/2, y + h*0.30, body, ha="center", va="center",
                fontsize=8.5, color=COLORS["annotation"])

    yb, hb = 3.0, 2.2
    pbox(0.0, yb, 2.8, hb, GRAY, "samples", "$\\xi_1,\\dots,\\xi_N$")
    pbox(3.3, yb, 3.4, hb, GRAY, "residuals", "$y_j(x)=g(x,\\xi_j)$")
    pbox(7.2, yb, 2.6, hb, BLUE, "KDE", "$\\hat\\rho_h(\\,\\cdot\\,;x)$")
    pbox(10.3, yb, 3.7, hb, BLUE, "smooth constraint",
         "$\\hat p_h(x)\\geq 1-\\varepsilon$")
    pbox(4.9, 0.0, 4.6, 1.7, GRAY, "NLP solver",
         "updates $x$ (uses $\\nabla_x \\hat p_h$)")

    arr = dict(arrowstyle="->", lw=0.8, color=ARROW, shrinkA=0, shrinkB=0)
    for x1, x2 in [(2.8, 3.3), (6.7, 7.2), (9.8, 10.3)]:
        ax.annotate("", xy=(x2, yb + hb/2), xytext=(x1, yb + hb/2),
                    arrowprops=arr)

    # feedback arrows
    ax.annotate("", xy=(9.5, 0.85), xytext=(12.15, yb),
                arrowprops=dict(arr, connectionstyle="arc3,rad=-0.25"))
    ax.annotate("", xy=(5.0, yb), xytext=(5.6, 1.7),
                arrowprops=dict(arr, connectionstyle="arc3,rad=-0.25"))

    ax.text(12.3, 1.0, "value and\ngradient", fontsize=8.5,
            color=COLORS["annotation"], ha="center", va="center")
    ax.text(2.7, 1.0, "new iterate\n$x^{(k+1)}$", fontsize=8.5,
            color=COLORS["annotation"], ha="center", va="center")

    finish(fig, "fig_pipeline")


# ===========================================================================
# Figure 4  fig_smoothing  (Chapter 4)
# Left: KDE smoothed constraint vs piecewise-constant empirical estimator.
# Right: its derivative vs the piecewise-constant estimator's undefined derivative.
# ===========================================================================
def fig_smoothing():
    N = 200
    samp = LAW.rvs(N, random_state=7)
    xs = np.linspace(95, 165, 900)
    p_true = LAW.cdf(xs)
    p_emp  = (samp[None, :] <= xs[:, None]).mean(axis=1)
    h_silv = 0.9*min(samp.std(ddof=1), stats.iqr(samp)/1.34)*N**(-0.2)

    def p_kde(x, h):
        return stats.norm.cdf((x[:, None] - samp[None, :]) / h).mean(axis=1)

    def dp_kde(x, h):
        return (stats.norm.pdf((x[:, None] - samp[None, :]) / h)
                .mean(axis=1) / h)

    KDES = [(2.0, tint(BLUE, 0.45)), (h_silv, BLUE), (15.0, tint(BLUE, 0.7))]
    fig, axes = plt.subplots(1, 2, figsize=(width_in(1.0), 2.75))

    # panel (a): constraint functions
    ax = axes[0]
    ax.step(xs, p_emp, where="post", color=GRAY, lw=1.0,
            label="empirical (piecewise constant)")
    ax.plot(xs, p_true, color=TRUTH, ls="--", lw=1.3, label="true $p(x)$")
    for h, c in KDES:
        lab = (f"KDE, $h={h:.0f}$" if h != h_silv
               else f"KDE, $h={h_silv:.1f}$ (Silverman)")
        ax.plot(xs, p_kde(xs, h), color=c, lw=1.5, label=lab)
    ax.axhline(1 - EPS, color=TRUTH, ls="--", lw=1.0)
    ax.text(152, 0.90, "$1-\\varepsilon$",
            color=COLORS["annotation"], fontsize=8.5, bbox=WBOX, zorder=7)
    ax.set_xlabel("decision $x$ [MW]")
    ax.set_ylabel("$\\hat p_h(x)$")
    ax.set_ylim(0.35, 1.03)
    ax.set_title("(a) The constraint function")
    h_, l_ = ax.get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False,
               columnspacing=1.2, handlelength=1.8)

    # panel (b): derivatives
    ax = axes[1]
    # piecewise-constant empirical estimator derivative: zero everywhere (marked by tick lines at samples)
    ax.axhline(0.0, color=GRAY, lw=1.6, zorder=3)
    ax.plot(samp, np.zeros_like(samp), linestyle="none", marker="|",
            ms=7, color=GRAY, zorder=4)
    for h, c in KDES:
        ax.plot(xs, dp_kde(xs, h), color=c, lw=1.5, zorder=5)
    ax.set_xlabel("decision $x$ [MW]")
    ax.set_ylabel("$d\\hat p_h/dx$")
    ax.set_title("(b) Its derivative")
    ax.set_ylim(-0.006, 0.040)
    ax.set_xlim(95, 165)
    ax.text(147, 0.0315,
            "piecewise-constant\nderivative: zero between\n"
            "samples, undefined at\nthem (ticks)",
            fontsize=8.5, color=COLORS["annotation"], ha="center", va="top", bbox=WBOX, zorder=7)

    fig.tight_layout(rect=(0, 0.16, 1, 1))
    finish(fig, "fig_smoothing")
    print(f"    h_Silverman = {h_silv:.2f}")


# ===========================================================================
# Figure 5  fig_optimism  (Chapter 4)
# True reliability of unbiased KDE certificates over 3000 trials.
# ===========================================================================
def fig_optimism():
    N = 100
    trials = 3000
    rels = []
    for _ in range(trials):
        st = LAW.rvs(N, random_state=rng.integers(int(1e9)))
        h  = 0.9*min(st.std(ddof=1), stats.iqr(st)/1.34)*N**(-0.2)
        grid = np.linspace(st.min() - 3*h, st.max() + 3*h, 700)
        cdf  = stats.norm.cdf(
            (grid[:, None] - st[None, :]) / h).mean(axis=1)
        xk = np.interp(1 - EPS, cdf, grid)
        rels.append(LAW.cdf(xk))
    rels = np.array(rels)
    frac_below = (rels < 1 - EPS).mean()

    fig, ax = plt.subplots(figsize=(width_in(0.82), 2.7))
    bins = np.linspace(0.88, 1.0, 49)
    ax.hist(rels[rels >= 1 - EPS], bins=bins, color=GREEN, alpha=0.75,
            label="meets target")
    ax.hist(rels[rels < 1 - EPS],  bins=bins, color=RED,   alpha=0.75,
            label=f"violates target ({frac_below*100:.0f}% of trials)")
    ax.axvline(1 - EPS, color=TRUTH, ls="--", lw=1.0)

    # label placed in upper region, clear of both histogram bars
    ax.set_ylim(0, 1.25 * ax.get_ylim()[1])
    ymax = ax.get_ylim()[1]
    ax.text(0.9515, ymax * 0.90, "target $0.95$",
            fontsize=8.5, color=COLORS["annotation"])

    ax.set_xlabel("true reliability of the KDE solution ($N=100$)")
    ax.set_ylabel("trials")
    ax.legend(loc="upper left")

    fig.tight_layout()
    finish(fig, "fig_optimism")
    print(f"    fraction below target: {frac_below*100:.1f}%")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating Chapter 4 figures ...")
    fig_pipeline()
    fig_smoothing()
    fig_optimism()         # ~20 s for 3000 trials
    print(f"\nAll figures written to ./{OUT}/ (PDF + PNG).")
