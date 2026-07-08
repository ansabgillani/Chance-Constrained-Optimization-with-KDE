#!/usr/bin/env python3
"""
make_chapter3_figures.py
========================
Generates the three figures for Chapter 3 (the unbiased-KDE chapter) of the thesis
"Chance-Constrained Optimization with Kernel Density Estimation".

Usage (from any directory):
    pip install numpy scipy matplotlib
    python make_chapter3_figures.py

Outputs five PDFs and matching PNG previews in ./figures/.

Figures produced
----------------
fig_pipeline       -- KDE reformulation computational loop
fig_smoothing      -- KDE smoothed constraint and its derivative
fig_optimism       -- unbiased KDE certificate failure rate
(fig_staircase and fig_scenariogrowth moved to the Chapter 2 script)

House style
-----------
- Serif fonts, no top/right spines, dpi=200
- All in-axes labels use a semi-opaque white backplate so that curves
  and lines never strike through text
- Annotations point at features from clear space with short arrows
- LaTeX is NOT required
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")        # remove this line for an interactive window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ------------------------------------------------------------------ style
plt.rcParams.update({
    "font.size": 9.5,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "axes.titlesize": 10,
    "legend.fontsize": 8.3,
})

BLUE   = "#1f4e79"
RED    = "#b3392e"
GRAY   = "#7a7a7a"
GREEN  = "#2e7d4f"
ORANGE = "#d9842b"
PURPLE = "#6a4c93"

WBOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.88)

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Fixed seed for reproducibility
rng = np.random.default_rng(21)

# Shared demand law used across all five figures
LAW  = stats.lognorm(s=0.18, scale=100)
EPS  = 0.05
TRUE_Q = LAW.ppf(1 - EPS)


def finish(fig, name):
    """Save PDF and PNG preview then close."""
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def styled_legend(ax, **kw):
    """Framed legend with white background at z-order 7."""
    leg = ax.legend(frameon=True, framealpha=0.9, edgecolor="none",
                    facecolor="white", **kw)
    leg.set_zorder(7)
    return leg


# ===========================================================================
# Figure 1  fig_staircase  (Chapter 3)
# Left panel: empirical indicator is a staircase.
# Right panel: three sample draws near the feasibility boundary.
# ===========================================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def pbox(x, y, w, h, color, lines, fs=8.0):
        r = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12",
            fc=color, ec=color, alpha=0.18, lw=1.0)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, lines,
                ha="center", va="center", fontsize=fs, linespacing=1.25)

    yb, hb = 3.0, 2.0
    pbox(0.2,  yb, 2.4, hb, BLUE,
         "samples\n$\\xi_1,\\dots,\\xi_N$")
    pbox(3.2,  yb, 2.6, hb, BLUE,
         "residuals\n$y_j(x)=g(x,\\xi_j)$")
    pbox(6.4,  yb, 2.5, hb, GREEN,
         "KDE\n$\\hat\\rho_h(\\,\\cdot\\,;x)$")
    pbox(9.5,  yb, 4.2, hb, GREEN,
         "smooth constraint\n$\\hat p_h(x)\\geq 1-\\varepsilon$")
    pbox(5.0, 0.2, 4.2, 1.6, ORANGE,
         "NLP solver updates $x$\n(uses $\\nabla_x \\hat p_h$)", fs=8.0)

    arr = dict(arrowstyle="->", lw=1.2, color="k")
    for x1, x2 in [(2.6, 3.2), (5.8, 6.4), (8.9, 9.5)]:
        ax.annotate("", xy=(x2, yb + hb/2), xytext=(x1, yb + hb/2),
                    arrowprops=arr)

    # feedback arrows
    ax.annotate("", xy=(9.2, 1.0), xytext=(11.6, 3.0),
                arrowprops=dict(arrowstyle="->", lw=1.1, color=ORANGE,
                                connectionstyle="arc3,rad=-0.25"))
    ax.annotate("", xy=(4.5, 3.0), xytext=(5.0, 1.0),
                arrowprops=dict(arrowstyle="->", lw=1.1, color=ORANGE,
                                connectionstyle="arc3,rad=-0.25"))

    ax.text(11.9, 1.45, "value and\ngradient",
            fontsize=7.4, color=ORANGE, ha="center", bbox=WBOX, zorder=7)
    ax.text(3.65, 1.45, "new iterate\n$x^{(k+1)}$",
            fontsize=7.4, color=ORANGE, ha="center", bbox=WBOX, zorder=7)

    fig.tight_layout()
    finish(fig, "fig_pipeline")


# ===========================================================================
# Figure 4  fig_smoothing  (Chapter 4)
# Left: KDE smoothed constraint vs staircase.
# Right: its derivative vs the staircase's undefined derivative.
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

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # panel (a): constraint functions
    ax = axes[0]
    ax.step(xs, p_emp, where="post", color=GRAY, lw=1.0, alpha=0.8,
            label="empirical (staircase)")
    ax.plot(xs, p_true, color="k", ls="--", lw=1.3, label="true $p(x)$")
    for h, c in [(2.0, RED), (h_silv, BLUE), (15.0, ORANGE)]:
        lab = (f"KDE, $h={h:.0f}$" if h != h_silv
               else f"KDE, $h={h_silv:.1f}$ (Silverman)")
        ax.plot(xs, p_kde(xs, h), color=c, lw=1.5, label=lab)
    ax.axhline(1 - EPS, color=GREEN, ls=":", lw=1.2)
    ax.text(96.5, 0.905, "$1-\\varepsilon$",
            color=GREEN, fontsize=8.5, bbox=WBOX, zorder=7)
    ax.set_xlabel("decision $x$ [MW]")
    ax.set_ylabel("$\\hat p_h(x)$")
    ax.set_ylim(0.35, 1.03)
    ax.set_title("(a) The constraint function")
    styled_legend(ax, loc="lower right", fontsize=7.0)

    # panel (b): derivatives
    ax = axes[1]
    # staircase derivative: zero everywhere (marked by tick lines at samples)
    ax.axhline(0.0, color=GRAY, lw=1.6, zorder=3)
    ax.plot(samp, np.zeros_like(samp), linestyle="none", marker="|",
            ms=7, color=GRAY, alpha=0.7, zorder=4)
    for h, c in [(2.0, RED), (h_silv, BLUE), (15.0, ORANGE)]:
        ax.plot(xs, dp_kde(xs, h), color=c, lw=1.5, zorder=5)
    ax.set_xlabel("decision $x$ [MW]")
    ax.set_ylabel("$d\\hat p_h/dx$")
    ax.set_title("(b) Its derivative")
    ax.set_ylim(-0.006, 0.040)
    ax.set_xlim(95, 165)
    ax.text(148.5, 0.0265,
            "staircase derivative:\nzero between samples,\n"
            "undefined at them (ticks)",
            fontsize=7.2, color=GRAY, ha="center", bbox=WBOX, zorder=7)
    ax.annotate("", xy=(146, 0.0012), xytext=(148.5, 0.021),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=GRAY))

    fig.tight_layout()
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

    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    bins = np.linspace(0.88, 1.0, 49)
    ax.hist(rels[rels >= 1 - EPS], bins=bins, color=GREEN, alpha=0.55,
            label="meets target")
    ax.hist(rels[rels < 1 - EPS],  bins=bins, color=RED,   alpha=0.55,
            label=f"violates target ({frac_below*100:.0f}% of trials)")
    ax.axvline(1 - EPS, color="k", ls="--", lw=1.2)

    # label placed in upper region, clear of both histogram bars
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 200
    ax.text(0.9505, ymax * 0.90, "target $0.95$",
            fontsize=8.5, bbox=WBOX, zorder=7)

    ax.set_xlabel("true reliability of the KDE solution ($N=100$)")
    ax.set_ylabel("trials")
    styled_legend(ax, loc="upper left", fontsize=8.2)

    fig.tight_layout()
    finish(fig, "fig_optimism")
    print(f"    fraction below target: {frac_below*100:.1f}%")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating Chapter 3 figures ...")
    fig_pipeline()
    fig_smoothing()
    fig_optimism()         # ~20 s for 3000 trials
    print(f"\nAll figures written to ./{OUT}/ (PDF + PNG).")