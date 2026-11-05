#!/usr/bin/env python3
"""
make_chapter1_figures.py
========================
Regenerates all six figures for Chapter 1 of the thesis
"Chance-Constrained Optimization with Kernel Density Estimation".

This is a standalone script. From a terminal:

    pip install numpy scipy matplotlib
    python make_chapter1_figures.py

It writes six PDFs (and matching PNG previews) into ./figures/.

Every figure uses a shared house style:
  - serif fonts, no top/right spines
  - all in-axes text labels sit on a semi-opaque white "backplate" so that
    grid lines, curves, and arrows never strike through text
  - annotations use short arrows that start away from the text and point at
    the feature, so no line ever crosses a label
  - LaTeX is NOT required; percent signs and math are written so they render
    with matplotlib's default mathtext

The numbers are produced by Monte Carlo / direct computation, so the figures
are fully reproducible (a fixed RNG seed is used throughout).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # safe default; remove if you want interactive
import matplotlib.pyplot as plt
from scipy import stats

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thesis_style import apply_style, COLORS, HATCH_UNSAFE, LINE, SAFE_FILL, UNSAFE_FILL, WBOX, TEXT, style, width_in
from matplotlib.colors import to_rgba
apply_style()

BLUE, RED, GRAY = COLORS["unbiased"], COLORS["violation"], COLORS["nominal"]
GREEN, ORANGE, TRUTH = COLORS["safe"], COLORS["scenario"], COLORS["truth"]
ANN = COLORS["annotation"]

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)


def finish(fig, name):
    """Save a figure as PDF (for LaTeX) and PNG (for quick preview)."""
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


# ==========================================================================
# Figure 1  --  fig_meanvalue
# The mean-value fallacy: density with two decisions, and the price curve.
# ==========================================================================
def fig_meanvalue():
    mu_d, sg = np.log(100), 0.18
    xi = rng.lognormal(mu_d, sg, 200_000)
    mean_xi = xi.mean()
    viol = (xi > mean_xi).mean()
    q95 = np.quantile(xi, 0.95)

    fig, axes = plt.subplots(1, 2, figsize=(width_in(1.0), 2.8))

    # ---- panel (a): density and two vertical decision lines ----
    ax = axes[0]
    xs = np.linspace(60, 190, 600)
    kde = stats.gaussian_kde(xi)
    dens = kde(xs)
    ax.plot(xs, dens, lw=1.7, zorder=4, **style("truth"))
    ax.fill_between(xs, dens, where=xs > mean_xi, facecolor=UNSAFE_FILL,
                    edgecolor=RED, hatch=HATCH_UNSAFE, lw=0, zorder=2)

    ax.axvline(mean_xi, ymax=0.64, lw=1.4, zorder=3, **style("nominal", every=False))
    ax.axvline(q95, ymax=0.64, lw=1.6, zorder=3, **style("unbiased", every=False))

    ymax = dens.max()
    ax.set_ylim(0, ymax * 1.75)          # headroom so labels clear the curve

    # mean-value label: upper-left, arrow down to the dashed line
    ax.annotate("mean-value\nplan\n$x=\\bar{\\xi}$",
                xy=(mean_xi, ymax * 1.02), xytext=(mean_xi + 3, ymax * 1.45),
                ha="right", va="center", color=TEXT["nominal"], zorder=8)
    # 95%-reliable label: upper-right, arrow down to the green line
    ax.annotate("95% reliable\nplan\n$x=q_{0.95}$",
                xy=(q95, ymax * 0.78), xytext=(q95 - 3, ymax * 1.45),
                ha="left", va="center", color=TEXT["unbiased"], zorder=8)
    # shortfall label inside the shaded region, clear of both lines
    ax.text(166, ymax * 0.50, f"shortfall\n{viol * 100:.1f}%",
            color=TEXT["violation"], ha="center", va="center", zorder=8)

    ax.set_xlabel("net demand $\\xi$ [MW]")
    ax.set_ylabel("density")
    ax.set_title("(a) Two plans vs. demand density")

    # ---- panel (b): price of reliability ----
    ax = axes[1]
    levels = np.linspace(0.50, 0.999, 200)
    xq = np.quantile(xi, levels)
    ax.plot((1 - levels) * 100, xq, lw=1.7, zorder=4, **style("truth"))

    ax.scatter([viol * 100], [mean_xi], color=GRAY, s=22, zorder=6,
               marker=LINE["nominal"]["marker"])
    ax.scatter([5], [q95], color=BLUE, s=22, zorder=6,
               marker=LINE["unbiased"]["marker"])

    ax.annotate("mean-value plan",
                xy=(viol * 100, mean_xi), xytext=(62, 122),
                ha="center", color=TEXT["nominal"], zorder=8,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=GRAY, shrinkB=3))
    ax.annotate("$\\varepsilon = 0.05$",
                xy=(5, q95), xytext=(16, q95 - 1),
                color=TEXT["unbiased"], ha="center", zorder=8,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=BLUE,
                                shrinkB=3))
    ax.set_xlabel("accepted shortfall probability $\\varepsilon$ [%]")
    ax.set_ylabel("required capacity [MW]")
    ax.set_title("(b) Price of reliability")
    ax.invert_xaxis()

    fig.tight_layout()
    finish(fig, "fig_meanvalue")


# ==========================================================================
# Figure 2  --  fig_violations
# Violation frequency of the mean-value solution in three stylized problems.
# ==========================================================================
def fig_violations():
    # energy dispatch: skewed (lognormal) net demand
    xi = rng.lognormal(np.log(100), 0.18, 200_000)
    v1 = (xi > xi.mean()).mean()
    # landing corridor: bimodal thrust disturbance
    land = np.concatenate([rng.normal(-0.5, 0.3, 100_000),
                           rng.normal(0.6, 0.35, 100_000)])
    v2 = (land > land.mean()).mean()
    # portfolio loss cap: heavy-tailed (Student-t) returns
    loss = -(stats.t(df=3).rvs(200_000, random_state=1) * 0.01 + 0.0004)
    v3 = (loss > loss.mean()).mean()

    labels = ["energy dispatch\n(skewed demand)",
              "landing corridor\n(bimodal thrust)",
              "portfolio loss cap\n(heavy-tailed returns)"]
    vals = [v1 * 100, v2 * 100, v3 * 100]

    fig, ax = plt.subplots(figsize=(width_in(0.95), 3.0))
    bars = ax.bar(labels, vals, color=[BLUE, ORANGE, RED], alpha=0.85,
                  width=0.55, zorder=3)
    ax.axhline(5, color=TRUTH, lw=1.2, ls="--", zorder=2)

    ax.set_ylim(0, max(vals) * 1.32)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.03,
                "%.0f%%" % v, ha="center", zorder=6)
    # target label parked in clear space on the right, above the dashed line
    ax.text(2.32, 3.0, "target $\\varepsilon = 5\\%$", color=ANN,
            ha="center", va="bottom", zorder=6, bbox=WBOX)

    ax.set_ylabel("violation frequency of\nmean-value solution [%]")
    fig.tight_layout()
    finish(fig, "fig_violations")


# ==========================================================================
# Figure 3  --  fig_spectrum
# The reliability spectrum: cost vs reliability, three regimes.
# ==========================================================================
def fig_spectrum():
    fig, ax = plt.subplots(figsize=(width_in(0.82), 3.0))
    rel = np.linspace(0.50, 0.99995, 500)
    cost = 100 + 35 * stats.norm.ppf(rel)
    ax.plot(rel * 100, cost, lw=1.8, zorder=4, **style("truth"))

    det_y = 100.0
    rob_y = 100 + 35 * stats.norm.ppf(0.99995)

    # shaded operating band
    ax.axvspan(90, 99.9, facecolor=to_rgba(BLUE, 0.10), edgecolor=BLUE,
               lw=0.6, zorder=1)

    ax.scatter([50], [det_y], color=GRAY, s=38, zorder=6,
               marker=LINE["nominal"]["marker"])
    ax.scatter([99.995], [rob_y], color=ORANGE, s=38, zorder=6,
               marker=LINE["scenario"]["marker"], ec=TEXT["scenario"], lw=0.8)

    ax.set_ylim(85, rob_y * 1.05)
    ax.set_xlim(48, 103)

    # deterministic label: lower-middle, arrow to the red dot
    ax.annotate("deterministic (mean value):\ncheap, no guarantee",
                xy=(50, det_y), xytext=(49.5, 123),
                color=TEXT["nominal"], ha="left", va="bottom", zorder=8,
                arrowprops=dict(arrowstyle="->", lw=0.9, color=GRAY,
                                shrinkB=4))
    # bounded-set robust proxy: upper-left of the dot, arrow pointing right to it,
    # so the arrow never overlaps the text
    ax.annotate("bounded-set robust design:\nhigh-quantile proxy",
                xy=(100, rob_y), xytext=(82, rob_y * 1),
                color=TEXT["scenario"], ha="center", va="center",
                zorder=8, bbox=WBOX,
                arrowprops=dict(arrowstyle="->", lw=0.9, color=TEXT["scenario"],
                                shrinkB=4))
    # operating-range label inside the shaded band, low so it clears the curve
    ax.text(89.5, 104, "chance-constrained\noperating range\n"
            "$1-\\varepsilon \\in [0.90,\\ 0.999]$",
            ha="center", va="center", color=TEXT["unbiased"],
            zorder=8, bbox=WBOX)

    ax.set_xlabel("reliability level $1-\\varepsilon$ [%]")
    ax.set_ylabel("objective cost (lower is better)")
    fig.tight_layout()
    finish(fig, "fig_spectrum")


# ==========================================================================
# Figure 4  --  fig_hardness
# (a) non-convex feasible set; (b) indicator vs smooth surrogates.
# ==========================================================================
def fig_hardness():
    a = np.array([1.0, 0.15])
    b = np.array([0.15, 1.0])
    g = np.linspace(-0.5, 3.0, 500)
    X, Y = np.meshgrid(g, g)
    inA = (a[0] * X + a[1] * Y <= 1)
    inB = (b[0] * X + b[1] * Y <= 1)
    feas = (inA | inB).astype(float)
    both = (inA & inB).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(width_in(0.9), 2.75))

    # ---- panel (a): union of half-spaces ----
    ax = axes[0]
    ax.contourf(X, Y, feas, levels=[0.5, 1.5], colors=[SAFE_FILL],
                zorder=1)
    ax.contourf(X, Y, both, levels=[0.5, 1.5], colors=[COLORS["safe"]],
                alpha=0.25, zorder=1)
    ax.plot(g, (1 - a[0] * g) / a[1], lw=1.3, zorder=3, **style("truth"),
            label="$\\xi^{(1)\\top}x=1$")
    ax.plot(g, (1 - b[0] * g) / b[1], color=TRUTH, lw=1.3, ls=(0, (5, 2)), zorder=3,
            label="$\\xi^{(2)\\top}x=1$")

    p1 = np.array([0.05, 2.4])
    p2 = np.array([2.4, 0.05])
    mid = 0.5 * (p1 + p2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=TRUTH, ls="--", lw=1.0, zorder=3)
    ax.scatter(*p1, color=TRUTH, s=22, zorder=6)
    ax.scatter(*p2, color=TRUTH, s=22, zorder=6)
    ax.scatter(*mid, facecolor="white", edgecolor=TRUTH, s=32, zorder=6)

    ax.text(-0.1, 2.62, "feasible", color=ANN, zorder=8, bbox=WBOX)
    ax.text(2.15, 0.30, "feasible", color=ANN, zorder=8, bbox=WBOX)
    ax.annotate("midpoint\ninfeasible", xy=mid, xytext=(2.2, 1.3),
                ha="center", color=ANN, zorder=8, bbox=WBOX,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=ANN, shrinkB=5))

    ax.set_xlim(-0.3, 3.0)
    ax.set_ylim(-0.3, 3.0)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("(a) Union of half-spaces")
    ax.legend(loc="upper right")

    # ---- panel (b): indicator vs smooth surrogates ----
    ax = axes[1]
    y = np.linspace(-3, 3, 600)
    ax.step(y, (y <= 0).astype(float), where="post", lw=1.7,
            zorder=4, **style("truth"), label="indicator $\\mathbf{1}\\{y\\leq 0\\}$")
    for h, ls in [(0.6, "-"), (1.2, (0, (5, 2)))]:
        kw = style("unbiased") if h == 0.6 else dict(color=BLUE, ls=ls)
        ax.plot(y, stats.norm.cdf(-y / h), lw=1.6, zorder=3, **kw,
                label="smooth surrogate, $h=%.1f$" % h)
    ax.set_xlabel("constraint residual $y$")
    ax.set_ylabel("contribution to $P(Y\\leq 0)$")
    ax.set_title("(b) Indicator vs. smooth surrogates")
    ax.set_ylim(-0.05, 1.8)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper center")

    fig.tight_layout()
    finish(fig, "fig_hardness")


# ==========================================================================
# Figure 5  --  fig_mismatch
# (a) bimodal truth vs Gaussian fit; (b) skewed tail underestimate (log y).
# ==========================================================================
def fig_mismatch():
    fig, axes = plt.subplots(1, 2, figsize=(width_in(0.82), 2.5))

    # ---- panel (a): bimodal shape mismatch ----
    m = np.concatenate([rng.normal(-1.6, 0.45, 60_000),
                        rng.normal(1.1, 0.7, 40_000)])
    xs = np.linspace(-3.6, 3.6, 700)
    kdem = stats.gaussian_kde(m)
    gfit = stats.norm(m.mean(), m.std())
    ax = axes[0]
    ax.plot(xs, kdem(xs), lw=1.8, zorder=4, **style("unbiased"),
            label="KDE of bimodal sample")
    ax.plot(xs, gfit.pdf(xs), lw=1.8, zorder=3, **{**style("nominal"), "mec": TEXT["nominal"]},
            label="fitted Gaussian")
    ax.set_xlabel("disturbance value")
    ax.set_ylabel("density")
    ax.set_ylim(0, kdem(xs).max() * 1.65)
    ax.set_title("(a) Shape mismatch")
    ax.legend(loc="upper right")

    # ---- panel (b): skewed law, Gaussian tail underestimate (log scale) ----
    gdist = stats.gamma(2.0)
    samp = gdist.rvs(400_000, random_state=3) - 2.0
    gf = stats.norm(samp.mean(), samp.std())
    thr = 4.0
    p_true = (samp > thr).mean()
    p_g = 1 - gf.cdf(thr)
    xs2 = np.linspace(-3, 8, 800)

    ax = axes[1]
    ax.semilogy(xs2, gdist.pdf(xs2 + 2.0), lw=1.8, zorder=4, **style("truth"),
                label="true law (skewed)")
    ax.semilogy(xs2, gf.pdf(xs2), lw=1.8, zorder=3, **{**style("nominal"), "mec": TEXT["nominal"]},
                label="fitted Gaussian")
    ax.axvline(thr, ymax=0.66, lw=1.0, zorder=2, **style("truth"))
    ax.fill_between(xs2, gdist.pdf(xs2 + 2.0), 1e-7, where=xs2 > thr,
                    facecolor=UNSAFE_FILL, edgecolor=RED, hatch=HATCH_UNSAFE,
                    lw=0, zorder=1)
    ax.set_ylim(1e-6, 1e3)
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1])

    # two stacked labels parked low and just right of the threshold line,
    # well clear of the legend in the upper-right corner
    ax.text(3.65, 4e-4, "true tail:\n%.1f%%" % (p_true * 100),
            color=TRUTH, ha="right", zorder=8, bbox=WBOX)
    ax.text(3.65, 6e-6, "Gaussian:\n%.2f%%" % (p_g * 100),
            color=TEXT["annotation"], ha="right", zorder=8, bbox=WBOX)
    ax.set_xlabel("forecast error")
    ax.set_ylabel("density (log scale)")
    ax.set_title("(b) Tail underestimated ~%.0fx" % (p_true / p_g))
    ax.legend(loc="upper right")

    fig.tight_layout()
    finish(fig, "fig_mismatch")


# ==========================================================================
# Figure 6  --  fig_residual
# Residual-space construction: samples, KDE, shaded safe region.
# ==========================================================================
def fig_residual():
    N = 400
    ximix = np.concatenate([rng.normal(-0.9, 0.35, int(N * 0.65)),
                            rng.normal(0.7, 0.5, N - int(N * 0.65))])
    ynodes = np.linspace(-2.6, 2.6, 700)
    kdey = stats.gaussian_kde(ximix, bw_method=0.25)
    dens = kdey(ynodes)
    psafe = kdey.integrate_box_1d(-np.inf, 0.0)

    fig, ax = plt.subplots(figsize=(width_in(0.88), 2.9))
    ax.hist(ximix, bins=34, density=True, fill=False, edgecolor=GRAY, lw=0.6, zorder=1,
            label="residual samples $y_j=g(x,\\xi_j)$, $N=%d$" % N)
    ax.plot(ynodes, dens, lw=1.9, zorder=4, **style("unbiased"),
            label="KDE $\\hat{\\rho}_{Y(x)}$")
    ax.fill_between(ynodes, dens, where=ynodes <= 0, color=SAFE_FILL, zorder=2)
    ax.axvline(0, lw=1.1, zorder=3, **style("truth"))

    ax.set_ylim(0, dens.max() * 1.7)

    # probability label parked top-left, clear of the curve
    ax.text(-2.75, dens.max() * 1.52,
            "$\\hat{P}(Y(x)\\leq 0)=\\int_{-\\infty}^{0}\\hat{\\rho}\\,dy"
            "\\approx %.2f$" % psafe,
            color=TEXT["truth"], ha="left", va="center",
            zorder=8, bbox=WBOX)
    ax.text(-1.0, 0.06, "safe", color=TEXT["safe"], ha="center",
            zorder=8, bbox=WBOX)
    ax.text(0.95, 0.06, "unsafe", color=TEXT["violation"], ha="center",
            zorder=8, bbox=WBOX)

    ax.set_xlabel("constraint residual $y$")
    ax.set_ylabel("density")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01))
    fig.tight_layout()
    finish(fig, "fig_residual")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    fig_meanvalue()
    fig_violations()
    fig_spectrum()
    fig_hardness()
    fig_mismatch()
    fig_residual()
    print("\nAll six figures written to ./%s/ (PDF)." % OUT)
