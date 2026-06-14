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

# --------------------------------------------------------------------------
# Shared style
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 9.5,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "axes.titlesize": 10,
    "legend.fontsize": 8.3,
    "savefig.bbox": "tight",
})

BLUE   = "#1f4e79"
RED    = "#b3392e"
GRAY   = "#7a7a7a"
GREEN  = "#2e7d4f"
ORANGE = "#d9842b"

# semi-opaque white backplate for any label that sits over plotted content
WBOX = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85)

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)


def finish(fig, name):
    """Save a figure as PDF (for LaTeX) and PNG (for quick preview)."""
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


def styled_legend(ax, **kw):
    """A legend with a white, borderless, high-z-order background."""
    leg = ax.legend(frameon=True, framealpha=0.9, edgecolor="none",
                    facecolor="white", **kw)
    leg.set_zorder(7)
    return leg


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

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # ---- panel (a): density and two vertical decision lines ----
    ax = axes[0]
    xs = np.linspace(60, 175, 600)
    kde = stats.gaussian_kde(xi)
    dens = kde(xs)
    ax.plot(xs, dens, color=BLUE, lw=1.7, zorder=4)
    ax.fill_between(xs, dens, where=xs > mean_xi, color=RED, alpha=0.25,
                    zorder=2)

    ax.axvline(mean_xi, color="k", lw=1.2, ls="--", zorder=3)
    ax.axvline(q95, color=GREEN, lw=1.6, zorder=3)

    ymax = dens.max()
    ax.set_ylim(0, ymax * 1.32)          # headroom so labels clear the curve

    # mean-value label: upper-left, arrow down to the dashed line
    ax.annotate("mean-value plan\n$x=\\bar{\\xi}$",
                xy=(mean_xi, ymax * 1.02), xytext=(100, ymax * 1.20),
                fontsize=8.3, ha="center", va="center", zorder=8, bbox=WBOX,
                )
    # 95%-reliable label: upper-right, arrow down to the green line
    ax.annotate("95% reliable plan\n$x=q_{0.95}$",
                xy=(q95, ymax * 0.78), xytext=(135, ymax * 1.0),
                fontsize=8.3, ha="center", va="center", color=GREEN,
                zorder=8, bbox=WBOX,
                )
    # shortfall label inside the shaded region, clear of both lines
    ax.text(117, ymax * 0.20, "shortfall" % (viol * 100),
            fontsize=8.0, color=RED, ha="center", va="center",
            zorder=8, bbox=WBOX)

    ax.set_xlabel("net demand $\\xi$ [MW]")
    ax.set_ylabel("density")
    ax.set_title("(a) Two plans against the demand density")

    # ---- panel (b): price of reliability ----
    ax = axes[1]
    levels = np.linspace(0.50, 0.999, 200)
    xq = np.quantile(xi, levels)
    ax.plot((1 - levels) * 100, xq, color=BLUE, lw=1.7, zorder=4)

    ax.scatter([50], [np.quantile(xi, 0.5)], color="k", s=22, zorder=6)
    ax.scatter([5], [q95], color=GREEN, s=22, zorder=6)

    ax.annotate("mean-value plan",
                xy=(50, np.quantile(xi, 0.5)), xytext=(50, 112),
                fontsize=8.3, ha="center", zorder=8, bbox=WBOX,
                arrowprops=dict(arrowstyle="->", lw=0.8, shrinkB=3))
    ax.annotate("$\\varepsilon = 0.05$",
                xy=(5, q95), xytext=(16, q95 - 1),
                fontsize=8.3, color=GREEN, ha="center", zorder=8, bbox=WBOX,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=GREEN,
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

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    bars = ax.bar(labels, vals, color=[BLUE, ORANGE, RED], alpha=0.85,
                  width=0.55, zorder=3)
    ax.axhline(5, color=GREEN, lw=1.4, ls="--", zorder=2)

    ax.set_ylim(0, max(vals) * 1.32)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.03,
                "%.0f%%" % v, ha="center", fontsize=9.5, zorder=6)
    # target label parked in clear space on the right, above the dashed line
    ax.text(2.32, 3.0, "target $\\varepsilon = 5\\%$", color=GREEN,
            fontsize=9.0, ha="center", va="bottom", zorder=6, bbox=WBOX)

    ax.set_ylabel("violation frequency of\nmean-value solution [%]")
    fig.tight_layout()
    finish(fig, "fig_violations")


# ==========================================================================
# Figure 3  --  fig_spectrum
# The reliability spectrum: cost vs reliability, three regimes.
# ==========================================================================
def fig_spectrum():
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    rel = np.linspace(0.50, 0.99995, 500)
    cost = 100 + 35 * stats.norm.ppf(rel)
    ax.plot(rel * 100, cost, color=BLUE, lw=2.0, zorder=4)

    det_y = 100.0
    rob_y = 100 + 35 * stats.norm.ppf(0.99995)

    # shaded operating band
    ax.axvspan(90, 99.9, color=GREEN, alpha=0.10, zorder=1)

    ax.scatter([50], [det_y], color=RED, s=38, zorder=6)
    ax.scatter([99.995], [rob_y], color=GRAY, s=38, zorder=6)

    ax.set_ylim(85, rob_y * 1.05)
    ax.set_xlim(48, 101.5)

    # deterministic label: lower-middle, arrow to the red dot
    ax.annotate("deterministic (mean value):\ncheap, no guarantee",
                xy=(50, det_y), xytext=(58, 120),
                fontsize=8.7, color=RED, ha="center", zorder=8, bbox=WBOX,
                arrowprops=dict(arrowstyle="->", lw=0.9, color=RED,
                                shrinkB=4))
    # worst-case label: upper-left of the dot, arrow pointing right to it,
    # so the arrow never overlaps the text
    ax.annotate("worst-case robust:\nguaranteed, expensive",
                xy=(100, rob_y), xytext=(82, rob_y * 1),
                fontsize=8.7, color=GRAY, ha="center", va="center",
                zorder=8, bbox=WBOX,
                arrowprops=dict(arrowstyle="->", lw=0.9, color=GRAY,
                                shrinkB=4))
    # operating-range label inside the shaded band, low so it clears the curve
    ax.text(94.95, 112, "chance-constrained\noperating range\n"
            "$1-\\varepsilon \\in [0.90,\\ 0.999]$",
            ha="center", va="center", fontsize=6.4, color=GREEN,
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

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    # ---- panel (a): union of half-spaces ----
    ax = axes[0]
    ax.contourf(X, Y, feas, levels=[0.5, 1.5], colors=[BLUE], alpha=0.18,
                zorder=1)
    ax.contourf(X, Y, both, levels=[0.5, 1.5], colors=[BLUE], alpha=0.18,
                zorder=1)
    ax.plot(g, (1 - a[0] * g) / a[1], color=RED, lw=1.4, zorder=3,
            label="$\\xi^{(1)\\top}x=1$")
    ax.plot(g, (1 - b[0] * g) / b[1], color=ORANGE, lw=1.4, zorder=3,
            label="$\\xi^{(2)\\top}x=1$")

    p1 = np.array([0.05, 2.4])
    p2 = np.array([2.4, 0.05])
    mid = 0.5 * (p1 + p2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", lw=1.0, zorder=3)
    ax.scatter(*p1, color="k", s=22, zorder=6)
    ax.scatter(*p2, color="k", s=22, zorder=6)
    ax.scatter(*mid, facecolor="white", edgecolor="k", s=32, zorder=6)

    ax.text(-0.1, 2.62, "feasible", fontsize=8.3, zorder=8, bbox=WBOX)
    ax.text(2.15, 0.30, "feasible", fontsize=8.3, zorder=8, bbox=WBOX)
    ax.annotate("midpoint\ninfeasible", xy=mid, xytext=(1.95, 1.75),
                fontsize=8.3, ha="center", zorder=8, bbox=WBOX,
                arrowprops=dict(arrowstyle="->", lw=0.8, shrinkB=5))

    ax.set_xlim(-0.3, 3.0)
    ax.set_ylim(-0.3, 3.0)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("(a) Feasible set is a union of half-spaces")
    styled_legend(ax, loc="upper right", fontsize=8.0)

    # ---- panel (b): indicator vs smooth surrogates ----
    ax = axes[1]
    y = np.linspace(-3, 3, 600)
    ax.step(y, (y <= 0).astype(float), where="post", color=RED, lw=1.7,
            zorder=4, label="indicator $\\mathbf{1}\\{y\\leq 0\\}$")
    for h, c in [(0.6, BLUE), (1.2, GREEN)]:
        ax.plot(y, stats.norm.cdf(-y / h), color=c, lw=1.6, zorder=3,
                label="smooth surrogate, $h=%.1f$" % h)
    ax.set_xlabel("constraint residual $y$")
    ax.set_ylabel("contribution to $P(Y\\leq 0)$")
    ax.set_title("(b) The discontinuity at the heart of the problem")
    styled_legend(ax, loc="upper right", fontsize=6.0)

    fig.tight_layout()
    finish(fig, "fig_hardness")


# ==========================================================================
# Figure 5  --  fig_mismatch
# (a) bimodal truth vs Gaussian fit; (b) skewed tail underestimate (log y).
# ==========================================================================
def fig_mismatch():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # ---- panel (a): bimodal shape mismatch ----
    m = np.concatenate([rng.normal(-1.6, 0.45, 60_000),
                        rng.normal(1.1, 0.7, 40_000)])
    xs = np.linspace(-3.6, 3.6, 700)
    kdem = stats.gaussian_kde(m)
    gfit = stats.norm(m.mean(), m.std())
    ax = axes[0]
    ax.plot(xs, kdem(xs), color=BLUE, lw=1.8, zorder=4,
            label="true law (bimodal)")
    ax.plot(xs, gfit.pdf(xs), color=RED, lw=1.6, ls="--", zorder=3,
            label="fitted Gaussian")
    ax.set_xlabel("disturbance value")
    ax.set_ylabel("density")
    ax.set_title("(a) Shape mismatch")
    styled_legend(ax, loc="upper right", fontsize=8.0)

    # ---- panel (b): skewed law, Gaussian tail underestimate (log scale) ----
    gdist = stats.gamma(2.0)
    samp = gdist.rvs(400_000, random_state=3) - 2.0
    gf = stats.norm(samp.mean(), samp.std())
    thr = 4.0
    p_true = (samp > thr).mean()
    p_g = 1 - gf.cdf(thr)
    xs2 = np.linspace(-3, 8, 800)

    ax = axes[1]
    ax.semilogy(xs2, gdist.pdf(xs2 + 2.0), color=BLUE, lw=1.8, zorder=4,
                label="true law (skewed)")
    ax.semilogy(xs2, gf.pdf(xs2), color=RED, lw=1.6, ls="--", zorder=3,
                label="fitted Gaussian")
    ax.axvline(thr, color="k", lw=1.0, zorder=2)
    ax.fill_between(xs2, gdist.pdf(xs2 + 2.0), 1e-7, where=xs2 > thr,
                    color=BLUE, alpha=0.25, zorder=1)
    ax.set_ylim(1e-6, 1.2)

    # two stacked labels parked low and just right of the threshold line,
    # well clear of the legend in the upper-right corner
    ax.text(5, 3e-3, "true tail: %.1f%%" % (p_true * 100),
            fontsize=8.3, color=BLUE, ha="left", zorder=8, bbox=WBOX)
    ax.text(4.35, 6e-5, "Gaussian: %.2f%%" % (p_g * 100),
            fontsize=8.3, color=RED, ha="left", zorder=8, bbox=WBOX)
    ax.set_xlabel("forecast error")
    ax.set_ylabel("density (log scale)")
    ax.set_title("(b) Tail mass underestimated ~%.0fx" % (p_true / p_g))
    styled_legend(ax, loc="upper right", fontsize=8.0)

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

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.hist(ximix, bins=34, density=True, color=GRAY, alpha=0.30, zorder=1,
            label="residual samples $y_j=g(x,\\xi_j)$, $N=%d$" % N)
    ax.plot(ynodes, dens, color=BLUE, lw=1.9, zorder=4,
            label="KDE $\\hat{\\rho}_{Y(x)}$")
    ax.fill_between(ynodes, dens, where=ynodes <= 0, color=GREEN, alpha=0.28,
                    zorder=2)
    ax.axvline(0, color="k", lw=1.1, zorder=3)

    ax.set_ylim(0, dens.max() * 1.28)

    # probability label parked top-left, clear of the curve
    ax.text(-2.75, dens.max() * 1.15,
            "$\\hat{P}(Y(x)\\leq 0)=\\int_{-\\infty}^{0}\\hat{\\rho}\\,dy"
            "\\approx %.2f$" % psafe,
            fontsize=8.0, color=GREEN, ha="left", va="center",
            zorder=8, bbox=WBOX)
    ax.text(-1.0, 0.06, "safe", fontsize=9.0, color=GREEN, ha="center",
            zorder=8, bbox=WBOX)
    ax.text(0.95, 0.06, "unsafe", fontsize=9.0, color=RED, ha="center",
            zorder=8, bbox=WBOX)

    ax.set_xlabel("constraint residual $y$")
    ax.set_ylabel("density")
    styled_legend(ax, loc="upper right", fontsize=8.0)
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