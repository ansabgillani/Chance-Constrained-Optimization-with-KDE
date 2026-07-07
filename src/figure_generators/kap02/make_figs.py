#!/usr/bin/env python3
"""
make_chapter2_figures.py
========================
Generates all seven figures for Chapter 2 of the thesis
"Chance-Constrained Optimization with Kernel Density Estimation".

Usage (from any directory):
    pip install numpy scipy matplotlib
    python make_chapter2_figures.py

Outputs seven PDFs and matching PNG previews in ./figures/.

Figures produced
----------------
fig_taxonomy     -- four-paradigm conceptual map
fig_levelsets    -- Prekopa log-concavity and its boundary
fig_methods      -- survey experiment: achieved reliability vs N
fig_kernels      -- kernel functions and their integrated forms
fig_bandwidth    -- bias-variance trade-off for KDE
fig_curse        -- curse of dimensionality: samples vs dimension
fig_litmap       -- KDE chance-constraint literature timeline

House style
-----------
- Serif fonts, no top/right spines, dpi=200
- All in-axes labels use a semi-opaque white backplate (WBOX) so that
  grid lines, curves, and arrows never strike through text
- Annotations use short arrows pointing *at* the feature, starting in
  clear space, so no arrow crosses a label
- LaTeX is NOT required; mathtext handles all math symbols
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")        # remove this line if you want an interactive window
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

# White backplate: place this on every label that sits over plotted content
WBOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.88)

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Fixed seed for reproducibility
rng = np.random.default_rng(11)


def finish(fig, name):
    """Save PDF (for LaTeX) and PNG (for quick preview) and close."""
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    # fig.savefig(os.path.join(OUT, name + ".png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def styled_legend(ax, **kw):
    """Legend with opaque white background at high z-order."""
    leg = ax.legend(frameon=True, framealpha=0.9, edgecolor="none",
                    facecolor="white", **kw)
    leg.set_zorder(7)
    return leg


# ===========================================================================
# Figure 1  fig_taxonomy
# Four-paradigm conceptual map.
# ===========================================================================
def fig_taxonomy():
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.0, 10)
    ax.axis("off")

    # axis arrows
    ax.annotate("", xy=(9.8, 0.6), xytext=(0.4, 0.6),
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(0.4, 9.8), xytext=(0.4, 0.6),
                arrowprops=dict(arrowstyle="->", lw=1.2))

    ax.text(5.1, -0.75, "distributional information used",
            ha="center", fontsize=9)
    ax.text(-0.05, 9.7, "protection level (conservatism)",
            rotation=90, va="top", fontsize=9)
    for xp, lab in [(1.9, "none"), (5.0, "support set only"),
                    (8.25, "full law / samples")]:
        ax.text(xp, 0.05, lab, ha="center", fontsize=8, color=GRAY)

    def box(x, y, w, h, color, label, sub, tfrac=0.72, sfrac=0.26):
        r = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12",
            fc=color, ec="none", alpha=0.20)
        ax.add_patch(r)
        ax.text(x + w/2, y + h*tfrac, label, ha="center", va="center",
                fontsize=9.3, color=color, weight="bold", linespacing=1.05)
        ax.text(x + w/2, y + h*sfrac, sub, ha="center", va="center",
                fontsize=7.8, color="k", linespacing=1.1)

    box(0.8, 1.2, 2.3, 2.0, RED,    "Deterministic",
        "mean values,\nno guarantee")
    box(3.8, 7.2, 2.5, 2.0, GRAY,   "Robust",
        "all of $\\Xi$,\nworst case")
    box(7.0, 1.6, 2.6, 2.0, ORANGE, "Stochastic\nprogramming",
        "expected cost\n(+ recourse)", tfrac=0.76, sfrac=0.24)

    # chance-constrained: tall box with double-headed arrow on the left
    r = mpatches.FancyBboxPatch(
        (7.0, 4.4), 2.6, 4.4, boxstyle="round,pad=0.12",
        fc=GREEN, ec=GREEN, alpha=0.20, lw=1.0)
    ax.add_patch(r)
    ax.text(8.55, 8.05, "Chance-\nconstrained", ha="center", va="center",
            fontsize=9.3, color=GREEN, weight="bold", linespacing=1.05)
    ax.annotate("", xy=(7.45, 8.55), xytext=(7.45, 4.65),
                arrowprops=dict(arrowstyle="<->", lw=1.1, color=GREEN))
    ax.text(8.6, 6.0, "reliability dial\n$1-\\varepsilon$",
            ha="center", va="center", fontsize=7.8, linespacing=1.15)

    fig.tight_layout()
    finish(fig, "fig_taxonomy")


# ===========================================================================
# Figure 2  fig_levelsets
# Prekopa's theorem and the boundary of log-concavity.
# ===========================================================================
def fig_levelsets():
    g = np.linspace(-0.6, 1.6, 240)
    X1, X2 = np.meshgrid(g, g)

    def p_gauss(mu, Sig):
        s = (np.sqrt(Sig[0, 0]*X1**2 + 2*Sig[0, 1]*X1*X2
                     + Sig[1, 1]*X2**2) + 1e-12)
        return stats.norm.cdf((1 - (mu[0]*X1 + mu[1]*X2)) / s)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))

    # panel (a): Gaussian -> convex level set
    ax = axes[0]
    P = p_gauss(np.array([0.8, 0.8]), np.array([[0.16, 0.0], [0.0, 0.16]]))
    ax.contourf(X1, X2, P, levels=[0.9, 1.001], colors=[GREEN], alpha=0.30)
    ax.contour(X1, X2, P, levels=[0.9], colors=[GREEN], linewidths=1.6)
    ax.text(-0.30, 0.05, "$\\{x: p(x)\\geq 0.9\\}$",
            fontsize=9, color=GREEN, bbox=WBOX, zorder=7)
    ax.set_title("(a) Gaussian $\\xi$: level set is convex")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    # panel (b): mixture -> non-convex level set
    ax = axes[1]
    PA = p_gauss(np.array([1.9, 0.1]), np.array([[0.05, 0], [0, 0.05]]))
    PB = p_gauss(np.array([0.1, 1.9]), np.array([[0.05, 0], [0, 0.05]]))
    P2 = 0.5*PA + 0.5*PB
    ax.contourf(X1, X2, P2, levels=[0.55, 1.001], colors=[RED], alpha=0.25)
    ax.contour(X1, X2, P2, levels=[0.55], colors=[RED], linewidths=1.6)
    ax.text(-0.42, -0.30, "$\\{x: p(x)\\geq 0.55\\}$",
            fontsize=9, color=RED, bbox=WBOX, zorder=7)
    ax.set_title("(b) Mixture $\\xi$: level set is non-convex")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    fig.tight_layout()
    finish(fig, "fig_levelsets")


# ===========================================================================
# Figure 3  fig_methods
# Survey experiment: five methods, achieved reliability vs sample size.
# ===========================================================================
def fig_methods():
    law = stats.lognorm(s=0.18, scale=100)
    eps = 0.05
    Ns = [50, 100, 200, 500, 1000]
    trials = 400

    res = {m: [] for m in ["Gaussian", "Scenario", "SAA", "CVaR", "KDE"]}
    for N in Ns:
        rels = {m: [] for m in res}
        for _ in range(trials):
            samp = law.rvs(N, random_state=rng.integers(int(1e9)))
            xg = samp.mean() + stats.norm.ppf(1 - eps)*samp.std(ddof=1)
            xs = samp.max()
            xq = np.quantile(samp, 1 - eps)
            tail = np.sort(samp)[max(0, int(np.ceil((1 - eps)*N))):]
            xc = tail.mean() if len(tail) else samp.max()
            h = 0.9*min(samp.std(ddof=1), stats.iqr(samp)/1.34)*N**(-0.2)
            grid = np.linspace(samp.min() - 3*h, samp.max() + 3*h, 800)
            cdf = stats.norm.cdf(
                (grid[:, None] - samp[None, :]) / h).mean(axis=1)
            xk = np.interp(1 - eps, cdf, grid)
            for m, x in zip(res, [xg, xs, xq, xc, xk]):
                rels[m].append(law.cdf(x))
        for m in res:
            r = rels[m]
            res[m].append((np.mean(r),
                           np.percentile(r, 10),
                           np.percentile(r, 90)))

    colors = {"Gaussian": RED, "Scenario": GRAY, "SAA": ORANGE,
              "CVaR": PURPLE, "KDE": BLUE}
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for m in res:
        arr = np.array(res[m])
        ax.plot(Ns, arr[:, 0], "-o", ms=3.5, lw=1.5,
                color=colors[m], label=m)
        ax.fill_between(Ns, arr[:, 1], arr[:, 2],
                        color=colors[m], alpha=0.10)

    ax.axhline(1 - eps, color="k", ls="--", lw=1.1)
    ax.annotate("target $1-\\varepsilon = 0.95$",
                xy=(115, 1 - eps), xytext=(115, 0.9645),
                fontsize=8.5, ha="center", zorder=7, bbox=WBOX,
                arrowprops=dict(arrowstyle="->", lw=0.8, shrinkB=2))
    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels(Ns)
    ax.set_xlabel("sample size $N$")
    ax.set_ylabel("achieved reliability $\\mathbb{P}(\\xi \\leq x_N)$")
    styled_legend(ax, ncol=1, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), fontsize=8.3)
    fig.tight_layout()
    finish(fig, "fig_methods")


# ===========================================================================
# Figure 4  fig_kernels
# Kernel functions and their integrated forms vs the indicator.
# ===========================================================================
def fig_kernels():
    u = np.linspace(-2.2, 2.2, 600)
    shift = 0.55

    gauss  = stats.norm.pdf(u)
    epan   = np.where(np.abs(u) <= 1, 0.75*(1 - u**2), 0)
    epan_b = np.where(np.abs(u - shift) <= 1,
                      0.75*(1 - (u - shift)**2), 0)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # panel (a): kernel shapes
    ax = axes[0]
    ax.plot(u, gauss,  color=BLUE,  lw=1.7, label="Gaussian")
    ax.plot(u, epan,   color=GREEN, lw=1.7, label="Epanechnikov")
    ax.plot(u, epan_b, color=RED,   lw=1.7, ls="--",
            label="biased (shifted support)")
    ax.set_xlabel("$u$")
    ax.set_ylabel("$K(u)$")
    ax.set_title("(a) Kernel functions")
    ax.set_ylim(0, 0.97)
    styled_legend(ax, loc="upper left", fontsize=6.3)

    # panel (b): integrated kernels vs indicator
    ax = axes[1]
    cg  = stats.norm.cdf(u)
    ce  = np.where(u < -1, 0, np.where(u > 1, 1, 0.25*(2 + 3*u - u**3)))
    ub  = u - shift
    cb  = np.where(ub < -1, 0, np.where(ub > 1, 1,
                                         0.25*(2 + 3*ub - ub**3)))

    ax.step([-2.2, 0, 0, 2.2], [0, 0, 1, 1], where="post",
            color="k", lw=1.2, label="indicator $\\mathbf{1}\\{u\\geq 0\\}$")
    ax.plot(u, cg, color=BLUE,  lw=1.6, label="Gaussian")
    ax.plot(u, ce, color=GREEN, lw=1.6, label="Epanechnikov")
    ax.plot(u, cb, color=RED,   lw=1.6, ls="--", label="biased")
    ax.fill_between(u, cb, np.where(u >= 0, 1, 0),
                    where=(cb <= np.where(u >= 0, 1, 0)) & (u > -1.7),
                    color=RED, alpha=0.08)
    ax.set_xlabel("$u$")
    ax.set_ylabel("$\\mathcal{K}(u)$")
    ax.set_title("(b) Integrated kernels vs the indicator")
    styled_legend(ax, loc="lower right", fontsize=6.3)

    fig.tight_layout()
    finish(fig, "fig_kernels")


# ===========================================================================
# Figure 5  fig_bandwidth
# Bias-variance trade-off for kernel density estimation.
# ===========================================================================
def fig_bandwidth():
    def truth(y):
        return (0.65*stats.norm.pdf(y, -0.9, 0.35)
                + 0.35*stats.norm.pdf(y, 0.7, 0.5))

    def sample_truth(N, seed):
        r = np.random.default_rng(seed)
        n1 = r.binomial(N, 0.65)
        return np.concatenate([r.normal(-0.9, 0.35, n1),
                               r.normal(0.7, 0.5, N - n1)])

    ygrid = np.linspace(-2.8, 2.8, 500)
    N = 200
    samp = sample_truth(N, 5)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # panel (a): three bandwidths on one sample
    ax = axes[0]
    ax.plot(ygrid, truth(ygrid), "k--", lw=1.3, label="true density")
    for h, c, lab in [(0.05, RED,    "$h=0.05$ (undersmoothed)"),
                      (0.25, BLUE,   "$h=0.25$ (near optimal)"),
                      (0.90, ORANGE, "$h=0.90$ (oversmoothed)")]:
        kde = (stats.norm.pdf((ygrid[:, None] - samp[None, :])/h)
               .mean(axis=1) / h)
        ax.plot(ygrid, kde, color=c, lw=1.5, label=lab)
    ax.set_xlabel("$y$")
    ax.set_ylabel("density")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"(a) One sample ($N={N}$), three bandwidths")
    styled_legend(ax, loc="upper right", fontsize=6.3)

    # panel (b): MISE decomposition over 120 replications
    hs = np.geomspace(0.03, 1.4, 26)
    T  = 120
    f_true = truth(ygrid)
    est = np.zeros((T, len(hs), len(ygrid)))
    for t in range(T):
        st = sample_truth(N, 100 + t)
        for k, h in enumerate(hs):
            est[t, k] = (stats.norm.pdf(
                (ygrid[:, None] - st[None, :])/h).mean(axis=1) / h)

    mean_est = est.mean(axis=0)
    bias2 = np.trapezoid((mean_est - f_true[None, :])**2, ygrid, axis=1)
    var   = np.trapezoid(est.var(axis=0), ygrid, axis=1)
    mise  = bias2 + var
    hstar = hs[np.argmin(mise)]

    ax = axes[1]
    ax.loglog(hs, bias2, color=RED,  lw=1.6, label="squared bias")
    ax.loglog(hs, var,   color=BLUE, lw=1.6, label="variance")
    ax.loglog(hs, mise,  color="k",  lw=1.8, label="MISE")
    ax.axvline(hstar, color=GREEN, lw=1.2, ls="--")
    ax.text(hstar*1.12, 1.45e-4, "$h^{*}$",
            color=GREEN, fontsize=10, bbox=WBOX, zorder=7)
    ax.set_xlabel("bandwidth $h$")
    ax.set_ylabel("integrated error")
    ax.set_title(f"(b) Bias-variance trade-off ({T} replications)")
    styled_legend(ax, loc="center right", fontsize=6.3)

    fig.tight_layout()
    finish(fig, "fig_bandwidth")
    print(f"    h* = {hstar:.3f}")


# ===========================================================================
# Figure 6  fig_curse
# Curse of dimensionality: sample size required vs dimension.
# ===========================================================================
def fig_curse():
    d  = np.arange(1, 11)
    N1 = 100.0
    Nd = N1**((d + 4) / 5)          # AMISE rate: N ∝ N_1^{(d+4)/5}

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.semilogy(d, Nd, "-o", color=BLUE, ms=4, lw=1.6)
    ax.axhline(1e6, color=GRAY, lw=1.0, ls=":")
    ax.text(1.0, 1.7e6, "$10^6$ samples", color=GRAY, fontsize=8.5)

    for dd, label, xy_text in [
            (1,  "$d=1$: $N\\approx 10^{2}$", (0.7, 9e2)),
            (5,  "$d=5$: $N\\approx 10^{4}$", (4.15,  1.1e5)),
            (10, "$d=10$: $N\\approx 10^{6}$", (8.4,  1.5e4))]:
        ax.annotate(label,
                    xy=(dd, N1**((dd + 4)/5)),
                    xytext=xy_text,
                    fontsize=8.3, bbox=WBOX, zorder=7,
                    arrowprops=dict(arrowstyle="->", lw=0.7, shrinkB=3))

    ax.set_xlabel("dimension $d$ of the estimated variable")
    ax.set_ylabel("samples for fixed accuracy")
    ax.set_xticks(d)
    ax.set_ylim(50, 4e6)
    fig.tight_layout()
    finish(fig, "fig_curse")


# ===========================================================================
# Figure 7  fig_litmap
# KDE chance-constraint literature timeline.
# Includes Gugat & Schuster (2018) added from the council research briefing.
# ===========================================================================
def fig_litmap():
    lanes = {
        "Process systems": (6.4, ORANGE),
        "Aerospace":       (4.8, BLUE),
        "Energy/networks": (3.2, GREEN),
        "Finance":         (1.6, PURPLE),
        "Theory":          (0.0, RED),
    }

    # (year, lane, label, above_the_dot?)
    papers = [
        (2015.0, "Process systems", "Calfa et al.",             True),
        (2018.0, "Aerospace",       "Caillau et al.",           True),
        # Gugat & Schuster (2018): energy/networks, below to avoid Caillau
        (2018.5, "Energy/networks", "Gugat,\nSchuster",        False),
        (2017.8, "Energy/networks", "Ciftci",                   True),
        (2019.6, "Energy/networks", "Ciftci et al.",            False),
        (2020.0, "Aerospace",       "Keil et al.\n(biased KDE)",False),
        (2021.3, "Aerospace",       "Keil et al.\n(OCAM)",      True),
        (2022.0, "Aerospace",       "Keil, Rao\n(warm start)",  False),
        (2021.5, "Energy/networks", "Wu et al.",                True),
        (2022.4, "Energy/networks", "Schuster et al.",          False),
        (2022.7, "Finance",         "Liu et al.",               True),
        (2023.3, "Energy/networks", "Wu, Kargarian",            True),
        (2025.4, "Theory",          "Schuster (convergence)",   True),
        (2026.0, "Energy/networks", "Qi et al.",                True),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))

    for name, (y, c) in lanes.items():
        ax.axhline(y, color=c, lw=0.8, alpha=0.4, zorder=1)
        ax.text(2013.6, y + 0.18, name,
                fontsize=8.8, color=c, weight="bold",
                zorder=6, bbox=WBOX)

    # narrative arrow (drawn before labels so labels sit on top)
    ax.annotate("", xy=(2025.35, 0.0), xytext=(2021.30, 4.8),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=GRAY,
                                connectionstyle="arc3,rad=-0.22"),
                zorder=2)
    ax.text(2024.85, 2.35,
            "engineering practice\nfeeds convergence theory",
            fontsize=7.5, color=GRAY, ha="center", style="italic",
            zorder=6, bbox=WBOX)

    for yr, lane, lab, above in papers:
        y, c = lanes[lane]
        ax.scatter(yr, y, s=46, color=c, zorder=7, ec="white", lw=0.4)
        off = 0.36 if above else -0.36
        va  = "bottom" if above else "top"
        ax.text(yr, y + off, lab, ha="center", fontsize=7.0, va=va,
                color="k", linespacing=1.1, zorder=6, bbox=WBOX)

    ax.set_xlim(2013.0, 2027.4)
    ax.set_ylim(-1.3, 7.6)
    ax.set_yticks([])
    ax.set_xlabel("year")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    finish(fig, "fig_litmap")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating Chapter 2 figures ...")
    fig_taxonomy()
    fig_levelsets()
    fig_methods()      # ~30 s for 400 trials × 5 sample sizes
    fig_kernels()
    fig_bandwidth()    # ~20 s for 120 replications
    fig_curse()
    fig_litmap()
    print(f"\nAll figures written to ./{OUT}/ (PDF + PNG).")