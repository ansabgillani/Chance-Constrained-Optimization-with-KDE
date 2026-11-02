#!/usr/bin/env python3
"""
Figures for Chapter 2 (survey) and Chapter 3 (KDE background).

Usage:  python make_figs.py   (writes PDFs to ./figures/)

fig_taxonomy, fig_levelsets, fig_staircase  -> kap02/figures
fig_kernels, fig_bandwidth, fig_curse, fig_litmap -> kap03/figures
fig_methods and fig_scenariogrowth are kept for regeneration but are not
part of the default run.

Style comes from thesis_style (Charter, open axes, unframed legends); every
figure is drawn at the width LaTeX includes it, so 1 pt here is 1 pt on paper.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thesis_style import apply_style, COLORS, WBOX, width_in
apply_style()

BLUE = COLORS["unbiased"]
GREEN = COLORS["biased"]
RED = COLORS["violation"]
GRAY = COLORS["nominal"]
ORANGE = COLORS["scenario"]
PURPLE = COLORS["stochastic"]
TRUTH = COLORS["truth"]
ANNOT = COLORS["annotation"]
TXTGRAY = COLORS["muted"]
BLUE_TINT = "#6f9cc6"      # lighter tint of the unbiased-KDE blue


def tint(c, f):
    """Blend colour c with white; f is the fraction of c kept."""
    import matplotlib.colors as mc
    r = np.array(mc.to_rgb(c))
    return tuple(1 - f * (1 - r))


OUT = "figures"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(11)


def finish(fig, name):
    """Save at the exact figure size (printed width = figsize width)."""
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches=None)
    plt.close(fig)
    print(f"  wrote {name}")


def styled_legend(ax, **kw):
    return ax.legend(**kw)


# ===========================================================================
# fig_taxonomy: four-paradigm conceptual map (schematic box style)
# ===========================================================================
def fig_taxonomy():
    fig, ax = plt.subplots(figsize=(width_in(0.85), 3.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.0, 10)
    ax.axis("off")
    arrow = dict(arrowstyle="->", lw=0.8, color=TXTGRAY)

    ax.annotate("", xy=(9.8, 0.6), xytext=(0.4, 0.6), arrowprops=arrow)
    ax.annotate("", xy=(0.4, 9.8), xytext=(0.4, 0.6), arrowprops=arrow)
    ax.text(5.1, -0.75, "distributional information used",
            ha="center", fontsize=9, color=ANNOT)
    ax.text(-0.05, 9.7, "protection level (conservatism)",
            rotation=90, va="top", fontsize=9, color=ANNOT)
    for xp, lab in [(1.9, "none"), (5.0, "support set only"),
                    (8.25, "full law / samples")]:
        ax.text(xp, 0.05, lab, ha="center", fontsize=8.5, color=TXTGRAY)

    def box(x, y, w, h, color, label, sub, tfrac=0.70, sfrac=0.26,
            tcolor=None):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.25",
            fc=tint(color, 0.14), ec=color, lw=0.8))
        ax.text(x + w/2, y + h*tfrac, label, ha="center", va="center",
                fontsize=9, color=tcolor or color, weight="bold",
                linespacing=1.05)
        if sub:
            ax.text(x + w/2, y + h*sfrac, sub, ha="center", va="center",
                    fontsize=8.5, color=ANNOT, linespacing=1.1)

    box(0.8, 1.2, 2.3, 2.0, GRAY, "Deterministic",
        "mean values,\nno guarantee", tcolor=TXTGRAY)
    box(3.8, 7.2, 2.5, 2.0, ORANGE, "Robust",
        "all of $\\Xi$,\nworst case")
    box(7.0, 1.6, 2.6, 2.0, PURPLE, "Stochastic\nprogramming",
        "expected cost\n(+ recourse)", tfrac=0.76, sfrac=0.24)
    box(7.0, 4.4, 2.6, 4.4, COLORS["chance"], "Chance-\nconstrained", "",
        tfrac=0.84)
    ax.annotate("", xy=(7.45, 8.55), xytext=(7.45, 4.65),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=BLUE))
    ax.text(8.6, 6.0, "reliability dial\n$1-\\varepsilon$",
            ha="center", va="center", fontsize=8.5, color=ANNOT,
            linespacing=1.15)

    fig.tight_layout(pad=0.3)
    finish(fig, "fig_taxonomy")


# ===========================================================================
# fig_levelsets: Prekopa's theorem and the boundary of log-concavity
# One two-panel figure at full text width.
# ===========================================================================
def fig_levelsets():
    g = np.linspace(-0.6, 1.6, 240)
    X1, X2 = np.meshgrid(g, g)

    def p_gauss(mu, Sig):
        s = (np.sqrt(Sig[0, 0]*X1**2 + 2*Sig[0, 1]*X1*X2
                     + Sig[1, 1]*X2**2) + 1e-12)
        return stats.norm.cdf((1 - (mu[0]*X1 + mu[1]*X2)) / s)

    def draw_panel(ax, mixture=False):
        if mixture:
            PA = p_gauss(np.array([1.9, 0.1]), np.array([[0.05, 0], [0, 0.05]]))
            PB = p_gauss(np.array([0.1, 1.9]), np.array([[0.05, 0], [0, 0.05]]))
            P = 0.5 * PA + 0.5 * PB
            level, label_x, label_y = 0.55, -0.42, -0.30
            title = "(b) Two-component mixture"
        else:
            P = p_gauss(np.array([0.8, 0.8]),
                        np.array([[0.16, 0.0], [0.0, 0.16]]))
            level, label_x, label_y = 0.9, -0.30, 0.05
            title = "(a) Gaussian law"
        c = COLORS["safe"]
        ax.contourf(X1, X2, P, levels=[level, 1.001], colors=[c], alpha=0.25)
        ax.contour(X1, X2, P, levels=[level], colors=[c], linewidths=1.6)
        ax.text(label_x, label_y, rf"$\{{x: p(x)\geq {level:g}\}}$",
                fontsize=8.5, color=c, bbox=WBOX, zorder=7)
        ax.set_title(title)
        ticks = [-0.5, 0, 0.5, 1, 1.5]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(g[0], g[-1])
        ax.set_ylim(g[0], g[-1])
        ax.set_aspect("equal")

    fig, axes = plt.subplots(1, 2, figsize=(width_in(1.0), 2.85))
    draw_panel(axes[0], mixture=False)
    draw_panel(axes[1], mixture=True)
    fig.tight_layout(pad=0.3, w_pad=1.5)
    finish(fig, "fig_levelsets")




# ===========================================================================
# fig_kernels: kernel functions and their integrals vs the indicator
# ===========================================================================
def fig_kernels():
    u = np.linspace(-2.2, 2.2, 600)
    shift = 1.0

    gauss = stats.norm.pdf(u)
    epan = np.where(np.abs(u) <= 1, 0.75*(1 - u**2), 0)
    epan_b = np.where(np.abs(u - shift) <= 1, 0.75*(1 - (u - shift)**2), 0)

    fig, axes = plt.subplots(1, 2, figsize=(width_in(1.0), 2.8))

    ax = axes[0]
    ax.plot(u, gauss, color=BLUE, label="Gaussian")
    ax.plot(u, epan, color=BLUE_TINT, ls="--", label="Epanechnikov")
    ax.plot(u, epan_b, color=GREEN, label="biased (shifted support)")
    ax.set_xlabel("$u$")
    ax.set_ylabel("$K(u)$")
    ax.set_title("(a) Kernel functions")
    ax.set_ylim(0, 1.3)
    ax.legend(loc="upper left")

    ax = axes[1]
    cg = stats.norm.cdf(u)
    ce = np.where(u < -1, 0, np.where(u > 1, 1, 0.25*(2 + 3*u - u**3)))
    ub = u - shift
    cb = np.where(ub < -1, 0, np.where(ub > 1, 1, 0.25*(2 + 3*ub - ub**3)))
    ax.step([-2.2, 0, 0, 2.2], [0, 0, 1, 1], where="post", color=TRUTH,
            lw=1.2, label="indicator $\\mathbf{1}\\{u\\geq 0\\}$")
    ax.plot(u, cg, color=BLUE, label="Gaussian")
    ax.plot(u, ce, color=BLUE_TINT, ls="--", label="Epanechnikov")
    ax.plot(u, cb, color=GREEN, label="biased")
    ax.fill_between(u, cb, np.where(u >= 0, 1, 0),
                    where=(cb <= np.where(u >= 0, 1, 0)) & (u > -1.7),
                    color=GREEN, alpha=0.10, lw=0)
    ax.set_xlabel("$u$")
    ax.set_ylabel("$F_K(u)$")
    ax.set_title("(b) Integrated kernels vs indicator")
    ax.set_ylim(-0.05, 1.75)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper left")

    fig.tight_layout(pad=0.3)
    finish(fig, "fig_kernels")


# ===========================================================================
# fig_bandwidth: bias-variance trade-off for kernel density estimation
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

    fig, axes = plt.subplots(1, 2, figsize=(width_in(0.82), 2.75),
                             gridspec_kw=dict(width_ratios=[1.3, 1]))

    ax = axes[0]
    ax.plot(ygrid, truth(ygrid), color=TRUTH, ls="--", lw=1.3,
            label="true density")
    for h, c, ls, lw, lab in [
            (0.05, tint(BLUE, 0.45), "-", 1.2, "$h=0.05$ (undersmoothed)"),
            (0.25, BLUE, "-", 1.8, "$h=0.25$ (near optimal)"),
            (0.90, tint(BLUE, 0.7), (0, (4, 1.5, 1, 1.5)), 1.6,
             "$h=0.90$ (oversmoothed)")]:
        kde = (stats.norm.pdf((ygrid[:, None] - samp[None, :])/h)
               .mean(axis=1) / h)
        ax.plot(ygrid, kde, color=c, ls=ls, lw=lw, label=lab)
    ax.set_xlabel("$y$")
    ax.set_ylabel("density")
    ax.set_ylim(0, 2.0)
    ax.set_yticks([0, 0.4, 0.8])
    ax.set_title(f"(a) Three bandwidths, $N={N}$")
    ax.legend(loc="upper right", handlelength=1.6, labelspacing=0.25)

    hs = np.geomspace(0.03, 1.4, 26)
    T = 120
    f_true = truth(ygrid)
    est = np.zeros((T, len(hs), len(ygrid)))
    for t in range(T):
        st = sample_truth(N, 100 + t)
        for k, h in enumerate(hs):
            est[t, k] = (stats.norm.pdf(
                (ygrid[:, None] - st[None, :])/h).mean(axis=1) / h)

    mean_est = est.mean(axis=0)
    bias2 = np.trapezoid((mean_est - f_true[None, :])**2, ygrid, axis=1)
    var = np.trapezoid(est.var(axis=0), ygrid, axis=1)
    mise = bias2 + var
    hstar = hs[np.argmin(mise)]

    ax = axes[1]
    ax.loglog(hs, bias2, color=GRAY, ls="--", label="squared bias")
    ax.loglog(hs, var, color=BLUE, ls=":", lw=2.0, label="variance")
    ax.loglog(hs, mise, color=TRUTH, lw=1.8, label="MISE")
    ax.axvline(hstar, color=TRUTH, lw=0.8, ls=(0, (1, 2)))
    ax.text(hstar*1.12, 1.45e-4, "$h^{*}$", color=ANNOT, fontsize=9.5)
    ax.set_xlabel("bandwidth $h$")
    ax.set_ylabel("integrated error")
    ax.set_title("(b) Bias-variance trade-off")
    ax.set_ylim(top=40)
    ax.legend(loc="upper right")

    fig.tight_layout(pad=0.3)
    finish(fig, "fig_bandwidth")
    print(f"    h* = {hstar:.3f}")


# ===========================================================================
# fig_curse: sample size required vs dimension
# ===========================================================================
def fig_curse():
    d = np.arange(1, 11)
    N1 = 100.0
    Nd = N1**((d + 4) / 5)

    fig, ax = plt.subplots(figsize=(width_in(0.8), 2.7))
    ax.semilogy(d, Nd, "-o", color=BLUE)
    ax.axhline(1e6, color=GRAY, lw=1.0, ls=":")
    ax.text(1.0, 1.7e6, "$10^6$ samples", color=TXTGRAY, fontsize=8.5)

    for dd, label, xy_text in [
            (1, "$d=1$: $N\\approx 10^{2}$", (1.15, 3.5e3)),
            (5, "$d=5$: $N\\approx 4\\times10^{3}$", (5.6, 1.1e3)),
            (10, "$d=10$: $N\\approx 4\\times10^{5}$", (4.9, 2.6e5))]:
        ax.annotate(label, xy=(dd, N1**((dd + 4)/5)), xytext=xy_text,
                    fontsize=8.5, color=ANNOT, zorder=7,
                    arrowprops=dict(arrowstyle="->", lw=0.7, shrinkB=3,
                                    color=TXTGRAY))

    ax.set_xlabel("dimension $d$ of the estimated variable")
    ax.set_ylabel("samples for fixed accuracy")
    ax.set_xticks(d)
    ax.set_ylim(50, 4e6)
    fig.tight_layout(pad=0.3)
    finish(fig, "fig_curse")


# ===========================================================================
# fig_litmap: KDE chance-constraint literature timeline
# ===========================================================================
def fig_litmap():
    lanes = {
        "Process systems": (6.4, TRUTH),
        "Aerospace":       (4.8, TRUTH),
        "Energy/networks": (3.2, TRUTH),
        "Finance":         (1.6, TRUTH),
        "Theory":          (0.0, TRUTH),
    }

    # (year, lane, label, above_the_dot?, label_ha, label_anchor_year)
    papers = [
        (2015.0, "Process systems", "Calfa et al.",              True,  "center", 2015.0),
        (2018.0, "Aerospace",       "Caillau et al.",            True,  "right",  2018.4),
        (2018.5, "Energy/networks", "Gugat,\nSchuster",          False, "right",  2018.9),
        (2017.8, "Energy/networks", "Ciftci",                    True,  "center", 2017.8),
        (2019.6, "Energy/networks", "Ciftci\net al.",            False, "center", 2019.6),
        (2020.0, "Aerospace",       "Keil et al.\n(biased KDE)", False, "center", 2020.0),
        (2021.3, "Aerospace",       "Keil et al.\n(OCAM)",       True,  "center", 2021.3),
        (2022.0, "Aerospace",       "Keil, Rao\n(warm start)",   False, "center", 2022.3),
        (2021.5, "Energy/networks", "Wu et al.",                 False, "right",  2022.0),
        (2022.4, "Energy/networks", "Schuster et al.",           False, "left",   2022.3),
        (2022.7, "Finance",         "Liu et al.",                False, "center", 2022.7),
        (2023.3, "Energy/networks", "Wu, Kargarian",             True,  "right",  2023.8),
        (2025.4, "Theory",          "Schuster (convergence)",    False, "center", 2025.4),
        (2026.0, "Energy/networks", "Qi et al.",                 True,  "left",   2025.9),
    ]

    fig, ax = plt.subplots(figsize=(width_in(0.88), 3.7))

    for name, (y, c) in lanes.items():
        ax.axhline(y, color=c, lw=0.8, alpha=0.4, zorder=1)
        ax.annotate(name, xy=(2014.0, y), xytext=(0, -4),
                    textcoords="offset points", va="top",
                    fontsize=8.5, color=c, zorder=6, bbox=WBOX)

    ax.annotate("", xy=(2025.4, 0.3), xytext=(2025.25, 4.65),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=TXTGRAY),
                zorder=2)
    ax.text(2025.0, 0.45, "engineering practice\nfeeds convergence theory",
            fontsize=8.5, color=TXTGRAY, ha="right", va="center",
            zorder=6, bbox=WBOX)

    for yr, lane, lab, above, ha, xl in papers:
        y, c = lanes[lane]
        ax.scatter(yr, y, s=46, color=c, zorder=7, ec="white", lw=0.4)
        ax.annotate(lab, xy=(xl, y), xytext=(0, 5 if above else -5),
                    textcoords="offset points", ha=ha, fontsize=8.5,
                    va="bottom" if above else "top",
                    color=ANNOT, linespacing=1.1, zorder=6, bbox=WBOX)

    ax.set_xlim(2013.9, 2027.4)
    ax.set_ylim(-0.95, 7.25)
    ax.set_yticks([])
    ax.set_xlabel("year")
    ax.spines["left"].set_visible(False)
    fig.tight_layout(pad=0.3)
    finish(fig, "fig_litmap")


# ---------------------------------------------------------------------------
# Survey experiment shared setup (seed 21 preserved so published numbers
# reproduce exactly).
# ---------------------------------------------------------------------------
LAW = stats.lognorm(s=0.18, scale=100)
EPS = 0.05
TRUE_Q = LAW.ppf(1 - EPS)
rng_bb = np.random.default_rng(21)


def fig_staircase():
    xs = np.linspace(88, 162, 1200)
    true_v = 1 - LAW.cdf(xs)

    fig, axes = plt.subplots(1, 2, figsize=(width_in(0.82), 2.6))

    ax = axes[0]
    N = 60
    samp = LAW.rvs(N, random_state=3)
    vhat = (samp[None, :] > xs[:, None]).mean(axis=1)
    ax.plot(xs, true_v, color=TRUTH, lw=1.8, label="true $v(x)$")
    ax.step(xs, vhat, where="post", color=ORANGE, lw=1.4,
            label=f"$\\hat v_N(x)$, $N={N}$")
    ax.axhline(EPS, color=TRUTH, ls="--", lw=1.0)
    ax.text(91, EPS + 0.03, "$\\varepsilon = 0.05$", fontsize=8.5,
            color=ANNOT, bbox=WBOX, zorder=7)
    ax.set_xlabel("decision $x$ [MW]")
    ax.set_ylabel("violation probability")
    ax.set_title("(a) Empirical estimator")
    ax.legend(loc="upper right")

    ax = axes[1]
    for k, (seed, c, lw) in enumerate([(11, tint(ORANGE, 0.55), 1.2),
                                       (12, ORANGE, 1.2),
                                       (13, "#8a4e0e", 1.2)]):
        sk = LAW.rvs(N, random_state=seed)
        vk = (sk[None, :] > xs[:, None]).mean(axis=1)
        ax.step(xs, vk, where="post", lw=lw, color=c, label=f"draw {k+1}")
    ax.plot(xs, true_v, color=TRUTH, lw=1.8, label="true $v(x)$")
    ax.axhline(EPS, color=TRUTH, ls="--", lw=1.0)
    ax.set_xlim(120, 155)
    ax.set_ylim(0, 0.18)
    ax.set_xlabel("decision $x$ [MW]")
    ax.set_ylabel("violation probability")
    ax.set_title("(b) Resampling noise")
    ax.legend(loc="upper right")

    fig.tight_layout(pad=0.3)
    finish(fig, "fig_staircase")


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
# Figure 2  fig_scenariogrowth  (survey comparison)
# Scenario conservatism grows with N; SAA empirical quantile stays on target.


def fig_scenariogrowth():
    Ns = np.unique(np.geomspace(20, 2000, 16).astype(int))
    trials = 600
    scen_mean, scen_lo, scen_hi, saa_mean = [], [], [], []

    for N in Ns:
        mx = np.array([
            LAW.rvs(N, random_state=rng_bb.integers(int(1e9))).max()
            for _ in range(trials)])
        qq = np.array([
            np.quantile(LAW.rvs(N, random_state=rng_bb.integers(int(1e9))),
                        1 - EPS)
            for _ in range(trials)])
        scen_mean.append(mx.mean())
        scen_lo.append(np.percentile(mx, 10))
        scen_hi.append(np.percentile(mx, 90))
        saa_mean.append(qq.mean())

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.plot(Ns, scen_mean, "-o", ms=3.2, lw=1.6, color=GRAY,
            label="scenario: $x_N = \\max_i \\xi_i$")
    ax.fill_between(Ns, scen_lo, scen_hi, color=GRAY, alpha=0.15)
    ax.plot(Ns, saa_mean, "-s", ms=3.2, lw=1.6, color=ORANGE,
            label="SAA: empirical $q_{0.95}$")
    ax.axhline(TRUE_Q, color=GREEN, lw=1.4, ls="--")
    ax.text(24, TRUE_Q - 7.5, "exact requirement $q_{0.95}$",
            color=GREEN, fontsize=8.5, bbox=WBOX, zorder=7)
    ax.set_xscale("log")
    ax.set_xlabel("sample size $N$ (= number of hard constraints)")
    ax.set_ylabel("required capacity [MW]")
    styled_legend(ax, loc="upper left", fontsize=8.2)

    fig.tight_layout()
    finish(fig, "fig_scenariogrowth")
    print(f"    scenario at N=2000: {scen_mean[-1]:.1f}, "
          f"SAA: {saa_mean[-1]:.1f}, true q: {TRUE_Q:.1f}")




if __name__ == "__main__":
    print("Generating Chapter 2/3 figures ...")
    fig_taxonomy()
    fig_levelsets()
    fig_staircase()
    fig_kernels()
    fig_bandwidth()
    fig_curse()
    fig_litmap()
    print(f"\nAll figures written to ./{OUT}/.")
