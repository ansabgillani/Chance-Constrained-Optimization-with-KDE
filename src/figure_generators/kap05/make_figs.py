"""
make_chapter5_figures.py
========================
Figures for Chapter 5 (convergence theory).
Code reused from the thesis history; only the output path is relative.

Usage:  pip install numpy scipy matplotlib
        python make_chapter5_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

plt.rcParams.update({
    "font.size": 9.5, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "axes.titlesize": 10, "legend.fontsize": 8.3,
})
BLUE = "#1f4e79"; RED = "#b3392e"; GRAY = "#7a7a7a"
GREEN = "#2e7d4f"; ORANGE = "#d9842b"; PURPLE = "#6a4c93"
WBOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)
import os
F6 = "figures/"
os.makedirs(F6, exist_ok=True)
rng = np.random.default_rng(41)

# ===================================================== fig_chain
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.set_xlim(0, 14); ax.set_ylim(0, 12)
ax.axis("off")

def box(x, y, w, h, color, text, fs=7.8, weight=None):
    r = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.13", fc=color, ec=color, alpha=0.16, lw=1.0)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.25, weight=weight)

def arrow(x1, y1, x2, y2, color="k", rad=0.0, lw=1.1):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=lw, color=color,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=3)

# statistical foundation layer (bottom)
box(0.3, 0.3, 4.0, 1.7, GRAY,
    "$L^1$ consistency of the KDE\n(Parzen 1962;\nDevroye and Gy\u00f6rfi 1985)")
box(5.0, 0.3, 3.8, 1.7, GRAY,
    "Scheff\u00e9's lemma:\ndensity convergence gives\nprobability convergence")
box(9.5, 0.3, 4.2, 1.7, GRAY,
    "uniform convergence\nof $\\hat P_N$ on $X$, a.s.\n(Wied and Wei\u00dfbach 2012)")
arrow(4.3, 1.15, 5.0, 1.15)
arrow(8.8, 1.15, 9.5, 1.15)

# middle layer: the two engines
box(1.4, 4.0, 4.6, 2.2, PURPLE,
    "convexity of the epigraph\n+ strict feasibility margin\n"
    "$P(g(x^*,\\xi)\\leq 0) > \\alpha$")
box(7.6, 4.0, 5.2, 2.2, PURPLE,
    "Alvarez-Mena and\nHern\u00e1ndez-Lerma (2005):\n"
    "three lemmas, Painlev\u00e9-\nKuratowski set convergence")
arrow(2.3, 2.0, 3.0, 4.0, color=GRAY, rad=0.12)
arrow(11.6, 2.0, 10.6, 4.0, color=GRAY, rad=-0.12)

# top layer: three theorems
box(0.3, 8.2, 4.1, 3.0, GREEN,
    "Theorem 6.1\n(passive constraint)\n$x^*$ solves $(P_N)$\nfor all "
    "$N \\geq N^*$", fs=8.0, weight="bold")
box(4.9, 8.2, 4.1, 3.0, GREEN,
    "Theorem 6.2\n(limits are optimal)\nlimits of solutions\nof $(P_N)$ "
    "solve $(P_\\infty)$", fs=8.0, weight="bold")
box(9.5, 8.2, 4.2, 3.0, GREEN,
    "Theorem 6.3\n(existence)\nsharp minimum gives\na convergent sequence",
    fs=8.0, weight="bold")
arrow(3.0, 6.2, 2.4, 8.2, color=PURPLE, rad=0.10)
arrow(9.4, 6.2, 6.9, 8.2, color=PURPLE, rad=0.14)
arrow(10.8, 6.2, 11.5, 8.2, color=PURPLE, rad=-0.10)
arrow(7.0, 11.2, 9.6, 11.2, color=GREEN, lw=1.0)
ax.text(8.3, 11.55, "feeds", fontsize=7.0, color=GREEN, ha="center",
        bbox=WBOX, zorder=7)
fig.tight_layout()
fig.savefig(F6 + "fig_chain.pdf", bbox_inches="tight")
plt.close(fig)
print("chain ok")

# ===================================================== fig_passive
law = stats.lognorm(s=0.18, scale=100)
alpha = 0.95
xs = np.linspace(118, 170, 700)
p = law.cdf(xs)
xstar = 146.0
pstar = law.cdf(xstar)
gamma = pstar - alpha

fig, ax = plt.subplots(figsize=(6.2, 3.1))
for half, alpha_f, lab in [(0.60*gamma, 0.13, None),
                           (0.30*gamma, 0.22,
                            "uniform KDE error tube, shrinking in $N$")]:
    ax.fill_between(xs, p - half, p + half, color=BLUE, alpha=alpha_f,
                    label=lab)
ax.plot(xs, p, color="k", lw=1.7, label="true probability $p(x)$")
ax.axhline(alpha, color=GREEN, ls="--", lw=1.3)
ax.text(119.5, alpha - 0.0085, "$\\alpha = 1-\\varepsilon$",
        color=GREEN, fontsize=8.5, bbox=WBOX, zorder=7)
ax.scatter([xstar], [pstar], color=RED, s=42, zorder=6, ec="white")
ax.annotate("$x^*$: constraint strictly inactive",
            xy=(xstar, pstar), xytext=(128, 1.001), fontsize=8.3,
            color=RED, ha="center", bbox=WBOX, zorder=7,
            arrowprops=dict(arrowstyle="->", lw=0.9, color=RED,
                            shrinkB=6))
ax.annotate("", xy=(163, pstar), xytext=(163, alpha),
            arrowprops=dict(arrowstyle="<->", lw=1.1, color=GRAY))
ax.text(164.0, (pstar + alpha)/2, "margin\n$\\gamma > 0$",
        fontsize=7.8, color=GRAY, va="center", bbox=WBOX, zorder=7)
ax.text(140.5, 0.912,
        "once $\\sup_x |\\hat p_N - p| < \\gamma$,\n"
        "$x^*$ stays feasible for $(P_N)$",
        fontsize=7.8, ha="center", bbox=WBOX, zorder=7)
ax.set_xlim(118, 170); ax.set_ylim(0.875, 1.015)
ax.set_xlabel("decision $x$")
ax.set_ylabel("probability")
leg = ax.legend(loc="lower right", fontsize=7.6, frameon=True,
                framealpha=0.9, edgecolor="none", facecolor="white")
leg.set_zorder(7)
fig.tight_layout()
fig.savefig(F6 + "fig_passive.pdf", bbox_inches="tight")
plt.close(fig)
print("passive ok")

# ===================================================== fig_sharpmin
t = np.linspace(-2.2, 2.2, 600)
f_sharp = 0.55*t**2 + 1.0
f_flat = 1.0 + 0.06*t**4

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for ax, f, title, ok in [
        (axes[0], f_sharp, "(a) Sharp minimum: condition holds", True),
        (axes[1], f_flat, "(b) Shallow valley: condition fails", False)]:
    ax.plot(t, f, color="k", lw=1.7)
    fstar = 1.0
    delta = 0.55
    ebar = (f_sharp if ok else f_flat)[np.argmin(np.abs(t - delta))] - fstar
    ax.scatter([0], [fstar], color=GREEN, s=40, zorder=6, ec="white")
    ax.axvspan(-delta, delta, color=GREEN, alpha=0.10)
    ax.axhline(fstar + ebar, color=GRAY, lw=1.0, ls=":")
    # approximate solutions
    if ok:
        xn = np.array([0.45, -0.3, 0.18, -0.08])
    else:
        xn = np.array([1.6, -1.3, 1.0, -1.75])
    fn = np.interp(xn, t, f)
    ax.scatter(xn, fn, color=ORANGE, s=26, zorder=6, ec="white")
    ax.set_title(title)
    ax.set_xlabel("decision space")
    ax.set_xlim(-2.2, 2.2)
axes[0].set_ylabel("objective $f$")
axes[0].set_ylim(0.7, 3.6)
axes[0].text(0, 0.84, "$B_\\delta(x^*)$", color=GREEN, fontsize=8.3,
             ha="center", bbox=WBOX, zorder=7)
axes[0].text(-1.32, 2.95,
             "outside the ball,\n$f > f(x^*) + \\bar\\varepsilon$:\n"
             "approximate solutions\nare trapped near $x^*$",
             fontsize=7.4, ha="center", bbox=WBOX, zorder=7)
axes[1].text(0, 0.84, "$B_\\delta(x^*)$", color=GREEN, fontsize=8.3,
             ha="center", bbox=WBOX, zorder=7)
axes[1].text(0.05, 2.75,
             "near-optimal points exist\nfar from $x^*$: the sequence\n"
             "may wander without converging",
             fontsize=7.4, ha="center", bbox=WBOX, zorder=7)
fig.tight_layout()
fig.savefig(F6 + "fig_sharpmin.pdf", bbox_inches="tight")
plt.close(fig)
print("sharpmin ok")

# ===================================================== fig_empconv
eps = 0.05
xstar_true = law.ppf(1 - eps)
Ns = np.unique(np.geomspace(50, 6400, 8).astype(int))
trials = 500
err_x, err_f = [], []
for N in Ns:
    ex = []
    for _ in range(trials):
        s = law.rvs(N, random_state=rng.integers(int(1e9)))
        h = 0.9*min(s.std(ddof=1), stats.iqr(s)/1.34)*N**(-0.2)
        grid = np.linspace(s.min()-3*h, s.max()+3*h, 700)
        cdf = stats.norm.cdf((grid[:, None]-s[None, :])/h).mean(axis=1)
        xN = np.interp(1-eps, cdf, grid)
        ex.append(abs(xN - xstar_true))
    err_x.append((np.mean(ex), np.percentile(ex, 10),
                  np.percentile(ex, 90)))
err_x = np.array(err_x)

# fit slope
slope, intercept = np.polyfit(np.log(Ns), np.log(err_x[:, 0]), 1)

fig, ax = plt.subplots(figsize=(6.0, 3.1))
ax.loglog(Ns, err_x[:, 0], "-o", ms=4, lw=1.6, color=BLUE,
          label="mean $|x_N - x^*|$")
ax.fill_between(Ns, err_x[:, 1], err_x[:, 2], color=BLUE, alpha=0.13)
fit = np.exp(intercept)*Ns**slope
ax.loglog(Ns, fit, color=RED, lw=1.2, ls="--",
          label=f"fitted slope $\\approx {slope:.2f}$")
ax.text(900, err_x[0, 0]*0.75,
        "consistency is guaranteed;\nthe rate is not: no theorem\n"
        "predicts this slope",
        fontsize=7.8, ha="center", bbox=WBOX, zorder=7)
ax.set_xlabel("sample size $N$")
ax.set_ylabel("optimization error")
leg = ax.legend(loc="lower left", fontsize=8.0, frameon=True,
                framealpha=0.9, edgecolor="none", facecolor="white")
leg.set_zorder(7)
fig.tight_layout()
fig.savefig(F6 + "fig_empconv.pdf", bbox_inches="tight")
plt.close(fig)
print("empconv ok, slope:", round(slope, 3))
print("all done")