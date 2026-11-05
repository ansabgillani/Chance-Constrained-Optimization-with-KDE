"""
make_chapter6_figures.py
========================
Figures for Chapter 6 (convergence theory).
Code reused from the thesis history; only the output path is relative.

Usage:  pip install numpy scipy matplotlib
        python make_chapter6_figures.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import to_rgba

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thesis_style import apply_style, COLORS, HATCH_UNSAFE, SAFE_FILL, LINE, WBOX, style, width_in, TEXT
apply_style()
BLUE = COLORS["unbiased"]; RED = COLORS["violation"]; GRAY = COLORS["nominal"]
GREEN = COLORS["safe"]; ORANGE = COLORS["scenario"]; TRUTH = COLORS["truth"]
ANN = COLORS["annotation"]
F6 = "figures/"
os.makedirs(F6, exist_ok=True)
rng = np.random.default_rng(41)

# ===================================================== fig_passive
law = stats.lognorm(s=0.18, scale=100)
# Standalone scalar diagnostic law; these figures are not generated from
# the dispatch experiment ledger.
alpha = 0.95
xs = np.linspace(118, 170, 700)
p = law.cdf(xs)
xstar = 146.0
pstar = law.cdf(xstar)
gamma = pstar - alpha

fig, ax = plt.subplots(figsize=(width_in(0.86), 3.05))
for half, alpha_f, lab in [(0.60*gamma, 0.13, None),
                           (0.30*gamma, 0.22,
                            "uniform KDE error tube, shrinking in $N$")]:
    ax.fill_between(xs, p - half, p + half, color=BLUE, alpha=alpha_f,
                    label=lab)
    ax.fill_between(xs, p - half, p + half, facecolor="none",
                    edgecolor=BLUE, lw=0.6)
ax.plot(xs, p, color=TRUTH, lw=1.7, label="true probability $p(x)$")
ax.axhline(alpha, color=TRUTH, ls="--", lw=1.1)
ax.text(119.5, alpha - 0.013, "$\\alpha = 1-\\varepsilon$",
        color=TRUTH, bbox=WBOX, zorder=7)
ax.scatter([xstar], [pstar], color=TRUTH, s=42, zorder=6, ec="white")
ax.annotate("$x^*$: constraint strictly inactive",
            xy=(xstar, pstar), xytext=(119, 1.022),
            color=TRUTH, ha="left", bbox=WBOX, zorder=7,
            arrowprops=dict(arrowstyle="->", lw=0.9, color=TRUTH,
                            shrinkB=6))
ax.annotate("", xy=(163, pstar), xytext=(163, alpha),
            arrowprops=dict(arrowstyle="<->", lw=1.0, color=ANN))
ax.text(164.0, (pstar + alpha)/2, "margin\n$\\gamma > 0$",
        color=ANN, va="center", bbox=WBOX, zorder=7)
ax.text(154, 0.915,
        "once $\\sup_x |\\hat p_N - p| < \\gamma$,\n"
        "$x^*$ stays feasible for $(P_N)$",
        color=ANN, ha="center", zorder=7)
ax.set_xlim(118, 170); ax.set_ylim(0.855, 1.035)
ax.set_xlabel("decision $x$")
ax.set_ylabel("probability")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(F6 + "fig_passive.pdf", bbox_inches="tight")
plt.close(fig)
print("passive ok")

# ===================================================== fig_sharpmin
t = np.linspace(-2.2, 2.2, 600)
f_sharp = 0.55*t**2 + 1.0
f_flat = 1.0 + 0.06*t**4

fig, axes = plt.subplots(1, 2, figsize=(width_in(1.0), 2.85), sharey=True)
for ax, f, title, ok in [
        (axes[0], f_sharp, "(a) Sharp minimum: condition holds", True),
        (axes[1], f_flat, "(b) Shallow valley: condition fails", False)]:
    ax.plot(t, f, color=TRUTH, lw=1.7)
    fstar = 1.0
    delta = 0.55
    ebar = (f_sharp if ok else f_flat)[np.argmin(np.abs(t - delta))] - fstar
    ax.scatter([0], [fstar], color=GREEN, marker="*", s=110, zorder=7, ec=TRUTH, lw=0.6)
    ax.axvspan(-delta, delta, color=SAFE_FILL, lw=0)
    ax.axhline(fstar + ebar, color=TRUTH, lw=1.0, ls=":")
    # approximate solutions
    if ok:
        xn = np.array([0.45, -0.3, 0.18, -0.08])
    else:
        xn = np.array([1.6, -1.3, 1.0, -1.75])
    fn = np.interp(xn, t, f)
    ax.scatter(xn, fn, facecolor="white", edgecolor=TRUTH, lw=0.8, s=30, zorder=6)
    ax.set_title(title)
    ax.set_xlabel("decision space")
    ax.set_xlim(-2.2, 2.2)
axes[0].set_ylabel("objective $f$")
axes[0].set_ylim(0.45, 4.8)
axes[0].text(0, 0.62, "$B_\\delta(x^*)$", color=TEXT["truth"],
             ha="center", zorder=7)
axes[0].text(0, 3.2,
             "outside the ball,\n$f > f(x^*) + \\kappa$:\n"
             "approximate solutions\nare trapped near $x^*$",
             color=ANN, ha="center", zorder=7)
axes[1].text(0, 0.62, "$B_\\delta(x^*)$", color=TEXT["truth"],
             ha="center", zorder=7)
axes[1].text(0, 3.2,
             "near-optimal points exist\nfar from $x^*$: the sequence\n"
             "may wander without converging",
             color=ANN, ha="center", zorder=7)
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

fig, ax = plt.subplots(figsize=(width_in(0.84), 3.0))
ax.loglog(Ns, err_x[:, 0], ms=4, lw=1.6, **style("unbiased", every=None),
          label="mean $|x_N - x^*|$")
ax.fill_between(Ns, err_x[:, 1], err_x[:, 2], facecolor=to_rgba(BLUE, 0.13),
                edgecolor=BLUE, lw=0.6, ls=style("unbiased")["ls"])
fit = np.exp(intercept)*Ns**slope
ax.loglog(Ns, fit, color=TRUTH, lw=1.2, ls="--",
          label=f"fitted slope $\\approx {slope:.2f}$")
ax.text(900, 13,
        "consistency is guaranteed;\nthe rate is not: no theorem\n"
        "predicts this slope",
        color=ANN, ha="center", zorder=7)
ax.set_ylim(0.08, 40)
ax.set_xlabel("sample size $N$")
ax.set_ylabel("optimization error")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(F6 + "fig_empconv.pdf", bbox_inches="tight")
plt.close(fig)
print("empconv ok, slope:", round(slope, 3))
print("all done")
