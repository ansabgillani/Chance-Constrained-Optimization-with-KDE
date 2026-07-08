"""
make_chapter4_figures.py
========================
Figures for Chapter 4 (biased KDE and conservative reformulation).
Code reused from the thesis history; only the output path is relative.

Usage:  pip install numpy scipy matplotlib
        python make_chapter4_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    "font.size": 9.5, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "axes.titlesize": 10, "legend.fontsize": 8.3,
})
BLUE = "#1f4e79"; RED = "#b3392e"; GRAY = "#7a7a7a"
GREEN = "#2e7d4f"; ORANGE = "#d9842b"; PURPLE = "#6a4c93"
WBOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.88)
import os
F5 = "figures/"
os.makedirs(F5, exist_ok=True)
rng = np.random.default_rng(31)

def C_epan(u):
    """Integrated Epanechnikov kernel."""
    u = np.asarray(u, dtype=float)
    out = np.where(u < -1, 0.0, np.where(u > 1, 1.0,
                   0.25*(2 + 3*u - u**3)))
    return out

# ===================================================== fig_domination
y = np.linspace(-2.6, 2.6, 800)
h = 1.0
ind = (y <= 0).astype(float)            # indicator 1{y<=0}
phi_unb = stats.norm.cdf(-y/h)          # unbiased Gaussian surrogate
phi_bias = C_epan((-y - h)/h)           # biased Epanechnikov, b = h

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
ax = axes[0]
ax.step([-2.6, 0, 0, 2.6], [1, 1, 0, 0], where="post", color="k",
        lw=1.4, label="indicator $\\mathbf{1}\\{y\\leq 0\\}$")
ax.plot(y, phi_unb, color=RED, lw=1.6, label="unbiased (Gaussian)")
ax.plot(y, phi_bias, color=GREEN, lw=1.6,
        label="biased (Epanechnikov, $b=h$)")
ax.fill_between(y, phi_unb, ind, where=(y > 0), color=RED, alpha=0.18)
ax.fill_between(y, phi_bias, ind, where=(phi_bias < ind),
                color=GREEN, alpha=0.18)
ax.text(1.62, 0.62, "optimism:\ncredit for\nviolating samples",
        fontsize=7.4, color=RED, ha="center", bbox=WBOX, zorder=7)
ax.text(-1.85, 0.52, "safety margin:\ndiscount for\nboundary samples",
        fontsize=7.4, color=GREEN, ha="center", bbox=WBOX, zorder=7)
ax.set_xlabel("residual $y$ (units of $h$)")
ax.set_ylabel("per-sample contribution to $\\hat p$")
ax.set_title("(a) The domination condition")
leg = ax.legend(loc="lower left", fontsize=7.0, frameon=True,
                framealpha=0.9, edgecolor="none", facecolor="white")
leg.set_zorder(7)

ax = axes[1]
bh = np.linspace(0.5, 4.5, 300)
overshoot = stats.norm.cdf(-bh)
ax.semilogy(bh, overshoot, color=BLUE, lw=1.7)
ax.scatter([3.0], [stats.norm.cdf(-3.0)], color=RED, s=30, zorder=6)
ax.annotate("$b = 3h$: residual optimism\n$\\Phi(-3) \\approx 0.13\\%$",
            xy=(3.0, stats.norm.cdf(-3.0)), xytext=(1.65, 4e-4),
            fontsize=7.8, bbox=WBOX, zorder=7,
            arrowprops=dict(arrowstyle="->", lw=0.8, shrinkB=4))
ax.set_xlabel("bias $b$ in units of $h$")
ax.set_ylabel("max per-sample optimism")
ax.set_title("(b) The Gaussian kernel: no finite bias suffices")
fig.tight_layout()
fig.savefig(F5 + "fig_domination.pdf", bbox_inches="tight")
plt.close(fig)
print("domination ok")

# ===================================================== fig_safetycost
law = stats.lognorm(s=0.18, scale=100)
eps = 0.05
true_q = law.ppf(1-eps)
Ns = [50, 100, 200, 500, 1000]
trials = 800

def decide_unbiased(s, h):
    grid = np.linspace(s.min()-4*h, s.max()+4*h, 900)
    cdf = stats.norm.cdf((grid[:, None]-s[None, :])/h).mean(axis=1)
    return np.interp(1-eps, cdf, grid)

def decide_biased(s, h):
    # p_b(x) = mean C_E(((x - xi_j) - h)/h), nondecreasing in x
    grid = np.linspace(s.min()-h, s.max()+4*h, 900)
    cdf = C_epan(((grid[:, None]-s[None, :]) - h)/h).mean(axis=1)
    return np.interp(1-eps, cdf, grid)

res = {"unbiased": {"rel": [], "cap": []},
       "biased": {"rel": [], "cap": []}}
for N in Ns:
    ru, rb, cu, cb = [], [], [], []
    for _ in range(trials):
        s = law.rvs(N, random_state=rng.integers(int(1e9)))
        h = 0.9*min(s.std(ddof=1), stats.iqr(s)/1.34)*N**(-0.2)
        xu = decide_unbiased(s, h)
        xb = decide_biased(s, h)
        ru.append(law.cdf(xu)); rb.append(law.cdf(xb))
        cu.append(xu); cb.append(xb)
    res["unbiased"]["rel"].append(
        (np.mean(ru), np.percentile(ru, 10), np.percentile(ru, 90),
         np.mean(np.array(ru) < 1-eps)))
    res["biased"]["rel"].append(
        (np.mean(rb), np.percentile(rb, 10), np.percentile(rb, 90),
         np.mean(np.array(rb) < 1-eps)))
    res["unbiased"]["cap"].append(np.mean(cu))
    res["biased"]["cap"].append(np.mean(cb))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
ax = axes[0]
for key, c, lab in [("unbiased", RED, "unbiased (Gaussian)"),
                    ("biased", GREEN, "biased (Epan., $b=h$)")]:
    arr = np.array([r[:3] for r in res[key]["rel"]])
    ax.plot(Ns, arr[:, 0], "-o", ms=3.4, lw=1.5, color=c, label=lab)
    ax.fill_between(Ns, arr[:, 1], arr[:, 2], color=c, alpha=0.13)
ax.axhline(1-eps, color="k", ls="--", lw=1.1)
ax.set_xscale("log"); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
ax.set_xlabel("sample size $N$")
ax.set_ylabel("achieved reliability")
ax.set_title("(a) Where the certificates land")
leg = ax.legend(loc="lower right", fontsize=7.6, frameon=True,
                framealpha=0.9, edgecolor="none", facecolor="white")
leg.set_zorder(7)

ax = axes[1]
fr_u = [r[3]*100 for r in res["unbiased"]["rel"]]
fr_b = [r[3]*100 for r in res["biased"]["rel"]]
w = 0.32
xpos = np.arange(len(Ns))
ax.bar(xpos - w/2, fr_u, width=w, color=RED, alpha=0.8,
       label="unbiased")
ax.bar(xpos + w/2, fr_b, width=w, color=GREEN, alpha=0.8,
       label="biased")
ax.set_xticks(xpos); ax.set_xticklabels(Ns)
ax.set_xlabel("sample size $N$")
ax.set_ylabel("trials violating target [%]")
ax.set_title("(b) Certificate failure rate")
leg = ax.legend(loc="upper right", fontsize=8.0, frameon=True,
                framealpha=0.9, edgecolor="none", facecolor="white")
leg.set_zorder(7)
fig.tight_layout()
fig.savefig(F5 + "fig_safetycost.pdf", bbox_inches="tight")
plt.close(fig)
prem = [(b-u)/u*100 for u, b in
        zip(res["unbiased"]["cap"], res["biased"]["cap"])]
print("safetycost ok; fail% unb:", [round(f,1) for f in fr_u],
      "bias:", [round(f,1) for f in fr_b],
      "premium%:", [round(p,2) for p in prem])

# ===================================================== fig_lse
# two-constraint toy along a 1D slice
t = np.linspace(-2, 2, 600)
g1 = t - 0.8
g2 = -1.2*t - 0.6
mx = np.maximum(g1, g2)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
ax = axes[0]
ax.plot(t, g1, color=GRAY, lw=1.0, ls=":", label="$g_1$")
ax.plot(t, g2, color=GRAY, lw=1.0, ls="--", label="$g_2$")
ax.plot(t, mx, color="k", lw=1.8, label="$s=\\max(g_1, g_2)$")
for tau, c in [(0.4, ORANGE), (0.15, BLUE)]:
    lse = tau*np.log(np.exp(g1/tau) + np.exp(g2/tau))
    ax.plot(t, lse, color=c, lw=1.5,
            label=f"$s_\\tau$, $\\tau={tau}$")
ax.axhline(0, color=GREEN, lw=0.9, ls=":")
ax.set_xlabel("slice through decision space")
ax.set_ylabel("violation statistic")
ax.set_title("(a) Log-sum-exp dominates the max")
leg = ax.legend(loc="lower left", fontsize=7.2, frameon=True,
                framealpha=0.9, edgecolor="none", facecolor="white")
leg.set_zorder(7)

ax = axes[1]
tau = 0.15
p1 = np.exp(g1/tau) / (np.exp(g1/tau) + np.exp(g2/tau))
ax.plot(t, p1, color=BLUE, lw=1.7, label="$\\pi_1$ (weight on $g_1$)")
ax.plot(t, 1-p1, color=ORANGE, lw=1.7, label="$\\pi_2$ (weight on $g_2$)")
xc = (0.8 - 0.6/1.2) / (1 + 1.2) * 0 + ( -0.6 + 0.8*0)  # solve g1=g2
xstar = ( -0.6 - (-0.8)*1 ) / (1 + 1.2)
xstar = ( -0.6 + 0.8 ) / (1 + 1.2)
ax.axvline(xstar, color=GRAY, lw=0.9, ls=":")
ax.annotate("active constraint\nswitches here",
            xy=(xstar, 0.5), xytext=(-1.45, 0.55), fontsize=7.4,
            color=GRAY, ha="center", bbox=WBOX, zorder=7,
            arrowprops=dict(arrowstyle="->", lw=0.8, color=GRAY,
                            shrinkB=6))
ax.set_xlabel("slice through decision space")
ax.set_ylabel("softmax weight")
ax.set_ylim(-0.05, 1.05)
ax.set_title("(b) Smooth gradient blending ($\\tau = 0.15$)")
leg = ax.legend(loc="center right", fontsize=7.4, frameon=True,
                framealpha=0.9, edgecolor="none", facecolor="white")
leg.set_zorder(7)
fig.tight_layout()
fig.savefig(F5 + "fig_lse.pdf", bbox_inches="tight")
plt.close(fig)
print("lse ok")

# ===================================================== fig_continuation
stages = np.arange(1, 6)
hs = np.array([8.0, 4.0, 2.0, 1.0, 1.0])
kernels = [BLUE, BLUE, BLUE, BLUE, GREEN]

fig, ax = plt.subplots(figsize=(5.8, 2.9))
ax.semilogy(stages, hs, color=GRAY, lw=1.0, ls="--", zorder=2)
for st, hh, c in zip(stages, hs, kernels):
    ax.scatter(st, hh, s=70, color=c, zorder=6, ec="white", lw=0.6)
for st in stages[:-1]:
    ax.annotate("", xy=(st+0.86, hs[st]*1.0), xytext=(st+0.14, hs[st-1]),
                arrowprops=dict(arrowstyle="->", lw=0.9, color=ORANGE,
                                connectionstyle="arc3,rad=-0.25"),
                zorder=3)
ax.text(2.5, 9.5, "warm start: each solution\ninitializes the next stage",
        fontsize=7.8, color=ORANGE, ha="center", bbox=WBOX, zorder=7)
ax.text(1.0, 5.6, "Gaussian kernel\n(smooth, fast)", fontsize=7.6,
        color=BLUE, ha="center", bbox=WBOX, zorder=7)
ax.text(4.62, 0.55, "switch to split-Bernstein\n(exact safety)",
        fontsize=7.6, color=GREEN, ha="center", bbox=WBOX, zorder=7)
ax.text(4.85, 6.8, "$N$ increases as\n$h$ decreases", fontsize=7.6,
        color=GRAY, ha="center", bbox=WBOX, zorder=7)
ax.set_xticks(stages)
ax.set_xlabel("continuation stage")
ax.set_ylabel("bandwidth $h$ (log scale)")
ax.set_xlim(0.5, 5.7)
ax.set_ylim(0.35, 14)
fig.tight_layout()
fig.savefig(F5 + "fig_continuation.pdf", bbox_inches="tight")
plt.close(fig)
print("continuation ok")
print("all done")