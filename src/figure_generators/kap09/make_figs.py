"""Generate publication-style Chapter 9 figures from saved result CSV data.

The script reads ``results/summary.csv`` and does not recompute experiment values.
It uses PGFPlots because the local environment has TeX but no Matplotlib.
"""
from __future__ import annotations
import csv
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(os.environ.get("THESIS_RESULTS_DIR", ROOT / "results"))
FIGURES = Path(os.environ.get("THESIS_FIGURES_DIR", ROOT / "kap09" / "figures"))
PREAMBLE = (r"""\usepackage[T1]{fontenc}\usepackage[bitstream-charter]{mathdesign}
\usepackage{pgfplots}\pgfplotsset{compat=1.18}
\definecolor{cnominal}{HTML}{7A7A7A}\definecolor{cscenario}{HTML}{D9842B}
\definecolor{cunbiased}{HTML}{1F4E79}\definecolor{cbiased}{HTML}{2E7D4F}
\definecolor{ctruth}{HTML}{222222}\definecolor{cannot}{HTML}{333333}
"""
        # House style: 8.5 pt ticks/legends, 9 pt labels, 9.5 pt regular titles, open axes, no grid, unframed legends.
        r"\pgfplotsset{every axis/.append style={axis lines=left,axis line style={draw=cannot,line width=0.8pt,-},"
        r"tick style={draw=cannot,line width=0.8pt},tick label style={font=\fontsize{8.5}{10}\selectfont},"
        r"label style={font=\fontsize{9}{11}\selectfont},title style={font=\fontsize{9.5}{11}\selectfont,yshift=-2pt},"
        r"legend style={font=\fontsize{8.5}{10}\selectfont,draw=none,fill=none,legend cell align=left}}}" "\n")
BAR = {"nominal": "cnominal", "scenario": "cscenario", "unbiased_kde": "cunbiased",
       "local_shifted_epanechnikov": "cbiased"}
METHODS = {
    "nominal": "Nominal",
    "scenario": "Scenario",
    "unbiased_kde": "Unbiased KDE",
    "local_shifted_epanechnikov": "Biased KDE",
}
DISTS = ("gaussian", "bimodal", "skewed", "heavy_tailed")


def rows_from_csv():
    path = RESULTS / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"run the experiment first: {path}")
    return list(csv.DictReader(path.open(newline="")))


def num(row, key):
    value = row.get(key, "")
    return None if value in ("", "None", "null") else float(value)


def esc(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def static_panel_tex(rows):
    panels = []
    for index, dist in enumerate(DISTS):
        selected = {r["method"]: num(r, "test_violation") for r in rows
                    if r.get("benchmark") == "static_nonlinear" and r.get("distribution") == dist}
        bars = "".join(f"\\addplot[ybar,bar width=9pt,bar shift=0pt,fill={BAR[m]},draw={BAR[m]}] "
                       f"coordinates {{({esc(METHODS[m])},{selected[m]:.6f})}};\n"
                       for m in METHODS if selected.get(m) is not None)
        top = index < 2
        labels = r"xticklabels=\empty," if top else ""
        title = f"title={{({chr(97 + index)}) {esc(dist.replace('_', ' ').title())}}},"
        note = ("\\node[anchor=east,font=\\fontsize{8.5}{10}\\selectfont,text=cannot] at (axis description cs:0.97,0.92) "
                "{dashed line: $\\varepsilon=0.1$};\n") if index == 0 else ""
        panels.append(
            f"\\nextgroupplot[{title}symbolic x coords={{Nominal,Scenario,Unbiased KDE,Biased KDE}},"
            f"xtick={{Nominal,Scenario,Unbiased KDE,Biased KDE}},enlarge x limits=0.15,{labels}xticklabel style={{rotate=35,anchor=east}},"
            f"ymin=0,ymax=.65,ytick={{0,0.2,0.4,0.6}},xlabel={{}}]\n"
            + bars +
            "\\addplot[ctruth,dashed,thick,forget plot] coordinates "
            "{(Nominal,.1) (Biased KDE,.1)};\n" + note
        )
    return "".join(panels)


def tradeoff_tex(rows):
    out = ["\\addplot[ctruth,dashed,thick] coordinates {(0.1,0) (0.1,1.1)};"]
    colors = {"nominal": "cnominal", "scenario": "cscenario", "unbiased_kde": "cunbiased", "local_shifted_epanechnikov": "cbiased"}
    for method, color in colors.items():
        points = []
        for row in rows:
            if row.get("benchmark") == "static_nonlinear" and row.get("method") == method:
                x, y = num(row, "test_violation"), num(row, "objective")
                if x is not None and y is not None:
                    points.append(f"({x:.6f},{y:.6f})")
        if points:
            out += [f"\\addplot[only marks,mark=*,mark size=2.5pt,color={color}] coordinates {{{' '.join(points)}}};"]
    return "\n".join(out)


def sensitivity_series(rows, method):
    values = [r for r in rows if r.get("benchmark") == "residual_sensitivity" and r.get("method") == method]
    key = "bandwidth" if method == "bandwidth_sweep" else "n_train"
    values.sort(key=lambda r: float(r[key]))
    return " ".join(f"({float(r[key]):.6f},{num(r, 'estimated_violation'):.6f})" for r in values if num(r, 'estimated_violation') is not None)


def documents(rows):
    static = r"""\documentclass[border=3pt]{standalone}
""" + PREAMBLE + r"""\usepgfplotslibrary{groupplots}\usetikzlibrary{calc}
\begin{document}\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.5cm,vertical sep=1.0cm,group name=g},width=7.0cm,height=3.8cm]
""" + static_panel_tex(rows) + r"""\end{groupplot}
\node[rotate=90,anchor=south,font=\fontsize{9}{11}\selectfont] at ($(g c1r1.west)!0.5!(g c1r2.west)+(-0.8cm,0)$) {independent test violation};
\end{tikzpicture}\end{document}
"""
    tradeoff = r"""\documentclass[border=3pt]{standalone}
""" + PREAMBLE + r"""
\begin{document}\begin{tikzpicture}\begin{axis}[width=13.3cm,height=7.0cm,
 xlabel={independent test violation},ylabel={objective},xmin=0,xmax=.65,
 scaled x ticks=false,tick label style={/pgf/number format/fixed},
 legend entries={target $\varepsilon=0.1$,Nominal,Scenario,Unbiased KDE,Biased KDE},
 legend style={at={(0.98,0.97)},anchor=north east},
 xtick={0,0.1,0.2,0.3,0.4,0.5,0.6}]
""" + tradeoff_tex(rows) + r"""
\end{axis}\end{tikzpicture}\end{document}
"""
    sensitivity = r"""\documentclass[border=3pt]{standalone}
""" + PREAMBLE + r"""\usepgfplotslibrary{groupplots}
\begin{document}\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.7cm},width=6.5cm,height=5.0cm,enlarge x limits=0.06,enlarge y limits=0.06]
\nextgroupplot[xlabel={bandwidth $h$},ylabel={estimated violation},title={(a) Bandwidth sweep}]
""" + f"\\addplot[cunbiased,mark=*,thick] coordinates {{{sensitivity_series(rows, 'bandwidth_sweep')}}};\n" + r"""\nextgroupplot[xlabel={training sample size $N$},title={(b) Sample-size sweep}]
""" + f"\\addplot[cunbiased,mark=square*,thick] coordinates {{{sensitivity_series(rows, 'sample_size_sweep')}}};\n" + r"""\end{groupplot}\end{tikzpicture}\end{document}
"""
    return static, tradeoff, sensitivity


def compile_pdf(tex, name, work):
    source = work / f"{name}.tex"
    source.write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", source.name],
                   cwd=work, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    target = FIGURES / f"{name}.pdf"
    target.write_bytes((work / f"{name}.pdf").read_bytes())


def main():
    rows = rows_from_csv()
    FIGURES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kap09-pgfplots-") as directory:
        work = Path(directory)
        for name, tex in zip(("fig_static_test_violation", "fig_static_tradeoff", "fig_sensitivity"), documents(rows)):
            compile_pdf(tex, name, work)
    (FIGURES / "figure_source.txt").write_text(
        "All plotted values are read from results/summary.csv; no values are recomputed.\n"
        "Dashed lines mark epsilon=0.1 in the static comparison plots.\n"
    )


if __name__ == "__main__":
    main()
