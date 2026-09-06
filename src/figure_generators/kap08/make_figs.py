"""Generate publication-style Chapter 8 figures from saved result CSV data.

The script reads ``results/summary.csv`` and does not recompute experiment values.
It uses PGFPlots because the local environment has TeX but no Matplotlib.
"""
from __future__ import annotations
import csv
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
FIGURES = ROOT / "kap08" / "figures"
METHODS = {
    "nominal": "Nominal",
    "scenario": "Scenario",
    "unbiased_kde": "Unbiased KDE",
    "local_shifted_epanechnikov": "Local shifted Epanechnikov",
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
        coords = " ".join(f"({esc(METHODS[m])},{selected[m]:.6f})" for m in METHODS if selected.get(m) is not None)
        top = index < 2
        ylabel = "ylabel={independent test violation}," if index in (0, 2) else ""
        labels = r"xticklabels=\empty," if top else ""
        title = f"title={{({chr(97 + index)}) {esc(dist.replace('_', ' ').title())}}},"
        note = ("\\node[anchor=west,font=\\scriptsize] at (axis description cs:0.03,0.92) "
                "{dashed line: $\\varepsilon=.1$};\n") if index == 0 else ""
        panels.append(
            f"\\nextgroupplot[{title}symbolic x coords={{Nominal,Scenario,Unbiased KDE,Local shifted Epanechnikov}},"
            f"xtick=data,{labels}xticklabel style={{rotate=35,anchor=east,font=\\scriptsize}},"
            f"ymin=0,ymax=.65,grid=major,{ylabel}xlabel={{}}]\n"
            f"\\addplot[ybar,bar width=8pt,fill=blue!55,draw=blue!80!black] coordinates {{{coords}}};\n"
            "\\addplot[black!65,dashed,thick,forget plot] coordinates "
            "{(Nominal,.1) (Local shifted Epanechnikov,.1)};\n" + note
        )
    return "".join(panels)


def tradeoff_tex(rows):
    out = ["\\addplot[black!65,dashed,thick] coordinates {(0.1,0) (0.1,1.1)};"]
    colors = {"nominal": "red", "scenario": "gray", "unbiased_kde": "blue", "local_shifted_epanechnikov": "green"}
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
\usepackage{pgfplots}\usepgfplotslibrary{groupplots}\pgfplotsset{compat=1.18}
\definecolor{blue}{HTML}{1F4E79}
\begin{document}\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.0cm,vertical sep=1.05cm},width=7.1cm,height=4.7cm]
""" + static_panel_tex(rows) + r"""\end{groupplot}\end{tikzpicture}\end{document}
"""
    tradeoff = r"""\documentclass[border=3pt]{standalone}
\usepackage{pgfplots}\pgfplotsset{compat=1.18}
\definecolor{blue}{HTML}{1F4E79}\definecolor{green}{HTML}{2E7D4F}
\begin{document}\begin{tikzpicture}\begin{axis}[width=13cm,height=7cm,grid=major,
 xlabel={independent test violation},ylabel={objective},xmin=0,xmax=.65,
 scaled x ticks=false,tick label style={/pgf/number format/fixed},
 legend entries={target $\varepsilon=.1$,Nominal,Scenario,Unbiased KDE,Local shifted Epanechnikov},
 legend style={at={(0.02,0.98)},anchor=north west,font=\scriptsize,draw=black,fill=white,fill opacity=.92,text opacity=1},
title={Static nonlinear benchmark: safety--objective trade-off}]
""" + tradeoff_tex(rows) + r"""
\end{axis}\end{tikzpicture}\end{document}
"""
    sensitivity = r"""\documentclass[border=3pt]{standalone}
\usepackage{pgfplots}\usepgfplotslibrary{groupplots}\pgfplotsset{compat=1.18}
\definecolor{blue}{HTML}{1F4E79}\definecolor{red}{HTML}{B3392E}
\begin{document}\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.2cm},width=7.0cm,height=5.5cm,grid=major]
\nextgroupplot[xlabel={bandwidth $h$},ylabel={estimated violation},title={(a) Bandwidth sweep}]
""" + f"\\addplot[blue,mark=*,thick] coordinates {{{sensitivity_series(rows, 'bandwidth_sweep')}}};\n" + r"""\nextgroupplot[xlabel={training sample size $N$},title={(b) Sample-size sweep}]
""" + f"\\addplot[red,mark=square*,thick] coordinates {{{sensitivity_series(rows, 'sample_size_sweep')}}};\n" + r"""\end{groupplot}\end{tikzpicture}\end{document}
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
    with tempfile.TemporaryDirectory(prefix="kap08-pgfplots-") as directory:
        work = Path(directory)
        for name, tex in zip(("fig_static_test_violation", "fig_static_tradeoff", "fig_sensitivity"), documents(rows)):
            compile_pdf(tex, name, work)
    (FIGURES / "figure_source.txt").write_text(
        "All plotted values are read from results/summary.csv; no values are recomputed.\n"
        "Dashed lines mark epsilon=0.1 in the static comparison plots.\n"
    )


if __name__ == "__main__":
    main()
