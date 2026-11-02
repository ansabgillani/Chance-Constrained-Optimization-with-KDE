"""Generate figures for the Schuster/Keil-aligned sensitivity run.

The script reads the saved CSV and only formats stored values.  It keeps the
paper-aligned figures separate from the original baseline figures so both
experiment versions remain auditable.
"""
from __future__ import annotations

import csv
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(os.environ.get("THESIS_RESULTS_DIR", ROOT / "results_paper_aligned_20260906"))
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
XTICKS = "xmode=log,log basis x=10,xtick={250,1000,5000},xticklabels={250,1000,5000},minor tick num=0,"
DISTS = ("gaussian", "bimodal", "skewed", "heavy_tailed")
METHODS = {
    "schuster_gaussian": "Unbiased KDE",
    "keil_biased_epanechnikov": "Biased KDE",
}


def rows():
    with (RESULTS / "summary.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def val(row, key):
    return float(row[key])


def static_tex(data):
    panels = []
    for i, dist in enumerate(DISTS):
        top = i < 2
        ylabel = "ylabel={test violation}," if i in (0, 2) else ""
        labels = r"xticklabels=\empty," if top else ""
        legend = ("legend style={at={(1.12,1.38)},anchor=south,legend columns=2}," if i == 0 else "")
        xlabel = "xlabel={}," if top else "xlabel={training size $N$},"
        panel = [
            f"\\nextgroupplot[title={{({chr(97+i)}) {dist.replace('_', ' ').title()}}},"
            f"{xlabel}{ylabel}{labels}"
            f"{XTICKS}"
            "ytick={0,0.05,0.1},yticklabel style={/pgf/number format/fixed},"
            f"xmin=200,xmax=6500,ymin=0,ymax=.12,{legend}]"
        ]
        for method, label in METHODS.items():
            points = []
            selected = [r for r in data if r.get("benchmark") == "static_nonlinear_paper_aligned"
                        and r.get("distribution") == dist and r.get("method") == method]
            selected.sort(key=lambda r: val(r, "n_train"))
            for row in selected:
                points.append(f"({row['n_train']},{val(row, 'test_violation'):.6f})")
            color = "cunbiased" if method.startswith("schuster") else "cbiased"
            style = "solid,mark=*" if method.startswith("schuster") else "dashed,mark=square*"
            panel.append(f"\\addplot[{color},{style},thick] coordinates {{{' '.join(points)}}};")
            if i == 0:
                panel.append(f"\\addlegendentry{{{label}}}")
        panel.append("\\addplot[ctruth,dashed,thick,forget plot] coordinates {(250,.1) (5000,.1)};")
        panels.append("\n".join(panel))
    return "\n".join(panels)


def dispatch_tex(data):
    series = []
    for method, label, color, style in (
        ("schuster_gaussian", "Unbiased KDE", "cunbiased", "solid,mark=*"),
        ("keil_biased_epanechnikov", "Biased KDE", "cbiased", "dashed,mark=square*"),
    ):
        selected = [r for r in data if r.get("benchmark") == "energy_dispatch_paper_aligned"
                    and r.get("method") == method]
        selected.sort(key=lambda r: val(r, "n_train"))
        objective = " ".join(f"({r['n_train']},{val(r, 'objective'):.6f})" for r in selected)
        violation = " ".join(f"({r['n_train']},{val(r, 'test_violation'):.6f})" for r in selected)
        series.append((label, color, style, objective, violation))
    left = []
    right = []
    for label, color, style, objective, violation in series:
        left += [f"\\addplot[{color},{style},thick] coordinates {{{objective}}};",
                 f"\\addlegendentry{{{label}}}"]
        right.append(f"\\addplot[{color},{style},thick] coordinates {{{violation}}};")
    return "\n".join(left), "\n".join(right)


def documents(data):
    static = r"""\documentclass[border=3pt]{standalone}
""" + PREAMBLE + r"""\usepgfplotslibrary{groupplots}
\begin{document}\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.6cm,vertical sep=1.5cm},
width=7.0cm,height=4.4cm]
""" + static_tex(data) + r"""
\end{groupplot}
\end{tikzpicture}\end{document}
"""
    left, right = dispatch_tex(data)
    dispatch = r"""\documentclass[border=3pt]{standalone}
""" + PREAMBLE + r"""\usepgfplotslibrary{groupplots}
\begin{document}\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.9cm},width=6.85cm,height=5.0cm,
""" + XTICKS + r"""xmin=200,xmax=6500]
\nextgroupplot[xlabel={training size $N$},ylabel={objective},title={(a) Objective},ymax=940,
legend style={at={(0.97,0.97)},anchor=north east}]
""" + left + r"""
\nextgroupplot[xlabel={training size $N$},ylabel={test violation},title={(b) Test violation},ymin=0,ymax=.1,
ytick={0,0.02,0.04,0.06,0.08,0.1},yticklabel style={/pgf/number format/fixed}]
""" + right + r"""
\addplot[ctruth,dashed,thick,forget plot] coordinates {(250,.1) (5000,.1)};
\end{groupplot}\end{tikzpicture}\end{document}
"""
    return {"fig_paper_aligned_static": static, "fig_paper_aligned_dispatch": dispatch}


def main():
    data = rows()
    FIGURES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paper-aligned-pgfplots-") as directory:
        work = Path(directory)
        for name, source in documents(data).items():
            tex = work / f"{name}.tex"
            tex.write_text(source)
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                           cwd=work, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (FIGURES / f"{name}.pdf").write_bytes((work / f"{name}.pdf").read_bytes())


if __name__ == "__main__":
    main()
