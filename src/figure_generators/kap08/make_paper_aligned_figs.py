"""Generate figures for the Schuster/Keil-aligned sensitivity run.

The script reads the saved CSV and only formats stored values.  It keeps the
paper-aligned figures separate from the original baseline figures so both
experiment versions remain auditable.
"""
from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results_paper_aligned_20260906"
FIGURES = ROOT / "kap08" / "figures"
DISTS = ("gaussian", "bimodal", "skewed", "heavy_tailed")
METHODS = {
    "schuster_gaussian": "Schuster Gaussian",
    "keil_biased_epanechnikov": "Keil biased Epanechnikov",
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
        legend = ("legend style={at={(0.03,0.97)},anchor=north west,font=\\scriptsize,"
                  "draw=black,fill=white}," if i == 0 else "")
        xlabel = "xlabel={}," if top else "xlabel={training size $N$},"
        panel = [
            f"\\nextgroupplot[title={{({chr(97+i)}) {dist.replace('_', ' ').title()}}},"
            f"{xlabel}{ylabel}{labels}"
            "xmode=log,log basis x=10,xtick={250,1000,5000},"
            f"xticklabel style={{font=\\scriptsize,/pgf/number format/fixed}},"
            f"xlabel style={{yshift=-4pt}},xmin=200,xmax=6500,ymin=0,ymax=.12,grid=major,{legend}]"
        ]
        for method, label in METHODS.items():
            points = []
            selected = [r for r in data if r.get("benchmark") == "static_nonlinear_paper_aligned"
                        and r.get("distribution") == dist and r.get("method") == method]
            selected.sort(key=lambda r: val(r, "n_train"))
            for row in selected:
                points.append(f"({row['n_train']},{val(row, 'test_violation'):.6f})")
            color = "blue!75!black" if method.startswith("schuster") else "red!75!black"
            style = "solid,mark=*" if method.startswith("schuster") else "dashed,mark=square*"
            panel.append(f"\\addplot[{color},{style},thick] coordinates {{{' '.join(points)}}};")
            if i == 0:
                panel.append(f"\\addlegendentry{{{label}}}")
        panel.append("\\addplot[black!60,dashed,forget plot] coordinates {(250,.1) (5000,.1)};")
        panels.append("\n".join(panel))
    return "\n".join(panels)


def dispatch_tex(data):
    series = []
    for method, label, color, style in (
        ("schuster_gaussian", "Schuster Gaussian", "blue!75!black", "solid,mark=*"),
        ("keil_biased_epanechnikov", "Keil biased Epanechnikov", "red!75!black", "dashed,mark=square*"),
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
\usepackage{pgfplots}\usepgfplotslibrary{groupplots}\pgfplotsset{compat=1.18}
\begin{document}\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.15cm,vertical sep=1.05cm},
width=7.0cm,height=4.8cm,grid=major]
""" + static_tex(data) + r"""
\end{groupplot}
\end{tikzpicture}\end{document}
"""
    left, right = dispatch_tex(data)
    dispatch = r"""\documentclass[border=3pt]{standalone}
\usepackage{pgfplots}\usepgfplotslibrary{groupplots}\pgfplotsset{compat=1.18}
\begin{document}\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.3cm},width=7.2cm,height=5.2cm,grid=major,
xmode=log,log basis x=10,xtick={250,1000,5000},xmin=200,xmax=6500]
\nextgroupplot[xlabel={training size $N$},ylabel={objective},title={Dispatch objective}]
""" + left + r"""
\nextgroupplot[xlabel={training size $N$},ylabel={test violation},title={Dispatch test violation},ymin=0,ymax=.1]
""" + right + r"""
\addplot[black!60,dashed,forget plot] coordinates {(250,.1) (5000,.1)};
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
