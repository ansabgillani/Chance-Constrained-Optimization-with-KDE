"""Create a deterministic PDF summary from the saved experiment CSV."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
FIGURES = ROOT / "kap08" / "figures"


def _pdf(lines, bars):
    stream=["BT /F1 10 Tf 48 560 Td", "(Chapter 8 local evidence) Tj", "0 -18 Td"]
    for line in lines:
        safe=str(line).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.append(f"({safe}) Tj 0 -14 Td")
    stream += ["0 4 Td"]
    for i,value in enumerate(bars):
        height=max(1.0,min(260.0,float(value)*2600.0)); stream.append(f"{70+i*105} 180 55 {height:.2f} re f")
    body="\n".join(stream)+"\nET"
    objects=["<< /Type /Catalog /Pages 2 0 R >>","<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
      "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 650] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
      "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",f"<< /Length {len(body.encode())} >>\nstream\n{body}\nendstream"]
    out=bytearray(b"%PDF-1.4\n"); offsets=[]
    for i,obj in enumerate(objects,1): offsets.append(len(out)); out+=f"{i} 0 obj\n{obj}\nendobj\n".encode()
    xref=len(out); out+=f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode()
    for off in offsets: out+=f"{off:010d} 00000 n \n".encode()
    out+=f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(); return bytes(out)

def main() -> None:
    summary=RESULTS/"summary.csv"
    if not summary.exists(): raise FileNotFoundError(f"run the experiment first: {summary}")
    rows=list(csv.DictReader(summary.open())); FIGURES.mkdir(parents=True,exist_ok=True)
    selected=[r for r in rows if r.get("benchmark")=="static_nonlinear" and r.get("distribution")=="gaussian"]
    names=[r.get("method","") for r in selected]; vals=[float(r.get("test_violation") or 0) for r in selected]
    (FIGURES/"fig_static_test_violation.pdf").write_bytes(_pdf([f"{n}: test violation={v:.6f}" for n,v in zip(names,vals)],vals))
    (FIGURES/"figure_source.txt").write_text("Source: results/summary.csv; no values recomputed.\n")


if __name__ == "__main__":
    main()
