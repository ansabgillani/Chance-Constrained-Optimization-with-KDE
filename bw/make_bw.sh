#!/usr/bin/env bash
# Black-and-white print edition: grayscale copies of every figure the thesis
# includes, and of the built thesis PDF. Run after building Masterarbeit.pdf:
#   bash bw/make_bw.sh
set -euo pipefail
cd "$(dirname "$0")/.."

gray() {
  mkdir -p "$(dirname "$2")"
  gs -q -o "$2" -sDEVICE=pdfwrite -sColorConversionStrategy=Gray \
     -dProcessColorModel=/DeviceGray "$1"
}

count=0
for tex in kap*/kapitel*.tex; do
  chap=$(dirname "$tex")
  for f in $(grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]*\}' "$tex" | sed -E 's/.*\{([^}]*)\}$/\1/'); do
    case $f in */*) ;; *) f="$chap/figures/$f" ;; esac
    f="${f%.pdf}.pdf"
    [ -f "$f" ] || { echo "missing figure: $f" >&2; exit 1; }
    gray "$f" "bw/$f"
    count=$((count + 1))
  done
done
[ -f Masterarbeit.pdf ] || { echo "build Masterarbeit.pdf first" >&2; exit 1; }
gray Masterarbeit.pdf bw/Masterarbeit_bw.pdf
echo "wrote $count figures and bw/Masterarbeit_bw.pdf"
