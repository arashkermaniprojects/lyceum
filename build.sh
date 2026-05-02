#!/usr/bin/env bash
# Build SeVim article PDF.
# Usage: ./build.sh  (run from the directory that contains main.tex)
set -euo pipefail
cd "$(dirname "$0")"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error main.tex
else
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
fi

echo "Built: $(pwd)/main.pdf"
