#!/bin/bash
# Build draft_mert.
#
# The appendix uses `bibunits`, so bibtex has to be run once on 0_main.aux (main
# reference list) and once on each bu*.aux (Online Appendix reference list).
# latexmk does not pick the bu*.aux files up on its own, hence this script.

set -e
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode -file-line-error 0_main.tex
bibtex 0_main || true
for f in bu*.aux; do
    [ -e "$f" ] || continue
    bibtex "${f%.aux}" || true
done
pdflatex -interaction=nonstopmode -file-line-error 0_main.tex
pdflatex -interaction=nonstopmode -file-line-error 0_main.tex
pdflatex -interaction=nonstopmode -file-line-error 0_main.tex

echo
echo "Built: $(pwd)/0_main.pdf"
echo "LaTeX errors:            $(grep -cE '^(! |[^ ]*\.tex:[0-9]+: )' 0_main.log)"
echo "undefined refs/citations: $(grep -cE 'Warning:.*(undefined|Citation)' 0_main.log)"
echo "float placement warnings: $(grep -cE 'Float too large|Too many unprocessed floats|h. float specifier' 0_main.log)" 
