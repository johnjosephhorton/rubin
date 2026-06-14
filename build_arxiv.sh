#!/usr/bin/env bash
#
# Build the arXiv source bundle + upload tarball for the FULL version of the
# paper (the one in writeup/, which is what peymanshahidi.github.io hosts).
#
#   Usage:  ./build_arxiv.sh          # run from the repo root
#   Output: arxiv/                    # staging dir (git-ignored)
#           arxiv_submission.tar.gz   # <-- upload THIS to arXiv (git-ignored)
#
# Both outputs are regenerable, so they are git-ignored; edit writeup/ and
# re-run this script to refresh them. Submission metadata lives in ARXIV.md.

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/writeup"
DST="$ROOT/arxiv"
TARBALL="$ROOT/arxiv_submission.tar.gz"

# TeX bin dir is not always on PATH in non-interactive shells; find latexmk.
for d in /Library/TeX/texbin /usr/local/texlive/*/bin/*; do
  if [ -x "$d/latexmk" ]; then PATH="$d:$PATH"; break; fi
done
export PATH

echo ">> staging clean bundle in $DST"
rm -rf "$DST" "$TARBALL"
mkdir -p "$DST/plots"

# 1) LaTeX sources at writeup/ root (sections + appendices), the .bib, and the
#    prebuilt .bbl (so arXiv resolves references without a bibtex pass).
cp "$SRC"/*.tex "$DST"/
cp "$SRC"/rubin.bib "$SRC"/main.bbl "$DST"/

# Strip the self-referential "most recent version" website link from the title
# page -- inappropriate on arXiv, which is itself versioned. The writeup/ source
# keeps it (the hosted PDF wants it so stale copies can find the latest).
sed '/for the most recent version/d' "$DST/main.tex" > "$DST/main.tex.tmp" && mv "$DST/main.tex.tmp" "$DST/main.tex"

# 2) Small asset dirs, wholesale.
rsync -a --exclude='.DS_Store' "$SRC"/tables "$DST"/ 2>/dev/null
rsync -a --exclude='.DS_Store' "$SRC"/images "$DST"/ 2>/dev/null

# 3) Only the figures actually cited -- writeup/plots is ~1GB of analysis
#    output, but the paper references ~15MB of it.
grep -rhoE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$DST"/*.tex \
  | sed -E 's/.*\{([^}]+)\}$/\1/' | sort -u | while IFS= read -r p; do
    for cand in "$SRC/$p" "$SRC/$p".png "$SRC/$p".pdf "$SRC/$p".jpg "$SRC/$p".jpeg; do
      if [ -f "$cand" ]; then
        mkdir -p "$DST/$(dirname "$p")"; cp "$cand" "$DST/$(dirname "$p")/"; break
      fi
    done
  done

# 4) TikZ figure sources that are \input from subdirectories.
grep -rhoE '\\input\{plots/[^}]+\}' "$DST"/*.tex \
  | sed -E 's/.*\{([^}]+)\}/\1/' | sort -u | while IFS= read -r p; do
    if [ -f "$SRC/$p" ]; then mkdir -p "$DST/$(dirname "$p")"; cp "$SRC/$p" "$DST/$(dirname "$p")/"; fi
  done

# 5) Bundle the one non-guaranteed package so arXiv's TeX Live compiles it.
CESTY="$(find /usr/local/texlive -name color-edits.sty 2>/dev/null | head -1)"
if [ -n "$CESTY" ]; then cp "$CESTY" "$DST"/; fi

# 6) Verify it compiles standalone -- exactly what arXiv will do.
echo ">> compiling to verify"
( cd "$DST" && latexmk -pdf -interaction=nonstopmode main.tex >/tmp/arxiv_build.log 2>&1 )
if [ ! -f "$DST/main.pdf" ]; then
  echo "!! compile failed -- see /tmp/arxiv_build.log"; exit 1
fi
PAGES="$(grep -oE 'Output written on main\.pdf \([0-9]+ page' "$DST/main.log" | grep -oE '[0-9]+' | head -1)"
echo ">> compiled OK: ${PAGES:-?} pages"

# 7) Strip build artifacts, then tar with source at the archive's top level
#    (arXiv expects main.tex at the root, not nested in a folder).
( cd "$DST" && find . \( -name '*.aux' -o -name '*.log' -o -name '*.out' \
    -o -name '*.fls' -o -name '*.fdb_latexmk' -o -name '*.blg' -o -name '*.toc' \
    -o -name '*.synctex.gz' -o -name 'main.pdf' \) -delete )
( cd "$DST" && COPYFILE_DISABLE=1 tar -czf "$TARBALL" --exclude='.DS_Store' --exclude='._*' ./* )

echo ">> done -> $TARBALL ($(du -h "$TARBALL" | cut -f1))"
