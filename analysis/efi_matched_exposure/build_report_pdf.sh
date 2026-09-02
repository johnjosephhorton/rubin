#!/usr/bin/env bash
# Rebuild writeup/EFI_matched_specification_report.pdf from report_print.html.
#
# report_source.html is the artifact version (no <html>/<head>/<body>; those are added at
# publish time). report_print.html is the same content wrapped in a full document with a
# print stylesheet that forces the light palette, sets A4 margins and keeps tables off page
# breaks. Edit report_source.html for content and mirror the change into report_print.html,
# or edit report_print.html directly if the change is print-only.
#
# The PDF is gitignored by the repo's "*pdf" rule, like every other build artifact. Run this
# to regenerate it; use "git add -f" if you ever want to commit a copy for co-authors.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SRC="$HERE/report_print.html"
PDF="$REPO/writeup/EFI_matched_specification_report.pdf"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ ! -x "$CHROME" ]; then
  echo "Google Chrome not found at $CHROME" >&2
  echo "Point CHROME at any Chromium build that supports --print-to-pdf." >&2
  exit 1
fi

"$CHROME" --headless --disable-gpu --no-pdf-header-footer \
  --virtual-time-budget=20000 \
  --print-to-pdf="$PDF" "file://$SRC" 2>&1 | grep -i "written\|error" || true

if [ -s "$PDF" ]; then
  echo "wrote $PDF"
else
  echo "PDF was not produced" >&2
  exit 1
fi
