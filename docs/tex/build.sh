#!/usr/bin/env bash
# Compile the technical report.  Two passes so the table of contents and the
# cross-references to figures and equations resolve.
#
# Needs xelatex with ctex and a CJK font; on Debian/Ubuntu that is
#   apt install texlive-xetex texlive-lang-chinese fonts-noto-cjk
set -euo pipefail
cd "$(dirname "$0")"
for pass in 1 2; do
  xelatex -interaction=nonstopmode report.tex > "build${pass}.log" 2>&1 \
    || { echo "pass ${pass} failed; see build${pass}.log"; tail -25 "build${pass}.log"; exit 1; }
done
rm -f report.aux report.out report.toc build1.log build2.log
echo "wrote $(pwd)/report.pdf ($(pdfinfo report.pdf | awk '/Pages/{print $2}') pages)"
