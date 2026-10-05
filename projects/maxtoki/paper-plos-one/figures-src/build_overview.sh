#!/bin/bash
# Build the overview diagram (TikZ) as a PLOS-compliant TIFF.
# Needs xelatex (fontspec, Arial). Rendered at 300 dpi and never resized, so 8 pt text prints at 8 pt.
set -e
cd "$(dirname "$0")"
W=$(mktemp -d)
cp fig_overview.tex "$W/"
( cd "$W" && xelatex -interaction=nonstopmode fig_overview.tex >/dev/null && pdftoppm -r 300 -png -singlefile fig_overview.pdf fig_overview )
cp "$W/fig_overview.pdf" preview/Fig_overview.pdf; cp "$W/fig_overview.png" preview/Fig_overview.png
../../.venv/bin/python - <<'PY'
from PIL import Image
im = Image.open('preview/Fig_overview.png').convert('RGB'); w, h = im.size
assert 789 <= w <= 2250 and h <= 2625, ('Fig_overview', w, h)  # no resizing: it would shrink the 8 pt text
im.save('../figures/Fig_overview.tif', compression='tiff_lzw', dpi=(300, 300)); print('Fig_overview', im.size)
PY
