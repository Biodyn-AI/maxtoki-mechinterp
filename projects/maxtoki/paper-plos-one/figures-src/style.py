"""Shared figure style for the PLOS ONE manuscript.

Colours follow a validated categorical palette (fixed slot order) and a
single-hue blue ramp for ordinal severity. Text uses ink colours only.
Figures are exported as 300-dpi LZW TIFF within PLOS limits
(width 789-2250 px, height <= 2625 px) plus a PDF preview.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
SRC = os.path.join(PAPER, 'source-data')
OUT = os.path.join(PAPER, 'figures')
PREVIEW = os.path.join(HERE, 'preview')

INK = '#0b0b0b'; INK2 = '#52514e'; MUTED = '#898781'; GRID = '#e1e0d9'; AXIS = '#c3c2b7'; SURFACE = '#ffffff'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
ARM = {'generic': SERIES[0], 'checklist': SERIES[1], 'deployed': SERIES[2], 'original': MUTED}
ARM_LABEL = {'generic': 'Generic review', 'checklist': 'Generic checklist', 'deployed': 'Checklist as deployed', 'original': 'In-session audit'}
EXEC = {'paper': SERIES[0], 'contract': SERIES[1]}
SEV = {'minor': '#86b6ef', 'major': '#2a78d6', 'critical': '#104281'}

# PLOS ONE figure text: Arial, Times or Symbol, 8-12 pt at the printed size.
# All text here is Arial, and no text is smaller than MIN_PT (checked in save()).
MIN_PT = 8
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'legend.title_fontsize': 8, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
    'figure.dpi': 150, 'savefig.dpi': 300, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
})
from matplotlib import font_manager as _fm
_ARIAL = _fm.findfont(_fm.FontProperties(family='Arial'), fallback_to_default=False)  # raises if Arial is missing
assert os.path.basename(_ARIAL).startswith('Arial'), _ARIAL


def check_text(fig):
    """Fail if any visible text is below MIN_PT or does not resolve to an Arial font file."""
    from matplotlib.text import Text
    bad = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        f = os.path.basename(_fm.findfont(t.get_fontproperties(), fallback_to_default=False))
        if t.get_fontsize() < MIN_PT or not f.startswith('Arial'):
            bad.append((t.get_text()[:40], t.get_fontsize(), f))
    assert not bad, f'text below {MIN_PT} pt or not Arial: {bad}'

def panel_label(ax, s):
    ax.text(-0.12, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left', color=INK)

def save(fig, name):
    check_text(fig)
    os.makedirs(OUT, exist_ok=True); os.makedirs(PREVIEW, exist_ok=True)
    fig.savefig(os.path.join(PREVIEW, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(PREVIEW, name + '.png'), dpi=150, bbox_inches='tight')
    tif = os.path.join(OUT, name + '.tif')
    fig.savefig(tif, dpi=300, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    from PIL import Image
    im = Image.open(tif).convert('RGB'); w, h = im.size
    # No resizing: shrinking the image would print the text below MIN_PT. Make the figure narrower instead.
    assert w <= 2250, f'{name}: {w} px wide at 300 dpi; reduce figsize so text keeps its point size'
    im.save(tif, compression='tiff_lzw', dpi=(300, 300))
    w, h = Image.open(tif).size
    assert 789 <= w <= 2250 and h <= 2625, (name, w, h)
    print(f'{name}: {w}x{h}px')
