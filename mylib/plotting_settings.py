"""
plotstyle.py  --  one place for every plotting decision in the CuGa paper.

Usage
-----
    import plotstyle as ps
    ps.apply("paper")                       # or "slides"

    fig, ax = plt.subplots(figsize=ps.size("single"))
    ax.plot(x, y, color=ps.PGA_COLOR[15], marker=ps.N_MARKER[700])
    im = ax.pcolormesh(X, Y, Z, **ps.cmap_kw("coverage"))
    ps.save(fig, "figures/coverage_map")

Design rules collected from earlier work
----------------------------------------
- No titles. Captions carry the description.
- Top/right spines hidden, no grid, legend without frame.
- lw 1.8, marker size 5, white marker edge (mew 1).
- Units in parentheses, e.g. "Ga content (at. %)", "T (K)".
- Save as PDF and PNG at 300 dpi.
- Figures are drawn at their final printed width, so fonts in the PDF
  are exactly the fonts in the paper. Always include them in LaTeX with
  width=\\columnwidth or width=\\textwidth, never with a scale factor.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize



# ═════════════════════════════════════════════════════════════════════════════
# 1. Figure sizes (inches)
# ═════════════════════════════════════════════════════════════════════════════
# RSC two-column layout (ESI terminology suggests an RSC journal).
# Single column 8.3 cm, double column 17.1 cm.
# For ACS use 3.25 and 7.0 instead.
SINGLE_W = 8.3 / 2.54      # 3.27 in, one column
DOUBLE_W = 17.1 / 2.54     # 6.73 in, full text width
HALF_GAP = 0.3 / 2.54      # space between two half panels in one column
HALF_W   = (SINGLE_W - HALF_GAP) / 2   # ~1.58 in, two side by side in one column
SLIDE_W  = 8.0

ASPECT = 0.72              # height / width of one panel, ~ the old (5.5, 3.8)
_DEFAULT_ASPECT = {"single": ASPECT, "double": ASPECT, "slide": ASPECT,
                   "half": 0.85}   # narrow panels need to be squarer to stay readable


def size(kind: str = "single", nrows: int = 1, ncols: int = 1,
         aspect: float | None = None) -> tuple[float, float]:
    """
    Figure size for a given layout.

    kind   "single"  one column wide
           "double"  full text width
           "half"    half a column, two side by side (subfigure at 0.48\\columnwidth)
           "slide"   presentation figure
    Height scales with nrows/ncols so every panel keeps the same aspect.
    """
    width = {"single": SINGLE_W, "double": DOUBLE_W,
             "half": HALF_W, "slide": SLIDE_W}[kind]
    if aspect is None:
        aspect = _DEFAULT_ASPECT[kind]
    panel_w = width / ncols
    return width, panel_w * aspect * nrows


def compact(ax, nx: int = 3, ny: int = 4) -> None:
    """Fewer ticks for narrow panels. Font sizes stay those of the global style."""
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nx, min_n_ticks=2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ny, min_n_ticks=2))
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4), useMathText=True)

# ═════════════════════════════════════════════════════════════════════════════
# 2. Fonts, lines, axes  (rcParams)
# ═════════════════════════════════════════════════════════════════════════════
# "paper" sizes are for figures drawn at true print width.
# They match the old look (12 pt labels on a 5.5 in figure, scaled down to
# column width, is ~7-8 pt in print).
_FONTS = {
    "paper":  dict(label=8,  tick=7,  legend=7,  annot=7),
    "slides": dict(label=24, tick=22, legend=20, annot=20),
}

LW       = 1.8
MS       = 5
MEW      = 1.0
DPI      = 300


def apply(context: str = "paper") -> None:
    """Set global rcParams. Call once at the top of a notebook or script."""
    f = _FONTS[context]
    lw_scale = 1.0 if context == "slides" else 0.75   # thinner lines at print size

    mpl.rcParams.update({
        # fonts
        "font.size":             f["label"],
        "axes.labelsize":        f["label"],
        "xtick.labelsize":       f["tick"],
        "ytick.labelsize":       f["tick"],
        "legend.fontsize":       f["legend"],
        "legend.title_fontsize": f["legend"],
        "mathtext.default":      "regular",

        # lines and markers
        "lines.linewidth":       LW * lw_scale,
        "lines.markersize":      MS * (1.0 if context == "slides" else 0.8),
        "lines.markeredgecolor": "white",
        "lines.markeredgewidth": MEW * lw_scale,

        # axes
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.grid":             False,
        "axes.titlesize":        f["label"],
        "axes.linewidth":        0.8 * lw_scale,
        "axes.prop_cycle":       mpl.cycler(color=SERIES_COLORS),
        "xtick.major.width":     0.8 * lw_scale,
        "ytick.major.width":     0.8 * lw_scale,

        # legend
        "legend.frameon":        False,

        # images / colormaps
        "image.cmap":            "viridis",

        # output
        "savefig.dpi":           DPI,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.02,
        "pdf.fonttype":          42,     # editable text in Illustrator/Inkscape
        "ps.fonttype":           42,
        "figure.constrained_layout.use": True,
    })


# ═════════════════════════════════════════════════════════════════════════════
# 3. Colors and markers
# ═════════════════════════════════════════════════════════════════════════════
# Generic series (used when nothing more specific applies)
SERIES_COLORS  = ["#2a78d6", "#eb6834", "#1baf7a", "#9467bd", "#8c564b"]
SERIES_MARKERS = ["o", "s", "^", "D", "v"]
REFERENCE_COLOR = "#333333"          # pure Cu / reference data

# Species in line plots (diffusion, Arrhenius, ...)
SPECIES_COLOR  = {"Cu": "#3A78C3", "Ga": "#E8833A", "O": "#C44E52"}
SPECIES_MARKER = {"Cu": "s",       "Ga": "o",       "O": "^"}

# Atom colors matching the OVITO renders (structure figures, legends in LaTeX)
ELEMENT_COLOR  = {"Cu": "#C57E32", "Ga": "#C18E8E", "O": "#FF0D0D"}

# Simulation parameters -> fixed visual identity, identical in every figure
NS    = [500, 700, 1500, 3000]
PGAS  = [5, 12, 15, 19, 22, 24]
NOS   = [0, 50, 75, 100, 125, 150]

_pga_cmap = plt.get_cmap("viridis")
PGA_COLOR = {p: _pga_cmap(i / (len(PGAS) - 1) * 0.9) for i, p in enumerate(PGAS)}
N_MARKER  = dict(zip(NS, SERIES_MARKERS))
N_COLOR   = dict(zip(NS, SERIES_COLORS))

_no_cmap = plt.get_cmap("tab10")
NO_COLOR = {no: _no_cmap(i) for i, no in enumerate(NOS)}


def no_cmap_norm():
    """Discrete colormap + norm for categorical O/Ga maps (phase diagrams)."""
    cmap = ListedColormap([NO_COLOR[no] for no in NOS])
    norm = BoundaryNorm([i - 0.5 for i in range(len(NOS) + 1)], cmap.N)
    return cmap, norm


# ═════════════════════════════════════════════════════════════════════════════
# 4. Fixed color ranges per quantity
# ═════════════════════════════════════════════════════════════════════════════
# Every map of the same quantity uses the same cmap and the same vmin/vmax,
# so colorbars are comparable across figures.
# Fill in the None entries once, e.g. with global_clim() over ALL configs.
CLIM = {
    # quantity         (vmin,  vmax,  cmap)
    "coverage":        (0.0,   None,  "viridis"),
    "shadow_coverage": (0.0,   None,  "viridis"),
    "f_ox":            (0.0,   None,  "viridis"),
    "Ea":              (None,  None,  "Greens"),
    "depth_diff":      (None,  None,  "viridis"),
    "dCN_GaCu":        (None,  None,  "viridis"),
    "asphericity":     (None,  None,  "viridis"),
    "warren_cowley":   (None,  None,  "viridis"),
}


def global_clim(values, pad: float = 0.0, symmetric: bool = False):
    """
    Suggest (vmin, vmax) over the full data set. Run once, then hardcode in CLIM.
    symmetric=True centres the range on zero (for signed differences).
    """
    import numpy as np
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    lo, hi = v.min(), v.max()
    if symmetric:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    span = hi - lo
    return lo - pad * span, hi + pad * span


def cmap_kw(quantity: str) -> dict:
    """
    kwargs for pcolormesh / imshow / scatter, e.g.
        ax.pcolormesh(X, Y, Z, **ps.cmap_kw("coverage"))
    Warns if the range is not fixed yet, so autoscaled figures don't slip
    into the paper unnoticed.
    """
    vmin, vmax, cmap = CLIM[quantity]
    if vmin is None or vmax is None:
        warnings.warn(f"CLIM['{quantity}'] is not fully fixed, colorbar will autoscale",
                      stacklevel=2)
    return dict(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))


# ═════════════════════════════════════════════════════════════════════════════
# 5. Labels
# ═════════════════════════════════════════════════════════════════════════════
LABEL = {
    "pga":      "Ga content (at. %)",
    "T":        "T (K)",
    "invT":     r"1/T (K$^{-1}$)",
    "N":        "N (atoms)",
    "no":       "O/Ga (%)",
    "dmu_O":    r"$\Delta\mu_O$ (eV)",
    "dG":       r"$\Delta G$ (eV)",
    "E_atom":   "E (eV/atom)",
    "Ea":       r"$E_a$ (eV)",
    "lnD":      r"ln D",
    "coverage": "coverage",
    "f_ox":     "oxide volume fraction",
}


# ═════════════════════════════════════════════════════════════════════════════
# 6. Saving
# ═════════════════════════════════════════════════════════════════════════════
def save(fig, path, formats=("pdf", "png"), transparent: bool = False) -> None:
    """Save to every format at DPI. path without extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(path.with_suffix(f".{ext}"), transparent=transparent)