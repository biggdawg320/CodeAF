"""Shared figure style for pareto-crewing.tex.

Sizes are the paper's real text width (A4, 2.2 cm margins: 6.5 in) and half width (3.15 in, the 0.49 textwidth
minipage); figures are saved at exactly these sizes (constrained layout, no tight-bbox resizing) and included in
LaTeX at natural size. Text is Liberation Serif (a TrueType Times clone matching mathptmx, embedded cleanly as Type 42) at 7-8 pt, math in STIX.
"""
import matplotlib as mpl
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

TEXT_W = 6.5
COL_W = 3.15

# Okabe-Ito based roles, fixed per entity across all figures
PAL = dict(
    ours="#0072B2",          # our router / crew-level routing / sequential policy
    ours_light="#56B4E9",    # a second variant of our method
    verify="#D55E00",        # crews with an upgraded verifier (strong checker)
    verify_light="#E69F00",  # crews with a low-cost second verifier
    alt="#CC79A7",           # other off-diagonal crews
    diag_low="#4D4D4D",      # diagonal (uniform) low-cost crew / single-configuration baselines
    diag_front="#9A9A9A",    # diagonal (uniform) frontier crew
    points="#C8C8C8",        # background points (single configurations)
    ref="#000000",           # reference lines and theory
    context="#8C8C8C",
)


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "axes.titlelocation": "left", "axes.titlepad": 4,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "lines.linewidth": 1.4, "lines.markersize": 3.8, "lines.markeredgewidth": 0.5,
        "errorbar.capsize": 2,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
    })


def _money(v):
    if v >= 1:
        return f"\\${v:g}"
    return f"\\${v:.2f}".rstrip("0").rstrip(".") if v >= 0.1 else f"\\${v:.2f}"


def log_axis(ax, axis, ticks, fmt="money"):
    """Log scale with major ticks exactly at `ticks`, clean labels, unlabeled minor ticks."""
    set_scale = ax.set_xscale if axis == "x" else ax.set_yscale
    a = ax.xaxis if axis == "x" else ax.yaxis
    set_scale("log")
    a.set_major_locator(FixedLocator(ticks))
    if fmt == "money":
        a.set_major_formatter(FuncFormatter(lambda v, _: _money(v)))
    elif fmt == "times":
        a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}$\\times$"))
    elif fmt == "pct":
        a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    else:
        a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def title(ax, letter, text=""):
    ax.set_title(f"$\\mathbf{{({letter})}}$ {text}".rstrip(), loc="left")


def refline(ax, value, axis="y", label=None, **kw):
    style = dict(color=PAL["ref"], lw=0.7, ls=(0, (4, 2)))
    style.update(kw)
    (ax.axhline if axis == "y" else ax.axvline)(value, **style)
