"""
Shared dark-theme plotting style, consistent with the rest of the ML portfolio.

Import FG/BG/PANEL/GRID/C0/C1/TEAL/GOLD and call style() once at the top of a
notebook, before any plotting -- every notebook in this project uses the same
palette so the whole portfolio reads as one consistent body of work rather than
a pile of differently-styled one-off notebooks.
"""

import matplotlib.pyplot as plt

BG = "#0d1117"
PANEL = "#161b22"
FG = "#e6edf3"
GRID = "#30363d"
C0 = "#58a6ff"   # primary series
C1 = "#f78166"   # secondary series
TEAL = "#39d3bb"
GOLD = "#e3b341"

PALETTE = [C0, C1, TEAL, GOLD, "#bc8cff", "#7ee787"]


def style() -> None:
    """Apply the portfolio-wide dark theme to matplotlib's rcParams."""
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": FG,
        "axes.titlecolor": FG,
        "text.color": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "grid.color": GRID,
        "grid.alpha": 0.4,
        "axes.grid": True,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "legend.labelcolor": FG,
        "font.size": 11,
    })
