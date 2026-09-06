"""Shared matplotlib styling for the figures.

Colours come from a validated categorical palette: assigned in fixed slot
order, never cycled, and checked for colour-vision-deficiency separation
(worst adjacent pair dE 9.1 protan, normal-vision floor 19.6) rather than
chosen by eye. Three of the six slots sit below 3:1 contrast on the light
surface, so every series also carries a direct label at the right edge --
identity is never colour alone.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: Categorical slots, in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

SEQUENTIAL = "#2a78d6"
CRITICAL = "#d03b3b"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 10.5,
            "axes.titleweight": "medium",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
        }
    )


def declutter(positions: np.ndarray, min_gap: float) -> np.ndarray:
    """Nudge label positions apart while preserving their order.

    Repeatedly pushes overlapping neighbours apart and re-centres, so a
    cluster of near-equal end-of-line values stays readable without any
    label jumping past another.
    """
    order = np.argsort(positions)
    values = positions[order].astype(float).copy()
    for _ in range(200):
        gaps = np.diff(values)
        tight = gaps < min_gap
        if not tight.any():
            break
        for i in np.flatnonzero(tight):
            shortfall = (min_gap - gaps[i]) / 2.0
            values[i] -= shortfall
            values[i + 1] += shortfall
    out = np.empty_like(values)
    out[order] = values
    return out
