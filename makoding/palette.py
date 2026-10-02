"""Shared colour tokens for DataLab Pro.

One source of truth for the interface theme (``makoding.styling``) and the
charts (``makoding.visualization``), so a chart never looks like it came
from a different product than the page it is embedded in.

This module has no dependencies, so both a Streamlit layer and a
framework-independent plotting layer can import it.
"""

from __future__ import annotations

__all__ = [
    "INK",
    "NAVY",
    "TEAL",
    "TEAL_DARK",
    "TEAL_LIGHT",
    "PAPER",
    "WHITE",
    "MUTED",
    "BORDER",
    "ACCENT",
    "CATEGORICAL",
]

# Interface tokens.
INK = "#17212B"
NAVY = "#173B4D"
TEAL = "#176B87"
TEAL_DARK = "#0F5268"
TEAL_LIGHT = "#E8F2F5"
PAPER = "#F7F7F4"
WHITE = "#FFFFFF"
MUTED = "#68737D"
BORDER = "#D9E0E3"

# Warm counterpoint to the teal primary: reference lines, fences, forecasts,
# the "bad" side of a diverging scale.
ACCENT = "#B5533C"

# Ten distinguishable colours for discrete groups (clusters, hue categories).
# The first is the primary teal so a single-group chart matches the theme.
CATEGORICAL: tuple[str, ...] = (
    "#176B87",  # teal
    "#D98E32",  # amber
    "#5B8E7D",  # sage
    "#8C5E9E",  # plum
    "#B5533C",  # terracotta
    "#4F6D7A",  # slate
    "#A8B545",  # olive
    "#C9788F",  # rose
    "#B8A27A",  # sand
    "#7B858C",  # grey
)