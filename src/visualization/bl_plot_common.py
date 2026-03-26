"""Shared helpers for boundary-layer plotting."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

_PROFILE_COLORS: Dict[str, str] = {
    "Blasius": "#1f77b4",
    "Pohlhausen": "#ff7f0e",
    "Falkner-Skan": "#2ca02c",
    "Thwaites": "#d62728",
    "Power-law 1/7": "#9467bd",
}

_LABELS: Dict[str, str] = {
    "cf": r"$c_f$",
    "delta_star": r"$\delta^*$ [m]",
    "theta": r"$\theta$ [m]",
    "H": r"$H = \delta^*/\theta$",
    "Re_theta": r"$Re_\theta$",
}


def _color_for(name: str, idx: int = 0) -> str:
    """Deterministic color for a profile name."""
    if name in _PROFILE_COLORS:
        return _PROFILE_COLORS[name]
    tab = plt.cm.tab10.colors  # type: ignore[attr-defined]
    return tab[idx % len(tab)]


def _cell_edges(centers: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute cell-edge coordinates from centers for pcolormesh."""
    n_centers = len(centers)
    if n_centers == 0:
        return np.empty(1, dtype=np.float64)
    if n_centers == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=np.float64)
    edges = np.empty(n_centers + 1, dtype=np.float64)
    mid = 0.5 * (centers[:-1] + centers[1:])
    edges[1:-1] = mid
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges
