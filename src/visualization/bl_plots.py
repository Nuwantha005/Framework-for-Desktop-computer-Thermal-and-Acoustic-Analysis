"""
Boundary layer visualization — line plots, envelope plots, profile comparisons.

All plotting functions work with the two-sided
:class:`BoundaryLayerCaseResult` returned by :class:`BoundaryLayerRunner`.

* **Line plots**: quantity vs arc-length for one path (upper or lower).
* **Two-sided line plots**: upper and lower columns side-by-side.
* **Envelope plots**: BL quantity distribution wrapped around the full body.
* **Comparison figure**: two-sided lines + envelope + Ue reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from visualization.surface_envelope import (
    compute_outward_normals,
    plot_surface_envelope,
    plot_surface_envelope_comparison,
)


# -------------------------------------------------------------------------
# Colour palette
# -------------------------------------------------------------------------

_PROFILE_COLORS: Dict[str, str] = {
    "Blasius": "#1f77b4",
    "Pohlhausen": "#ff7f0e",
    "Falkner-Skan": "#2ca02c",
    "Thwaites": "#d62728",
    "Power-law 1/7": "#9467bd",
}


def _color_for(name: str, idx: int = 0) -> str:
    """Deterministic colour for a profile name."""
    if name in _PROFILE_COLORS:
        return _PROFILE_COLORS[name]
    tab = plt.cm.tab10.colors  # type: ignore[attr-defined]
    return tab[idx % len(tab)]


_LABELS: Dict[str, str] = {
    "cf": r"$c_f$",
    "delta_star": r"$\delta^*$ [m]",
    "theta": r"$\theta$ [m]",
    "H": r"$H = \delta^*/\theta$",
    "Re_theta": r"$Re_\theta$",
}


# -------------------------------------------------------------------------
# Single-path line plot
# -------------------------------------------------------------------------

def plot_bl_line(
    path_result,  # BoundaryLayerPathResult
    quantity: str = "cf",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot a BL quantity vs arc length for one side, all profiles overlaid.

    Args:
        path_result: :class:`BoundaryLayerPathResult` (one side).
        quantity: One of ``"cf"``, ``"delta_star"``, ``"theta"``, ``"H"``,
            ``"Re_theta"``.
        ax: Existing axes (creates new figure if *None*).
        title: Plot title.
        output_path: If given, save figure to this path.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    for idx, (name, res) in enumerate(path_result.results.items()):
        vals = getattr(res, quantity)
        color = _color_for(name, idx)
        ax.plot(res.s, vals, color=color, linewidth=1.5, label=name)

        # Mark transition if available
        tr = path_result.transitions.get(name)
        if tr is not None and tr.transition_s is not None:
            s_tr = tr.transition_s
            vi = np.searchsorted(res.s, s_tr)
            vi = min(vi, len(vals) - 1)
            ax.axvline(s_tr, color=color, ls="--", alpha=0.4)
            ax.plot(s_tr, vals[vi], "o", color=color, ms=6)

    ax.set_xlabel("Arc length $s$ [m]")
    ax.set_ylabel(_LABELS.get(quantity, quantity))
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_bl_lines_multi(
    path_result,
    quantities: Optional[List[str]] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, List[Axes]]:
    """
    Multi-panel line plots of several BL quantities for one side.

    Args:
        path_result: :class:`BoundaryLayerPathResult` (one side).
        quantities: Defaults to ``["cf", "delta_star", "theta", "H"]``.
        title: Super-title.
        output_path: Save path.

    Returns:
        ``(fig, axes_list)`` tuple.
    """
    if quantities is None:
        quantities = ["cf", "delta_star", "theta", "H"]
    n = len(quantities)

    fig, axes = plt.subplots(n, 1, figsize=(8, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, q in zip(axes, quantities):
        plot_bl_line(path_result, quantity=q, ax=ax)

    axes[-1].set_xlabel("Arc length $s$ [m]")
    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, list(axes)


# -------------------------------------------------------------------------
# Two-sided line plots (upper | lower)
# -------------------------------------------------------------------------

def plot_bl_two_sides(
    case_result,  # BoundaryLayerCaseResult
    quantities: Optional[List[str]] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, NDArray]:
    """
    Side-by-side line plots: upper (left column) and lower (right column).

    Layout::

        ┌──────────────┬──────────────┐
        │ Upper — cf   │ Lower — cf   │
        ├──────────────┼──────────────┤
        │ Upper — δ*   │ Lower — δ*   │
        ├──────────────┼──────────────┤
        │ Upper — θ    │ Lower — θ    │
        ├──────────────┼──────────────┤
        │ Upper — H    │ Lower — H    │
        └──────────────┴──────────────┘

    Args:
        case_result: :class:`BoundaryLayerCaseResult`.
        quantities: Defaults to ``["cf", "delta_star", "theta", "H"]``.
        title: Super-title.
        output_path: Save path.

    Returns:
        ``(fig, axes)`` where axes is ``(nq, 2)`` array.
    """
    if quantities is None:
        quantities = ["cf", "delta_star", "theta", "H"]
    nq = len(quantities)

    fig, axes = plt.subplots(nq, 2, figsize=(14, 3.2 * nq), sharex="col")
    if nq == 1:
        axes = axes.reshape(1, 2)

    for i, q in enumerate(quantities):
        plot_bl_line(case_result.upper, quantity=q, ax=axes[i, 0])
        plot_bl_line(case_result.lower, quantity=q, ax=axes[i, 1])
        # Only left column gets y-label
        axes[i, 1].set_ylabel("")

    axes[0, 0].set_title("Upper side", fontsize=12)
    axes[0, 1].set_title("Lower side", fontsize=12)
    axes[-1, 0].set_xlabel("Arc length $s$ [m]")
    axes[-1, 1].set_xlabel("Arc length $s$ [m]")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, axes


# -------------------------------------------------------------------------
# Envelope plots (full body)
# -------------------------------------------------------------------------

def plot_bl_envelope(
    case_result,
    quantity: str = "cf",
    profile_name: Optional[str] = None,
    scale: float = 0.15,
    colormap: Optional[str] = "magma",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """
    Envelope plot of a BL quantity on the full body for one profile.

    Panels with *NaN* (stagnation skip, separated region) are shown
    with zero displacement.

    Args:
        case_result: :class:`BoundaryLayerCaseResult`.
        quantity: ``"cf"``, ``"delta_star"``, ``"theta"``, ``"H"``.
        profile_name: Which profile to plot.  Defaults to the first.
        scale: Envelope displacement scale factor.
        colormap: Matplotlib colormap name, or *None* for single colour.
        ax: Existing axes.
        title: Plot title.
        output_path: Save path.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if profile_name is None:
        profile_name = case_result.profile_names[0]

    vals = case_result.full_body_quantity(quantity, profile_name)
    x = case_result.surface_x
    y = case_result.surface_y

    # Replace NaN with 0 for envelope display
    vals_plot = np.where(np.isnan(vals), 0.0, vals)

    fig, ax = plot_surface_envelope(
        x, y, vals_plot,
        scale=scale,
        quantity_name=f"{quantity} ({profile_name})",
        colormap=colormap,
        ax=ax,
        title=title or f"{quantity} envelope — {profile_name}",
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_bl_envelope_comparison(
    case_result,
    quantity: str = "cf",
    scale: float = 0.15,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """
    Overlay envelope plots of multiple profiles on the full body.

    Args:
        case_result: :class:`BoundaryLayerCaseResult`.
        quantity: BL quantity to plot.
        scale: Envelope scale.
        ax: Existing axes (creates new figure if *None*).
        title: Plot title.
        output_path: Save path.

    Returns:
        ``(fig, ax)`` tuple.
    """
    x_list, y_list, values_list, labels = [], [], [], []

    for name in case_result.profile_names:
        vals = case_result.full_body_quantity(quantity, name)
        vals_plot = np.where(np.isnan(vals), 0.0, vals)
        x_list.append(case_result.surface_x)
        y_list.append(case_result.surface_y)
        values_list.append(vals_plot)
        labels.append(name)

    colors = [_color_for(name, i) for i, name in enumerate(labels)]

    fig, ax = plot_surface_envelope_comparison(
        x_list, y_list, values_list, labels,
        scale=scale,
        quantity_name=quantity,
        colors=colors,
        ax=ax,
        title=title or f"{quantity} — profile comparison",
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


# -------------------------------------------------------------------------
# Full comparison figure
# -------------------------------------------------------------------------

def plot_bl_comparison(
    case_result,
    quantities: Optional[List[str]] = None,
    envelope_quantity: str = "cf",
    envelope_scale: float = 0.15,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Figure:
    """
    Full comparison figure: two-sided lines + envelope + Ue.

    Layout::

        ┌──────────┬──────────┬──────────────┐
        │ U cf     │ L cf     │              │
        ├──────────┼──────────┤  cf envelope │
        │ U δ*     │ L δ*     │  (full body) │
        ├──────────┼──────────┤              │
        │ U θ      │ L θ      ├──────────────┤
        ├──────────┼──────────┤ Ue upper     │
        │ U H      │ L H      │ Ue lower     │
        └──────────┴──────────┴──────────────┘

    Args:
        case_result: :class:`BoundaryLayerCaseResult`.
        quantities: Line-plot quantities (default ``["cf","delta_star","theta","H"]``).
        envelope_quantity: Quantity for the envelope plot.
        envelope_scale: Envelope displacement scale.
        title: Super-title.
        output_path: Save path.
        show: Call ``plt.show()``.

    Returns:
        Figure.
    """
    if quantities is None:
        quantities = ["cf", "delta_star", "theta", "H"]

    n_rows = max(len(quantities), 2)

    fig = plt.figure(figsize=(18, 3.5 * n_rows))
    gs = fig.add_gridspec(
        n_rows, 3, width_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.30,
    )

    # Left two columns: upper | lower line plots
    for i, q in enumerate(quantities):
        ax_u = fig.add_subplot(gs[i, 0])
        plot_bl_line(case_result.upper, quantity=q, ax=ax_u)
        if i == 0:
            ax_u.set_title("Upper side", fontsize=11)
        if i < len(quantities) - 1:
            ax_u.set_xlabel("")

        ax_l = fig.add_subplot(gs[i, 1])
        plot_bl_line(case_result.lower, quantity=q, ax=ax_l)
        ax_l.set_ylabel("")
        if i == 0:
            ax_l.set_title("Lower side", fontsize=11)
        if i < len(quantities) - 1:
            ax_l.set_xlabel("")

    # Right top: envelope comparison (spans most of right column)
    ax_env = fig.add_subplot(gs[: n_rows - 1, 2])
    plot_bl_envelope_comparison(
        case_result,
        quantity=envelope_quantity,
        scale=envelope_scale,
        ax=ax_env,
        title=f"{envelope_quantity} envelope",
    )

    # Right bottom: Ue(s) for both sides
    ax_ue = fig.add_subplot(gs[n_rows - 1, 2])
    ax_ue.plot(
        case_result.upper.s, case_result.upper.Ue,
        "b-", lw=1.5, label="Upper $U_e$",
    )
    ax_ue.plot(
        case_result.lower.s, case_result.lower.Ue,
        "r-", lw=1.5, label="Lower $U_e$",
    )
    ax_ue.set_xlabel("Arc length $s$ [m]")
    ax_ue.set_ylabel("$U_e$ [m/s]")
    ax_ue.legend(fontsize=9)
    ax_ue.grid(True, alpha=0.3)
    ax_ue.set_title("Edge velocity (panel method)")

    if title:
        fig.suptitle(title, fontsize=15, y=1.02)

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
