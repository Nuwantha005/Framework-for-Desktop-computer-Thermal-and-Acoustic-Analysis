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
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

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


# =========================================================================
#  Phase 5 — velocity-field visualizations (require BLFieldData)
# =========================================================================


def plot_bl_velocity_contour(
    field,  # BLFieldData
    ax: Optional[Axes] = None,
    cmap: str = "viridis",
    show_delta: bool = True,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    n_levels: int = 64,
) -> Tuple[Figure, Axes]:
    """s-y velocity contour plot for one BL path.

    X-axis is arc-length *s*, y-axis is wall-normal distance *y*.
    Velocity magnitude is shown as a filled contour / pcolormesh.
    Optionally overlays δ(s) as a white dashed line.

    Args:
        field: :class:`BLFieldData` from :func:`reconstruct_bl_field`.
        ax: Existing axes (creates new figure if *None*).
        cmap: Matplotlib colormap name.
        show_delta: Draw δ(s) curve on top.
        title: Plot title.
        output_path: Save path.
        n_levels: Number of contour levels.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    # field.s (M,), field.y (M, Ny), field.u (M, Ny)
    M, Ny = field.u.shape

    if M < 2:
        ax.text(
            0.5, 0.5,
            f"Insufficient data ({M} station{'s' if M != 1 else ''})",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="0.4",
        )
        if title:
            ax.set_title(title, fontsize=11)
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    # Build 2-D coordinate grids for pcolormesh
    # Extend s to cell edges for pcolormesh (M+1 edges)
    s_edges = _cell_edges(field.s)  # (M+1,)
    # For each station the y grid is the same length Ny; build edges (Ny+1)
    # We use per-station y ranges, so we construct a full (M+1, Ny+1) mesh
    y_edge_grid = np.zeros((M + 1, Ny + 1), dtype=np.float64)
    for i in range(M):
        y_edge_grid[i] = _cell_edges(field.y[i])
    y_edge_grid[M] = y_edge_grid[M - 1]  # repeat last row

    S_grid = np.broadcast_to(s_edges[:, np.newaxis], (M + 1, Ny + 1))

    pcm = ax.pcolormesh(
        S_grid, y_edge_grid, field.u,
        cmap=cmap, shading="flat", rasterized=True,
    )
    fig.colorbar(pcm, ax=ax, label=r"$u$ [m/s]", shrink=0.85, pad=0.02)

    if show_delta:
        ax.plot(field.s, field.delta, "w--", lw=1.5, label=r"$\delta(s)$")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    ax.set_xlabel("Arc length $s$ [m]")
    ax.set_ylabel("Wall-normal $y$ [m]")
    ax.set_xlim(field.s[0], field.s[-1])
    ax.set_ylim(0, None)

    if title:
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(
            f"BL velocity contour — {field.profile_name}", fontsize=11,
        )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_bl_velocity_contour_normalized(
    field,  # BLFieldData
    ax: Optional[Axes] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """Normalized s-(y/δ) velocity contour for one BL path.

    Same as :func:`plot_bl_velocity_contour` but with the y-axis
    replaced by y/δ(s), producing a uniform rectangle.  This makes
    the thin parts of the BL easier to inspect.

    Args:
        field: :class:`BLFieldData`.
        ax: Existing axes.
        cmap: Colormap.
        title: Plot title.
        output_path: Save path.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    M, Ny = field.u.shape

    if M < 2:
        ax.text(
            0.5, 0.5,
            f"Insufficient data ({M} station{'s' if M != 1 else ''})",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="0.4",
        )
        if title:
            ax.set_title(title, fontsize=11)
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    # Normalise y by delta at each station → eta in [0, 1] (or up to extend)
    eta = np.zeros_like(field.y)
    for i in range(M):
        if field.delta[i] > 0:
            eta[i] = field.y[i] / field.delta[i]
        else:
            eta[i] = np.linspace(0.0, 1.0, Ny)

    # Build edge grids
    s_edges = _cell_edges(field.s)
    eta_edge_grid = np.zeros((M + 1, Ny + 1), dtype=np.float64)
    for i in range(M):
        eta_edge_grid[i] = _cell_edges(eta[i])
    eta_edge_grid[M] = eta_edge_grid[M - 1]

    S_grid = np.broadcast_to(s_edges[:, np.newaxis], (M + 1, Ny + 1))

    pcm = ax.pcolormesh(
        S_grid, eta_edge_grid, field.u,
        cmap=cmap, shading="flat", rasterized=True,
    )
    fig.colorbar(pcm, ax=ax, label=r"$u$ [m/s]", shrink=0.85, pad=0.02)

    # Horizontal line at η = 1 (BL edge)
    ax.axhline(1.0, color="w", ls="--", lw=1.0, alpha=0.7, label=r"$\delta$")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    ax.set_xlabel("Arc length $s$ [m]")
    ax.set_ylabel(r"$y / \delta$")
    ax.set_xlim(field.s[0], field.s[-1])
    ax.set_ylim(0, None)

    if title:
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(
            f"Normalized BL velocity — {field.profile_name}", fontsize=11,
        )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_bl_velocity_envelope(
    field,  # BLFieldData
    surface_x: NDArray[np.float64],
    surface_y: NDArray[np.float64],
    panel_indices: List[int],
    scale: float = 0.15,
    cmap: str = "viridis",
    ax: Optional[Axes] = None,
    show_body: bool = True,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    n_y_vis: int = 20,
) -> Tuple[Figure, Axes]:
    """Wrapped velocity-coloured envelope plot around the body.

    This extends the plain δ-envelope plot by filling the region between
    the body surface and δ(s) with the reconstructed velocity field,
    colour-mapped to show the velocity distribution.

    The plot is produced for a **single path** (upper or lower).  Use
    :func:`plot_bl_velocity_envelope_two_sides` for both.

    Args:
        field: :class:`BLFieldData` for one path.
        surface_x: Full body x-coordinates (M_body,).
        surface_y: Full body y-coordinates (M_body,).
        panel_indices: Indices mapping path panels to full-body panels.
        scale: Geometric scale factor for the envelope displacement.
        cmap: Colormap for velocity.
        ax: Existing axes.
        show_body: Draw body outline.
        title: Plot title.
        output_path: Save path.
        n_y_vis: Number of wall-normal layers to draw in the envelope.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    if field.u.shape[0] < 2:
        ax.text(
            0.5, 0.5,
            f"Insufficient data ({field.u.shape[0]} station"
            f"{'s' if field.u.shape[0] != 1 else ''})",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="0.4",
        )
        if show_body:
            bx = np.append(surface_x, surface_x[0])
            by = np.append(surface_y, surface_y[0])
            ax.plot(bx, by, "k-", lw=2.0, zorder=10, label="Body")
            ax.set_aspect("equal")
        if title:
            ax.set_title(title, fontsize=11)
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    # Compute outward normals for the full body
    normals = compute_outward_normals(surface_x, surface_y, closed=True)

    # Draw body outline
    if show_body:
        bx = np.append(surface_x, surface_x[0])
        by = np.append(surface_y, surface_y[0])
        ax.plot(bx, by, "k-", lw=2.0, zorder=10, label="Body")

    # We draw the velocity field as stacked coloured quads between
    # wall-normal layers (y_j, y_{j+1}) at each arc-length station.
    # Only stations that appear in BLFieldData are drawn.
    M_field, Ny = field.u.shape

    # Map field stations back to panel indices.
    # field.s corresponds to valid stations from BoundaryLayerResult.
    # We need the (x, y) position and outward normal for each station.
    # Build a mapping: for each field station i, find the closest panel
    # index in the path.
    path_s = np.zeros(len(panel_indices))
    path_x = surface_x[panel_indices]
    path_y = surface_y[panel_indices]
    ds = np.sqrt(np.diff(path_x) ** 2 + np.diff(path_y) ** 2)
    path_s[1:] = np.cumsum(ds)
    path_normals = normals[panel_indices]

    # For each field station, interpolate position and normal on the path
    # (field.s may not exactly match path panel midpoints because of
    #  re-zeroing, but the relative ordering is the same)
    # We use the field.s values to interpolate into the path geometry.
    # First, re-zero path_s to match field.s domain
    # field.s starts at the first valid station (s > 0 after stagnation)
    # We match by index: field has M_field stations, path has len(panel_indices) panels.
    # The safest approach: find the closest path panel for each field.s value.
    # But field.s was computed from the same path — we just need to shift path_s
    # to the same zero point.

    # Match field.s[0] to the nearest path_s and compute offset
    # Actually, BLFieldData.s comes from result.s[valid_idx] which is the
    # same array as the path.s (from BoundaryLayerPathResult) — same re-zeroing.
    # So we can directly interpolate x, y, nx, ny as functions of path.s.
    # But path.s may have been re-zeroed differently. Let's use the original
    # panel_indices and compute s from the same geometry.

    # Simplest robust approach: for each field station, find nearest path panel
    field_px = np.interp(field.s, path_s - path_s[0] + field.s[0], path_x)
    field_py = np.interp(field.s, path_s - path_s[0] + field.s[0], path_y)
    field_nx = np.interp(field.s, path_s - path_s[0] + field.s[0], path_normals[:, 0])
    field_ny = np.interp(field.s, path_s - path_s[0] + field.s[0], path_normals[:, 1])

    # Normalise field normals
    n_len = np.sqrt(field_nx**2 + field_ny**2)
    n_len = np.where(n_len < 1e-12, 1.0, n_len)
    field_nx /= n_len
    field_ny /= n_len

    # Global velocity range for consistent coloring
    u_min = float(np.nanmin(field.u))
    u_max = float(np.nanmax(field.u))
    norm = mcolors.Normalize(vmin=u_min, vmax=u_max)
    cmap_obj = plt.cm.get_cmap(cmap)

    # Determine how many y-layers to draw (subsample if Ny > n_y_vis)
    if Ny > n_y_vis:
        y_idx = np.linspace(0, Ny - 1, n_y_vis + 1, dtype=int)
    else:
        y_idx = np.arange(Ny)

    # Scale: map delta to geometric displacement
    delta_max = float(np.nanmax(field.delta)) if np.any(field.delta > 0) else 1.0

    for i in range(M_field - 1):
        for jj in range(len(y_idx) - 1):
            j0 = y_idx[jj]
            j1 = y_idx[jj + 1]
            # Mean velocity for this quad's colour
            u_avg = 0.25 * (
                field.u[i, j0] + field.u[i, j1]
                + field.u[i + 1, j0] + field.u[i + 1, j1]
            )
            color = cmap_obj(norm(u_avg))

            # y displacement scaled to geometry
            d00 = field.y[i, j0] / delta_max * scale
            d01 = field.y[i, j1] / delta_max * scale
            d10 = field.y[i + 1, j0] / delta_max * scale
            d11 = field.y[i + 1, j1] / delta_max * scale

            quad_x = [
                field_px[i] + d00 * field_nx[i],
                field_px[i] + d01 * field_nx[i],
                field_px[i + 1] + d11 * field_nx[i + 1],
                field_px[i + 1] + d10 * field_nx[i + 1],
            ]
            quad_y = [
                field_py[i] + d00 * field_ny[i],
                field_py[i] + d01 * field_ny[i],
                field_py[i + 1] + d11 * field_ny[i + 1],
                field_py[i + 1] + d10 * field_ny[i + 1],
            ]
            ax.fill(quad_x, quad_y, color=color, edgecolor="none", zorder=2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$u$ [m/s]", shrink=0.8, pad=0.02)

    # Draw delta envelope line
    env_x = field_px + (field.delta / delta_max * scale) * field_nx
    env_y = field_py + (field.delta / delta_max * scale) * field_ny
    ax.plot(env_x, env_y, "w--", lw=1.2, zorder=5, label=r"$\delta(s)$")

    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(
            f"Velocity envelope — {field.profile_name}", fontsize=11,
        )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


# -------------------------------------------------------------------------
# Two-side convenience wrappers
# -------------------------------------------------------------------------


def plot_bl_velocity_contour_two_sides(
    field_upper,  # BLFieldData
    field_lower,  # BLFieldData
    cmap: str = "viridis",
    show_delta: bool = True,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Side-by-side s-y velocity contour for upper and lower paths.

    Args:
        field_upper: :class:`BLFieldData` for the upper path.
        field_lower: :class:`BLFieldData` for the lower path.
        cmap: Colormap.
        show_delta: Show δ(s) overlay.
        title: Super-title.
        output_path: Save path.

    Returns:
        ``(fig, (ax_upper, ax_lower))`` tuple.
    """
    fig, (ax_u, ax_l) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    plot_bl_velocity_contour(
        field_upper, ax=ax_u, cmap=cmap, show_delta=show_delta,
        title=f"Upper — {field_upper.profile_name}",
    )
    plot_bl_velocity_contour(
        field_lower, ax=ax_l, cmap=cmap, show_delta=show_delta,
        title=f"Lower — {field_lower.profile_name}",
    )

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, (ax_u, ax_l)


def plot_bl_velocity_contour_normalized_two_sides(
    field_upper,  # BLFieldData
    field_lower,  # BLFieldData
    cmap: str = "viridis",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Side-by-side normalised s-(y/δ) velocity contour for both paths.

    Args:
        field_upper: :class:`BLFieldData` for the upper path.
        field_lower: :class:`BLFieldData` for the lower path.
        cmap: Colormap.
        title: Super-title.
        output_path: Save path.

    Returns:
        ``(fig, (ax_upper, ax_lower))`` tuple.
    """
    fig, (ax_u, ax_l) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    plot_bl_velocity_contour_normalized(
        field_upper, ax=ax_u, cmap=cmap,
        title=f"Upper — {field_upper.profile_name}",
    )
    plot_bl_velocity_contour_normalized(
        field_lower, ax=ax_l, cmap=cmap,
        title=f"Lower — {field_lower.profile_name}",
    )

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, (ax_u, ax_l)


def plot_bl_velocity_envelope_two_sides(
    field_upper,  # BLFieldData
    field_lower,  # BLFieldData
    case_result,  # BoundaryLayerCaseResult
    scale: float = 0.15,
    cmap: str = "viridis",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    n_y_vis: int = 20,
) -> Tuple[Figure, Axes]:
    """Wrapped velocity envelope for both upper and lower paths.

    Both paths are drawn on the same body outline.

    Args:
        field_upper: :class:`BLFieldData` for the upper path.
        field_lower: :class:`BLFieldData` for the lower path.
        case_result: :class:`BoundaryLayerCaseResult` for body geometry.
        scale: Geometric scale factor.
        cmap: Colormap.
        title: Super-title.
        output_path: Save path.
        n_y_vis: Number of wall-normal layers to draw.

    Returns:
        ``(fig, ax)`` tuple.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw upper side (body drawn here)
    plot_bl_velocity_envelope(
        field_upper,
        surface_x=case_result.surface_x,
        surface_y=case_result.surface_y,
        panel_indices=case_result.upper.panel_indices,
        scale=scale, cmap=cmap, ax=ax, show_body=True,
        n_y_vis=n_y_vis,
    )

    # Draw lower side (body already drawn)
    plot_bl_velocity_envelope(
        field_lower,
        surface_x=case_result.surface_x,
        surface_y=case_result.surface_y,
        panel_indices=case_result.lower.panel_indices,
        scale=scale, cmap=cmap, ax=ax, show_body=False,
        n_y_vis=n_y_vis,
    )

    if title:
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(
            f"Velocity envelope — {field_upper.profile_name}", fontsize=11,
        )

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


# -------------------------------------------------------------------------
# Fluent/CFD comparison plots
# -------------------------------------------------------------------------


def plot_bl_fluent_comparison(
    field,  # BLFieldData from panel-method BL solver
    fluent_field=None,  # InterpolatedBLField from Fluent
    cmap: str = "RdBu_r",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    show_colorbar: bool = True,
) -> Tuple[Figure, Axes]:
    """Velocity difference contour between panel-method BL and Fluent.

    Shows the difference (BL solver - Fluent) in tangential velocity as
    a filled contour plot in (s, y) coordinates. Positive values indicate
    the BL solver predicts higher velocity than Fluent.

    Args:
        field: :class:`BLFieldData` from the panel-method BL solver.
        fluent_field: :class:`InterpolatedBLField` from Fluent comparison.
            If *None*, shows only the BL solver field with a note.
        cmap: Colormap for the difference plot.
        ax: Existing axes.
        title: Plot title.
        output_path: Save path.
        show_colorbar: Whether to add a colorbar.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if fluent_field is None:
        # No Fluent data — just show the panel-method contour with a note
        fig, ax = plot_bl_velocity_contour(
            field, ax=ax, cmap="viridis",
            title=title or f"BL velocity — {field.profile_name} (no Fluent data)",
        )
        ax.annotate(
            "Fluent comparison not available",
            xy=(0.5, 0.95), xycoords="axes fraction",
            ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9),
        )
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    # --- Actual comparison ------------------------------------------------
    M, Ny = field.u.shape
    diff = field.u - fluent_field.u  # element-wise difference

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    s_edges = _cell_edges(field.s)
    y_edge_grid = np.zeros((M + 1, Ny + 1), dtype=np.float64)
    for i in range(M):
        y_edge_grid[i] = _cell_edges(field.y[i])
    y_edge_grid[M] = y_edge_grid[M - 1]

    S_grid = np.broadcast_to(s_edges[:, np.newaxis], (M + 1, Ny + 1))

    # Use symmetric colorbar centered on zero
    vmax = np.nanmax(np.abs(diff))
    vmin = -vmax

    pcm = ax.pcolormesh(
        S_grid, y_edge_grid, diff,
        cmap=cmap, shading="flat", rasterized=True,
        vmin=vmin, vmax=vmax,
    )

    if show_colorbar:
        fig.colorbar(
            pcm, ax=ax,
            label=r"$u_{\mathrm{BL}} - u_{\mathrm{Fluent}}$ [m/s]",
            shrink=0.85, pad=0.02,
        )

    # Overlay δ(s) from both sources
    ax.plot(field.s, field.delta, "k-", lw=1.5, label="δ (BL solver)")
    ax.plot(
        fluent_field.s, fluent_field.delta,
        "k--", lw=1.5, label="δ (Fluent)",
    )

    ax.set_xlabel("Arc length $s$ [m]")
    ax.set_ylabel("Wall-normal $y$ [m]")
    ax.set_xlim(field.s[0], field.s[-1])
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=8)

    if title:
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(
            f"Velocity difference — {field.profile_name} vs Fluent", fontsize=11,
        )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_bl_fluent_comparison_two_sides(
    bl_result,  # BoundaryLayerCaseResult
    comparison_result,  # BLComparisonResult
    profile_name: str = "thwaites",
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Two-panel velocity difference plot (upper and lower sides).

    Args:
        bl_result: Panel-method BL solver result.
        comparison_result: Fluent comparison result.
        profile_name: Name of the BL profile to compare.
        cmap: Colormap for difference plots.
        title: Overall figure title.
        output_path: Save path.

    Returns:
        ``(fig, (ax_upper, ax_lower))`` tuple.
    """
    fig, (ax_u, ax_l) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"hspace": 0.15},
    )

    # Upper side
    field_u = bl_result.upper.fields.get(profile_name)
    fluent_u = comparison_result.upper_fluent_field

    if field_u is not None:
        plot_bl_fluent_comparison(
            field_u, fluent_u, cmap=cmap, ax=ax_u,
            title="Upper side", show_colorbar=True,
        )

    # Lower side
    field_l = bl_result.lower.fields.get(profile_name)
    fluent_l = comparison_result.lower_fluent_field

    if field_l is not None:
        plot_bl_fluent_comparison(
            field_l, fluent_l, cmap=cmap, ax=ax_l,
            title="Lower side", show_colorbar=True,
        )

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, (ax_u, ax_l)


def plot_bl_wall_comparison(
    bl_result,  # BoundaryLayerCaseResult
    fluent_result,  # FluentBLResult
    quantities: Optional[List[str]] = None,
    profile_name: Optional[str] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, NDArray]:
    """Line plots comparing wall quantities (Ue, Cf, δ) vs arc-length.

    Creates a multi-panel figure with BL solver and Fluent results overlaid
    for each quantity.

    Args:
        bl_result: Panel-method BL solver result.
        fluent_result: Extracted Fluent BL result.
        quantities: Which quantities to plot. Default: ["Ue", "Cf", "delta"].
        profile_name: BL profile for δ comparison. If None, uses first available.
        title: Overall figure title.
        output_path: Save path.

    Returns:
        ``(fig, axes)`` tuple where axes is a 2D array of shape (n_quantities, 2).
    """
    if quantities is None:
        quantities = ["Ue", "Cf", "delta"]

    n_qty = len(quantities)
    fig, axes = plt.subplots(
        n_qty, 2, figsize=(12, 3 * n_qty), sharex="col",
        gridspec_kw={"hspace": 0.1, "wspace": 0.25},
    )

    if n_qty == 1:
        axes = axes.reshape(1, 2)

    qty_labels = {
        "Ue": r"$U_e$ [m/s]",
        "Cf": r"$C_f$ [-]",
        "delta": r"$\delta$ [m]",
        "delta_star": r"$\delta^*$ [m]",
        "theta": r"$\theta$ [m]",
        "H": r"$H$ [-]",
    }

    for side_idx, side in enumerate(["upper", "lower"]):
        bl_path = bl_result.sides[side]
        fluent_path = fluent_result.sides[side]

        # Get profile result
        if profile_name and profile_name in bl_path.results:
            bl_res = bl_path.results[profile_name]
            pname = profile_name
        elif bl_path.results:
            pname = list(bl_path.results.keys())[0]
            bl_res = bl_path.results[pname]
        else:
            continue

        for qty_idx, qty in enumerate(quantities):
            ax = axes[qty_idx, side_idx]

            # Get BL solver data
            if qty == "Ue":
                bl_s, bl_val = bl_path.s, bl_path.Ue
                fl_s, fl_val = fluent_path.s, fluent_path.Ue
            elif qty == "Cf":
                bl_s, bl_val = bl_res.s, bl_res.cf
                fl_s, fl_val = fluent_path.s, fluent_path.Cf
            elif qty == "delta":
                if pname in bl_path.fields:
                    bl_s = bl_path.fields[pname].s
                    bl_val = bl_path.fields[pname].delta
                else:
                    bl_s, bl_val = np.array([]), np.array([])
                fl_s, fl_val = fluent_path.s, fluent_path.delta
            elif qty in ["delta_star", "theta", "H"]:
                bl_s = bl_res.s
                bl_val = getattr(bl_res, qty)
                # Fluent doesn't have these directly
                fl_s, fl_val = np.array([]), np.array([])
            else:
                continue

            # Plot BL solver
            if len(bl_val) > 0:
                ax.plot(bl_s, bl_val, "-", lw=1.5, color="#1f77b4", label="BL solver")

            # Plot Fluent
            if len(fl_val) > 0:
                ax.plot(fl_s, fl_val, "--", lw=1.5, color="#d62728", label="Fluent")

            # Labels
            if qty_idx == 0:
                ax.set_title(f"{side.capitalize()} side", fontsize=11)
            if qty_idx == n_qty - 1:
                ax.set_xlabel("Arc length $s$ [m]")
            ax.set_ylabel(qty_labels.get(qty, qty))

            if qty_idx == 0 and side_idx == 0:
                ax.legend(loc="best", fontsize=8)

            ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, axes


# -------------------------------------------------------------------------
# Fluent comparison — envelope and normalized plots
# -------------------------------------------------------------------------


def _draw_envelope_comparison_quads(
    ax: Axes,
    field,  # BLFieldData
    fluent_field,  # InterpolatedBLField
    surface_x: NDArray[np.float64],
    surface_y: NDArray[np.float64],
    panel_indices: List[int],
    normals: NDArray[np.float64],
    scale: float,
    delta_max: float,
    cmap_obj,
    norm: mcolors.Normalize,
    n_y_vis: int,
    draw_delta_lines: bool = True,
) -> None:
    """Draw velocity-difference envelope quads for one BL path (internal helper).

    This is factored out to allow drawing multiple paths on the same axes
    with consistent color normalization.
    """
    M_field, Ny = field.u.shape
    if M_field < 2:
        return

    # Build geometry mapping
    path_s = np.zeros(len(panel_indices))
    path_x = surface_x[panel_indices]
    path_y = surface_y[panel_indices]
    ds = np.sqrt(np.diff(path_x) ** 2 + np.diff(path_y) ** 2)
    path_s[1:] = np.cumsum(ds)
    path_normals = normals[panel_indices]

    field_px = np.interp(field.s, path_s - path_s[0] + field.s[0], path_x)
    field_py = np.interp(field.s, path_s - path_s[0] + field.s[0], path_y)
    field_nx = np.interp(field.s, path_s - path_s[0] + field.s[0], path_normals[:, 0])
    field_ny = np.interp(field.s, path_s - path_s[0] + field.s[0], path_normals[:, 1])

    # Normalise field normals
    n_len = np.sqrt(field_nx**2 + field_ny**2)
    n_len = np.where(n_len < 1e-12, 1.0, n_len)
    field_nx /= n_len
    field_ny /= n_len

    # Compute velocity difference
    diff = field.u - fluent_field.u

    # Determine y-layers to draw
    if Ny > n_y_vis:
        y_idx = np.linspace(0, Ny - 1, n_y_vis + 1, dtype=int)
    else:
        y_idx = np.arange(Ny)

    # Scale: map delta to geometric displacement (use provided delta_max for consistency)
    local_delta_max = float(np.nanmax(field.delta)) if np.any(field.delta > 0) else 1.0

    for i in range(M_field - 1):
        for jj in range(len(y_idx) - 1):
            j0 = y_idx[jj]
            j1 = y_idx[jj + 1]
            # Mean difference for this quad's colour
            diff_avg = 0.25 * (
                diff[i, j0] + diff[i, j1]
                + diff[i + 1, j0] + diff[i + 1, j1]
            )
            color = cmap_obj(norm(diff_avg))

            # y displacement scaled to geometry
            d00 = field.y[i, j0] / delta_max * scale
            d01 = field.y[i, j1] / delta_max * scale
            d10 = field.y[i + 1, j0] / delta_max * scale
            d11 = field.y[i + 1, j1] / delta_max * scale

            quad_x = [
                field_px[i] + d00 * field_nx[i],
                field_px[i] + d01 * field_nx[i],
                field_px[i + 1] + d11 * field_nx[i + 1],
                field_px[i + 1] + d10 * field_nx[i + 1],
            ]
            quad_y = [
                field_py[i] + d00 * field_ny[i],
                field_py[i] + d01 * field_ny[i],
                field_py[i + 1] + d11 * field_ny[i + 1],
                field_py[i + 1] + d10 * field_ny[i + 1],
            ]
            ax.fill(quad_x, quad_y, color=color, edgecolor="none", zorder=2)

    # Draw delta envelope lines
    if draw_delta_lines:
        env_x_bl = field_px + (field.delta / delta_max * scale) * field_nx
        env_y_bl = field_py + (field.delta / delta_max * scale) * field_ny
        ax.plot(env_x_bl, env_y_bl, "k-", lw=1.2, zorder=5)

        env_x_fl = field_px + (fluent_field.delta / delta_max * scale) * field_nx
        env_y_fl = field_py + (fluent_field.delta / delta_max * scale) * field_ny
        ax.plot(env_x_fl, env_y_fl, "k--", lw=1.2, zorder=5)


def plot_bl_velocity_envelope_comparison_two_sides(
    bl_result,  # BoundaryLayerCaseResult
    comparison_result,  # BLComparisonResult
    profile_name: str = "thwaites",
    scale: float = 0.15,
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    n_y_vis: int = 20,
) -> Tuple[Figure, Axes]:
    """Wrapped velocity-difference envelope for both upper and lower paths.

    Shows the velocity difference (BL solver - Fluent) as a colour-mapped
    envelope wrapped around the full body geometry, with both upper and
    lower boundary layer paths displayed continuously for a complete
    visualization.

    Positive values (red) indicate BL solver predicts higher velocity;
    negative values (blue) indicate Fluent is higher.

    Args:
        bl_result: Panel-method BL solver result with fields for both sides.
        comparison_result: Fluent comparison result with interpolated fields.
        profile_name: Name of the BL profile to compare.
        scale: Geometric scale factor for the envelope displacement.
        cmap: Colormap for velocity difference.
        title: Plot title.
        output_path: Save path.
        n_y_vis: Number of wall-normal layers to draw in the envelope.

    Returns:
        ``(fig, ax)`` tuple.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Check if Fluent data is available
    if not comparison_result.has_fluent_data:
        # Fall back to regular two-sided envelope
        field_u = bl_result.upper.fields.get(profile_name)
        field_l = bl_result.lower.fields.get(profile_name)
        if field_u is not None and field_l is not None:
            plot_bl_velocity_envelope_two_sides(
                field_u, field_l, bl_result,
                scale=scale, cmap="viridis",
                title=title or f"BL velocity envelope — {profile_name}",
                output_path=None, n_y_vis=n_y_vis,
            )
        ax.annotate(
            "Fluent comparison not available",
            xy=(0.5, 0.95), xycoords="axes fraction",
            ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9),
        )
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    # Get fields for both sides
    field_upper = bl_result.upper.fields.get(profile_name)
    field_lower = bl_result.lower.fields.get(profile_name)
    fluent_upper = comparison_result.upper_fluent_field
    fluent_lower = comparison_result.lower_fluent_field

    if field_upper is None or field_lower is None:
        ax.text(
            0.5, 0.5,
            f"BL field not available for profile '{profile_name}'",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="0.4",
        )
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    if fluent_upper is None or fluent_lower is None:
        ax.text(
            0.5, 0.5,
            "Fluent interpolated fields not available",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="0.4",
        )
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    surface_x = bl_result.surface_x
    surface_y = bl_result.surface_y

    # Compute outward normals for the full body (once)
    normals = compute_outward_normals(surface_x, surface_y, closed=True)

    # Draw body outline
    bx = np.append(surface_x, surface_x[0])
    by = np.append(surface_y, surface_y[0])
    ax.plot(bx, by, "k-", lw=2.0, zorder=10, label="Body")

    # Compute global velocity difference range for consistent coloring
    diff_upper = field_upper.u - fluent_upper.u
    diff_lower = field_lower.u - fluent_lower.u
    vmax = max(np.nanmax(np.abs(diff_upper)), np.nanmax(np.abs(diff_lower)))
    vmin = -vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.cm.get_cmap(cmap)

    # Compute global delta_max for consistent scaling
    delta_max = max(
        float(np.nanmax(field_upper.delta)) if np.any(field_upper.delta > 0) else 1.0,
        float(np.nanmax(field_lower.delta)) if np.any(field_lower.delta > 0) else 1.0,
    )

    # Draw upper side
    _draw_envelope_comparison_quads(
        ax, field_upper, fluent_upper,
        surface_x, surface_y, bl_result.upper.panel_indices,
        normals, scale, delta_max, cmap_obj, norm, n_y_vis,
        draw_delta_lines=True,
    )

    # Draw lower side
    _draw_envelope_comparison_quads(
        ax, field_lower, fluent_lower,
        surface_x, surface_y, bl_result.lower.panel_indices,
        normals, scale, delta_max, cmap_obj, norm, n_y_vis,
        draw_delta_lines=True,
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(
        sm, ax=ax,
        label=r"$u_{\mathrm{BL}} - u_{\mathrm{Fluent}}$ [m/s]",
        shrink=0.8, pad=0.02,
    )

    # Add legend entries for delta lines
    ax.plot([], [], "k-", lw=1.2, label=r"$\delta$ (BL)")
    ax.plot([], [], "k--", lw=1.2, label=r"$\delta$ (Fluent)")

    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(
            f"Velocity difference envelope — {profile_name} vs Fluent", fontsize=11,
        )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_bl_velocity_envelope_comparison(
    field,  # BLFieldData from panel-method BL solver
    fluent_field=None,  # InterpolatedBLField from Fluent
    surface_x: Optional[NDArray[np.float64]] = None,
    surface_y: Optional[NDArray[np.float64]] = None,
    panel_indices: Optional[List[int]] = None,
    scale: float = 0.15,
    cmap: str = "RdBu_r",
    ax: Optional[Axes] = None,
    show_body: bool = True,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    n_y_vis: int = 20,
) -> Tuple[Figure, Axes]:
    """Wrapped velocity-difference envelope plot for one side.

    Shows the velocity difference (BL solver - Fluent) as a colour-mapped
    envelope wrapped around the body geometry. Positive values (red) indicate
    BL solver predicts higher velocity; negative values (blue) indicate Fluent
    is higher.

    For a complete visualization with both upper and lower sides, use
    :func:`plot_bl_velocity_envelope_comparison_two_sides` instead.

    Args:
        field: :class:`BLFieldData` from the panel-method BL solver.
        fluent_field: :class:`InterpolatedBLField` from Fluent comparison.
            If *None*, shows BL solver velocity with a note.
        surface_x: Full body x-coordinates (M_body,).
        surface_y: Full body y-coordinates (M_body,).
        panel_indices: Indices mapping path panels to full-body panels.
        scale: Geometric scale factor for the envelope displacement.
        cmap: Colormap for velocity difference.
        ax: Existing axes.
        show_body: Draw body outline.
        title: Plot title.
        output_path: Save path.
        n_y_vis: Number of wall-normal layers to draw in the envelope.

    Returns:
        ``(fig, ax)`` tuple.
    """
    # Validate required geometry
    if surface_x is None or surface_y is None or panel_indices is None:
        # Fall back to regular velocity envelope without comparison
        fig, ax_out = plot_bl_velocity_envelope(
            field,
            surface_x=surface_x if surface_x is not None else np.zeros(1),
            surface_y=surface_y if surface_y is not None else np.zeros(1),
            panel_indices=panel_indices if panel_indices is not None else [0],
            scale=scale,
            cmap="viridis",
            ax=ax,
            show_body=show_body,
            title=title or f"BL velocity envelope — {field.profile_name} (no geometry)",
            output_path=output_path,
            n_y_vis=n_y_vis,
        )
        return fig, ax_out

    if fluent_field is None:
        # No Fluent data — show BL solver envelope with a note
        fig, ax_out = plot_bl_velocity_envelope(
            field,
            surface_x=surface_x,
            surface_y=surface_y,
            panel_indices=panel_indices,
            scale=scale,
            cmap="viridis",
            ax=ax,
            show_body=show_body,
            title=title or f"BL velocity envelope — {field.profile_name}",
            output_path=None,
            n_y_vis=n_y_vis,
        )
        ax_out.annotate(
            "Fluent comparison not available",
            xy=(0.5, 0.95), xycoords="axes fraction",
            ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9),
        )
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax_out

    # --- Actual comparison: velocity difference envelope ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    M_field, Ny = field.u.shape

    if M_field < 2:
        ax.text(
            0.5, 0.5,
            f"Insufficient data ({M_field} station"
            f"{'s' if M_field != 1 else ''})",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="0.4",
        )
        if show_body:
            bx = np.append(surface_x, surface_x[0])
            by = np.append(surface_y, surface_y[0])
            ax.plot(bx, by, "k-", lw=2.0, zorder=10, label="Body")
            ax.set_aspect("equal")
        if title:
            ax.set_title(title, fontsize=11)
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    # Compute outward normals for the full body
    normals = compute_outward_normals(surface_x, surface_y, closed=True)

    # Draw body outline
    if show_body:
        bx = np.append(surface_x, surface_x[0])
        by = np.append(surface_y, surface_y[0])
        ax.plot(bx, by, "k-", lw=2.0, zorder=10, label="Body")

    # Compute velocity difference for color normalization
    diff = field.u - fluent_field.u
    vmax = np.nanmax(np.abs(diff))
    vmin = -vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.cm.get_cmap(cmap)

    # Compute delta_max for scaling
    delta_max = float(np.nanmax(field.delta)) if np.any(field.delta > 0) else 1.0

    # Draw envelope quads
    _draw_envelope_comparison_quads(
        ax, field, fluent_field,
        surface_x, surface_y, panel_indices,
        normals, scale, delta_max, cmap_obj, norm, n_y_vis,
        draw_delta_lines=True,
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$u_{\mathrm{BL}} - u_{\mathrm{Fluent}}$ [m/s]", shrink=0.8, pad=0.02)

    # Add legend entries
    ax.plot([], [], "k-", lw=1.2, label=r"$\delta$ (BL)")
    ax.plot([], [], "k--", lw=1.2, label=r"$\delta$ (Fluent)")

    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(
            f"Velocity difference envelope — {field.profile_name} vs Fluent", fontsize=11,
        )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_bl_velocity_contour_normalized_comparison(
    field,  # BLFieldData from panel-method BL solver
    fluent_field=None,  # InterpolatedBLField from Fluent
    cmap: str = "RdBu_r",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    show_colorbar: bool = True,
) -> Tuple[Figure, Axes]:
    """Normalized s-(y/δ) velocity difference contour for Fluent comparison.

    Same as :func:`plot_bl_velocity_contour_normalized` but shows the
    difference (BL solver - Fluent) in the normalized (s, y/δ) space.
    This makes thin parts of the BL easier to compare by stretching the
    y-axis to a uniform [0, 1] range at each station.

    Args:
        field: :class:`BLFieldData` from the panel-method BL solver.
        fluent_field: :class:`InterpolatedBLField` from Fluent comparison.
            If *None*, shows BL solver field with a note.
        cmap: Colormap for difference plot.
        ax: Existing axes.
        title: Plot title.
        output_path: Save path.
        show_colorbar: Whether to add a colorbar.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if fluent_field is None:
        # No Fluent data — fall back to regular normalized contour
        fig, ax_out = plot_bl_velocity_contour_normalized(
            field, ax=ax, cmap="viridis",
            title=title or f"Normalized BL velocity — {field.profile_name} (no Fluent data)",
        )
        ax_out.annotate(
            "Fluent comparison not available",
            xy=(0.5, 0.95), xycoords="axes fraction",
            ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9),
        )
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax_out

    M, Ny = field.u.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    if M < 2:
        ax.text(
            0.5, 0.5,
            f"Insufficient data ({M} station{'s' if M != 1 else ''})",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="0.4",
        )
        if title:
            ax.set_title(title, fontsize=11)
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    # Compute velocity difference
    diff = field.u - fluent_field.u

    # Normalise y by delta at each station → eta in [0, 1]
    eta = np.zeros_like(field.y)
    for i in range(M):
        if field.delta[i] > 0:
            eta[i] = field.y[i] / field.delta[i]
        else:
            eta[i] = np.linspace(0.0, 1.0, Ny)

    # Build edge grids
    s_edges = _cell_edges(field.s)
    eta_edge_grid = np.zeros((M + 1, Ny + 1), dtype=np.float64)
    for i in range(M):
        eta_edge_grid[i] = _cell_edges(eta[i])
    eta_edge_grid[M] = eta_edge_grid[M - 1]

    S_grid = np.broadcast_to(s_edges[:, np.newaxis], (M + 1, Ny + 1))

    # Symmetric colorbar for difference
    vmax = np.nanmax(np.abs(diff))
    vmin = -vmax

    pcm = ax.pcolormesh(
        S_grid, eta_edge_grid, diff,
        cmap=cmap, shading="flat", rasterized=True,
        vmin=vmin, vmax=vmax,
    )

    if show_colorbar:
        fig.colorbar(
            pcm, ax=ax,
            label=r"$u_{\mathrm{BL}} - u_{\mathrm{Fluent}}$ [m/s]",
            shrink=0.85, pad=0.02,
        )

    # Horizontal line at η = 1 (BL edge from BL solver)
    ax.axhline(1.0, color="k", ls="-", lw=1.0, alpha=0.7, label=r"$\delta$ (BL)")

    # Draw Fluent δ normalized by BL solver δ
    eta_fluent_delta = np.where(
        field.delta > 0,
        fluent_field.delta / field.delta,
        1.0,
    )
    ax.plot(field.s, eta_fluent_delta, "k--", lw=1.0, alpha=0.7, label=r"$\delta$ (Fluent)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    ax.set_xlabel("Arc length $s$ [m]")
    ax.set_ylabel(r"$y / \delta$")
    ax.set_xlim(field.s[0], field.s[-1])
    ax.set_ylim(0, None)

    if title:
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(
            f"Normalized velocity difference — {field.profile_name} vs Fluent", fontsize=11,
        )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_bl_comparison_report(
    comparison_result,  # BLComparisonResult
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, NDArray]:
    """Comprehensive metrics report for BL solver vs Fluent comparison.

    Creates a summary figure with:
    - Table of RMS/relative errors for all quantities (Ue, Cf, δ, velocity field)
    - Bar chart visualization of relative errors by side
    - Text summary of comparison statistics

    Args:
        comparison_result: :class:`BLComparisonResult` with computed metrics.
        title: Overall figure title.
        output_path: Save path.

    Returns:
        ``(fig, axes)`` tuple where axes is a 2D array of axes.
    """
    if not comparison_result.has_fluent_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5,
            "Fluent comparison data not available",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="0.4",
        )
        ax.axis("off")
        if title:
            fig.suptitle(title, fontsize=12)
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, np.array([[ax]])

    # Create figure with 2x2 layout:
    # Top row: bar charts for upper and lower sides
    # Bottom row: metrics table spanning both columns
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], hspace=0.3, wspace=0.25)

    ax_bar_upper = fig.add_subplot(gs[0, 0])
    ax_bar_lower = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    # Colors for quantities
    qty_colors = {
        "Ue": "#1f77b4",
        "Cf": "#ff7f0e",
        "delta": "#2ca02c",
        "u": "#d62728",
    }

    # Collect all metrics into a table structure
    table_data = []
    table_cols = ["Side", "Quantity", "RMS", "MAE", "L∞", "Rel. L2 (%)", "N points"]

    # Bar chart data
    bar_data = {"upper": {}, "lower": {}}

    for side in ["upper", "lower"]:
        # Wall metrics
        if side in comparison_result.wall_metrics:
            for qty, metrics in comparison_result.wall_metrics[side].items():
                table_data.append([
                    side.capitalize(),
                    qty,
                    f"{metrics.RMS:.4g}",
                    f"{metrics.MAE:.4g}",
                    f"{metrics.L_inf:.4g}",
                    f"{metrics.relative_L2 * 100:.1f}" if not np.isnan(metrics.relative_L2) else "N/A",
                    str(metrics.n_points),
                ])
                if not np.isnan(metrics.relative_L2):
                    bar_data[side][qty] = metrics.relative_L2 * 100

        # Velocity metrics
        if side in comparison_result.velocity_metrics:
            for qty, metrics in comparison_result.velocity_metrics[side].items():
                table_data.append([
                    side.capitalize(),
                    f"{qty} (field)",
                    f"{metrics.RMS:.4g}",
                    f"{metrics.MAE:.4g}",
                    f"{metrics.L_inf:.4g}",
                    f"{metrics.relative_L2 * 100:.1f}" if not np.isnan(metrics.relative_L2) else "N/A",
                    str(metrics.n_points),
                ])
                if not np.isnan(metrics.relative_L2):
                    bar_data[side][f"{qty}_field"] = metrics.relative_L2 * 100

    # Plot bar charts
    for ax_bar, side in [(ax_bar_upper, "upper"), (ax_bar_lower, "lower")]:
        data = bar_data[side]
        if data:
            quantities = list(data.keys())
            values = list(data.values())
            colors = [qty_colors.get(q.split("_")[0], "#888888") for q in quantities]

            x = np.arange(len(quantities))
            bars = ax_bar.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)

            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(quantities, rotation=45, ha="right", fontsize=9)
            ax_bar.set_ylabel("Relative Error (%)")
            ax_bar.set_title(f"{side.capitalize()} Side", fontsize=11)
            ax_bar.grid(True, axis="y", alpha=0.3)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax_bar.annotate(
                    f"{val:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8,
                )
        else:
            ax_bar.text(
                0.5, 0.5, "No metrics available",
                transform=ax_bar.transAxes, ha="center", va="center",
                fontsize=11, color="0.4",
            )
            ax_bar.set_title(f"{side.capitalize()} Side", fontsize=11)

    # Plot metrics table
    ax_table.axis("off")

    if table_data:
        table = ax_table.table(
            cellText=table_data,
            colLabels=table_cols,
            loc="center",
            cellLoc="center",
            colColours=["#f0f0f0"] * len(table_cols),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)

        # Style header row
        for j, col in enumerate(table_cols):
            cell = table[(0, j)]
            cell.set_text_props(weight="bold")
    else:
        ax_table.text(
            0.5, 0.5, "No metrics computed",
            transform=ax_table.transAxes, ha="center", va="center",
            fontsize=11, color="0.4",
        )

    # Title
    if title:
        fig.suptitle(title, fontsize=13, y=0.98)
    else:
        fig.suptitle(
            f"BL Solver vs Fluent — {comparison_result.profile_name}",
            fontsize=13, y=0.98,
        )

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, np.array([[ax_bar_upper, ax_bar_lower], [ax_table, ax_table]])


# Backward compatibility alias
def plot_bl_of_comparison(
    field,  # BLFieldData
    of_field=None,  # Optional BLFieldData/InterpolatedBLField
    cmap: str = "RdBu_r",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """Alias for :func:`plot_bl_fluent_comparison` (backward compatibility).

    The original function was a placeholder for OpenFOAM comparison.
    It now uses the Fluent comparison implementation.
    """
    return plot_bl_fluent_comparison(
        field, of_field, cmap=cmap, ax=ax, title=title, output_path=output_path,
    )


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _cell_edges(centers: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute cell-edge coordinates from cell centres for pcolormesh.

    Given N centre values, returns N+1 edge values using midpoints
    between adjacent centres, with linear extrapolation at the ends.

    For a single centre value, returns edges at ±0.5 around it.
    """
    N = len(centers)
    if N == 0:
        return np.empty(1, dtype=np.float64)
    if N == 1:
        # Can't compute a spacing from neighbours; use unit half-width.
        return np.array(
            [centers[0] - 0.5, centers[0] + 0.5], dtype=np.float64,
        )
    edges = np.empty(N + 1, dtype=np.float64)
    mid = 0.5 * (centers[:-1] + centers[1:])
    edges[1:-1] = mid
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges
