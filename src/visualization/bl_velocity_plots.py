"""Boundary-layer reconstructed velocity-field visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from visualization.bl_plot_common import _cell_edges
from visualization.surface_envelope import compute_outward_normals


def plot_bl_velocity_contour(
    field,
    ax: Optional[Axes] = None,
    cmap: str = "viridis",
    show_delta: bool = True,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    n_levels: int = 64,
) -> Tuple[Figure, Axes]:
    """s-y velocity contour plot for one BL path."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    m_stations, ny = field.u.shape
    if m_stations < 2:
        ax.text(
            0.5,
            0.5,
            f"Insufficient data ({m_stations} station{'s' if m_stations != 1 else ''})",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="0.4",
        )
        if title:
            ax.set_title(title, fontsize=11)
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    s_edges = _cell_edges(field.s)
    y_edge_grid = np.zeros((m_stations + 1, ny + 1), dtype=np.float64)
    for i in range(m_stations):
        y_edge_grid[i] = _cell_edges(field.y[i])
    y_edge_grid[m_stations] = y_edge_grid[m_stations - 1]

    s_grid = np.broadcast_to(s_edges[:, np.newaxis], (m_stations + 1, ny + 1))
    pcm = ax.pcolormesh(s_grid, y_edge_grid, field.u, cmap=cmap, shading="flat", rasterized=True)
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
        ax.set_title(f"BL velocity contour - {field.profile_name}", fontsize=11)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_bl_velocity_contour_normalized(
    field,
    ax: Optional[Axes] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """Normalized s-(y/delta) velocity contour for one BL path."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    m_stations, ny = field.u.shape
    if m_stations < 2:
        ax.text(
            0.5,
            0.5,
            f"Insufficient data ({m_stations} station{'s' if m_stations != 1 else ''})",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="0.4",
        )
        if title:
            ax.set_title(title, fontsize=11)
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    eta = np.zeros_like(field.y)
    for i in range(m_stations):
        if field.delta[i] > 0:
            eta[i] = field.y[i] / field.delta[i]
        else:
            eta[i] = np.linspace(0.0, 1.0, ny)

    s_edges = _cell_edges(field.s)
    eta_edge_grid = np.zeros((m_stations + 1, ny + 1), dtype=np.float64)
    for i in range(m_stations):
        eta_edge_grid[i] = _cell_edges(eta[i])
    eta_edge_grid[m_stations] = eta_edge_grid[m_stations - 1]

    s_grid = np.broadcast_to(s_edges[:, np.newaxis], (m_stations + 1, ny + 1))
    pcm = ax.pcolormesh(s_grid, eta_edge_grid, field.u, cmap=cmap, shading="flat", rasterized=True)
    fig.colorbar(pcm, ax=ax, label=r"$u$ [m/s]", shrink=0.85, pad=0.02)

    ax.axhline(1.0, color="w", ls="--", lw=1.0, alpha=0.7, label=r"$\delta$")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.set_xlabel("Arc length $s$ [m]")
    ax.set_ylabel(r"$y / \delta$")
    ax.set_xlim(field.s[0], field.s[-1])
    ax.set_ylim(0, None)

    if title:
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(f"Normalized BL velocity - {field.profile_name}", fontsize=11)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_bl_velocity_envelope(
    field,
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
    """Wrapped velocity-colored envelope plot around the body."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    if field.u.shape[0] < 2:
        ax.text(
            0.5,
            0.5,
            f"Insufficient data ({field.u.shape[0]} station{'s' if field.u.shape[0] != 1 else ''})",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="0.4",
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

    normals = compute_outward_normals(surface_x, surface_y, closed=True)

    if show_body:
        bx = np.append(surface_x, surface_x[0])
        by = np.append(surface_y, surface_y[0])
        ax.plot(bx, by, "k-", lw=2.0, zorder=10, label="Body")

    m_field, ny = field.u.shape
    path_s = np.zeros(len(panel_indices))
    path_x = surface_x[panel_indices]
    path_y = surface_y[panel_indices]
    ds = np.sqrt(np.diff(path_x) ** 2 + np.diff(path_y) ** 2)
    path_s[1:] = np.cumsum(ds)
    path_normals = normals[panel_indices]

    interp_s = path_s - path_s[0] + field.s[0]
    field_px = np.interp(field.s, interp_s, path_x)
    field_py = np.interp(field.s, interp_s, path_y)
    field_nx = np.interp(field.s, interp_s, path_normals[:, 0])
    field_ny = np.interp(field.s, interp_s, path_normals[:, 1])

    n_len = np.sqrt(field_nx ** 2 + field_ny ** 2)
    n_len = np.where(n_len < 1e-12, 1.0, n_len)
    field_nx /= n_len
    field_ny /= n_len

    u_min = float(np.nanmin(field.u))
    u_max = float(np.nanmax(field.u))
    norm = mcolors.Normalize(vmin=u_min, vmax=u_max)
    cmap_obj = plt.cm.get_cmap(cmap)

    if ny > n_y_vis:
        y_idx = np.linspace(0, ny - 1, n_y_vis + 1, dtype=int)
    else:
        y_idx = np.arange(ny)

    delta_max = float(np.nanmax(field.delta)) if np.any(field.delta > 0) else 1.0

    for i in range(m_field - 1):
        for jj in range(len(y_idx) - 1):
            j0 = y_idx[jj]
            j1 = y_idx[jj + 1]
            u_avg = 0.25 * (
                field.u[i, j0] + field.u[i, j1] + field.u[i + 1, j0] + field.u[i + 1, j1]
            )
            color = cmap_obj(norm(u_avg))

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

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$u$ [m/s]", shrink=0.8, pad=0.02)

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
        ax.set_title(f"Velocity envelope - {field.profile_name}", fontsize=11)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_bl_velocity_contour_two_sides(
    field_upper,
    field_lower,
    cmap: str = "viridis",
    show_delta: bool = True,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Side-by-side s-y velocity contour for upper and lower paths."""
    fig, (ax_u, ax_l) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    plot_bl_velocity_contour(field_upper, ax=ax_u, cmap=cmap, show_delta=show_delta, title=f"Upper - {field_upper.profile_name}")
    plot_bl_velocity_contour(field_lower, ax=ax_l, cmap=cmap, show_delta=show_delta, title=f"Lower - {field_lower.profile_name}")

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, (ax_u, ax_l)


def plot_bl_velocity_contour_normalized_two_sides(
    field_upper,
    field_lower,
    cmap: str = "viridis",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Side-by-side normalized s-(y/delta) contour for both paths."""
    fig, (ax_u, ax_l) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    plot_bl_velocity_contour_normalized(field_upper, ax=ax_u, cmap=cmap, title=f"Upper - {field_upper.profile_name}")
    plot_bl_velocity_contour_normalized(field_lower, ax=ax_l, cmap=cmap, title=f"Lower - {field_lower.profile_name}")

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, (ax_u, ax_l)


def plot_bl_velocity_envelope_two_sides(
    field_upper,
    field_lower,
    case_result,
    scale: float = 0.15,
    cmap: str = "viridis",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    n_y_vis: int = 20,
) -> Tuple[Figure, Axes]:
    """Wrapped velocity envelope for both upper and lower paths."""
    fig, ax = plt.subplots(figsize=(10, 8))

    plot_bl_velocity_envelope(
        field_upper,
        surface_x=case_result.surface_x,
        surface_y=case_result.surface_y,
        panel_indices=case_result.upper.panel_indices,
        scale=scale,
        cmap=cmap,
        ax=ax,
        show_body=True,
        n_y_vis=n_y_vis,
    )
    plot_bl_velocity_envelope(
        field_lower,
        surface_x=case_result.surface_x,
        surface_y=case_result.surface_y,
        panel_indices=case_result.lower.panel_indices,
        scale=scale,
        cmap=cmap,
        ax=ax,
        show_body=False,
        n_y_vis=n_y_vis,
    )

    if title:
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(f"Velocity envelope - {field_upper.profile_name}", fontsize=11)

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax
