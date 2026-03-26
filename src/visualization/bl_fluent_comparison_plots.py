"""Fluent comparison plots for boundary-layer validation."""

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
from visualization.bl_velocity_plots import (
    plot_bl_velocity_contour,
    plot_bl_velocity_contour_normalized,
    plot_bl_velocity_envelope_two_sides,
)
from visualization.surface_envelope import compute_outward_normals


def plot_bl_fluent_comparison(
    field,
    fluent_field=None,
    cmap: str = "RdBu_r",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    show_colorbar: bool = True,
) -> Tuple[Figure, Axes]:
    """Velocity difference contour between panel-method BL and Fluent."""
    if fluent_field is None:
        fig, ax = plot_bl_velocity_contour(
            field,
            ax=ax,
            cmap="viridis",
            title=title or f"BL velocity - {field.profile_name} (no Fluent data)",
        )
        ax.annotate(
            "Fluent comparison not available",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9),
        )
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    m_stations, ny = field.u.shape
    diff = field.u - fluent_field.u

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    s_edges = _cell_edges(field.s)
    y_edge_grid = np.zeros((m_stations + 1, ny + 1), dtype=np.float64)
    for i in range(m_stations):
        y_edge_grid[i] = _cell_edges(field.y[i])
    y_edge_grid[m_stations] = y_edge_grid[m_stations - 1]
    s_grid = np.broadcast_to(s_edges[:, np.newaxis], (m_stations + 1, ny + 1))

    vmax = np.nanmax(np.abs(diff))
    vmin = -vmax
    pcm = ax.pcolormesh(
        s_grid,
        y_edge_grid,
        diff,
        cmap=cmap,
        shading="flat",
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
    )

    if show_colorbar:
        fig.colorbar(
            pcm,
            ax=ax,
            label=r"$u_{\mathrm{BL}} - u_{\mathrm{Fluent}}$ [m/s]",
            shrink=0.85,
            pad=0.02,
        )

    ax.plot(field.s, field.delta, "k-", lw=1.5, label="delta (BL solver)")
    ax.plot(fluent_field.s, fluent_field.delta, "k--", lw=1.5, label="delta (Fluent)")
    ax.set_xlabel("Arc length $s$ [m]")
    ax.set_ylabel("Wall-normal $y$ [m]")
    ax.set_xlim(field.s[0], field.s[-1])
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=8)

    if title:
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(f"Velocity difference - {field.profile_name} vs Fluent", fontsize=11)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_bl_fluent_comparison_two_sides(
    bl_result,
    comparison_result,
    profile_name: str = "thwaites",
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Two-panel velocity difference plot (upper and lower sides)."""
    fig, (ax_u, ax_l) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"hspace": 0.15})

    field_u = bl_result.upper.fields.get(profile_name)
    fluent_u = comparison_result.upper_fluent_field
    if field_u is not None:
        plot_bl_fluent_comparison(field_u, fluent_u, cmap=cmap, ax=ax_u, title="Upper side", show_colorbar=True)

    field_l = bl_result.lower.fields.get(profile_name)
    fluent_l = comparison_result.lower_fluent_field
    if field_l is not None:
        plot_bl_fluent_comparison(field_l, fluent_l, cmap=cmap, ax=ax_l, title="Lower side", show_colorbar=True)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, (ax_u, ax_l)


def plot_bl_wall_comparison(
    bl_result,
    fluent_result,
    quantities: Optional[List[str]] = None,
    profile_name: Optional[str] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, NDArray]:
    """Line plots comparing wall quantities (Ue, Cf, delta) vs arc length."""
    if quantities is None:
        quantities = ["Ue", "Cf", "delta"]

    n_qty = len(quantities)
    fig, axes = plt.subplots(n_qty, 2, figsize=(12, 3 * n_qty), sharex="col", gridspec_kw={"hspace": 0.1, "wspace": 0.25})
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
                fl_s, fl_val = np.array([]), np.array([])
            else:
                continue

            # Plot only over the BL-valid region for this quantity so Fluent
            # data beyond BL separation/termination does not dominate the
            # visual comparison.
            if len(bl_val) > 0:
                valid_bl = np.isfinite(bl_val)
                if np.any(valid_bl):
                    bl_s_plot = bl_s[valid_bl]
                    bl_val_plot = bl_val[valid_bl]
                else:
                    bl_s_plot = np.array([])
                    bl_val_plot = np.array([])
            else:
                bl_s_plot = np.array([])
                bl_val_plot = np.array([])

            fl_s_plot = fl_s
            fl_val_plot = fl_val
            if len(bl_s_plot) > 0 and len(fl_val_plot) > 0:
                s_bl_min = float(np.min(bl_s_plot))
                s_bl_max = float(np.max(bl_s_plot))
                in_range = (
                    np.isfinite(fl_val_plot)
                    & (fl_s_plot >= s_bl_min)
                    & (fl_s_plot <= s_bl_max)
                )
                fl_s_plot = fl_s_plot[in_range]
                fl_val_plot = fl_val_plot[in_range]

            if len(bl_val_plot) > 0:
                ax.plot(bl_s_plot, bl_val_plot, "-", lw=1.5, color="#1f77b4", label="BL solver")
            if len(fl_val_plot) > 0:
                ax.plot(fl_s_plot, fl_val_plot, "--", lw=1.5, color="#d62728", label="Fluent")

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


def _draw_envelope_comparison_quads(
    ax: Axes,
    field,
    fluent_field,
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
    """Draw velocity-difference envelope quads for one BL path."""
    m_field, ny = field.u.shape
    if m_field < 2:
        return

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

    diff = field.u - fluent_field.u
    if ny > n_y_vis:
        y_idx = np.linspace(0, ny - 1, n_y_vis + 1, dtype=int)
    else:
        y_idx = np.arange(ny)

    for i in range(m_field - 1):
        for jj in range(len(y_idx) - 1):
            j0 = y_idx[jj]
            j1 = y_idx[jj + 1]
            diff_avg = 0.25 * (diff[i, j0] + diff[i, j1] + diff[i + 1, j0] + diff[i + 1, j1])
            color = cmap_obj(norm(diff_avg))

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

    if draw_delta_lines:
        env_x_bl = field_px + (field.delta / delta_max * scale) * field_nx
        env_y_bl = field_py + (field.delta / delta_max * scale) * field_ny
        ax.plot(env_x_bl, env_y_bl, "k-", lw=1.2, zorder=5)

        env_x_fl = field_px + (fluent_field.delta / delta_max * scale) * field_nx
        env_y_fl = field_py + (fluent_field.delta / delta_max * scale) * field_ny
        ax.plot(env_x_fl, env_y_fl, "k--", lw=1.2, zorder=5)


def _draw_envelope_absolute_quads(
    ax: Axes,
    field,
    u_arr: NDArray[np.float64],
    delta_arr: NDArray[np.float64],
    surface_x: NDArray[np.float64],
    surface_y: NDArray[np.float64],
    panel_indices: List[int],
    normals: NDArray[np.float64],
    scale: float,
    delta_max: float,
    cmap_obj,
    norm: mcolors.Normalize,
    n_y_vis: int,
    draw_delta_line: bool = True,
    line_style: str = "k-",
) -> None:
    """Draw velocity envelope quads for one BL path using absolute velocity."""
    m_field, ny = field.u.shape
    if m_field < 2:
        return

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

    if ny > n_y_vis:
        y_idx = np.linspace(0, ny - 1, n_y_vis + 1, dtype=int)
    else:
        y_idx = np.arange(ny)

    for i in range(m_field - 1):
        for jj in range(len(y_idx) - 1):
            j0 = y_idx[jj]
            j1 = y_idx[jj + 1]
            u_avg = 0.25 * (u_arr[i, j0] + u_arr[i, j1] + u_arr[i + 1, j0] + u_arr[i + 1, j1])
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

    if draw_delta_line:
        env_x = field_px + (delta_arr / delta_max * scale) * field_nx
        env_y = field_py + (delta_arr / delta_max * scale) * field_ny
        ax.plot(env_x, env_y, line_style, lw=1.2, zorder=5)


def plot_bl_velocity_envelope_comparison(
    bl_result,
    comparison_result,
    profile_name: str = "thwaites",
    scale: float = 0.15,
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    n_y_vis: int = 20,
) -> Tuple[Figure, Axes]:
    """Wrapped velocity-difference envelope for both upper and lower paths."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if not comparison_result.has_fluent_data:
        field_u = bl_result.upper.fields.get(profile_name)
        field_l = bl_result.lower.fields.get(profile_name)
        if field_u is not None and field_l is not None:
            plot_bl_velocity_envelope_two_sides(
                field_u,
                field_l,
                bl_result,
                scale=scale,
                cmap="viridis",
                title=title or f"BL velocity envelope - {profile_name}",
                output_path=None,
                n_y_vis=n_y_vis,
            )
        ax.annotate(
            "Fluent comparison not available",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9),
        )
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    field_upper = bl_result.upper.fields.get(profile_name)
    field_lower = bl_result.lower.fields.get(profile_name)
    fluent_upper = comparison_result.upper_fluent_field
    fluent_lower = comparison_result.lower_fluent_field

    if field_upper is None or field_lower is None:
        ax.text(
            0.5,
            0.5,
            f"BL field not available for profile '{profile_name}'",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="0.4",
        )
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    if fluent_upper is None or fluent_lower is None:
        ax.text(
            0.5,
            0.5,
            "Fluent interpolated fields not available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="0.4",
        )
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    surface_x = bl_result.surface_x
    surface_y = bl_result.surface_y
    normals = compute_outward_normals(surface_x, surface_y, closed=True)
    bx = np.append(surface_x, surface_x[0])
    by = np.append(surface_y, surface_y[0])
    ax.plot(bx, by, "k-", lw=2.0, zorder=10, label="Body")

    diff_upper = field_upper.u - fluent_upper.u
    diff_lower = field_lower.u - fluent_lower.u
    vmax = max(np.nanmax(np.abs(diff_upper)), np.nanmax(np.abs(diff_lower)))
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap_obj = plt.cm.get_cmap(cmap)

    delta_max = max(
        float(np.nanmax(field_upper.delta)) if np.any(field_upper.delta > 0) else 1.0,
        float(np.nanmax(field_lower.delta)) if np.any(field_lower.delta > 0) else 1.0,
    )

    _draw_envelope_comparison_quads(
        ax,
        field_upper,
        fluent_upper,
        surface_x,
        surface_y,
        bl_result.upper.panel_indices,
        normals,
        scale,
        delta_max,
        cmap_obj,
        norm,
        n_y_vis,
        draw_delta_lines=True,
    )
    _draw_envelope_comparison_quads(
        ax,
        field_lower,
        fluent_lower,
        surface_x,
        surface_y,
        bl_result.lower.panel_indices,
        normals,
        scale,
        delta_max,
        cmap_obj,
        norm,
        n_y_vis,
        draw_delta_lines=True,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$u_{\mathrm{BL}} - u_{\mathrm{Fluent}}$ [m/s]", shrink=0.8, pad=0.02)

    ax.plot([], [], "k-", lw=1.2, label="delta (BL)")
    ax.plot([], [], "k--", lw=1.2, label="delta (Fluent)")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(f"Velocity difference envelope - {profile_name} vs Fluent", fontsize=11)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_bl_velocity_contour_normalized_comparison(
    field,
    fluent_field=None,
    cmap: str = "RdBu_r",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    show_colorbar: bool = True,
) -> Tuple[Figure, Axes]:
    """Normalized s-(y/delta) velocity difference contour for Fluent comparison."""
    if fluent_field is None:
        fig, ax_out = plot_bl_velocity_contour_normalized(
            field,
            ax=ax,
            cmap="viridis",
            title=title or f"Normalized BL velocity - {field.profile_name} (no Fluent data)",
        )
        ax_out.annotate(
            "Fluent comparison not available",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9),
        )
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax_out

    m_stations, ny = field.u.shape
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

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

    diff = field.u - fluent_field.u
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

    vmax = np.nanmax(np.abs(diff))
    pcm = ax.pcolormesh(s_grid, eta_edge_grid, diff, cmap=cmap, shading="flat", rasterized=True, vmin=-vmax, vmax=vmax)

    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label=r"$u_{\mathrm{BL}} - u_{\mathrm{Fluent}}$ [m/s]", shrink=0.85, pad=0.02)

    ax.axhline(1.0, color="k", ls="-", lw=1.0, alpha=0.7, label="delta (BL)")
    eta_fluent_delta = np.where(field.delta > 0, fluent_field.delta / field.delta, 1.0)
    ax.plot(field.s, eta_fluent_delta, "k--", lw=1.0, alpha=0.7, label="delta (Fluent)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    ax.set_xlabel("Arc length $s$ [m]")
    ax.set_ylabel(r"$y / \delta$")
    ax.set_xlim(field.s[0], field.s[-1])
    ax.set_ylim(0, None)

    if title:
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(f"Normalized velocity difference - {field.profile_name} vs Fluent", fontsize=11)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_bl_fluent_envelope_side_by_side(
    bl_result,
    comparison_result,
    profile_name: str = "thwaites",
    scale: float = 0.15,
    cmap: str = "viridis",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    n_y_vis: int = 20,
) -> Tuple[Figure, NDArray]:
    """Side-by-side wrapped velocity envelope (BL Solver vs Fluent)."""
    fig, (ax_bl, ax_fl) = plt.subplots(1, 2, figsize=(14, 6))

    if not comparison_result.has_fluent_data:
        ax_fl.text(0.5, 0.5, "Fluent data not available", ha="center", va="center")
        ax_fl.axis("off")
        return fig, np.array([ax_bl, ax_fl])

    field_u = bl_result.upper.fields.get(profile_name)
    field_l = bl_result.lower.fields.get(profile_name)
    fluent_u = comparison_result.upper_fluent_field
    fluent_l = comparison_result.lower_fluent_field

    if None in (field_u, field_l, fluent_u, fluent_l):
        ax_bl.text(0.5, 0.5, "Data unavailable", ha="center", va="center")
        return fig, np.array([ax_bl, ax_fl])

    surface_x = bl_result.surface_x
    surface_y = bl_result.surface_y
    normals = compute_outward_normals(surface_x, surface_y, closed=True)
    bx = np.append(surface_x, surface_x[0])
    by = np.append(surface_y, surface_y[0])

    vmax = 0.0
    vmin = float("inf")
    for f in [field_u, field_l, fluent_u, fluent_l]:
        vmax = max(vmax, float(np.nanmax(f.u)))
        vmin = min(vmin, float(np.nanmin(f.u)))

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.cm.get_cmap(cmap)
    delta_max = max(
        float(np.nanmax(field_u.delta)) if np.any(field_u.delta > 0) else 1.0,
        float(np.nanmax(field_l.delta)) if np.any(field_l.delta > 0) else 1.0,
        float(np.nanmax(fluent_u.delta)) if np.any(fluent_u.delta > 0) else 1.0,
        float(np.nanmax(fluent_l.delta)) if np.any(fluent_l.delta > 0) else 1.0,
    )

    for ax, name in [(ax_bl, "BL Solver"), (ax_fl, "Fluent CFD")]:
        ax.plot(bx, by, "k-", lw=2.0, zorder=10)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

    _draw_envelope_absolute_quads(
        ax_bl,
        field_u,
        field_u.u,
        field_u.delta,
        surface_x,
        surface_y,
        bl_result.upper.panel_indices,
        normals,
        scale,
        delta_max,
        cmap_obj,
        norm,
        n_y_vis,
        line_style="k-",
    )
    _draw_envelope_absolute_quads(
        ax_bl,
        field_l,
        field_l.u,
        field_l.delta,
        surface_x,
        surface_y,
        bl_result.lower.panel_indices,
        normals,
        scale,
        delta_max,
        cmap_obj,
        norm,
        n_y_vis,
        line_style="k-",
    )
    _draw_envelope_absolute_quads(
        ax_fl,
        field_u,
        fluent_u.u,
        fluent_u.delta,
        surface_x,
        surface_y,
        bl_result.upper.panel_indices,
        normals,
        scale,
        delta_max,
        cmap_obj,
        norm,
        n_y_vis,
        line_style="k--",
    )
    _draw_envelope_absolute_quads(
        ax_fl,
        field_l,
        fluent_l.u,
        fluent_l.delta,
        surface_x,
        surface_y,
        bl_result.lower.panel_indices,
        normals,
        scale,
        delta_max,
        cmap_obj,
        norm,
        n_y_vis,
        line_style="k--",
    )

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_bl, ax_fl], label="u [m/s]", shrink=0.8, pad=0.02)

    ax_bl.plot([], [], "k-", lw=1.2, label="delta")
    ax_bl.legend(loc="upper right", fontsize=8)
    ax_fl.plot([], [], "k--", lw=1.2, label="delta")
    ax_fl.legend(loc="upper right", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, np.array([ax_bl, ax_fl])


def plot_bl_fluent_contour_side_by_side(
    field,
    fluent_field,
    cmap: str = "viridis",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, NDArray]:
    """Side-by-side s-y absolute velocity contour (BL Solver vs Fluent)."""
    fig, (ax_bl, ax_fl) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

    m_stations, ny = field.u.shape
    if fluent_field is None:
        ax_fl.text(0.5, 0.5, "Fluent data not available", ha="center", va="center")
        ax_fl.axis("off")
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, np.array([ax_bl, ax_fl])

    vmax = max(float(np.nanmax(field.u)), float(np.nanmax(fluent_field.u)))
    vmin = min(float(np.nanmin(field.u)), float(np.nanmin(fluent_field.u)))
    s_edges = _cell_edges(field.s)
    y_edge_grid = np.zeros((m_stations + 1, ny + 1), dtype=np.float64)
    for i in range(m_stations):
        y_edge_grid[i] = _cell_edges(field.y[i])
    y_edge_grid[m_stations] = y_edge_grid[m_stations - 1]
    s_grid = np.broadcast_to(s_edges[:, np.newaxis], (m_stations + 1, ny + 1))

    ax_bl.pcolormesh(s_grid, y_edge_grid, field.u, cmap=cmap, shading="flat", rasterized=True, vmin=vmin, vmax=vmax)
    ax_bl.plot(field.s, field.delta, "w-", lw=1.5, label="delta (BL)", alpha=0.7)
    ax_bl.legend(loc="upper left", fontsize=8, framealpha=0.8)

    pcm = ax_fl.pcolormesh(
        s_grid,
        y_edge_grid,
        fluent_field.u,
        cmap=cmap,
        shading="flat",
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
    )
    ax_fl.plot(fluent_field.s, fluent_field.delta, "w--", lw=1.5, label="delta (Fluent)", alpha=0.7)
    ax_fl.legend(loc="upper left", fontsize=8, framealpha=0.8)

    fig.colorbar(pcm, ax=[ax_bl, ax_fl], label="u [m/s]", shrink=0.8)
    ax_bl.set_title("BL Solver")
    ax_fl.set_title("Fluent CFD")
    ax_fl.set_xlabel("Arc length $s$ [m]")
    ax_bl.set_ylabel("Wall-normal $y$ [m]")
    ax_fl.set_ylabel("Wall-normal $y$ [m]")
    ax_bl.set_xlim(field.s[0], field.s[-1])
    ax_bl.set_ylim(0, None)

    if title:
        fig.suptitle(title, fontsize=12)
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, np.array([ax_bl, ax_fl])


def plot_bl_fluent_contour_normalized_side_by_side(
    field,
    fluent_field,
    cmap: str = "viridis",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, NDArray]:
    """Side-by-side normalized s-(y/delta) velocity contour (BL Solver vs Fluent)."""
    fig, (ax_bl, ax_fl) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

    m_stations, ny = field.u.shape
    if fluent_field is None:
        ax_fl.text(0.5, 0.5, "Fluent data not available", ha="center", va="center")
        ax_fl.axis("off")
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, np.array([ax_bl, ax_fl])

    vmax = max(float(np.nanmax(field.u)), float(np.nanmax(fluent_field.u)))
    vmin = min(float(np.nanmin(field.u)), float(np.nanmin(fluent_field.u)))
    s_edges = _cell_edges(field.s)
    s_grid = np.broadcast_to(s_edges[:, np.newaxis], (m_stations + 1, ny + 1))

    eta_bl = np.zeros_like(field.y)
    for i in range(m_stations):
        if field.delta[i] > 0:
            eta_bl[i] = field.y[i] / field.delta[i]
        else:
            eta_bl[i] = np.linspace(0.0, 1.0, ny)
    eta_edge_bl = np.zeros((m_stations + 1, ny + 1), dtype=np.float64)
    for i in range(m_stations):
        eta_edge_bl[i] = _cell_edges(eta_bl[i])
    eta_edge_bl[m_stations] = eta_edge_bl[m_stations - 1]

    ax_bl.pcolormesh(s_grid, eta_edge_bl, field.u, cmap=cmap, shading="flat", rasterized=True, vmin=vmin, vmax=vmax)
    ax_bl.axhline(1.0, color="w", ls="-", lw=1.5, label="delta (BL)", alpha=0.7)
    ax_bl.legend(loc="upper left", fontsize=8, framealpha=0.8)

    eta_fl = np.zeros_like(field.y)
    for i in range(m_stations):
        if fluent_field.delta[i] > 0:
            eta_fl[i] = field.y[i] / fluent_field.delta[i]
        else:
            eta_fl[i] = np.linspace(0.0, 1.0, ny)
    eta_edge_fl = np.zeros((m_stations + 1, ny + 1), dtype=np.float64)
    for i in range(m_stations):
        eta_edge_fl[i] = _cell_edges(eta_fl[i])
    eta_edge_fl[m_stations] = eta_edge_fl[m_stations - 1]

    pcm = ax_fl.pcolormesh(
        s_grid,
        eta_edge_fl,
        fluent_field.u,
        cmap=cmap,
        shading="flat",
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
    )
    ax_fl.axhline(1.0, color="w", ls="--", lw=1.5, label="delta (Fluent)", alpha=0.7)
    ax_fl.legend(loc="upper left", fontsize=8, framealpha=0.8)

    fig.colorbar(pcm, ax=[ax_bl, ax_fl], label="u [m/s]", shrink=0.8)
    ax_bl.set_title("BL Solver")
    ax_fl.set_title("Fluent CFD")
    ax_fl.set_xlabel("Arc length $s$ [m]")
    ax_bl.set_ylabel(r"$y / \delta$")
    ax_fl.set_ylabel(r"$y / \delta$")
    ax_bl.set_xlim(field.s[0], field.s[-1])
    ax_bl.set_ylim(0, None)

    if title:
        fig.suptitle(title, fontsize=12)
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, np.array([ax_bl, ax_fl])


def plot_bl_comparison_report(
    comparison_result,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, NDArray]:
    """Comprehensive metrics report for BL solver vs Fluent comparison."""
    if not comparison_result.has_fluent_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "Fluent comparison data not available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="0.4",
        )
        ax.axis("off")
        if title:
            fig.suptitle(title, fontsize=12)
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, np.array([[ax]])

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], hspace=0.3, wspace=0.25)
    ax_bar_upper = fig.add_subplot(gs[0, 0])
    ax_bar_lower = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    qty_colors = {"Ue": "#1f77b4", "Cf": "#ff7f0e", "delta": "#2ca02c", "u": "#d62728"}
    table_data = []
    table_cols = ["Side", "Quantity", "RMS", "MAE", "Linf", "Rel. L2 (%)", "N points"]
    bar_data = {"upper": {}, "lower": {}}

    for side in ["upper", "lower"]:
        if side in comparison_result.wall_metrics:
            for qty, metrics in comparison_result.wall_metrics[side].items():
                table_data.append(
                    [
                        side.capitalize(),
                        qty,
                        f"{metrics.RMS:.4g}",
                        f"{metrics.MAE:.4g}",
                        f"{metrics.L_inf:.4g}",
                        f"{metrics.relative_L2 * 100:.1f}" if not np.isnan(metrics.relative_L2) else "N/A",
                        str(metrics.n_points),
                    ]
                )
                if not np.isnan(metrics.relative_L2):
                    bar_data[side][qty] = metrics.relative_L2 * 100

        if side in comparison_result.velocity_metrics:
            for qty, metrics in comparison_result.velocity_metrics[side].items():
                table_data.append(
                    [
                        side.capitalize(),
                        f"{qty} (field)",
                        f"{metrics.RMS:.4g}",
                        f"{metrics.MAE:.4g}",
                        f"{metrics.L_inf:.4g}",
                        f"{metrics.relative_L2 * 100:.1f}" if not np.isnan(metrics.relative_L2) else "N/A",
                        str(metrics.n_points),
                    ]
                )
                if not np.isnan(metrics.relative_L2):
                    bar_data[side][f"{qty}_field"] = metrics.relative_L2 * 100

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

            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax_bar.annotate(
                    f"{val:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        else:
            ax_bar.text(0.5, 0.5, "No metrics available", transform=ax_bar.transAxes, ha="center", va="center", fontsize=11, color="0.4")
            ax_bar.set_title(f"{side.capitalize()} Side", fontsize=11)

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
        for j, _col in enumerate(table_cols):
            cell = table[(0, j)]
            cell.set_text_props(weight="bold")
    else:
        ax_table.text(0.5, 0.5, "No metrics computed", transform=ax_table.transAxes, ha="center", va="center", fontsize=11, color="0.4")

    if title:
        fig.suptitle(title, fontsize=13, y=0.98)
    else:
        fig.suptitle(f"BL Solver vs Fluent - {comparison_result.profile_name}", fontsize=13, y=0.98)

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, np.array([[ax_bar_upper, ax_bar_lower], [ax_table, ax_table]])


def plot_bl_of_comparison(
    field,
    of_field=None,
    cmap: str = "RdBu_r",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """Alias for :func:`plot_bl_fluent_comparison` for compatibility."""
    return plot_bl_fluent_comparison(field, of_field, cmap=cmap, ax=ax, title=title, output_path=output_path)
