"""Fluent comparison plots for thermal boundary-layer validation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from visualization.bl_plot_common import _cell_edges
from visualization.surface_envelope import (
    compute_outward_normals,
    plot_dual_surface_envelope,
    plot_surface_envelope,
)


_WALL_META = {
    "wall_temperature": {"label": r"$T_w$", "unit": "K"},
    "heat_transfer_coeff": {"label": r"$h$", "unit": "W/m^2K"},
}


def _quantity_label(quantity: str) -> str:
    meta = _WALL_META.get(quantity, {"label": quantity, "unit": ""})
    return f"{meta['label']} [{meta['unit']}]" if meta["unit"] else meta["label"]


def _wall_full_arrays(comparison_result, quantity: str) -> tuple[np.ndarray, np.ndarray]:
    surface_x = comparison_result.bl_result.surface_x
    n = len(surface_x)
    bl_full = np.full(n, np.nan, dtype=np.float64)
    fl_full = np.full(n, np.nan, dtype=np.float64)

    if quantity not in ("wall_temperature", "heat_transfer_coeff"):
        raise ValueError(f"Unsupported thermal wall quantity '{quantity}'")

    upper_bl = np.asarray(getattr(comparison_result.upper_thermal_result, quantity), dtype=np.float64)
    lower_bl = np.asarray(getattr(comparison_result.lower_thermal_result, quantity), dtype=np.float64)
    for i, idx in enumerate(comparison_result.upper_panel_indices):
        if i < len(upper_bl) and idx < n:
            bl_full[idx] = upper_bl[i]
    for i, idx in enumerate(comparison_result.lower_panel_indices):
        if i < len(lower_bl) and idx < n:
            bl_full[idx] = lower_bl[i]

    if comparison_result.fluent_wall_result is not None:
        upper_fl = np.asarray(getattr(comparison_result.fluent_wall_result.upper, quantity), dtype=np.float64)
        lower_fl = np.asarray(getattr(comparison_result.fluent_wall_result.lower, quantity), dtype=np.float64)
        for i, idx in enumerate(comparison_result.upper_panel_indices):
            if i < len(upper_fl) and idx < n:
                fl_full[idx] = upper_fl[i]
        for i, idx in enumerate(comparison_result.lower_panel_indices):
            if i < len(lower_fl) and idx < n:
                fl_full[idx] = lower_fl[i]

    return bl_full, fl_full


def plot_thermal_wall_envelope_side_by_side(
    comparison_result,
    quantity: str = "wall_temperature",
    scale: float = 0.15,
    cmap: str = "coolwarm",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, np.ndarray]:
    """Side-by-side wall-quantity envelope (Thermal solver vs Fluent)."""
    fig, (ax_bl, ax_fl) = plt.subplots(1, 2, figsize=(14, 6))
    if not comparison_result.has_fluent_data:
        ax_fl.text(0.5, 0.5, "Fluent data not available", ha="center", va="center")
        ax_fl.axis("off")
        return fig, np.array([ax_bl, ax_fl])

    surface_x = comparison_result.bl_result.surface_x
    surface_y = comparison_result.bl_result.surface_y
    normals = compute_outward_normals(surface_x, surface_y, closed=True)
    bl_full, fl_full = _wall_full_arrays(comparison_result, quantity)

    bl_plot = np.where(np.isnan(bl_full), 0.0, bl_full)
    fl_plot = np.where(np.isnan(fl_full), 0.0, fl_full)
    all_vals = np.concatenate([bl_plot, fl_plot])
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))
    qty_label = _quantity_label(quantity)

    for ax, vals, name in [(ax_bl, bl_plot, "Thermal Solver"), (ax_fl, fl_plot, "Fluent CFD")]:
        plot_surface_envelope(
            surface_x,
            surface_y,
            vals,
            normals=normals,
            ax=ax,
            scale=scale,
            quantity_name=qty_label,
            colormap=cmap,
            vmin=vmin,
            vmax=vmax,
            show_colorbar=False,
            title=name,
        )

    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap), norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_bl, ax_fl], label=qty_label, shrink=0.8, pad=0.02)

    if title:
        fig.suptitle(title, fontsize=12)
    else:
        fig.suptitle(f"{_WALL_META[quantity]['label']} envelope - Thermal vs Fluent", fontsize=12)
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, np.array([ax_bl, ax_fl])


def plot_thermal_wall_envelope_overlay(
    comparison_result,
    quantity: str = "wall_temperature",
    scale: float = 0.15,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    show_difference: bool = True,
) -> Tuple[Figure, Axes]:
    """Overlay wall-quantity envelope (Thermal solver and Fluent)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    if not comparison_result.has_fluent_data:
        ax.text(0.5, 0.5, "Fluent data not available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax

    surface_x = comparison_result.bl_result.surface_x
    surface_y = comparison_result.bl_result.surface_y
    bl_full, fl_full = _wall_full_arrays(comparison_result, quantity)
    bl_plot = np.where(np.isnan(bl_full), 0.0, bl_full)
    fl_plot = np.where(np.isnan(fl_full), 0.0, fl_full)

    plot_dual_surface_envelope(
        surface_x,
        surface_y,
        bl_plot,
        fl_plot,
        label1="Thermal Solver",
        label2="Fluent CFD",
        ax=ax,
        scale=scale,
        quantity_name=_quantity_label(quantity),
        color1="#1f77b4",
        color2="#d62728",
        show_difference=show_difference,
        title=title or f"{_WALL_META[quantity]['label']} envelope - Thermal vs Fluent",
        invert_values=False,
    )
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_thermal_wall_line_comparison(
    comparison_result,
    quantity: str = "wall_temperature",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, np.ndarray]:
    """Line comparison of thermal wall quantities vs arc length for both sides."""
    if quantity not in ("wall_temperature", "heat_transfer_coeff"):
        raise ValueError(f"Unsupported thermal wall quantity '{quantity}'")

    fig, (ax_u, ax_l) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    axes = np.array([ax_u, ax_l])
    qty_label = _quantity_label(quantity)

    side_data = [
        ("upper", comparison_result.upper_thermal_result, ax_u),
        ("lower", comparison_result.lower_thermal_result, ax_l),
    ]

    for side, thermal_result, ax in side_data:
        s_bl = np.asarray(thermal_result.arc_length)
        v_bl = np.asarray(getattr(thermal_result, quantity))
        ax.plot(s_bl, v_bl, "-", lw=2.0, color="#1f77b4", label="Thermal Solver")

        if comparison_result.fluent_wall_result is not None:
            fl_path = comparison_result.fluent_wall_result.sides[side]
            s_fl = np.asarray(fl_path.s)
            v_fl = np.asarray(getattr(fl_path, quantity))
            ax.plot(s_fl, v_fl, "--", lw=2.0, color="#d62728", label="Fluent CFD")

        ax.set_ylabel(qty_label)
        ax.set_title(f"{side.capitalize()} side", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("Arc length $s$ [m]")
    if title is None:
        title = f"{_WALL_META[quantity]['label']} vs arc length - Thermal vs Fluent"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, axes


def _plot_sy_contour(
    s: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    *,
    ax: Axes,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> any:
    m, ny = values.shape
    s_edges = _cell_edges(s)
    y_edge_grid = np.zeros((m + 1, ny + 1), dtype=np.float64)
    for i in range(m):
        y_edge_grid[i] = _cell_edges(y_grid[i])
    y_edge_grid[m] = y_edge_grid[m - 1]
    s_grid = np.broadcast_to(s_edges[:, np.newaxis], (m + 1, ny + 1))
    return ax.pcolormesh(s_grid, y_edge_grid, values, cmap=cmap, shading="flat", rasterized=True, vmin=vmin, vmax=vmax)


def plot_thermal_fluent_contour_side_by_side(
    thermal_result,
    fluent_field,
    cmap: str = "coolwarm",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, np.ndarray]:
    """Side-by-side absolute temperature contour on s-y coordinates."""
    fig, (ax_bl, ax_fl) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    if not thermal_result.has_field or thermal_result.field is None:
        ax_bl.text(0.5, 0.5, "Thermal field data not available", transform=ax_bl.transAxes, ha="center", va="center")
        return fig, np.array([ax_bl, ax_fl])
    if fluent_field is None:
        ax_fl.text(0.5, 0.5, "Fluent field data not available", transform=ax_fl.transAxes, ha="center", va="center")
        return fig, np.array([ax_bl, ax_fl])

    field = thermal_result.field
    vmax = max(float(np.nanmax(field.T)), float(np.nanmax(fluent_field.T)))
    vmin = min(float(np.nanmin(field.T)), float(np.nanmin(fluent_field.T)))
    _plot_sy_contour(field.s, field.y_normal, field.T, ax=ax_bl, cmap=cmap, vmin=vmin, vmax=vmax)
    pcm = _plot_sy_contour(field.s, fluent_field.y, fluent_field.T, ax=ax_fl, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_bl.plot(field.s, thermal_result.thermal_bl_thickness, "w-", lw=1.2, alpha=0.8, label=r"$\delta_T$ solver")
    ax_fl.plot(field.s, fluent_field.delta, "w--", lw=1.2, alpha=0.8, label=r"$\delta_T$ Fluent")
    ax_bl.legend(loc="upper left", fontsize=8)
    ax_fl.legend(loc="upper left", fontsize=8)
    fig.colorbar(pcm, ax=[ax_bl, ax_fl], label=r"$T$ [K]", shrink=0.8)
    ax_bl.set_title("Thermal Solver")
    ax_fl.set_title("Fluent CFD")
    ax_fl.set_xlabel("Arc length $s$ [m]")
    ax_bl.set_ylabel("Wall-normal $y$ [m]")
    ax_fl.set_ylabel("Wall-normal $y$ [m]")
    if title:
        fig.suptitle(title, fontsize=12)
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, np.array([ax_bl, ax_fl])


def plot_thermal_fluent_contour_difference(
    thermal_result,
    fluent_field,
    cmap: str = "RdBu_r",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """Temperature difference contour (thermal solver - Fluent) on s-y."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()
    if not thermal_result.has_field or thermal_result.field is None or fluent_field is None:
        ax.text(0.5, 0.5, "Missing thermal/fluent field data", transform=ax.transAxes, ha="center", va="center")
        return fig, ax

    field = thermal_result.field
    diff = field.T - fluent_field.T
    vmax = float(np.nanmax(np.abs(diff)))
    pcm = _plot_sy_contour(field.s, field.y_normal, diff, ax=ax, cmap=cmap, vmin=-vmax, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label=r"$T_{solver} - T_{Fluent}$ [K]", shrink=0.85, pad=0.02)
    ax.plot(field.s, thermal_result.thermal_bl_thickness, "k-", lw=1.2, label=r"$\delta_T$ solver")
    ax.plot(field.s, fluent_field.delta, "k--", lw=1.2, label=r"$\delta_T$ Fluent")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Arc length $s$ [m]")
    ax.set_ylabel("Wall-normal $y$ [m]")
    ax.set_xlim(field.s[0], field.s[-1])
    ax.set_ylim(0, None)
    ax.set_title(title or "Temperature difference - Thermal vs Fluent", fontsize=11)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, ax


def _eta_grid(y_grid: np.ndarray, delta: np.ndarray) -> np.ndarray:
    eta = np.zeros_like(y_grid)
    for i in range(y_grid.shape[0]):
        if i < len(delta) and delta[i] > 0:
            eta[i] = y_grid[i] / delta[i]
        else:
            eta[i] = np.linspace(0.0, 1.0, y_grid.shape[1])
    return eta


def plot_thermal_fluent_contour_normalized_side_by_side(
    thermal_result,
    fluent_field,
    cmap: str = "coolwarm",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, np.ndarray]:
    """Side-by-side absolute temperature contour on normalized s-eta coordinates."""
    fig, (ax_bl, ax_fl) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    if not thermal_result.has_field or thermal_result.field is None:
        ax_bl.text(0.5, 0.5, "Thermal field data not available", transform=ax_bl.transAxes, ha="center", va="center")
        return fig, np.array([ax_bl, ax_fl])
    if fluent_field is None:
        ax_fl.text(0.5, 0.5, "Fluent field data not available", transform=ax_fl.transAxes, ha="center", va="center")
        return fig, np.array([ax_bl, ax_fl])

    field = thermal_result.field
    eta_bl = _eta_grid(field.y_normal, thermal_result.thermal_bl_thickness)
    eta_fl = _eta_grid(fluent_field.y, fluent_field.delta)
    vmax = max(float(np.nanmax(field.T)), float(np.nanmax(fluent_field.T)))
    vmin = min(float(np.nanmin(field.T)), float(np.nanmin(fluent_field.T)))
    _plot_sy_contour(field.s, eta_bl, field.T, ax=ax_bl, cmap=cmap, vmin=vmin, vmax=vmax)
    pcm = _plot_sy_contour(field.s, eta_fl, fluent_field.T, ax=ax_fl, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_bl.axhline(1.0, color="w", ls="-", lw=1.2, alpha=0.8)
    ax_fl.axhline(1.0, color="w", ls="--", lw=1.2, alpha=0.8)
    fig.colorbar(pcm, ax=[ax_bl, ax_fl], label=r"$T$ [K]", shrink=0.8)
    ax_bl.set_title("Thermal Solver")
    ax_fl.set_title("Fluent CFD")
    ax_fl.set_xlabel("Arc length $s$ [m]")
    ax_bl.set_ylabel(r"$\eta = y / \delta_T$")
    ax_fl.set_ylabel(r"$\eta = y / \delta_T$")
    if title:
        fig.suptitle(title, fontsize=12)
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, np.array([ax_bl, ax_fl])


def plot_thermal_fluent_contour_normalized_difference(
    thermal_result,
    fluent_field,
    cmap: str = "RdBu_r",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """Temperature difference contour on normalized s-eta coordinates."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()
    if not thermal_result.has_field or thermal_result.field is None or fluent_field is None:
        ax.text(0.5, 0.5, "Missing thermal/fluent field data", transform=ax.transAxes, ha="center", va="center")
        return fig, ax

    field = thermal_result.field
    diff = field.T - fluent_field.T
    eta_bl = _eta_grid(field.y_normal, thermal_result.thermal_bl_thickness)
    vmax = float(np.nanmax(np.abs(diff)))
    pcm = _plot_sy_contour(field.s, eta_bl, diff, ax=ax, cmap=cmap, vmin=-vmax, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label=r"$T_{solver} - T_{Fluent}$ [K]", shrink=0.85, pad=0.02)
    ax.axhline(1.0, color="k", ls="-", lw=1.0, alpha=0.7, label=r"$\delta_T$ solver")
    ratio = np.where(thermal_result.thermal_bl_thickness > 0, fluent_field.delta / thermal_result.thermal_bl_thickness, 1.0)
    ax.plot(field.s, ratio, "k--", lw=1.0, alpha=0.7, label=r"$\delta_T$ Fluent")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Arc length $s$ [m]")
    ax.set_ylabel(r"$\eta = y / \delta_T$")
    ax.set_xlim(field.s[0], field.s[-1])
    ax.set_ylim(0, None)
    ax.set_title(title or "Normalized temperature difference - Thermal vs Fluent", fontsize=11)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, ax


def _draw_temperature_envelope_quads(
    ax: Axes,
    thermal_result,
    T_arr: np.ndarray,
    delta_arr: np.ndarray,
    surface_x: np.ndarray,
    surface_y: np.ndarray,
    panel_indices: list[int],
    normals: np.ndarray,
    scale: float,
    delta_max: float,
    cmap_obj,
    norm: mcolors.Normalize,
    n_y_vis: int,
    line_style: str,
) -> None:
    if not thermal_result.has_field or thermal_result.field is None:
        return
    field = thermal_result.field
    m, ny = field.T.shape
    if m < 2:
        return

    path_x = surface_x[panel_indices]
    path_y = surface_y[panel_indices]
    path_s = np.zeros(len(panel_indices), dtype=np.float64)
    path_s[1:] = np.cumsum(np.hypot(np.diff(path_x), np.diff(path_y)))
    path_normals = normals[panel_indices]
    interp_s = path_s - path_s[0] + field.s[0]

    px = np.interp(field.s, interp_s, path_x)
    py = np.interp(field.s, interp_s, path_y)
    nx = np.interp(field.s, interp_s, path_normals[:, 0])
    nyv = np.interp(field.s, interp_s, path_normals[:, 1])
    nlen = np.hypot(nx, nyv)
    nlen = np.where(nlen < 1e-12, 1.0, nlen)
    nx /= nlen
    nyv /= nlen

    if ny > n_y_vis:
        y_idx = np.linspace(0, ny - 1, n_y_vis + 1, dtype=int)
    else:
        y_idx = np.arange(ny)

    for i in range(m - 1):
        for jj in range(len(y_idx) - 1):
            j0 = y_idx[jj]
            j1 = y_idx[jj + 1]
            t_avg = 0.25 * (T_arr[i, j0] + T_arr[i, j1] + T_arr[i + 1, j0] + T_arr[i + 1, j1])
            color = cmap_obj(norm(t_avg))

            d00 = field.y_normal[i, j0] / delta_max * scale
            d01 = field.y_normal[i, j1] / delta_max * scale
            d10 = field.y_normal[i + 1, j0] / delta_max * scale
            d11 = field.y_normal[i + 1, j1] / delta_max * scale

            quad_x = [px[i] + d00 * nx[i], px[i] + d01 * nx[i], px[i + 1] + d11 * nx[i + 1], px[i + 1] + d10 * nx[i + 1]]
            quad_y = [py[i] + d00 * nyv[i], py[i] + d01 * nyv[i], py[i + 1] + d11 * nyv[i + 1], py[i + 1] + d10 * nyv[i + 1]]
            ax.fill(quad_x, quad_y, color=color, edgecolor="none", zorder=2)

    env_x = px + (delta_arr / delta_max * scale) * nx
    env_y = py + (delta_arr / delta_max * scale) * nyv
    ax.plot(env_x, env_y, line_style, lw=1.2, zorder=5)


def plot_thermal_fluent_envelope_side_by_side(
    comparison_result,
    scale: float = 0.15,
    cmap: str = "coolwarm",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    n_y_vis: int = 20,
) -> Tuple[Figure, np.ndarray]:
    """Side-by-side wrapped thermal-field envelope (absolute temperature)."""
    fig, (ax_bl, ax_fl) = plt.subplots(1, 2, figsize=(14, 6))
    if not comparison_result.has_fluent_data:
        ax_fl.text(0.5, 0.5, "Fluent data not available", ha="center", va="center")
        ax_fl.axis("off")
        return fig, np.array([ax_bl, ax_fl])

    if comparison_result.upper_fluent_field is None or comparison_result.lower_fluent_field is None:
        ax_fl.text(0.5, 0.5, "Fluent thermal field unavailable", ha="center", va="center")
        ax_fl.axis("off")
        return fig, np.array([ax_bl, ax_fl])

    surface_x = comparison_result.bl_result.surface_x
    surface_y = comparison_result.bl_result.surface_y
    normals = compute_outward_normals(surface_x, surface_y, closed=True)
    bx = np.append(surface_x, surface_x[0])
    by = np.append(surface_y, surface_y[0])

    fields = [
        comparison_result.upper_thermal_result.field,
        comparison_result.lower_thermal_result.field,
        comparison_result.upper_fluent_field,
        comparison_result.lower_fluent_field,
    ]
    tmin = min(float(np.nanmin(f.T)) for f in fields if f is not None)
    tmax = max(float(np.nanmax(f.T)) for f in fields if f is not None)
    norm = mcolors.Normalize(vmin=tmin, vmax=tmax)
    cmap_obj = plt.cm.get_cmap(cmap)
    delta_max = max(
        float(np.nanmax(comparison_result.upper_thermal_result.thermal_bl_thickness)),
        float(np.nanmax(comparison_result.lower_thermal_result.thermal_bl_thickness)),
        float(np.nanmax(comparison_result.upper_fluent_field.delta)),
        float(np.nanmax(comparison_result.lower_fluent_field.delta)),
    )

    for ax, name in [(ax_bl, "Thermal Solver"), (ax_fl, "Fluent CFD")]:
        ax.plot(bx, by, "k-", lw=2.0, zorder=10)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.grid(True, alpha=0.3)

    _draw_temperature_envelope_quads(
        ax_bl,
        comparison_result.upper_thermal_result,
        comparison_result.upper_thermal_result.field.T,
        comparison_result.upper_thermal_result.thermal_bl_thickness,
        surface_x,
        surface_y,
        comparison_result.upper_panel_indices,
        normals,
        scale,
        delta_max,
        cmap_obj,
        norm,
        n_y_vis,
        "k-",
    )
    _draw_temperature_envelope_quads(
        ax_bl,
        comparison_result.lower_thermal_result,
        comparison_result.lower_thermal_result.field.T,
        comparison_result.lower_thermal_result.thermal_bl_thickness,
        surface_x,
        surface_y,
        comparison_result.lower_panel_indices,
        normals,
        scale,
        delta_max,
        cmap_obj,
        norm,
        n_y_vis,
        "k-",
    )
    _draw_temperature_envelope_quads(
        ax_fl,
        comparison_result.upper_thermal_result,
        comparison_result.upper_fluent_field.T,
        comparison_result.upper_fluent_field.delta,
        surface_x,
        surface_y,
        comparison_result.upper_panel_indices,
        normals,
        scale,
        delta_max,
        cmap_obj,
        norm,
        n_y_vis,
        "k--",
    )
    _draw_temperature_envelope_quads(
        ax_fl,
        comparison_result.lower_thermal_result,
        comparison_result.lower_fluent_field.T,
        comparison_result.lower_fluent_field.delta,
        surface_x,
        surface_y,
        comparison_result.lower_panel_indices,
        normals,
        scale,
        delta_max,
        cmap_obj,
        norm,
        n_y_vis,
        "k--",
    )

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_bl, ax_fl], label=r"$T$ [K]", shrink=0.8, pad=0.02)
    ax_bl.plot([], [], "k-", lw=1.2, label=r"$\delta_T$")
    ax_fl.plot([], [], "k--", lw=1.2, label=r"$\delta_T$")
    ax_bl.legend(loc="upper right", fontsize=8)
    ax_fl.legend(loc="upper right", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=12)
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, np.array([ax_bl, ax_fl])


def plot_thermal_fluent_envelope_difference(
    comparison_result,
    scale: float = 0.15,
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    n_y_vis: int = 20,
) -> Tuple[Figure, Axes]:
    """Wrapped thermal-field envelope difference (solver - Fluent)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    if (
        not comparison_result.has_fluent_data
        or comparison_result.upper_fluent_field is None
        or comparison_result.lower_fluent_field is None
    ):
        ax.text(0.5, 0.5, "Fluent thermal field unavailable", transform=ax.transAxes, ha="center", va="center")
        return fig, ax

    surface_x = comparison_result.bl_result.surface_x
    surface_y = comparison_result.bl_result.surface_y
    normals = compute_outward_normals(surface_x, surface_y, closed=True)
    bx = np.append(surface_x, surface_x[0])
    by = np.append(surface_y, surface_y[0])
    ax.plot(bx, by, "k-", lw=2.0, zorder=10, label="Body")

    diff_u = comparison_result.upper_thermal_result.field.T - comparison_result.upper_fluent_field.T
    diff_l = comparison_result.lower_thermal_result.field.T - comparison_result.lower_fluent_field.T
    vmax = max(float(np.nanmax(np.abs(diff_u))), float(np.nanmax(np.abs(diff_l))))
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap_obj = plt.cm.get_cmap(cmap)
    delta_max = max(
        float(np.nanmax(comparison_result.upper_thermal_result.thermal_bl_thickness)),
        float(np.nanmax(comparison_result.lower_thermal_result.thermal_bl_thickness)),
    )

    _draw_temperature_envelope_quads(
        ax,
        comparison_result.upper_thermal_result,
        diff_u,
        comparison_result.upper_thermal_result.thermal_bl_thickness,
        surface_x,
        surface_y,
        comparison_result.upper_panel_indices,
        normals,
        scale,
        delta_max,
        cmap_obj,
        norm,
        n_y_vis,
        "k-",
    )
    _draw_temperature_envelope_quads(
        ax,
        comparison_result.lower_thermal_result,
        diff_l,
        comparison_result.lower_thermal_result.thermal_bl_thickness,
        surface_x,
        surface_y,
        comparison_result.lower_panel_indices,
        normals,
        scale,
        delta_max,
        cmap_obj,
        norm,
        n_y_vis,
        "k-",
    )

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$T_{solver} - T_{Fluent}$ [K]", shrink=0.8, pad=0.02)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.grid(True, alpha=0.3)
    ax.set_title(title or "Temperature difference envelope - Thermal vs Fluent", fontsize=11)
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, ax
