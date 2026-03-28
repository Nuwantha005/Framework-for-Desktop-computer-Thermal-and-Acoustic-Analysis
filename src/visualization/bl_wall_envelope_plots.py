"""Wall quantity envelope plots for BL solver vs Fluent comparison.

This module provides envelope plots for wall quantities (Ue, Cf, delta, Cp)
wrapped around the body geometry, comparing BL solver results with Fluent CFD.

Plot types:
- Side-by-side: BL solver on left, Fluent on right
- Overlay: Both results on the same body
- Grid: 2x2 grid of all four quantities
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from visualization.surface_envelope import (
    compute_outward_normals,
    plot_dual_surface_envelope,
    plot_surface_envelope,
)


# Quantity metadata: label, units, and whether to invert for envelope display
_WALL_QUANTITY_META = {
    "Ue": {"label": r"$U_e$", "unit": "m/s", "invert": False},
    "Cf": {"label": r"$C_f$", "unit": "-", "invert": False},
    "delta": {"label": r"$\delta$", "unit": "m", "invert": False},
    "Cp": {"label": r"$C_p$", "unit": "-", "invert": True},  # suction peaks outward
}


def _get_wall_quantity(
    bl_path,
    fluent_path,
    quantity: str,
    profile_name: Optional[str] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Extract wall quantity arrays from BL and Fluent path results.

    Returns:
        (bl_s, bl_val, fl_s, fl_val) — arc-length and value arrays for each source.
        Arrays may be empty if the quantity is unavailable.
    """
    # BL solver data
    if quantity == "Ue":
        bl_s, bl_val = bl_path.s, bl_path.Ue
    elif quantity == "Cf":
        # Cf is stored in the profile result
        if profile_name and profile_name in bl_path.results:
            res = bl_path.results[profile_name]
            bl_s, bl_val = res.s, res.cf
        elif bl_path.results:
            pname = list(bl_path.results.keys())[0]
            res = bl_path.results[pname]
            bl_s, bl_val = res.s, res.cf
        else:
            bl_s, bl_val = np.array([]), np.array([])
    elif quantity == "delta":
        if profile_name and profile_name in bl_path.fields:
            field = bl_path.fields[profile_name]
            bl_s, bl_val = field.s, field.delta
        elif bl_path.fields:
            pname = list(bl_path.fields.keys())[0]
            field = bl_path.fields[pname]
            bl_s, bl_val = field.s, field.delta
        else:
            bl_s, bl_val = np.array([]), np.array([])
    elif quantity == "Cp":
        # Cp from Bernoulli: Cp = 1 - (Ue / U_inf)^2
        # We need U_inf from the BL result; assume it's max(Ue) for now
        Ue = bl_path.Ue
        U_inf = float(np.nanmax(Ue)) if len(Ue) > 0 else 1.0
        bl_s = bl_path.s
        bl_val = 1.0 - (Ue / U_inf) ** 2 if U_inf > 0 else np.zeros_like(Ue)
    else:
        bl_s, bl_val = np.array([]), np.array([])

    # Fluent data
    if quantity == "Ue":
        fl_s, fl_val = fluent_path.s, fluent_path.Ue
    elif quantity == "Cf":
        fl_s, fl_val = fluent_path.s, fluent_path.Cf
    elif quantity == "delta":
        fl_s, fl_val = fluent_path.s, fluent_path.delta
    elif quantity == "Cp":
        Ue_fl = fluent_path.Ue
        U_inf_fl = float(np.nanmax(Ue_fl)) if len(Ue_fl) > 0 else 1.0
        fl_s = fluent_path.s
        fl_val = 1.0 - (Ue_fl / U_inf_fl) ** 2 if U_inf_fl > 0 else np.zeros_like(Ue_fl)
    else:
        fl_s, fl_val = np.array([]), np.array([])

    return bl_s, bl_val, fl_s, fl_val


def _interpolate_to_surface(
    s_data: NDArray[np.float64],
    val_data: NDArray[np.float64],
    surface_x: NDArray[np.float64],
    surface_y: NDArray[np.float64],
    panel_indices: List[int],
) -> NDArray[np.float64]:
    """Interpolate arc-length-based data back to surface panel coordinates.

    Returns an (M,) array where M = len(surface_x), with NaN for panels
    not covered by the BL data.
    """
    M = len(surface_x)
    full = np.full(M, np.nan)

    if len(s_data) == 0 or len(val_data) == 0:
        return full

    # Compute arc-length along the path
    path_x = surface_x[panel_indices]
    path_y = surface_y[panel_indices]
    path_s = np.zeros(len(panel_indices))
    ds = np.sqrt(np.diff(path_x) ** 2 + np.diff(path_y) ** 2)
    path_s[1:] = np.cumsum(ds)

    # Map s_data onto path_s coordinate system
    s_offset = s_data[0] - path_s[0] if len(s_data) > 0 else 0.0
    interp_s = path_s + s_offset

    # Interpolate
    valid = np.isfinite(val_data)
    if np.sum(valid) < 2:
        return full

    interp_val = np.interp(
        interp_s,
        s_data[valid],
        val_data[valid],
        left=np.nan,
        right=np.nan,
    )

    for j, idx in enumerate(panel_indices):
        full[idx] = interp_val[j]

    return full


def plot_wall_quantity_envelope_side_by_side(
    bl_result,
    comparison_result,
    quantity: str = "Ue",
    profile_name: Optional[str] = None,
    scale: float = 0.15,
    cmap: str = "viridis",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, NDArray]:
    """Side-by-side wall quantity envelope (BL Solver vs Fluent).

    Plots the specified wall quantity (Ue, Cf, delta, or Cp) as an envelope
    wrapped around the body geometry, with BL solver result on the left
    and Fluent CFD result on the right.

    Args:
        bl_result: BoundaryLayerCaseResult from the BL solver.
        comparison_result: BLComparisonResult from the Fluent comparison pipeline.
        quantity: Wall quantity to plot — one of "Ue", "Cf", "delta", "Cp".
        profile_name: BL profile name (uses first available if None).
        scale: Envelope displacement scale factor.
        cmap: Colormap name.
        title: Optional suptitle.
        output_path: If provided, save figure to this path.

    Returns:
        (fig, axes) where axes is a (2,) ndarray [ax_bl, ax_fluent].
    """
    fig, (ax_bl, ax_fl) = plt.subplots(1, 2, figsize=(14, 6))

    meta = _WALL_QUANTITY_META.get(quantity, {"label": quantity, "unit": "", "invert": False})
    qty_label = f"{meta['label']} [{meta['unit']}]" if meta["unit"] else meta["label"]
    invert = meta["invert"]

    if not comparison_result.has_fluent_data:
        ax_fl.text(0.5, 0.5, "Fluent data not available", ha="center", va="center", fontsize=11, color="0.4")
        ax_fl.axis("off")
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, np.array([ax_bl, ax_fl])

    surface_x = bl_result.surface_x
    surface_y = bl_result.surface_y
    normals = compute_outward_normals(surface_x, surface_y, closed=True)

    # Resolve profile name
    if profile_name is None and bl_result.profile_names:
        profile_name = bl_result.profile_names[0]

    # Collect data for both sides
    bl_full = np.full(len(surface_x), np.nan)
    fl_full = np.full(len(surface_x), np.nan)

    fluent_result = comparison_result.fluent_result
    for side in ["upper", "lower"]:
        bl_path = bl_result.sides[side]
        fl_path = fluent_result.sides[side]
        bl_s, bl_val, fl_s, fl_val = _get_wall_quantity(bl_path, fl_path, quantity, profile_name)

        bl_interp = _interpolate_to_surface(bl_s, bl_val, surface_x, surface_y, bl_path.panel_indices)
        fl_interp = _interpolate_to_surface(fl_s, fl_val, surface_x, surface_y, bl_path.panel_indices)

        valid_bl = np.isfinite(bl_interp)
        valid_fl = np.isfinite(fl_interp)
        bl_full[valid_bl] = bl_interp[valid_bl]
        fl_full[valid_fl] = fl_interp[valid_fl]

    # Replace NaN with 0 for envelope display
    bl_plot = np.where(np.isnan(bl_full), 0.0, bl_full)
    fl_plot = np.where(np.isnan(fl_full), 0.0, fl_full)

    # Shared colormap normalization
    all_vals = np.concatenate([bl_plot, fl_plot])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    if invert:
        # For Cp, swap so negative (suction) becomes positive displacement
        bl_plot = -bl_plot
        fl_plot = -fl_plot
        vmin, vmax = -vmax, -vmin

    for ax, vals, name in [(ax_bl, bl_plot, "BL Solver"), (ax_fl, fl_plot, "Fluent CFD")]:
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
            invert_values=False,  # already handled above
        )

    # Shared colorbar
    cmap_obj = plt.cm.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar_label = f"-{qty_label}" if invert else qty_label
    fig.colorbar(sm, ax=[ax_bl, ax_fl], label=cbar_label, shrink=0.8, pad=0.02)

    if title:
        fig.suptitle(title, fontsize=12)
    else:
        fig.suptitle(f"{meta['label']} envelope — BL Solver vs Fluent", fontsize=12)

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, np.array([ax_bl, ax_fl])


def plot_wall_quantity_envelope_overlay(
    bl_result,
    comparison_result,
    quantity: str = "Ue",
    profile_name: Optional[str] = None,
    scale: float = 0.15,
    color_bl: str = "#1f77b4",
    color_fl: str = "#d62728",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    show_difference: bool = False,
) -> Tuple[Figure, Axes]:
    """Overlay wall quantity envelopes (BL Solver and Fluent on same body).

    Both results are plotted on the same body geometry for direct visual
    comparison.

    Args:
        bl_result: BoundaryLayerCaseResult from the BL solver.
        comparison_result: BLComparisonResult from the Fluent comparison pipeline.
        quantity: Wall quantity to plot — one of "Ue", "Cf", "delta", "Cp".
        profile_name: BL profile name (uses first available if None).
        scale: Envelope displacement scale factor.
        color_bl: Color for BL solver envelope.
        color_fl: Color for Fluent envelope.
        title: Optional title.
        output_path: If provided, save figure to this path.
        show_difference: If True, highlight regions with large differences.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    meta = _WALL_QUANTITY_META.get(quantity, {"label": quantity, "unit": "", "invert": False})
    qty_label = f"{meta['label']} [{meta['unit']}]" if meta["unit"] else meta["label"]
    invert = meta["invert"]

    surface_x = bl_result.surface_x
    surface_y = bl_result.surface_y

    if not comparison_result.has_fluent_data:
        ax.text(0.5, 0.5, "Fluent data not available", ha="center", va="center", fontsize=11, color="0.4", transform=ax.transAxes)
        ax.axis("off")
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig, ax

    # Resolve profile name
    if profile_name is None and bl_result.profile_names:
        profile_name = bl_result.profile_names[0]

    # Collect data for both sides
    bl_full = np.full(len(surface_x), np.nan)
    fl_full = np.full(len(surface_x), np.nan)

    fluent_result = comparison_result.fluent_result
    for side in ["upper", "lower"]:
        bl_path = bl_result.sides[side]
        fl_path = fluent_result.sides[side]
        bl_s, bl_val, fl_s, fl_val = _get_wall_quantity(bl_path, fl_path, quantity, profile_name)

        bl_interp = _interpolate_to_surface(bl_s, bl_val, surface_x, surface_y, bl_path.panel_indices)
        fl_interp = _interpolate_to_surface(fl_s, fl_val, surface_x, surface_y, bl_path.panel_indices)

        valid_bl = np.isfinite(bl_interp)
        valid_fl = np.isfinite(fl_interp)
        bl_full[valid_bl] = bl_interp[valid_bl]
        fl_full[valid_fl] = fl_interp[valid_fl]

    # Replace NaN with 0 for envelope display
    bl_plot = np.where(np.isnan(bl_full), 0.0, bl_full)
    fl_plot = np.where(np.isnan(fl_full), 0.0, fl_full)

    plot_dual_surface_envelope(
        surface_x,
        surface_y,
        bl_plot,
        fl_plot,
        label1="BL Solver",
        label2="Fluent CFD",
        ax=ax,
        scale=scale,
        quantity_name=qty_label,
        color1=color_bl,
        color2=color_fl,
        show_difference=show_difference,
        title=title or f"{meta['label']} envelope — BL Solver vs Fluent",
        invert_values=invert,
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_wall_quantity_envelopes_grid(
    bl_result,
    comparison_result,
    quantities: Optional[List[str]] = None,
    profile_name: Optional[str] = None,
    scale: float = 0.12,
    mode: str = "overlay",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, NDArray]:
    """Grid of wall quantity envelope plots (2x2 for Ue, Cf, delta, Cp).

    Args:
        bl_result: BoundaryLayerCaseResult from the BL solver.
        comparison_result: BLComparisonResult from the Fluent comparison pipeline.
        quantities: List of quantities to plot (default: ["Ue", "Cf", "delta", "Cp"]).
        profile_name: BL profile name (uses first available if None).
        scale: Envelope displacement scale factor.
        mode: "overlay" for both on same body, "side_by_side" for separate panels.
        title: Optional suptitle.
        output_path: If provided, save figure to this path.

    Returns:
        (fig, axes) where axes is a 2D ndarray of Axes.
    """
    if quantities is None:
        quantities = ["Ue", "Cf", "delta", "Cp"]

    n_qty = len(quantities)
    if mode == "side_by_side":
        fig, axes = plt.subplots(n_qty, 2, figsize=(12, 4 * n_qty))
        if n_qty == 1:
            axes = axes.reshape(1, 2)
    else:
        ncols = 2
        nrows = (n_qty + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
        axes = np.atleast_2d(axes)

    surface_x = bl_result.surface_x
    surface_y = bl_result.surface_y
    normals = compute_outward_normals(surface_x, surface_y, closed=True)

    if profile_name is None and bl_result.profile_names:
        profile_name = bl_result.profile_names[0]

    fluent_result = comparison_result.fluent_result if comparison_result.has_fluent_data else None

    for idx, qty in enumerate(quantities):
        meta = _WALL_QUANTITY_META.get(qty, {"label": qty, "unit": "", "invert": False})
        qty_label = f"{meta['label']} [{meta['unit']}]" if meta["unit"] else meta["label"]
        invert = meta["invert"]

        # Collect data
        bl_full = np.full(len(surface_x), np.nan)
        fl_full = np.full(len(surface_x), np.nan)

        for side in ["upper", "lower"]:
            bl_path = bl_result.sides[side]
            if fluent_result is not None:
                fl_path = fluent_result.sides[side]
                bl_s, bl_val, fl_s, fl_val = _get_wall_quantity(bl_path, fl_path, qty, profile_name)
                fl_interp = _interpolate_to_surface(fl_s, fl_val, surface_x, surface_y, bl_path.panel_indices)
                fl_full[np.isfinite(fl_interp)] = fl_interp[np.isfinite(fl_interp)]
            else:
                bl_s, bl_val, _, _ = _get_wall_quantity(bl_path, bl_path, qty, profile_name)

            bl_interp = _interpolate_to_surface(bl_s, bl_val, surface_x, surface_y, bl_path.panel_indices)
            bl_full[np.isfinite(bl_interp)] = bl_interp[np.isfinite(bl_interp)]

        bl_plot = np.where(np.isnan(bl_full), 0.0, bl_full)
        fl_plot = np.where(np.isnan(fl_full), 0.0, fl_full)

        if mode == "side_by_side":
            ax_bl = axes[idx, 0]
            ax_fl = axes[idx, 1]

            all_vals = np.concatenate([bl_plot, fl_plot])
            vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
            if invert:
                bl_plot_vis = -bl_plot
                fl_plot_vis = -fl_plot
                vmin, vmax = -vmax, -vmin
            else:
                bl_plot_vis = bl_plot
                fl_plot_vis = fl_plot

            for ax, vals, name in [(ax_bl, bl_plot_vis, "BL Solver"), (ax_fl, fl_plot_vis, "Fluent")]:
                plot_surface_envelope(
                    surface_x,
                    surface_y,
                    vals,
                    normals=normals,
                    ax=ax,
                    scale=scale,
                    quantity_name=qty_label,
                    colormap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    show_colorbar=True,
                    title=f"{name} — {meta['label']}",
                    invert_values=False,
                )
        else:
            row, col = divmod(idx, 2)
            ax = axes[row, col]

            plot_dual_surface_envelope(
                surface_x,
                surface_y,
                bl_plot,
                fl_plot,
                label1="BL Solver",
                label2="Fluent",
                ax=ax,
                scale=scale,
                quantity_name=qty_label,
                color1="#1f77b4",
                color2="#d62728",
                show_difference=False,
                title=meta["label"],
                invert_values=invert,
            )

    # Hide unused axes
    if mode != "side_by_side":
        for idx in range(n_qty, axes.size):
            row, col = divmod(idx, 2)
            axes[row, col].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, axes
