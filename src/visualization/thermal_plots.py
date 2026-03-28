"""
Thermal boundary layer visualization module.

Provides plotting functions for thermal BL results:
- Envelope plots: temperature or h(s) wrapped around body geometry
- Line plots: T_w, h, Nu vs arc-length
- Two-sided comparison plots: upper and lower surfaces

All functions accept ThermalResult or ThermalCaseResult objects and
follow the same patterns as bl_line_plots and bl_envelope_plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from visualization.surface_envelope import plot_surface_envelope


# ---------------------------------------------------------------------------
# Result container for multi-side thermal results
# ---------------------------------------------------------------------------

@dataclass
class ThermalCaseResult:
    """
    Container for thermal results from both surface sides.
    
    Mirrors BoundaryLayerCaseResult structure for consistency.
    
    Attributes:
        case_name: Case identifier
        surface_x: Full body x-coordinates [m], shape (M,)
        surface_y: Full body y-coordinates [m], shape (M,)
        upper: Thermal result for upper surface
        lower: Thermal result for lower surface
        solver_type: Name of the thermal solver used
    """
    case_name: str
    surface_x: NDArray[np.float64]
    surface_y: NDArray[np.float64]
    upper: "ThermalResult"  # Forward reference
    lower: "ThermalResult"
    solver_type: str = ""
    
    @property
    def sides(self) -> Dict[str, "ThermalResult"]:
        """Convenience mapping {"upper": ..., "lower": ...}."""
        return {"upper": self.upper, "lower": self.lower}


# ---------------------------------------------------------------------------
# Labels for thermal quantities
# ---------------------------------------------------------------------------

_THERMAL_LABELS = {
    "wall_temperature": r"$T_w$ [K]",
    "heat_transfer_coeff": r"$h$ [W/m²K]",
    "nusselt": r"$Nu$",
    "wall_heat_flux": r"$q_w$ [W/m²]",
    "thermal_bl_thickness": r"$\delta_T$ [m]",
}

_THERMAL_COLORS = {
    "upper": "#E63946",  # Red
    "lower": "#1D3557",  # Blue
}


def _get_side_color(side: str) -> str:
    """Get consistent color for a surface side."""
    return _THERMAL_COLORS.get(side.lower(), "#2A9D8F")


# ---------------------------------------------------------------------------
# Line plots
# ---------------------------------------------------------------------------

def plot_thermal_line(
    thermal_result,
    quantity: str = "wall_temperature",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot a thermal quantity vs arc length.
    
    Args:
        thermal_result: ThermalResult object
        quantity: One of "wall_temperature", "heat_transfer_coeff", "nusselt",
            "wall_heat_flux", "thermal_bl_thickness"
        ax: Existing axes (creates new if None)
        title: Plot title
        color: Line color (default: based on side)
        label: Legend label (default: solver_type)
        output_path: If provided, save figure to this path
    
    Returns:
        (Figure, Axes) tuple
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()
    
    s = np.asarray(thermal_result.arc_length)
    vals = np.asarray(getattr(thermal_result, quantity))
    
    sort_idx = np.argsort(s)
    s_plot = s[sort_idx]
    vals_plot = vals[sort_idx]
    
    if color is None:
        color = _get_side_color(thermal_result.side)
    if label is None:
        label = thermal_result.solver_type or thermal_result.side
    
    ax.plot(s_plot, vals_plot, color=color, linewidth=1.5, label=label)
    
    ax.set_xlabel("Arc length $s$ [m]")
    ax.set_ylabel(_THERMAL_LABELS.get(quantity, quantity))
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=11)
    
    fig.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig, ax


def plot_thermal_lines_multi(
    thermal_result,
    quantities: Optional[List[str]] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, List[Axes]]:
    """
    Multi-panel line plots of several thermal quantities.
    
    Args:
        thermal_result: ThermalResult object
        quantities: List of quantities to plot (default: T_w, h, Nu, q_w)
        title: Figure title
        output_path: If provided, save figure to this path
    
    Returns:
        (Figure, list of Axes) tuple
    """
    if quantities is None:
        quantities = ["wall_temperature", "heat_transfer_coeff", "nusselt", "wall_heat_flux"]
    
    n_quantities = len(quantities)
    fig, axes = plt.subplots(n_quantities, 1, figsize=(8, 3.2 * n_quantities), sharex=True)
    
    if n_quantities == 1:
        axes = [axes]
    
    color = _get_side_color(thermal_result.side)
    
    for ax, quantity in zip(axes, quantities):
        plot_thermal_line(thermal_result, quantity=quantity, ax=ax, color=color)
    
    axes[-1].set_xlabel("Arc length $s$ [m]")
    
    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    
    fig.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig, list(axes)


def plot_thermal_two_sides(
    upper_result,
    lower_result,
    quantities: Optional[List[str]] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, NDArray]:
    """
    Side-by-side line plots: upper (left) and lower (right).
    
    Args:
        upper_result: ThermalResult for upper surface
        lower_result: ThermalResult for lower surface
        quantities: List of quantities to plot
        title: Figure title
        output_path: If provided, save figure to this path
    
    Returns:
        (Figure, axes array) tuple
    """
    if quantities is None:
        quantities = ["wall_temperature", "heat_transfer_coeff", "nusselt", "wall_heat_flux"]
    
    n_quantities = len(quantities)
    fig, axes = plt.subplots(n_quantities, 2, figsize=(14, 3.2 * n_quantities), sharex="col")
    
    if n_quantities == 1:
        axes = axes.reshape(1, 2)
    
    for i, quantity in enumerate(quantities):
        plot_thermal_line(upper_result, quantity=quantity, ax=axes[i, 0],
                         color=_get_side_color("upper"))
        plot_thermal_line(lower_result, quantity=quantity, ax=axes[i, 1],
                         color=_get_side_color("lower"))
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


# ---------------------------------------------------------------------------
# Envelope plots
# ---------------------------------------------------------------------------

def plot_thermal_envelope(
    thermal_result,
    surface_x: NDArray[np.float64],
    surface_y: NDArray[np.float64],
    panel_indices: List[int],
    quantity: str = "wall_temperature",
    scale: float = 0.15,
    colormap: Optional[str] = "coolwarm",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """
    Envelope plot of thermal quantity on body geometry.
    
    The thermal result values are mapped back to the full body surface
    using the panel_indices from the BL path result.
    
    Args:
        thermal_result: ThermalResult object
        surface_x: Full body x-coordinates [m], shape (M,)
        surface_y: Full body y-coordinates [m], shape (M,)
        panel_indices: Indices mapping thermal result stations to body panels
        quantity: Thermal quantity to plot
        scale: Envelope displacement scale
        colormap: Matplotlib colormap name
        ax: Existing axes (creates new if None)
        title: Plot title
        output_path: If provided, save figure to this path
    
    Returns:
        (Figure, Axes) tuple
    """
    # Map thermal values to full body array
    M = len(surface_x)
    vals_full = np.full(M, np.nan)
    
    thermal_vals = getattr(thermal_result, quantity)
    n_thermal = len(thermal_vals)
    
    for i, panel_idx in enumerate(panel_indices):
        if i < n_thermal and panel_idx < M:
            vals_full[panel_idx] = thermal_vals[i]
    
    # Use masked array instead of replacing with 0 to prevent 
    # plotting the envelope in regions without valid data (like the wake)
    vals_plot = np.ma.masked_invalid(vals_full)
    
    # Get units for colorbar label
    label = _THERMAL_LABELS.get(quantity, quantity)
    
    fig, ax = plot_surface_envelope(
        surface_x,
        surface_y,
        vals_plot,
        scale=scale,
        quantity_name=label,
        colormap=colormap,
        ax=ax,
        title=title or f"{quantity} envelope ({thermal_result.side})",
    )
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig, ax


def plot_thermal_envelope_two_sides(
    upper_result,
    lower_result,
    surface_x: NDArray[np.float64],
    surface_y: NDArray[np.float64],
    upper_indices: List[int],
    lower_indices: List[int],
    quantity: str = "wall_temperature",
    scale: float = 0.15,
    colormap: Optional[str] = "coolwarm",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """
    Combined envelope plot with both upper and lower thermal results.
    
    Both sides' thermal values are mapped to the full body and combined
    into a single envelope visualization.
    
    Args:
        upper_result: ThermalResult for upper surface
        lower_result: ThermalResult for lower surface
        surface_x: Full body x-coordinates [m], shape (M,)
        surface_y: Full body y-coordinates [m], shape (M,)
        upper_indices: Panel indices for upper surface
        lower_indices: Panel indices for lower surface
        quantity: Thermal quantity to plot
        scale: Envelope displacement scale
        colormap: Matplotlib colormap name
        ax: Existing axes (creates new if None)
        title: Plot title
        output_path: If provided, save figure to this path
    
    Returns:
        (Figure, Axes) tuple
    """
    M = len(surface_x)
    vals_full = np.full(M, np.nan)
    
    # Map upper values
    upper_vals = getattr(upper_result, quantity)
    for i, panel_idx in enumerate(upper_indices):
        if i < len(upper_vals) and panel_idx < M:
            vals_full[panel_idx] = upper_vals[i]
    
    # Map lower values
    lower_vals = getattr(lower_result, quantity)
    for i, panel_idx in enumerate(lower_indices):
        if i < len(lower_vals) and panel_idx < M:
            vals_full[panel_idx] = lower_vals[i]
    
    # Use masked array instead of replacing with 0 to prevent 
    # plotting the envelope in regions without valid data (like the wake)
    vals_plot = np.ma.masked_invalid(vals_full)
    label = _THERMAL_LABELS.get(quantity, quantity)
    
    fig, ax = plot_surface_envelope(
        surface_x,
        surface_y,
        vals_plot,
        scale=scale,
        quantity_name=label,
        colormap=colormap,
        ax=ax,
        title=title or f"{quantity} envelope",
    )
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig, ax


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Domain temperature contour plots (BDIM field visualization)
# ---------------------------------------------------------------------------

def plot_thermal_field_contour(
    thermal_result,
    quantity: str = "T",
    ax: Optional[Axes] = None,
    colormap: str = "coolwarm",
    levels: int = 50,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """
    Contour plot of temperature field in boundary layer (s, y) coordinates.
    
    Requires ThermalResult.field to be populated (from BDIM solver).
    Uses arc-length s and wall-normal distance y (like viscous BL plots).
    
    Args:
        thermal_result: ThermalResult with field data
        quantity: "T" for temperature, "T_norm" for normalized temperature
        ax: Existing axes (creates new if None)
        colormap: Matplotlib colormap name
        levels: Number of contour levels
        title: Plot title
        output_path: If provided, save figure to this path
    
    Returns:
        (Figure, Axes) tuple
    
    Raises:
        ValueError: If field data not available
    """
    if not thermal_result.has_field:
        raise ValueError(
            "Thermal field data not available. Use BDIM solver with "
            "field reconstruction to generate domain temperature."
        )
    
    field = thermal_result.field
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()
    
    # Get data to plot
    if quantity == "T":
        Z = field.T
        cbar_label = r"$T$ [K]"
    elif quantity == "T_norm":
        Z = field.T_normalized
        cbar_label = r"$(T - T_\infty) / (T_w - T_\infty)$"
    else:
        raise ValueError(f"Unknown quantity '{quantity}'. Use 'T' or 'T_norm'.")
    
    # Build meshgrid for s-y coordinates
    # field.s is shape (M,), field.y_normal is shape (M, Ny)
    M, Ny = field.y_normal.shape
    S = np.broadcast_to(field.s[:, None], (M, Ny))
    Y = field.y_normal * 1000  # Convert to mm for readability
    
    # Create filled contour plot in (s, y) coordinates
    contour = ax.contourf(
        S, Y, Z,
        levels=levels,
        cmap=colormap,
        extend="both",
    )
    
    # Add contour lines for clarity
    ax.contour(
        S, Y, Z,
        levels=10,
        colors="k",
        linewidths=0.3,
        alpha=0.5,
    )
    
    # Plot wall (y=0 line)
    ax.axhline(y=0, color="k", linewidth=1.5, label="Wall")
    
    # Colorbar
    cbar = fig.colorbar(contour, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    
    ax.set_xlabel(r"Arc length $s$ [m]")
    ax.set_ylabel(r"Wall-normal $y$ [mm]")
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=11)
    
    fig.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig, ax


def plot_thermal_field_profiles(
    thermal_result,
    stations: Optional[List[int]] = None,
    num_stations: int = 5,
    ax: Optional[Axes] = None,
    colormap: str = "viridis",
    normalized: bool = True,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot temperature profiles at selected arc-length stations.
    
    Shows T vs y_normal at multiple streamwise locations.
    
    Args:
        thermal_result: ThermalResult with field data
        stations: Specific station indices to plot (0-based)
        num_stations: Number of evenly-spaced stations if indices not provided
        ax: Existing axes (creates new if None)
        colormap: Matplotlib colormap for station colors
        normalized: If True, plot (T - T_inf)/(T_w - T_inf); else T [K]
        title: Plot title
        output_path: If provided, save figure to this path
    
    Returns:
        (Figure, Axes) tuple
    """
    if not thermal_result.has_field:
        raise ValueError("Thermal field data not available.")
    
    field = thermal_result.field
    M = field.num_stations
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()
    
    # Select stations
    if stations is None:
        stations = np.linspace(0, M - 1, num_stations, dtype=int)
    
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i / (len(stations) - 1)) for i in range(len(stations))]
    
    for idx, (station_idx, color) in enumerate(zip(stations, colors)):
        y_prof = field.y_normal[station_idx, :]
        T_prof = field.T[station_idx, :]
        
        if normalized:
            T_wall = T_prof[0]
            T_inf = field.T_inf
            if abs(T_wall - T_inf) > 1e-10:
                T_plot = (T_prof - T_inf) / (T_wall - T_inf)
            else:
                T_plot = np.zeros_like(T_prof)
            x_label = r"$(T - T_\infty) / (T_w - T_\infty)$"
        else:
            T_plot = T_prof
            x_label = r"$T$ [K]"
        
        s_val = field.s[station_idx]
        ax.plot(T_plot, y_prof * 1000, color=color, linewidth=1.5, 
                label=f"$s = {s_val:.3f}$ m")
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$y$ [mm]")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=11)
    
    fig.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig, ax


def plot_thermal_field_two_sides(
    upper_result,
    lower_result,
    quantity: str = "T",
    colormap: str = "coolwarm",
    levels: int = 50,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Figure, NDArray]:
    """
    Side-by-side contour plots of thermal field for both surfaces.
    
    Args:
        upper_result: ThermalResult for upper surface (with field)
        lower_result: ThermalResult for lower surface (with field)
        quantity: "T" or "T_norm"
        colormap: Matplotlib colormap name
        levels: Number of contour levels
        title: Figure title
        output_path: If provided, save figure to this path
    
    Returns:
        (Figure, axes array) tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if upper_result.has_field:
        plot_thermal_field_contour(
            upper_result, quantity=quantity, ax=axes[0],
            colormap=colormap, levels=levels, title=f"Upper ({upper_result.side})"
        )
    else:
        axes[0].text(0.5, 0.5, "No field data", ha="center", va="center",
                     transform=axes[0].transAxes)
        axes[0].set_title("Upper (no field)")
    
    if lower_result.has_field:
        plot_thermal_field_contour(
            lower_result, quantity=quantity, ax=axes[1],
            colormap=colormap, levels=levels, title=f"Lower ({lower_result.side})"
        )
    else:
        axes[1].text(0.5, 0.5, "No field data", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].set_title("Lower (no field)")
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    fig.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig, axes


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def plot_thermal_summary(
    upper_result,
    lower_result,
    surface_x: NDArray[np.float64],
    surface_y: NDArray[np.float64],
    upper_indices: List[int],
    lower_indices: List[int],
    envelope_quantity: str = "wall_temperature",
    envelope_scale: float = 0.15,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Figure:
    """
    Full thermal summary figure with line plots and envelope.
    
    Layout:
    - Row 1-4: Line plots for T_w, h, Nu, q_w (upper left, lower right)
    - Right column: Combined envelope plot
    
    Args:
        upper_result: ThermalResult for upper surface
        lower_result: ThermalResult for lower surface
        surface_x: Full body x-coordinates
        surface_y: Full body y-coordinates
        upper_indices: Panel indices for upper surface
        lower_indices: Panel indices for lower surface
        envelope_quantity: Quantity for envelope plot
        envelope_scale: Envelope displacement scale
        title: Figure title
        output_path: If provided, save figure to this path
    
    Returns:
        Figure object
    """
    quantities = ["wall_temperature", "heat_transfer_coeff", "nusselt", "wall_heat_flux"]
    n_rows = len(quantities)
    
    fig = plt.figure(figsize=(16, 3.5 * n_rows))
    gs = fig.add_gridspec(n_rows, 3, width_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.30)
    
    # Line plots
    for i, quantity in enumerate(quantities):
        # Upper side
        ax_upper = fig.add_subplot(gs[i, 0])
        plot_thermal_line(upper_result, quantity=quantity, ax=ax_upper,
                         color=_get_side_color("upper"))
        if i == 0:
            ax_upper.set_title("Upper side", fontsize=11)
        if i < n_rows - 1:
            ax_upper.set_xlabel("")
        
        # Lower side
        ax_lower = fig.add_subplot(gs[i, 1])
        plot_thermal_line(lower_result, quantity=quantity, ax=ax_lower,
                         color=_get_side_color("lower"))
        ax_lower.set_ylabel("")
        if i == 0:
            ax_lower.set_title("Lower side", fontsize=11)
        if i < n_rows - 1:
            ax_lower.set_xlabel("")
    
    # Envelope plot (spans all rows in right column)
    ax_env = fig.add_subplot(gs[:, 2])
    plot_thermal_envelope_two_sides(
        upper_result, lower_result,
        surface_x, surface_y,
        upper_indices, lower_indices,
        quantity=envelope_quantity,
        scale=envelope_scale,
        ax=ax_env,
        title=f"{envelope_quantity} envelope",
    )
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig
