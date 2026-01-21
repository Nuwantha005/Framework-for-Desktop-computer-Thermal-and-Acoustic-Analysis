"""
Comparison visualization module.

Provides tools for comparing fields from different sources:
- Panel method vs OpenFOAM (potentialFoam, simpleFoam, etc.)
- Different mesh resolutions
- Different geometries

Features:
- Side-by-side contour plots with unified colorbar
- Difference plots
- Line plot overlays
- Error metrics (L2, Linf, RMS)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Any, Literal
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy import interpolate

from core.geometry import Mesh


@dataclass
class FieldSeries:
    """
    A named field for comparison.
    
    Wraps scalar or vector field data with metadata for plotting.
    """
    name: str
    data: NDArray  # 2D scalar (ny, nx) or could be component of vector
    XX: NDArray
    YY: NDArray
    units: str = ""
    source: str = ""  # e.g., "Panel Method", "potentialFoam", "simpleFoam"
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape
    
    @property
    def x_range(self) -> Tuple[float, float]:
        return (float(self.XX.min()), float(self.XX.max()))
    
    @property
    def y_range(self) -> Tuple[float, float]:
        return (float(self.YY.min()), float(self.YY.max()))
    
    @classmethod
    def from_field_data(cls, fields: 'FieldData', field_name: str, 
                        source: str = "", component: str = "magnitude") -> 'FieldSeries':
        """Create from FieldData container."""
        from postprocessing.fields import VectorField
        
        fld = fields[field_name]
        
        if isinstance(fld, VectorField):
            if component == "magnitude":
                data = fld.magnitude
            elif component in ("x", "u"):
                data = fld.u
            elif component in ("y", "v"):
                data = fld.v
            else:
                raise ValueError(f"Unknown component: {component}")
            name = f"{field_name}_{component}"
            units = fld.units
        else:
            data = fld.data
            name = fld.name
            units = fld.units
        
        return cls(
            name=name,
            data=data,
            XX=fields.XX,
            YY=fields.YY,
            units=units,
            source=source
        )


@dataclass
class LineSeries:
    """
    A named line series for comparison (1D data).
    
    Used for Cp distributions, profiles along lines, etc.
    """
    name: str
    x: NDArray  # Independent variable
    y: NDArray  # Dependent variable
    source: str = ""
    x_label: str = ""
    y_label: str = ""
    style: str = "-"  # Line style
    marker: str = ""  # Marker style
    color: Optional[str] = None
    
    @property
    def length(self) -> int:
        return len(self.x)


@dataclass 
class ComparisonMetrics:
    """Error metrics between two fields."""
    l2_norm: float
    linf_norm: float  # Max absolute error
    rms: float
    mean_error: float
    max_error_location: Tuple[float, float]
    
    def summary(self) -> str:
        return (
            f"L2 norm: {self.l2_norm:.6g}\n"
            f"L∞ norm: {self.linf_norm:.6g}\n"
            f"RMS: {self.rms:.6g}\n"
            f"Mean error: {self.mean_error:.6g}\n"
            f"Max error at: ({self.max_error_location[0]:.3f}, {self.max_error_location[1]:.3f})"
        )


class ComparisonVisualizer:
    """
    Visualizer for comparing multiple fields.
    
    Supports:
    - Side-by-side contour plots with unified colorbar
    - Difference plots
    - Line plot overlays
    - Error metrics computation
    
    Usage:
        comp = ComparisonVisualizer()
        
        # Compare two scalar fields
        comp.compare_contours([field1, field2], mesh=mesh, title="Comparison")
        
        # Plot difference
        comp.plot_difference(field1, field2, mesh=mesh)
        
        # Compare line plots
        comp.compare_lines([line1, line2, line3], title="Cp Comparison")
        
        # Get error metrics
        metrics = comp.compute_metrics(field1, field2)
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Args:
            output_dir: Optional directory for saving figures
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.fig: Optional[Figure] = None
        self.axes: List[Axes] = []
    
    # =========================================================================
    # Contour Comparisons
    # =========================================================================
    
    def compare_contours(self,
                        fields: List[FieldSeries],
                        mesh: Optional[Mesh] = None,
                        levels: int = 20,
                        cmap: str = 'jet',
                        unified_colorbar: bool = True,
                        show_body: bool = True,
                        title: str = "Field Comparison",
                        figsize: Optional[Tuple[float, float]] = None) -> Figure:
        """
        Plot multiple fields side-by-side with optional unified colorbar.
        
        Args:
            fields: List of FieldSeries to compare
            mesh: Body mesh for outline (optional)
            levels: Number of contour levels
            cmap: Colormap
            unified_colorbar: Use same color scale for all plots
            show_body: Draw body outline
            title: Overall figure title
            figsize: Figure size (auto-calculated if None)
            
        Returns:
            Figure object
        """
        n = len(fields)
        if n < 2:
            raise ValueError("Need at least 2 fields to compare")
        
        # Auto figure size
        if figsize is None:
            figsize = (5 * n + 1, 5)  # Extra space for colorbar
        
        # Create figure with GridSpec for proper colorbar placement
        fig = plt.figure(figsize=figsize)
        
        if unified_colorbar:
            # Reserve space for shared colorbar
            gs = GridSpec(1, n + 1, width_ratios=[1]*n + [0.05], figure=fig)
        else:
            gs = GridSpec(1, n, figure=fig)
        
        # Compute unified color limits
        if unified_colorbar:
            vmin = min(np.nanmin(f.data) for f in fields)
            vmax = max(np.nanmax(f.data) for f in fields)
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = None
        
        axes = []
        last_cf = None
        
        for i, field in enumerate(fields):
            ax = fig.add_subplot(gs[0, i])
            axes.append(ax)
            
            ax.set_aspect('equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Title with source
            if field.source:
                ax.set_title(f"{field.name}\n({field.source})")
            else:
                ax.set_title(field.name)
            
            # Plot contours
            if unified_colorbar:
                cf = ax.contourf(field.XX, field.YY, field.data, 
                                levels=levels, cmap=cmap, norm=norm)
            else:
                cf = ax.contourf(field.XX, field.YY, field.data,
                                levels=levels, cmap=cmap)
                plt.colorbar(cf, ax=ax, label=field.units)
            
            last_cf = cf
            
            # Body outline
            if show_body and mesh is not None:
                self._draw_body_outline(ax, mesh)
            
            ax.set_xlim(field.x_range)
            ax.set_ylim(field.y_range)
            ax.grid(True, alpha=0.3)
        
        # Unified colorbar
        if unified_colorbar and last_cf is not None:
            cax = fig.add_subplot(gs[0, -1])
            cbar = fig.colorbar(last_cf, cax=cax)
            cbar.set_label(fields[0].units)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        self.fig = fig
        self.axes = axes
        
        return fig
    
    def plot_difference(self,
                       field1: FieldSeries,
                       field2: FieldSeries,
                       mesh: Optional[Mesh] = None,
                       levels: int = 20,
                       cmap: str = 'RdBu_r',
                       symmetric: bool = True,
                       show_body: bool = True,
                       show_originals: bool = True,
                       title: Optional[str] = None,
                       figsize: Optional[Tuple[float, float]] = None) -> Tuple[Figure, ComparisonMetrics]:
        """
        Plot difference between two fields (field1 - field2).
        
        Optionally shows both original fields for context.
        
        Args:
            field1: First field (reference)
            field2: Second field (to subtract)
            mesh: Body mesh for outline
            levels: Number of contour levels
            cmap: Colormap (diverging recommended)
            symmetric: Center colormap at zero
            show_body: Draw body outline
            show_originals: If True, show original fields alongside difference
            title: Figure title
            figsize: Figure size (auto-calculated if None)
            
        Returns:
            Tuple of (Figure, ComparisonMetrics)
        """
        # Interpolate field2 to field1's grid if different
        if field1.shape != field2.shape or not np.allclose(field1.XX, field2.XX):
            data2_interp = self._interpolate_to_grid(field2, field1.XX, field1.YY)
        else:
            data2_interp = field2.data
        
        # Compute difference
        diff = field1.data - data2_interp
        
        # Compute metrics
        metrics = self._compute_metrics_from_diff(diff, field1.XX, field1.YY)
        
        if show_originals:
            # Create figure: left column = 2 small plots stacked, right = 1 large difference
            if figsize is None:
                figsize = (14, 10)
            
            fig = plt.figure(figsize=figsize)
            # 2 rows, 3 cols: left col (width=1) for 2 plots, right 2 cols (width=2) for 1 large plot
            gs = GridSpec(2, 3, width_ratios=[1, 2, 0.08], figure=fig, 
                         hspace=0.25, wspace=0.35)
            
            # Common color scale for original fields
            vmin = min(np.nanmin(field1.data), np.nanmin(data2_interp))
            vmax = max(np.nanmax(field1.data), np.nanmax(data2_interp))
            norm = Normalize(vmin=vmin, vmax=vmax)
            
            # Top-left: Field 1
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_aspect('equal')
            cf1 = ax1.contourf(field1.XX, field1.YY, field1.data, levels=levels, 
                             cmap='viridis', norm=norm)
            ax1.set_title(f"{field1.name}\n({field1.source})" if field1.source else field1.name,
                         fontsize=10)
            if show_body and mesh:
                self._draw_body_outline(ax1, mesh)
            ax1.set_xlim(field1.x_range)
            ax1.set_ylim(field1.y_range)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            plt.colorbar(cf1, ax=ax1, label=field1.units, fraction=0.046, pad=0.04)
            
            # Bottom-left: Field 2
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.set_aspect('equal')
            cf2 = ax2.contourf(field1.XX, field1.YY, data2_interp, levels=levels,
                             cmap='viridis', norm=norm)
            ax2.set_title(f"{field2.name}\n({field2.source})" if field2.source else field2.name,
                         fontsize=10)
            if show_body and mesh:
                self._draw_body_outline(ax2, mesh)
            ax2.set_xlim(field1.x_range)
            ax2.set_ylim(field1.y_range)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            plt.colorbar(cf2, ax=ax2, label=field2.units, fraction=0.046, pad=0.04)
            
            # Right side: Difference plot (spans both rows)
            ax3 = fig.add_subplot(gs[:, 1])
            ax3.set_aspect('equal')
            
            if symmetric:
                diff_max = np.nanmax(np.abs(diff))
                if diff_max > 0:
                    diff_norm = TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max)
                else:
                    diff_norm = None
            else:
                diff_norm = None
            
            cf_diff = ax3.contourf(field1.XX, field1.YY, diff, levels=levels,
                                 cmap=cmap, norm=diff_norm)
            
            diff_title = f"Difference ({field1.source} − {field2.source})"
            if field1.source and field2.source:
                diff_title += f"\n{field1.name}"
            diff_title += f"\nL∞={metrics.linf_norm:.2e}, RMS={metrics.rms:.2e}"
            
            ax3.set_title(diff_title, fontsize=11, fontweight='bold')
            if show_body and mesh:
                self._draw_body_outline(ax3, mesh)
            ax3.set_xlim(field1.x_range)
            ax3.set_ylim(field1.y_range)
            ax3.grid(True, alpha=0.3)
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            
            # Colorbar for difference (in the rightmost column)
            cax_diff = fig.add_subplot(gs[:, 2])
            fig.colorbar(cf_diff, cax=cax_diff, label=f"Δ{field1.units}")
            
            axes = [ax1, ax2, ax3]
            
        else:
            # Only show difference plot
            if figsize is None:
                figsize = (8, 6)
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect('equal')
            
            if symmetric:
                diff_max = np.nanmax(np.abs(diff))
                if diff_max > 0:
                    diff_norm = TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max)
                else:
                    diff_norm = None
            else:
                diff_norm = None
            
            cf_diff = ax.contourf(field1.XX, field1.YY, diff, levels=levels,
                                 cmap=cmap, norm=diff_norm)
            ax.set_title(f"Difference: {field1.source} − {field2.source}\n"
                        f"L∞={metrics.linf_norm:.2e}, RMS={metrics.rms:.2e}")
            if show_body and mesh:
                self._draw_body_outline(ax, mesh)
            ax.set_xlim(field1.x_range)
            ax.set_ylim(field1.y_range)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            fig.colorbar(cf_diff, ax=ax, label=f"Δ{field1.units}")
            
            axes = [ax]
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        fig.tight_layout()
        
        self.fig = fig
        self.axes = axes
        
        return fig, metrics
    
    # =========================================================================
    # Line Plot Comparisons
    # =========================================================================
    
    def compare_lines(self,
                     series: List[LineSeries],
                     ax: Optional[Axes] = None,
                     title: str = "Comparison",
                     xlabel: Optional[str] = None,
                     ylabel: Optional[str] = None,
                     legend_loc: str = 'best',
                     grid: bool = True,
                     figsize: Tuple[float, float] = (10, 6)) -> Axes:
        """
        Overlay multiple line series on the same axes.
        
        Args:
            series: List of LineSeries to plot
            ax: Existing axes (creates new figure if None)
            title: Plot title
            xlabel: X-axis label (uses first series if None)
            ylabel: Y-axis label (uses first series if None)
            legend_loc: Legend location
            grid: Show grid
            figsize: Figure size (if creating new)
            
        Returns:
            Axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            self.fig = fig
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(series)))
        
        for i, s in enumerate(series):
            style = s.style + s.marker
            color = s.color if s.color else colors[i]
            
            label = f"{s.name}" if not s.source else f"{s.name} ({s.source})"
            ax.plot(s.x, s.y, style, color=color, label=label, linewidth=1.5)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel or series[0].x_label or 'X')
        ax.set_ylabel(ylabel or series[0].y_label or 'Y')
        ax.legend(loc=legend_loc)
        
        if grid:
            ax.grid(True, alpha=0.3)
        
        self.axes = [ax]
        return ax
    
    def compare_cp_distributions(self,
                                 cp_series: List[Tuple[NDArray, NDArray, str]],
                                 title: str = "Cp Distribution Comparison",
                                 figsize: Tuple[float, float] = (10, 6)) -> Axes:
        """
        Compare pressure coefficient distributions.
        
        Convenience method with Cp-specific defaults (inverted y-axis).
        
        Args:
            cp_series: List of (x, Cp, label) tuples
            title: Plot title
            figsize: Figure size
            
        Returns:
            Axes object
        """
        fig, ax = plt.subplots(figsize=figsize)
        self.fig = fig
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(cp_series)))
        
        for i, (x, cp, label) in enumerate(cp_series):
            ax.plot(x, cp, '-o', color=colors[i], label=label, 
                   markersize=4, linewidth=1.5)
        
        ax.set_title(title)
        ax.set_xlabel('x/c or Panel Index')
        ax.set_ylabel('Cp')
        ax.invert_yaxis()  # Cp convention
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        self.axes = [ax]
        return ax
    
    # =========================================================================
    # Mesh Convergence
    # =========================================================================
    
    def plot_convergence(self,
                        mesh_sizes: List[int],
                        values: List[float],
                        reference: Optional[float] = None,
                        xlabel: str = "Number of Panels",
                        ylabel: str = "Value",
                        title: str = "Mesh Convergence",
                        log_x: bool = True,
                        figsize: Tuple[float, float] = (8, 6)) -> Axes:
        """
        Plot mesh convergence study results.
        
        Args:
            mesh_sizes: List of mesh sizes (e.g., number of panels)
            values: Corresponding computed values
            reference: Optional reference/analytical value
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            log_x: Use log scale for x-axis
            figsize: Figure size
            
        Returns:
            Axes object
        """
        fig, ax = plt.subplots(figsize=figsize)
        self.fig = fig
        
        ax.plot(mesh_sizes, values, 'b-o', label='Computed', linewidth=2, markersize=8)
        
        if reference is not None:
            ax.axhline(y=reference, color='r', linestyle='--', 
                      label=f'Reference ({reference:.4f})', linewidth=1.5)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        if log_x:
            ax.set_xscale('log')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.axes = [ax]
        return ax
    
    # =========================================================================
    # Error Metrics
    # =========================================================================
    
    def compute_metrics(self,
                       field1: FieldSeries,
                       field2: FieldSeries) -> ComparisonMetrics:
        """
        Compute error metrics between two fields.
        
        Args:
            field1: Reference field
            field2: Field to compare
            
        Returns:
            ComparisonMetrics object
        """
        # Interpolate if needed
        if field1.shape != field2.shape or not np.allclose(field1.XX, field2.XX):
            data2_interp = self._interpolate_to_grid(field2, field1.XX, field1.YY)
        else:
            data2_interp = field2.data
        
        diff = field1.data - data2_interp
        
        return self._compute_metrics_from_diff(diff, field1.XX, field1.YY)
    
    def _compute_metrics_from_diff(self, diff: NDArray, XX: NDArray, YY: NDArray) -> ComparisonMetrics:
        """Compute metrics from difference array."""
        # Mask NaN values
        valid = ~np.isnan(diff)
        diff_valid = diff[valid]
        
        if len(diff_valid) == 0:
            return ComparisonMetrics(
                l2_norm=np.nan, linf_norm=np.nan, rms=np.nan,
                mean_error=np.nan, max_error_location=(np.nan, np.nan)
            )
        
        l2 = np.sqrt(np.sum(diff_valid**2))
        linf = np.max(np.abs(diff_valid))
        rms = np.sqrt(np.mean(diff_valid**2))
        mean_err = np.mean(diff_valid)
        
        # Find location of max error
        max_idx = np.nanargmax(np.abs(diff))
        max_idx_2d = np.unravel_index(max_idx, diff.shape)
        max_loc = (float(XX[max_idx_2d]), float(YY[max_idx_2d]))
        
        return ComparisonMetrics(
            l2_norm=float(l2),
            linf_norm=float(linf),
            rms=float(rms),
            mean_error=float(mean_err),
            max_error_location=max_loc
        )
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _interpolate_to_grid(self, field: FieldSeries, 
                            target_XX: NDArray, target_YY: NDArray) -> NDArray:
        """Interpolate field data to a different grid."""
        # Flatten source grid
        points = np.column_stack([field.XX.ravel(), field.YY.ravel()])
        values = field.data.ravel()
        
        # Remove NaN points
        valid = ~np.isnan(values)
        points_valid = points[valid]
        values_valid = values[valid]
        
        # Interpolate
        target_points = np.column_stack([target_XX.ravel(), target_YY.ravel()])
        interpolated = interpolate.griddata(
            points_valid, values_valid, target_points, 
            method='linear', fill_value=np.nan
        )
        
        return interpolated.reshape(target_XX.shape)
    
    def _draw_body_outline(self, ax: Axes, mesh: Mesh, **kwargs):
        """Draw body outline from mesh."""
        # Group panels by component
        component_ids = np.unique(mesh.component_ids)
        
        for comp_id in component_ids:
            comp_mask = mesh.component_ids == comp_id
            comp_panel_indices = np.where(comp_mask)[0]
            
            if len(comp_panel_indices) == 0:
                continue
            
            # Collect ordered nodes
            x_coords = []
            y_coords = []
            
            for panel_idx in comp_panel_indices:
                n1_idx = mesh.panels[panel_idx, 0]
                x_coords.append(mesh.nodes[n1_idx, 0])
                y_coords.append(mesh.nodes[n1_idx, 1])
            
            # Close the loop
            last_panel = comp_panel_indices[-1]
            n2_idx = mesh.panels[last_panel, 1]
            x_coords.append(mesh.nodes[n2_idx, 0])
            y_coords.append(mesh.nodes[n2_idx, 1])
            
            ax.fill(x_coords, y_coords, color='white', edgecolor='black', 
                   linewidth=1.5, zorder=10)
    
    # =========================================================================
    # Save/Show
    # =========================================================================
    
    def save(self, filename: str, dpi: int = 150):
        """Save current figure."""
        if self.fig is None:
            raise RuntimeError("No figure to save")
        
        if self.output_dir:
            filepath = self.output_dir / filename
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            filepath = Path(filename)
        
        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    def show(self):
        """Display current figure."""
        if self.fig is None:
            raise RuntimeError("No figure to show")
        plt.show()
    
    def finalize(self, save: Optional[str] = None, show: bool = True, dpi: int = 150):
        """Save and/or show figure."""
        if save:
            self.save(save, dpi=dpi)
        if show:
            self.show()
        else:
            plt.close(self.fig)
