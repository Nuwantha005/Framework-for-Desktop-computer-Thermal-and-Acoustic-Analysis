"""
Unified visualization facade for panel method solver.

Provides a single entry point for all visualization tasks:
- Pre-solve: Mesh plots (geometry verification)
- Post-solve: Streamlines, contours, Cp distributions

Handles common concerns:
- Figure creation and sizing
- Save vs display logic
- Subplot composition
- Output path management (with datetime override protection)
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Union, Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import NDArray

from core.geometry import Mesh, Scene


class OutputManager:
    """
    Manages output paths and save behavior.
    
    Features:
    - Auto-creates 'out' directory if missing
    - Optional datetime subfolder for overwrite protection
    - Consistent path resolution
    """
    
    def __init__(self, 
                 base_dir: Union[str, Path],
                 protect_overwrite: bool = False):
        """
        Args:
            base_dir: Base output directory (typically case_dir/out)
            protect_overwrite: If True, saves to timestamped subfolder
        """
        self.base_dir = Path(base_dir)
        self.protect_overwrite = protect_overwrite
        self._output_dir: Optional[Path] = None
    
    @property
    def output_dir(self) -> Path:
        """Get (and create) the output directory."""
        if self._output_dir is None:
            if self.protect_overwrite:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._output_dir = self.base_dir / timestamp
            else:
                self._output_dir = self.base_dir
            
            self._output_dir.mkdir(parents=True, exist_ok=True)
        
        return self._output_dir
    
    def get_path(self, filename: str) -> Path:
        """Get full path for a file in the output directory."""
        return self.output_dir / filename
    
    def reset(self):
        """Reset output directory (for new timestamp on next access)."""
        self._output_dir = None


class Visualizer:
    """
    Main visualization facade.
    
    Coordinates mesh plotting and post-processing visualizations with
    consistent interface for save/show behavior and subplot composition.
    
    Usage:
        # Single plot
        viz = Visualizer(output_dir='cases/my_case/out')
        viz.plot_mesh(mesh, save='mesh.png')
        
        # Combined subplot
        viz.create_figure(subplots=(1, 3), figsize=(18, 6))
        viz.plot_mesh(mesh, ax_index=0)
        viz.plot_contours(XX, YY, Vx, Vy, mesh, ax_index=1)
        viz.plot_streamlines(XX, YY, Vx, Vy, mesh, ax_index=2)
        viz.finalize(save='combined.png')
    """
    
    def __init__(self,
                 output_dir: Optional[Union[str, Path]] = None,
                 protect_overwrite: bool = False,
                 figsize: Tuple[float, float] = (10, 8)):
        """
        Args:
            output_dir: Base output directory for saves
            protect_overwrite: Save to timestamped subfolder
            figsize: Default figure size
        """
        self.default_figsize = figsize
        
        if output_dir is not None:
            self.output = OutputManager(output_dir, protect_overwrite)
        else:
            self.output = None
        
        # Current figure state
        self.fig: Optional[Figure] = None
        self.axes: Optional[Union[Axes, np.ndarray]] = None
        self._subplot_shape: Optional[Tuple[int, int]] = None
    
    # -------------------------------------------------------------------------
    # Figure Management
    # -------------------------------------------------------------------------
    
    def create_figure(self,
                     subplots: Tuple[int, int] = (1, 1),
                     figsize: Optional[Tuple[float, float]] = None,
                     title: Optional[str] = None) -> Tuple[Figure, Union[Axes, np.ndarray]]:
        """
        Create a new figure with optional subplots.
        
        Args:
            subplots: (rows, cols) subplot grid
            figsize: Figure size (uses default if None)
            title: Super title for figure
        
        Returns:
            (fig, axes) tuple
        """
        if figsize is None:
            # Scale figsize based on subplot count
            w, h = self.default_figsize
            figsize = (w * subplots[1], h * subplots[0])
        
        self.fig, self.axes = plt.subplots(subplots[0], subplots[1], figsize=figsize)
        self._subplot_shape = subplots
        
        if title:
            self.fig.suptitle(title, fontsize=14, fontweight='bold')
        
        return self.fig, self.axes
    
    def _get_ax(self, ax_index: Optional[int] = None) -> Axes:
        """Get axis for plotting."""
        if self.fig is None:
            # Auto-create single figure
            self.create_figure()
        
        if ax_index is None:
            if isinstance(self.axes, np.ndarray):
                return self.axes.flat[0]
            return self.axes
        
        if isinstance(self.axes, np.ndarray):
            return self.axes.flat[ax_index]
        
        if ax_index != 0:
            raise ValueError(f"ax_index={ax_index} invalid for single subplot")
        return self.axes
    
    def finalize(self,
                save: Optional[str] = None,
                show: bool = False,
                dpi: int = 150,
                tight_layout: bool = True):
        """
        Finalize figure: save and/or display.
        
        Args:
            save: Filename to save (in output_dir). None = don't save.
            show: Whether to display interactively
            dpi: Resolution for saved image
            tight_layout: Apply tight_layout before saving
        """
        if self.fig is None:
            raise ValueError("No figure to finalize. Call a plot method first.")
        
        if tight_layout:
            self.fig.tight_layout()
        
        if save is not None:
            if self.output is None:
                # Save to current directory
                save_path = Path(save)
            else:
                save_path = self.output.get_path(save)
            
            self.fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        
        if not show:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
    
    # -------------------------------------------------------------------------
    # Mesh Plotting (Pre-solve)
    # -------------------------------------------------------------------------
    
    def plot_mesh(self,
                 mesh: Mesh,
                 ax_index: Optional[int] = None,
                 show_nodes: bool = True,
                 show_panels: bool = True,
                 show_normals: bool = False,
                 show_centers: bool = False,
                 component_colors: bool = True,
                 title: str = "Mesh"):
        """
        Plot a 2D mesh.
        
        Args:
            mesh: Mesh to plot
            ax_index: Subplot index (None for single/first plot)
            show_nodes: Draw node markers
            show_panels: Draw panel edges
            show_normals: Draw normal vectors
            show_centers: Draw panel centers
            component_colors: Color by component ID
            title: Subplot title
        """
        if not mesh.is_2d:
            raise ValueError("plot_mesh only supports 2D meshes")
        
        ax = self._get_ax(ax_index)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        
        # Component colors
        component_ids = np.unique(mesh.component_ids)
        if component_colors and len(component_ids) > 1:
            colors = plt.cm.tab10(np.linspace(0, 1, len(component_ids)))
            color_map = {cid: colors[i] for i, cid in enumerate(component_ids)}
        else:
            color_map = {cid: 'black' for cid in component_ids}
        
        # Plot panels
        if show_panels:
            for i in range(mesh.num_panels):
                n1_idx, n2_idx = mesh.panels[i]
                p1 = mesh.nodes[n1_idx, :2]
                p2 = mesh.nodes[n2_idx, :2]
                color = color_map[mesh.component_ids[i]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2)
        
        # Plot nodes
        if show_nodes:
            ax.plot(mesh.nodes[:, 0], mesh.nodes[:, 1], 
                   'ko', markersize=4, zorder=10)
        
        # Plot centers
        if show_centers and mesh.centers is not None:
            ax.plot(mesh.centers[:, 0], mesh.centers[:, 1],
                   'rx', markersize=6, zorder=11)
        
        # Plot normals
        if show_normals and mesh.normals is not None and mesh.centers is not None:
            scale = self._compute_normal_scale(mesh)
            for i in range(mesh.num_panels):
                center = mesh.centers[i, :2]
                normal = mesh.normals[i, :2]
                ax.arrow(center[0], center[1],
                        scale * normal[0], scale * normal[1],
                        head_width=scale*0.3, head_length=scale*0.2,
                        fc='red', ec='red', alpha=0.7, zorder=12)
        
        self._auto_scale_axis(ax, mesh.nodes[:, :2])
        return ax
    
    def plot_scene(self,
                  scene: Scene,
                  ax_index: Optional[int] = None,
                  show_normals: bool = False,
                  show_freestream: bool = True,
                  title: Optional[str] = None):
        """
        Plot a scene (multi-component).
        
        Args:
            scene: Scene to plot
            ax_index: Subplot index
            show_normals: Show panel normals
            show_freestream: Show freestream arrow
            title: Subplot title
        """
        mesh = scene.assemble()
        
        if title is None:
            title = f"Scene: {scene.name}"
        
        self.plot_mesh(mesh, ax_index=ax_index, show_normals=show_normals, 
                      component_colors=True, title=title)
        
        ax = self._get_ax(ax_index)
        if show_freestream:
            self._draw_freestream_arrow(ax, scene.freestream)
        
        return ax
    
    # -------------------------------------------------------------------------
    # Post-Processing Plots (Post-solve)
    # -------------------------------------------------------------------------
    
    def plot_streamlines(self,
                        XX: NDArray, YY: NDArray, Vx: NDArray, Vy: NDArray,
                        mesh: Mesh,
                        ax_index: Optional[int] = None,
                        density: float = 1.0,
                        seed_style: Literal['left', 'uniform', 'auto'] = 'left',
                        show_body: bool = True,
                        title: str = "Streamlines",
                        cmap: str = 'viridis'):
        """
        Plot streamlines from precomputed velocity field.
        
        Args:
            XX, YY: Meshgrid coordinates
            Vx, Vy: Velocity components
            mesh: Body mesh (for outline)
            ax_index: Subplot index
            density: Streamline density
            seed_style: How to seed streamlines
            show_body: Draw body outline
            title: Subplot title
            cmap: Colormap for velocity magnitude
        """
        ax = self._get_ax(ax_index)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        
        x_range = (XX[0, 0], XX[0, -1])
        y_range = (YY[0, 0], YY[-1, 0])
        ny, nx = XX.shape
        
        # Seed points
        if seed_style == 'left':
            n_lines = int(ny * 0.3 * density)
            y_seeds = np.linspace(y_range[0], y_range[1], max(n_lines, 5))
            x_seeds = np.full_like(y_seeds, x_range[0])
            seed_points = np.column_stack([x_seeds, y_seeds])
        elif seed_style == 'uniform':
            n_lines = int(ny * 0.5 * density)
            y_seeds = np.linspace(y_range[0], y_range[1], max(n_lines, 5))
            x_seeds = np.full_like(y_seeds, x_range[0] * 0.9)
            seed_points = np.column_stack([x_seeds, y_seeds])
        else:
            seed_points = None
        
        speed = np.sqrt(Vx**2 + Vy**2)
        
        strm_kwargs = dict(
            color=speed, cmap=cmap,
            linewidth=1.0, density=density,
            arrowsize=1.2, arrowstyle='->',
            maxlength=10.0, broken_streamlines=False
        )
        
        if seed_points is not None:
            strm_kwargs['start_points'] = seed_points
            strm_kwargs['integration_direction'] = 'both'
        
        strm = ax.streamplot(XX[0, :], YY[:, 0], Vx, Vy, **strm_kwargs)
        plt.colorbar(strm.lines, ax=ax, label='|V|')
        
        if show_body:
            self._draw_body_outline(ax, mesh)
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.grid(True, alpha=0.3)
    
    def plot_contours(self,
                     XX: NDArray, YY: NDArray, Vx: NDArray, Vy: NDArray,
                     mesh: Mesh,
                     ax_index: Optional[int] = None,
                     levels: int = 20,
                     show_body: bool = True,
                     title: str = "Velocity Magnitude",
                     cmap: str = 'jet'):
        """
        Plot velocity magnitude contours.
        
        Args:
            XX, YY: Meshgrid coordinates
            Vx, Vy: Velocity components
            mesh: Body mesh (for outline)
            ax_index: Subplot index
            levels: Number of contour levels
            show_body: Draw body outline
            title: Subplot title
            cmap: Colormap
        """
        ax = self._get_ax(ax_index)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        
        speed = np.sqrt(Vx**2 + Vy**2)
        
        cf = ax.contourf(XX, YY, speed, levels=levels, cmap=cmap)
        plt.colorbar(cf, ax=ax, label='|V|')
        
        if show_body:
            self._draw_body_outline(ax, mesh, fill=True)
        
        ax.set_xlim(XX[0, 0], XX[0, -1])
        ax.set_ylim(YY[0, 0], YY[-1, 0])
        ax.grid(True, alpha=0.3)
    
    def plot_cp(self,
               mesh: Mesh,
               Cp: NDArray,
               ax_index: Optional[int] = None,
               title: str = "Pressure Coefficient"):
        """
        Plot Cp distribution around body.
        
        Args:
            mesh: Body mesh
            Cp: Pressure coefficients per panel
            ax_index: Subplot index
            title: Plot title
        """
        ax = self._get_ax(ax_index)
        ax.set_xlabel('Panel Index (or θ)')
        ax.set_ylabel('Cp')
        ax.set_title(title)
        
        x = np.arange(len(Cp))
        ax.plot(x, Cp, 'b-o', markersize=4)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.invert_yaxis()  # Convention: negative Cp (suction) at top
        ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Generic Field Plotting (for FieldData)
    # -------------------------------------------------------------------------
    
    def plot_scalar_field(self,
                         data: NDArray,
                         XX: NDArray,
                         YY: NDArray,
                         mesh: Optional[Mesh] = None,
                         ax_index: Optional[int] = None,
                         levels: int = 20,
                         show_body: bool = True,
                         show_iso: bool = False,
                         iso_levels: Optional[List[float]] = None,
                         title: str = "Field",
                         cmap: str = 'jet',
                         label: str = "",
                         symmetric: bool = False) -> Axes:
        """
        Plot any scalar field as contours.
        
        This is the generic method for plotting fields from FieldData.
        
        Args:
            data: 2D array of values (ny, nx)
            XX, YY: Meshgrid coordinates
            mesh: Body mesh (for outline), optional
            ax_index: Subplot index
            levels: Number of filled contour levels
            show_body: Draw body outline (if mesh provided)
            show_iso: Draw iso-contour lines
            iso_levels: Specific values for iso-contours (auto if None)
            title: Plot title
            cmap: Colormap
            label: Colorbar label
            symmetric: Center colormap at zero (for diverging data)
            
        Returns:
            Axes object for further customization
        """
        ax = self._get_ax(ax_index)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        
        # Handle symmetric colormap (for things like pressure_gauge)
        if symmetric:
            vmax = np.nanmax(np.abs(data))
            vmin = -vmax
        else:
            vmin, vmax = np.nanmin(data), np.nanmax(data)
        
        # Filled contours
        cf = ax.contourf(XX, YY, data, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(cf, ax=ax, label=label)
        
        # Iso-contour lines
        if show_iso:
            if iso_levels is None:
                iso_levels = levels // 4 if isinstance(levels, int) else 5
            cs = ax.contour(XX, YY, data, levels=iso_levels, colors='black', 
                           linewidths=0.5, alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.2g')
        
        if show_body and mesh is not None:
            self._draw_body_outline(ax, mesh, fill=True)
        
        ax.set_xlim(XX[0, 0], XX[0, -1])
        ax.set_ylim(YY[0, 0], YY[-1, 0])
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_field(self,
                  field_name: str,
                  fields: 'FieldData',
                  mesh: Optional[Mesh] = None,
                  ax_index: Optional[int] = None,
                  component: str = "magnitude",
                  **kwargs) -> Axes:
        """
        Plot a named field from FieldData container.
        
        This is the recommended interface for post-processing visualization.
        
        Args:
            field_name: Name of the field (e.g., "velocity", "pressure")
            fields: FieldData container
            mesh: Body mesh for outline
            ax_index: Subplot index
            component: For vector fields - "magnitude", "x", or "y"
            **kwargs: Passed to plot_scalar_field (levels, cmap, etc.)
            
        Returns:
            Axes object
            
        Usage:
            viz.plot_field("pressure", fields, mesh, title="Static Pressure")
            viz.plot_field("velocity", fields, mesh, component="x", title="Vx")
            viz.plot_field("pressure_coefficient", fields, mesh, symmetric=True)
        """
        # Import here to avoid circular dependency
        from postprocessing.fields import FieldData, ScalarField, VectorField
        
        if field_name not in fields:
            raise ValueError(f"Field '{field_name}' not found. Available: {fields.available}")
        
        fld = fields[field_name]
        
        # Handle vector fields
        if isinstance(fld, VectorField):
            scalar = fld.to_scalar(component)
            data = scalar.data
            default_title = scalar.name
            default_label = f"{scalar.name} [{scalar.units}]" if scalar.units else scalar.name
        else:
            data = fld.data
            default_title = fld.name
            default_label = f"{fld.name} [{fld.units}]" if fld.units else fld.name
        
        # Apply defaults
        kwargs.setdefault("title", default_title)
        kwargs.setdefault("label", default_label)
        
        return self.plot_scalar_field(
            data=data,
            XX=fields.XX,
            YY=fields.YY,
            mesh=mesh,
            ax_index=ax_index,
            **kwargs
        )
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _compute_normal_scale(self, mesh: Mesh) -> float:
        """Compute appropriate scale for normal arrows."""
        if mesh.areas is not None:
            return 0.3 * np.mean(mesh.areas)
        return 0.1
    
    def _auto_scale_axis(self, ax: Axes, points: NDArray, padding: float = 0.3):
        """Auto-scale axis with padding."""
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        x_range = max(x_max - x_min, 1e-10)
        y_range = max(y_max - y_min, 1e-10)
        
        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    
    def _draw_body_outline(self, ax: Axes, mesh: Mesh, fill: bool = False):
        """Draw body outline on axis, handling multiple components separately."""
        from matplotlib.patches import Polygon
        
        # Group panels by component
        component_ids = np.unique(mesh.component_ids)
        
        for comp_id in component_ids:
            # Get panel indices for this component
            comp_mask = mesh.component_ids == comp_id
            comp_panel_indices = np.where(comp_mask)[0]
            
            if len(comp_panel_indices) == 0:
                continue
            
            # Collect ordered nodes for this component
            # Panels should be in order, so we can build the outline
            comp_nodes = []
            for i in comp_panel_indices:
                n1_idx, n2_idx = mesh.panels[i]
                p1 = mesh.nodes[n1_idx, :2]
                p2 = mesh.nodes[n2_idx, :2]
                # Draw the panel edge
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=2, zorder=5)
                comp_nodes.append(p1)
            
            if fill and len(comp_nodes) > 2:
                # Close the polygon and fill
                comp_nodes = np.array(comp_nodes)
                poly = Polygon(comp_nodes, facecolor='white', edgecolor='black', 
                              linewidth=2, zorder=4, closed=True)
                ax.add_patch(poly)
    
    def _draw_freestream_arrow(self, ax: Axes, freestream: NDArray):
        """Draw freestream velocity indicator."""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        arrow_x = xlim[0] + 0.1 * (xlim[1] - xlim[0])
        arrow_y = ylim[1] - 0.1 * (ylim[1] - ylim[0])
        
        fs_norm = freestream[:2] / (np.linalg.norm(freestream[:2]) + 1e-10)
        arrow_length = 0.12 * (xlim[1] - xlim[0])
        
        ax.arrow(arrow_x, arrow_y,
                arrow_length * fs_norm[0], arrow_length * fs_norm[1],
                head_width=0.1, head_length=0.1,
                fc='blue', ec='blue', alpha=0.8, linewidth=2,
                length_includes_head=True, zorder=15)
        
        ax.text(arrow_x, arrow_y + 0.05 * (ylim[1] - ylim[0]),
               f'V∞', fontsize=10, color='blue', fontweight='bold')
