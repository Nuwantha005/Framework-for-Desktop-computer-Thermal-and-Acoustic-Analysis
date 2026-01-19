"""
Main visualization facade for 2D panel method results.
Coordinates field computation and plotting with caching.
"""

from typing import Tuple, Optional
from numpy.typing import NDArray

from core.geometry.mesh import Mesh
from .field2d import VelocityField2D
from .plotters import StreamlinePlotter, ContourPlotter


class PanelVisualizer2D:
    """
    Unified visualization interface for 2D panel methods.
    
    Separates expensive field computation from plotting, enabling:
    - Compute velocity field once, plot multiple times
    - Independent plot types (streamlines, contours, Cp, etc.)
    - Easy extension for new visualization types
    
    Usage:
        viz = PanelVisualizer2D(mesh, v_inf, aoa, sigma)
        
        # Option A: Compute once, plot many
        viz.compute_field((-2, 2), (-1.5, 1.5), (200, 150), num_cores=6)
        viz.plot_streamlines(save_path='streamlines.png')
        viz.plot_contours(save_path='contours.png')
        
        # Option B: Automatic (computes if needed)
        viz.plot_streamlines(x_range=(-2, 2), y_range=(-1.5, 1.5),
                            resolution=(200, 150), save_path='out.png')
    """
    
    def __init__(self,
                 mesh: Mesh,
                 v_inf: float,
                 aoa: float,
                 source_strengths: NDArray):
        """
        Initialize 2D panel visualizer.
        
        Args:
            mesh: Solved 2D mesh
            v_inf: Freestream velocity
            aoa: Angle of attack (degrees)
            source_strengths: Panel source strengths (sigma)
        """
        self.mesh = mesh
        self.v_inf = v_inf
        self.aoa = aoa
        
        # Field computer with caching
        self.field = VelocityField2D(mesh, v_inf, aoa, source_strengths)
        
        # Plotters
        self.streamline_plotter = StreamlinePlotter(mesh)
        self.contour_plotter = ContourPlotter(mesh)
    
    def compute_field(self,
                     x_range: Tuple[float, float],
                     y_range: Tuple[float, float],
                     resolution: Tuple[int, int] = (100, 100),
                     num_cores: int = 6,
                     force: bool = False):
        """
        Compute and cache velocity field.
        
        Args:
            x_range: (xmin, xmax) domain
            y_range: (ymin, ymax) domain
            resolution: (nx, ny) grid points
            num_cores: Parallel worker count
            force: Recompute even if cached
        """
        self.field.compute(x_range, y_range, resolution, num_cores, force)
    
    def plot_streamlines(self,
                        x_range: Optional[Tuple[float, float]] = None,
                        y_range: Optional[Tuple[float, float]] = None,
                        resolution: Optional[Tuple[int, int]] = None,
                        num_cores: int = 6,
                        density: float = 1.0,
                        seed_style: str = 'left',
                        figsize: Tuple[float, float] = (12, 8),
                        show_body: bool = True,
                        show_cp: bool = False,
                        save_path: Optional[str] = None):
        """
        Plot streamlines (computes field if needed).
        
        Args:
            x_range, y_range, resolution: Field parameters (None = use cached)
            num_cores: Parallel cores for computation
            density: Streamline density
            seed_style: 'left' | 'uniform' | 'auto'
            figsize: Figure size
            show_body: Draw body outline
            show_cp: Show Cp on body
            save_path: Output path (None = show)
        """
        # Get or compute field
        if x_range is not None and y_range is not None and resolution is not None:
            XX, YY, Vx, Vy = self.field.compute(
                x_range, y_range, resolution, num_cores
            )
        else:
            cached = self.field.get_cached()
            if cached is None:
                raise ValueError(
                    "No cached field. Provide x_range, y_range, resolution "
                    "or call compute_field() first."
                )
            XX, YY, Vx, Vy = cached
        
        # Plot
        self.streamline_plotter.plot(
            XX, YY, Vx, Vy,
            density=density,
            seed_style=seed_style,
            figsize=figsize,
            show_body=show_body,
            show_cp=show_cp,
            save_path=save_path
        )
    
    def plot_contours(self,
                     x_range: Optional[Tuple[float, float]] = None,
                     y_range: Optional[Tuple[float, float]] = None,
                     resolution: Optional[Tuple[int, int]] = None,
                     num_cores: int = 6,
                     levels: int = 20,
                     figsize: Tuple[float, float] = (10, 8),
                     show_body: bool = True,
                     save_path: Optional[str] = None):
        """
        Plot velocity magnitude contours (computes field if needed).
        
        Args:
            x_range, y_range, resolution: Field parameters (None = use cached)
            num_cores: Parallel cores for computation
            levels: Number of contour levels
            figsize: Figure size
            show_body: Draw body outline
            save_path: Output path (None = show)
        """
        # Get or compute field
        if x_range is not None and y_range is not None and resolution is not None:
            XX, YY, Vx, Vy = self.field.compute(
                x_range, y_range, resolution, num_cores
            )
        else:
            cached = self.field.get_cached()
            if cached is None:
                raise ValueError(
                    "No cached field. Provide x_range, y_range, resolution "
                    "or call compute_field() first."
                )
            XX, YY, Vx, Vy = cached
        
        # Plot
        self.contour_plotter.plot_velocity_magnitude(
            XX, YY, Vx, Vy,
            levels=levels,
            figsize=figsize,
            show_body=show_body,
            save_path=save_path
        )
    
    def plot_cp(self,
               figsize: Tuple[float, float] = (10, 6),
               save_path: Optional[str] = None):
        """
        Plot pressure coefficient distribution on body.
        
        Args:
            figsize: Figure size
            save_path: Output path (None = show)
        """
        self.contour_plotter.plot_pressure_coefficient(
            figsize=figsize,
            save_path=save_path
        )
    
    def clear_cache(self):
        """Clear cached velocity field to free memory."""
        self.field.clear_cache()
