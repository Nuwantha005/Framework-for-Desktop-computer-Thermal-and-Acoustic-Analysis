"""
Streamline plotting for 2D velocity fields.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from numpy.typing import NDArray

from core.geometry.mesh import Mesh


class StreamlinePlotter:
    """
    Plots streamlines from a precomputed 2D velocity field.
    """
    
    def __init__(self, mesh: Mesh):
        """
        Initialize streamline plotter.
        
        Args:
            mesh: Mesh geometry for body outline
        """
        self.mesh = mesh
    
    def plot(self,
             XX: NDArray,
             YY: NDArray,
             Vx: NDArray,
             Vy: NDArray,
             density: float = 1.0,
             seed_style: str = 'left',
             figsize: Tuple[float, float] = (12, 8),
             show_body: bool = True,
             show_cp: bool = False,
             save_path: Optional[str] = None):
        """
        Plot streamlines on a precomputed velocity field.
        
        Args:
            XX, YY: Meshgrid coordinates
            Vx, Vy: Velocity components on grid
            density: Streamline density multiplier
            seed_style: 'left' | 'uniform' | 'auto'
            figsize: Figure dimensions
            show_body: Draw body outline
            show_cp: Color body by Cp (if available)
            save_path: Output file path (None = show)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine streamline seeding
        ny, nx = XX.shape
        x_range = (XX.min(), XX.max())
        y_range = (YY.min(), YY.max())
        
        if seed_style == 'left':
            n_lines = int(ny * 0.3 * density)
            y_seeds = np.linspace(y_range[0], y_range[1], n_lines)
            x_seeds = np.full_like(y_seeds, x_range[0])
            seed_points = np.column_stack([x_seeds, y_seeds])
        elif seed_style == 'uniform':
            n_lines = int(ny * 0.5 * density)
            y_seeds = np.linspace(y_range[0], y_range[1], n_lines)
            x_seeds = np.full_like(y_seeds, x_range[0] * 0.9)
            seed_points = np.column_stack([x_seeds, y_seeds])
        else:
            seed_points = None
        
        # Velocity magnitude for coloring
        speed = np.sqrt(Vx**2 + Vy**2)
        
        # Plot streamlines
        if seed_points is not None:
            strm = ax.streamplot(
                XX[0, :], YY[:, 0], Vx, Vy,
                color=speed, cmap='viridis',
                linewidth=1.0, density=density,
                start_points=seed_points,
                arrowsize=1.2, arrowstyle='->',
                maxlength=10.0,
                integration_direction='both',
                broken_streamlines=False
            )
        else:
            strm = ax.streamplot(
                XX[0, :], YY[:, 0], Vx, Vy,
                color=speed, cmap='viridis',
                linewidth=1.0, density=density,
                arrowsize=1.2, arrowstyle='->',
                maxlength=10.0,
                broken_streamlines=False
            )
        
        plt.colorbar(strm.lines, ax=ax, label='Velocity Magnitude')
        
        # Draw body
        if show_body:
            if show_cp and 'Cp' in self.mesh.cell_data:
                cp_vals = self.mesh.cell_data['Cp']
                scatter = ax.scatter(
                    self.mesh.centers[:, 0],
                    self.mesh.centers[:, 1],
                    c=cp_vals, cmap='jet',
                    s=30, edgecolors='black', linewidths=0.5,
                    zorder=10
                )
                plt.colorbar(scatter, ax=ax, label='Cp')
            
            for panel in self.mesh.panels:
                panel_nodes = self.mesh.nodes[panel]
                ax.plot(panel_nodes[:, 0], panel_nodes[:, 1],
                       'k-', lw=1.5, zorder=5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Streamlines')
        ax.set_aspect('equal')
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Streamlines saved: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
