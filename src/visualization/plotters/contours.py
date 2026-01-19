"""
Contour plotting for 2D scalar fields.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from numpy.typing import NDArray

from core.geometry.mesh import Mesh


class ContourPlotter:
    """
    Plots contours (filled or line) for scalar fields.
    """
    
    def __init__(self, mesh: Mesh):
        """
        Initialize contour plotter.
        
        Args:
            mesh: Mesh geometry for body outline
        """
        self.mesh = mesh
    
    def plot_velocity_magnitude(self,
                                XX: NDArray,
                                YY: NDArray,
                                Vx: NDArray,
                                Vy: NDArray,
                                levels: int = 20,
                                figsize: Tuple[float, float] = (10, 8),
                                show_body: bool = True,
                                save_path: Optional[str] = None):
        """
        Plot velocity magnitude contours.
        
        Args:
            XX, YY: Meshgrid coordinates
            Vx, Vy: Velocity components
            levels: Number of contour levels
            figsize: Figure dimensions
            show_body: Draw body outline
            save_path: Output file path (None = show)
        """
        speed = np.sqrt(Vx**2 + Vy**2)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        contour = ax.contourf(XX, YY, speed, levels=levels, cmap='jet')
        plt.colorbar(contour, ax=ax, label='Velocity Magnitude')
        
        if show_body:
            ax.plot(self.mesh.nodes[:, 0], self.mesh.nodes[:, 1],
                   'k-', lw=2, label='Body')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Velocity Magnitude Contours')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Contours saved: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
    
    def plot_pressure_coefficient(self,
                                  figsize: Tuple[float, float] = (10, 6),
                                  save_path: Optional[str] = None):
        """
        Plot Cp distribution on body surface.
        
        Args:
            figsize: Figure dimensions
            save_path: Output file path (None = show)
        """
        if 'Cp' not in self.mesh.cell_data:
            raise ValueError("Mesh does not contain Cp data")
        
        cp_vals = self.mesh.cell_data['Cp']
        x_centers = self.mesh.centers[:, 0]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(x_centers, cp_vals, 'o-', markersize=5, linewidth=1.5)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Cp')
        ax.set_title('Pressure Coefficient Distribution')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Cp plot saved: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
