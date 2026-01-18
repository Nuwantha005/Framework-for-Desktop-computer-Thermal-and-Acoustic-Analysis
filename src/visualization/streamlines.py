"""
Streamline visualization for panel method solutions.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import path
from typing import Tuple, Optional
from numpy.typing import NDArray
from multiprocessing import Pool
from functools import partial

from core.geometry.mesh import Mesh

def compute_point_velocity(args):
    """
    Independent function for parallel velocity computation.
    
    Args:
        args: Tuple containing:
            XP, YP: Coordinates of point P
            sigma: Source strengths (N,)
            nodes_X: Panel start X coordinates (N,)
            nodes_Y: Panel start Y coordinates (N,)
            phi: Panel angles (N,)
            S: Panel lengths (N,)
            v_inf: Freestream velocity
            aoa_rad: Angle of attack (radians)
            
    Returns:
        (Vx, Vy): Velocity components at P
    """
    XP, YP, sigma, nodes_X, nodes_Y, phi, S, v_inf, aoa_rad = args
    
    # Vectorized computation for all panels at once for a single point P
    dx = XP - nodes_X
    dy = YP - nodes_Y
    
    # Intermediate terms
    # A = -(x-xb)cos(phi) - (y-yb)sin(phi)
    # B = (x-xb)^2 + (y-yb)^2
    # C = -cos(phi)
    # D = x-xb
    
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    A = -dx * cos_phi - dy * sin_phi
    B = dx**2 + dy**2
    
    # Avoid numerical issues with B close to 0
    B = np.maximum(B, 1e-12)
    
    E_sq = B - A**2
    # Numerical stability check - if point is on line of panel extension
    E_sq = np.maximum(E_sq, 0)
    E = np.sqrt(E_sq)
    
    # Terms for Mx (X-direction integral)
    # Ref: Katz & Plotkin
    Cx = -cos_phi
    Dx = dx
    
    # Terms for My (Y-direction integral)
    Cy = -sin_phi
    Dy = dy
    
    # Safe computation for logarithm
    # If point is extremely close to panel start (B~0) or end (S^2+2AS+B ~ 0), this blows up.
    # But B is clamped.
    
    log_term = np.log(np.maximum((S**2 + 2*A*S + B) / B, 1e-12))
    
    # Safe computation for atan2
    # Force E to be non-zero for division, handle result masking later
    E_safe = np.where(E < 1e-12, 1.0, E) # Avoiding div by zero
    atan_term = np.arctan2((S + A), E_safe) - np.arctan2(A, E_safe)
    
    term2_factor = 1.0 / E_safe
    
    # Compute Mx
    Mx = 0.5 * Cx * log_term + ((Dx - A*Cx) * term2_factor) * atan_term
    
    # Compute My
    My = 0.5 * Cy * log_term + ((Dy - A*Cy) * term2_factor) * atan_term
    
    # Zero out contributions where E was effectively 0 (on line)
    mask = E < 1e-12
    Mx[mask] = 0
    My[mask] = 0
    
    # Sum contributions
    u_induced = np.sum(sigma * Mx) / (2 * np.pi)
    v_induced = np.sum(sigma * My) / (2 * np.pi)
    
    Vx = v_inf * np.cos(aoa_rad) + u_induced
    Vy = v_inf * np.sin(aoa_rad) + v_induced
    
    return Vx, Vy


class StreamlineVisualizer:
    """
    Compute and visualize streamlines for 2D panel method solutions.
    """
    
    def __init__(self, mesh: Mesh, v_inf: float, aoa: float, source_strengths: NDArray):
        """
        Initialize streamline visualizer.
        
        Args:
            mesh: The solved mesh
            v_inf: Freestream velocity magnitude
            aoa: Angle of attack in degrees
            source_strengths: Source strength values (sigma) for each panel
        """
        if mesh.dimension != 2:
            raise ValueError("StreamlineVisualizer only supports 2D meshes")
            
        self.mesh = mesh
        self.v_inf = v_inf
        self.aoa_rad = np.radians(aoa)
        self.sigma = source_strengths
        
        # Extract geometry - Prepare arrays for parallel workers
        # Need panel start points
        # Assuming mesh.panels contains indices [i, j]
        panel_start_indices = self.mesh.panels[:, 0]
        self.nodes_X = self.mesh.nodes[panel_start_indices, 0]
        self.nodes_Y = self.mesh.nodes[panel_start_indices, 1]
        self.S = self.mesh.areas
        
        # Panel angles
        tx = mesh.tangents[:, 0]
        ty = mesh.tangents[:, 1]
        self.phi = np.arctan2(ty, tx)
        self.phi = np.where(self.phi < 0, self.phi + 2*np.pi, self.phi)
        
        # Create path for inside/outside detection
        boundary_points = np.column_stack([self.mesh.nodes[:, 0], self.mesh.nodes[:, 1]])
        self.boundary_path = path.Path(boundary_points)
        
        # Center points for plotting Cp
        self.XC = mesh.centers[:, 0]
        self.YC = mesh.centers[:, 1]
    
    def compute_velocity_field(self, 
                               x_range: Tuple[float, float],
                               y_range: Tuple[float, float],
                               grid_resolution: Tuple[int, int] = (100, 100),
                               num_cores: int = 6
                               ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Compute velocity field on a grid using parallel processing.
        
        Args:
            x_range: (xmin, xmax) for grid extent
            y_range: (ymin, ymax) for grid extent
            grid_resolution: (nx, ny) number of grid points
            num_cores: Number of processes to use
            
        Returns:
            (XX, YY, Vx, Vy): Grid coordinates and velocity components
        """
        nx, ny = grid_resolution
        xmin, xmax = x_range
        ymin, ymax = y_range
        
        x_grid = np.linspace(xmin, xmax, nx)
        y_grid = np.linspace(ymin, ymax, ny)
        XX, YY = np.meshgrid(x_grid, y_grid)
        
        # Flatten grid points for mapping
        points_flat = np.column_stack([XX.ravel(), YY.ravel()])
        n_points = len(points_flat)
        
        tasks = []
        for i in range(n_points):
            tasks.append((
                points_flat[i, 0],
                points_flat[i, 1],
                self.sigma,
                self.nodes_X,
                self.nodes_Y,
                self.phi,
                self.S,
                self.v_inf,
                self.aoa_rad
            ))
            
        print(f"Computing velocity field on {nx}x{ny} grid using {num_cores} cores...")
        
        with Pool(processes=num_cores) as pool:
            results = pool.map(compute_point_velocity, tasks)
            
        # Unpack results
        results_arr = np.array(results)
        Vx_flat = results_arr[:, 0]
        Vy_flat = results_arr[:, 1]
        
        # Mask points inside body
        is_inside = self.boundary_path.contains_points(points_flat)
        Vx_flat[is_inside] = np.nan
        Vy_flat[is_inside] = np.nan
        
        # Reshape to grid
        Vx = Vx_flat.reshape(ny, nx)
        Vy = Vy_flat.reshape(ny, nx)
        
        return XX, YY, Vx, Vy
    
    def plot_velocity_contours(self,
                              x_range: Tuple[float, float],
                              y_range: Tuple[float, float],
                              grid_resolution: Tuple[int, int] = (100, 100),
                              levels: int = 20,
                              figsize: Tuple[float, float] = (10, 8),
                              save_path: Optional[str] = None):
        """
        Plot velocity magnitude contours to check field smoothness.
        """
        XX, YY, Vx, Vy = self.compute_velocity_field(x_range, y_range, grid_resolution)
        speed = np.sqrt(Vx**2 + Vy**2)
        
        plt.figure(figsize=figsize)
        plt.contourf(XX, YY, speed, levels=levels, cmap='jet')
        plt.colorbar(label='Velocity Magnitude')
        
        # Overlay body
        plt.plot(self.mesh.nodes[:, 0], self.mesh.nodes[:, 1], 'k-', lw=2)
        
        plt.title('Velocity Magnitude Contours')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Contour plot saved to {save_path}")
        else:
            plt.show()    
    
    def plot_streamlines(self,
                         x_range: Tuple[float, float],
                         y_range: Tuple[float, float],
                         grid_resolution: Tuple[int, int] = (100, 100),
                         streamline_density: float = 1.0,
                         streamline_start: str = 'left',
                         figsize: Tuple[float, float] = (12, 8),
                         show_body: bool = True,
                         show_cp: bool = False,
                         save_path: Optional[str] = None,
                         num_cores: int = 6):
        """
        Plot streamlines around the body.
        
        Args:
            x_range: (xmin, xmax) for plot extent
            y_range: (ymin, ymax) for plot extent
            grid_resolution: (nx, ny) grid resolution for velocity field
            streamline_density: Density of streamlines (0.5 = half, 2.0 = double)
            streamline_start: Where to seed streamlines ('left', 'uniform', 'custom')
            figsize: Figure size
            show_body: Whether to show body outline
            show_cp: Whether to color body panels by Cp
            save_path: Path to save figure (if None, will show instead)
            num_cores: Number of parallel cores
        """
        # Compute velocity field
        XX, YY, Vx, Vy = self.compute_velocity_field(x_range, y_range, grid_resolution, num_cores)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot streamlines
        nx, ny = grid_resolution
        if streamline_start == 'left':
            # Seed streamlines from left boundary
            n_lines = int(ny * 0.3 * streamline_density)  # 30% of grid height by default
            y_seeds = np.linspace(y_range[0], y_range[1], n_lines)
            x_seeds = np.full_like(y_seeds, x_range[0])
            seed_points = np.column_stack([x_seeds, y_seeds])
        elif streamline_start == 'uniform':
            # Seed from entire upstream region
            n_lines = int(ny * 0.5 * streamline_density)
            y_seeds = np.linspace(y_range[0], y_range[1], n_lines)
            x_seeds = np.full_like(y_seeds, x_range[0] * 0.9)
            seed_points = np.column_stack([x_seeds, y_seeds])
        else:
            seed_points = None
        
        # Compute velocity magnitude for coloring
        speed = np.sqrt(Vx**2 + Vy**2)
        
        # Plot streamlines
        # Increased maxlength to prevent premature termination
        if seed_points is not None:
            strm = ax.streamplot(XX[0, :], YY[:, 0], Vx, Vy,
                                color=speed, cmap='viridis',
                                linewidth=1.0, density=streamline_density,
                                start_points=seed_points,
                                arrowsize=1.2, arrowstyle='->',
                                maxlength=50.0, integration_direction='both')
        else:
            strm = ax.streamplot(XX[0, :], YY[:, 0], Vx, Vy,
                                color=speed, cmap='viridis',
                                linewidth=1.0, density=streamline_density,
                                arrowsize=1.2, arrowstyle='->',
                                maxlength=50.0)
        
        # Add colorbar
        cbar = plt.colorbar(strm.lines, ax=ax, label='Velocity Magnitude')
        
        # Plot body
        if show_body:
            if show_cp and 'Cp' in self.mesh.cell_data:
                # Color panels by Cp
                cp_vals = self.mesh.cell_data['Cp']
                scatter = ax.scatter(self.XC, self.YC, c=cp_vals, cmap='jet', 
                                    s=30, edgecolors='black', linewidths=0.5,
                                    zorder=10)
                plt.colorbar(scatter, ax=ax, label='Cp (Body)')
            
            # Draw panel edges
            for panel in self.mesh.panels:
                panel_nodes = self.mesh.nodes[panel]
                ax.plot(panel_nodes[:, 0], panel_nodes[:, 1], 'k-', lw=1.5, zorder=5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Streamlines (V∞={self.v_inf:.1f}, α={np.degrees(self.aoa_rad):.1f}°)')
        ax.set_aspect('equal')
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Streamline plot saved to {save_path}")
        else:
            plt.show()
