"""
Structured grid comparison between panel method and OpenFOAM.

Provides functions for fair field comparison on a structured grid
in the far-field region (excluding near-body singularities).
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

from validation.convergence.metrics import compute_error_metrics, ErrorMetrics


def create_structured_grid(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    nx: int,
    ny: int
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Create a structured comparison grid.
    
    Args:
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        nx: Number of points in x direction
        ny: Number of points in y direction
    
    Returns:
        (XX, YY, points) where XX, YY are meshgrid arrays and
        points is (N, 2) array of all grid points
    
    Examples:
        >>> XX, YY, pts = create_structured_grid((-5, 5), (-5, 5), 50, 50)
        >>> print(f"Grid shape: {XX.shape}, Points: {len(pts)}")
        Grid shape: (50, 50), Points: 2500
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    XX, YY = np.meshgrid(x, y)
    points = np.column_stack([XX.ravel(), YY.ravel()])
    return XX, YY, points


def filter_points_near_body(
    points: NDArray,
    panel_centers: NDArray,
    min_distance: float = 0.1
) -> NDArray:
    """
    Filter out points that are too close to panel centers.
    
    Panel methods are inaccurate very close to the body surface
    due to singularity effects. This filter removes points within
    min_distance of any panel to allow fair comparison.
    
    Args:
        points: (N, 2) array of points to filter
        panel_centers: (M, 2) array of panel center coordinates
        min_distance: Minimum distance from body surface
    
    Returns:
        Boolean mask of points that are far enough from body
    
    Examples:
        >>> pts = np.array([[0, 0], [1, 0], [10, 0]])
        >>> centers = np.array([[0, 0]])
        >>> mask = filter_points_near_body(pts, centers, min_distance=2.0)
        >>> filtered = pts[mask]
    """
    # Build KD-tree from panel centers
    tree = cKDTree(panel_centers)
    
    # Query for nearest panel to each point
    distances, _ = tree.query(points, k=1)
    
    # Keep points that are far enough away
    return distances > min_distance


def interpolate_openfoam_to_points(
    of_cell_centers: NDArray,
    of_velocity: NDArray,
    target_points: NDArray,
    method: str = 'linear'
) -> NDArray:
    """
    Interpolate OpenFOAM velocity field to target points.
    
    Args:
        of_cell_centers: (N, 2) OpenFOAM cell center coordinates (2D)
        of_velocity: (N, 2) OpenFOAM velocity field (Vx, Vy)
        target_points: (M, 2) target point coordinates
        method: Interpolation method ('linear', 'nearest', 'cubic')
    
    Returns:
        (M, 2) interpolated velocity at target points
    
    Examples:
        >>> centers = np.random.rand(100, 2) * 10
        >>> velocity = np.random.rand(100, 2)
        >>> targets = np.array([[5, 5], [3, 3]])
        >>> interp_vel = interpolate_openfoam_to_points(centers, velocity, targets)
    """
    Vx = griddata(of_cell_centers, of_velocity[:, 0], target_points, method=method)
    Vy = griddata(of_cell_centers, of_velocity[:, 1], target_points, method=method)
    
    return np.column_stack([Vx, Vy])


def compute_panel_velocity_at_points(
    solver,
    points: NDArray
) -> NDArray:
    """
    Compute panel method velocity at arbitrary points using solver.
    
    Args:
        solver: Solved panel method solver instance
        points: (N, 2) array of (x, y) coordinates
    
    Returns:
        (N, 2) array of velocity vectors (Vx, Vy)
    
    Examples:
        >>> from solvers.panel2d.spm import SourcePanelSolver
        >>> # ... (create mesh and solve)
        >>> points = np.array([[1, 0], [2, 0]])
        >>> velocities = compute_panel_velocity_at_points(solver, points)
    """
    # Ensure points are (N, 3) with z=0
    if points.shape[1] == 2:
        points_3d = np.column_stack([points, np.zeros(len(points))])
    else:
        points_3d = points
    
    # Use solver's velocity_at method
    velocities_3d = solver.velocity_at(points_3d)
    
    # Return only x,y components
    return velocities_3d[:, :2]


def compare_on_structured_grid(
    solver,
    of_cell_centers: NDArray,
    of_velocity: NDArray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: Tuple[int, int],
    body_distance_filter: float = 0.5,
    verbose: bool = True
) -> Dict:
    """
    Compare panel method and OpenFOAM on a structured grid.
    
    This is the main function for structured grid comparison. It:
    1. Creates a uniform grid
    2. Filters points near bodies
    3. Interpolates OpenFOAM results to grid
    4. Computes panel method velocity on grid
    5. Computes error metrics
    
    Args:
        solver: Solved panel method solver instance
        of_cell_centers: (N, 3) OpenFOAM cell centers
        of_velocity: (N, 3) OpenFOAM velocity field
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        resolution: (nx, ny) grid resolution
        body_distance_filter: Exclude points within this distance of body
        verbose: Print progress messages
    
    Returns:
        Dictionary with comparison results and error metrics
    
    Examples:
        >>> results = compare_on_structured_grid(
        ...     solver, of_C, of_U,
        ...     (-5, 5), (-5, 5), (100, 80),
        ...     body_distance_filter=0.5
        ... )
        >>> print(f"RMS error: {results['metrics'].rms_error:.4f}")
    """
    if verbose:
        print("Structured Grid Comparison")
        print("=" * 60)
    
    # Create structured grid
    nx, ny = resolution
    XX, YY, grid_points = create_structured_grid(x_range, y_range, nx, ny)
    
    if verbose:
        print(f"  Grid: {nx}×{ny} = {len(grid_points)} points")
    
    # Filter points near bodies
    panel_centers_2d = solver.mesh.centers[:, :2]
    far_mask = filter_points_near_body(grid_points, panel_centers_2d, body_distance_filter)
    comparison_points = grid_points[far_mask]
    
    if verbose:
        print(f"  After body filter ({body_distance_filter}m): {len(comparison_points)} points")
    
    # Interpolate OpenFOAM to comparison points (extract 2D midplane)
    z_mid = 0.05
    z_mask = np.abs(of_cell_centers[:, 2] - z_mid) < 0.02
    of_points_2d = of_cell_centers[z_mask, :2]
    of_U_2d = of_velocity[z_mask, :2]
    
    U_of_interp = interpolate_openfoam_to_points(of_points_2d, of_U_2d, comparison_points)
    
    # Remove NaN from extrapolation
    valid_mask = ~(np.isnan(U_of_interp[:, 0]) | np.isnan(U_of_interp[:, 1]))
    comparison_points = comparison_points[valid_mask]
    U_of_interp = U_of_interp[valid_mask]
    
    if verbose:
        print(f"  Valid comparison points: {len(comparison_points)}")
    
    # Compute panel method velocity
    U_pm = compute_panel_velocity_at_points(solver, comparison_points)
    
    # Compute error metrics (using velocity magnitude)
    V_of = np.sqrt((U_of_interp**2).sum(axis=1))
    V_pm = np.sqrt((U_pm**2).sum(axis=1))
    
    metrics = compute_error_metrics(V_pm, V_of)
    
    if verbose:
        print("\n" + str(metrics))
    
    return {
        'comparison_points': comparison_points,
        'U_openfoam': U_of_interp,
        'U_panel': U_pm,
        'V_openfoam': V_of,
        'V_panel': V_pm,
        'metrics': metrics,
        'grid_resolution': resolution,
        'body_distance_filter': body_distance_filter
    }
