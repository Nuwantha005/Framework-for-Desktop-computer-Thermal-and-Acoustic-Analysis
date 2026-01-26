"""
Probe point comparison utilities.

Provides functions for extracting and comparing field values
at specific probe point locations.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import griddata

from validation.convergence.metrics import compute_error_metrics, ErrorMetrics


def extract_probe_values(
    points: NDArray,
    field_points: NDArray,
    field_values: NDArray,
    method: str = 'linear'
) -> NDArray:
    """
    Extract field values at probe point locations via interpolation.
    
    Args:
        points: (N, 2) probe point coordinates
        field_points: (M, 2) field sample point coordinates
        field_values: (M,) field values at sample points
        method: Interpolation method ('linear', 'nearest', 'cubic')
    
    Returns:
        (N,) field values at probe points
    
    Examples:
        >>> probes = np.array([[1, 0], [2, 0], [3, 0]])
        >>> field_pts = np.random.rand(100, 2) * 5
        >>> field_vals = np.random.rand(100)
        >>> probe_vals = extract_probe_values(probes, field_pts, field_vals)
    """
    return griddata(field_points, field_values, points, method=method)


def compare_at_probes(
    probe_points: NDArray,
    pm_field_points: NDArray,
    pm_field_values: NDArray,
    of_field_points: NDArray,
    of_field_values: NDArray,
    probe_names: List[str] = None
) -> Tuple[NDArray, NDArray, ErrorMetrics, Dict]:
    """
    Compare panel method and OpenFOAM at specific probe locations.
    
    Args:
        probe_points: (N, 2) probe coordinates
        pm_field_points: Panel method field sample points
        pm_field_values: Panel method field values
        of_field_points: OpenFOAM field sample points
        of_field_values: OpenFOAM field values
        probe_names: Optional names for probe locations
    
    Returns:
        (pm_probe_values, of_probe_values, metrics, probe_data)
    
    Examples:
        >>> probes = np.array([[1, 0], [0, 1], [-1, 0]])
        >>> names = ["right", "top", "left"]
        >>> pm_vals, of_vals, metrics, data = compare_at_probes(
        ...     probes, pm_pts, pm_vals, of_pts, of_vals, names
        ... )
        >>> print(f"RMS error at probes: {metrics.rms_error:.4f}")
    """
    # Extract values at probes
    pm_probe_vals = extract_probe_values(
        probe_points, pm_field_points, pm_field_values
    )
    
    of_probe_vals = extract_probe_values(
        probe_points, of_field_points, of_field_values
    )
    
    # Compute metrics
    metrics = compute_error_metrics(pm_probe_vals, of_probe_vals)
    
    # Build detailed data
    if probe_names is None:
        probe_names = [f"probe_{i}" for i in range(len(probe_points))]
    
    probe_data = {
        'names': probe_names,
        'coordinates': probe_points,
        'panel_method': pm_probe_vals,
        'openfoam': of_probe_vals,
        'error': pm_probe_vals - of_probe_vals,
        'relative_error': (pm_probe_vals - of_probe_vals) / of_probe_vals * 100
    }
    
    return pm_probe_vals, of_probe_vals, metrics, probe_data


def create_probe_line(
    start: Tuple[float, float],
    end: Tuple[float, float],
    num_points: int
) -> NDArray:
    """
    Create a line of probe points between two locations.
    
    Args:
        start: (x, y) start coordinate
        end: (x, y) end coordinate
        num_points: Number of probe points
    
    Returns:
        (num_points, 2) array of probe coordinates
    
    Examples:
        >>> probes = create_probe_line((0, 0), (10, 0), 11)
        >>> print(probes.shape)
        (11, 2)
    """
    x = np.linspace(start[0], end[0], num_points)
    y = np.linspace(start[1], end[1], num_points)
    return np.column_stack([x, y])


def create_probe_grid(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    nx: int,
    ny: int
) -> NDArray:
    """
    Create a grid of probe points.
    
    Args:
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        nx: Number of points in x
        ny: Number of points in y
    
    Returns:
        (nx*ny, 2) array of probe coordinates
    
    Examples:
        >>> probes = create_probe_grid((-5, 5), (-5, 5), 5, 5)
        >>> print(probes.shape)
        (25, 2)
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    XX, YY = np.meshgrid(x, y)
    return np.column_stack([XX.ravel(), YY.ravel()])


def print_probe_comparison(probe_data: Dict, verbose: bool = True):
    """
    Print formatted probe comparison results.
    
    Args:
        probe_data: Probe data dictionary from compare_at_probes
        verbose: If True, print detailed table
    """
    if not verbose:
        return
    
    print("\n" + "=" * 80)
    print("Probe Point Comparison")
    print("=" * 80)
    print(f"{'Name':<15} {'X':>8} {'Y':>8} {'PM':>10} {'OF':>10} {'Error':>10} {'Rel%':>8}")
    print("-" * 80)
    
    for i, name in enumerate(probe_data['names']):
        x, y = probe_data['coordinates'][i]
        pm_val = probe_data['panel_method'][i]
        of_val = probe_data['openfoam'][i]
        error = probe_data['error'][i]
        rel_err = probe_data['relative_error'][i]
        
        print(f"{name:<15} {x:>8.3f} {y:>8.3f} {pm_val:>10.4f} {of_val:>10.4f} "
              f"{error:>10.4f} {rel_err:>7.2f}%")
    
    print("=" * 80)


__all__ = [
    'extract_probe_values',
    'compare_at_probes',
    'create_probe_line',
    'create_probe_grid',
    'print_probe_comparison',
]
