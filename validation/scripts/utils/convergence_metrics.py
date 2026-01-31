"""
Convergence metrics computation for OpenFOAM and panel method studies.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray


def extract_monitoring_point_data(
    openfoam_case_dir: Path,
    monitoring_points: List[Dict[str, any]],
    time_idx: int = -1
) -> Dict[str, Dict[str, float]]:
    """
    Extract velocity and pressure at monitoring points from OpenFOAM case.
    
    Uses scipy.interpolate to get values at exact point locations from
    the cell-centered OpenFOAM data.
    
    Args:
        openfoam_case_dir: Path to OpenFOAM case
        monitoring_points: List of dicts with 'name' and 'coordinates'
        time_idx: Time index to extract (-1 for latest)
    
    Returns:
        Dict mapping point name to {'velocity': float, 'pressure': float}
    """
    from scipy.interpolate import griddata
    from validation.adapters.openfoam import OpenFOAMRunner
    
    # Load OpenFOAM data
    runner = OpenFOAMRunner(openfoam_case_dir, verbose=False)
    
    # Get cell centers
    cell_centers = runner.get_cell_centres(time_idx=time_idx)
    
    # Get velocity and pressure fields
    velocity = runner.get_velocity_field(time_idx=time_idx)
    pressure = runner.get_pressure_field(time_idx=time_idx)
    
    # Compute velocity magnitude
    velocity_mag = np.linalg.norm(velocity, axis=1)
    
    # Extract 2D coordinates (x, y)
    points_2d = cell_centers[:, :2]
    
    # Interpolate at monitoring points
    results = {}
    for point in monitoring_points:
        name = point['name']
        coords = np.array(point['coordinates'])
        
        # Interpolate velocity magnitude
        vel = griddata(points_2d, velocity_mag, coords.reshape(1, -1), method='linear')[0]
        
        # Interpolate pressure
        p = griddata(points_2d, pressure, coords.reshape(1, -1), method='linear')[0]
        
        results[name] = {
            'velocity': float(vel),
            'pressure': float(p),
        }
    
    return results


def compute_convergence_metrics(
    values: List[float],
    mesh_sizes: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute convergence metrics from a series of values at different mesh resolutions.
    
    Computes:
    - Change between consecutive levels
    - Relative change (percentage)
    - Grid Convergence Index (GCI) if 3+ levels
    - Asymptotic convergence ratio
    
    Args:
        values: List of values at different mesh levels (coarse to fine)
        mesh_sizes: Optional list of characteristic mesh sizes (for GCI)
    
    Returns:
        Dictionary of convergence metrics
    """
    if len(values) < 2:
        raise ValueError("Need at least 2 values for convergence analysis")
    
    values = np.array(values)
    n_levels = len(values)
    
    metrics = {
        'n_levels': n_levels,
        'values': values.tolist(),
        'changes': [],
        'relative_changes': [],
    }
    
    # Compute changes between consecutive levels
    for i in range(1, n_levels):
        change = values[i] - values[i-1]
        rel_change = change / values[i-1] if values[i-1] != 0 else 0.0
        metrics['changes'].append(float(change))
        metrics['relative_changes'].append(float(rel_change))
    
    # If we have 3+ levels, compute GCI
    if n_levels >= 3:
        # Use last 3 levels for GCI computation
        f3, f2, f1 = values[-3], values[-2], values[-1]  # coarse, medium, fine
        
        # Estimate apparent order of convergence
        if mesh_sizes is not None and len(mesh_sizes) >= 3:
            h3, h2, h1 = mesh_sizes[-3], mesh_sizes[-2], mesh_sizes[-1]
            r21 = h2 / h1
            r32 = h3 / h2
        else:
            # Assume uniform refinement ratio
            r21 = 1.5
            r32 = 1.5
        
        epsilon_21 = f2 - f1
        epsilon_32 = f3 - f2
        
        # Apparent order of convergence
        if abs(epsilon_32) > 1e-10:
            p = abs(np.log(abs(epsilon_21 / epsilon_32)) / np.log(r21))
        else:
            p = 1.0  # First-order as fallback
        
        # GCI for fine-medium
        if f1 != 0:
            gci_21 = 1.25 * abs(epsilon_21 / f1) / (r21**p - 1)
        else:
            gci_21 = float('inf')
        
        # Asymptotic convergence ratio
        if abs(epsilon_21) > 1e-10:
            gci_32 = 1.25 * abs(epsilon_32 / f2) / (r32**p - 1)
            asymptotic_ratio = gci_32 / (r21**p * gci_21) if gci_21 != 0 else 0
        else:
            asymptotic_ratio = 0
        
        metrics['gci'] = float(gci_21)
        metrics['order_of_convergence'] = float(p)
        metrics['asymptotic_ratio'] = float(asymptotic_ratio)
        metrics['converged'] = (gci_21 < 0.05)  # 5% threshold
    
    return metrics


def compute_panel_convergence_error(
    panel_values: List[float],
    reference_value: float
) -> Dict[str, any]:
    """
    Compute error metrics for panel method convergence against reference.
    
    Args:
        panel_values: List of panel method values at different resolutions
        reference_value: Reference value (typically from finest OpenFOAM)
    
    Returns:
        Dictionary with error metrics at each level
    """
    panel_values = np.array(panel_values)
    
    # Absolute errors
    abs_errors = np.abs(panel_values - reference_value)
    
    # Relative errors (percentage)
    if reference_value != 0:
        rel_errors = abs_errors / abs(reference_value) * 100
    else:
        rel_errors = np.zeros_like(abs_errors)
    
    return {
        'reference_value': float(reference_value),
        'panel_values': panel_values.tolist(),
        'absolute_errors': abs_errors.tolist(),
        'relative_errors': rel_errors.tolist(),
        'convergence_rate': _estimate_convergence_rate(panel_values),
    }


def _estimate_convergence_rate(values: NDArray) -> Optional[float]:
    """
    Estimate convergence rate from sequence of values.
    
    Fits a power law: error ~ N^(-rate) where N is panel count
    
    Returns:
        Estimated convergence rate or None if cannot compute
    """
    if len(values) < 3:
        return None
    
    # Compute successive differences (proxy for error)
    diffs = np.abs(np.diff(values))
    
    if len(diffs) < 2 or np.any(diffs == 0):
        return None
    
    # Fit power law to last 3 points
    log_diffs = np.log(diffs[-2:])
    x = np.array([0, 1])  # Sequential levels
    
    try:
        rate = -np.polyfit(x, log_diffs, 1)[0]
        return float(rate)
    except:
        return None


def format_convergence_table(
    metrics: Dict[str, Dict[str, any]],
    level_names: List[str],
    quantities: List[str] = ['velocity', 'pressure']
) -> str:
    """
    Format convergence results as a text table.
    
    Args:
        metrics: Dictionary mapping level names to metrics
        level_names: Ordered list of level names
        quantities: Quantities to include in table
    
    Returns:
        Formatted table string
    """
    lines = []
    lines.append("Convergence Analysis Results")
    lines.append("=" * 80)
    
    for qty in quantities:
        lines.append(f"\n{qty.upper()}:")
        lines.append("-" * 80)
        
        # Header
        lines.append(f"{'Level':<15} {'Value':>12} {'Change':>12} {'Rel Change':>12}")
        lines.append("-" * 80)
        
        # Data rows
        for i, level in enumerate(level_names):
            if level not in metrics:
                continue
            
            value = metrics[level].get(qty, 0)
            
            if i == 0:
                change_str = "—"
                rel_change_str = "—"
            else:
                prev_level = level_names[i-1]
                prev_value = metrics[prev_level].get(qty, 0)
                change = value - prev_value
                rel_change = (change / prev_value * 100) if prev_value != 0 else 0
                change_str = f"{change:+.6f}"
                rel_change_str = f"{rel_change:+.2f}%"
            
            lines.append(f"{level:<15} {value:12.6f} {change_str:>12} {rel_change_str:>12}")
    
    return "\n".join(lines)
