"""
Convergence metrics computation.

Provides functions for computing error metrics between fields,
Grid Convergence Index (GCI), and Richardson extrapolation.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Dict
from scipy.interpolate import griddata


@dataclass
class ErrorMetrics:
    """Container for error metrics between two fields."""
    l2_error: float  # L2 norm
    linf_error: float  # L-infinity norm (max absolute error)
    rms_error: float  # Root mean square error
    mae: float  # Mean absolute error
    relative_l2: float  # Relative L2 error (%)
    relative_rms: float  # Relative RMS error (%)
    num_points: int  # Number of comparison points
    
    def __str__(self) -> str:
        return (
            f"Error Metrics (n={self.num_points}):\n"
            f"  L2:        {self.l2_error:.6f}\n"
            f"  L∞:        {self.linf_error:.6f}\n"
            f"  RMS:       {self.rms_error:.6f}\n"
            f"  MAE:       {self.mae:.6f}\n"
            f"  Rel L2:    {self.relative_l2:.2f}%\n"
            f"  Rel RMS:   {self.relative_rms:.2f}%"
        )


@dataclass
class GCIResult:
    """Grid Convergence Index result."""
    gci_fine: float  # GCI on fine mesh
    gci_medium: float  # GCI on medium mesh
    observed_order: float  # Observed order of convergence
    asymptotic_ratio: float  # Ratio for checking asymptotic range
    refinement_ratio: float  # Grid refinement ratio
    extrapolated_value: Optional[float] = None  # Richardson extrapolation
    
    def is_asymptotic(self, tolerance: float = 0.05) -> bool:
        """Check if solution is in asymptotic range (ratio ≈ 1)."""
        return abs(self.asymptotic_ratio - 1.0) < tolerance


def compute_error_metrics(
    field1: NDArray,
    field2: NDArray,
    reference_norm: Optional[float] = None
) -> ErrorMetrics:
    """
    Compute error metrics between two fields.
    
    Args:
        field1: First field (e.g., computed solution)
        field2: Second field (e.g., reference solution)
        reference_norm: Reference value for relative errors (default: mean of field2)
    
    Returns:
        ErrorMetrics with all computed metrics
    
    Examples:
        >>> field1 = np.array([1.0, 2.0, 3.0])
        >>> field2 = np.array([1.1, 2.1, 3.1])
        >>> metrics = compute_error_metrics(field1, field2)
        >>> print(f"RMS error: {metrics.rms_error:.4f}")
    """
    field1 = np.asarray(field1).flatten()
    field2 = np.asarray(field2).flatten()
    
    if len(field1) != len(field2):
        raise ValueError(f"Field sizes must match: {len(field1)} != {len(field2)}")
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(field1) | np.isnan(field2))
    field1 = field1[valid_mask]
    field2 = field2[valid_mask]
    
    if len(field1) == 0:
        raise ValueError("No valid comparison points after NaN filtering")
    
    # Compute error
    diff = field1 - field2
    abs_diff = np.abs(diff)
    
    # Absolute metrics
    l2_error = np.sqrt(np.sum(diff**2))
    linf_error = np.max(abs_diff)
    rms_error = np.sqrt(np.mean(diff**2))
    mae = np.mean(abs_diff)
    
    # Relative metrics
    if reference_norm is None:
        reference_norm = np.mean(np.abs(field2))
    
    if reference_norm == 0:
        reference_norm = 1.0  # Avoid division by zero
    
    relative_l2 = (l2_error / (reference_norm * np.sqrt(len(field2)))) * 100
    relative_rms = (rms_error / reference_norm) * 100
    
    return ErrorMetrics(
        l2_error=l2_error,
        linf_error=linf_error,
        rms_error=rms_error,
        mae=mae,
        relative_l2=relative_l2,
        relative_rms=relative_rms,
        num_points=len(field1)
    )


def interpolate_to_common_grid(
    points1: NDArray,
    values1: NDArray,
    points2: NDArray,
    values2: NDArray,
    method: str = 'linear',
    fill_value: float = np.nan
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Interpolate two fields to a common set of points for comparison.
    
    Uses the union of both point sets, interpolating each field.
    
    Args:
        points1: (N1, 2) array of coordinates for field 1
        values1: (N1,) array of values for field 1
        points2: (N2, 2) array of coordinates for field 2
        values2: (N2,) array of values for field 2
        method: Interpolation method ('linear', 'nearest', 'cubic')
        fill_value: Value for extrapolated points
    
    Returns:
        (common_points, interp_values1, interp_values2)
    
    Examples:
        >>> pts1 = np.array([[0, 0], [1, 0], [0, 1]])
        >>> vals1 = np.array([1.0, 2.0, 3.0])
        >>> pts2 = np.array([[0.5, 0], [0, 0.5]])
        >>> vals2 = np.array([1.5, 2.5])
        >>> common_pts, v1, v2 = interpolate_to_common_grid(pts1, vals1, pts2, vals2)
    """
    # Use finer grid as common grid
    if len(points1) > len(points2):
        common_points = points1
        # Field 1 is already on target grid
        interp_values1 = values1
        # Interpolate field 2
        interp_values2 = griddata(points2, values2, common_points, 
                                   method=method, fill_value=fill_value)
    else:
        common_points = points2
        # Field 2 is already on target grid
        interp_values2 = values2
        # Interpolate field 1
        interp_values1 = griddata(points1, values1, common_points,
                                   method=method, fill_value=fill_value)
    
    return common_points, interp_values1, interp_values2


def compute_gci(
    values_fine: NDArray,
    values_medium: NDArray,
    values_coarse: NDArray,
    refinement_ratio: float,
    safety_factor: float = 1.25
) -> GCIResult:
    """
    Compute Grid Convergence Index using three nested grids.
    
    Follows the procedure in Roache (1994, 1998) for quantifying
    discretization uncertainty.
    
    Args:
        values_fine: Solution on finest grid
        values_medium: Solution on medium grid
        values_coarse: Solution on coarsest grid
        refinement_ratio: Grid spacing ratio (typically 2.0)
        safety_factor: Safety factor (1.25 for 3+ grids, 3.0 for 2 grids)
    
    Returns:
        GCIResult with convergence metrics
    
    References:
        Roache, P.J. (1998). "Verification of Codes and Calculations."
        AIAA Journal, 36(5), 696-702.
    
    Examples:
        >>> fine = np.array([1.00, 2.00, 3.00])
        >>> medium = np.array([1.02, 2.02, 3.02])
        >>> coarse = np.array([1.05, 2.05, 3.05])
        >>> gci = compute_gci(fine, medium, coarse, refinement_ratio=2.0)
        >>> print(f"Observed order: {gci.observed_order:.2f}")
    """
    # Ensure all arrays have same length (interpolate if needed)
    if not (len(values_fine) == len(values_medium) == len(values_coarse)):
        raise ValueError("All value arrays must have same length for GCI computation")
    
    # Representative values (use mean)
    f1 = np.mean(values_fine)
    f2 = np.mean(values_medium)
    f3 = np.mean(values_coarse)
    
    r = refinement_ratio
    
    # Compute epsilon (differences)
    eps21 = f2 - f1  # medium - fine
    eps32 = f3 - f2  # coarse - medium
    
    # Avoid division by zero
    if abs(eps21) < 1e-15 or abs(eps32) < 1e-15:
        # Essentially converged
        return GCIResult(
            gci_fine=0.0,
            gci_medium=0.0,
            observed_order=np.inf,
            asymptotic_ratio=1.0,
            refinement_ratio=r,
            extrapolated_value=f1
        )
    
    # Observed order of convergence
    p = np.log(eps32 / eps21) / np.log(r)
    
    # GCI on fine grid
    gci_fine = (safety_factor * abs(eps21)) / ((r**p - 1) * abs(f1))
    
    # GCI on medium grid
    gci_medium = (safety_factor * abs(eps32)) / ((r**p - 1) * abs(f2))
    
    # Asymptotic ratio (should be ~ 1.0 if in asymptotic range)
    asymptotic_ratio = gci_medium / (r**p * gci_fine)
    
    # Richardson extrapolation to zero grid spacing
    extrapolated_value = f1 + eps21 / (r**p - 1)
    
    return GCIResult(
        gci_fine=gci_fine,
        gci_medium=gci_medium,
        observed_order=p,
        asymptotic_ratio=asymptotic_ratio,
        refinement_ratio=r,
        extrapolated_value=extrapolated_value
    )


def compute_richardson_extrapolation(
    value_fine: float,
    value_coarse: float,
    refinement_ratio: float,
    order: float
) -> float:
    """
    Richardson extrapolation to estimate zero-grid-spacing value.
    
    Args:
        value_fine: Solution on fine grid
        value_coarse: Solution on coarse grid
        refinement_ratio: Grid spacing ratio
        order: Order of convergence
    
    Returns:
        Extrapolated value at h=0
    
    Examples:
        >>> fine = 1.00
        >>> coarse = 1.04
        >>> extrapolated = compute_richardson_extrapolation(fine, coarse, 2.0, 2.0)
        >>> print(f"Extrapolated: {extrapolated:.4f}")
    """
    eps = value_coarse - value_fine
    return value_fine + eps / (refinement_ratio**order - 1)


def compute_convergence_rate(
    errors: NDArray,
    grid_sizes: NDArray
) -> Tuple[float, float]:
    """
    Compute convergence rate from error vs grid size data.
    
    Fits power law: error = C * h^p where p is convergence order.
    
    Args:
        errors: Array of error values
        grid_sizes: Array of characteristic grid sizes (e.g., sqrt(1/N_panels))
    
    Returns:
        (convergence_order, coefficient) from fitted power law
    
    Examples:
        >>> errors = np.array([0.1, 0.025, 0.00625])
        >>> h = np.array([0.1, 0.05, 0.025])
        >>> order, coeff = compute_convergence_rate(errors, h)
        >>> print(f"Convergence order: {order:.2f}")
    """
    if len(errors) != len(grid_sizes):
        raise ValueError("errors and grid_sizes must have same length")
    
    if len(errors) < 2:
        raise ValueError("Need at least 2 data points for convergence rate")
    
    # Log-log fit: log(error) = log(C) + p * log(h)
    log_h = np.log(grid_sizes)
    log_err = np.log(errors)
    
    # Linear regression
    p, log_C = np.polyfit(log_h, log_err, 1)
    C = np.exp(log_C)
    
    return p, C


def compute_field_statistics(field: NDArray) -> Dict[str, float]:
    """
    Compute basic statistics for a field.
    
    Args:
        field: Array of field values
    
    Returns:
        Dictionary with min, max, mean, std, median
    """
    field = np.asarray(field).flatten()
    valid = field[~np.isnan(field)]
    
    if len(valid) == 0:
        return {
            'min': np.nan,
            'max': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'num_valid': 0
        }
    
    return {
        'min': float(np.min(valid)),
        'max': float(np.max(valid)),
        'mean': float(np.mean(valid)),
        'std': float(np.std(valid)),
        'median': float(np.median(valid)),
        'num_valid': len(valid)
    }
