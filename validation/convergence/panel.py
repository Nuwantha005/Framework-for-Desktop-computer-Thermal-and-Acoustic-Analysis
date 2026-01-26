"""
Panel method mesh convergence study utilities.

Provides functions for running the panel method solver at different
mesh resolutions and computing convergence metrics.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import time

from core.io import CaseLoader
from solvers.panel2d.spm import SourcePanelSolver


@dataclass
class PanelMeshConfig:
    """Configuration for a panel mesh resolution."""
    name: str
    n_panels_side: int
    n_panels_arc: int
    
    @property
    def total_panels(self) -> int:
        """Estimate total panels (assumes 2 components, 4 sides, 4 corners each)."""
        return 2 * (4 * self.n_panels_side + 4 * self.n_panels_arc)
    
    def __str__(self) -> str:
        return f"{self.name}: side={self.n_panels_side}, arc={self.n_panels_arc} (≈{self.total_panels} panels)"


@dataclass
class PanelConvergenceResult:
    """Result from a single panel method resolution."""
    config: PanelMeshConfig
    num_panels: int
    solve_time: float
    success: bool
    error_msg: Optional[str] = None
    
    # Solution data
    sigma: Optional[NDArray] = None  # Source strengths
    Cp: Optional[NDArray] = None  # Pressure coefficients
    mesh_data: Optional[Dict] = None  # Mesh for field computation
from visualization.field2d import VelocityField2D


def run_panel_convergence(
    case_path: Path,
    resolutions: List[int],
    compute_field: bool = True,
    num_cores: int = 6
) -> List[Dict]:
    """
    Run panel method convergence study at multiple resolutions.
    
    Args:
        case_path: Path to case directory
        resolutions: List of panel resolutions to test (e.g., [32, 64, 128, 256])
        compute_field: If True, compute velocity field for each resolution
        num_cores: Number of cores for parallel field computation
    
    Returns:
        List of result dictionaries, one per resolution
    """
    # Load case
    case = CaseLoader.load_case(case_path)
    
    results = []
    
    for res in resolutions:
        print(f"\n{'='*60}")
        print(f"Panel Method: Resolution {res}")
        print(f"{'='*60}")
        
        # For now, this assumes fixed mesh from JSON
        # TODO: Phase 5 will use parametric geometry
        result = run_single_panel_case(
            case,
            resolution_label=res,
            compute_field=compute_field,
            num_cores=num_cores
        )
        results.append(result)
    
    return results


def run_single_panel_case(
    case,
    resolution_label: int,
    compute_field: bool = True,
    num_cores: int = 6
) -> Dict:
    """
    Run panel method for a single resolution.
    
    Args:
        case: Loaded Case object
        resolution_label: Resolution identifier (for labeling)
        compute_field: If True, compute velocity field
        num_cores: Number of cores for parallel computation
    
    Returns:
        Dictionary with solver results and metrics
    """
    # Solve panel method
    solver = SourcePanelSolver(case.mesh, v_inf=case.v_inf, aoa=case.aoa)
    solver.solve()
    
    # Extract surface metrics
    metrics = compute_panel_metrics(solver, case.mesh)
    metrics["resolution"] = resolution_label
    metrics["num_panels"] = case.mesh.num_panels
    
    # Compute velocity field if requested
    if compute_field:
        print(f"  Computing velocity field...")
        vfield = VelocityField2D(case.mesh, case.v_inf, case.aoa, solver.sigma)
        XX, YY, Vx, Vy = vfield.compute(
            case.x_range,
            case.y_range,
            case.resolution,
            num_cores=num_cores
        )
        
        metrics["field"] = {
            "XX": XX,
            "YY": YY,
            "Vx": Vx,
            "Vy": Vy,
            "V_mag": np.sqrt(Vx**2 + Vy**2)
        }
    
    return metrics


def compute_panel_metrics(
    solver: SourcePanelSolver,
    mesh: Mesh
) -> Dict:
    """
    Compute metrics from solved panel method.
    
    Args:
        solver: Solved SourcePanelSolver instance
        mesh: Panel mesh
    
    Returns:
        Dictionary of metrics
    """
    Cp = solver.Cp
    Vt = solver.Vt
    
    metrics = {
        # Pressure coefficient statistics
        "Cp_min": float(np.min(Cp)),
        "Cp_max": float(np.max(Cp)),
        "Cp_mean": float(np.mean(Cp)),
        "Cp_std": float(np.std(Cp)),
        
        # Tangential velocity statistics
        "Vt_min": float(np.min(Vt)),
        "Vt_max": float(np.max(Vt)),
        "Vt_mean": float(np.mean(np.abs(Vt))),
        "Vt_std": float(np.std(Vt)),
        
        # Source strength statistics
        "sigma_min": float(np.min(solver.sigma)),
        "sigma_max": float(np.max(solver.sigma)),
        "sigma_mean": float(np.mean(np.abs(solver.sigma))),
    }
    
    return metrics


def compare_panel_resolutions(
    results: List[Dict],
    reference_idx: int = -1
) -> Dict:
    """
    Compare panel method results at different resolutions.
    
    Args:
        results: List of result dictionaries from run_panel_convergence
        reference_idx: Index of reference solution (default: finest, -1)
    
    Returns:
        Dictionary with convergence metrics
    """
    if len(results) < 2:
        raise ValueError("Need at least 2 resolutions for comparison")
    
    reference = results[reference_idx]
    
    convergence = {
        "resolutions": [r["resolution"] for r in results],
        "num_panels": [r["num_panels"] for r in results],
        "Cp_convergence": [],
        "Vt_convergence": [],
    }
    
    # Compute error metrics relative to reference
    for res_data in results[:-1]:  # Exclude reference from comparison
        # Cp error
        cp_error = abs(res_data["Cp_min"] - reference["Cp_min"]) / abs(reference["Cp_min"])
        convergence["Cp_convergence"].append(float(cp_error))
        
        # Vt error
        vt_error = abs(res_data["Vt_max"] - reference["Vt_max"]) / abs(reference["Vt_max"])
        convergence["Vt_convergence"].append(float(vt_error))
    
    return convergence
