"""
Surface comparison utilities.

Wrapper around Phase 1 surface comparison functionality for
tangential velocity and pressure coefficient along body surfaces.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

from postprocessing.surface import SurfaceData, SurfaceDataExtractor
from validation.adapters.openfoam.surface_extractor import OpenFOAMSurfaceExtractor
from visualization.comparison import compare_surface_distributions, compute_surface_metrics


def compare_surface_velocities(
    panel_case,
    panel_solver,
    openfoam_case_dir: Path,
    component_id: Optional[int] = None,
    output_dir: Optional[Path] = None,
    show: bool = True
) -> dict:
    """
    Compare surface velocities between panel method and OpenFOAM.
    
    This is a convenience wrapper around the Phase 1 surface comparison
    functionality.
    
    Args:
        panel_case: Panel method case object
        panel_solver: Solved panel method solver
        openfoam_case_dir: Path to OpenFOAM case directory
        component_id: Component to compare (None for all)
        output_dir: Directory to save plots (None to skip saving)
        show: Whether to display plots
    
    Returns:
        Dictionary with comparison metrics
    
    Examples:
        >>> from core.io import CaseLoader
        >>> from solvers.panel2d.spm import SourcePanelSolver
        >>> case = CaseLoader.load_case("cases/single_square")
        >>> solver = SourcePanelSolver(case.mesh, case.v_inf, case.aoa)
        >>> solver.solve()
        >>> results = compare_surface_velocities(
        ...     case, solver,
        ...     Path("validation_results/single_square/openfoam")
        ... )
    """
    # Extract panel method surface data
    pm_extractor = SurfaceDataExtractor(
        mesh=panel_case.mesh,
        solver=panel_solver
    )
    pm_surface = pm_extractor.extract()
    
    # Extract OpenFOAM surface data
    of_extractor = OpenFOAMSurfaceExtractor(
        openfoam_case_dir=openfoam_case_dir,
        panel_mesh=panel_case.mesh
    )
    of_surface = of_extractor.extract()
    
    # Filter by component if requested
    if component_id is not None:
        pm_mask = pm_surface.component_id == component_id
        of_mask = of_surface.component_id == component_id
        
        pm_surface = SurfaceData(
            x=pm_surface.x[pm_mask],
            y=pm_surface.y[pm_mask],
            s=pm_surface.s[pm_mask],
            Vt=pm_surface.Vt[pm_mask],
            Cp=pm_surface.Cp[pm_mask],
            Vn=pm_surface.Vn[pm_mask],
            component_id=pm_surface.component_id[pm_mask]
        )
        
        of_surface = SurfaceData(
            x=of_surface.x[of_mask],
            y=of_surface.y[of_mask],
            s=of_surface.s[of_mask],
            Vt=of_surface.Vt[of_mask],
            Cp=of_surface.Cp[of_mask],
            Vn=of_surface.Vn[of_mask],
            component_id=of_surface.component_id[of_mask]
        )
    
    # Compute metrics
    metrics = compute_surface_metrics(pm_surface, of_surface)
    
    # Plot comparison
    fig = compare_surface_distributions(
        pm_surface, of_surface,
        labels=["Panel Method", "OpenFOAM"]
    )
    
    # Save if requested
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comp_str = f"_comp{component_id}" if component_id is not None else ""
        save_path = output_dir / f"surface_comparison{comp_str}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return metrics


__all__ = [
    'compare_surface_velocities',
]
