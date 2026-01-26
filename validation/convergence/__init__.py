"""
Convergence study utilities.

Provides functions for running panel method and OpenFOAM convergence studies,
computing error metrics, and comparing results.
"""

from . import panel, openfoam, metrics

# Import key classes/functions for convenience
from .metrics import (
    ErrorMetrics,
    GCIResult,
    compute_error_metrics,
    compute_gci,
    compute_richardson_extrapolation,
    compute_convergence_rate,
    interpolate_to_common_grid
)

from .openfoam import (
    OpenFOAMMeshConfig,
    OpenFOAMConvergenceResult,
    create_mesh_configs,
    run_openfoam_convergence,
    extract_openfoam_fields
)

from .panel import (
    PanelMeshConfig,
    PanelConvergenceResult,
    run_panel_convergence,
    run_single_panel_case
)

__all__ = [
    'panel',
    'openfoam', 
    'metrics',
    # Metrics
    'ErrorMetrics',
    'GCIResult',
    'compute_error_metrics',
    'compute_gci',
    'compute_richardson_extrapolation',
    'compute_convergence_rate',
    'interpolate_to_common_grid',
    # OpenFOAM
    'OpenFOAMMeshConfig',
    'OpenFOAMConvergenceResult',
    'create_mesh_configs',
    'run_openfoam_convergence',
    'extract_openfoam_fields',
    # Panel
    'PanelMeshConfig',
    'PanelConvergenceResult',
    'run_panel_convergence',
    'run_single_panel_case',
]
