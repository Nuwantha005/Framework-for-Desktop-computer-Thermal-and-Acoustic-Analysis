"""Panel method solvers."""

from .panel2d.spm import SourcePanelSolver
from .panel2d.linear_source_solver import LinearSourcePanelSolver
from .panel2d.linear_vortex_solver import LinearVortexPanelSolver
from .panel2d.dirichlet_doublet_solver import DirichletDoubletSolver
from .panel2d.linear_source_doublet_solver import LinearSourceDoubletSolver
from .panel3d import SourcePanelSolver3D, PanelSolver3D
from .factory import SolverFactory
from .base import Solver

# Register available solvers
SolverFactory.register("source", "constant", "flat", SourcePanelSolver)
SolverFactory.register("source", "linear", "flat", LinearSourcePanelSolver)
SolverFactory.register("vortex", "linear", "flat", LinearVortexPanelSolver)
SolverFactory.register("source_doublet", "constant", "flat", DirichletDoubletSolver)
SolverFactory.register("source_doublet", "linear", "flat", LinearSourceDoubletSolver)

def __getattr__(name):
    """Lazy import for comparison and boundary_layer modules (avoids circular import at init time)."""
    if name == "SolverComparisonRunner":
        from .comparison import SolverComparisonRunner
        return SolverComparisonRunner
    if name == "ComparisonResult":
        from .comparison import ComparisonResult
        return ComparisonResult
    if name == "SolverResult":
        from .comparison import SolverResult
        return SolverResult
    if name == "extract_openfoam_reference":
        from .comparison import extract_openfoam_reference
        return extract_openfoam_reference
    # Boundary layer solver (lazy to avoid loading scipy at import time)
    if name == "BoundaryLayerSolver":
        from .boundary_layer import BoundaryLayerSolver
        return BoundaryLayerSolver
    if name == "BoundaryLayerResult":
        from .boundary_layer import BoundaryLayerResult
        return BoundaryLayerResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SourcePanelSolver",
    "LinearSourcePanelSolver",
    "LinearVortexPanelSolver",
    "DirichletDoubletSolver",
    "LinearSourceDoubletSolver",
    "SourcePanelSolver3D",
    "PanelSolver3D",
    "SolverFactory",
    "Solver",
    "SolverComparisonRunner",
    "ComparisonResult",
    "SolverResult",
    "extract_openfoam_reference",
    "BoundaryLayerSolver",
    "BoundaryLayerResult",
]
