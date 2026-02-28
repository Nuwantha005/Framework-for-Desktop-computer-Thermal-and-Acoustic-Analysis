"""Panel method solvers."""

from .panel2d.spm import SourcePanelSolver
from .panel2d.linear_source_solver import LinearSourcePanelSolver
from .panel2d.linear_vortex_solver import LinearVortexPanelSolver
from .factory import SolverFactory
from .base import Solver

# Register available solvers
SolverFactory.register("source", "constant", "flat", SourcePanelSolver)
SolverFactory.register("source", "linear", "flat", LinearSourcePanelSolver)
SolverFactory.register("vortex", "linear", "flat", LinearVortexPanelSolver)

def __getattr__(name):
    """Lazy import for comparison module (avoids circular import at init time)."""
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SourcePanelSolver",
    "LinearSourcePanelSolver",
    "LinearVortexPanelSolver",
    "SolverFactory",
    "Solver",
    "SolverComparisonRunner",
    "ComparisonResult",
    "SolverResult",
    "extract_openfoam_reference",
]
