"""Panel method solvers."""

from .panel2d.spm import SourcePanelSolver
from .factory import SolverFactory
from .base import Solver

# Register available solvers
SolverFactory.register("source", "constant", "flat", SourcePanelSolver)

__all__ = [
    "SourcePanelSolver",
    "SolverFactory",
    "Solver",
]
