"""
3D panel method solvers.

Provides solvers for 3D aerodynamic flows over bodies:
- SourcePanelSolver3D: Constant-strength source panels for non-lifting bodies
"""

from .base import PanelSolver3D
from .source_panel_solver import SourcePanelSolver3D

__all__ = [
    "PanelSolver3D",
    "SourcePanelSolver3D",
]
