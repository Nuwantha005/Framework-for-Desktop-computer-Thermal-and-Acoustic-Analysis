"""Visualization module for panel method solver."""

from .visualizer import Visualizer, OutputManager
from .field2d import VelocityField2D
from .panel2d import PanelVisualizer2D
from .comparison import ComparisonVisualizer, FieldSeries, LineSeries, ComparisonMetrics
from .surface_envelope import (
    plot_surface_envelope,
    plot_surface_envelope_comparison,
    plot_dual_surface_envelope,
    compute_outward_normals,
)
from .solver_comparison import SolverComparisonVisualizer

# Legacy exports (prefer Visualizer for new code)
from .mesh_plot import MeshPlotter, quick_plot_mesh, quick_plot_component, quick_plot_scene
from .streamlines import StreamlineVisualizer

__all__ = [
    # Primary API
    'Visualizer',
    'OutputManager',
    'VelocityField2D',
    'PanelVisualizer2D',
    # Comparison
    'ComparisonVisualizer',
    'FieldSeries',
    'LineSeries',
    'ComparisonMetrics',
    # Surface envelope plots
    'plot_surface_envelope',
    'plot_surface_envelope_comparison',
    'plot_dual_surface_envelope',
    'compute_outward_normals',
    # Solver comparison
    'SolverComparisonVisualizer',
    # Legacy
    'MeshPlotter',
    'quick_plot_mesh',
    'quick_plot_component',
    'quick_plot_scene',
    'StreamlineVisualizer',
]
