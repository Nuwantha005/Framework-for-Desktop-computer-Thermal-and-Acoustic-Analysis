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
from .bl_plots import (
    plot_bl_line,
    plot_bl_lines_multi,
    plot_bl_two_sides,
    plot_bl_envelope,
    plot_bl_envelope_comparison,
    plot_bl_comparison,
    # Phase 5 — velocity-field visualizations
    plot_bl_velocity_contour,
    plot_bl_velocity_contour_normalized,
    plot_bl_velocity_envelope,
    plot_bl_velocity_contour_two_sides,
    plot_bl_velocity_contour_normalized_two_sides,
    plot_bl_velocity_envelope_two_sides,
    plot_bl_of_comparison,
)

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
    # Boundary layer plots
    'plot_bl_line',
    'plot_bl_lines_multi',
    'plot_bl_two_sides',
    'plot_bl_envelope',
    'plot_bl_envelope_comparison',
    'plot_bl_comparison',
    # Boundary layer velocity-field plots (Phase 5)
    'plot_bl_velocity_contour',
    'plot_bl_velocity_contour_normalized',
    'plot_bl_velocity_envelope',
    'plot_bl_velocity_contour_two_sides',
    'plot_bl_velocity_contour_normalized_two_sides',
    'plot_bl_velocity_envelope_two_sides',
    'plot_bl_of_comparison',
    # Legacy
    'MeshPlotter',
    'quick_plot_mesh',
    'quick_plot_component',
    'quick_plot_scene',
    'StreamlineVisualizer',
]
