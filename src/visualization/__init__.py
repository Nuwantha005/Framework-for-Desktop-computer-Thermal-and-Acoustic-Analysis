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
    plot_bl_fluent_comparison,
    plot_bl_fluent_comparison_two_sides,
    plot_bl_wall_comparison,
    plot_bl_velocity_envelope_comparison,
    plot_bl_velocity_contour_normalized_comparison,
    plot_bl_comparison_report,
    plot_bl_fluent_envelope_side_by_side,
    plot_bl_fluent_contour_side_by_side,
    plot_bl_fluent_contour_normalized_side_by_side,
    plot_bl_of_comparison,
)
from .bl_wall_envelope_plots import (
    plot_wall_quantity_envelope_side_by_side,
    plot_wall_quantity_envelope_overlay,
    plot_wall_quantity_envelopes_grid,
)
from .thermal_plots import (
    ThermalCaseResult,
    plot_thermal_line,
    plot_thermal_lines_multi,
    plot_thermal_two_sides,
    plot_thermal_envelope,
    plot_thermal_envelope_two_sides,
    plot_thermal_summary,
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
    'plot_bl_fluent_comparison',
    'plot_bl_fluent_comparison_two_sides',
    'plot_bl_wall_comparison',
    'plot_bl_velocity_envelope_comparison',
    'plot_bl_velocity_contour_normalized_comparison',
    'plot_bl_comparison_report',
    'plot_bl_fluent_envelope_side_by_side',
    'plot_bl_fluent_contour_side_by_side',
    'plot_bl_fluent_contour_normalized_side_by_side',
    'plot_bl_of_comparison',
    # Thermal BL plots
    'ThermalCaseResult',
    'plot_thermal_line',
    'plot_thermal_lines_multi',
    'plot_thermal_two_sides',
    'plot_thermal_envelope',
    'plot_thermal_envelope_two_sides',
    'plot_thermal_summary',
    # Legacy
    'MeshPlotter',
    'quick_plot_mesh',
    'quick_plot_component',
    'quick_plot_scene',
    'StreamlineVisualizer',
]
