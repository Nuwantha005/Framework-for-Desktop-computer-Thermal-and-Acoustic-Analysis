"""Compatibility layer for boundary-layer plotting APIs.

This module re-exports plotting functions from smaller focused modules.
"""

from __future__ import annotations

from visualization.bl_envelope_plots import (
    plot_bl_envelope,
    plot_bl_envelope_comparison,
)
from visualization.bl_fluent_comparison_plots import (
    plot_bl_comparison_report,
    plot_bl_fluent_comparison,
    plot_bl_fluent_comparison_two_sides,
    plot_bl_fluent_contour_normalized_side_by_side,
    plot_bl_fluent_contour_side_by_side,
    plot_bl_fluent_envelope_side_by_side,
    plot_bl_of_comparison,
    plot_bl_velocity_contour_normalized_comparison,
    plot_bl_velocity_envelope_comparison,
    plot_bl_wall_comparison,
)
from visualization.bl_line_plots import (
    plot_bl_comparison,
    plot_bl_line,
    plot_bl_lines_multi,
    plot_bl_two_sides,
)
from visualization.bl_velocity_plots import (
    plot_bl_velocity_contour,
    plot_bl_velocity_contour_normalized,
    plot_bl_velocity_contour_normalized_two_sides,
    plot_bl_velocity_contour_two_sides,
    plot_bl_velocity_envelope,
    plot_bl_velocity_envelope_two_sides,
)
