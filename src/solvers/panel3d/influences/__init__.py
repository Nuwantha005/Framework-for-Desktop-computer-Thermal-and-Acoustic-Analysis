"""
3D influence coefficient computations.

Contains functions for computing influences of 3D singularity elements:
- Quadrilateral constant-strength source panels
"""

from .source3d import (
    compute_quad_source_potential,
    compute_quad_source_velocity,
    compute_source_influence_matrix,
    compute_source_velocity_influence,
)

__all__ = [
    "compute_quad_source_potential",
    "compute_quad_source_velocity", 
    "compute_source_influence_matrix",
    "compute_source_velocity_influence",
]
