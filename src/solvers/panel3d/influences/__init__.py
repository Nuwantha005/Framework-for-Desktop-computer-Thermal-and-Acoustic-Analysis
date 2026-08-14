"""
3D influence coefficient computations.

Contains functions for computing influences of 3D singularity elements:
- Quadrilateral constant-strength source panels
"""

from .source3d import (
    compute_source_influence_matrix,
    compute_all_velocities_influence,
)

__all__ = [
    "compute_source_influence_matrix",
    "compute_all_velocities_influence",
]
