"""
Influence coefficient computations for panel methods.

This package contains modules for computing geometric influence coefficients
for different singularity types:
- source.py: Source panel influences
- doublet.py: Constant-strength doublet panel influences (Dirichlet BC)
- linear_source.py: Linear-strength source panel influences
- linear_vortex.py: Linear-strength vortex panel influences
- linear_doublet.py: Linear-strength doublet panel influences (Dirichlet BC)
"""

from .source import (
    compute_source_influence_matrices,
    compute_source_velocity_influence,
    compute_source_potential_influence
)

from .doublet import (
    compute_doublet_potential_influence,
    compute_doublet_influence_matrix,
    compute_source_potential_matrix,
    compute_doublet_velocity_influence,
)

from .linear_doublet import (
    compute_linear_doublet_potential_influence,
    compute_linear_doublet_influence_matrix,
    compute_linear_source_potential_matrix as compute_linear_source_potential_matrix_dirichlet,
    compute_linear_doublet_velocity_influence,
    compute_linear_doublet_velocity_field,
    compute_linear_source_potential_influence,
)

__all__ = [
    'compute_source_influence_matrices',
    'compute_source_velocity_influence',
    'compute_source_potential_influence',
    'compute_doublet_potential_influence',
    'compute_doublet_influence_matrix',
    'compute_source_potential_matrix',
    'compute_doublet_velocity_influence',
    'compute_linear_doublet_potential_influence',
    'compute_linear_doublet_influence_matrix',
    'compute_linear_source_potential_matrix_dirichlet',
    'compute_linear_doublet_velocity_influence',
    'compute_linear_doublet_velocity_field',
    'compute_linear_source_potential_influence',
]
