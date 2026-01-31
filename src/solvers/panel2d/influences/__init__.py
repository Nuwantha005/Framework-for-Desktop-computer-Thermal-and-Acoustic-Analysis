"""
Influence coefficient computations for panel methods.

This package contains modules for computing geometric influence coefficients
for different singularity types:
- source.py: Source panel influences
- doublet.py: Doublet panel influences (TODO)
- vortex.py: Vortex panel influences (TODO)
"""

from .source import (
    compute_source_influence_matrices,
    compute_source_velocity_influence
)

__all__ = [
    'compute_source_influence_matrices',
    'compute_source_velocity_influence',
]
