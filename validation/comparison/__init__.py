"""
Comparison utilities for validation.

Provides functions for comparing panel method results with OpenFOAM
on structured grids, surfaces, and probe points.
"""

from . import grid, surface, probe

__all__ = ['grid', 'surface', 'probe']
